from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import onnxruntime as ort
except ImportError:  # ONNX Runtime is optional for non-export unit tests.
    ort = None

from configs.config import TrainConfig
from data.dataset import NumpyTrainTransform, split_train_val_test
from models.factory import build_model, load_model_from_checkpoint
from models.resnet_metric import SubCenterArcMarginProduct
from tools.export_onnx import package_artifact
from train import build_optimizer, build_scheduler


class SplitTests(unittest.TestCase):
    def test_three_way_group_split_is_disjoint_and_repeatable(self) -> None:
        rows = []
        for species_index in range(3):
            for group_index in range(10):
                for image_index in range(2):
                    rows.append(
                        {
                            "image": f"{species_index}_{group_index}_{image_index}.jpg",
                            "species": f"species_{species_index}",
                            "individual_id": f"id_{species_index}_{group_index}",
                        }
                    )
        frame = pd.DataFrame(rows)
        first = split_train_val_test(frame, "species", 0.2, 0.2, 42, "group", "individual_id")
        second = split_train_val_test(frame, "species", 0.2, 0.2, 42, "group", "individual_id")

        first_groups = [set(part["individual_id"]) for part in first]
        self.assertFalse(first_groups[0] & first_groups[1])
        self.assertFalse(first_groups[0] & first_groups[2])
        self.assertFalse(first_groups[1] & first_groups[2])
        self.assertEqual(sum(len(part) for part in first), len(frame))
        for left, right in zip(first, second):
            self.assertEqual(left["image"].tolist(), right["image"].tolist())

    def test_random_crop_fallback_shape_and_finiteness(self) -> None:
        transform = NumpyTrainTransform(image_size=64, cutout_p=0.0, crop_scale_min=0.8)
        image = np.full((80, 120, 3), 127, dtype=np.uint8)
        tensor = transform(image=image)["image"]
        self.assertEqual(tuple(tensor.shape), (3, 64, 64))
        self.assertTrue(torch.isfinite(tensor).all())


class ArcFaceTests(unittest.TestCase):
    def test_subcenter_arcface_forward_backward_and_inference(self) -> None:
        torch.manual_seed(7)
        head = SubCenterArcMarginProduct(8, 4, scale=16.0, margin=0.2, subcenters=2)
        features = torch.randn(5, 8, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 3, 0])
        train_logits = head(features, labels)
        inference_logits = head(features)
        self.assertEqual(tuple(train_logits.shape), (5, 4))
        self.assertEqual(tuple(inference_logits.shape), (5, 4))
        self.assertTrue(torch.isfinite(train_logits).all())
        torch.nn.functional.cross_entropy(train_logits, labels).backward()
        self.assertIsNotNone(features.grad)

    @unittest.skipIf(ort is None, "onnxruntime is not installed")
    def test_metric_checkpoint_round_trip(self) -> None:
        cfg = TrainConfig(
            model_type="metric",
            metric_head="arcface",
            metric_pooling="gem",
            embedding_dim=32,
            pretrained=False,
            arcface_subcenters=2,
            image_size=64,
            mixup_alpha=0.0,
        )
        model = build_model(cfg, num_classes=4, pretrained=False)
        model.train()
        train_sample = torch.randn(2, 3, 64, 64)
        train_labels = torch.tensor([0, 1])
        train_loss = torch.nn.functional.cross_entropy(model(train_sample, train_labels), train_labels)
        train_loss.backward()
        self.assertIsNotNone(next(model.head.parameters()).grad)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, total_epochs=5)
        for _ in range(3):
            learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
            self.assertAlmostEqual(
                learning_rates["metric_head"] / learning_rates["backbone"],
                cfg.head_lr_multiplier,
                places=6,
            )
            optimizer.step()
            scheduler.step()
        checkpoint = {
            "config": vars(cfg),
            "class_to_idx": {f"species_{index}": index for index in range(4)},
            "model_state_dict": model.state_dict(),
        }
        loaded = load_model_from_checkpoint(checkpoint, device="cpu")
        model.eval()
        sample = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            expected = model(sample)
            actual = loaded(sample)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

        onnx_buffer = io.BytesIO()
        torch.onnx.export(
            loaded,
            sample[:1],
            onnx_buffer,
            opset_version=17,
            input_names=["images"],
            output_names=["logits"],
            dynamo=False,
        )
        self.assertGreater(len(onnx_buffer.getvalue()), 1_000_000)
        session = ort.InferenceSession(onnx_buffer.getvalue(), providers=["CPUExecutionProvider"])
        ort_output = session.run(None, {"images": sample[:1].numpy()})[0]
        np.testing.assert_allclose(expected[:1].numpy(), ort_output, rtol=1e-3, atol=1e-4)


class ArtifactPackagingTests(unittest.TestCase):
    def test_package_artifact_preserves_checkpoint_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_dir = root / "artifact"
            output_path = artifact_dir / "model.onnx"
            output_path.parent.mkdir(parents=True)
            output_path.write_bytes(b"onnx-placeholder")
            class_map_path = root / "class_to_idx.json"
            class_map_path.write_text('{"humpback_whale": 0}', encoding="utf-8")
            checkpoint = {
                "config": {"dropout": 0.3, "split_mode": "group"},
                "class_to_idx": {"humpback_whale": 0},
                "best_val_macro_f1": 0.9487,
            }

            package_artifact(
                artifact_dir=artifact_dir,
                output_path=output_path,
                checkpoint=checkpoint,
                class_map_path=class_map_path,
                metrics_path=None,
                model_name="whale_resnet50_transformer",
                version="test-v1",
                image_size=512,
                model_type="transformer",
                opset=17,
            )

            config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
            manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(config["dropout"], 0.3)
            self.assertEqual(config["split_mode"], "group")
            self.assertEqual(config["image_size"], 512)
            self.assertEqual(manifest["model_file"], "model.onnx")
            self.assertEqual(metrics["best_val_macro_f1"], 0.9487)


if __name__ == "__main__":
    unittest.main()
