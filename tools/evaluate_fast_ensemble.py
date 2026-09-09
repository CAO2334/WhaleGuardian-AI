"""Validation-locked probability ensemble with optional horizontal-flip TTA.

This is deliberately a post-training experiment: it reuses completed checkpoints,
selects weights using only the frozen validation split, locks that choice, and then
evaluates the independent test split once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import WhaleSpeciesDataset, build_transforms, normalize_species_column
from models.factory import load_model_from_checkpoint
from tools.generate_model_report import (
    bootstrap_intervals,
    expected_calibration_error,
    observed_balanced_accuracy,
    save_confidence_plots,
    save_confusion_plot,
)


DEFAULT_EXPERIMENTS = (
    "02_resnet50_ce_crop",
    "04_metric_linear_gem_crop",
    "06_transformer_mean_ce_crop",
    "07_previous_best_transformer_recipe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复用已训练权重，按验证 Macro F1 锁定概率集成/TTA，再评估一次独立测试集"
    )
    parser.add_argument("--run-root", required=True, help="例如 outputs/autodl_research_20260909-023128")
    parser.add_argument("--image-dir", default="archive/train_images")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--group-col", default="individual_id")
    parser.add_argument("--max-samples", type=int, default=0, help="仅供冒烟测试；正式实验保持 0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--disable-hflip", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        return torch.load(path, map_location="cpu")


def discover_inputs(
    run_root: Path,
    experiments: Sequence[str],
    seed: int,
) -> Tuple[Dict[str, Path], Path, Path, Path, Dict[str, int], int]:
    checkpoints: Dict[str, Path] = {}
    reference_class_map: Dict[str, int] | None = None
    reference_val: pd.DataFrame | None = None
    reference_test: pd.DataFrame | None = None
    class_map_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None
    image_size: int | None = None

    for experiment in experiments:
        experiment_dir = run_root / "ablations" / experiment / f"seed_{seed}"
        checkpoint_path = experiment_dir / "best_model.pth"
        current_class_map_path = experiment_dir / "class_to_idx.json"
        current_val_path = experiment_dir / "splits" / "val.csv"
        current_test_path = experiment_dir / "splits" / "test.csv"
        for path in (checkpoint_path, current_class_map_path, current_val_path, current_test_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        current_class_map = {
            str(key): int(value)
            for key, value in json.loads(current_class_map_path.read_text(encoding="utf-8-sig")).items()
        }
        current_val = normalize_species_column(pd.read_csv(current_val_path), fix_typos=True)
        current_test = normalize_species_column(pd.read_csv(current_test_path), fix_typos=True)
        checkpoint = load_checkpoint(checkpoint_path)
        current_image_size = int(checkpoint.get("config", {}).get("image_size", 512))

        if reference_class_map is None:
            reference_class_map = current_class_map
            reference_val = current_val
            reference_test = current_test
            class_map_path = current_class_map_path
            val_path = current_val_path
            test_path = current_test_path
            image_size = current_image_size
        else:
            if current_class_map != reference_class_map:
                raise ValueError(f"类别映射不一致: {experiment}")
            if current_image_size != image_size:
                raise ValueError(f"输入尺寸不一致: {experiment}={current_image_size}, reference={image_size}")
            pd.testing.assert_frame_equal(current_val, reference_val, check_dtype=False)
            pd.testing.assert_frame_equal(current_test, reference_test, check_dtype=False)
        checkpoints[experiment] = checkpoint_path

    if not checkpoints or reference_class_map is None:
        raise ValueError("没有找到可用检查点。")
    assert class_map_path is not None and val_path is not None and test_path is not None and image_size is not None
    return checkpoints, class_map_path, val_path, test_path, reference_class_map, image_size


def make_loader(
    csv_path: Path,
    image_dir: Path,
    class_to_idx: Dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, DataLoader]:
    frame = normalize_species_column(pd.read_csv(csv_path), fix_typos=True)
    if max_samples > 0:
        frame = frame.head(max_samples).reset_index(drop=True)
    unknown = sorted(set(frame["species"]) - set(class_to_idx))
    if unknown:
        raise ValueError(f"评估 CSV 含类别映射外物种: {unknown}")
    _, transform = build_transforms(image_size=image_size, cutout_p=0.0)
    dataset = WhaleSpeciesDataset(frame, image_dir, class_to_idx, transforms=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return frame, loader


def cache_matches(
    cache: Mapping[str, np.ndarray],
    checkpoint: Path,
    split_hash: str,
    require_hflip: bool,
) -> bool:
    try:
        return (
            str(cache["checkpoint_path"].item()) == str(checkpoint.resolve())
            and int(cache["checkpoint_size"].item()) == checkpoint.stat().st_size
            and str(cache["split_sha256"].item()) == split_hash
            and (not require_hflip or bool(cache["hflip_computed"].item()))
        )
    except (KeyError, ValueError, OSError):
        return False


def predict_probabilities(
    name: str,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
    enable_hflip: bool,
    cache_path: Path,
    split_hash: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if cache_matches(cached, checkpoint_path, split_hash, enable_hflip):
                labels = cached["labels"]
                original = cached["original"]
                hflip = cached["hflip"]
                if len(labels) == len(loader.dataset) and original.shape == (len(labels), num_classes):
                    print(f"[{name}] 使用缓存: {cache_path}")
                    return labels, original, hflip, 0.0

    checkpoint = load_checkpoint(checkpoint_path)
    model = load_model_from_checkpoint(checkpoint, num_classes=num_classes, device=device)
    original_parts: List[np.ndarray] = []
    hflip_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc=f"{name}", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                original_logits = model(images)
                flipped_logits = model(torch.flip(images, dims=(3,))) if enable_hflip else original_logits
            original_parts.append(torch.softmax(original_logits.float(), dim=1).cpu().numpy())
            hflip_parts.append(torch.softmax(flipped_logits.float(), dim=1).cpu().numpy())
            label_parts.append(labels.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    labels_np = np.concatenate(label_parts)
    original_np = np.concatenate(original_parts)
    hflip_np = np.concatenate(hflip_parts)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        labels=labels_np,
        original=original_np,
        hflip=hflip_np,
        checkpoint_path=np.array(str(checkpoint_path.resolve())),
        checkpoint_size=np.array(checkpoint_path.stat().st_size, dtype=np.int64),
        split_sha256=np.array(split_hash),
        hflip_computed=np.array(enable_hflip),
    )
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return labels_np, original_np, hflip_np, elapsed


def integer_compositions(total: int, parts: int, prefix: Tuple[int, ...] = ()) -> Iterator[Tuple[int, ...]]:
    if parts == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from integer_compositions(total - value, parts - 1, prefix + (value,))


def search_probability_ensemble(
    probabilities: Mapping[str, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    num_classes: int,
    weight_step: float,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    reciprocal = round(1.0 / weight_step)
    if not np.isclose(reciprocal * weight_step, 1.0) or reciprocal < 1:
        raise ValueError("weight-step 必须能整除 1，例如 0.5、0.25、0.2、0.1。")
    names = list(probabilities)
    variants = list(next(iter(probabilities.values())).keys())
    records: List[Dict[str, object]] = []
    for variant in variants:
        model_arrays = [probabilities[name][variant] for name in names]
        for units in integer_compositions(reciprocal, len(names)):
            weights = np.asarray(units, dtype=np.float64) / reciprocal
            combined = np.zeros_like(model_arrays[0], dtype=np.float64)
            for weight, array in zip(weights, model_arrays):
                if weight > 0:
                    combined += weight * array
            preds = combined.argmax(axis=1)
            active_weights = {
                name: float(weight) for name, weight in zip(names, weights) if weight > 0
            }
            records.append(
                {
                    "tta": variant,
                    "tta_passes": 1 if variant == "none" else 2,
                    "weights": json.dumps(active_weights, ensure_ascii=False, sort_keys=True),
                    "active_models": len(active_weights),
                    "val_macro_f1": float(
                        f1_score(
                            labels,
                            preds,
                            labels=list(range(num_classes)),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "val_accuracy": float(accuracy_score(labels, preds)),
                }
            )
    table = pd.DataFrame(records).sort_values(
        ["val_macro_f1", "val_accuracy", "active_models", "tta_passes", "weights"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    best["weights"] = json.loads(str(best["weights"]))
    best["selection_metric"] = "validation_macro_f1"
    best["test_set_used_for_selection"] = False
    best["weight_step"] = weight_step
    return best, table


def combine_probabilities(
    probabilities: Mapping[str, Mapping[str, np.ndarray]],
    selection: Mapping[str, object],
) -> np.ndarray:
    variant = str(selection["tta"])
    weights = {str(key): float(value) for key, value in dict(selection["weights"]).items()}
    first = next(iter(probabilities.values()))[variant]
    combined = np.zeros_like(first, dtype=np.float64)
    for name, weight in weights.items():
        combined += weight * probabilities[name][variant]
    return combined / np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)


def save_test_report(
    output_dir: Path,
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_to_idx: Dict[str, int],
    train_csv: Path,
    selection: Mapping[str, object],
    checkpoint_paths: Mapping[str, Path],
    elapsed: float,
    device: torch.device,
    group_col: str,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    class_names = [idx_to_class[index] for index in range(len(idx_to_class))]
    preds = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    top3 = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]
    present_labels = sorted(np.unique(labels).tolist())
    precision, recall, per_f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names))), zero_division=0
    )
    train_df = normalize_species_column(pd.read_csv(train_csv), fix_typos=True)
    train_counts = train_df["species"].value_counts()
    tail_count = max(1, int(np.ceil(len(class_names) * 0.20)))
    tail_species = train_counts.reindex(class_names, fill_value=0).sort_values().head(tail_count).index.tolist()
    per_class = pd.DataFrame(
        {
            "class_index": range(len(class_names)),
            "species": class_names,
            "precision": precision,
            "recall": recall,
            "f1": per_f1,
            "support": support.astype(int),
        }
    )
    per_class["train_count"] = per_class["species"].map(train_counts).fillna(0).astype(int)
    per_class["is_tail_20_percent"] = per_class["species"].isin(tail_species)
    observed = per_class["support"] > 0
    tail_observed = observed & per_class["is_tail_20_percent"]
    head_observed = observed & ~per_class["is_tail_20_percent"]
    intervals = bootstrap_intervals(frame, labels, preds, group_col, bootstrap_iters, bootstrap_seed)
    top3_correct = np.any(top3 == labels[:, None], axis=1)
    active_checkpoints = {
        name: str(checkpoint_paths[name]) for name in dict(selection["weights"])
    }
    metrics: Dict[str, object] = {
        "split_name": "independent_test",
        "independent_test": True,
        "test_set_used_for_selection": False,
        "selection_metric": "validation_macro_f1",
        "ensemble_weights": selection["weights"],
        "tta": selection["tta"],
        "checkpoints": active_checkpoints,
        "num_samples": int(len(labels)),
        "num_classes_total": len(class_names),
        "num_classes_present": len(present_labels),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": observed_balanced_accuracy(labels, preds),
        "macro_f1_present_classes": float(
            f1_score(labels, preds, labels=present_labels, average="macro", zero_division=0)
        ),
        "macro_f1_all_classes": float(
            f1_score(labels, preds, labels=list(range(len(class_names))), average="macro", zero_division=0)
        ),
        "top3_accuracy": float(top3_correct.mean()),
        "negative_log_likelihood": float(
            log_loss(labels, probabilities, labels=list(range(len(class_names))) )
        ),
        "ece_15_bins": expected_calibration_error(labels, preds, confidence, bins=15),
        "head_macro_f1": float(per_class.loc[head_observed, "f1"].mean()) if head_observed.any() else None,
        "tail_macro_f1": float(per_class.loc[tail_observed, "f1"].mean()) if tail_observed.any() else None,
        "bootstrap_unit": group_col if group_col in frame.columns else "image",
        "bootstrap_iterations": bootstrap_iters,
        "bootstrap_95_ci": intervals,
        "device": str(device),
        "elapsed_seconds_all_selected_models": elapsed,
        "samples_per_second_end_to_end": len(labels) / max(elapsed, 1e-9),
    }

    predictions = frame.copy()
    predictions["true_index"] = labels
    predictions["pred_index"] = preds
    predictions["pred_species"] = [idx_to_class[int(index)] for index in preds]
    predictions["confidence"] = confidence
    predictions["correct"] = preds == labels
    for rank in range(3):
        predictions[f"top{rank + 1}_species"] = [idx_to_class[int(index)] for index in top3[:, rank]]
        predictions[f"top{rank + 1}_score"] = probabilities[np.arange(len(labels)), top3[:, rank]]
    matrix = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    predictions.loc[~predictions["correct"]].sort_values("confidence", ascending=False).to_csv(
        output_dir / "error_cases.csv", index=False, encoding="utf-8-sig"
    )
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_confusion_plot(matrix, class_names, output_dir / "confusion_matrix_normalized.png")
    save_confidence_plots(labels, preds, confidence, output_dir)
    return metrics


def write_summary(
    output_path: Path,
    selection: Mapping[str, object],
    metrics: Mapping[str, object],
    existing_metrics: Mapping[str, object] | None,
    checkpoints: Mapping[str, Path],
) -> None:
    weights = dict(selection["weights"])

    def metric_text(source: Mapping[str, object] | None, key: str) -> str:
        if not source or source.get(key) is None:
            return "N/A"
        return f"{float(source[key]):.4f}"

    lines = [
        "# 快速集成与 TTA 实验",
        "",
        "> 所有权重和 TTA 选择只使用固定验证集；独立测试集仅在方案锁定后评估一次。",
        "",
        "## 实验设计",
        "",
        "- 不重新训练，不增加随机种子；复用四个已有、结构互补的检查点。",
        "- 对原图与水平翻转分别推理；先对单模型 softmax 概率做 TTA 平均，再做非负概率集成。",
        f"- 集成权重步长: {float(selection['weight_step']):.2f}；选模指标: 验证 Macro F1。",
        "- 测试集从未参与权重搜索，避免测试泄漏。",
        "",
        "## 验证集锁定结果",
        "",
        f"- TTA: `{selection['tta']}`",
        f"- 验证 Accuracy: {float(selection['val_accuracy']):.4f}",
        f"- 验证 Macro F1: {float(selection['val_macro_f1']):.4f}",
        "- 锁定权重:",
    ]
    for name, weight in weights.items():
        lines.append(f"  - `{name}`: {float(weight):.2f}")
    lines.extend(
        [
            "",
            "## 独立测试结果",
            "",
            "| 指标 | 集成结果 | 原 07 单模型 |",
            "|---|---:|---:|",
        ]
    )
    lines.extend(
        [
            f"| Accuracy | {metric_text(metrics, 'accuracy')} | {metric_text(existing_metrics, 'accuracy')} |",
            f"| Macro F1 | {metric_text(metrics, 'macro_f1_present_classes')} | {metric_text(existing_metrics, 'macro_f1_present_classes')} |",
            f"| Top-3 Accuracy | {metric_text(metrics, 'top3_accuracy')} | {metric_text(existing_metrics, 'top3_accuracy')} |",
            f"| Tail Macro F1 | {metric_text(metrics, 'tail_macro_f1')} | {metric_text(existing_metrics, 'tail_macro_f1')} |",
            "",
            "## 使用的检查点",
            "",
        ]
    )
    for name, path in checkpoints.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## 解读边界",
            "",
            "- 这是一次预先限定候选的低成本后训练实验，不是新的训练消融组。",
            "- 若集成测试结果没有超过单模型，应如实报告‘验证集改善但独立测试未改善’，不要继续用测试集反复调权重。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_root = resolve(args.run_root)
    image_dir = resolve(args.image_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else run_root / "reports" / "fast_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_samples > 0:
        print("警告: max-samples > 0，仅供冒烟测试，结果不能作为正式实验。")

    checkpoints, _, val_csv, test_csv, class_to_idx, image_size = discover_inputs(
        run_root, args.experiments, args.seed
    )
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    train_csv = next(iter(checkpoints.values())).parent / "splits" / "train.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    enable_hflip = not args.disable_hflip
    print(f"device={device}, amp={use_amp}, hflip_search={enable_hflip}, image_size={image_size}")
    print("候选模型:", ", ".join(checkpoints))

    val_frame, val_loader = make_loader(
        val_csv,
        image_dir,
        class_to_idx,
        image_size,
        args.batch_size,
        args.num_workers,
        args.max_samples,
        device,
    )
    val_hash = sha256_file(val_csv) + f":{len(val_frame)}"
    val_probabilities: Dict[str, Dict[str, np.ndarray]] = {}
    validation_elapsed = 0.0
    reference_labels: np.ndarray | None = None
    for name, checkpoint in checkpoints.items():
        labels, original, flipped, elapsed = predict_probabilities(
            name,
            checkpoint,
            val_loader,
            device,
            len(class_to_idx),
            use_amp,
            enable_hflip,
            output_dir / "cache" / f"{name}_validation.npz",
            val_hash,
        )
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(labels, reference_labels):
            raise ValueError(f"验证标签顺序不一致: {name}")
        variants = {"none": original}
        if enable_hflip:
            variants["hflip"] = (original + flipped) / 2.0
        val_probabilities[name] = variants
        validation_elapsed += elapsed
    assert reference_labels is not None

    selection, search_table = search_probability_ensemble(
        val_probabilities, reference_labels, len(class_to_idx), args.weight_step
    )
    selection["candidate_models"] = list(checkpoints)
    selection["validation_csv"] = str(val_csv)
    selection["validation_split_sha256"] = sha256_file(val_csv)
    selection["validation_inference_seconds"] = validation_elapsed
    selection["independent_test_csv"] = str(test_csv)
    (output_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    search_table.to_csv(output_dir / "validation_search.csv", index=False, encoding="utf-8-sig")
    print("验证集已锁定:", json.dumps(selection, ensure_ascii=False))

    test_frame, test_loader = make_loader(
        test_csv,
        image_dir,
        class_to_idx,
        image_size,
        args.batch_size,
        args.num_workers,
        args.max_samples,
        device,
    )
    test_hash = sha256_file(test_csv) + f":{len(test_frame)}"
    selected_names = list(dict(selection["weights"]))
    test_probabilities: Dict[str, Dict[str, np.ndarray]] = {}
    test_labels: np.ndarray | None = None
    test_elapsed = 0.0
    for name in selected_names:
        labels, original, flipped, elapsed = predict_probabilities(
            name,
            checkpoints[name],
            test_loader,
            device,
            len(class_to_idx),
            use_amp,
            str(selection["tta"]) == "hflip",
            output_dir / "cache" / f"{name}_independent_test.npz",
            test_hash,
        )
        if test_labels is None:
            test_labels = labels
        elif not np.array_equal(labels, test_labels):
            raise ValueError(f"测试标签顺序不一致: {name}")
        test_probabilities[name] = {
            "none": original,
            "hflip": (original + flipped) / 2.0,
        }
        test_elapsed += elapsed
    assert test_labels is not None
    combined = combine_probabilities(test_probabilities, selection)
    metrics = save_test_report(
        output_dir / "independent_test",
        test_frame,
        test_labels,
        combined,
        class_to_idx,
        train_csv,
        selection,
        checkpoints,
        test_elapsed,
        device,
        args.group_col,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    existing_path = run_root / "reports" / "final_test" / "metrics.json"
    existing_metrics = (
        json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.is_file() else None
    )
    write_summary(output_dir / "report.md", selection, metrics, existing_metrics, checkpoints)
    print(f"报告: {(output_dir / 'report.md').resolve()}")
    print(
        f"Independent test Accuracy={float(metrics['accuracy']):.4f}, "
        f"Macro F1={float(metrics['macro_f1_present_classes']):.4f}"
    )


if __name__ == "__main__":
    main()
