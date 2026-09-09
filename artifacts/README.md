# 模型 Artifact 目录

推荐把每个可部署模型作为一个独立版本目录保存，例如:

```text
artifacts/
  whale_resnet50_transformer_v1/
    model.onnx
    class_to_idx.json
    config.json
    metrics.json
    manifest.json
```

生成方式:

```powershell
python tools/export_onnx.py `
  --checkpoint outputs/best_model.pth `
  --class-map outputs/class_to_idx.json `
  --artifact-dir artifacts/whale_resnet50_transformer_v1 `
  --version v1
```

Flask 默认优先加载当前最佳单模型 07:

```text
artifacts/final_model_07
```

如果该目录不存在，Flask 会自动搜索项目根目录或 `outputs/` 下最新的
`autodl_research_*/artifact`，并只接受 `experiment_name` 为
`07_previous_best_transformer_recipe` 的 artifact。

也可以通过环境变量切换模型版本:

```powershell
$env:WHALE_ARTIFACT_DIR="artifacts/whale_resnet50_transformer_v2"
python whale_web/app.py
```
