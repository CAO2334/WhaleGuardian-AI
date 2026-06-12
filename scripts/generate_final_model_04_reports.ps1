param(
    [string]$PythonExe = "python",
    [string]$DataRoot = "archive",
    [string]$OutputRoot = "outputs\reports\final_model_04",
    [string]$ArtifactDir = "artifacts\final_model_04",
    [int]$BatchSize = 8,
    [int]$NumWorkers = 0,
    [string]$ImagePath = ""
)

$ErrorActionPreference = "Stop"

function Resolve-FirstExistingPath {
    param(
        [string]$ProjectRoot,
        [string[]]$Candidates,
        [string]$Description
    )

    foreach ($candidate in $Candidates) {
        $full = Join-Path $ProjectRoot $candidate
        if (Test-Path -LiteralPath $full) {
            return $full
        }
    }

    throw "Cannot find $Description. Tried:`n$($Candidates -join "`n")"
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$OutputRootAbs = Join-Path $ProjectRoot $OutputRoot
$AnalysisDir = Join-Path $OutputRootAbs "analysis"
$EvaluationDir = Join-Path $OutputRootAbs "evaluation"
$InterpretabilityDir = Join-Path $OutputRootAbs "interpretability"
$ArtifactDirAbs = Join-Path $ProjectRoot $ArtifactDir
$LogDir = Join-Path $ProjectRoot "logs"
$LogPath = Join-Path $LogDir "generate_final_model_04_reports_$Timestamp.log"

Ensure-Dir $OutputRootAbs
Ensure-Dir $AnalysisDir
Ensure-Dir $EvaluationDir
Ensure-Dir $InterpretabilityDir
Ensure-Dir $ArtifactDirAbs
Ensure-Dir $LogDir

$Checkpoint = Resolve-FirstExistingPath -ProjectRoot $ProjectRoot -Description "checkpoint for experiment 04" -Candidates @(
    "outputs\ablation_all_results\outputs\ablations\04_transformer_mean_focal_mixup_cutout\best_model.pth",
    "outputs\ablations\04_transformer_mean_focal_mixup_cutout\best_model.pth"
)

$ClassMap = Resolve-FirstExistingPath -ProjectRoot $ProjectRoot -Description "class map for experiment 04" -Candidates @(
    "outputs\ablation_all_results\outputs\ablations\04_transformer_mean_focal_mixup_cutout\class_to_idx.json",
    "outputs\ablations\04_transformer_mean_focal_mixup_cutout\class_to_idx.json"
)

$Metrics = Resolve-FirstExistingPath -ProjectRoot $ProjectRoot -Description "metrics file for experiment 04" -Candidates @(
    "outputs\ablation_all_results\outputs\ablations\04_transformer_mean_focal_mixup_cutout\metrics.json",
    "outputs\ablations\04_transformer_mean_focal_mixup_cutout\metrics.json"
)

$TrainCsv = Join-Path $ProjectRoot "$DataRoot\train.csv"
$TrainImageDir = Join-Path $ProjectRoot "$DataRoot\train_images"

if (-not (Test-Path -LiteralPath $TrainCsv)) {
    throw "Cannot find train CSV: $TrainCsv"
}

if (-not (Test-Path -LiteralPath $TrainImageDir)) {
    throw "Cannot find train image directory: $TrainImageDir"
}

if ([string]::IsNullOrWhiteSpace($ImagePath)) {
    $DefaultImage = Join-Path $ProjectRoot "$DataRoot\train_images\90f1655bca651f.jpg"
    if (Test-Path -LiteralPath $DefaultImage) {
        $ImagePath = $DefaultImage
    } else {
        $FirstImage = Get-ChildItem -LiteralPath $TrainImageDir -File | Select-Object -First 1
        if ($null -eq $FirstImage) {
            throw "Cannot find a sample image for Grad-CAM / Attention Map."
        }
        $ImagePath = $FirstImage.FullName
    }
} elseif (-not [System.IO.Path]::IsPathRooted($ImagePath)) {
    $ImagePath = Join-Path $ProjectRoot $ImagePath
}

if (-not (Test-Path -LiteralPath $ImagePath)) {
    throw "Cannot find sample image: $ImagePath"
}

Start-Transcript -Path $LogPath -Force | Out-Null

Push-Location $ProjectRoot
try {
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Checkpoint: $Checkpoint"
    Write-Host "Class map: $ClassMap"
    Write-Host "Metrics: $Metrics"
    Write-Host "Sample image: $ImagePath"
    Write-Host "Report dir: $OutputRootAbs"
    Write-Host "Artifact dir: $ArtifactDirAbs"

    & $PythonExe tools/analyze_dataset.py `
        --csv $TrainCsv `
        --image-dir $TrainImageDir `
        --output-dir $AnalysisDir `
        --checkpoint $Checkpoint `
        --class-map $ClassMap `
        --batch-size $BatchSize `
        --num-workers $NumWorkers `
        --split-strategy group `
        --group-col individual_id

    & $PythonExe tools/eval_confusion_matrix.py `
        --checkpoint $Checkpoint `
        --class-map $ClassMap `
        --eval-csv $TrainCsv `
        --image-dir $TrainImageDir `
        --output (Join-Path $EvaluationDir "confusion_matrix.png") `
        --batch-size $BatchSize `
        --num-workers $NumWorkers `
        --split-strategy group `
        --group-col individual_id `
        --normalize

    & $PythonExe tools/generate_gradcam.py `
        --image $ImagePath `
        --checkpoint $Checkpoint `
        --class-map $ClassMap `
        --output (Join-Path $InterpretabilityDir "gradcam.jpg")

    & $PythonExe tools/generate_attention_map.py `
        --image $ImagePath `
        --checkpoint $Checkpoint `
        --class-map $ClassMap `
        --output (Join-Path $InterpretabilityDir "attention_map.jpg")

    & $PythonExe tools/export_onnx.py `
        --checkpoint $Checkpoint `
        --class-map $ClassMap `
        --metrics $Metrics `
        --artifact-dir $ArtifactDirAbs `
        --version "v_final_04"

    Copy-Item -LiteralPath $Metrics -Destination (Join-Path $OutputRootAbs "metrics.json") -Force
    Copy-Item -LiteralPath $Checkpoint -Destination (Join-Path $OutputRootAbs "best_model.pth") -Force
    Copy-Item -LiteralPath $ClassMap -Destination (Join-Path $OutputRootAbs "class_to_idx.json") -Force

    Write-Host ""
    Write-Host "Done."
    Write-Host "Report dir: $OutputRootAbs"
    Write-Host "Artifact dir: $ArtifactDirAbs"
    Write-Host "Log file: $LogPath"
}
finally {
    Pop-Location
    Stop-Transcript | Out-Null
}
