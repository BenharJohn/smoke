$ErrorActionPreference = "Stop"

param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$RunName = "bigmodel",

    [int]$PilotExamples = 32,

    [string]$Device = "cuda"
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$extract3Config = Join-Path $root "configs\_generated_extract_${RunName}_3prompt.json"
$extractPilotConfig = Join-Path $root "configs\_generated_extract_${RunName}_pilot.json"
$trainSmokeConfig = Join-Path $root "configs\_generated_train_${RunName}_smoke.json"

$extract3Output = "runs/extract_${RunName}_3prompt"
$extractPilotOutput = "runs/extract_${RunName}_pilot"
$trainSmokeOutput = "runs/train_${RunName}_smoke"

$extract3 = @{
    dataset = "data/gemma_smoke_prompts.jsonl"
    teacher_model_path = $ModelPath
    teacher_quantization = "none"
    output_dir = $extract3Output
    max_examples = 3
    max_length = 256
    device = $Device
} | ConvertTo-Json

$extractPilot = @{
    dataset = "data/gemma_smoke_prompts.jsonl"
    teacher_model_path = $ModelPath
    teacher_quantization = "none"
    output_dir = $extractPilotOutput
    max_examples = $PilotExamples
    max_length = 512
    device = $Device
} | ConvertTo-Json

$trainSmoke = @{
    manifest = "$extract3Output/manifest.jsonl"
    output_dir = $trainSmokeOutput
    projection_model_path = $ModelPath
    epochs = 1
    max_examples = 3
    batch_size = 1
    grad_accum_steps = 1
    learning_rate = 0.0001
    checkpoint_every = 1
    device = $Device
    dtype = "float16"
} | ConvertTo-Json

Set-Content -Path $extract3Config -Value $extract3 -Encoding UTF8
Set-Content -Path $extractPilotConfig -Value $extractPilot -Encoding UTF8
Set-Content -Path $trainSmokeConfig -Value $trainSmoke -Encoding UTF8

Write-Host ""
Write-Host "Step 1: 3-prompt extraction"
python -m peagle_q.cli extract --config $extract3Config

$manifestPath = Join-Path $root ($extract3Output + "\manifest.jsonl")
$tensorDir = Join-Path $root ($extract3Output + "\tensors")

if (-not (Test-Path $manifestPath)) {
    throw "Expected manifest not found: $manifestPath"
}
if (-not (Test-Path $tensorDir)) {
    throw "Expected tensor directory not found: $tensorDir"
}

$tensorCount = (Get-ChildItem $tensorDir -Filter *.pt -File | Measure-Object).Count
if ($tensorCount -lt 3) {
    throw "Expected at least 3 tensor files, found $tensorCount"
}

Write-Host ""
Write-Host "Verified first step:"
Write-Host "  manifest: $manifestPath"
Write-Host "  tensors : $tensorDir ($tensorCount files)"

Write-Host ""
Write-Host "Next commands:"
Write-Host "  1) Scale extraction:"
Write-Host "     python -m peagle_q.cli extract --config $extractPilotConfig"
Write-Host "  2) Run tiny training smoke:"
Write-Host "     python -m peagle_q.cli train --config $trainSmokeConfig"

