$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$vendorPath = Join-Path $root ".vendor-smoke"
$tempPath = Join-Path $root ".tmp"

Write-Host "Workspace: $root"
New-Item -ItemType Directory -Force -Path $vendorPath, $tempPath | Out-Null

$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "Checking workspace-local smoke-test dependencies"
$env:PYTHONPATH = "$vendorPath;$root"
$probe = @'
import importlib.util
required = {
    "sentencepiece": "sentencepiece",
    "protobuf": "google.protobuf",
}
missing = [package for package, module_name in required.items() if importlib.util.find_spec(module_name) is None]
print(",".join(missing))
'@
$probeOutput = $probe | python -
if ($LASTEXITCODE -ne 0) {
    throw "Failed to probe workspace-local smoke-test dependencies."
}
$missing = ($probeOutput | Out-String).Trim()

if ($missing) {
    Write-Host "Installing missing packages into $vendorPath"
    pip install --upgrade --target $vendorPath sentencepiece "protobuf<6"
} else {
    Write-Host "Required packages already available"
}

Write-Host ""
Write-Host "If Gemma download fails, open a separate shell and run:"
Write-Host "  hf auth login"
Write-Host "Then make sure you have accepted the Gemma terms for the model on Hugging Face."
Write-Host ""
Write-Host "Running Gemma 1B CPU smoke test..."

& python -m peagle_q.cli smoke --config (Join-Path $root "configs\smoke_gemma_1b_cpu.json")
