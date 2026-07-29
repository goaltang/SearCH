#Requires -Version 5.1
# Launch the PhotoFinder Web UI with the local virtual environment.

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Error: virtual environment not found. Please run .\install.ps1 first." -ForegroundColor Red
    exit 1
}

# Show a friendly warning if models are missing.
$detModel = Join-Path $projectRoot "models\det_10g.onnx"
$recModel = Join-Path $projectRoot "models\w600k_r50.onnx"
if ((-not (Test-Path $detModel)) -or (-not (Test-Path $recModel))) {
    Write-Host "Warning: model files not found. Make sure models/ contains det_10g.onnx and w600k_r50.onnx." -ForegroundColor Yellow
    Write-Host "You can download them later with .\download_models.ps1." -ForegroundColor Yellow
}

Write-Host "Starting PhotoFinder Web UI (http://127.0.0.1:7860) ..." -ForegroundColor Green
& $pythonExe -m photofinder.webui
