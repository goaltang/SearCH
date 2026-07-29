#Requires -Version 5.1
# Install script: create a virtual environment and install photofinder.

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"

# Check Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "Error: python not found. Please install Python 3.10+ (https://www.python.org/downloads/)" -ForegroundColor Red
    exit 1
}

$ver = & python --version 2>&1
Write-Host "Detected Python: $ver"

# Create virtual environment
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment .venv ..."
    & python -m venv $venvDir
} else {
    Write-Host "Virtual environment already exists, skipping creation."
}

# Upgrade pip and install the project (dev+fast extras match the Docker image)
Write-Host "Installing dependencies..."
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -e "$projectRoot[dev,fast]"

# Download models if missing
$detModel = Join-Path $projectRoot "models\det_10g.onnx"
$recModel = Join-Path $projectRoot "models\w600k_r50.onnx"
if ((-not (Test-Path $detModel)) -or (-not (Test-Path $recModel))) {
    Write-Host "Downloading models (first run, ~300MB+) ..."
    & $pythonExe "$projectRoot\download_models.py" --models-dir "$projectRoot\models"
} else {
    Write-Host "Model files already exist, skipping download."
}

Write-Host "`nInstallation complete." -ForegroundColor Green
Write-Host "Start the Web UI with: .\start-webui.ps1" -ForegroundColor Cyan
