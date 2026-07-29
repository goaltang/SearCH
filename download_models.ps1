#Requires -Version 5.1
# Download the InsightFace buffalo_l models into models/.

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Error: virtual environment not found. Please run .\install.ps1 first." -ForegroundColor Red
    exit 1
}

& $pythonExe "$projectRoot\download_models.py" --models-dir "$projectRoot\models"
