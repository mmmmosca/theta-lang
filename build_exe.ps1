# Build one-folder Theta executable for Windows using PyInstaller
param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [switch]$RebuildCython
)

Write-Host "Activating venv and installing PyInstaller";
& ".\.venv\Scripts\Activate.ps1"; pip install --upgrade pyinstaller

if ($RebuildCython) {
    Write-Host "Rebuilding Cython extensions";
    python .\src\setup.py build_ext --inplace
}

Write-Host "Cleaning previous build outputs";
if (Test-Path .\dist) { Remove-Item .\dist -Recurse -Force }
if (Test-Path .\build) { Remove-Item .\build -Recurse -Force }

Write-Host "Building one-folder executable with PyInstaller";
Push-Location .\src
pyinstaller -y --clean --distpath ..\dist --workpath ..\build pyinstaller.spec
Pop-Location
 
Write-Host "Done. Output in .\\dist\\theta";
