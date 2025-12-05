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

Write-Host "Building one-folder executable with PyInstaller";
pyinstaller --onedir -y .\src\pyinstaller.spec

Write-Host "Done. Output in .\dist\theta\";
