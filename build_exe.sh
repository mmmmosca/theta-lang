#!/usr/bin/env bash
set -euo pipefail

# Build one-folder Theta executables for macOS/Linux using PyInstaller
# Ensure you are on the target OS and have built Cython extensions in-place

python3 -m pip install --upgrade pyinstaller

# Optional: rebuild Cython extensions
if [[ "${REBUILD_CYTHON:-}" == "1" ]]; then
  python3 src/setup.py build_ext --inplace
fi

# Clean previous outputs
rm -rf dist build || true

# Build one-folder from src to ensure correct resolution
pushd src >/dev/null
pyinstaller -y --clean --distpath ../dist --workpath ../build pyinstaller.spec
popd >/dev/null

echo "Done. Output in ./dist/theta/"