# PyInstaller spec for one-folder builds of Theta
# Uses theta.py as entrypoint; includes Cython extensions fastpaths and fastpaths_vm
# Generate: pyinstaller --onedir -y pyinstaller.spec

# Note: On macOS/Linux, ensure Cython extensions are built for that platform before packaging.

block_cipher = None

import sys
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = [
    'fastpaths',
    'fastpaths_vm',
]

# Explicitly include compiled extensions (Windows example names)
datas = [
    ('fastpaths.cp313-win_amd64.pyd', '.'),
    ('fastpaths_vm.cp313-win_amd64.pyd', '.'),
]

# Entry script
entry_script = 'theta.py'

# Build the Analysis
a = Analysis(
    [entry_script],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='theta',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='theta'
)
