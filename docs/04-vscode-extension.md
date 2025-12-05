# VS Code Extension (syntax highlighting)

A minimal TextMate-based grammar is provided in `vscode-theta-syntax/` which highlights comments, strings, numbers, keywords, and function names.

Quick test (dev host)
1. Open the `vscode-theta-syntax` folder in VS Code.
2. Press F5 to launch the Extension Development Host.
3. Open a `.th` file in the host window; the grammar should apply automatically.

Optional runtime integration
- The interpreter now lives under `src/theta.py`. For quick experiments while developing the extension, you can run sample files via:
```powershell
python .\src\theta.py .\examples\string_match.th
```
or on macOS/Linux:
```bash
python3 src/theta.py examples/string_match.th
```

Package & install
```powershell
cd vscode-theta-syntax
npm install -g vsce
vsce package
code --install-extension vscode-theta-syntax-0.0.1.vsix
```

Configure file association
- In your settings JSON add:
```json
"files.associations": {"*.th": "theta"}
```

Notes
- This grammar is intentionally compact. Contributions welcome to improve highlighting (operators, keywords, semantic tokens, snippets, etc.).
 - Project packaging: use the root `build_exe.ps1` / `build_exe.sh` which reference `src/pyinstaller.spec`.
