# Theta Syntax Highlighting (VS Code)

This is a minimal TextMate-based grammar for Theta `.th` files. It provides basic highlighting for comments, strings, numbers, keywords, and function names, plus an optional file icon theme.

What’s highlighted
- Core keywords: `let`, `def`, `return`, `when`, `else`, `matches`, `blueprint`, `import`.
- Operators: arrow `->`.
- Numbers: integers and floats.
- Blueprints used as qualifiers: `io.`, `tm.`, `python.`
- Star-rest pattern identifiers like `*rest`.

How to use during development

- From this folder run `vsce package` to produce a `.vsix` (you must have `vsce` installed).
- Install the produced `.vsix` in VS Code: `code --install-extension your-extension-0.0.1.vsix`.

Publishing

1. Install `vsce` (or use `npm`/`pnpm` toolchains):

```powershell
npm install -g vsce
```

2. Update `package.json` with your `publisher` name and version (currently `0.0.2`).
3. Create a Personal Access Token on the Visual Studio Marketplace and follow the `vsce` publish docs to login and publish.

This grammar is intentionally small; contributions and PRs welcome to expand language features (semantic tokens, snippets, or a full LSP server).

File icon theme
----------------
Version `0.0.4` adds a lightweight icon theme (structural fix) so `.th` files can display the Theta symbol in the VS Code explorer.

How to enable:
1. Install the extension (packaged `.vsix` or published version).
2. Open Command Palette: `Preferences: File Icon Theme`.
3. Select `Theta Icons`.

Implementation notes:
- The theme is contributed via `iconThemes` in `package.json`.
- Icon theme file declares a default `file` icon plus mapping: extension `th` → icon definition `theta` → `media/theta.svg`.
- SVG can be replaced with a higher-resolution PNG if desired (`media/theta.png`).

Replacing the icon:
Troubleshooting:
- If the icon does not appear, verify you selected `Theta Icons` under File Icon Theme (it will not auto-activate).
- Ensure no other icon theme overrides `.th` (selecting another theme will hide the Theta icon until re-selected).
- Reload window (`Developer: Reload Window`) after installing a new version if icons do not update.
1. Drop a new SVG/PNG into `media/`.
2. Update `theta-icon-theme.json` iconPath if you changed the filename.
3. Bump version in `package.json`.
