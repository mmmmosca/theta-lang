# Theta Syntax Highlighting (VS Code)

This is a minimal TextMate-based grammar for Theta `.th` files. It provides basic highlighting for comments, strings, numbers, keywords, and function names.

How to use during development

- From this folder run `vsce package` to produce a `.vsix` (you must have `vsce` installed).
- Install the produced `.vsix` in VS Code: `code --install-extension your-extension-0.0.1.vsix`.

Publishing

1. Install `vsce` (or use `npm`/`pnpm` toolchains):

```powershell
npm install -g vsce
```

2. Update `package.json` with your `publisher` name and version.
3. Create a Personal Access Token on the Visual Studio Marketplace and follow the `vsce` publish docs to login and publish.

This grammar is intentionally small; contributions and PRs welcome to expand language features (semantic tokens, snippets, or a full LSP server).
