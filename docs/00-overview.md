# Theta — Overview

Theta is an expression-oriented language implemented as a Python interpreter (`src/theta.py`). It emphasizes concise, readable expressions where control flow constructs and blocks are also expressions.

This `docs/` folder provides quick-start instructions, a language reference, blueprint documentation, and notes about the VS Code syntax extension.

Goals
- Keep the interpreter focused and safe (AST-based evaluation).
- Provide a pleasant REPL and a `.th` script runner supporting multi-line constructs.
- Allow side effects via pluggable "blueprints" implemented in Python.

Highlights
- Expression-oriented semantics (`when/else`, blocks return values).
- Pattern matching with list patterns and star-rest (`matches`).

Project layout
- `src/` — runtime sources and assets:
	- `src/theta.py`, `src/fastpaths.py`, `src/fastpaths_vm.pyx`, `src/theta_types.py`
	- `src/tests/` example `.th` programs and test runner, `src/test_debug.py`
	- compiled extensions (e.g., `src/fastpaths*.pyd` on Windows)
- `docs/` — this documentation folder.
- `examples/` — sample Theta scripts.
- `build_exe.ps1`, `build_exe.sh` — one-folder packaging scripts.
- `vscode-theta-syntax/` — TextMate grammar and packaging scaffold.

License
Add a `LICENSE` file if you wish to publish this project under an open-source license.
