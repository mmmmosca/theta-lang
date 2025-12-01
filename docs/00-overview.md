# Theta — Overview

Theta is an expression-oriented language implemented as a Python interpreter (`theta.py`). It emphasizes concise, readable expressions where control flow constructs and blocks are also expressions.

This `docs/` folder provides quick-start instructions, a language reference, blueprint documentation, and notes about the VS Code syntax extension.

Goals
- Keep the interpreter focused and safe (AST-based evaluation).
- Provide a pleasant REPL and a `.th` script runner supporting multi-line constructs.
- Allow side effects via pluggable "blueprints" implemented in Python.

Project layout
- `theta.py` — main interpreter, REPL, and script runner.
- `tests/` — example `.th` programs and a test runner.
- `vscode-theta-syntax/` — TextMate grammar and packaging scaffold.
- `docs/` — this documentation folder.

License
Add a `LICENSE` file if you wish to publish this project under an open-source license.
