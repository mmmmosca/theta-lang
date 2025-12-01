# Theta

Theta is a functional-expression language implemented in Python. It is expression-oriented (everything is an expression) and designed for readable, concise code. You can use it interactively from the REPL or run `.th` script files.

**Key features**
- Expression-based language (control flow and blocks are expressions).
- Safe AST-based expression evaluation (no direct `eval` of user code).
- Immutable `let` variables with lazy evaluation and cycle detection.
- Function definitions with both single-expression returns and block bodies.
- `when ... else` conditional syntax (transformed into Python ternary expressions).
- OCaml-style arrays using semicolons (e.g., `[1;2;3]`) and special double-bracket display semantics.
- Blueprints: pluggable Python-backed modules accessible as `name.method(...)` from Theta.
  - Built-in `io` blueprint for simple input/output.
  - A Python-side `tm` blueprint that runs Turing machines.
- REPL plus `.th` source-code runner that supports multi-line, bracketed constructs (for transition tables and multi-line function/blueprint blocks).

Files of interest
- `theta.py` — the interpreter / REPL. The single, primary file that implements parsing, AST evaluation, the `ThetaArray` type, blueprints, and the `run_file` script runner.
- `tests/` — example `.th` scripts and a simple test runner (factorial, fibonacci, ackermann, and Turing machine examples).

Quick start

Requirements
- Python 3.10+ (developed and tested on recent Python 3.13).

Run the REPL

```powershell
python theta.py
```

Define a function and call it:

```text
theta> inc(x) -> x + 1
theta> inc(5)
6
```

Run a `.th` script

```powershell
python theta.py tests/factorial.th
```

Enable verbose debugging

If you need to see internal debug traces (for parsing, multi-line handling, etc.), run with `--debug` or `--verbose`:

```powershell
python theta.py --debug tests/tm_pure.th
```

Working with blueprints

Built-in blueprints provide side-effectful behavior accessible from Theta expressions:
- `io.out(...)` — prints values to stdout.
- `tm.run(transitions, tape, head, state, max_steps)` — Python-side Turing Machine runner.

`transitions` for the `tm` blueprint can be a Theta array of 5-tuples/lists: `[state, symbol, new_symbol, direction, next_state]`.

Tests and examples

There is a `tests` folder with small `.th` programs and a Python test runner that executes them and reports success. See `tests/run_tests.py` for how tests are invoked.

Contributing and development notes

- Contributions and bug reports are welcome. When editing `theta.py`, keep changes focused and run the included tests in `tests/` to validate behavior.
- If you add debug printing, use the module `DEBUG` flag and the `log()` helper so output can be toggled with `--debug`/`--verbose`.

License

MIT License