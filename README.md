# Theta


Theta is a functional-expression language implemented in Python. It is expression-oriented (everything is an expression) and designed for readable, concise code. You can use it interactively from the REPL or run `.th` script files.

**Key features**
- Expression-based language (control flow and blocks are expressions).
- Safe AST-based expression evaluation (no direct `eval` of user code).
- Immutable `let` variables with lazy evaluation and cycle detection.
- Function definitions with both single-expression returns and block bodies.
- `when ... else` conditional syntax (transformed into Python ternary expressions).
- Pattern matching with `matches`, list patterns, and star-rest (e.g., `[head; *rest]`).
- OCaml-style arrays using semicolons (e.g., `[1;2;3]`) and special double-bracket display semantics.
 - Booleans and logical operators: `true`/`false`, `and`/`or`/`not` (also `&&`, `||`, `!`).
 - Type casting helpers: `Int(x)`, `Float(x)`, `String(x)`, `Bool(x)` and `typeof(expr)`.
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

Pattern matching (example)

```
doubleList(xs) -> xs matches [head; *rest] return [head * 2] + doubleList(rest) else []

doubleList([1;2;3])  # -> [2;4;6]
```

Run a `.th` script
Language Cheatsheet

- Arrays: `[1;2;3]`, indexing `a[0]`
- Functions: `foo(x,y) -> x + y` or block `{ ...; return v }`
- Conditionals: `A when B else C`
- Pattern match: `xs matches [h; *t] return [h] + t else []`
- Booleans: `true`, `false`; `and`, `or`, `not` (also `&&`, `||`, `!`)
- Casting: `Int("123")`, `Float("3.14")`, `String([1;2])`, `Bool("true")`, `typeof(expr)`
- Blueprints: `io.out(x)`, `tm.run(...)`, `python.call("math.sqrt", 9)`


```powershell
python .\theta.py .\examples\double_list.th
```

Booleans and casting (quick examples)

```
import io
io.out(true && !false)      # -> True
io.out(Int("123") + 1)     # -> 124
io.out(typeof([1;2;3]))     # -> [Int]
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

There is an `examples/` folder with runnable programs (pattern matching, Python interop, TM demo) and a `tests/` folder with small `.th` programs plus a Python test runner.

Run an example:

```powershell
python .\theta.py .\examples\python_blueprint.th
```

Run the test scripts:

```powershell
python .\tests\run_tests.py
```

Contributing and development notes

- Contributions and bug reports are welcome. When editing `theta.py`, keep changes focused and run the included tests in `tests/` to validate behavior.
- If you add debug printing, use the module `DEBUG` flag and the `log()` helper so output can be toggled with `--debug`/`--verbose`.

License

MIT License
