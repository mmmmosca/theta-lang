# Usage

Requirements
- Python 3.10+ (tested on 3.13).

Run the REPL

```powershell
python .\theta.py
```

Defining variables with `let`

```
theta> let xs = [1;2;3]
Defined variable 'xs'   # no value printed; definition returns None
theta> xs
[1;2;3]
```

Notes:
- `let` stores the expression lazily; evaluation happens on first access.
- Use plain assignment inside block bodies: `{ x = 1; return x }` (do not write `let x = 1` inside blocks).

Run a `.th` script

```powershell
python .\theta.py tests\example.th
```

Enable debug traces

```powershell
python .\theta.py --debug tests\tm_pure.th
```

Common workflows
- Edit `.th` files in your editor and run them via `python theta.py file.th`.
- Use the REPL for quick experiments and define `let` variables or `->` functions interactively.
- Import blueprints from Theta files by name: `import foo` loads `foo.th` when it contains `blueprint foo [ ... ]`. The blueprint name must match the file name.
 - Use boolean literals and operators: `true`, `false`, `!x` (negation), `a && b`, `a || b`, and `not a`.
 - Cast values with built-in functions: `Int("123")`, `Float("3.14")`, `String([1;2;3])`, `Bool("true")`.

Files and tests
- `tests/run_tests.py` runs the example programs (factorial, fibonacci, ackermann, TMs).
 - See `examples/booleans.th` and `examples/casting.th` for usage of booleans and type casting.

Reporting issues
- For bugs, include the Theta script that reproduces the problem and run with `--debug` to include the Python traceback.
