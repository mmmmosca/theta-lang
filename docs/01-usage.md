# Usage

Requirements
- Python 3.10+ (tested on 3.13).

Run the REPL

```powershell
python .\theta.py
```

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

Files and tests
- `tests/run_tests.py` runs the example programs (factorial, fibonacci, ackermann, TMs).

Reporting issues
- For bugs, include the Theta script that reproduces the problem and run with `--debug` to include the Python traceback.
