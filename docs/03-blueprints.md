# Blueprints (side effects and host integration)

What are blueprints?
- Blueprints are plain Python objects registered with `register_blueprint(name, obj)` in `theta.py`.
- From Theta you can call `name.method(...)` and the call is dispatched to the Python object.

Built-in blueprints
- `io` — simple I/O convenience:
  - `io.out(...)` prints values to stdout.
  - `io.in(prompt)` reads a line from stdin.
- `tm` — a Python-side Turing Machine runner: `tm.run(transitions, tape, head, state, max_steps)`.

Note: built-in blueprints such as `io`, `tm`, and `python` are available by default; you do not need to `import tm` in your `.th` scripts to use them. The `import` statement is used to register external Python modules as blueprints when needed.

Importing blueprints from Theta files
------------------------------------

You can import a blueprint defined in another Theta file using:

```
import foo
```

Requirements:
- File naming: the interpreter looks for a file named `foo.th` in the current working directory.
- Blueprint name: inside `foo.th` there must be a block `blueprint foo [ ... ]` — the blueprint name must match the file name.
- Methods: define methods with `def name(params) -> expr` inside the blueprint block.

Example (`math_utils.th`):
```
blueprint math_utils [
  def twice(x) -> x * 2
]
```

Usage from another file:
```
import math_utils
io.out(math_utils.twice(21))  # -> 42
```

Import resolution order:
- First, Theta tries to load `foo.th` and register `blueprint foo` from it.
- If no matching `.th` file is found, it falls back to importing a Python module named `foo` and registering it as a blueprint.

Creating blueprints from `.th`
- The file runner supports `blueprint name [ ... ]` blocks where each `def name(params) -> expr` is turned into a Python-callable method on the blueprint object.

Safety
- Blueprints can perform arbitrary Python actions; do not register untrusted objects if running untrusted Theta code.

Example (using `tm` blueprint from Theta):
```
let table = [[ 'q0'; 1; 1; 1; 'q0' ]; [ 'q0'; 0; 1; 0; 'qa' ]]
let tape = [1;1;1;0;0]
let res = tm.run(table, tape, 0, 'q0', 100)
io.out(res)
```

Python interop blueprint
------------------------

For convenience, a `python` blueprint is provided which exposes a single
method `python.call(path, args...)` that resolves a dotted path and calls
the target Python callable.

Examples:

```
io.out(python.call('math.sqrt', 16))      # prints 4.0
io.out(python.call('os.path.join', 'a', 'b'))
```

The call performs a safe import of the top-level module and walks attributes
to resolve the final object; if it's callable it will be invoked with the
provided arguments, otherwise the resolved Python object is returned to
Theta.
