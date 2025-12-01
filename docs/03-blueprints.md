# Blueprints (side effects and host integration)

What are blueprints?
- Blueprints are plain Python objects registered with `register_blueprint(name, obj)` in `theta.py`.
- From Theta you can call `name.method(...)` and the call is dispatched to the Python object.

Built-in blueprints
- `io` — simple I/O convenience:
  - `io.out(...)` prints values to stdout.
  - `io.in(prompt)` reads a line from stdin.
- `tm` — a Python-side Turing Machine runner: `tm.run(transitions, tape, head, state, max_steps)`.

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
