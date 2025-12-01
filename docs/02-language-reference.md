# Language Reference (quick)

Basics
- Expressions only: everything returns a value.
- `let name = expr` — define an immutable variable lazily evaluated on first use.
- Function definition shorthand: `foo(x,y) -> x + y` or block form `bar(x) -> { expr1; expr2; return x }`.
- `return <expr>` inside a block returns from that block. Guarded returns are supported: `return v when cond`.

Conditionals
- `A when B else C` — Theta syntax for conditional expressions (transformed to Python `A if B else C`).

Arrays
- OCaml-style arrays: `[1;2;3]` — semicolons are accepted inside `[]` and internally mapped to Python lists.
- Double-bracket display: `[[1;2;3]]` will preserve a double-bracketed visual representation.

Operators
- Arithmetic: `+ - * / % ** //` (common Python operators supported)
- Comparisons and booleans: `== != < <= > >=`, `and`, `or`, `not` (via the AST evaluator)

BluePrints
- Blueprints are Python objects registered under a name; callable attributes are available as `name.attr(...)` from Theta.
- Built-in: `io.out(...)` and `io.in(...)` (input) plus `tm.run(...)` for running Turing machines (Python-backed blueprint).

Parsing notes
- Inline `#` comments are stripped unless inside quotes.
- Multi-line bracketed constructs are supported in files (the runner collects until brackets balance).

Safety
- Expressions are parsed into Python AST and evaluated by a restricted walker — no arbitrary `eval` of source.
- Only safe builtins (`abs`, `min`, `max`, `pow`, `len`) and registered blueprints are permitted.

Examples
- Increment function:
```
inc(x) -> x + 1
inc(5)  # -> 6
```
- Let variable:
```
let xs = [1;2;3]
xs[0]   # -> 1
```
