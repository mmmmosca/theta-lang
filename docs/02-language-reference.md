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

Pattern Matching
- Syntax: `<subject> matches <pattern> return <then_expr> else <else_expr>`.
- Patterns:
	- Literals: `0`, `'ok'`, etc.
	- Variables: `x` binds the matched value.
	- Wildcard: `_` matches anything and binds nothing.
	- Lists: `[a;b;c]` matches a list of length 3; `[head;*rest]` uses a star-rest to capture the tail.
- Scope: In the success/else expressions you can reference
	- bindings created by the pattern (e.g., `head`, `rest`), and
	- names in the current scope (globals and function parameters).
- Else arm: If omitted, it defaults to `None` and a warning is printed. Prefer writing an explicit `else`.
- Subject/pattern order: The canonical form is `subject matches pattern ...`.
	For convenience, writing `pattern matches subject ...` is also accepted; the interpreter will swap sides when it detects pattern variables on the left.

Examples
```
# double the elements of a list
doubleList(xs) -> xs matches [head; *rest] return [head * 2] + doubleList(rest) else []

doubleList([1;2;3])  # -> [2;4;6]

# pick first two when present, else 0
sum2(xs) -> xs matches [a; b; *_] return a + b else 0
```

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
