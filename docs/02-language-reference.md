# Language Reference (quick)

Basics
- Expressions only: everything returns a value.
- `let name = expr` — define an immutable variable lazily evaluated on first use.
	- Returns no immediate value (internally `None`); the bound expression is evaluated only when the variable is first referenced later.
	- Not allowed inside block bodies; inside `{ ... }` use plain assignment `name = expr` for local temporaries.
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
	- Strings as sequences: list patterns also work on strings; `[head;*rest]` over a string binds `head` to the first character and `rest` to the remaining substring.
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

# string prefix example
firstRest(s) -> s matches ["R"; *rest] return rest else 0
firstRest("R8")  # -> "8"
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

Booleans
- Literals: `true`, `false` evaluate to Bool values (also accept `True`/`False`).
- Operators: `and`, `or`, `not` and their C-style aliases `&&`, `||`, `!`.

Type Casting
- Built-in casting functions:
	- `Int(x)` — to integer. Accepts numbers and numeric strings (e.g., `"123"`).
	- `Float(x)` — to float. Accepts numbers and decimal strings.
	- `String(x)` — to string. Theta arrays render as `[a;b;c]`.
	- `Bool(x)` — to boolean. Accepts `"true"/"false"`, `1/0`, `yes/no`, `on/off`.
	- `typeof(expr)` — returns a Hindley–Milner type summary (e.g., `Bool`, `Int`, `[Int]`).

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
