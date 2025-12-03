# Theta Cheatsheet

Core syntax at a glance. See `02-language-reference.md` for details.

- Arrays: `[1;2;3]`, indexing `a[0]`
- Functions: `foo(x,y) -> x + y` or block `{ ...; return v }`
- Conditionals: `A when B else C`
- Pattern match: `xs matches [h; *t] return [h] + t else []`
- Booleans: `true`, `false`; `and`, `or`, `not` (also `&&`, `||`, `!`)
- Casting: `Int("123")`, `Float("3.14")`, `String([1;2])`, `Bool("true")`, `typeof(expr)`
- Blueprints: `io.out(x)`, `tm.run(...)`, `python.call("math.sqrt", 9)`

Examples

```
# double a list
doubleList(xs) -> xs matches [head; *rest] return [head * 2] + doubleList(rest) else []

# factorial
fact(n) -> n * fact(n - 1) when n > 1 else 1 when n == 0 || n == 1 else n

# booleans & casting
let a = true; let b = false
io.out(a && !b)         # True
io.out(Int("123") + 1) # 124
io.out(typeof([1;2;3])) # [Int]
```
