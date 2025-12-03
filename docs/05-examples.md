# Examples

Create an `example.th` to try features quickly:

```
# factorial example
fact(n) -> n * fact(n-1) when n > 1 else n

io.out(fact(5))

# arrays and indexing
let a = [1;2;3;4]
io.out(a[2])  # -> 3
```

Pattern matching: doubling a list
```
doubleList(xs) -> xs matches [head; *rest] return [head * 2] + doubleList(rest) else []

io.out(doubleList([1;2;3;4;5]))  # -> [2;4;6;8;10]
```

Turing machine example (using `tm` blueprint)

```
let table = [
  [ 'q0'; 1; 1; 1; 'q0' ];
  [ 'q0'; 0; 1; 0; 'qa' ]
]
let tape = [1;1;1;0;0]
let res = tm.run(table, tape, 0, 'q0', 100)
io.out(res)
```

Booleans and casting

```
# booleans
let a = true; let b = false
io.out(a && !b)   # -> True
io.out(not a)     # -> False

# casting
io.out(Int("123"))     # -> 123
io.out(Float("3.14"))  # -> 3.14
io.out(String([1;2]))  # -> [1;2]
io.out(Bool("false"))  # -> False
```
