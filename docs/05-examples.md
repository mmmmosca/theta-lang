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
