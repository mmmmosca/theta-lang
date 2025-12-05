# distutils: language=c
# Cython VM for Theta fast path

cdef enum:
    OP_PUSH_CONST = 1
    OP_LOAD_NAME = 2
    OP_STORE_NAME = 3
    OP_ADD = 4
    OP_SUB = 5
    OP_MUL = 6
    OP_DIV = 7
    OP_EQ = 8
    OP_NE = 9
    OP_LT = 10
    OP_LE = 11
    OP_GT = 12
    OP_GE = 13
    OP_MAKE_LIST = 14
    OP_NEG = 15
    OP_POS = 16
    OP_NOT = 17
    OP_AND = 18
    OP_OR = 19
    OP_MAKE_TUPLE = 20
    OP_INDEX_CONST = 21
    OP_INDEX_STACK = 22
    OP_CALL_NAME = 23
    OP_CALL_ATTR = 24
    OP_JMP_IF_FALSE = 25
    OP_SLICE = 26
    OP_JMP = 27

from cpython.list cimport PyList_GET_ITEM
from cpython.object cimport PyObject_IsTrue
from cpython.long cimport PyLong_AsSsize_t

# We expose a cpdef so Python can call it
cpdef object vm_exec(object opcodes, object consts, object names, object scope):
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t L = len(opcodes)
    # Manual stack with preallocation and pointer
    cdef Py_ssize_t cap = 64
    cdef list stack = [None] * cap
    cdef Py_ssize_t sp = 0
    cdef object a, b, val
    cdef int op
    cdef Py_ssize_t count, count2
    # simple hot name cache: last name index -> resolved object
    cdef Py_ssize_t last_name_idx = -1
    cdef object last_name_obj = None
    # micro cache for CALL_ATTR: last object + attr idx -> bound method
    cdef Py_ssize_t last_attr_idx = -1
    cdef object last_attr_obj = None
    cdef object last_attr_bound = None
    # Note: avoid Python closures; inline push/pop logic below
    while i < L:
        op = <int>PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1
        if op == OP_PUSH_CONST:
            val = <object>PyList_GET_ITEM(consts, PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i))); i += 1
            if sp >= cap:
                cap = cap * 2
                stack.extend([None] * (cap - len(stack)))
            stack[sp] = val
            sp += 1
        elif op == OP_LOAD_NAME:
            count = PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1
            if count == last_name_idx and last_name_obj is not None:
                val = last_name_obj
            else:
                val = scope[ <object>PyList_GET_ITEM(names, count) ]
                last_name_idx = count
                last_name_obj = val
            if sp >= cap:
                cap = cap * 2
                stack.extend([None] * (cap - len(stack)))
            stack[sp] = val
            sp += 1
        elif op == OP_STORE_NAME:
            sp -= 1
            count = PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1
            scope[ <object>PyList_GET_ITEM(names, count) ] = stack[sp]
            if count == last_name_idx:
                last_name_obj = stack[sp]
        elif op == OP_ADD:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            val = a + b
            if sp >= cap:
                cap = cap * 2
                stack.extend([None] * (cap - len(stack)))
            stack[sp] = val
            sp += 1
        elif op == OP_SUB:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            val = a - b
            if sp >= cap:
                cap = cap * 2
                stack.extend([None] * (cap - len(stack)))
            stack[sp] = val
            sp += 1
        elif op == OP_MUL:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            val = a * b
            if sp >= cap:
                cap = cap * 2
                stack.extend([None] * (cap - len(stack)))
            stack[sp] = val
            sp += 1
        elif op == OP_DIV:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            val = a / b
            if sp >= cap:
                cap = cap * 2
                stack.extend([None] * (cap - len(stack)))
            stack[sp] = val
            sp += 1
        elif op == OP_EQ:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            stack[sp] = (a == b); sp += 1
        elif op == OP_NE:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            stack[sp] = (a != b); sp += 1
        elif op == OP_LT:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            stack[sp] = (a < b); sp += 1
        elif op == OP_LE:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            stack[sp] = (a <= b); sp += 1
        elif op == OP_GT:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            stack[sp] = (a > b); sp += 1
        elif op == OP_GE:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            stack[sp] = (a >= b); sp += 1
        elif op == OP_MAKE_LIST:
            count = opcodes[i]; i += 1
            if count:
                # Build list from top count items without slicing
                val = [None] * count
                for idx in range(count-1, -1, -1):
                    sp -= 1
                    val[idx] = stack[sp]
                if sp >= cap:
                    cap = cap * 2
                    stack.extend([None] * (cap - len(stack)))
                stack[sp] = val
                sp += 1
            else:
                if sp >= cap:
                    cap = cap * 2
                    stack.extend([None] * (cap - len(stack)))
                stack[sp] = []
                sp += 1
        elif op == OP_MAKE_TUPLE:
            count2 = opcodes[i]; i += 1
            if count2:
                val = [None] * count2
                for idx2 in range(count2-1, -1, -1):
                    sp -= 1
                    val[idx2] = stack[sp]
                if sp >= cap:
                    cap = cap * 2
                    stack.extend([None] * (cap - len(stack)))
                stack[sp] = tuple(val)
                sp += 1
            else:
                if sp >= cap:
                    cap = cap * 2
                    stack.extend([None] * (cap - len(stack)))
                stack[sp] = ()
                sp += 1
        elif op == OP_NEG:
            sp -= 1; a = stack[sp]; stack[sp] = -a; sp += 1
        elif op == OP_POS:
            sp -= 1; a = stack[sp]; stack[sp] = +a; sp += 1
        elif op == OP_NOT:
            sp -= 1; a = stack[sp]; stack[sp] = (not a); sp += 1
        elif op == OP_AND:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            stack[sp] = True if (PyObject_IsTrue(a) and PyObject_IsTrue(b)) else False; sp += 1
        elif op == OP_OR:
            sp -= 1; b = stack[sp]
            sp -= 1; a = stack[sp]
            stack[sp] = True if (PyObject_IsTrue(a) or PyObject_IsTrue(b)) else False; sp += 1
        elif op == OP_INDEX_CONST:
            sp -= 1; val = stack[sp]
            stack[sp] = val[ <object>PyList_GET_ITEM(consts, PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i))) ]; sp += 1; i += 1
        elif op == OP_INDEX_STACK:
            sp -= 1; a = stack[sp]
            sp -= 1; val = stack[sp]
            stack[sp] = val[a]; sp += 1
        elif op == OP_CALL_NAME:
            # name index, argc inline
            count = PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1  # name_idx
            count2 = PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1  # argc
            # collect args
            val = []
            if count2:
                val = [None] * count2
                for idx3 in range(count2-1, -1, -1):
                    sp -= 1
                    val[idx3] = stack[sp]
            # resolve function and call
            if count == last_name_idx and last_name_obj is not None:
                a = last_name_obj
            else:
                a = scope[ <object>PyList_GET_ITEM(names, count) ]
                last_name_idx = count
                last_name_obj = a
            # call function
            if sp >= cap:
                cap = cap * 2
                stack.extend([None] * (cap - len(stack)))
            stack[sp] = a(*val); sp += 1
        elif op == OP_CALL_ATTR:
            # attr name index, argc inline; object then args are on stack
            count = PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1  # attr idx
            count2 = PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1  # argc
            val = []
            if count2:
                val = [None] * count2
                for idx4 in range(count2-1, -1, -1):
                    sp -= 1
                    val[idx4] = stack[sp]
            # object
            sp -= 1; a = stack[sp]
            b = <object>PyList_GET_ITEM(names, count)
            if last_attr_idx == count and last_attr_obj is a and last_attr_bound is not None:
                # reuse cached bound method
                val2 = last_attr_bound
            else:
                val2 = getattr(a, b)
                last_attr_idx = count
                last_attr_obj = a
                last_attr_bound = val2
            if sp >= cap:
                cap = cap * 2
                stack.extend([None] * (cap - len(stack)))
            stack[sp] = val2(*val); sp += 1
        elif op == OP_JMP_IF_FALSE:
            # conditional forward jump; offset is relative to next index
            count = PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1
            sp -= 1; a = stack[sp]
            if not PyObject_IsTrue(a):
                i += count
        elif op == OP_JMP:
            count = <Py_ssize_t>PyList_GET_ITEM(opcodes, i); i += 1
            i += count
        elif op == OP_SLICE:
            count = PyLong_AsSsize_t(<object>PyList_GET_ITEM(opcodes, i)); i += 1  # flags
            # fast paths:
            if count == 0:
                sp -= 1; val = stack[sp]
                stack[sp] = val[:]; sp += 1
            elif count == 3:
                # only lower and upper (1|2)
                sp -= 1; upper = stack[sp]
                sp -= 1; lower = stack[sp]
                sp -= 1; val = stack[sp]
                stack[sp] = val[lower:upper]; sp += 1
            else:
                # generic slow path
                step = upper = lower = None
                if count & 4:
                    sp -= 1; step = stack[sp]
                if count & 2:
                    sp -= 1; upper = stack[sp]
                if count & 1:
                    sp -= 1; lower = stack[sp]
                sp -= 1; val = stack[sp]
                stack[sp] = val[slice(lower, upper, step)]; sp += 1
        else:
            raise RuntimeError(f"Unknown opcode {op}")
    return stack[sp-1] if sp > 0 else None
