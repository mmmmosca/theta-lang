# fastpaths.py
# Pure-Python implementations designed to be Cythonized for speed.
# You can build this as a C extension via setup.py (cythonize), and
# theta.py will import the accelerated functions when available.

from typing import List, Tuple, Any, Dict
import ast

def cy_tokenize_and_transform(s: str) -> str:
    # Inline, quote/bracket-aware scan; identical to Python fallback in theta.py
    # Minimal dependencies to keep Cython compilation simple.
    def strip_comments_local(t: str) -> str:
        out = []
        in_sq = False
        in_dq = False
        i = 0
        L = len(t)
        while i < L:
            ch = t[i]
            if ch == "'" and not in_dq:
                in_sq = not in_sq
                out.append(ch)
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
                out.append(ch)
            elif ch == '#' and not in_sq and not in_dq:
                break
            else:
                out.append(ch)
            i += 1
        return ''.join(out).rstrip()

    t = strip_comments_local(s)
    # Semicolons-in-brackets: keep simple replacement respecting quotes
    out2 = []
    depth_sq = 0
    in_sq = False
    in_dq = False
    i = 0
    L = len(t)
    while i < L:
        ch = t[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            out2.append(ch)
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
            out2.append(ch)
        elif not in_sq and not in_dq:
            if ch == '[':
                depth_sq += 1
                out2.append(ch)
            elif ch == ']':
                depth_sq -= 1
                out2.append(ch)
            elif ch == ';' and depth_sq > 0:
                out2.append(',')
            else:
                out2.append(ch)
        else:
            out2.append(ch)
        i += 1
    t = ''.join(out2).replace('.in(', '.__in__(')

    # Blocks: simple transform to __BLOCK__(...)
    # Keep identical logic to theta.py for safety
    def transform_blocks_local(s2: str) -> str:
        out_parts = []
        i2 = 0
        in_sq2 = False
        in_dq2 = False
        depth = 0
        while i2 < len(s2):
            ch2 = s2[i2]
            if ch2 == "'" and not in_dq2:
                in_sq2 = not in_sq2
                out_parts.append(ch2)
                i2 += 1
                continue
            if ch2 == '"' and not in_sq2:
                in_dq2 = not in_dq2
                out_parts.append(ch2)
                i2 += 1
                continue
            if not in_sq2 and not in_dq2 and ch2 == '{':
                # capture until matching '}'
                i2 += 1
                depth = 1
                body_buf = []
                in_sq3 = False
                in_dq3 = False
                while i2 < len(s2):
                    ch3 = s2[i2]
                    if ch3 == "'" and not in_dq3:
                        in_sq3 = not in_sq3
                        body_buf.append(ch3)
                        i2 += 1
                        continue
                    if ch3 == '"' and not in_sq3:
                        in_dq3 = not in_dq3
                        body_buf.append(ch3)
                        i2 += 1
                        continue
                    if not in_sq3 and not in_dq3:
                        if ch3 == '{':
                            depth += 1
                            body_buf.append(ch3)
                            i2 += 1
                            continue
                        if ch3 == '}':
                            depth -= 1
                            if depth == 0:
                                i2 += 1
                                break
                            body_buf.append(ch3)
                            i2 += 1
                            continue
                    body_buf.append(ch3)
                    i2 += 1
                body_str = ''.join(body_buf)
                out_parts.append(f"__BLOCK__({repr(body_str)})")
                continue
            out_parts.append(ch2)
            i2 += 1
        return ''.join(out_parts)

    t = transform_blocks_local(t)

    # Operators and NOT
    out = []
    in_sq = False
    in_dq = False
    depth_round = 0
    depth_sq = 0
    depth_curly = 0
    i = 0
    L = len(t)
    while i < L:
        ch = t[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            out.append(ch)
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            out.append(ch)
            i += 1
            continue
        if not in_sq and not in_dq:
            if ch == '(':
                depth_round += 1
                out.append(ch)
                i += 1
                continue
            if ch == ')':
                depth_round -= 1
                out.append(ch)
                i += 1
                continue
            if ch == '[':
                depth_sq += 1
                out.append(ch)
                i += 1
                continue
            if ch == ']':
                depth_sq -= 1
                out.append(ch)
                i += 1
                continue
            if ch == '{':
                depth_curly += 1
                out.append(ch)
                i += 1
                continue
            if ch == '}':
                depth_curly -= 1
                out.append(ch)
                i += 1
                continue
            if ch == '&' and i + 1 < L and t[i+1] == '&':
                out.append(' and ')
                i += 2
                continue
            if ch == '|' and i + 1 < L and t[i+1] == '|':
                out.append(' or ')
                i += 2
                continue
            if ch == '!':
                if i + 1 < L and t[i+1] == '=':
                    out.append('!')
                    i += 1
                    continue
                out.append(' not ')
                i += 1
                continue
        out.append(ch)
        i += 1
    s2 = ''.join(out)
    # Leave complex constructs to Python helpers (will be faster overall when Cythonized here)
    # but we keep this function focused on the hot operator path.
    return s2

def cy_transform_when_else(s: str) -> str:
    # Quote/bracket-aware transform: 'A when B else C' -> '(A) if (B) else (C)'
    out = s
    while True:
        idx_when = -1
        depth = 0
        in_sq = False
        in_dq = False
        i = 0
        L = len(out)
        while i < L:
            ch = out[i]
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif not in_sq and not in_dq:
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                if out.startswith(' when ', i):
                    idx_when = i
                    break
            i += 1
        if idx_when == -1:
            break

        j = idx_when + len(' when ')
        depth = 0
        in_sq = False
        in_dq = False
        idx_else = -1
        while j < L:
            ch = out[j]
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif not in_sq and not in_dq:
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                if depth == 0 and out.startswith(' else ', j):
                    idx_else = j
                    break
            j += 1
        if idx_else == -1:
            raise SyntaxError("Malformed 'when' expression: missing corresponding 'else'")

        left = out[:idx_when]
        mid = out[idx_when + len(' when '):idx_else]
        right = out[idx_else + len(' else '):]
        out = f"({left.strip()}) if ({mid.strip()}) else ({right.strip()})"
    return out

def cy_transform_matches(s: str) -> str:
    out = s
    while True:
        idx_when = -1
        depth = 0
        in_sq = False
        in_dq = False
        i = 0
        L = len(out)
        while i < L:
            ch = out[i]
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif not in_sq and not in_dq:
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                if out.startswith(' matches ', i):
                    idx_when = i
                    break
            i += 1
        if idx_when == -1:
            break

        j = idx_when + len(' matches ')
        depth = 0
        in_sq = False
        in_dq = False
        idx_return = -1
        while j < L:
            ch = out[j]
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif not in_sq and not in_dq:
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                if depth == 0 and out.startswith(' return ', j):
                    idx_return = j
                    break
            j += 1
        if idx_return == -1:
            raise SyntaxError("Malformed 'matches' expression: missing 'return'")

        k = idx_return + len(' return ')
        depth = 0
        in_sq = False
        in_dq = False
        idx_else = -1
        while k < L:
            ch = out[k]
            if ch == "'" and not in_dq:
                in_sq = not in_sq
            elif ch == '"' and not in_sq:
                in_dq = not in_dq
            elif not in_sq and not in_dq:
                if ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                if depth == 0 and out.startswith(' else ', k):
                    idx_else = k
                    break
            k += 1

        if idx_else == -1:
            left = out[:idx_when]
            pattern = out[idx_when + len(' matches '):idx_return]
            ret = out[idx_return + len(' return '):].strip()
            else_expr = 'None'
            suffix = ''
        else:
            left = out[:idx_when]
            pattern = out[idx_when + len(' matches '):idx_return]
            ret = out[idx_return + len(' return '):idx_else]
            m = idx_else + len(' else ')
            depth3 = 0
            in_sq3 = False
            in_dq3 = False
            else_end = L
            while m < L:
                ch3 = out[m]
                if ch3 == "'" and not in_dq3:
                    in_sq3 = not in_sq3
                elif ch3 == '"' and not in_sq3:
                    in_dq3 = not in_dq3
                elif not in_sq3 and not in_dq3:
                    if ch3 in '([{':
                        depth3 += 1
                    elif ch3 in ')]}':
                        if depth3 == 0:
                            else_end = m
                            break
                        depth3 -= 1
                    elif depth3 == 0 and ch3 == ',':
                        else_end = m
                        break
                m += 1
            else_expr = out[idx_else + len(' else '):else_end]
            suffix = out[else_end:]

        # find subject start by scanning back
        k2 = idx_when - 1
        depth2 = 0
        in_sq2 = False
        in_dq2 = False
        subj_start = 0
        while k2 >= 0:
            ch2 = out[k2]
            if ch2 == "'" and not in_dq2:
                in_sq2 = not in_sq2
            elif ch2 == '"' and not in_sq2:
                in_dq2 = not in_dq2
            elif not in_sq2 and not in_dq2:
                if ch2 in ')]}':
                    depth2 += 1
                elif ch2 in '([{':
                    if depth2 == 0:
                        subj_start = k2 + 1
                        break
                    depth2 -= 1
                elif depth2 == 0 and ch2 == ',':
                    subj_start = k2 + 1
                    break
            k2 -= 1
        else:
            subj_start = 0

        prefix = out[:subj_start]
        subject_fragment = out[subj_start:idx_when]
        subj_str = subject_fragment.strip()
        patt_str = pattern.strip()
        try:
            left_ast = ast.parse(subj_str, mode='eval').body
            right_ast = ast.parse(patt_str, mode='eval').body
        except Exception:
            left_ast = right_ast = None

        def _contains_name(node):
            if node is None:
                return False
            if isinstance(node, ast.Name):
                return True
            for child in ast.iter_child_nodes(node):
                if _contains_name(child):
                    return True
            return False

        if left_ast is not None and right_ast is not None and _contains_name(left_ast) and not _contains_name(right_ast):
            subj_str, patt_str = patt_str, subj_str

        new_expr = f"match.matches({subj_str}, {repr(patt_str)}, {repr(ret.strip())}, {repr(else_expr.strip())}, __GLOBALS__)"
        out = prefix + new_expr + suffix
    return out

def cy_transform_semicolons_in_brackets(s: str) -> str:
    out = []
    depth_sq = 0
    in_sq = False
    in_dq = False
    i = 0
    L = len(s)
    while i < L:
        ch = s[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            out.append(ch)
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
            out.append(ch)
        elif not in_sq and not in_dq:
            if ch == '[':
                depth_sq += 1
                out.append(ch)
            elif ch == ']':
                depth_sq -= 1
                out.append(ch)
            elif ch == ';' and depth_sq > 0:
                out.append(',')
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return ''.join(out)

# Optional: Cython-friendly list pattern matcher helper
# (we keep using the Python version in theta.py for correctness; this is here for future use)

def cy_list_match_ops(seq, ops, bindings, is_str: bool):
    # Supports simple ops natively; handles 'sub'/'node' by recursive pattern matching here.
    def match_node(val, pat) -> bool:
        kind = pat[0]
        if kind == 'lit':
            return val == pat[1]
        if kind == 'wild':
            return True
        if kind == 'var':
            bindings[pat[1]] = val
            return True
        if kind == 'list':
            parts = pat[1]
            seq2 = val if isinstance(val, str) else list(val) if isinstance(val, (list, tuple)) else None
            if seq2 is None:
                return False
            has_rest = bool(parts and parts[-1][0] == 'rest')
            core = parts[:-1] if has_rest else parts
            ops2 = []
            for p in core:
                if isinstance(val, str) and p[0] == 'lit' and isinstance(p[1], str):
                    ops2.append(('strlit', p[1]))
                elif p[0] in ('lit','var','wild'):
                    ops2.append((p[0], p[1] if p[0] != 'wild' else None))
                elif p[0] == 'list':
                    ops2.append(('sub', p))
                else:
                    ops2.append(('node', p))
            idx2 = cy_list_match_ops(seq2, ops2, bindings, isinstance(val, str))
            if idx2 < 0:
                return False
            if has_rest:
                rest_name = parts[-1][1]
                tail = seq2[idx2:]
                bindings[rest_name] = tail
                return True
            return idx2 == (len(seq2) if not isinstance(val, str) else len(seq2))
        return False

    idx = 0
    n = len(seq)
    for op, arg in ops:
        if op == 'strlit':
            chunk = arg
            end = idx + len(chunk)
            if seq[idx:end] != chunk:
                return -1
            idx = end
        elif op == 'lit':
            if idx >= n or seq[idx] != arg:
                return -1
            idx += 1
        elif op == 'var':
            if idx >= n:
                return -1
            bindings[arg] = seq[idx]
            idx += 1
        elif op == 'wild':
            if idx >= n:
                return -1
            idx += 1
        elif op in ('sub','node'):
            if idx >= n:
                return -1
            if not match_node(seq[idx], arg):
                return -1
            idx += 1
        else:
            return -2
    return idx

def cy_split_top_level_semicolons(body_str: str):
    parts = []
    buf = []
    depth_round = 0
    depth_sq = 0
    depth_curly = 0
    in_sq = False
    in_dq = False
    i = 0
    L = len(body_str)
    while i < L:
        ch = body_str[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            buf.append(ch)
            i += 1
            continue
        if not in_sq and not in_dq:
            if ch == '(':
                depth_round += 1
                buf.append(ch)
                i += 1
                continue
            if ch == ')':
                depth_round -= 1
                buf.append(ch)
                i += 1
                continue
            if ch == '[':
                depth_sq += 1
                buf.append(ch)
                i += 1
                continue
            if ch == ']':
                depth_sq -= 1
                buf.append(ch)
                i += 1
                continue
            if ch == '{':
                depth_curly += 1
                buf.append(ch)
                i += 1
                continue
            if ch == '}':
                depth_curly -= 1
                buf.append(ch)
                i += 1
                continue
            if ch == ';' and depth_round == 0 and depth_sq == 0 and depth_curly == 0:
                parts.append(''.join(buf).strip())
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append(''.join(buf).strip())
    return parts

def cy_balance_brackets(s: str) -> int:
    depth_round = 0
    depth_sq = 0
    depth_curly = 0
    in_sq = False
    in_dq = False
    i = 0
    L = len(s)
    while i < L:
        ch = s[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == '(':
                depth_round += 1
            elif ch == ')':
                depth_round -= 1
            elif ch == '[':
                depth_sq += 1
            elif ch == ']':
                depth_sq -= 1
            elif ch == '{':
                depth_curly += 1
            elif ch == '}':
                depth_curly -= 1
        i += 1
    return depth_round + depth_sq + depth_curly

def cy_parse_guarded_return(expr: str):
    # Parse 'X when Y' at top level (no 'else'), respecting quotes/brackets.
    # Returns (left_expr, cond_expr) or (None, None) if no top-level ' when ' found.
    depth_round = 0
    depth_sq = 0
    depth_curly = 0
    in_sq = False
    in_dq = False
    i = 0
    L = len(expr)
    idx_when = -1
    while i < L:
        ch = expr[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            i += 1
            continue
        if not in_sq and not in_dq:
            if ch == '(':
                depth_round += 1
            elif ch == ')':
                depth_round -= 1
            elif ch == '[':
                depth_sq += 1
            elif ch == ']':
                depth_sq -= 1
            elif ch == '{':
                depth_curly += 1
            elif ch == '}':
                depth_curly -= 1
            if depth_round == 0 and depth_sq == 0 and depth_curly == 0:
                if expr.startswith(' when ', i):
                    idx_when = i
                    break
        i += 1
    if idx_when == -1:
        return None, None
    left = expr[:idx_when].strip()
    cond = expr[idx_when + len(' when '):].strip()
    return left, cond

# ----------------------
# Minimal Bytecode VM
# ----------------------

# Opcodes
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
OP_MAKE_LIST = 14  # arg: count
OP_NEG = 15
OP_POS = 16
OP_NOT = 17
OP_AND = 18
OP_OR = 19
OP_MAKE_TUPLE = 20  # arg: count
OP_INDEX_CONST = 21  # arg: const index
OP_INDEX_STACK = 22  # index value from stack
OP_CALL_NAME = 23    # args: name_idx, argc
OP_CALL_ATTR = 24    # args: attr_name_idx, argc (object and args on stack)
OP_JMP_IF_FALSE = 25 # args: offset (relative forward jump)
OP_SLICE = 26        # args: flags bitmask (1=has_lower,2=has_upper,4=has_step)
OP_JMP = 27          # args: offset (relative forward jump)

def vm_exec(opcodes, consts, names, scope):
    """Execute a compact opcode program.
    Optimized Python implementation: minimizes attribute lookups and branching.
    """
    stack = []
    push = stack.append
    pop = stack.pop
    getc = consts.__getitem__
    getn = names.__getitem__
    gets = scope.__getitem__
    sets = scope.__setitem__
    i = 0
    L = len(opcodes)
    while i < L:
        op = opcodes[i]; i += 1
        if op == OP_PUSH_CONST:
            idx = opcodes[i]; i += 1
            push(getc(idx))
        elif op == OP_LOAD_NAME:
            idx = opcodes[i]; i += 1
            push(gets(getn(idx)))
        elif op == OP_STORE_NAME:
            idx = opcodes[i]; i += 1
            sets(getn(idx), pop())
        elif op == OP_ADD:
            b = pop(); a = pop(); push(a + b)
        elif op == OP_SUB:
            b = pop(); a = pop(); push(a - b)
        elif op == OP_MUL:
            b = pop(); a = pop(); push(a * b)
        elif op == OP_DIV:
            b = pop(); a = pop(); push(a / b)
        elif op == OP_EQ:
            b = pop(); a = pop(); push(a == b)
        elif op == OP_NE:
            b = pop(); a = pop(); push(a != b)
        elif op == OP_LT:
            b = pop(); a = pop(); push(a < b)
        elif op == OP_LE:
            b = pop(); a = pop(); push(a <= b)
        elif op == OP_GT:
            b = pop(); a = pop(); push(a > b)
        elif op == OP_GE:
            b = pop(); a = pop(); push(a >= b)
        elif op == OP_MAKE_LIST:
            count = opcodes[i]; i += 1
            if count:
                items = stack[-count:]
                del stack[-count:]
                push(items)
            else:
                push([])
        elif op == OP_MAKE_TUPLE:
            count = opcodes[i]; i += 1
            if count:
                items = stack[-count:]
                del stack[-count:]
                push(tuple(items))
            else:
                push(())
        elif op == OP_NEG:
            a = pop(); push(-a)
        elif op == OP_POS:
            a = pop(); push(+a)
        elif op == OP_NOT:
            a = pop(); push(not a)
        elif op == OP_AND:
            b = pop(); a = pop();
            # Inline truthiness via PyObject_IsTrue equivalent; Python-level fallback
            push((a is True or (a and a != 0)) and (b is True or (b and b != 0)))
        elif op == OP_OR:
            b = pop(); a = pop();
            push((a is True or (a and a != 0)) or (b is True or (b and b != 0)))
        elif op == OP_INDEX_CONST:
            idx = opcodes[i]; i += 1
            val = pop(); push(val[getc(idx)])
        elif op == OP_INDEX_STACK:
            idx_val = pop(); val = pop(); push(val[idx_val])
        elif op == OP_CALL_NAME:
            name_idx = opcodes[i]; i += 1
            argc = opcodes[i]; i += 1
            args = []
            for _ in range(argc):
                args.append(pop())
            args.reverse()
            func = gets(getn(name_idx))
            push(func(*args))
        elif op == OP_CALL_ATTR:
            attr_idx = opcodes[i]; i += 1
            argc = opcodes[i]; i += 1
            args = []
            for _ in range(argc):
                args.append(pop())
            args.reverse()
            obj = pop()
            func = getattr(obj, getn(attr_idx))
            push(func(*args))
        elif op == OP_JMP_IF_FALSE:
            offset = opcodes[i]; i += 1
            cond = pop()
            if not cond:
                i += offset
        elif op == OP_JMP:
            offset = opcodes[i]; i += 1
            i += offset
        elif op == OP_SLICE:
            flags = opcodes[i]; i += 1
            if flags == 0:
                seq = pop(); push(seq[:])
            elif flags == 3:
                upper = pop(); lower = pop(); seq = pop(); push(seq[lower:upper])
            else:
                step = upper = lower = None
                if flags & 4:
                    step = pop()
                if flags & 2:
                    upper = pop()
                if flags & 1:
                    lower = pop()
                seq = pop()
                push(seq[slice(lower, upper, step)])
        else:
            raise RuntimeError(f"Unknown opcode {op}")
    return stack[-1] if stack else None

# Prefer Cython vm_exec if available
try:
    from fastpaths_vm import vm_exec as cy_vm_exec
    vm_exec = cy_vm_exec
except Exception:
    pass

def simple_expr_to_bytecode(expr_tokens):
    """Attempt to compile a simple token sequence to bytecode.
    Supports: integers, floats, strings (already parsed tokens), names, + - * /, == != < <= > >=, list literals [ ... ] with commas.
    expr_tokens: a token list structure as produced by tokenizer (assumed simple).
    Returns (opcodes, consts, names) or None on unsupported.
    """
    # Minimal shunting-yard to RPN then to opcodes.
    # Expect tokens as dicts: {'type': 'number'|'name'|'string'|'op'|'['|']'|',' , 'value': ...}
    output = []
    ops = []
    precedence = {'==':1,'!=':1,'<':1,'<=':1,'>':1,'>=':1,'+':2,'-':2,'*':3,'/':3}
    def push_op(o):
        while ops and ops[-1] != '(' and precedence.get(ops[-1],0) >= precedence.get(o,0):
            output.append({'type':'op','value':ops.pop()})
        ops.append(o)
    i = 0
    # Simple list literal handling: when encountering '[', parse items split by commas at zero bracket depth.
    try:
        while i < len(expr_tokens):
            t = expr_tokens[i]; i += 1
            tt = t.get('type')
            if tt in ('number','string'):
                output.append(t)
            elif tt == 'name':
                output.append(t)
            elif tt == 'op':
                o = t['value']
                if o in precedence:
                    push_op(o)
                elif o == '(':
                    ops.append('(')
                elif o == ')':
                    while ops and ops[-1] != '(':
                        output.append({'type':'op','value':ops.pop()})
                    if not ops or ops[-1] != '(':
                        return None
                    ops.pop()
                else:
                    return None
            elif tt == '[':
                # parse list items until matching ']'
                depth = 1
                items = []
                current = []
                while i < len(expr_tokens) and depth > 0:
                    t2 = expr_tokens[i]; i += 1
                    if t2.get('type') == '[':
                        depth += 1; current.append(t2)
                    elif t2.get('type') == ']':
                        depth -= 1
                        if depth == 0:
                            if current:
                                items.append(current)
                            break
                        else:
                            current.append(t2)
                    elif t2.get('type') == 'comma':
                        if depth == 1:
                            items.append(current); current = []
                        else:
                            current.append(t2)
                    else:
                        current.append(t2)
                if depth != 0:
                    return None
                # For each item tokens, we only support simple literals/names for now
                output.append({'type':'list','items':items})
            else:
                return None
        while ops:
            o = ops.pop()
            if o == '(':
                return None
            output.append({'type':'op','value':o})
    except Exception:
        return None

    # Convert RPN to opcodes
    consts = []
    names = []
    opcodes = []

    def const_index(val):
        try:
            return consts.index(val)
        except ValueError:
            consts.append(val); return len(consts)-1

    def name_index(n):
        try:
            return names.index(n)
        except ValueError:
            names.append(n); return len(names)-1

    for node in output:
        tt = node['type']
        if tt == 'number' or tt == 'string':
            idx = const_index(node['value'])
            opcodes.extend([OP_PUSH_CONST, idx])
        elif tt == 'name':
            idx = name_index(node['value'])
            opcodes.extend([OP_LOAD_NAME, idx])
        elif tt == 'list':
            # Each item must compile to a single push; for now only support literal tokens inside
            item_count = 0
            for item_tokens in node['items']:
                # only single literal or name supported for MVP
                if len(item_tokens) == 1 and item_tokens[0]['type'] in ('number','string','name'):
                    it = item_tokens[0]
                    if it['type'] in ('number','string'):
                        idx = const_index(it['value'])
                        opcodes.extend([OP_PUSH_CONST, idx])
                    elif it['type'] == 'name':
                        idx = name_index(it['value'])
                        opcodes.extend([OP_LOAD_NAME, idx])
                    item_count += 1
                else:
                    return None
            opcodes.extend([OP_MAKE_LIST, item_count])
        elif tt == 'op':
            o = node['value']
            table = {'+':OP_ADD,'-':OP_SUB,'*':OP_MUL,'/':OP_DIV,
                     '==':OP_EQ,'!=':OP_NE,'<':OP_LT,'<=':OP_LE,'>':OP_GT,'>=':OP_GE}
            if o not in table:
                return None
            opcodes.append(table[o])
        else:
            return None

    return (opcodes, consts, names)
