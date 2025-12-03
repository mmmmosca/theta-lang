print("Theta v1.2.0 - made by Mosca\n")
import math
import ast
import importlib
import sys
import traceback
import theta_types as tt

# Simple debug/verbose toggle. Set via command-line flags `--verbose` or `--debug`.
DEBUG = False

def log(*args, **kwargs):
    """Conditional logger controlled by the module-level `DEBUG` flag.

    Behaves like `print()` when `DEBUG` is True; otherwise does nothing.
    """
    if DEBUG:
        print(*args, **kwargs)


def report_error(exc: Exception, context: str | None = None):
    """Print a concise error message with optional context and show a full
    traceback only when `DEBUG` is True.

    `context` should be a short string such as `"file.th:23"` or
    `"while evaluating function foo"` to help the user locate the error.
    """
    ctx = f" ({context})" if context else ""
    # concise one-line message
    try:
        msg = str(exc)
    except Exception:
        msg = repr(exc)
    print(f"Error{ctx}: {exc.__class__.__name__}: {msg}")
    if DEBUG:
        # full traceback for debugging
        traceback.print_exc()

class ThetaArray:
    """A lightweight wrapper for Theta arrays that preserves display style.

    Behaves like a Python list for indexing/len/iteration, but prints using
    Theta's semicolon notation. If `double_brackets` is True, the printed
    form will include an extra outer pair of brackets (e.g. `[[1;2;3]]`).
    """
    def __init__(self, items, double_brackets=False):
        # store as a plain list for operations
        self._items = list(items)
        self._double = bool(double_brackets)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __eq__(self, other):
        if isinstance(other, ThetaArray):
            return self._items == other._items
        if isinstance(other, list):
            return self._items == other
        return False

    def __add__(self, other):
        # Support concatenation with another ThetaArray or a Python list/tuple
        if isinstance(other, ThetaArray):
            return ThetaArray(self._items + other._items, double_brackets=self._double)
        if isinstance(other, list):
            return ThetaArray(self._items + other, double_brackets=self._double)
        if isinstance(other, tuple):
            return ThetaArray(self._items + list(other), double_brackets=self._double)
        return NotImplemented

    def __radd__(self, other):
        # Support list + ThetaArray
        if isinstance(other, list):
            return ThetaArray(other + self._items, double_brackets=self._double)
        return NotImplemented

    def __repr__(self):
        # Use semicolon-separated items for display
        def fmt(x):
            if isinstance(x, ThetaArray):
                return x.__repr__()
            return repr(x)

        inner = ';'.join(fmt(x) for x in self._items)
        s = f"[{inner}]"
        if self._double:
            return f"[{s}]"
        return s


# Simple in-memory function registry: name -> dict with keys:
#   'params': [param names], 'body': body string, 'is_block': bool
FUNCTIONS = {}
# Immutable variables registry: name -> {'expr': str, 'value': evaluated_value}
VARS = {}
# Blueprints registry (struct-like objects with behavior / side effects)
BLUEPRINTS = {}


def register_blueprint(name, obj):
    """Register a blueprint object under `name`.

    A blueprint is any Python object whose attributes are callable and handle
    side effects. Blueprints are accessible in expressions as `name.attr(...)`.
    """
    if not name.isidentifier():
        raise SyntaxError(f"Invalid blueprint name '{name}'")
    BLUEPRINTS[name] = obj


def _import_blueprint_from_th(name: str, base_dir: str | None = None) -> bool:
    """Attempt to import a blueprint definition from a Theta `.th` file.

    Looks for `<name>.th` in `base_dir` (or current working directory if None),
    scans for a `blueprint <name> [ ... ]` block, and registers the resulting
    blueprint. Returns True on success, False otherwise.
    """
    import os
    dir_to_use = base_dir or os.getcwd()
    th_path = os.path.join(dir_to_use, f"{name}.th")
    if not os.path.isfile(th_path):
        return False
    try:
        with open(th_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return False

    # find blueprint <name> [ ... ] and parse defs
    found = False
    i = 0
    while i < len(lines):
        raw = lines[i]; i += 1
        line = strip_comments(raw).strip()
        if not line or not line.startswith('blueprint '):
            continue
        rest = line[len('blueprint '):].strip()
        if not rest.startswith(name):
            # different blueprint block
            continue
        # expect '[' and collect until ']'
        if '[' not in rest:
            continue
        parts = rest.split('[', 1)
        after = parts[1]
        body_lines = []
        if ']' in after:
            inner = after.split(']', 1)[0]
            body_lines = inner.split('\n')
        else:
            if after.strip():
                body_lines.append(after)
            while i < len(lines):
                more = lines[i]; i += 1
                if ']' in more:
                    body_lines.append(more.split(']', 1)[0])
                    break
                body_lines.append(more)
        # parse defs
        defs = []
        for raw2 in body_lines:
            line2 = strip_comments(raw2).strip()
            if not line2:
                continue
            if line2.startswith('def '):
                fn_def = line2[len('def '):].strip()
                fname, fparams, fexpr = parseFunction(fn_def)
                defs.append((fname, fparams, fexpr))
        bp_obj = create_blueprint_from_defs(name, defs)
        register_blueprint(name, bp_obj)
        found = True
        break
    return found


def create_blueprint_from_defs(name, defs):
    """Create a blueprint object from a list of function defs.

    `defs` is a list of tuples (method_name, params, return_expr).
    The returned object has callable attributes for each method.
    """
    methods = {}

    for mname, params, body in defs:
        is_block = body.startswith('{') and body.endswith('}')
        body_content = body[1:-1].strip() if is_block else body.strip()

        def make_method(body_str, params_list, is_blocked):
            def method(*call_args):
                if len(call_args) != len(params_list):
                    raise TypeError(f"{mname}() takes {len(params_list)} positional argument(s) but {len(call_args)} were given")
                # Build local vars: globals then parameters
                local = get_global_var_values()
                local.update({p: v for p, v in zip(params_list, call_args)})
                if is_blocked:
                    # block body uses process_block; pass only param locals to avoid mutating globals
                    param_locals = {p: local[p] for p in params_list}
                    return process_block(body_str, param_locals)
                else:
                    return evaluate_expression(body_str, local)

            return method

        meth = make_method(body_content, params, is_block)
        methods[mname] = meth
        # If user named method 'in', also expose it as '__in__' for parsing
        if mname == 'in':
            methods['__in__'] = meth

    # Create a simple object to hold methods
    bp = type(f"Blueprint_{name}", (), {})()
    for k, fn in methods.items():
        setattr(bp, k, fn)
    return bp


class IoBlueprint:
    """A minimal IO blueprint exposing side-effectful functions."""

    def out(self, *args):
        # Simple output function: print the joined args with spaces
        print(*args)
        return None


# Register a default `io` blueprint
_io_instance = IoBlueprint()
# assign an `in` attribute (can't use 'in' as a Python def name) to read from stdin
setattr(_io_instance, 'in', lambda prompt=None: input(prompt))
# also provide a valid Python identifier fallback for parsing: '__in__'
setattr(_io_instance, '__in__', getattr(_io_instance, 'in'))
register_blueprint('io', _io_instance)


# --- TM blueprint (Python-side) -------------------------------------------
class _TMBlueprint:
    """Simple Turing Machine runner exposed to Theta as blueprint `tm`.

    Usage from Theta:
      tm.run(transitions, tape, head, state, max_steps)

    `transitions` should be a list/array of 5-tuples/lists:
      [state, symbol, new_symbol, direction, next_state]
    Direction should be -1, 0, or +1.
    """

    def _to_py(self, obj):
        # Convert ThetaArray-like objects (or nested lists) into native Python lists/values
        try:
            from types import SimpleNamespace
        except Exception:
            SimpleNamespace = None
        # Heuristic: if object has '_items' attribute (ThetaArray), unwrap
        if hasattr(obj, '_items'):
            return [self._to_py(x) for x in obj._items]
        # If it's a list/tuple already, map elements
        if isinstance(obj, (list, tuple)):
            return [self._to_py(x) for x in obj]
        return obj

    def run(self, transitions, tape, head, state, steps=1000):
        trans = self._to_py(transitions)
        tape_py = self._to_py(tape)
        # ensure tape is a list
        if not isinstance(tape_py, list):
            tape_py = [tape_py]

        # If no transitions provided, use a default unary-increment TM
        if not trans:
            trans = [[ 'q0', 1, 1, 1, 'q0' ], [ 'q0', 0, 1, 0, 'qa' ]]

        # Build transition dict for quick lookup
        table = {}
        for entry in trans:
            if not entry or len(entry) < 5:
                continue
            s, sym, new_sym, direction, ns = entry
            table[(s, sym)] = (new_sym, int(direction), ns)

        head = int(head)
        for _ in range(int(steps)):
            sym = tape_py[head] if 0 <= head < len(tape_py) else 0
            key = (state, sym)
            if key not in table:
                # halt
                return (tape_py, head, state)
            new_sym, direction, next_state = table[key]
            # write
            if 0 <= head < len(tape_py):
                tape_py[head] = new_sym
            else:
                # extend tape with blanks until head index
                while len(tape_py) <= head:
                    tape_py.append(0)
                tape_py[head] = new_sym
            head = head + direction
            state = next_state
        return (tape_py, head, state)


register_blueprint('tm', _TMBlueprint())


class MatchBlueprint:
    """Runtime helper that performs pattern matching and evaluates branches.

    The `matches(subject, pattern_str, success_expr_str, else_expr_str, globals_map)`
    method parses `pattern_str` using Python's AST (after semicolon->comma
    transform), attempts to match `subject` against the pattern and, on a
    successful match, evaluates `success_expr_str` with the found bindings
    merged into the provided `globals_map`.
    """

    def _pattern_from_ast(self, node):
        # Literals
        if isinstance(node, ast.Constant):
            return ("lit", node.value)

        # Backwards-compat: older Python versions used ast.Num / ast.Str
        if node.__class__.__name__ == 'Num':
            return ("lit", getattr(node, 'n'))
        if node.__class__.__name__ == 'Str':
            return ("lit", getattr(node, 's'))
        if isinstance(node, ast.Name):
            if node.id == '_':
                return ("wild", None)
            return ("var", node.id)
        if isinstance(node, ast.List):
            parts = []
            for elt in node.elts:
                if isinstance(elt, ast.Starred):
                    if isinstance(elt.value, ast.Name):
                        parts.append(("rest", elt.value.id))
                    else:
                        raise SyntaxError("Invalid rest pattern")
                else:
                    parts.append(self._pattern_from_ast(elt))
            return ("list", parts)
        # Fallback: try tuple as list
        if isinstance(node, ast.Tuple):
            return ("list", [self._pattern_from_ast(e) for e in node.elts])
        raise SyntaxError(f"Unsupported pattern element: {type(node).__name__}")

    def _match(self, value, pattern, bindings):
        kind = pattern[0]
        if kind == 'lit':
            return value == pattern[1]
        if kind == 'wild':
            return True
        if kind == 'var':
            bindings[pattern[1]] = value
            return True
        if kind == 'list':
            parts = pattern[1]
            # handle rest pattern if present
            if parts and parts[-1][0] == 'rest':
                fixed = parts[:-1]
                rest_name = parts[-1][1]
                if not isinstance(value, (list, tuple, ThetaArray)):
                    return False
                vals = list(value)
                if len(vals) < len(fixed):
                    return False
                for p, v in zip(fixed, vals[:len(fixed)]):
                    if not self._match(v, p, bindings):
                        return False
                bindings[rest_name] = vals[len(fixed):]
                return True
            else:
                if not isinstance(value, (list, tuple, ThetaArray)):
                    return False
                vals = list(value)
                if len(vals) != len(parts):
                    return False
                for p, v in zip(parts, vals):
                    if not self._match(v, p, bindings):
                        return False
                return True
        return False

    def matches(self, subj, pattern_str, success_expr_str, else_expr_str, globals_map):
        # Pre-process pattern string to convert semicolons and then parse AST
        pat_s = transform_semicolons_in_brackets(pattern_str)
        try:
            pat_ast = ast.parse(pat_s, mode='eval').body
        except Exception as e:
            raise SyntaxError(f"Invalid pattern: {e}")
        pattern = self._pattern_from_ast(pat_ast)

        bindings = {}
        ok = self._match(subj, pattern, bindings)
        if ok:
            # Merge globals and bindings and evaluate the success expression string
            local = dict(globals_map or {})
            # copy bindings so ThetaArray and others are native
            local.update(bindings)
            return evaluate_expression(success_expr_str, local)
        else:
            return evaluate_expression(else_expr_str, globals_map or {})


register_blueprint('match', MatchBlueprint())


class PythonBlueprint:
    """Expose controlled access to Python modules/functions from Theta.

    Usage from Theta:
      python.call('math.sqrt', 9)

    This resolves the dotted path by importing the top-level module and
    walking attributes. Only explicit calls via `call` are supported to
    avoid enabling arbitrary attribute-based calls accidentally.
    """
    def call(self, dotted_name, *args):
        if not isinstance(dotted_name, str):
            raise TypeError("python.call first argument must be a string like 'math.sqrt'")
        if dotted_name.strip() == '':
            raise ValueError('Empty module path')
        parts = dotted_name.split('.')
        mod_name = parts[0]
        try:
            obj = importlib.import_module(mod_name)
        except Exception as e:
            raise ImportError(f"Cannot import module '{mod_name}': {e}")
        for attr in parts[1:]:
            try:
                obj = getattr(obj, attr)
            except Exception as e:
                raise AttributeError(f"Module/object has no attribute '{attr}': {e}")
        # If target is callable, call it; otherwise return the resolved object.
        if callable(obj):
            return obj(*args)
        return obj


register_blueprint('python', PythonBlueprint())



def get_global_var_values(visited=None):
    """Return a mapping of variable names to their evaluated values.

    Evaluates variables lazily; `visited` is an optional set used to detect
    circular dependencies during evaluation.
    """
    # Only include already-evaluated variables in the globals mapping.
    # Variables that are defined but not yet evaluated will be evaluated
    # on-demand by `evaluate_variable` during expression evaluation to
    # avoid eager recursive evaluation and unexpected cycles.
    return {k: info['value'] for k, info in VARS.items() if info.get('evaluated')}


def define_variable(name, expr):
    """Define an immutable variable `name` with expression `expr` lazily.

    The expression is stored and evaluated only when the variable is used.
    Redeclaration raises an error (immutable by default).
    """
    if not name.isidentifier():
        raise SyntaxError(f"Invalid variable name '{name}'")
    if name in VARS:
        raise NameError(f"Variable '{name}' is immutable and already defined")
    expr = strip_comments(expr).strip()
    VARS[name] = {'expr': expr, 'evaluated': False, 'value': None}
    return None


def evaluate_variable(name, visited=None):
    """Evaluate variable `name` lazily, detecting cycles via `visited` set."""
    if name not in VARS:
        raise NameError(f"Variable '{name}' is not defined")
    info = VARS[name]
    if info.get('evaluated'):
        return info['value']
    if visited is None:
        visited = set()
    if name in visited:
        raise RecursionError(f"Circular dependency while evaluating variable '{name}'")
    visited.add(name)
    expr = info['expr']
    # Evaluate using globals (which will call evaluate_variable recursively as needed)
    value = evaluate_expression(expr, get_global_var_values(visited), visited)
    info['value'] = value
    info['evaluated'] = True
    visited.remove(name)
    return value


def register_function(name, params, return_expr):
    # Strip inline comments from the function body before storing
    return_expr = strip_comments(return_expr).strip()
    is_block = return_expr.startswith('{') and return_expr.endswith('}')
    body = return_expr[1:-1].strip() if is_block else return_expr.strip()
    FUNCTIONS[name] = {'params': params, 'body': body, 'is_block': is_block}


SAFE_FUNCS = {
    'abs': abs,
    'min': min,
    'max': max,
    'pow': pow,
    'len': len,
}


def _value_to_type(value):
    """Map a runtime Python/Theta value to a Hindley–Milner `Type`.

    This is a heuristic runtime-based typer used for `typeof(expr)` until
    a full AST-based HM inference is implemented.
    """
    # primitives
    if isinstance(value, bool):
        return tt.TypeConst('Bool')
    if isinstance(value, int) and not isinstance(value, bool):
        return tt.TypeConst('Int')
    if isinstance(value, float):
        return tt.TypeConst('Float')
    if isinstance(value, str):
        return tt.TypeConst('String')
    if value is None:
        return tt.TypeConst('None')
    # lists / ThetaArray
    if isinstance(value, (list, tuple, ThetaArray)):
        items = list(value)
        if not items:
            return tt.TypeList(tt.fresh_type_var())
        first_t = _value_to_type(items[0])
        homogeneous = all(repr(_value_to_type(x)) == repr(first_t) for x in items)
        if homogeneous:
            return tt.TypeList(first_t)
        return tt.TypeList(tt.TypeConst('Any'))
    # fallback
    return tt.TypeConst('Any')


def typeof(expr_str: str):
    """Infer a type for `expr_str` by evaluating it and mapping the result.

    This is a pragmatic first step: it executes the expression and returns
    a derived HM `Type`. It does not perform static HM inference yet.
    """
    if not isinstance(expr_str, str):
        # if called with a non-string, treat it as a runtime value
        t = _value_to_type(expr_str)
        return repr(t)
    # try to evaluate the expression first in the current globals
    try:
        val = evaluate_expression(expr_str, get_global_var_values())
    except Exception as e:
        # If evaluation fails due to an undefined name (e.g., user passed
        # a raw string like "Hello" which becomes Hello), fall back to
        # treating the original python string as a runtime value.
        if isinstance(e, NameError):
            val = expr_str
        else:
            raise RuntimeError(f"Cannot evaluate expression for typeof: {e}")
    t = _value_to_type(val)
    return repr(t)


# expose typeof as a safe function callable from Theta expressions
SAFE_FUNCS['typeof'] = typeof

# Casting helpers
def _cast_int(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        # allow underscores and leading +/-, reject non-digits
        try:
            return int(s.replace('_',''))
        except Exception:
            raise ValueError(f"Cannot cast to Int: {x}")
    raise ValueError(f"Cannot cast to Int: {x}")

def _cast_float(x):
    if isinstance(x, (int, float, bool)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s.replace('_',''))
        except Exception:
            raise ValueError(f"Cannot cast to Float: {x}")
    raise ValueError(f"Cannot cast to Float: {x}")

def _cast_string(x):
    # ThetaArray pretty repr
    if isinstance(x, ThetaArray):
        return x.__repr__()
    return str(x)

def _cast_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ('true','1','yes','y','on'):
            return True
        if s in ('false','0','no','n','off'):
            return False
        raise ValueError(f"Cannot cast to Bool: {x}")
    # non-empty lists/arrays are truthy
    if isinstance(x, (list, tuple, ThetaArray)):
        return len(list(x)) > 0
    return bool(x)

# Register casting functions as safe funcs
SAFE_FUNCS['Int'] = _cast_int
SAFE_FUNCS['Float'] = _cast_float
SAFE_FUNCS['String'] = _cast_string
SAFE_FUNCS['Bool'] = _cast_bool


def strip_comments(s: str) -> str:
    """Remove inline comments starting with '#' unless inside quotes.

    Returns the string up to the first unquoted '#'. Trailing whitespace is stripped.
    """
    out = []
    in_sq = False
    in_dq = False
    i = 0
    while i < len(s):
        ch = s[i]
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


def split_top_level_semicolons(s: str):
    """Split string `s` on top-level semicolons (ignore semicolons inside brackets/quotes)."""
    parts = []
    buf = []
    depth = 0
    in_sq = False
    in_dq = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            buf.append(ch)
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
            buf.append(ch)
        elif not in_sq and not in_dq:
            if ch in '([{':
                depth += 1
                buf.append(ch)
            elif ch in ')]}':
                depth -= 1
                buf.append(ch)
            elif ch == ';' and depth == 0:
                part = ''.join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    last = ''.join(buf).strip()
    if last:
        parts.append(last)
    return parts


def transform_semicolons_in_brackets(s: str) -> str:
    """Convert semicolon-separated lists inside square brackets to comma-separated.

    Example: '[1;2;3]' -> '[1,2,3]'. Handles nested brackets and respects quotes.
    """
    out = []
    depth = 0
    in_sq = False
    in_dq = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            out.append(ch)
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
            out.append(ch)
        elif not in_sq and not in_dq:
            if ch == '[':
                depth += 1
                out.append(ch)
            elif ch == ']':
                depth -= 1
                out.append(ch)
            elif ch == ';' and depth > 0:
                out.append(',')
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return ''.join(out)


def transform_matches(s: str) -> str:
    """Transform `A matches P return X else Y` into a call to the match blueprint.

    We keep this transformation simple and quote the pattern and branch expressions
    so the runtime match handler can evaluate them under the appropriate
    bindings.
    """
    out = s
    while True:
        idx_when = -1
        depth = 0
        in_sq = False
        in_dq = False
        i = 0
        while i < len(out):
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

        # find ' return ' and ' else ' after this ' matches '
        j = idx_when + len(' matches ')
        depth = 0
        in_sq = False
        in_dq = False
        idx_return = -1
        idx_else = -1
        while j < len(out):
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
        while k < len(out):
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

        # If there's no explicit 'else' arm, default to `None`.
        if idx_else == -1:
            left = out[:idx_when]
            pattern = out[idx_when + len(' matches '):idx_return]
            ret = out[idx_return + len(' return '):].strip()
            # Emit a clear lint-style warning so the user knows the else-arm
            # was omitted intentionally or accidentally. Show a compact
            # snippet of the expression to help them locate the issue.
            snippet = f"{left.strip()} matches {pattern.strip()} return {ret}"
            print(f"Warning: 'matches' expression missing 'else' — defaulting to None. Expression: {snippet}")
            other = 'None'
        else:
            left = out[:idx_when]
            pattern = out[idx_when + len(' matches '):idx_return]
            ret = out[idx_return + len(' return '):idx_else]
            other = out[idx_else + len(' else '):]

        # Allow a permissive ordering: users may write either
        #   <subject> matches <pattern> return ...
        # or
        #   <pattern> matches <subject> return ...
        # Detect the likely pattern side by parsing both fragments and
        # checking for the presence of `Name` nodes (pattern variables).
        subj_str = left.strip()
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

        # If the left side contains names (pattern vars) and the right side
        # does not, assume the user wrote pattern-first and swap.
        if left_ast is not None and right_ast is not None and _contains_name(left_ast) and not _contains_name(right_ast):
            subj_str, patt_str = patt_str, subj_str

        # Quote the pattern and branch expressions so the runtime handler can
        # parse and evaluate them under bindings.
        new_expr = f"match.matches({subj_str}, {repr(patt_str)}, {repr(ret.strip())}, {repr(other.strip())}, __GLOBALS__)"
        out = new_expr
    return out


def balance_brackets(s: str) -> int:
    """Return total unclosed bracket depth (round + square + curly).

    Positive means there are unclosed openings. Respects single/double quotes.
    """
    depth_round = 0
    depth_sq = 0
    depth_curly = 0
    in_sq = False
    in_dq = False
    i = 0
    while i < len(s):
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


def _eval_ast(node, local_vars, visited=None):
    # Evaluate AST nodes safely.
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, local_vars, visited)
        right = _eval_ast(node.right, local_vars, visited)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        raise ValueError(f"Unsupported binary operator {node.op}")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand, local_vars, visited)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.Not):
            return not operand
        raise ValueError(f"Unsupported unary operator {node.op}")
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for v in node.values:
                if not _eval_ast(v, local_vars, visited):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for v in node.values:
                if _eval_ast(v, local_vars, visited):
                    return True
            return False
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, local_vars, visited)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_ast(comp, local_vars, visited)
            if isinstance(op, ast.Eq) and not (left == right):
                return False
            if isinstance(op, ast.NotEq) and not (left != right):
                return False
            if isinstance(op, ast.Lt) and not (left < right):
                return False
            if isinstance(op, ast.LtE) and not (left <= right):
                return False
            if isinstance(op, ast.Gt) and not (left > right):
                return False
            if isinstance(op, ast.GtE) and not (left >= right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        cond = _eval_ast(node.test, local_vars, visited)
        return _eval_ast(node.body, local_vars, visited) if cond else _eval_ast(node.orelse, local_vars, visited)
    if isinstance(node, ast.Call):
        # Only allow simple name calls or math.<name>
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            # special-case typeof to avoid relying on SAFE_FUNCS being populated
            if fname == 'typeof':
                args = [_eval_ast(arg, local_vars, visited) for arg in node.args]
                # typeof accepts either a string expression or a runtime value
                return typeof(*args)
            # user-defined function
            if fname in FUNCTIONS:
                args = [_eval_ast(arg, local_vars, visited) for arg in node.args]
                return call_user_function(fname, args)
            if fname in SAFE_FUNCS:
                args = [_eval_ast(arg, local_vars, visited) for arg in node.args]
                return SAFE_FUNCS[fname](*args)
            raise NameError(f"Function '{fname}' is not defined or not allowed")
        if isinstance(node.func, ast.Attribute):
            # allow math.sin, math.cos, etc., and blueprint.method()
            val = node.func
            # math.<fn>
            if isinstance(val.value, ast.Name) and val.value.id == 'math':
                func_name = val.attr
                func = getattr(math, func_name, None)
                if func is None:
                    raise NameError(f"math has no function '{func_name}'")
                args = [_eval_ast(arg, local_vars, visited) for arg in node.args]
                return func(*args)
            # blueprint.method(), e.g., io.out(...)
            if isinstance(val.value, ast.Name) and val.value.id in BLUEPRINTS:
                bp = BLUEPRINTS[val.value.id]
                func_name = val.attr
                func = getattr(bp, func_name, None)
                if func is None or not callable(func):
                    raise NameError(f"Blueprint '{val.value.id}' has no callable '{func_name}'")
                args = [_eval_ast(arg, local_vars, visited) for arg in node.args]
                return func(*args)
        raise ValueError("Unsupported function call")
    if isinstance(node, ast.Name):
        # boolean literals (both lowercase Theta style and Python style)
        if node.id in ("true", "True"):
            return True
        if node.id in ("false", "False"):
            return False
        if node.id in local_vars:
            return local_vars[node.id]
        if node.id == 'math':
            return math
        if node.id == 'typeof':
            return typeof
        # If a variable exists in VARS, evaluate it lazily
        if node.id in VARS:
            return evaluate_variable(node.id, visited or set())
        # If a blueprint exists with this name, return the blueprint object
        if node.id in BLUEPRINTS:
            return BLUEPRINTS[node.id]
        raise NameError(f"Name '{node.id}' is not defined")
    if isinstance(node, ast.Attribute):
        # Support math.pi etc.
        value = _eval_ast(node.value, local_vars, visited)
        return getattr(value, node.attr)
    if isinstance(node, ast.Subscript):
        # Support indexing like list[0] and nested subscripting
        value = _eval_ast(node.value, local_vars, visited)
        # In recent Python ASTs, slice may be ast.Constant, ast.Index wrapper, or direct
        idx_node = node.slice
        # ast.Index was removed in newer versions; handle common cases
        if isinstance(idx_node, ast.Constant):
            idx = idx_node.value
        elif isinstance(idx_node, ast.Tuple):
            idx = tuple(_eval_ast(elt, local_vars, visited) for elt in idx_node.elts)
        else:
            try:
                idx = _eval_ast(idx_node, local_vars, visited)
            except Exception:
                # Fallback: if slice has 'value' attribute (older ast.Index wrapper)
                if hasattr(idx_node, 'value'):
                    idx = _eval_ast(idx_node.value, local_vars, visited)
                else:
                    raise ValueError(f"Unsupported subscript index AST node: {type(idx_node).__name__}")
        return value[idx]
    if isinstance(node, ast.Constant):
        return node.value
    # Avoid direct reference to deprecated ast.Num class to prevent
    # DeprecationWarning on newer Python versions. Fall back to checking
    # the node class name for compatibility with older ASTs.
    if node.__class__.__name__ == 'Num':
        return getattr(node, 'n')
    if isinstance(node, ast.List):
        items = [_eval_ast(elt, local_vars, visited) for elt in node.elts]
        return ThetaArray(items)
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(elt, local_vars, visited) for elt in node.elts)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def evaluate_expression(expr, local_vars=None, visited=None):
    """Parse an expression string into AST and evaluate safely.

    `visited` is an optional set used to detect circular variable dependencies
    during lazy variable evaluation.
    """
    if local_vars is None:
        local_vars = {}
    else:
        # copy to avoid mutating caller's mapping
        local_vars = dict(local_vars)
    # Ensure a `__GLOBALS__` mapping is available for runtime transforms
    # (e.g., `matches` is transformed into a call that passes __GLOBALS__).
    # Include BOTH evaluated globals and current local variables (e.g.,
    # function parameters) so branch expressions can access them.
    try:
        global_snapshot = get_global_var_values()
    except Exception:
        global_snapshot = {}
    merged_scope = dict(global_snapshot)
    # merge current locals (excluding any pre-existing __GLOBALS__ to avoid cycles)
    for k, v in list(local_vars.items()):
        if k != '__GLOBALS__':
            merged_scope[k] = v
    local_vars['__GLOBALS__'] = merged_scope
    # Strip inline comments from the expression first
    expr = strip_comments(expr)
    # Transform OCaml-style semicolon-separated arrays into Python lists
    expr = transform_semicolons_in_brackets(expr)
    # Support Theta's 'when' conditional syntax: 'A when B else C' -> 'A if B else C'
    def transform_when_else(s: str) -> str:
        # Transform occurrences of 'A when B else C' into 'A if B else C'.
        # This implementation scans for ' when ' tokens outside of quoted
        # strings and matches them to the nearest corresponding ' else '
        # at the same nesting level (respecting parentheses/brackets/braces).
        out = s
        while True:
            idx_when = -1
            # scan to find a ' when ' not inside quotes
            depth = 0
            in_sq = False
            in_dq = False
            i = 0
            while i < len(out):
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
                    # detect ' when ' at any nesting level (we'll match else at same level)
                    if out.startswith(' when ', i):
                        idx_when = i
                        break
                i += 1
            if idx_when == -1:
                break

            # find matching ' else ' after this ' when ' at the same nesting/quote level
            j = idx_when + len(' when ')
            depth = 0
            in_sq = False
            in_dq = False
            idx_else = -1
            while j < len(out):
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
            # Replace this when/else with Python conditional expression and continue
            new_expr = f"({left.strip()}) if ({mid.strip()}) else ({right.strip()})"
            out = new_expr
        return out

    # Rewrite reserved attribute calls: allow `.<in>(` by mapping to `.__in__(`
    expr = expr.replace('.in(', '.__in__(')
    # transform boolean operators, matches, and when/else constructs before parsing
    def transform_boolean_ops(s: str) -> str:
        # Replace '||' with ' or ' and '&&' with ' and ' outside of quotes.
        out = []
        in_sq = False
        in_dq = False
        i = 0
        while i < len(s):
            ch = s[i]
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
                if ch == '|' and i + 1 < len(s) and s[i+1] == '|':
                    out.append(' or ')
                    i += 2
                    continue
                if ch == '&' and i + 1 < len(s) and s[i+1] == '&':
                    out.append(' and ')
                    i += 2
                    continue
            out.append(ch)
            i += 1
        return ''.join(out)

    def transform_not_operator(s: str) -> str:
        # Replace unary '!' with ' not ' outside of quotes, but keep '!=' intact.
        out = []
        in_sq = False
        in_dq = False
        i = 0
        while i < len(s):
            ch = s[i]
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
            if not in_sq and not in_dq and ch == '!':
                # if next char is '=' then it's '!='; leave as is
                if i + 1 < len(s) and s[i+1] == '=':
                    out.append('!')
                    i += 1
                    continue
                # otherwise treat as unary not
                out.append(' not ')
                i += 1
                continue
            out.append(ch)
            i += 1
        return ''.join(out)
    expr = transform_not_operator(expr)
    expr = transform_boolean_ops(expr)
    expr = transform_matches(expr)
    expr2 = transform_when_else(expr)
    # (debug print removed for normal runs)
    parsed = ast.parse(expr2, mode='eval')
    parsed_body = parsed.body
    double_brackets = False
    # If the original expression used the double-bracket array syntax
    # (an outer list with a single inner list), evaluate the inner list
    # for semantics but remember to display it as double-bracketed.
    if isinstance(parsed_body, ast.List) and len(parsed_body.elts) == 1 and isinstance(parsed_body.elts[0], ast.List):
        parsed_body = parsed_body.elts[0]
        double_brackets = True

    result = _eval_ast(parsed_body, local_vars, visited)
    if double_brackets:
        # Ensure the result is a ThetaArray and mark it for double-bracket display
        if isinstance(result, ThetaArray):
            result._double = True
        elif isinstance(result, list):
            result = ThetaArray(result, double_brackets=True)
    return result


def process_block(body_str, local_vars):
    # Split by semicolons; support assignments and 'return <expr>'
    parts = split_top_level_semicolons(body_str)
    for part in parts:
        if part.startswith('return '):
            expr = part[len('return '):].strip()
            base = get_global_var_values()
            base.update(local_vars)
            # Support guarded return syntax: 'return <expr> when <cond>' (no else)
            # which should behave like:
            #   if <cond>: return <expr>
            # else continue to next statement.
            if ' when ' in expr and ' else ' not in expr:
                # split only on the first top-level ' when '
                # respect brackets/quotes is unnecessary here because
                # split_top_level_semicolons already delivered a single statement
                idx = expr.find(' when ')
                left = expr[:idx].strip()
                cond = expr[idx + len(' when '):].strip()
                cond_val = evaluate_expression(cond, base)
                if cond_val:
                    return evaluate_expression(left, base)
                else:
                    # condition false -> continue to next statement
                    continue
            # Normal return (may include 'when ... else ...' which transform handles)
            return evaluate_expression(expr, base)
        if '=' in part:
            left, right = part.split('=', 1)
            name = left.strip()
            if not name.isidentifier():
                raise SyntaxError(f"Invalid assignment target '{name}'")
            # Merge globals with current locals for evaluation
            base = get_global_var_values()
            base.update(local_vars)
            value = evaluate_expression(right.strip(), base)
            local_vars[name] = value
            continue
        # otherwise evaluate expression and ignore result
        base = get_global_var_values()
        base.update(local_vars)
        evaluate_expression(part, base)
    return None


def parseFunction(function_definition):
    s = function_definition.strip()
    # split name and the rest starting at first '('
    if '(' not in s:
        raise ValueError("Invalid function definition: missing '('.")
    name, rest = s.split('(', 1)
    # find closing ')'
    close_idx = rest.find(')')
    if close_idx == -1:
        raise ValueError("Invalid function definition: missing ')'.")
    params_str = rest[:close_idx].strip()
    after = rest[close_idx+1:].strip()
    if '->' not in after:
        raise ValueError("Invalid function definition: missing '->'.")
    return_expr = after.split('->', 1)[1].strip()
    params = []
    if params_str:
        params = [param.strip() for param in params_str.split(',') if param.strip()]
    return name.strip(), params, return_expr


def handle_line(line, interactive=True):
    """Handle a single input line (REPL or file) and return True if handled.

    Returns True when the line was recognized and processed; False otherwise.
    """
    # line should already be comment-stripped and trimmed by caller
    if line == '':
        return True
    if line.lower() == 'exit':
        print("Exiting Theta. Goodbye!")
        sys.exit(0)

    # Variable definition
    if line.startswith('let '):
        try:
            rest = line[len('let '):].strip()
            if '=' not in rest:
                raise SyntaxError("Invalid let statement; expected 'let name = expr'")
            name, expr = rest.split('=', 1)
            name = name.strip()
            expr = expr.strip()
            define_variable(name, expr)
            if interactive:
                print(f"Defined variable '{name}'")
        except Exception as e:
            report_error(e, context=f"defining variable '{name if 'name' in locals() else ''}'")
        return True

    # Import blueprint or python module
    if line.startswith('import '):
        try:
            name = line[len('import '):].strip()
            if name == 'io':
                if 'io' not in BLUEPRINTS:
                    register_blueprint('io', _io_instance)
                if interactive:
                    print("Imported blueprint 'io'")
            else:
                # If a blueprint with this name is already registered, don't try to
                # import a Python module. Just acknowledge the existing blueprint.
                if name in BLUEPRINTS:
                    if interactive:
                        print(f"Blueprint '{name}' is already available")
                else:
                    # First, try to import from a local Theta file `<name>.th`.
                    ok = _import_blueprint_from_th(name)
                    if ok:
                        if interactive:
                            print(f"Imported Theta blueprint '{name}' from {name}.th")
                    else:
                        # Fallback: import a Python module and register as blueprint
                        mod = importlib.import_module(name)
                        register_blueprint(name, mod)
                        if interactive:
                            print(f"Imported module '{name}' as blueprint")
        except Exception as e:
            report_error(e, context=f"importing '{name}'")
        return True

    # Function definition
    if '->' in line:
        try:
            name, params, return_expr = parseFunction(line)
            register_function(name, params, return_expr)
            if interactive:
                print(f"Defined function '{name}' with parameters {params} returning '{return_expr}'")
        except Exception as e:
            report_error(e, context=f"defining function from: {line}")
        return True

    # Function or blueprint call (or dotted expression)
    if '(' in line and line.endswith(')'):
        func_name = line.split('(', 1)[0].strip()
        args_raw = line.split('(', 1)[1][:-1]
        args = [arg.strip() for arg in args_raw.split(',')] if args_raw.strip() != '' else []
        # Only treat this as a top-level function/blueprint call when the
        # portion before '(' is a valid identifier or dotted identifier like
        # `io.out` or `module.attr`. Otherwise fall through and evaluate the
        # whole line as an expression (covers cases like `... else ()`).
        import re
        if not re.match(r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$', func_name):
            # Not a simple (possibly dotted) identifier: evaluate as expression
            try:
                result = evaluate_expression(line, get_global_var_values())
                if result is not None:
                    print(result)
            except Exception as e:
                report_error(e, context=f"evaluating expression: {line}")
            return True

        try:
            if '.' in func_name:
                # dotted call -> evaluate as expression (handles blueprints)
                result = evaluate_expression(line, get_global_var_values())
                if result is not None:
                    print(result)
            else:
                result = call_function(func_name, args)
                if result is not None:
                    print(result)
        except Exception as e:
            report_error(e, context=f"calling function '{func_name}'")
        return True

    # Try to evaluate the line as an expression (includes variable names)
    try:
        result = evaluate_expression(line, get_global_var_values())
        print(result)
        return True
    except Exception as e:
        if isinstance(e, NameError):
            return False
        else:
            report_error(e, context=f"evaluating expression: {line}")
            return True


def run_file(path):
    """Run a Theta script file (.th)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        report_error(e, context=f"reading file '{path}'")
        return

    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        line = strip_comments(raw).strip()
        if line == '' or line.startswith('#'):
            continue
        # Show processed line (debug) to locate parse issues (printed only when DEBUG=True)
        log(f"[run_file] processing: {line}")
        # Support multi-line bracketed expressions and block bodies
        # If the line contains any opening paren/brace/bracket whose matching
        # closer isn't present yet, keep reading lines until the entire
        # expression's brackets are balanced. This respects single- and
        # double-quotes to avoid counting brackets inside strings.
        def _balance_all(s: str) -> int:
            depth_round = 0
            depth_sq = 0
            depth_curly = 0
            in_sq = False
            in_dq = False
            i2 = 0
            while i2 < len(s):
                ch2 = s[i2]
                if ch2 == "'" and not in_dq:
                    in_sq = not in_sq
                elif ch2 == '"' and not in_sq:
                    in_dq = not in_dq
                elif not in_sq and not in_dq:
                    if ch2 == '(':
                        depth_round += 1
                    elif ch2 == ')':
                        depth_round -= 1
                    elif ch2 == '[':
                        depth_sq += 1
                    elif ch2 == ']':
                        depth_sq -= 1
                    elif ch2 == '{':
                        depth_curly += 1
                    elif ch2 == '}':
                        depth_curly -= 1
                i2 += 1
            return depth_round + depth_sq + depth_curly

        # If any bracket depth is positive (unclosed openings), keep reading
        if _balance_all(line) > 0:
            parts = [line]
            while i < len(lines):
                more_raw = lines[i]
                i += 1
                more = strip_comments(more_raw).rstrip()
                parts.append(more)
                joined = '\n'.join(parts)
                if _balance_all(joined) <= 0:
                    break
            # update `line` after we've collected the full balanced chunk
            line = ' '.join(p.strip() for p in parts).strip()
        # Handle blueprint block specially so we can read until ']'
        if line.startswith('blueprint '):
            rest = line[len('blueprint '):].strip()
            if '[' in rest:
                parts = rest.split('[', 1)
                bp_name = parts[0].strip()
                after = parts[1]
                body_lines = []
                if ']' in after:
                    inner = after.split(']', 1)[0]
                    body_lines = inner.split('\n')
                else:
                    if after.strip():
                        body_lines.append(after)
                    # consume lines until ']' found
                    while i < len(lines):
                        more = lines[i]
                        i += 1
                        if ']' in more:
                            body_lines.append(more.split(']', 1)[0])
                            break
                        body_lines.append(more)

                # parse defs in body_lines
                defs = []
                for raw2 in body_lines:
                    line2 = strip_comments(raw2).strip()
                    if not line2:
                        continue
                    if line2.startswith('def '):
                        fn_def = line2[len('def '):].strip()
                        fname, fparams, fexpr = parseFunction(fn_def)
                        defs.append((fname, fparams, fexpr))
                bp_obj = create_blueprint_from_defs(bp_name, defs)
                register_blueprint(bp_name, bp_obj)
                continue
            else:
                print("Invalid blueprint definition in file; missing '['")
                continue

        # Support multi-line function bodies introduced with a trailing '->'
        # Example:
        #   foo(x) ->
        #       x + 1
        #   bar(y) ->
        #       y * 2 when y > 1 else y
        # We collect subsequent lines until we reach a blank line or a new
        # top-level statement (let/import/blueprint/another function def).
        if '->' in line:
            lhs, rhs = line.split('->', 1)
            if rhs.strip() == '':
                body_lines = []
                # consume following lines for body
                start_i = i
                while i < len(lines):
                    nxt_raw = lines[i]
                    nxt = strip_comments(nxt_raw).strip()
                    # stop on empty line or a new top-level construct
                    if nxt == '' or nxt.startswith('let ') or nxt.startswith('import ') or nxt.startswith('blueprint ') or '->' in nxt:
                        break
                    body_lines.append(nxt)
                    i += 1
                # join body lines with spaces to allow expressions like
                # '... matches ... return ... else ...' split across lines
                body_expr = ' '.join(body_lines).strip()
                try:
                    name, params, _ = parseFunction(line)
                    # register the function using the collected expression
                    register_function(name, params, body_expr)
                    continue
                except Exception as e:
                    report_error(e, context=f"defining function from: {line}")
                    # fall through to default handling

        # Non-blueprint lines: delegate to handle_line (non-interactive)
        try:
            handled = handle_line(line, interactive=False)
            if not handled:
                print(f"Unrecognized input in script: {line}")
        except Exception as e:
            # include file and line number context
            lineno = max(1, i)
            report_error(e, context=f"{path}:{lineno}")



def call_user_function(name, evaluated_args):
    info = FUNCTIONS.get(name)
    if info is None:
        raise NameError(f"Function '{name}' is not defined.")
    params = info['params']
    if len(evaluated_args) != len(params):
        raise TypeError(f"{name}() takes {len(params)} positional argument(s) but {len(evaluated_args)} were given")
    # Local variables for this function: parameters overlaid on globals
    local_vars = get_global_var_values()
    local_vars.update({p: v for p, v in zip(params, evaluated_args)})
    # allow recursion by having FUNCTIONS available at module level
    if info['is_block']:
        # pass only the parameter locals into block execution
        param_locals = {p: local_vars[p] for p in params}
        try:
            result = process_block(info['body'], param_locals)
            return result
        except Exception as e:
            report_error(e, context=f"in function '{name}'")
            raise
    else:
        try:
            return evaluate_expression(info['body'], local_vars)
        except Exception as e:
            report_error(e, context=f"in function '{name}'")
            raise


def call_function(name, arg_strs):
    # Evaluate arg expressions first, then call the user function
    base = get_global_var_values()
    evaluated_args = [evaluate_expression(a, base) for a in arg_strs]
    # Allow calling safe builtin functions (e.g., typeof) as top-level names
    if name in SAFE_FUNCS:
        return SAFE_FUNCS[name](*evaluated_args)
    return call_user_function(name, evaluated_args)


def main():
    while True:
        try:
            line = input("theta> ")
        except EOFError:
            print()
            break
        if line is None:
            break
        # Remove inline comments before further processing
        line = strip_comments(line).strip()
        if line == '':
            continue
        if line.lower() == 'exit':
            print("Exiting Theta. Goodbye!")
            break
        # REPL: type query
        if line.startswith(':type ') or line.startswith(':t '):
            try:
                expr = line.split(None, 1)[1]
                t = typeof(expr)
                print(t)
            except Exception as e:
                report_error(e, context='typeof')
            continue

        # Function definition
        if line.startswith('let '):
            # variable definition: let name = expr
            try:
                rest = line[len('let '):].strip()
                if '=' not in rest:
                    raise SyntaxError("Invalid let statement; expected 'let name = expr'")
                name, expr = rest.split('=', 1)
                name = name.strip()
                expr = expr.strip()
                val = define_variable(name, expr)
            except Exception as e:
                print(f"Error defining variable: {e}")
            continue

        # Import blueprint or python module
        if line.startswith('import '):
            try:
                name = line[len('import '):].strip()
                # support importing built-in blueprint names like 'io'
                if name == 'io':
                    # register our default io instance (preserve __in__ mapping)
                    if 'io' not in BLUEPRINTS:
                        register_blueprint('io', _io_instance)
                    print("Imported blueprint 'io'")
                else:
                    # If the blueprint name already exists, acknowledge it instead
                    # of attempting to import a Python module (avoids ModuleNotFoundError
                    # when using built-in blueprints like 'tm').
                    if name in BLUEPRINTS:
                        print(f"Blueprint '{name}' is already available")
                    else:
                        # Try `.th` file first, then Python module.
                        ok = _import_blueprint_from_th(name)
                        if ok:
                            print(f"Imported Theta blueprint '{name}' from {name}.th")
                        else:
                            # try to import a python module and register it as a blueprint
                            try:
                                mod = importlib.import_module(name)
                                register_blueprint(name, mod)
                                print(f"Imported module '{name}' as blueprint")
                            except Exception as e:
                                report_error(e, context=f"importing '{name}'")
            except Exception as e:
                report_error(e, context=f"importing '{name}'")
            continue

        # Blueprint definition: multi-line block starting with 'blueprint name [' and ending with ']'
        if line.startswith('blueprint '):
            try:
                rest = line[len('blueprint '):].strip()
                # Expect: name [  (maybe same-line body or multi-line)
                if '[' in rest:
                    parts = rest.split('[', 1)
                    bp_name = parts[0].strip()
                    after = parts[1]
                    body_lines = []
                    if ']' in after:
                        inner = after.split(']', 1)[0]
                        body_lines = inner.split('\n')
                    else:
                        # read following lines until a line containing ']'
                        if after.strip():
                            body_lines.append(after)
                        while True:
                            more = input('... ')  # continuation prompt
                            if ']' in more:
                                body_lines.append(more.split(']', 1)[0])
                                break
                            body_lines.append(more)
                else:
                    raise SyntaxError("Invalid blueprint definition; missing '['")

                # Parse body_lines to find `def` function definitions
                defs = []
                for raw in body_lines:
                    line2 = strip_comments(raw).strip()
                    if not line2:
                        continue
                    if line2.startswith('def '):
                        # use parseFunction to parse 'def name(params) -> expr'
                        # remove leading 'def '
                        fn_def = line2[len('def '):].strip()
                        fname, fparams, fexpr = parseFunction(fn_def)
                        defs.append((fname, fparams, fexpr))
                    else:
                        # ignore other lines inside blueprint for now
                        continue

                bp_obj = create_blueprint_from_defs(bp_name, defs)
                register_blueprint(bp_name, bp_obj)
                print(f"Defined blueprint '{bp_name}' with methods {[d[0] for d in defs]}")
            except Exception as e:
                print(f"Error defining blueprint: {e}")
            continue

        if '->' in line:
            # Support interactive multi-line function block bodies in the REPL.
            try:
                # If the right-hand side is a block that is not closed, read continuation lines
                parts = line.split('->', 1)
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                if rhs.startswith('{') and balance_brackets(rhs) > 0:
                    body_lines = [rhs]
                    while True:
                        more = input('... ')
                        body_lines.append(more)
                        joined = '\n'.join(body_lines)
                        if balance_brackets(joined) <= 0:
                            rhs = joined
                            break
                    # reconstruct a single-line function definition for parsing
                    line_full = f"{lhs} -> {rhs}"
                else:
                    line_full = line

                name, params, return_expr = parseFunction(line_full)
                register_function(name, params, return_expr)
            except Exception as e:
                report_error(e, context=f"defining function from: {line}")
            continue

        # Function call
        if '(' in line and line.endswith(')'):
            func_name = line.split('(', 1)[0].strip()
            args_raw = line.split('(', 1)[1][:-1]
            args = [arg.strip() for arg in args_raw.split(',')] if args_raw.strip() != '' else []
            try:
                        # If this is an attribute call (blueprint.method), evaluate as expression
                        import re
                        if not re.match(r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$', func_name):
                            # not a simple identifier/dotted name -> evaluate as expression
                            result = evaluate_expression(line, get_global_var_values())
                            if result is not None:
                                print(result)
                        elif '.' in func_name:
                            result = evaluate_expression(line, get_global_var_values())
                            if result is not None:
                                print(result)
                        else:
                            result = call_function(func_name, args)
                            if result is not None:
                                print(result)
            except Exception as e:
                print(f"Error calling function: {e}")
            continue

        # Try to evaluate the line as an expression (includes variable names)
        try:
            result = evaluate_expression(line, get_global_var_values())
            print(result)
            continue
        except Exception as e:
            # If it's a simple NameError or similar, fall through to unrecognized message;
            # otherwise show evaluation errors (e.g., circular deps)
            if isinstance(e, NameError):
                pass
            else:
                print(f"Error evaluating expression: {e}")
                continue

        print("Unrecognized input. Please define a function, variable, or call a function.")


if __name__ == '__main__':
    # Command-line flags: support `--debug` or `--verbose` to enable debug logging.
    args = sys.argv[1:]
    if '--debug' in args or '--verbose' in args:
        # enable debug logging and remove the flags from args list
        DEBUG = True
        args = [a for a in args if a not in ('--debug', '--verbose')]

    # If filenames provided, run them (support .th scripts), otherwise start REPL.
    if len(args) > 0:
        for arg in args:
            if arg.endswith('.th'):
                run_file(arg)
            else:
                print(f"Skipping unsupported file type: {arg}")
    else:
        main()
        
