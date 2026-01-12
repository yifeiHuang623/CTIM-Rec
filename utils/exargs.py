import os
import re
import yaml
import json
import ast
import operator as op
from collections import defaultdict
from typing import Any
from typing_extensions import Self

SAFE_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Mod: op.mod, ast.Pow: op.pow,
    ast.BitXor: op.xor, ast.And: lambda a, b: a and b, ast.Or: lambda a, b: a or b,
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge,
    ast.USub: op.neg, ast.Not: op.not_,
}
SAFE_FUNCTIONS = {
    'min': min, 'max': max, 'abs': abs, 'int': int, 'float': float, 'bool': bool
}

def _preprocess_expr(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"&&", " and ", expr)
    expr = re.sub(r"\|\|", " or ", expr)
    return expr


def _build_attr_path(node: ast.Attribute) -> str:
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        raise ValueError("Unsupported attribute chain")
    return ".".join(reversed(parts))

def _as_number(x):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return x
    raise TypeError(f"Non-numeric operand: {x!r}")

def _binop_numeric(opnode, a, b):
    a, b = _as_number(a), _as_number(b)
    float_infects = isinstance(a, float) or isinstance(b, float)

    if isinstance(opnode, ast.Add):
        return a + b  
    if isinstance(opnode, ast.Sub):
        return a - b
    if isinstance(opnode, ast.Mult):
        return a * b

    if isinstance(opnode, ast.Div):
        if float_infects:
            return a / b
        if b == 0:
            raise ZeroDivisionError("division by zero")
        q, r = divmod(a, b)
        return q if r == 0 else a / b

    if isinstance(opnode, ast.Mod):
        return a % b  

    if isinstance(opnode, ast.Pow):
        if float_infects:
            return a ** b
        if b >= 0:
            return int(a ** b)
        else:
            return float(a ** b)

    if isinstance(opnode, ast.BitXor):
        return int(a) ^ int(b)

    raise ValueError(f"Unsupported numeric operator: {opnode!r}")


def _eval_expr(expr: str, local_vars: dict):
    expr = _preprocess_expr(expr)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in local_vars:
                return local_vars[node.id]
            if node.id in os.environ:
                return os.environ[node.id]
            raise KeyError(node.id)
        elif isinstance(node, ast.Attribute):
            dotted = _build_attr_path(node)
            if dotted in local_vars:
                return local_vars[dotted]
            base_val = _eval(node.value)
            if isinstance(base_val, dict):
                if node.attr in base_val:
                    return base_val[node.attr]
            raise KeyError(dotted)
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            return _binop_numeric(node.op, left, right)

        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                v = _as_number(_eval(node.operand))
                return -v
            return SAFE_OPERATORS[type(node.op)](_eval(node.operand))
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(_eval(v) for v in node.values)
            elif isinstance(node.op, ast.Or):
                return any(_eval(v) for v in node.values)
        elif isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op_, right_ in zip(node.ops, node.comparators):
                if not SAFE_OPERATORS[type(op_)](left, _eval(right_)):
                    return False
                left = _eval(right_)
            return True
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name not in SAFE_FUNCTIONS:
                raise ValueError(f"Function '{func_name}' not allowed in expressions")
            func = SAFE_FUNCTIONS[func_name]
            args = [_eval(arg) for arg in node.args]
            return func(*args)
        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expr, mode='eval')
        return _eval(tree)
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {e}")


class ConfigResolver:
    VAR_PATTERN = re.compile(r"\$\{(?!\{)([^}]+)\}")  
    EXPR_PATTERN = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.raw_config = self._load_config()
        self.flat_config = self._flatten(self.raw_config)
        self.dependencies = self._extract_dependencies()
        self.resolved = None

    def parse(self) -> dict[str, Any]:
        order = self._topo_sort_with_cycle_check()
        # include all flat keys
        for k in self.flat_config:
            if k not in order:
                order.append(k)

        resolved = {}
        for key in order:
            if key not in self.flat_config:
                continue
            resolved[key] = self._resolve_value_recursively(self.flat_config[key], resolved)
        self.resolved = resolved
        return self._unflatten(resolved)

    def add_variable(self, key: str, value: Any) -> Self:
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        self.flat_config[key] = value
        self.dependencies = self._extract_dependencies()
        return self.parse()

    def _resolve_value_recursively(self, value, resolved):
        if isinstance(value, list):
            return [self._resolve_value_recursively(v, resolved) for v in value]
        if isinstance(value, dict):
            return {k: self._resolve_value_recursively(v, resolved) for k, v in value.items()}

        def substitute_vars(val):
            if not isinstance(val, str):
                return val

            m = self.VAR_PATTERN.fullmatch(val)
            if m:
                var = m.group(1)
                if var in resolved:
                    return resolved[var]
                elif var in self.flat_config:
                    v = self._resolve_value_recursively(self.flat_config[var], resolved)
                    resolved[var] = v
                    return v
                elif var in os.environ:
                    return os.environ[var]
                else:
                    raise ValueError(f"Unresolved variable: {var}")

            prev = None
            while isinstance(val, str) and self.VAR_PATTERN.search(val):
                if val == prev:
                    raise ValueError(f"Unresolved variables remain in value: {val}")
                prev = val

                def _repl(m):
                    var = m.group(1)
                    if var in resolved:
                        v = resolved[var]
                    elif var in self.flat_config:
                        v = self._resolve_value_recursively(self.flat_config[var], resolved)
                        resolved[var] = v
                    elif var in os.environ:
                        v = os.environ[var]
                    else:
                        raise ValueError(f"Unresolved variable: {var}")
                    return str(v) 

                val = self.VAR_PATTERN.sub(_repl, val)
            return val

        value = substitute_vars(value)

        if isinstance(value, str) and self.EXPR_PATTERN.search(value):
            m_full = self.EXPR_PATTERN.fullmatch(value)
            if m_full:
                return _eval_expr(m_full.group(1), resolved)
            value = self.EXPR_PATTERN.sub(lambda m: str(_eval_expr(m.group(1), resolved)), value)

        return value

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith(('.yaml', '.yml')):
                return yaml.safe_load(f)
            elif self.config_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError("Only .yaml, .yml, or .json files are supported.")

    def _flatten(self, d, parent_key='', sep='.'):  # dict → flat
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten(v, new_key, sep))
            else:
                items[new_key] = v
        return items

    def _unflatten(self, d, sep='.'):  # flat → nested
        result = {}
        for k, v in d.items():
            keys = k.split(sep)
            tgt = result
            for sub in keys[:-1]:
                tgt = tgt.setdefault(sub, {})
            tgt[keys[-1]] = v
        return result

    def _extract_dependencies(self):
        deps = defaultdict(set)
        dotted_name_re = r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"

        for key, val in self.flat_config.items():
            if isinstance(val, str):
                deps[key].update(self.VAR_PATTERN.findall(val))
                for expr in self.EXPR_PATTERN.findall(val):
                    deps[key].update(re.findall(dotted_name_re, expr))

        all_nodes = set(deps.keys()).union(*deps.values()) if deps else set()
        for n in all_nodes:
            deps.setdefault(n, set())
        return deps

    def _topo_sort_with_cycle_check(self):
        visited: dict[str, int] = {}
        order: list[str] = []
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]):
            if visited.get(node) == 1:
                cycles.append(path + [node])
                return
            if visited.get(node) == 2:
                return
            visited[node] = 1 
            for nxt in self.dependencies.get(node, []):
                dfs(nxt, path + [node])
            visited[node] = 2 
            order.append(node)

        for k in sorted(self.dependencies):
            if visited.get(k, 0) == 0:
                dfs(k, [])

        if cycles:
            readable = [" -> ".join(c) for c in cycles]
            raise ValueError("Cycle(s) detected in variable references: " + str(readable))

        return order 

def format_nested(obj, indent: int = 0, indent_step: int = 2) -> str:
    lines: list[str] = []

    def _fmt(value, level: int):
        pad = " " * (level * indent_step)
        
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    _fmt(v, level + 1)
                else:
                    lines.append(f"{pad}{k}: {v}")
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}-")
                    _fmt(item, level + 1)
                else:
                    lines.append(f"{pad}- {item}")

        else:
            lines.append(f"{pad}{value}")

    _fmt(obj, indent // indent_step)
    return "\n".join(lines)
