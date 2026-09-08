"""Static check: every name used in ibkr_engine.py (and credit_gate.py) is defined, imported, a builtin,
or one of the ib_insync star-import names. pyflakes is not installed on this machine; this is the
equivalent undefined-name pass, restricted to module-level names (attribute chains are not resolved).
Run: python3 tests/lint_engine_names.py"""
import ast
import builtins
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IB_NAMES = {"IB", "MarketOrder", "Position", "Stock", "util", "Contract", "Trade", "LimitOrder", "Order"}


def defined_names(tree):
    names = set(dir(builtins)) | IB_NAMES | {"__file__", "__name__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if not isinstance(node, ast.ClassDef):
                for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                    names.add(arg.arg)
                if node.args.vararg:
                    names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    names.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    return names


def check(path):
    src = open(os.path.join(ROOT, path)).read()
    tree = ast.parse(src)
    names = defined_names(tree)
    bad = sorted({(n.id, n.lineno) for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in names})
    return bad


fails = 0
for f in ("ibkr_engine.py", "credit_gate.py", "live_config.py", "strategies/multi_strategy_engine.py", "signal_builder.py"):
    bad = check(f)
    print(f"  [{'PASS' if not bad else 'FAIL'}] {f}: {len(bad)} undefined names" + (f" -> {bad[:12]}" if bad else ""))
    fails += bool(bad)
sys.exit(1 if fails else 0)
