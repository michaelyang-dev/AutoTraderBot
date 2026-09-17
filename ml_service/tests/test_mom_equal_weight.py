"""EXP-059 equal-weight momentum flag: strategy1's weighting helper. Run: python3 tests/test_mom_equal_weight.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("MOM_EQUAL_WEIGHT", None); os.environ.pop("PROD_BULL_WEIGHTS", None)
from strategies import multi_strategy_engine as M  # noqa: E402
fails = 0
def check(name, cond, detail=""):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond: fails += 1
comp = {"A": 5.0, "B": 3.0, "C": 1.0, "D": 0.5, "E": 0.5}; syms = ["A", "B", "C", "D", "E"]
w = M._weight_picks(syms, comp)
check("default is score-weighted (cap 2/N before renormalisation; A raw 0.5 -> 0.4 -> 0.444), sums to 1", abs(sum(w.values()) - 1) < 1e-9 and w["A"] == max(w.values()) and abs(w["A"] - 0.4 / 0.9) < 1e-9 and w["A"] > w["C"], str(w))
check("default flag is OFF (deployed behaviour unchanged)", M.MOM_EQUAL_WEIGHT is False)
we = M._weight_picks(syms, comp, equal=True)
check("equal=True -> 1/N each", all(abs(v - 0.2) < 1e-12 for v in we.values()) and set(we) == set(syms))
check("empty picks -> {}", M._weight_picks([], comp) == {} and M._weight_picks([], comp, equal=True) == {})
check("PROD_WEIGHTS_BULL unchanged without the env override", M.PROD_WEIGHTS_BULL["s1_momentum"] == 0.70 and M.PROD_WEIGHTS_BULL["s7_value"] == 0.21 and M.PROD_WEIGHTS_BULL["s5_lowvol"] == 0.09)
# env override path (fresh import in a subprocess so module globals are re-evaluated)
import subprocess, json
code = "import os,sys; sys.path.insert(0, %r); from strategies import multi_strategy_engine as M; print(json.dumps([M.PROD_WEIGHTS_BULL, M.MOM_EQUAL_WEIGHT]))" % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = subprocess.run([sys.executable, "-c", "import json;" + code], env={**os.environ, "PROD_BULL_WEIGHTS": "0.80,0.15,0.05", "MOM_EQUAL_WEIGHT": "1"}, capture_output=True, text=True)
try:
    pw, mq = json.loads(out.stdout.strip().splitlines()[-1])
    check("env override -> 80/15/5 and equal-weight ON", abs(pw["s1_momentum"] - 0.80) < 1e-9 and abs(pw["s7_value"] - 0.15) < 1e-9 and abs(pw["s5_lowvol"] - 0.05) < 1e-9 and mq is True, str(pw))
except Exception as e:
    check("env override subprocess", False, out.stderr[-300:])
out = subprocess.run([sys.executable, "-c", "import json;" + code], env={**os.environ, "PROD_BULL_WEIGHTS": "0.80,0.15,0.10"}, capture_output=True, text=True)
check("override that does not sum to 1 is rejected", out.returncode != 0 and "must sum to 1" in out.stderr)
print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAIL'}"); sys.exit(1 if fails else 0)
