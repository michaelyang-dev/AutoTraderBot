"""Live fundamentals use each ticker's WHOLE latest quarter (backtest parity), never a per-column mix.

WHY THIS TEST EXISTS (2026-09-30)
  signal_builder took `fund.sort_values(rdq).groupby("tic").last()`. GroupBy.last() returns the last
  NON-NULL value per column, so a blank field in the newest quarter was silently filled from an older
  quarter (dlcq for 208 of 1,496 pool names; seqq for VSXY, MDT; cogsq for CPB). The backtest keeps one
  record per quarter (NaN stays NaN; missing debt counts as 0).
Run: python3 tests/test_fund_rowwise.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _m in ("ib_insync",):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import signal_builder as SB  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


tmp = Path(tempfile.mkdtemp(prefix="fund_rowwise_"))
(tmp / "wrds").mkdir()
rows = [
    # tic, datadate, rdq, niq, seqq, saleq, cogsq, dlttq, dlcq
    ("AAA", "2026-03-31", "2026-04-30", 10.0, 100.0, 50.0, 20.0, 30.0, 40.0),
    ("AAA", "2026-06-30", "2026-07-30", 12.0, 120.0, 60.0, 24.0, 36.0, np.nan),     # newest: dlcq blank
    ("BBB", "2026-03-31", "2026-04-28", 5.0, 80.0, 40.0, 10.0, 8.0, 2.0),
    ("BBB", "2026-06-30", "2026-07-28", 6.0, np.nan, 44.0, np.nan, 9.0, 3.0),        # newest: seqq & cogsq blank
]
cols = ["tic", "datadate", "rdq", "niq", "seqq", "saleq", "cogsq", "dlttq", "dlcq"]
pd.DataFrame(rows, columns=cols).to_parquet(tmp / "wrds" / "compustat_fundamentals_quarterly.parquet")
feats = pd.DataFrame({"symbol": ["AAA", "BBB"], "date": pd.Timestamp("2026-09-29")})
for c in ("roe", "gross_margin", "debt_to_equity"):
    feats[c] = np.nan
out = SB._fill_fundamentals(feats.copy(), tmp).set_index("symbol")

check("AAA ROE from the newest quarter (12*4/120)", abs(out.loc["AAA", "roe"] - 0.4) < 1e-12, out.loc["AAA", "roe"])
check("AAA D/E: blank newest dlcq counts as 0 (36/120), NOT last quarter's 40 ((36+40)/120)",
      abs(out.loc["AAA", "debt_to_equity"] - 0.3) < 1e-12, out.loc["AAA", "debt_to_equity"])
check("AAA gross margin from the newest quarter ((60-24)/60)", abs(out.loc["AAA", "gross_margin"] - 0.6) < 1e-12)
check("BBB ROE is NaN when newest seqq is blank (not 6*4/80 from the old equity)", pd.isna(out.loc["BBB", "roe"]), out.loc["BBB", "roe"])
check("BBB gross margin is NaN when newest cogsq is blank (not mixed with the old cogs)", pd.isna(out.loc["BBB", "gross_margin"]), out.loc["BBB", "gross_margin"])
check("BBB D/E is NaN when newest seqq is blank", pd.isna(out.loc["BBB", "debt_to_equity"]), out.loc["BBB", "debt_to_equity"])

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all row-wise fundamentals tests passed"))
sys.exit(1 if FAILS else 0)
