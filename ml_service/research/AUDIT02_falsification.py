"""AUDIT 02 — falsification battery on the LIVE-MIRROR harness.

Mandatory gate tests from the research contract:
  1. SHIFT   — delay EVERY signal by one full bar (features, closes-for-signals, UMD,
               breadth) while execution/valuation stay at the true date-t close.
               A real edge degrades gracefully; a look-ahead bug dies or flips.
  2. SHUFFLE — replace every sleeve's picks with a RANDOM draw from the same eligible
               membership, same count. Long-only equity cannot collapse to zero (you
               still hold beta), so the null is: random ~ market-at-same-leverage, and
               the real book must beat it by a wide margin.
  3. COST    — 1x / 2x / 3x / 5x the modelled cost. Where does the edge die?
  4. OUTLIER — drop the best 5 days of the equity curve. Does it still work?

Run: python3 research/AUDIT02_falsification.py [8yr|26yr]
"""
import os, sys, time, copy
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "research"))
import numpy as np, pandas as pd  # noqa: E402
import livemirror_backtest as LM  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402

B = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
     "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
     "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
     "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
     "financing_rate": 0.063}

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH, STARTS = (("data/wrds/complete_sp1500_universe.pkl",
                 ["2018-01-03", "2018-07-03", "2019-01-03", "2019-07-03"])
                if HZ == "8yr" else
                ("data/wrds/sp1500_universe_2000.pkl",
                 ["2001-01-03", "2001-07-03", "2002-01-03", "2002-07-03"]))


def stat(v):
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    return dict(cagr=(v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
                sharpe=dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0,
                dd=((v - v.cummax()) / v.cummax()).min())


def show(tag, rows):
    c = np.mean([r["cagr"] for r in rows]); s = np.mean([r["sharpe"] for r in rows])
    d = np.mean([r["dd"] for r in rows])
    print(f"  {tag:<34}{c:>+9.2%}{s:>8.2f}{d:>9.1%}", flush=True)
    return c, s, d


bt = LiveMirrorBacktester(universe_path=PATH)
print(f"\n{'='*70}\nAUDIT02 — {HZ}  ({len(STARTS)} starts)\n{'='*70}", flush=True)
print(f"  {'arm':<34}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}", flush=True)

# ---------------- baseline ----------------
base_curves = []
for st in STARTS:
    m = bt.run(st, "2025-12-31", dict(B))
    base_curves.append(m["daily_values"])
base = [stat(v) for v in base_curves]
bc, bs, bd = show("BASELINE (deployed)", base)

# ---------------- 3. cost sensitivity ----------------
for mult in (2, 3, 5):
    rows = [stat(bt.run(st, "2025-12-31", {**B, "cost_mult": mult})["daily_values"])
            for st in STARTS]
    c, s, d = show(f"cost x{mult}", rows)
    print(f"      -> dCAGR {c-bc:+.2%}  dSharpe {s-bs:+.3f}", flush=True)

# ---------------- 4. outlier check ----------------
print("\n  --- outlier check: drop best-N days from the equity curve ---", flush=True)
for n_drop in (5, 10, 20):
    out = []
    for v in base_curves:
        dr = v.pct_change().dropna()
        keep = dr.drop(dr.nlargest(n_drop).index)
        yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
        out.append(dict(cagr=(1 + keep).prod() ** (1 / yrs) - 1,
                        sharpe=keep.mean() / keep.std() * np.sqrt(252), dd=np.nan))
    c = np.mean([r["cagr"] for r in out]); s = np.mean([r["sharpe"] for r in out])
    print(f"  {'drop best %d days'%n_drop:<34}{c:>+9.2%}{s:>8.2f}", flush=True)

# ---------------- 1. SHIFT TEST ----------------
print("\n  --- SHIFT: every signal lagged one full bar, fills still at date-t close ---",
      flush=True)
_orig_feat = bt.uni._feat_by_date
_orig_fbd = bt.features_by_date
_orig_close = bt.uni._close
_orig_umd = bt.umd_20d

dts = sorted(_orig_feat.keys())
lag_feat = {dts[i]: _orig_feat[dts[i - 1]] for i in range(1, len(dts))}
lag_fbd = {dts[i]: _orig_fbd[dts[i - 1]] for i in range(1, len(dts)) if dts[i - 1] in _orig_fbd}
lag_close = {s: ser.shift(1) for s, ser in _orig_close.items()}
bt.uni._feat_by_date = lag_feat
bt.features_by_date = lag_fbd
bt.uni._close = lag_close
bt.umd_20d = _orig_umd.shift(1)
try:
    rows = [stat(bt.run(st, "2025-12-31", dict(B))["daily_values"]) for st in STARTS]
    c, s, d = show("SHIFT +1 bar", rows)
    print(f"      -> dCAGR {c-bc:+.2%}  dSharpe {s-bs:+.3f}   "
          f"({'GRACEFUL - ok' if s > bs - 0.15 else 'COLLAPSE - INVESTIGATE'})", flush=True)
finally:
    bt.uni._feat_by_date = _orig_feat; bt.features_by_date = _orig_fbd
    bt.uni._close = _orig_close; bt.umd_20d = _orig_umd

# ---------------- 2. SHUFFLE TEST ----------------
print("\n  --- SHUFFLE: sleeves return RANDOM members (same count, equal weight) ---",
      flush=True)
_s1, _s3, _s5, _sv = (LM.strategy1_momentum_reversal, LM.strategy3_sector_rotation,
                      LM.strategy5_lowvol_quality, LM.strategy_value)


def _rand_from(uni, date, k, seed):
    mem = sorted(uni.get_sp500(date))
    if len(mem) < k:
        return {}
    rs = np.random.RandomState((hash(str(date)) ^ seed) & 0x7FFFFFFF)
    pick = rs.choice(len(mem), size=k, replace=False)
    return {mem[i]: 1.0 / k for i in pick}


rows_all = []
for rep in range(3):
    LM.strategy1_momentum_reversal = (
        lambda date, uni, di, top_n=8, rebal_days=10, _r=rep, **kw:
        None if di % rebal_days else _rand_from(uni, date, top_n, 1000 + _r))
    LM.strategy_value = (lambda uni, date, members, top_n=10, _r=rep:
                         _rand_from(uni, date, top_n, 2000 + _r))
    LM.strategy5_lowvol_quality = (lambda date, uni, di, _r=rep, **kw:
                                   _rand_from(uni, date, 10, 3000 + _r))
    LM.strategy3_sector_rotation = lambda date, uni, di, **kw: {}
    try:
        rows_all += [stat(bt.run(st, "2025-12-31", dict(B))["daily_values"]) for st in STARTS]
    finally:
        (LM.strategy1_momentum_reversal, LM.strategy3_sector_rotation,
         LM.strategy5_lowvol_quality, LM.strategy_value) = _s1, _s3, _s5, _sv
c, s, d = show("SHUFFLE (random picks, 3 reps)", rows_all)
print(f"      -> dCAGR {c-bc:+.2%}  dSharpe {s-bs:+.3f}   "
      f"({'edge is real vs random' if bs - s > 0.15 else 'NO EDGE OVER RANDOM - INVESTIGATE'})",
      flush=True)
print("\ndone", flush=True)
