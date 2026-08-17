"""VERIFY — are the universes this program used the CLEAN rebuilt ones?

The user recalls 26yr CAGR dropping because the SP1500 universe was poisoned. It was: two build
bugs (look-ahead membership via an ignored `Index Constituent` flag, and survivorship because the
price filter matched suffixed WRDS tickers that CRSP never carries) overstated 26yr CAGR by
~10.4pp. Both were fixed and BOTH pickles were rebuilt 2026-07-28 by one audited builder.

This re-checks the SIX defects that are detectable from the pickle alone, so the conclusion does
not rest on a filename or a date. If any check fails, every number in LOG.md is void.

  1  membership sizes ~500/400/600 (look-ahead membership inflated these to 866/1136/1508)
  2  no date-suffixed tickers in membership (the survivorship vector)
  3  delistings PRESENT -- terminated price series exist and some end in a real crash
  4  no zero prices (defect 8: CRSP writes DlyPrc=0 for 'no valid price')
  5  coverage of index members by price data, early vs late (survivorship shows as a ramp)
  6  fundamentals coverage early vs late (defect 7: ticker-joined roe ramped 66%->88%)

Run:  python3 research/VERIFY_universe_integrity.py [8yr|26yr]
"""
import os
import sys
import pickle
import bisect

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PKL = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
       else "data/wrds/sp1500_universe_2000.pkl")

print(f"\n{'='*88}\nVERIFY UNIVERSE INTEGRITY — {HZ}  ({PKL})\n{'='*88}", flush=True)
import time
print(f"  file mtime: {time.ctime(os.path.getmtime(PKL))}   "
      f"size {os.path.getsize(PKL)/1073741824:.2f} GB", flush=True)
print("  (the audited rebuild of BOTH pickles completed 2026-07-28)", flush=True)

with open(PKL, "rb") as f:
    d = pickle.load(f)
px = d["prices_df"]
mems = {k: d[k] for k in ("sp500_mem", "sp400_mem", "sp600_mem")}
fbd = d["features_by_date"]
print(f"  prices {px.shape}  {px.index[0].date()} -> {px.index[-1].date()}", flush=True)

fails = []

# --- 1 membership sizes -------------------------------------------------------------
print("\n[1] MEMBERSHIP SIZES (look-ahead inflated these to 866/1136/1508 vs true 500/400/600)",
      flush=True)
keys = {k: sorted(v.keys()) for k, v in mems.items()}
for probe in (px.index[10], px.index[len(px.index) // 2], px.index[-10]):
    sizes = {}
    for k, v in mems.items():
        i = bisect.bisect_right(keys[k], probe) - 1
        sizes[k] = len(v[keys[k][i]]) if i >= 0 else 0
    tot = sum(sizes.values())
    ok = (450 <= sizes["sp500_mem"] <= 520 and 370 <= sizes["sp400_mem"] <= 420
          and 550 <= sizes["sp600_mem"] <= 620)
    print(f"    {probe.date()}  sp500={sizes['sp500_mem']:>4} sp400={sizes['sp400_mem']:>4} "
          f"sp600={sizes['sp600_mem']:>4}  total={tot:>5}   {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        fails.append(f"membership size @{probe.date()}: {sizes}")

# --- 2 suffix pollution -------------------------------------------------------------
allsyms = set()
for k, v in mems.items():
    for lst in v.values():
        allsyms |= set(lst)
suf = [s for s in allsyms if "-" in str(s) and str(s).split("-")[-1].isdigit()]
print(f"\n[2] SUFFIXED TICKERS in membership (the survivorship vector; was 59.1% of symbols)",
      flush=True)
print(f"    {len(suf)} of {len(allsyms)} symbols carry a date suffix   "
      f"{'OK' if len(suf) == 0 else 'FAIL'}", flush=True)
if suf:
    fails.append(f"{len(suf)} suffixed tickers still present, e.g. {suf[:5]}")

# --- 3 delistings present -----------------------------------------------------------
v = px.notna().values
last = v.shape[0] - 1 - np.argmax(v[::-1], axis=0)
has = v.any(axis=0)
ended = int(((last < v.shape[0] - 20) & has).sum())
crashes = 0
for i in np.where((last < v.shape[0] - 20) & has)[0]:
    s = px.iloc[:, i].dropna()
    if len(s) > 5 and s.iloc[-1] / s.iloc[-2] - 1 < -0.50:
        crashes += 1
print(f"\n[3] DELISTINGS PRESENT (defect 6 deleted the final -100% return)", flush=True)
print(f"    {ended} series terminate early; {crashes} end in a >50% single-bar crash   "
      f"{'OK' if ended > 100 and crashes > 0 else 'FAIL'}", flush=True)
if not (ended > 100 and crashes > 0):
    fails.append(f"delisting evidence weak: ended={ended} crashes={crashes}")

# --- 4 zero prices ------------------------------------------------------------------
nonpos = int(np.nansum(px.values <= 0))
print(f"\n[4] ZERO/NEGATIVE PRICES (defect 8 zeroed 666,293 cells)", flush=True)
print(f"    {nonpos} non-positive price cells   {'OK' if nonpos == 0 else 'FAIL'}", flush=True)
if nonpos:
    fails.append(f"{nonpos} non-positive prices")

# --- 5 coverage ramp = survivorship --------------------------------------------------
print(f"\n[5] PRICE COVERAGE of index members, EARLY vs LATE "
      f"(survivorship shows as a rising ramp; pre-fix 26yr ran 35%->96%)", flush=True)
cols = set(px.columns)
cov = []
for probe in (px.index[260], px.index[len(px.index) // 2], px.index[-10]):
    mem = set()
    for k, vv in mems.items():
        i = bisect.bisect_right(keys[k], probe) - 1
        if i >= 0:
            mem |= set(vv[keys[k][i]])
    c = len([s for s in mem if s in cols]) / max(len(mem), 1)
    cov.append(c)
    print(f"    {probe.date()}  {c:.1%} of members have prices", flush=True)
ramp = cov[-1] - cov[0]
print(f"    early->late ramp {ramp:+.1%}   "
      f"{'OK (flat)' if ramp < 0.15 else 'FAIL (survivorship ramp)'}", flush=True)
if ramp >= 0.15:
    fails.append(f"coverage ramp {ramp:.1%} suggests survivorship")

# --- 6 fundamentals coverage ---------------------------------------------------------
print(f"\n[6] FUNDAMENTALS (roe) COVERAGE, EARLY vs LATE (defect 7 ramped 66%->88%)", flush=True)
fdates = sorted(fbd.keys())
fc = []
for probe in (fdates[260], fdates[len(fdates) // 2], fdates[-10]):
    fd = fbd[probe]
    n = len(fd)
    r = sum(1 for x in fd.values() if "roe" in x and np.isfinite(x.get("roe", np.nan)))
    fc.append(r / max(n, 1))
    print(f"    {probe.date()}  roe present for {r/max(n,1):.1%} of {n} names", flush=True)
framp = fc[-1] - fc[0]
print(f"    early->late ramp {framp:+.1%}   "
      f"{'OK' if framp < 0.20 else 'FAIL'}", flush=True)
if framp >= 0.20:
    fails.append(f"fundamentals ramp {framp:.1%}")

print(f"\n{'='*88}")
if fails:
    print("  RESULT: FAIL — this universe is NOT clean. Every number computed on it is void.")
    for f_ in fails:
        print("    -", f_)
else:
    print("  RESULT: PASS — all six poisoning signatures absent. This is the audited rebuild.")
print(f"{'='*88}\n", flush=True)
