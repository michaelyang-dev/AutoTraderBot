"""AUDIT 01 — structural leak audit of the backtest harness DATA LAYER.

No strategy code. Just: does the price matrix / feature panel contain the classic
inflation mechanisms? Written to be re-runnable and cheap-ish (one pickle load).
"""
import os, sys, pickle, json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd

PKL = sys.argv[1] if len(sys.argv) > 1 else "data/wrds/complete_sp1500_universe.pkl"
OUT = {}
print(f"loading {PKL} ...", flush=True)
with open(PKL, "rb") as f:
    d = pickle.load(f)
px = d["prices_df"]
print(f"prices_df {px.shape}  {px.index[0].date()} -> {px.index[-1].date()}", flush=True)
OUT["shape"] = list(px.shape)

# ---------- A. NaN structure: gaps vs true delisting ----------
v = px.notna().values
n_dates, n_syms = v.shape
first = np.argmax(v, axis=0)
has = v.any(axis=0)
last = n_dates - 1 - np.argmax(v[::-1], axis=0)
span = np.where(has, last - first + 1, 0)
obs = v.sum(axis=0)
gaps = np.where(has, span - obs, 0)
print("\n=== A. price matrix NaN structure ===")
print(f"  symbols with any data      : {has.sum()}/{n_syms}")
print(f"  symbols with INTERIOR gaps : {(gaps > 0).sum()}  (total gap-days {int(gaps.sum()):,})")
print(f"  median gap-days | max      : {np.median(gaps[gaps>0]) if (gaps>0).any() else 0} | {gaps.max()}")
print(f"  series ENDING before final date (delist/removal): {(last < n_dates-1).sum()}")
OUT["interior_gap_syms"] = int((gaps > 0).sum()); OUT["gap_days"] = int(gaps.sum())
OUT["ended_early"] = int((last < n_dates - 1).sum())

# ---------- B. what happens at the end of a terminated series ----------
ends = np.where(has & (last < n_dates - 20))[0]
cols = px.columns.values
term_last_ret = []
for i in ends:
    s = px.iloc[:, i].dropna()
    if len(s) > 5:
        term_last_ret.append(s.iloc[-1] / s.iloc[-2] - 1)
tr = np.array(term_last_ret)
print("\n=== B. terminal-bar return of series that stop early ===")
print(f"  n={len(tr)}  mean {tr.mean():+.4f}  median {np.median(tr):+.4f}  "
      f"p05 {np.percentile(tr,5):+.3f}  p95 {np.percentile(tr,95):+.3f}")
print(f"  share with last bar < -20%: {(tr < -0.20).mean():.2%}   < -50%: {(tr < -0.50).mean():.2%}")
print("  -> if ~0 and symmetric, CRSP price series do NOT encode the delisting crash;")
print("     the harness's `today.get(sym, entry_px)` fallback then liquidates at ENTRY price.")
OUT["term_ret_mean"] = float(tr.mean()); OUT["term_ret_median"] = float(np.median(tr))

# ---------- C. zero / negative / absurd prices ----------
arr = px.values
bad_nonpos = np.nansum(arr <= 0)
with np.errstate(invalid="ignore"):
    rets = px.pct_change().values
absurd = np.nansum(np.abs(rets) > 1.0)
print("\n=== C. bad ticks ===")
print(f"  non-positive prices: {int(bad_nonpos)}")
print(f"  |1-day return| > 100%: {int(absurd)}  ({int(absurd)/max(np.isfinite(rets).sum(),1):.6%} of obs)")
OUT["nonpos"] = int(bad_nonpos); OUT["absurd_rets"] = int(absurd)

# ---------- D. feature panel timestamp alignment ----------
fbd = d["features_by_date"]
fdates = sorted(fbd.keys())
print("\n=== D. features_by_date ===")
print(f"  dates {len(fdates)}  {fdates[0].date()} -> {fdates[-1].date()}")
sample = fbd[fdates[len(fdates)//2]]
k0 = next(iter(sample))
print(f"  features per symbol ({len(sample[k0])}): {sorted(sample[k0].keys())}")
OUT["features"] = sorted(sample[k0].keys())

# does a PRICE feature on date D use date D's close? test ret_20d vs px
dtest = fdates[len(fdates)//2]
fm = fbd[dtest]
chk = []
for sym, fd in list(fm.items())[:4000]:
    if "ret_20d" not in fd or sym not in px.columns:
        continue
    s = px[sym].loc[:dtest].dropna()
    if len(s) < 25:
        continue
    # trailing, INCLUDING dtest close
    r_incl = s.iloc[-1] / s.iloc[-21] - 1
    r_excl = s.iloc[-2] / s.iloc[-22] - 1
    chk.append((fd["ret_20d"], r_incl, r_excl))
c = np.array(chk)
if len(c):
    print(f"  ret_20d vs trailing INCLUDING today  : corr {np.corrcoef(c[:,0],c[:,1])[0,1]:.6f}  "
          f"mad {np.abs(c[:,0]-c[:,1]).mean():.6f}")
    print(f"  ret_20d vs trailing EXCLUDING today  : corr {np.corrcoef(c[:,0],c[:,2])[0,1]:.6f}  "
          f"mad {np.abs(c[:,0]-c[:,2]).mean():.6f}")
    print("  -> INCLUDING-today match => decision uses date-D close; harness also FILLS at")
    print("     date-D close. Zero-latency close-to-close. Faithful to live MOC, but the")
    print("     shift test is mandatory.")

# forward-looking check: is any feature correlated with the NEXT 20d return more than
# is plausible? (a crude leak detector across the whole panel)
print("\n=== E. crude leak sweep: |IC| of every feature vs FORWARD 20d return ===")
sel = fdates[::63]
feats = OUT["features"]
fwd_ic = {f: [] for f in feats}
pxs = px
for dt in sel:
    j = pxs.index.searchsorted(dt)
    if j + 20 >= len(pxs.index):
        continue
    p0 = pxs.iloc[j]; p1 = pxs.iloc[j + 20]
    fr = (p1 / p0 - 1)
    fm = fbd.get(dt, {})
    syms = [s for s in fm if s in pxs.columns]
    if len(syms) < 100:
        continue
    frs = fr.reindex(syms)
    for f in feats:
        vv = pd.Series({s: fm[s].get(f, np.nan) for s in syms}, dtype=float)
        m = vv.notna() & frs.notna()
        if m.sum() > 100:
            fwd_ic[f].append(np.corrcoef(vv[m].rank(), frs[m].rank())[0, 1])
rows = []
for f in feats:
    a = np.array(fwd_ic[f])
    if len(a) > 5:
        rows.append((f, a.mean(), a.std() / np.sqrt(len(a)), len(a)))
rows.sort(key=lambda r: -abs(r[1]))
print(f"  {'feature':<28}{'meanIC':>9}{'sem':>8}{'n':>5}")
for f, m, se, n in rows:
    flag = "  <-- SUSPICIOUS" if abs(m) > 0.15 else ""
    print(f"  {f:<28}{m:>+9.4f}{se:>8.4f}{n:>5}{flag}")
OUT["fwd_ic"] = {f: float(m) for f, m, _, _ in rows}

with open("research/_audit01.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nwrote research/_audit01.json")
