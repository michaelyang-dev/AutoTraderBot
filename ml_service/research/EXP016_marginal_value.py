"""EXP-016 — DOES THE STRATEGY ADD ANYTHING TO A PORTFOLIO THAT ALREADY HOLDS THE INDEX?

THE QUESTION NO PRIOR WORK IN THIS REPO ASKS
  Everything here compares the strategy to a benchmark as a REPLACEMENT. That is the wrong
  comparison for an owner who could hold both. The right one is MARGINAL: starting from a
  passive index portfolio, does adding some of this strategy improve it?

  This matters because of what cycles 3-14 established:
    - standalone 26yr the strategy is roughly a wash with passive on return and behind on risk
      (EXP-004 / EXP-010: +11.97% / 0.531 / -64.1% vs EW SP1500 +11.17% / 0.590 / -58.5%)
    - but it is a CONCENTRATED top-5 momentum book, so its return stream is NOT the index's
  A strategy can be worse standalone and still be valuable at the margin, purely through
  imperfect correlation. That is a different, and much more forgiving, test -- and it is the
  one that decides whether any of this work should be held at all.

METHOD
  Blend the strategy's daily NAV path with a passive path, rebalancing the mix MONTHLY (a real
  account would drift otherwise, and drift is not free). Sweep w = 0 .. 1 in 0.1 steps for two
  passive legs: SPY and quarterly-reconstituted EW SP1500. Report the full frontier so the
  optimum is visible rather than asserted, and report the correlation that drives it.

HONESTY GUARDS
  1. The mix weight is NOT optimised and then reported as a result. The whole frontier is shown.
     Any "best w" is in-sample by construction and is labelled as such.
  2. Both legs run over the identical calendar from the identical price matrix.
  3. The passive leg pays NO costs and holds 1,500 names -- it is a favourable benchmark, so any
     marginal value found here is a LOWER bound on the strategy's contribution.
  4. Strategy metrics are averaged over 12 starts; the passive leg is one path. Stated, not hidden.

Run:  python3 research/EXP016_marginal_value.py [8yr|26yr]
"""
import os
import sys
import pickle
import bisect
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
WS = [round(x, 2) for x in np.arange(0, 1.01, 0.1)]


def met(r):
    """r = daily simple returns."""
    v = (1 + r).cumprod()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    sd = r.std()
    neg = r[r < 0].std()
    return dict(cagr=v.iloc[-1] ** (1 / yrs) - 1,
                sharpe=r.mean() / sd * np.sqrt(252) if sd > 0 else 0.0,
                sortino=r.mean() / neg * np.sqrt(252) if neg > 0 else 0.0,
                dd=((v - v.cummax()) / v.cummax()).min())


def ew_series(px, mems, idx):
    """quarterly reconstitution, buy-and-hold in between (EXP-004 convention)."""
    keys = {k: sorted(v.keys()) for k, v in mems.items()}
    sub = px.reindex(idx)
    dr = sub.pct_change(fill_method=None)
    w, cur, out, last_p = None, [], [], None
    for i, dt in enumerate(sub.index):
        p = (dt.year, (dt.month - 1) // 3)
        if p != last_p:
            names = set()
            for k, v in mems.items():
                kk = keys[k]
                j = bisect.bisect_right(kk, dt) - 1
                if j >= 0:
                    names |= set(v[kk[j]])
            row0 = sub.iloc[i]
            cur = [x for x in names if x in sub.columns and np.isfinite(row0.get(x, np.nan))]
            w = np.full(len(cur), 1.0 / len(cur)) if cur else None
            last_p = p
        if i == 0 or w is None or not len(cur):
            out.append(0.0); continue
        r_i = dr.iloc[i].reindex(cur).values
        r_i = np.where(np.isfinite(r_i), r_i, 0.0)
        out.append(float((w * r_i).sum()))
        w = w * (1.0 + r_i); t = w.sum()
        if t > 0:
            w = w / t
    return pd.Series(out, index=sub.index).fillna(0.0)


def blend(rs, rp, w):
    """w in strategy, 1-w in passive, rebalanced MONTHLY (drift is not free)."""
    idx = rs.index
    grp = pd.Series(idx.year * 12 + idx.month, index=idx)
    out = np.empty(len(idx)); a, b = w, 1 - w
    last = None
    for i in range(len(idx)):
        if grp.iloc[i] != last:
            a, b = w, 1 - w
            last = grp.iloc[i]
        tot = a + b
        out[i] = (a * rs.iloc[i] + b * rp.iloc[i]) / tot if tot > 0 else 0.0
        a *= (1 + rs.iloc[i]); b *= (1 + rp.iloc[i])
    return pd.Series(out, index=idx)


def main():
    t0 = time.time()
    with open(PATH, "rb") as f:
        d = pickle.load(f)
    px = d["prices_df"]
    mems = {k: d[k] for k in ("sp500_mem", "sp400_mem", "sp600_mem")}
    del d

    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*96}\nEXP-016 MARGINAL VALUE — {HZ}, {len(STARTS)} strategy starts\n{'='*96}",
          flush=True)

    strat, gs = [], []
    for st in STARTS:
        m = bt.run(st, END, {**DEPLOYED, "financing_curve": True})
        v = m["daily_values"]
        strat.append(v.pct_change().dropna()); gs.append(m["avg_gross"])
    global G_STRAT, CTRL_X, CTRL_S
    G_STRAT = float(np.mean(gs))
    print(f"  strategy curves built, avgGross {G_STRAT:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # control curve: pure strategy across a leverage grid, same financing, same starts
    CTRL_X, CTRL_S = [], []
    for L in (0.90, 1.10, 1.30, 1.49, 1.75):
        ss, gg = [], []
        for st in STARTS:
            m = bt.run(st, END, {**DEPLOYED, "leverage": L, "financing_curve": True})
            ss.append(met(m["daily_values"].pct_change().dropna())["sharpe"])
            gg.append(m["avg_gross"])
        CTRL_X.append(float(np.mean(gg))); CTRL_S.append(float(np.mean(ss)))
    o = np.argsort(CTRL_X)
    CTRL_X = list(np.array(CTRL_X)[o]); CTRL_S = list(np.array(CTRL_S)[o])
    print("  control (pure strategy, leverage grid): " +
          "  ".join(f"g={x:.3f}->Sh {y:.3f}" for x, y in zip(CTRL_X, CTRL_S)), flush=True)

    for leg in ("SPY", "EW"):
        print(f"\n  === passive leg: {leg} ===", flush=True)
        print(f"  {'w_strat':>8}{'CAGR':>10}{'Sharpe':>9}{'Sortino':>9}{'MaxDD':>9}"
              f"{'dSharpe vs w=0':>16}", flush=True)
        rows = {w: [] for w in WS}
        corrs = []
        for rs in strat:
            idx = rs.index
            if leg == "SPY":
                sp = px["SPY"].reindex(idx).ffill()
                rp = sp.pct_change().fillna(0.0)
            else:
                rp = ew_series(px, mems, idx)
            rp = rp.reindex(idx).fillna(0.0)
            corrs.append(float(np.corrcoef(rs.values, rp.values)[0, 1]))
            for w in WS:
                rows[w].append(met(blend(rs, rp, w)))
        base_s = np.mean([r["sharpe"] for r in rows[0.0]])
        for w in WS:
            c = np.mean([r["cagr"] for r in rows[w]])
            s = np.mean([r["sharpe"] for r in rows[w]])
            so = np.mean([r["sortino"] for r in rows[w]])
            dd = np.mean([r["dd"] for r in rows[w]])
            star = "  <- max Sharpe" if s == max(
                np.mean([r["sharpe"] for r in rows[x]]) for x in WS) else ""
            print(f"  {w:>8.1f}{c:>+10.2%}{s:>9.3f}{so:>9.3f}{dd:>9.1%}"
                  f"{s-base_s:>+16.3f}{star}", flush=True)
        print(f"  daily corr(strategy, {leg}) = {np.mean(corrs):.3f}", flush=True)
        # ---- MANDATORY CONTROL: is the blend's Sharpe gain just DE-LEVERING? ----------
        # At w in the strategy, effective gross = w*G_strat + (1-w)*1.0, which is LOWER than
        # the strategy's own gross. Since Sharpe rises as leverage falls (EXP-010), part of any
        # blend gain could be pure de-levering with no diversification content at all. This
        # runs the pure strategy at the leverage that reproduces the blend's OWN gross and
        # compares. If the blend does not beat that, it is a level effect and not a finding.
        print(f"  {'--- exposure-matched control ---':<36}", flush=True)
        for w in (0.2, 0.3, 0.4):
            eff = w * G_STRAT + (1 - w) * 1.0
            ms = float(np.interp(eff, CTRL_X, CTRL_S))
            bs_ = np.mean([r["sharpe"] for r in rows[w]])
            print(f"  w={w:.1f}: blend gross {eff:.3f}  blend Sharpe {bs_:.3f}  vs pure "
                  f"strategy at the SAME gross {ms:.3f}   -> diversification adds "
                  f"{bs_-ms:+.3f}", flush=True)
        print(f"  NOTE: any 'best w' above is IN-SAMPLE by construction. The frontier's SHAPE "
              f"is\n        the finding; the argmax is not a recommendation.", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
