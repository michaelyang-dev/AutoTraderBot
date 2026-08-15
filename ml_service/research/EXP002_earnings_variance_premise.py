"""EXP-002 (IDEAS I-07) — PREMISE CHECK before building anything.

CLAIM TO TEST, cheaply, on the cross-section alone:
  A holding whose earnings announcement falls inside the coming 20-session window takes a
  large idiosyncratic gamble. We have DELIBERATELY zeroed every earnings-surprise boost
  (BUGS B4) -- i.e. we assert no ability to predict the print. If that is true, the position
  carries extra VARIANCE at ZERO extra MEAN. Removing it should raise Sharpe.

Two things must both hold or the idea is void:
  (a) forward 20d realised variance is MATERIALLY higher for names reporting in the window;
  (b) forward 20d MEAN return is NOT materially higher (else we would be selling a premium
      we are actually being paid for -- e.g. an earnings-announcement risk premium, which is
      documented in the literature and would flip the sign of this idea).

LEAK DISCIPLINE
  `rdq` is Compustat's report date. It is the date the number became public, so using
  "is there an rdq in (t, t+20]" at time t requires knowing a FUTURE announcement date.
  That is legitimate ONLY because announcement dates are pre-announced by the company weeks
  ahead -- but Compustat's `rdq` is the REALISED date, which can differ from the scheduled
  one. To stay honest this study uses the PRIOR quarter's rdq + ~91 days as the ESTIMATE of
  the next one, which is information available at time t. The naive version (true future rdq)
  is computed alongside ONLY to bound how much the estimate loses.

Run:  python3 research/EXP002_earnings_variance_premise.py
"""
import os
import sys
import pickle

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PKL = "data/wrds/complete_sp1500_universe.pkl"
H = 20          # holding horizon in sessions


def main():
    print("loading compustat rdq ...", flush=True)
    cq = pd.read_parquet("data/wrds/compustat_quarterly.parquet",
                         columns=["tic", "datadate", "rdq"])
    cq = cq.dropna(subset=["tic", "rdq"])
    cq["rdq"] = pd.to_datetime(cq["rdq"])
    cq = cq.drop_duplicates(subset=["tic", "rdq"]).sort_values(["tic", "rdq"])
    print(f"  {len(cq):,} rows, {cq['tic'].nunique():,} tickers, "
          f"{cq['rdq'].min().date()} -> {cq['rdq'].max().date()}", flush=True)

    print("loading price matrix ...", flush=True)
    with open(PKL, "rb") as f:
        px = pickle.load(f)["prices_df"]
    dates = px.index
    print(f"  prices {px.shape}", flush=True)

    # rdq per ticker as a sorted array, plus the PRIOR rdq (causal estimate source)
    by_tic = {t: g["rdq"].values for t, g in cq.groupby("tic")}
    cols = [c for c in px.columns if c in by_tic]
    print(f"  {len(cols):,} of {px.shape[1]:,} price columns have rdq history", flush=True)

    # sample one cross-section every 10 sessions, need t+H to exist
    sample_idx = list(range(252, len(dates) - H, 10))
    print(f"  {len(sample_idx)} cross-sections", flush=True)

    rows = []
    logpx = np.log(px[cols])
    ret_h = (px[cols].shift(-H) / px[cols] - 1)
    for i in sample_idx:
        t = dates[i]
        t_end = dates[i + H]
        r = ret_h.iloc[i]
        # realised variance over the window, per name
        win = logpx.iloc[i:i + H + 1].diff().iloc[1:]
        rv = win.std() * np.sqrt(252)
        for sym in cols:
            arr = by_tic[sym]
            # ---- CAUSAL estimate: last rdq strictly BEFORE t, + 91 days ----
            j = np.searchsorted(arr, np.datetime64(t), side="left") - 1
            if j < 0:
                continue
            est_next = arr[j] + np.timedelta64(91, "D")
            est_in = bool(np.datetime64(t) < est_next <= np.datetime64(t_end))
            # ---- NAIVE (uses the true future date) -- upper bound only ----
            k = np.searchsorted(arr, np.datetime64(t), side="right")
            true_in = bool(k < len(arr) and arr[k] <= np.datetime64(t_end))
            rr, vv = r.get(sym), rv.get(sym)
            if rr is None or vv is None or not np.isfinite(rr) or not np.isfinite(vv):
                continue
            rows.append((i, est_in, true_in, rr, vv))

    df = pd.DataFrame(rows, columns=["xs", "est_in", "true_in", "fwd_ret", "fwd_vol"])
    # ---- MANDATORY CONFOUND CONTROL --------------------------------------------------
    # The raw in/out comparison is NOT cross-sectional: the share of names reporting swings
    # from ~5% to ~60% across the calendar, so dates with many reporters dominate the "in"
    # group and dates with few dominate the "out" group. Forward 20d return is mostly market
    # beta, so that turns a stock-level question into a "were earnings-season months good
    # months" question. The first pass of this script did exactly that and produced t=+20.
    # De-mean every quantity WITHIN its cross-section: now "in" and "out" are compared only
    # against names observed on the SAME day.
    df["ret_dm"] = df["fwd_ret"] - df.groupby("xs")["fwd_ret"].transform("mean")
    df["vol_dm"] = df["fwd_vol"] / df.groupby("xs")["fwd_vol"].transform("mean")
    print(f"\n  n = {len(df):,} name-dates\n", flush=True)

    for flag, name in (("true_in", "TRUE rdq inside window (upper bound, uses future)"),
                       ("est_in", "ESTIMATED rdq inside window (causal, prior+91d)")):
        a = df[df[flag]]
        b = df[~df[flag]]
        print(f"  --- {name} ---", flush=True)
        print(f"    share flagged        : {df[flag].mean():.1%}  "
              f"(per-date range {df.groupby('xs')[flag].mean().min():.0%}"
              f"-{df.groupby('xs')[flag].mean().max():.0%})", flush=True)
        print(f"    RAW  fwd mean in/out : {a.fwd_ret.mean():+.3%} / {b.fwd_ret.mean():+.3%}"
              f"   diff {(a.fwd_ret.mean()-b.fwd_ret.mean())*100:+.3f}pp   "
              f"<-- CONFOUNDED, do not use", flush=True)
        # within-cross-section: per-date difference of means, then t-stat on the 214 dates
        g = df.groupby(["xs", flag])["fwd_ret"].mean().unstack()
        gd = (g[True] - g[False]).dropna()
        gv = df.groupby(["xs", flag])["vol_dm"].mean().unstack()
        gvd = (gv[True] / gv[False]).dropna()
        print(f"    WITHIN-DATE vol ratio: {gvd.mean():.3f}  "
              f"(per-date, n={len(gvd)} dates)", flush=True)
        print(f"    WITHIN-DATE mean diff: {gd.mean()*100:+.3f}pp  "
              f"t={gd.mean()/(gd.std(ddof=1)/np.sqrt(len(gd))):+.2f}  "
              f"(n={len(gd)} dates, {100*(gd>0).mean():.0f}% of dates positive)", flush=True)
        print(f"    de-meaned sd  in/out : {a.ret_dm.std():.2%} / {b.ret_dm.std():.2%}   "
              f"ratio {a.ret_dm.std()/b.ret_dm.std():.3f}\n", flush=True)

    print("  VERDICT GUIDE", flush=True)
    print("   vol ratio < 1.10                       -> premise (a) fails, idea is void.",
          flush=True)
    print("   mean diff clearly POSITIVE (t > 2)     -> an earnings risk PREMIUM exists and we",
          flush=True)
    print("                                             would be selling something we are paid",
          flush=True)
    print("                                             for. Idea flips sign.", flush=True)
    print("   vol up, mean flat, return/risk < 1     -> proceed to a full backtest arm.",
          flush=True)


if __name__ == "__main__":
    main()
