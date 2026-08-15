"""evalkit — the shared validation protocol. Every experiment in this program uses it.

This module exists so the bar cannot drift experiment to experiment. It encodes, in one
place, the four controls that have each independently killed a result in this repo:

  1. MANY-START           >=12 monthly starts, judged on SIGN-CONSISTENCY, not the mean.
                          8yr per-start CAGR sigma is 7.07pp; a delta's sigma is ~2.30pp.
                          A 3-4 start A/B once reported the WRONG SIGN (gross-margin bound).
  2. TWO-HORIZON          positive on BOTH 8yr and 26yr. `umd_crash` (+0.026 / -0.020) and
                          `mkt_vol` (+0.019 / -0.010) both passed a single-period test and
                          were wrong.
  3. MATCHED EXPOSURE     anything that moves gross is scored against a constant-leverage
                          curve interpolated at its OWN realised avg_gross. Without this,
                          every de-risking rule looks like genius for merely running less.
  4. EVENT CONCENTRATION  what share of total EXCESS comes from the top 5/10/20 days?
                          A finding once passed 23/24 starts, walk-forward OOS, a sensitivity
                          plateau and all three sub-periods -- with 99.6% of its 26yr excess
                          coming from 5 days.

Do not loosen any of these to let a result through. If a rule is wrong, argue the case in
LOG.md and change it deliberately.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

# ---------------------------------------------------------------- the deployed config
DEPLOYED = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
            "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10,
            "use_rp": False, "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40,
            "vol_scale_cap": 1.0, "initial_capital": 50_000.0, "leverage": 1.49,
            "integer_shares": True, "financing_rate": 0.063}

HORIZONS = {
    "8yr":  dict(path="data/wrds/complete_sp1500_universe.pkl", years=[2018, 2019]),
    "26yr": dict(path="data/wrds/sp1500_universe_2000.pkl",     years=[2001, 2002]),
}
END = "2025-12-31"


def starts(horizon, n_per_year=6):
    """12 monthly starts by default. Consecutive monthly starts shift the 20-session
    rebalance phase by ~1 session, so this samples phase as well as calendar."""
    step = max(1, 12 // n_per_year)
    return [f"{y}-{m:02d}-03" for y in HORIZONS[horizon]["years"]
            for m in range(1, 13, step)]


# ---------------------------------------------------------------- metrics
def stat(v):
    dr = v.pct_change().dropna()
    yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
    sd = dr.std()
    neg = dr[dr < 0].std()
    return dict(
        cagr=(v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
        sharpe=dr.mean() / sd * np.sqrt(252) if sd > 0 else 0.0,
        sortino=dr.mean() / neg * np.sqrt(252) if neg > 0 else 0.0,
        vol=sd * np.sqrt(252),
        dd=((v - v.cummax()) / v.cummax()).min(),
    )


def sign_verdict(delta_cagr, delta_sharpe, n):
    """The house rule. 6/12 is a coin flip. SURVIVES needs to be clearly better than one."""
    nc = int((np.asarray(delta_cagr) > 0).sum())
    ns = int((np.asarray(delta_sharpe) > 0).sum())
    if ns >= int(np.ceil(0.75 * n)) and nc >= int(np.ceil(0.67 * n)):
        v = "SURVIVES"
    elif ns >= int(np.ceil(0.58 * n)):
        v = "marginal"
    else:
        v = "dead"
    return nc, ns, v


# ---------------------------------------------------------------- control 3
def matched_exposure_curve(bt, start, end, avg_gross, cfg=None, grid=None, _cache={}):
    """Constant-leverage baseline interpolated at `avg_gross`.

    Anything that changes exposure must be compared to THIS, not to the deployed curve.
    Runs the deployed strategy with vol_scaling OFF at several fixed leverages, then
    linearly interpolates CAGR/Sharpe/MaxDD at the variant's own realised gross.
    """
    cfg = dict(cfg or DEPLOYED)
    grid = grid or [0.7, 0.9, 1.1, 1.3, 1.5]
    key = (id(bt), start, end, tuple(grid), tuple(sorted(cfg.items(), key=str)))
    if key not in _cache:
        pts = []
        for lev in grid:
            m = bt.run(start, end, {**cfg, "leverage": lev, "vol_scaling": False})
            pts.append((m["avg_gross"], stat(m["daily_values"])))
        pts.sort(key=lambda p: p[0])
        _cache[key] = pts
    pts = _cache[key]
    xs = np.array([p[0] for p in pts])
    out = {}
    for k in ("cagr", "sharpe", "sortino", "vol", "dd"):
        out[k] = float(np.interp(avg_gross, xs, np.array([p[1][k] for p in pts])))
    out["_grid"] = [(float(x), float(p[1]["cagr"])) for x, p in zip(xs, pts)]
    return out


# ---------------------------------------------------------------- control 4
def event_concentration(base_curve, var_curve, tops=(5, 10, 20)):
    """Share of the variant's TOTAL EXCESS log-return contributed by its best N days.

    Deliberately measured on EXCESS, not on total return: a levered long-only equity book
    inherits the index's own 'miss the best days' property, so concentration of the TOTAL
    return measures beta and means nothing (see BUGS.md A4).

    >50% from 5 days => the 'edge' is a handful of events, not a process. Reject.
    """
    b = np.log(base_curve).diff().dropna()
    v = np.log(var_curve.reindex(base_curve.index).ffill()).diff().dropna()
    idx = b.index.intersection(v.index)
    ex = (v[idx] - b[idx])
    total = float(ex.sum())
    out = {"total_excess_log": total,
           "n_days": int(len(ex)),
           "pos_day_share": float((ex > 0).mean())}
    # GUARD (BUGS A8a): the share is top-N / TOTAL. When the two arms are nearly identical the
    # denominator approaches zero and the ratio explodes -- EXP-017 printed -1525% and +256%
    # for arms whose excess was a rounding error. A near-zero excess is not "concentrated", it
    # is ABSENT, and reporting a huge share there inverts the conclusion. Below the threshold
    # the share is undefined and is returned as NaN.
    # Two conditions, both necessary:
    #   (i)  the net excess is materially non-zero at all, and
    #   (ii) the net is a decent fraction of the GROSS daily movement. If the two arms differ
    #        a lot day to day but almost cancel, top-5 positive days can exceed the net and the
    #        share blows past 100% -- EXP-017/018 printed -241% and +233% that way even with a
    #        net excess well above the 2% floor. That is a signal of a SMALL NET, not of event
    #        concentration, and reporting it as concentration inverts the conclusion.
    gross = float(ex.abs().sum())
    meaningful = abs(total) > 0.02 and (gross == 0 or abs(total) > 0.10 * gross)
    out["gross_excess_log"] = gross
    out["net_to_gross"] = (abs(total) / gross) if gross > 0 else float("nan")
    out["excess_meaningful"] = bool(meaningful)
    for n in tops:
        out[f"top{n}_share"] = (float(ex.nlargest(n).sum() / total)
                                if meaningful else float("nan"))
    return out


# ---------------------------------------------------------------- the driver
def ab(bt, variant_cfg, horizon, base_cfg=None, n_starts=12, end=END,
       label="variant", matched=False, concentration=True, verbose=True):
    """Full A/B of one variant against the baseline on one horizon.

    Returns a dict with per-start deltas, the sign verdict, and (optionally) the
    matched-exposure residual and the event-concentration profile.
    """
    base_cfg = dict(base_cfg or DEPLOYED)
    sts = starts(horizon, n_per_year=max(1, n_starts // 2) if n_starts <= 12 else 6)
    sts = sts[:n_starts]
    rec = {"label": label, "horizon": horizon, "starts": sts,
           "base": [], "var": [], "avg_gross": [], "conc": [], "matched": []}
    for st in sts:
        mb = bt.run(st, end, base_cfg)
        mv = bt.run(st, end, {**base_cfg, **variant_cfg})
        cb, cv = mb["daily_values"], mv["daily_values"]
        rec["base"].append(stat(cb))
        rec["var"].append(stat(cv))
        rec["avg_gross"].append((mb["avg_gross"], mv["avg_gross"]))
        if concentration:
            rec["conc"].append(event_concentration(cb, cv))
        if matched:
            rec["matched"].append(matched_exposure_curve(bt, st, end, mv["avg_gross"],
                                                         cfg=base_cfg))
    dc = np.array([v["cagr"] - b["cagr"] for b, v in zip(rec["base"], rec["var"])])
    ds = np.array([v["sharpe"] - b["sharpe"] for b, v in zip(rec["base"], rec["var"])])
    dd = np.array([v["dd"] - b["dd"] for b, v in zip(rec["base"], rec["var"])])
    nc, ns, verdict = sign_verdict(dc, ds, len(sts))
    rec.update(dcagr=dc, dsharpe=ds, ddd=dd, n_cagr=nc, n_sharpe=ns, verdict=verdict)
    if matched:
        rec["m_dcagr"] = np.array([v["cagr"] - m["cagr"]
                                   for v, m in zip(rec["var"], rec["matched"])])
        rec["m_dsharpe"] = np.array([v["sharpe"] - m["sharpe"]
                                     for v, m in zip(rec["var"], rec["matched"])])
    if verbose:
        report(rec)
    return rec


def report(r):
    n = len(r["starts"])
    print(f"\n  {r['label']}  [{r['horizon']}, {n} starts]", flush=True)
    print(f"    base  CAGR {np.mean([b['cagr'] for b in r['base']]):+7.2%}  "
          f"Sharpe {np.mean([b['sharpe'] for b in r['base']]):.3f}  "
          f"MaxDD {np.mean([b['dd'] for b in r['base']]):.1%}", flush=True)
    print(f"    var   CAGR {np.mean([v['cagr'] for v in r['var']]):+7.2%}  "
          f"Sharpe {np.mean([v['sharpe'] for v in r['var']]):.3f}  "
          f"MaxDD {np.mean([v['dd'] for v in r['var']]):.1%}  "
          f"avgGross {np.mean([g[1] for g in r['avg_gross']]):.4f}"
          f" (base {np.mean([g[0] for g in r['avg_gross']]):.4f})", flush=True)
    print(f"    DELTA dCAGR {r['dcagr'].mean()*100:+.2f}pp  dSharpe {r['dsharpe'].mean():+.3f}  "
          f"dMaxDD {r['ddd'].mean()*100:+.2f}pp   "
          f"sign {r['n_cagr']}/{n} CAGR, {r['n_sharpe']}/{n} Sharpe  -> {r['verdict']}",
          flush=True)
    if r.get("matched"):
        print(f"    MATCHED-EXPOSURE residual: dCAGR {r['m_dcagr'].mean()*100:+.2f}pp  "
              f"dSharpe {r['m_dsharpe'].mean():+.3f}  "
              f"(vs constant-leverage curve at the variant's own gross)", flush=True)
    if r.get("conc"):
        c = r["conc"]
        print(f"    CONCENTRATION of excess: top5 {np.mean([x['top5_share'] for x in c]):.1%}  "
              f"top10 {np.mean([x['top10_share'] for x in c]):.1%}  "
              f"top20 {np.mean([x['top20_share'] for x in c]):.1%}  "
              f"| days beating base {np.mean([x['pos_day_share'] for x in c]):.1%}", flush=True)
        if np.mean([x["top5_share"] for x in c]) > 0.50:
            print("    ^^ >50% of the excess comes from 5 days. This is an EVENT, not a "
                  "process. REJECT.", flush=True)
