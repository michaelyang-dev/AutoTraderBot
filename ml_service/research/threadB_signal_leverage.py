"""
THREAD B — Futures/macro/credit as a LEVERAGE signal.

Question: can a regime signal the equity book is BLIND to (HY credit spread, macro
PC1, VIX term) time de-risking / re-levering BETTER than the live vol-scaled 1.49x
policy? De-levering in bad states is convex (big DD win); levering up is Kelly-flat
(little CAGR gain) -> value is on the de-risk side.

METHOD (extends leverage_policy_search.py + exposure_lib.py):
  r_off = 1x UNSCALED book daily returns (fork-run, deployed data OFF).
  vs    = live vol-scale path (logged from a vol_scaling run; piecewise-constant, causal).
  base_gross_t = BASE_L * min(cap, vs_t)                       # = the live 1.49x policy
  regime signal -> causal expanding-percentile risk pr_t (acts NEXT day), then one of:
    override: gross = base * (derisk if pr>=pct else 1)         # hard cut in the tail
    gate    : gross = min(base, 1.0) where pr>=pct else base    # drop ALL leverage in tail
    blend   : gross = base * (1-(1-derisk)*ramp(pr; pct->1))    # smooth de-risk
  optional lever-up: gross *= up_mult where pr<=calm_pct (CAGR-first variants only).
  r_t = gross_t*r_off_t - max(0,gross_t-1)*RATE/252. Financing 6.3% borrowed; cash 0%.

THE HONEST BAR (the point of the file):
  A de-risk signal that merely LOWERS AVERAGE leverage cuts DD for free (Kelly, not skill).
  So every policy is judged vs TWO baselines, BOTH periods, one param set:
    (1) LIVE 1.49x vol-scaled (base, cap1.0) — the incumbent.
    (2) CONSTANT leverage at the policy's OWN average gross — the timing control.
  Real DD-first win = matches incumbent CAGR (>= -1.5pp) with lower MaxDD, AND beats the
  matched-gross constant on MaxDD (proves the de-risk is TIMED, not just smaller).
  Pareto win = CAGR up AND MaxDD up vs incumbent, both periods.

Signals + honest history:
  hy_oas  (1996->2026 spliced WRDS+local)  BOTH periods.  high=risk-off.
  baa_aaa (Moody's, 1986->2025)            BOTH periods.  high=risk-off.
  pc1     (_macro_pcs, 2000->2026)         BOTH periods.  LOW =risk-off.
  vix_term(^VIX/^VIX3M, 2016->)            2018-only; reported, flagged.
"""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML); sys.path.insert(0, os.path.join(ML, "research"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SRC = os.path.join(ML, "main_production_backtest.py")
FORK = os.path.join(ML, "research", "_threadb_fork.py")
A1 = '        self._rebal_log = []  # research: (date, {sym: target_weight}) per rebalance when record_targets set'
P1 = A1 + '\n        self._vs_log = []'
A2 = '                    vol_scale = min(1.5, max(0.3, vol_target / realized_vol))'
P2 = ('                    vol_scale = min(config.get("vol_scale_cap", 1.5), max(0.3, vol_target / realized_vol))\n'
      '                    self._vs_log.append((date, vol_scale))')
src = open(SRC).read()
assert src.count(A1) == 1 and src.count(A2) == 1, "fork anchors changed — re-check"
open(FORK, "w").write(src.replace(A1, P1).replace(A2, P2))
print("fork written", flush=True)
from _threadb_fork import FastBacktester  # noqa: E402

V12 = {"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
       "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
       "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10}}
VS = {"vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0}
PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl",
     ["2018-01-02", "2018-01-17", "2018-02-01"], "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl",
     ["2001-01-02", "2001-01-17"], "2025-12-31"),
]
BASE_L, RATE = 1.49, 0.063
DERISK_HALF, DERISK_FLOOR = 0.50, 0.34   # halve leverage / go to live vol-scale floor
UP_MULT = 1.07                            # ~cap 1.6 / 1.49 — Kelly-flat by prior
TAIL, CALM = 0.80, 0.20                   # round, pre-committed percentiles (not tuned)


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats(r):
    r = r.dropna()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    c = (1 + r).cumprod()
    return (c.iloc[-1] ** (1 / yrs) - 1, r.std() * np.sqrt(252),
            (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            ((c - c.cummax()) / c.cummax()).min())


def load_signals():
    cr = pd.read_parquet(os.path.join(ML, "research", "_credit_signal.parquet"))
    pc = pd.read_parquet(os.path.join(ML, "research", "_macro_pcs.parquet"))[["PC1"]]
    out = {"hy_oas": cr["hy_oas"].dropna(), "baa_aaa": cr["baa_aaa"].dropna(),
           "pc1": pc["PC1"].dropna()}
    try:
        vx = pd.read_parquet(os.path.join(ML, "data/enhanced_data/vix_cache.parquet"))
        out["vix_term"] = (vx["^VIX"] / vx["^VIX3M"]).dropna()
    except Exception:
        pass
    return out


def risk_pct(sig, idx, high_is_risk, expanding_min=252):
    """Causal expanding-percentile of RISK in [0,1], acts next day (shift 1).
    high_is_risk: risk rises with sig (credit); else risk rises as sig falls (PC1)."""
    s = sig.reindex(idx.union(sig.index)).sort_index().ffill().reindex(idx)
    x = s if high_is_risk else -s
    pr = x.expanding(min_periods=expanding_min).apply(lambda a: (a[-1] >= a).mean(), raw=True)
    return pr.shift(1).fillna(0.5)


# ---------------- leverage overlay primitives ----------------
def base_gross(r_off, vs, cap=1.0):
    return BASE_L * np.minimum(cap, vs.values)


def apply_gross(r_off, gross, rate=RATE):
    rr = r_off.values
    r = gross * rr - np.maximum(0.0, gross - 1.0) * (rate / 252)
    return pd.Series(r, index=r_off.index), float(np.mean(gross))


def ramp(pr, lo, hi):
    return np.clip((pr - lo) / max(1e-9, (hi - lo)), 0.0, 1.0)


def build_gross(r_off, vs, pr, mode="override", pct=TAIL, derisk=DERISK_HALF,
                pr_up=None, calm=CALM, up_mult=1.0, cap=1.0):
    base = base_gross(r_off, vs, cap)
    prv = pr.values
    off = prv >= pct
    if mode == "override":
        g = base * np.where(off, derisk, 1.0)
    elif mode == "gate":
        g = np.where(off, np.minimum(base, 1.0), base)
    elif mode == "blend":
        g = base * (1.0 - (1.0 - derisk) * ramp(prv, pct, 1.0))
    else:
        raise ValueError(mode)
    if pr_up is not None and up_mult != 1.0:
        calm_flag = (pr_up.values <= calm) & (~off)
        g = np.where(calm_flag, g * up_mult, g)
    return g


# ------------- data prep (one expensive run set per period) -------------
def prep_runs(bt, starts, end, signals):
    runs = []
    for st in starts:
        r_off = bt.run(st, end, dict(V12))["daily_values"].pct_change().dropna()
        bt._vs_log = []
        cfg = dict(V12); cfg.update(VS)
        bt.run(st, end, cfg)
        vs = pd.Series({d: v for d, v in bt._vs_log}); vs.index = pd.to_datetime(vs.index)
        vs = vs.reindex(r_off.index.union(vs.index)).ffill().reindex(r_off.index).fillna(1.0)
        idx = r_off.index
        prc = {"hy_oas": risk_pct(signals["hy_oas"], idx, True),
               "baa_aaa": risk_pct(signals["baa_aaa"], idx, True),
               "pc1": risk_pct(signals["pc1"], idx, False)}
        if "vix_term" in signals:
            prc["vix_term"] = risk_pct(signals["vix_term"], idx, True)
        runs.append((r_off, vs, idx, prc))
    return runs


# ------------- policy suite (economically-motivated, round thresholds) -------------
# each: name, fn(r_off, vs, prc)->gross array, only8yr flag
def make_policies():
    P = []

    def override(sig, pct, derisk):
        return lambda ro, vs, prc: build_gross(ro, vs, prc[sig], "override", pct, derisk)

    def gate(sig, pct):
        return lambda ro, vs, prc: build_gross(ro, vs, prc[sig], "gate", pct)

    def blend(sig, pct, derisk):
        return lambda ro, vs, prc: build_gross(ro, vs, prc[sig], "blend", pct, derisk)

    def two_signal(a, b, pct, derisk):  # de-risk only when BOTH in tail (AND-confirm)
        def f(ro, vs, prc):
            base = base_gross(ro, vs)
            off = (prc[a].values >= pct) & (prc[b].values >= pct)
            return base * np.where(off, derisk, 1.0)
        return f

    def bidir(sig, pct, derisk, up):  # de-lever in tail + lever-up in calm
        return lambda ro, vs, prc: build_gross(ro, vs, prc[sig], "override", pct, derisk,
                                               pr_up=prc[sig], calm=CALM, up_mult=up)

    # B2 credit-override (top candidate, DD-first) — two derisk depths x two tails
    P += [("B2 credit override p80 x0.50", override("hy_oas", 0.80, DERISK_HALF), False),
          ("B2 credit override p80 floor", override("hy_oas", 0.80, DERISK_FLOOR), False),
          ("B2 credit override p90 x0.50", override("hy_oas", 0.90, DERISK_HALF), False),
          ("B2 baa-aaa override p80 x0.50", override("baa_aaa", 0.80, DERISK_HALF), False)]
    # B1 PC1 (both-period workhorse) de-lever only
    P += [("B1 pc1 override p80 x0.50", override("pc1", 0.80, DERISK_HALF), False),
          ("B1 pc1 override p90 x0.50", override("pc1", 0.90, DERISK_HALF), False)]
    # B6 two-signal confirmation (credit AND pc1)
    P += [("B6 credit&pc1 p80 x0.50", two_signal("hy_oas", "pc1", 0.80, DERISK_HALF), False),
          ("B6 credit&pc1 p70 x0.50", two_signal("hy_oas", "pc1", 0.70, DERISK_HALF), False)]
    # B4 fusion ablation (same signal, three mechanisms)
    P += [("B4 credit GATE p80", gate("hy_oas", 0.80), False),
          ("B4 credit BLEND p80 x0.50", blend("hy_oas", 0.80, DERISK_HALF), False)]
    # B5 the two required variants (credit): DD-first vs CAGR-first
    P += [("B5a credit de-lever ONLY (DDfirst)", override("hy_oas", 0.80, DERISK_HALF), False),
          ("B5b credit + lever-up (CAGRfirst)", bidir("hy_oas", 0.80, DERISK_HALF, UP_MULT), False)]
    # B1-up bidirectional PC1
    P += [("B1up pc1 bidir p80 x0.50 up1.07", bidir("pc1", 0.80, DERISK_HALF, UP_MULT), False)]
    # B3 VIX-term gate (2018-only — flagged)
    P += [("B3 vix-term override p80 x0.50 [8yr]", override("vix_term", 0.80, DERISK_HALF), True)]
    return P


def fmt(v, pct=True):
    return f"{v:+.1%}" if pct else f"{v:.2f}"


def main():
    signals = load_signals()
    print("signals:", {k: (str(v.index.min())[:10], str(v.index.max())[:10], len(v))
                        for k, v in signals.items()}, flush=True)
    policies = make_policies()
    results = {}   # name -> {period -> (cagr,vol,sh,dd,gross, mc_dd, mc_cagr)}
    incumb = {}    # period -> (cagr,vol,sh,dd,gross)

    for pname, path, starts, end in PERIODS:
        print("\n" + "=" * 110, flush=True)
        print(f"PERIOD {pname} | {len(starts)}-start | financing {RATE:.1%} borrowed | incumbent=LIVE 1.49x vol-scaled", flush=True)
        print("=" * 110, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path); clear_deployed(bt)
        runs = prep_runs(bt, starts, end, signals)
        print(f"(loaded+ran {time.time()-t0:.0f}s)", flush=True)

        # incumbent = base gross (cap1.0), the live policy
        ic, iv, ish, idd, ig = [], [], [], [], []
        for ro, vs, idx, prc in runs:
            ser, g = apply_gross(ro, base_gross(ro, vs))
            c, v, s, d = stats(ser); ic.append(c); iv.append(v); ish.append(s); idd.append(d); ig.append(g)
        incumb[pname] = (np.mean(ic), np.mean(iv), np.mean(ish), np.mean(idd), np.mean(ig))
        hdr = (f"{'policy':<40}{'CAGR':>8}{'Vol':>7}{'Shrp':>6}{'MaxDD':>8}{'gross':>7}"
               f"{'|match-gross':>13}{'DDdelta':>9}")
        print(f"{'INCUMBENT LIVE 1.49x':<40}{fmt(incumb[pname][0])}{incumb[pname][1]:>7.1%}"
              f"{incumb[pname][2]:>6.2f}{fmt(incumb[pname][3]):>8}{incumb[pname][4]:>7.2f}", flush=True)
        print(hdr); print("-" * len(hdr), flush=True)

        for name, fn, only8 in policies:
            if only8 and pname.startswith("26yr"):
                continue
            cs, vls, ss, ds, gs, mcd, mcc = [], [], [], [], [], [], []
            for ro, vs, idx, prc in runs:
                g = fn(ro, vs, prc)
                ser, gav = apply_gross(ro, g)
                c, v, s, d = stats(ser)
                # matched-gross constant control
                mser, _ = apply_gross(ro, np.full(len(ro), gav))
                mc, _, _, md = stats(mser)
                cs.append(c); vls.append(v); ss.append(s); ds.append(d); gs.append(gav)
                mcd.append(md); mcc.append(mc)
            cagr, vol, sh, dd, gr = np.mean(cs), np.mean(vls), np.mean(ss), np.mean(ds), np.mean(gs)
            mdd, mcagr = np.mean(mcd), np.mean(mcc)
            results.setdefault(name, {})[pname] = (cagr, vol, sh, dd, gr, mdd, mcagr)
            beats_match = dd > mdd + 0.002   # policy DD shallower than matched-gross constant
            tag = " *timed" if beats_match else ""
            print(f"{name:<40}{fmt(cagr)}{vol:>7.1%}{sh:>6.2f}{fmt(dd):>8}{gr:>7.2f}"
                  f"{fmt(mdd):>13}{(dd-mdd)*100:>+8.1f}pp{tag}", flush=True)
        del bt

    # -------- verdict --------
    print("\n" + "=" * 110, flush=True)
    print("VERDICT — classify each policy vs incumbent (both periods) + matched-gross timing test", flush=True)
    print("  PARETO = CAGR>=incumbent AND MaxDD shallower, both periods", flush=True)
    print("  DD-FIRST = CAGR within 1.5pp of incumbent AND MaxDD shallower AND beats matched-gross, both periods", flush=True)
    print("=" * 110, flush=True)
    for name, per in results.items():
        periods = [p for p in per]  # policy may be 8yr-only
        def ok(cond):
            return all(cond(per[p], incumb[p]) for p in periods)
        pareto = ok(lambda a, b: a[0] >= b[0] - 0.001 and a[3] > b[3] + 0.002)
        ddfirst = ok(lambda a, b: a[0] >= b[0] - 0.015 and a[3] > b[3] + 0.002 and a[3] > a[5] + 0.002)
        cls = "PARETO" if pareto else ("DD-FIRST" if ddfirst else "-")
        if cls == "-":
            continue
        deltas = " | ".join(f"{p}: {(per[p][0]-incumb[p][0])*100:+.1f}pp CAGR "
                            f"{(per[p][3]-incumb[p][3])*100:+.1f}pp DD "
                            f"(vs match {(per[p][3]-per[p][5])*100:+.1f}pp)" for p in periods)
        scope = "" if len(periods) == 2 else "  [8yr-ONLY]"
        print(f"  {cls:8s} {name}{scope}: {deltas}", flush=True)
    print("\n(no PARETO/DD-FIRST line = policy failed the both-period + timing bar; that is itself a finding)", flush=True)


if __name__ == "__main__":
    main()
