"""EXP-059 — FRONTIER SEARCH on top of the deployed structure (FINAL @1.49x on the v2 universes).
Switches patched into the clean room (all default OFF = identical to the deployed engine):
  exit_all      : at every tranche day, EVERY book sells names no sleeve wants today (mid-cycle exit, I-29, now cheap: the stride is 5)
  overlay_all   : at every tranche day, ALL books are rescaled to today's vol_scale x gate (prompt overlay; I-03 spirit, cycle 8/13 finding)
  waterfill     : I-02 — cap at 0.10 with the excess redistributed to uncapped names until convergence (instead of clip-then-renormalise)
  min_trade     : no-trade band for resizes as a fraction of book NAV (deployed 0.003)
  slow_vl       : value + lowvol targets refreshed only every N-th rebuild of a book (momentum every rebuild)
  vol_stop      : I-05 — per-name stop = k x annualised vol_60d, clamped [0.25, 0.55] (deployed flat 0.40)
  excl_adds     : I-06 — exclude names that joined the SP1500 within the last N sessions from the momentum sleeve
Run: python3 research/EXP059_frontier.py <8yr|26yr> <stage1|stage2> [arm ...]"""
import os, sys, inspect, textwrap, json, time, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research")
HZ = sys.argv[1]; STAGE = sys.argv[2]; ONLY = sys.argv[3:]; sys.argv = ["x", HZ]
import EXP057_final_on_v2 as E
from main_production_backtest import FastBacktester
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
CACHE = f"research/_v2_{HZ}/exp059"; os.makedirs(CACHE, exist_ok=True)
STARTS = (E.STARTS[0::3] if STAGE == "stage1" else E.STARTS)        # stage1: 8 starts incl. odd months mostly; stage2: all 24
BASE = dict(E.FIN, leverage=1.49)
ARMS = {
    "base":            dict(BASE),
    "exit_all":        dict(BASE, exit_all=True),
    "overlay_all":     dict(BASE, overlay_all=True),
    "exit+overlay":    dict(BASE, exit_all=True, overlay_all=True),
    "waterfill":       dict(BASE, waterfill=True),
    "min_trade_1pct":  dict(BASE, min_trade=0.01),
    "min_trade_2pct":  dict(BASE, min_trade=0.02),
    "slow_vl_2":       dict(BASE, slow_vl=2),
    "vol_stop_1.5":    dict(BASE, vol_stop=1.5),
    "vol_stop_2.0":    dict(BASE, vol_stop=2.0),
    "excl_adds_60":    dict(BASE, excl_adds=60),
    "excl_adds_120":   dict(BASE, excl_adds=120),
    # K re-sweep on the corrected data (EXP-052 chose K=4 on the old universes)
    "K2_s10":          dict(BASE, tranches=2, tranche_stride=10),
    "K5_s4":           dict(BASE, tranches=5, tranche_stride=4),
    "K8_s2":           dict(BASE, tranches=8, tranche_stride=2),
    "K10_s2":          dict(BASE, tranches=10, tranche_stride=2),
    # I-03 ex-ante holdings vol for the overlay
    "exante_vol":      dict(BASE, exante_vol=True),
    "exante_max":      dict(BASE, exante_max=True),
    "exante+overlay":  dict(BASE, exante_max=True, overlay_all=True),
    # I-09 multi-horizon momentum rank ensemble (12-1, 6-1, ~4-1) over the sleeve's top-30 candidates -> top-5
    "mom_ens":         dict(BASE, top_n=30, mom_ens=True),
    # I-10 52-week-high proximity as the ranking over the sleeve's top-30 candidates -> top-5
    "mom_52wh":        dict(BASE, top_n=30, mom_52wh=True),
    # I-14 VIX term structure (VIX > VIX3M = backwardation) -> gross x0.5 on rebuild days (8yr only: data from 2016)
    "vix_gate_0.5":    dict(BASE, vix_gate=0.5),
    # stage-2 combinations of the stage-1 survivors
    "exit+overlay+slow":      dict(BASE, exit_all=True, overlay_all=True, slow_vl=2),
    "exit+overlay+slow+wf":   dict(BASE, exit_all=True, overlay_all=True, slow_vl=2, waterfill=True),
    "overlay+slow":           dict(BASE, overlay_all=True, slow_vl=2),
    "exit+overlay_K5":        dict(BASE, exit_all=True, overlay_all=True, tranches=5, tranche_stride=4),
    # exposure-matched versions: the prompt switches cut exposure in stress, so re-lever until realized vol matches the base
    "exit+overlay_L1.65":     dict(BASE, exit_all=True, overlay_all=True, leverage=1.65),
    "exit+overlay_L1.80":     dict(BASE, exit_all=True, overlay_all=True, leverage=1.80),
    "exit+overlay+slow_L1.65": dict(BASE, exit_all=True, overlay_all=True, slow_vl=2, leverage=1.65),
    # cost-2x versions of the leading candidates (realistic cost level for this account size)
    "exit+overlay+slow_cost2": dict(BASE, exit_all=True, overlay_all=True, slow_vl=2, cost_mult=2.0),
    "base_cost2":              dict(BASE, cost_mult=2.0),
    # DE-RISK-ONLY prompt overlay (never re-levers other books between rebuilds; the deployed overlay is de-risk-only too)
    "overlay_down":            dict(BASE, overlay_down=True),
    "exit+overlay_down":       dict(BASE, exit_all=True, overlay_down=True),
    "overlay_down_L1.65":      dict(BASE, overlay_down=True, leverage=1.65),
    "exit_all_L1.65":          dict(BASE, exit_all=True, leverage=1.65),
    "exit+overlay_down_L1.65": dict(BASE, exit_all=True, overlay_down=True, leverage=1.65),
    "overlay_all_cost2":       dict(BASE, overlay_all=True, cost_mult=2.0),
    "overlay_down_cost2":      dict(BASE, overlay_down=True, cost_mult=2.0),
    "exit_all_cost2":          dict(BASE, exit_all=True, cost_mult=2.0),
    "exit+overlay_down_cost2": dict(BASE, exit_all=True, overlay_down=True, cost_mult=2.0),
    # with the overlay applied promptly, does the gate still need to be fully flat?
    "overlay_down_gate0.25":   dict(BASE, overlay_down=True, credit_derisk=0.25),
    "overlay_down_gate0.50":   dict(BASE, overlay_down=True, credit_derisk=0.50),
    "overlay_all_gate0.25":    dict(BASE, overlay_all=True, credit_derisk=0.25),
    # batch 3 (structural/cost, not signal): drop sub-2% names; equal-weight the momentum picks; wider stop under prompt exits
    "min_weight_2pct":         dict(BASE, min_weight=0.02),
    "mom_equal":               dict(BASE, mom_equal=True),
    "exit_all_stop50":         dict(BASE, exit_all=True, stop=0.50),
    "overlay_down_lb20":       dict(BASE, overlay_down=True, vol_lookback=20),
    "overlay_down_lb60":       dict(BASE, overlay_down=True, vol_lookback=60),
    # batch 4 (stage 2 refinements of the surviving de-risk-only overlay): trigger threshold (hysteresis), combos
    "overlay_down_thr0.90":    dict(BASE, overlay_down=True, overlay_thr=0.90),
    "overlay_down_thr0.85":    dict(BASE, overlay_down=True, overlay_thr=0.85),
    "overlay_down+mom_equal":  dict(BASE, overlay_down=True, mom_equal=True),
    "overlay_down_lb20+mom_equal": dict(BASE, overlay_down=True, vol_lookback=20, mom_equal=True),
    "overlay_down_lb20_cost2": dict(BASE, overlay_down=True, vol_lookback=20, cost_mult=2.0),
    # batch 5: threshold sweep + the 0.85 threshold's cost-2x / exposure-matched / combo versions
    "overlay_down_thr0.80":    dict(BASE, overlay_down=True, overlay_thr=0.80),
    "overlay_down_thr0.75":    dict(BASE, overlay_down=True, overlay_thr=0.75),
    "overlay_down_thr0.85_cost2": dict(BASE, overlay_down=True, overlay_thr=0.85, cost_mult=2.0),
    "overlay_down_thr0.85_L1.65": dict(BASE, overlay_down=True, overlay_thr=0.85, leverage=1.65),
    "overlay_down_thr0.85+mom_equal": dict(BASE, overlay_down=True, overlay_thr=0.85, mom_equal=True),
    "mom_equal_cost2":         dict(BASE, mom_equal=True, cost_mult=2.0),
    "overlay_down+mom_equal_cost2": dict(BASE, overlay_down=True, mom_equal=True, cost_mult=2.0),
    "overlay_down_thr0.90+mom_equal": dict(BASE, overlay_down=True, overlay_thr=0.90, mom_equal=True),
    # batch 6: DAILY de-risk overlay (prompt, every session, not just tranche days); value/lowvol equal weight; momentum count under equal weight
    "overlay_daily":           dict(BASE, overlay_daily=True),
    "overlay_daily+mom_equal": dict(BASE, overlay_daily=True, mom_equal=True),
    "overlay_daily_cost2":     dict(BASE, overlay_daily=True, cost_mult=2.0),
    "vl_equal":                dict(BASE, vl_equal=True),
    "all_equal":               dict(BASE, mom_equal=True, vl_equal=True),
    "mom_equal_n4":            dict(BASE, mom_equal=True, top_n=4),
    "mom_equal_n6":            dict(BASE, mom_equal=True, top_n=6),
    "mom_equal_n7":            dict(BASE, mom_equal=True, top_n=7),
}
def _engine():
    src = inspect.getsource(CleanRoom.run)
    reps = [
        ("dd=float(((v - v.cummax()) / v.cummax()).min()))", "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v)"),
        ("cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0", 'cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0 * float(cfg.get("cost_mult", 1.0))'),
        # per-name stop level (vol_stop) + stop check uses it
        ("        stop = 0.40\n", "        stop = 0.40; vol_stop_k = cfg.get('vol_stop'); stop_of = {}\n"),
        ("                        if p <= peaks[t][s] * (1.0 - stop):\n", "                        if p <= peaks[t][s] * (1.0 - stop_of.get(s, stop)):\n"),
        # remember last sleeve targets per tranche for slow_vl, and expose today's 'wanted' set for exit_all
        ("        navs, last = [], {}\n", "        navs, last = [], {}; vl_cache = {}; vl_count = {}; adds_seen = {}; prev_mem = None\n"),
        ("                mem = self.uni.get_sp500(d)\n                mv = strategy_value(self.uni, d, mem, top_n=10)\n",
         "                mem = self.uni.get_sp500(d)\n                if cfg.get('excl_adds'):\n                    if prev_mem is not None:\n                        for s_ in mem - prev_mem: adds_seen[s_] = i\n                    prev_mem = set(mem)\n                    m1 = {s_: w_ for s_, w_ in (m1 or {}).items() if i - adds_seen.get(s_, -10**9) > int(cfg['excl_adds'])}\n                mv = strategy_value(self.uni, d, mem, top_n=10)\n                if cfg.get('slow_vl'):\n                    vl_count[t] = vl_count.get(t, 0) + 1\n                    if (vl_count[t] - 1) % int(cfg['slow_vl']) == 0 or t not in vl_cache: vl_cache[t] = (mv, m5)\n                    else: mv, m5 = vl_cache[t]\n"),
        # I-09 / I-10: re-rank the momentum sleeve's candidates (sleeve called with top_n=30) and keep 5 equal-weight
        ("                if m1 is None:\n                    m1 = last.get(\"m\", {})\n",
         "                if m1 is None:\n                    m1 = last.get(\"m\", {})\n                if m1 and (cfg.get('mom_ens') or cfg.get('mom_52wh')):\n                    fd0 = self.bt.features_by_date.get(d, {}); sc = {}\n                    if cfg.get('mom_ens'):\n                        import scipy.stats as _ss\n                        cands = [s_ for s_ in m1 if s_ in fd0]\n                        cols_ = [('ret_252d', 'ret_20d'), ('ret_126d', 'ret_20d'), ('ret_60d', 'ret_20d')]\n                        ranks = np.zeros(len(cands))\n                        for a_, b_ in cols_:\n                            v_ = np.array([fd0[s_].get(a_, np.nan) - fd0[s_].get(b_, 0.0) for s_ in cands], dtype=float); v_ = np.where(np.isnan(v_), -9, v_); ranks += _ss.rankdata(v_)\n                        sc = dict(zip(cands, ranks))\n                    else:\n                        for s_ in m1:\n                            h_ = px[s_].loc[:d].dropna().tail(252)\n                            if len(h_) >= 200 and h_.max() > 0: sc[s_] = float(h_.iloc[-1] / h_.max())\n                    top_ = sorted(sc, key=sc.get, reverse=True)[:5]\n                    m1 = {s_: 1.0 / len(top_) for s_ in top_} if top_ else m1\n"),
        # I-14 VIX term structure gate
        ("                dr = dr * vs\n", "                dr = dr * vs\n                if cfg.get('vix_gate') and getattr(self, '_vix', None) is not None:\n                    vv = self._vix.loc[:d]\n                    if len(vv) > 1 and vv.iloc[-2, 0] > vv.iloc[-2, 1]: dr = dr * float(cfg['vix_gate'])\n"),
        # batch 3: equal-weight momentum picks; min-weight filter; stop level; overlay lookback
        ("                comb = {s: v for s, v in comb.items() if v >= 0.005}\n", "                comb = {s: v for s, v in comb.items() if v >= float(cfg.get('min_weight', 0.005))}\n"),
        ("                m5 = strategy5_lowvol_quality(d, self.uni, di)\n", "                if cfg.get('mom_equal') and m1: m1 = {s_: 1.0 / len(m1) for s_ in m1}\n                m5 = strategy5_lowvol_quality(d, self.uni, di)\n"),
        ("        stop = 0.40; vol_stop_k", "        stop = float(cfg.get('stop', 0.40)); vol_stop_k"),
        # batch 6: value/lowvol sleeves equal-weighted
        ("                m5 = strategy5_lowvol_quality(d, self.uni, di)\n",
         "                m5 = strategy5_lowvol_quality(d, self.uni, di)\n                if cfg.get('vl_equal'):\n                    if mv: mv = {s_: 1.0 / len(mv) for s_ in mv}\n                    if m5: m5 = {s_: 1.0 / len(m5) for s_ in m5}\n"),
        # batch 6: DAILY de-risk-only overlay on non-tranche days (same maths as the weekly one: today's vol x gate target per book)
        ("            nav = cash + mtm()\n\n            if i in sched and nav > 0:\n",
         "            nav = cash + mtm()\n            if cfg.get('overlay_daily') and i not in sched and nav > 0 and len(navhist) >= 41:\n                dr_ = derisk_v if (gate_pct and gate.get(d, 0.5) >= gate_pct) else 1.0\n                rr_ = np.diff(np.array(navhist[-41:])) / np.array(navhist[-41:-1]); rv_ = float(np.std(rr_)) * np.sqrt(252)\n                vs_ = min(1.0, max(0.30, VOL_TARGET / rv_)) if (use_ov and rv_ > 0.01) else 1.0\n                tgt_ = (nav / K) * lev * dr_ * vs_\n                for t2 in range(K):\n                    if not books[t2]: continue\n                    gross2 = sum(q2 * prc.get(s2, lastpx.get(s2, 0.0)) for s2, q2 in books[t2].items())\n                    if gross2 <= 0: continue\n                    f2 = tgt_ / gross2\n                    if f2 < float(cfg.get('overlay_thr', 0.95)):\n                        for s2 in list(books[t2]):\n                            p2 = prc.get(s2)\n                            if not p2: continue\n                            q_old = books[t2][s2]; q_new = int(q_old * f2); dq2 = q_new - q_old\n                            if dq2 == 0 or abs(dq2 * p2) < (nav / K) * 0.003: continue\n                            cash -= dq2 * p2; cash -= abs(dq2 * p2) * cost_r\n                            if q_new > 0: books[t2][s2] = q_new\n                            else: books[t2].pop(s2, None); peaks[t2].pop(s2, None)\n                nav = cash + mtm()\n\n            if i in sched and nav > 0:\n"),
        # water-filling cap (I-02)
        ("                comb = {s: min(v, 0.10) for s, v in comb.items() if v > 0}\n                g = sum(comb.values())\n                if g > 1.0:\n                    comb = {s: v / g for s, v in comb.items()}\n",
         "                comb = {s: v for s, v in comb.items() if v > 0}\n                if cfg.get('waterfill'):\n                    g0 = sum(comb.values()); target = min(g0, 1.0); comb = {s: v / g0 * target for s, v in comb.items()} if g0 > 0 else {}\n                    for _ in range(20):\n                        over = {s for s, v in comb.items() if v > 0.10 + 1e-12}\n                        if not over: break\n                        excess = sum(comb[s] - 0.10 for s in over); free = {s: v for s, v in comb.items() if s not in over}; fs = sum(free.values())\n                        for s in over: comb[s] = 0.10\n                        if fs <= 0: break\n                        for s in free: comb[s] += excess * free[s] / fs\n                else:\n                    comb = {s: min(v, 0.10) for s, v in comb.items()}\n                    g = sum(comb.values())\n                    if g > 1.0:\n                        comb = {s: v / g for s, v in comb.items()}\n"),
        # vol-normalised stop levels for today's targets
        ("                dr = derisk_v if (gate_pct and gate.get(d, 0.5) >= gate_pct) else 1.0\n",
         "                dr = derisk_v if (gate_pct and gate.get(d, 0.5) >= gate_pct) else 1.0\n                if vol_stop_k:\n                    fd_ = self.bt.features_by_date.get(d, {})\n                    for s_ in comb:\n                        v60 = fd_.get(s_, {}).get('vol_60d', np.nan)\n                        if v60 == v60 and v60 > 0: stop_of[s_] = min(0.55, max(0.25, vol_stop_k * v60))\n"),
        # min-trade band
        ("                    if abs(dq * p) < tnav * 0.003:\n", "                    if abs(dq * p) < tnav * float(cfg.get('min_trade', 0.003)):\n"),
        # I-03 ex-ante (holdings-based) vol instead of / in addition to 40d realised NAV vol
        ("                vs = 1.0\n                if use_ov and len(navhist) >= 41:\n                    rr = np.diff(np.array(navhist[-41:])) / np.array(navhist[-41:-1])\n                    rv = float(np.std(rr)) * np.sqrt(252)\n                    if rv > 0.01:\n                        vs = min(1.0, max(0.30, VOL_TARGET / rv))\n",
         "                vs = 1.0\n                LB_ = int(cfg.get('vol_lookback', 40))\n                if use_ov and len(navhist) >= LB_ + 1:\n                    rr = np.diff(np.array(navhist[-(LB_+1):])) / np.array(navhist[-(LB_+1):-1])\n                    rv = float(np.std(rr)) * np.sqrt(252)\n                    if cfg.get('exante_vol') or cfg.get('exante_max'):\n                        hold = {}\n                        for b_ in books:\n                            for s_, q_ in b_.items(): hold[s_] = hold.get(s_, 0) + q_\n                        syms_ = [s_ for s_ in hold if s_ in px.columns]\n                        if syms_ and nav > 0:\n                            R_ = px[syms_].loc[:d].tail(61).pct_change().dropna(how='all').fillna(0.0)\n                            w_ = np.array([hold[s_] * prc.get(s_, lastpx.get(s_, 0.0)) / nav for s_ in syms_])\n                            if len(R_) >= 30:\n                                pr_ = R_.values @ w_; ex_ = float(np.std(pr_)) * np.sqrt(252)\n                                rv = ex_ if cfg.get('exante_vol') else max(rv, ex_)\n                    if rv > 0.01:\n                        vs = min(1.0, max(0.30, VOL_TARGET / rv))\n"),
        # exit_all + overlay_all: after this book's rebuild, touch the OTHER books
        ("                nav = cash + mtm()\n\n            prev_debit = max(0.0, -cash)\n",
         "                nav = cash + mtm()\n                wanted = set(comb)\n                if cfg.get('exit_all') or cfg.get('overlay_all') or cfg.get('overlay_down'):\n                    for t2 in range(K):\n                        if t2 == t: continue\n                        if cfg.get('exit_all'):\n                            for s2 in list(books[t2]):\n                                if s2 not in wanted:\n                                    p2 = prc.get(s2, lastpx.get(s2, 0.0)); q2 = books[t2].pop(s2); peaks[t2].pop(s2, None)\n                                    cash += q2 * p2; cash -= abs(q2 * p2) * cost_r\n                        if (cfg.get('overlay_all') or cfg.get('overlay_down')) and books[t2]:\n                            gross2 = sum(q2 * prc.get(s2, lastpx.get(s2, 0.0)) for s2, q2 in books[t2].items()); tgt2 = tnav * lev * dr\n                            f2 = (tgt2 / gross2) if gross2 > 0 else None\n                            if f2 is not None and (tgt2 <= 0 or abs(f2 - 1) > 0.05) and (not cfg.get('overlay_down') or f2 < float(cfg.get('overlay_thr', 0.95))):\n                                for s2 in list(books[t2]):\n                                    p2 = prc.get(s2)\n                                    if not p2: continue\n                                    q_old = books[t2][s2]; q_new = int(q_old * f2); dq2 = q_new - q_old\n                                    if dq2 == 0 or abs(dq2 * p2) < tnav * 0.003: continue\n                                    cash -= dq2 * p2; cash -= abs(dq2 * p2) * cost_r\n                                    if q_new > 0: books[t2][s2] = q_new\n                                    else: books[t2].pop(s2, None); peaks[t2].pop(s2, None)\n                    nav = cash + mtm()\n\n            prev_debit = max(0.0, -cash)\n"),
    ]
    for a, b in reps: assert src.count(a) == 1, a[:80]; src = src.replace(a, b)
    V.END = E.END; ns = dict(V.__dict__); assert ns["END"] == E.END; exec(compile(textwrap.dedent(src), "<exp059>", "exec"), ns); CleanRoom.run = ns["run"]
def st(v): return E.st(v)
def main():
    t0 = time.time(); _engine(); cr = None
    names = [a for a in ARMS if (not ONLY or a in ONLY)]
    for nm in names:
        f = f"{CACHE}/{STAGE}_{nm}.parquet"
        if os.path.exists(f): continue
        if nm == "base" and STAGE == "stage2" and os.path.exists(f"{E.CACHE}/FINAL_1.49.parquet"):
            pd.read_parquet(f"{E.CACHE}/FINAL_1.49.parquet").to_parquet(f); continue
        if cr is None:
            cr = CleanRoom(FastBacktester(universe_path=E.PATH))
            try:
                vx = pd.read_parquet("data/enhanced_data/vix_cache.parquet"); vx.index = pd.to_datetime(vx.index); cr._vix = vx[["^VIX", "^VIX3M"]].dropna().sort_index()
            except Exception as e:
                cr._vix = None; print("  (vix cache unavailable:", e, ")")
        curves = {}
        for s_ in STARTS:
            t1 = time.time(); curves[s_] = cr.run(s_, ARMS[nm])["curve"]; dt = time.time() - t1
            print(f"    {nm} start {s_} {dt:5.0f}s", file=sys.stderr, flush=True)
            if dt > 600: print(f"    !!! {nm} start {s_} took {dt:.0f}s — pathological run", file=sys.stderr, flush=True)
        pd.DataFrame(curves).to_parquet(f); print(f"  {nm:<16} {time.time()-t0:6.0f}s", flush=True)
    # analyse
    B = pd.read_parquet(f"{CACHE}/{STAGE}_base.parquet"); cols = [c for c in B.columns if c in STARTS]; b = np.array([st(B[c]) for c in cols])
    print(f"\n{HZ} {STAGE} ({len(cols)} starts) — FINAL@1.49 base {b[:,0].mean():+.2%} / {b[:,1].mean():.3f} / {b[:,2].mean():.1%}")
    print(f"  {'arm':<16}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'dCAGR':>8}{'dShrp':>8}{'+Shrp':>7}{'dMaxDD':>8}{'+DD':>6}{'yrs>':>6}")
    for nm in names:
        f = f"{CACHE}/{STAGE}_{nm}.parquet"
        if nm == "base" or not os.path.exists(f): continue
        A = pd.read_parquet(f); a = np.array([st(A[c]) for c in cols]); d = a - b
        if np.abs(d).max() < 1e-12: print(f"  !! {nm}: IDENTICAL to base on every start — the switch never fired (harness bug); cache removed"); os.remove(f); continue
        yl = [E.yearly(B[c]) for c in cols]; ya = [E.yearly(A[c]) for c in cols]; ys = sorted(set().union(*[set(x) for x in yl])); wins = sum(1 for y in ys if np.mean([xa[y] - xb[y] for xa, xb in zip(ya, yl) if y in xa and y in xb]) > 0)
        print(f"  {nm:<16}{a[:,0].mean():>+9.2%}{a[:,1].mean():>8.3f}{a[:,2].mean():>8.1%}{d[:,0].mean()*100:>+7.2f}p{d[:,1].mean():>+8.3f}{int((d[:,1]>0).sum()):>4}/{len(cols)}{d[:,2].mean()*100:>+7.2f}p{int((d[:,2]>0).sum()):>3}/{len(cols)}{wins:>4}/{len(ys)}")
    print(f"  total {time.time()-t0:.0f}s")
if __name__ == "__main__": main()
