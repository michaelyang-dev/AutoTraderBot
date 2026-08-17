"""VERIFY2 — CLEAN-ROOM INDEPENDENT SIMULATOR. Written from the spec, not copied.

WHY
  Every number in this program came from ONE portfolio simulator that I wrote
  (livemirror_backtest.py, extended by livemirror_tranche.py). If that code has a systematic
  error -- a mis-ordered cost, a stale mark, a double-counted financing charge -- every result
  inherits it, and no amount of start-consistency, cost-sensitivity or year-by-year testing
  would reveal it. Those tests all run THROUGH the same engine.

  The only way to detect a harness-level error is a SECOND implementation that does not share
  code with the first, and then to check whether the two agree.

SCOPE, chosen deliberately
  The production SLEEVES (strategy1_momentum_reversal, strategy_value, strategy5_lowvol_quality)
  are NOT reimplemented. They are the same functions the LIVE signal server calls, so rewriting
  them would test my copy of the strategy rather than the strategy. What IS reimplemented, from
  scratch and with different internal structure, is everything I wrote:
      position sizing / closed-loop calibration   integer-share truncation
      transaction costs                           margin debit + daily financing
      trailing stops                              the credit gate
      the K-tranche ledger and rebalance cycle    NAV accounting and metrics

DELIBERATE IMPLEMENTATION DIFFERENCES (so a shared bug cannot survive in both)
  1. NAV is rebuilt from scratch each day as cash + Sum(shares x price), never carried forward
     incrementally. The original accumulates. A drift bug shows as disagreement.
  2. Costs are charged as an explicit separate cash deduction AFTER the share change, rather
     than folded into the same expression.
  3. Financing is charged on the PREVIOUS day's closing debit (accrual convention), not the
     current day's -- a genuinely different, slightly more conservative convention.
  4. Stops are evaluated against a peak dict rebuilt from a rolling max of actual holdings.
  5. Tranche selection uses an explicit schedule list built up-front, not a modulo on the loop
     index.

  Because of (3) especially, the two will NOT agree to the penny. The question is whether they
  agree to within a small tolerance and, far more importantly, whether the DELTA vs LIVE is the
  same. A shared conclusion under two independent accountings is strong evidence; an identical
  number would actually be suspicious given (3).

PLUS: UNTOUCHED HOLDOUT
  Every experiment in this program used ODD-month starts (Jan/Mar/May/Jul/Sep/Nov). This runs
  EVEN months (Feb/Apr/Jun/Aug/Oct/Dec) -- a sample of entry dates never examined once. If the
  edge is a product of the start dates I happened to pick, it will not survive here.

Run:  python3 research/VERIFY2_cleanroom.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester, COST_BPS, SLIPPAGE_BPS, _short_weights  # noqa: E402
from strategies.multi_strategy_engine import (  # noqa: E402
    strategy1_momentum_reversal, strategy3_sector_rotation, strategy5_lowvol_quality,
    strategy_value, PROD_WEIGHTS_BEAR, PROD_WEIGHTS_CRASH, UMD_CRASH_THRESHOLD)

HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
PATH = ("data/wrds/complete_sp1500_universe.pkl" if HZ == "8yr"
        else "data/wrds/sp1500_universe_2000.pkl")
YRS = [2018, 2019] if HZ == "8yr" else [2001, 2002]
END = pd.Timestamp("2025-12-31")
# UNTOUCHED: even months. Every prior experiment used odd months.
HOLDOUT = [f"{y}-{m:02d}-03" for y in YRS for m in (2, 4, 6, 8, 10, 12)]
USED = [f"{y}-{m:02d}-03" for y in YRS for m in (1, 3, 5, 7, 9, 11)]


class CleanRoom:
    """Independent simulator. Shares only the production sleeve functions and the price data."""

    def __init__(self, bt):
        self.bt = bt
        self.px = bt.prices
        self.fin = pd.read_parquet("research/_fin_rate.parquet")["fin_rate"].dropna()
        cr = pd.read_parquet("research/_credit_signal.parquet")["hy_oas"].dropna()
        self.cr = cr

    def _gate(self, dates):
        s = self.cr.reindex(pd.DatetimeIndex(dates).union(self.cr.index)).sort_index()
        s = s.ffill().reindex(pd.DatetimeIndex(dates))
        pr = s.expanding(min_periods=252).apply(lambda a: (a[-1] >= a).mean(), raw=True).shift(1)
        return {d: (float(v) if pd.notna(v) else 0.5) for d, v in zip(dates, pr.values)}

    def run(self, start, cfg):
        px = self.px
        dates = [d for d in sorted(px.index) if pd.Timestamp(start) <= d <= END]
        K = int(cfg.get("tranches", 1))
        stride = int(cfg.get("tranche_stride", 20 // max(K, 1)))
        lev = float(cfg["leverage"])
        derisk_v = float(cfg.get("credit_derisk", 0.0))
        gate_pct = cfg.get("credit_pct")
        cap_pos = 0.15
        cost_r = (COST_BPS + SLIPPAGE_BPS) / 10000.0
        stop = 0.40
        gate = self._gate(dates) if gate_pct else {}
        fr = self.fin.reindex(pd.DatetimeIndex(dates).union(self.fin.index)).sort_index()
        fr = fr.ffill().reindex(pd.DatetimeIndex(dates)).ffill().bfill()

        # DIFFERENCE 5: explicit up-front schedule, no modulo on the loop counter
        sched = {i: (i // stride) % K for i in range(len(dates)) if i % stride == 0}

        # VOL OVERLAY -- implemented here to the LIVE engine's own definition
        # (ibkr_engine.compute_vol_scale): vol_scale = clamp(VOL_TARGET / realised NAV vol,
        # 0.30, 1.0) with VOL_TARGET = 0.15 * 1.49 over a 40-day window of ACCOUNT NAV returns.
        # This is deliberately a DIFFERENT construction from the original harness, which derives
        # the signal from a synthetic unlevered "shadow" book. If the two agree on the overlay's
        # cost, that cost is not an artefact of either construction.
        use_ov = bool(cfg.get("vol_overlay"))
        VOL_TARGET = 0.15 * 1.49
        navhist = []
        books = [dict() for _ in range(K)]     # tranche -> {sym: shares}
        peaks = [dict() for _ in range(K)]     # tranche -> {sym: peak price}
        cash = float(cfg.get("initial_capital", 50_000.0))
        lastpx = {}                            # last observed price, for marking stale holdings
        prev_debit = 0.0                       # DIFFERENCE 3: accrue on yesterday's debit
        navs, last = [], {}
        self.uni = self.bt.uni
        self.uni.get_sp500 = self.bt._get_sp1500

        for i, d in enumerate(dates):
            row = px.loc[d] if d in px.index else None
            prc = {} if row is None else {s: v for s, v in row.dropna().items()}
            # ⚠️ BUG IN THIS VERIFIER, found 2026-08-17 and fixed. The first version defaulted a
            # missing price to ZERO (`prc.get(s, 0.0)`), so any held name without a print that
            # day was marked to nil and then sold for nil at the next rebalance. 577 of 2,740
            # symbols lack a print on a typical day, so the book was being destroyed: the
            # reference arm returned +1.36% CAGR / -70.3% MaxDD against the original engine's
            # +23.88%. That 22pp gap was MY new code, not the engine under test.
            # Fixed by carrying the last observed price, which is also what the original does
            # (it falls back to entry_px). Recorded because an independent verifier that is
            # itself broken is worse than no verifier -- it would have "refuted" a good result.
            for _s, _p in prc.items():
                lastpx[_s] = _p

            # DIFFERENCE 3: financing on YESTERDAY's debit, charged before anything else
            if prev_debit > 0:
                cash -= prev_debit * float(fr.get(d, 0.063)) / 252.0

            # DIFFERENCE 4: stops from an explicit peak dict per tranche
            if stop:
                for t in range(K):
                    for s in list(books[t]):
                        p = prc.get(s)
                        if p is None:
                            continue
                        peaks[t][s] = max(peaks[t].get(s, p), p)
                        if p <= peaks[t][s] * (1.0 - stop):
                            q = books[t].pop(s); peaks[t].pop(s, None)
                            gross_ = q * p
                            cash += gross_
                            cash -= gross_ * cost_r          # DIFFERENCE 2: separate deduction
            # DIFFERENCE 1: NAV rebuilt from scratch, never carried
            def mtm():
                tot = 0.0
                for t in range(K):
                    for s, q in books[t].items():
                        tot += q * prc.get(s, lastpx.get(s, 0.0))
                return tot
            nav = cash + mtm()

            if i in sched and nav > 0:
                t = sched[i]
                di = i - t * stride
                m1 = strategy1_momentum_reversal(d, self.uni, di, top_n=cfg.get("top_n", 5),
                                                 rebal_days=20)
                if m1 is None:
                    m1 = last.get("m", {})
                m5 = strategy5_lowvol_quality(d, self.uni, di)
                mem = self.uni.get_sp500(d)
                mv = strategy_value(self.uni, d, mem, top_n=10)
                m3 = strategy3_sector_rotation(d, self.uni, di)
                if m5 is None:
                    m5 = last.get("l", {})
                if m3 is None:
                    m3 = last.get("s", {})
                last = {"m": m1, "l": m5, "s": m3}

                nu = self.bt.umd_20d.loc[:d]
                crash = len(nu) and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < UMD_CRASH_THRESHOLD
                w = (_short_weights(PROD_WEIGHTS_CRASH) if crash
                     else {"mom": cfg.get("mom_w", .5), "val": cfg.get("val_w", .35),
                           "s5": cfg.get("lv_w", .15), "s3": 0.0})
                fd = self.bt.features_by_date.get(d, {})
                ab = sum(1 for x in fd.values() if x.get("dist_sma50", 0) > 0)
                tf = sum(1 for x in fd.values() if "dist_sma50" in x)
                bl = min(1.0, max(0.0, (ab / max(tf, 1) - 0.35) / 0.25))
                bw = _short_weights(PROD_WEIGHTS_BEAR)
                w = {k: w[k] * bl + bw.get(k, 0) * (1 - bl) for k in w}

                comb = {}
                for key, src in (("mom", m1), ("val", mv), ("s5", m5), ("s3", m3)):
                    for s, ww in (src or {}).items():
                        if ww > 0:
                            comb[s] = comb.get(s, 0.0) + ww * w[key]
                comb = {s: min(v, 0.10) for s, v in comb.items() if v > 0}
                g = sum(comb.values())
                if g > 1.0:
                    comb = {s: v / g for s, v in comb.items()}
                comb = {s: v for s, v in comb.items() if v >= 0.005}

                dr = derisk_v if (gate_pct and gate.get(d, 0.5) >= gate_pct) else 1.0
                vs = 1.0
                if use_ov and len(navhist) >= 41:
                    rr = np.diff(np.array(navhist[-41:])) / np.array(navhist[-41:-1])
                    rv = float(np.std(rr)) * np.sqrt(252)
                    if rv > 0.01:
                        vs = min(1.0, max(0.30, VOL_TARGET / rv))
                dr = dr * vs
                tnav = nav / K
                tot = sum(comb.values())
                base = {s: v / tot for s, v in comb.items()} if tot > 0 else {}
                mult = lev * dr
                tgt = {}
                for _ in range(4):
                    tgt, gg = {}, 0.0
                    for s, w0 in base.items():
                        p = prc.get(s)
                        if not p or p <= 0:
                            continue
                        q = int(tnav * min(w0 * mult, cap_pos) / p)   # integer shares
                        if q > 0:
                            tgt[s] = q; gg += q * p
                    if gg <= 0:
                        break
                    mult = min(mult * (tnav * lev * dr / gg), 1.80)

                for s in list(books[t]):
                    if s not in tgt:
                        p = prc.get(s, lastpx.get(s, 0.0))
                        q = books[t].pop(s); peaks[t].pop(s, None)
                        cash += q * p
                        cash -= abs(q * p) * cost_r
                for s, q in tgt.items():
                    p = prc.get(s)
                    if not p:
                        continue
                    cur = books[t].get(s, 0)
                    dq = q - cur
                    if abs(dq * p) < tnav * 0.003:
                        continue
                    cash -= dq * p
                    cash -= abs(dq * p) * cost_r
                    if q > 0:
                        books[t][s] = q; peaks[t].setdefault(s, p)
                    else:
                        books[t].pop(s, None); peaks[t].pop(s, None)
                nav = cash + mtm()

            prev_debit = max(0.0, -cash)
            navhist.append(max(nav, 1e-9))
            navs.append((d, max(nav, 1e-9)))

        v = pd.Series([x for _, x in navs], index=pd.DatetimeIndex([x for x, _ in navs]))
        r = v.pct_change().dropna()
        yrs = max((v.index[-1] - v.index[0]).days / 365.25, 1)
        return dict(cagr=(v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1,
                    sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0,
                    dd=float(((v - v.cummax()) / v.cummax()).min()))


def main():
    t0 = time.time()
    bt = FastBacktester(universe_path=PATH)
    cr = CleanRoom(bt)
    LIVE = dict(leverage=1.49, tranches=1, tranche_stride=20, credit_pct=0.95,
                credit_derisk=0.50, mom_w=.50, val_w=.35, lv_w=.15, initial_capital=50_000.0,
                vol_overlay=True)          # <- the TRUE deployed book, overlay included
    WIN = dict(leverage=1.25, tranches=4, tranche_stride=5, credit_pct=0.95,
               credit_derisk=0.00, mom_w=.70, val_w=.21, lv_w=.09, initial_capital=50_000.0)
    ARMS = [("LIVE @1.49+ovl", LIVE),
            ("WIN @1.00", {**WIN, "leverage": 1.00}),
            ("WIN @1.10", {**WIN, "leverage": 1.10}),
            ("WIN @1.25", WIN),
            ("WIN @1.49", {**WIN, "leverage": 1.49}),
            ("noOvl only @1.49", {**LIVE, "vol_overlay": False})]
    for label, sts in (("UNTOUCHED HOLDOUT (even months)", HOLDOUT),
                       ("previously-used (odd months)", USED)):
        print(f"\n  === {HZ} — {label} ===", flush=True)
        print(f"  {'arm':<18}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>9}{'dCAGR':>9}{'dShrp':>9}"
              f"{'dMaxDD':>9}{'+Shrp':>7}", flush=True)
        out = {}
        for nm, cfg in ARMS:
            rs = [cr.run(s, cfg) for s in sts]
            c = np.array([x["cagr"] for x in rs]); sh = np.array([x["sharpe"] for x in rs])
            dd = np.array([x["dd"] for x in rs])
            out[nm] = (c, sh, dd)
            a = out.get("LIVE @1.49+ovl")
            if nm == "LIVE @1.49+ovl":
                print(f"  {nm:<18}{c.mean():>+10.2%}{sh.mean():>9.3f}{dd.mean():>9.1%}",
                      flush=True)
            else:
                print(f"  {nm:<18}{c.mean():>+10.2%}{sh.mean():>9.3f}{dd.mean():>9.1%}"
                      f"{(c-a[0]).mean()*100:>+8.2f}p{(sh-a[1]).mean():>+9.3f}"
                      f"{(dd-a[2]).mean()*100:>+8.2f}p{int((sh>a[1]).sum()):>4}/{len(sts)}",
                      flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
