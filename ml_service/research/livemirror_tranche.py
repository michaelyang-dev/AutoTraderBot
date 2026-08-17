"""JOINT-TRANCHE BACKTESTER — what the LIVE engine will actually do with Option 2.

WHY THIS EXISTS
  Every tranching number in this program came from running K sub-books as SEPARATE backtests at
  capital/K and summing the curves. Those sub-books never share capital: a lucky phase stays big
  for 26 years. The live engine cannot work that way -- all K sub-books sit in ONE margin
  account, so a tranche rebalancing necessarily sizes off NAV/K of the CURRENT TOTAL NAV. That
  is continuous reallocation between tranches.

  Cross-phase 26yr CAGR sigma is 1.94pp, which compounds to ~1.6x between best and worst
  tranche, so the two designs are genuinely different products. This simulates the LIVE one.

WHAT IS MODELLED
  - K virtual sub-books, each with its OWN share ledger and its OWN per-name trailing-stop peak
    (a tranche that entered later has a different peak, so stops can fire at different times --
    the summed-independent version has this property and it must be preserved)
  - ONE shared cash balance and ONE margin debit, financed daily (a real account has one)
  - every `rebal_days / K` sessions exactly ONE tranche rebalances, cycling
  - the rebalancing tranche sizes off (shared NAV) / K, using the SAME closed-loop integer-share
    calibration the live engine runs (`live_sizing`), including the position cap
  - credit gate and vol-scaling apply exactly as in the single-book harness

PARITY GATES
  1. `tranches` unset or 1  -> delegates to the parent's validated run(), so there is ZERO
     regression risk to any previously published number.
  2. `tranches=1, force_joint=True, tranche_stride=rebal_days` runs the JOINT loop as a single
     book on the deployed cadence, so it must reproduce the parent's single-book result. This is
     the gate that actually validates this reimplementation; gate 1 cannot, because delegation
     means it never enters this code path.
     (An earlier version of gate 2 used K=4 with stride=rebal_days and expected a split single
     book. That was wrong: with stride=rebal_days and K=4 each tranche fires every K*stride=80
     sessions, not 20, so the book is 4x stale and the comparison is meaningless. Recorded
     because the failing number looked like a machinery bug and was a bad test.)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livemirror_backtest import LiveMirrorBacktester            # noqa: E402
from main_production_backtest import COST_BPS, SLIPPAGE_BPS, _short_weights  # noqa: E402
from strategies.multi_strategy_engine import (                  # noqa: E402
    strategy1_momentum_reversal, strategy3_sector_rotation, strategy5_lowvol_quality,
    strategy_value, PROD_WEIGHTS_BEAR, PROD_WEIGHTS_CRASH, UMD_CRASH_THRESHOLD,
)


class JointTrancheBacktester(LiveMirrorBacktester):

    def run(self, start="2018-01-01", end="2025-12-31", config=None):
        config = dict(config or {})
        K = int(config.get("tranches", 1) or 1)
        # `force_joint` runs K=1 THROUGH the joint loop instead of delegating. That is the only
        # way to test this reimplementation against the validated single-book harness: with K=1
        # and stride=rebal_days the joint loop is, by construction, one book on the deployed
        # cadence, so it must reproduce the parent. Without this the new code path is never
        # validated at all -- delegation would silently hide any bug in it.
        if K <= 1 and not config.get("force_joint"):
            return super().run(start, end, config)      # parity gate 1: no regression

        universe = config.get("universe", "sp1500")
        self.uni.get_sp500 = self._get_sp1500 if universe == "sp1500" else self._original_get_sp500
        dates = [d for d in sorted(self.prices.index)
                 if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        if not dates:
            return None

        cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000 * float(config.get("cost_mult", 1.0))
        mom_w = config.get("mom_w", 0.60); val_w = config.get("val_w", 0.15)
        lv_w = config.get("lv_w", 0.15);   sec_w = config.get("sec_w", 0.10)
        top_n = config.get("top_n", 8);    cap = config.get("cap", 0.15)
        rebal_days = int(config.get("rebal_days", 20))
        stride = int(config.get("tranche_stride", max(1, rebal_days // K)))
        trailing_stop = config.get("trailing_stop")
        capital = float(config.get("initial_capital", 50_000.0))
        leverage = float(config.get("leverage", 1.0))
        fin_rate = float(config.get("financing_rate", 0.0))
        pos_cap = float(config.get("live_position_cap", 0.15))
        lev_ceiling = float(config.get("live_lev_ceiling", 1.80))
        vol_scaling = bool(config.get("vol_scaling", False))
        vol_target = config.get("vol_target", 0.20) ; vol_lb = int(config.get("vol_lookback", 40))
        credit_pct = config.get("credit_pct"); credit_derisk = float(config.get("credit_derisk", 0.5))
        crmap = self._credit_pctile_map(dates) if credit_pct else {}

        _fin = None
        if config.get("financing_curve"):
            fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fin_rate.parquet")
            fs = pd.read_parquet(fp)["fin_rate"].dropna()
            idx = pd.DatetimeIndex(dates)
            _fin = fs.reindex(idx.union(fs.index)).sort_index().ffill().reindex(idx).ffill().bfill()

        # K independent ledgers: shares and per-name trailing peaks
        books = [{} for _ in range(K)]          # tranche -> {sym: {"shares","entry_px","peak_px"}}
        cash = capital
        port, last_tgt = [], {}
        recent, shadow_nav, shadow_gross = [], capital, 0.0
        prev_sh, prev_px, prev_gross = {}, {}, 0.0
        self._fin_paid = 0.0
        self._gross_path = []

        def mark(today, b):
            return sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in b.items())

        for di, date in enumerate(dates):
            today = {}
            if date in self.prices.index:
                row = self.prices.loc[date]
                for b in books:
                    for s, h in b.items():
                        v = row.get(s)
                        if v is not None and not np.isnan(v):
                            today[s] = v
                for s in row.dropna().index:
                    today[s] = row[s]

            # unlevered shadow book -> vol signal (identical convention to the single-book run)
            if prev_gross > 0:
                pdr = sum(prev_sh[s] * (today.get(s, prev_px[s]) - prev_px[s])
                          for s in prev_sh) / prev_gross
                pnl = shadow_gross * pdr
                recent.append(pnl / shadow_nav if shadow_nav > 0 else 0.0)
                if len(recent) > vol_lb:
                    recent.pop(0)
                shadow_nav += pnl
                shadow_gross *= (1 + pdr)

            # trailing stops — PER TRANCHE, each with its own peak (entry dates differ)
            if trailing_stop:
                for b in books:
                    for s in list(b):
                        px = today.get(s)
                        if not px:
                            continue
                        h = b[s]
                        h.setdefault("peak_px", px)
                        if px > h["peak_px"]:
                            h["peak_px"] = px
                        if (px - h["peak_px"]) / h["peak_px"] < -abs(trailing_stop):
                            cash += h["shares"] * px * (1 - cost_frac)
                            del b[s]

            fr = float(_fin.get(date, fin_rate)) if _fin is not None else fin_rate
            if fr and cash < 0:
                chg = (-cash) * (fr / 252.0)
                cash -= chg; self._fin_paid += chg

            nav = cash + sum(mark(today, b) for b in books)

            def snapshot():
                prev_sh.clear(); prev_px.clear(); g = 0.0
                for b in books:
                    for s, h in b.items():
                        px = today.get(s, h["entry_px"])
                        prev_sh[s] = prev_sh.get(s, 0) + h["shares"]
                        prev_px[s] = px
                        g += h["shares"] * px
                return g

            # ---- mid-cycle SIGNAL EXIT (EXP-026) -------------------------------------
            # The only mechanism in this program whose drawdown gain SURVIVED the
            # matched-exposure control (+5.6..+7.8pp) rather than being a level effect. It
            # costs Sharpe on its own (-0.016..-0.037 matched), so it is tested here only in
            # combination -- the question is whether tranching's variance budget pays for it.
            # ⚠️ BUG CAUGHT 2026-08-16 by the suspicious-roundness heuristic (BUGS E): the first
            # version gated on `di % sig_exit == 0 AND di % stride != 0`. With sig_exit=5 and
            # stride=5 those are the SAME days, so the exit could NEVER fire -- and the arms
            # returned numbers IDENTICAL to the no-exit reference (+14.46% twice). Same failure
            # as `park idle cash IEF` in EXP-021: an arm that tested nothing.
            # The exit only has anything to do on days when NO tranche is rebalancing, so its
            # cadence must be FINER than the stride. Now it fires on any non-rebalance day that
            # matches the cadence, and a cadence coarser than the stride is rejected loudly.
            sig_exit = config.get("signal_exit_every")
            if sig_exit and sig_exit >= stride:
                raise ValueError(
                    f"signal_exit_every={sig_exit} >= tranche_stride={stride}: every exit day "
                    f"is already a rebalance day, so the exit can never fire. Use a finer "
                    f"cadence (e.g. 1-{stride-1}).")
            if sig_exit and di > 0 and di % sig_exit == 0 and di % stride != 0:
                m1 = strategy1_momentum_reversal(date, self.uni, 0, top_n=top_n,
                                                 rebal_days=rebal_days) or {}
                mem_ = self.uni.get_sp500(date)
                wanted = (set(m1) | set(strategy_value(self.uni, date, mem_, top_n=10) or {})
                          | set(strategy5_lowvol_quality(date, self.uni, 0) or {}))
                for b_ in books:
                    for s_ in list(b_):
                        if s_ not in wanted:
                            px_ = today.get(s_, b_[s_]["entry_px"])
                            cash += b_[s_]["shares"] * px_ * (1 - cost_frac)
                            del b_[s_]
                nav = cash + sum(mark(today, bb) for bb in books)

            # exactly ONE tranche rebalances every `stride` sessions, cycling
            if di % stride != 0:
                prev_gross = snapshot()
                port.append((date, nav))
                continue
            t = (di // stride) % K

            # ---- signals (production code, identical call pattern to the single-book run) ----
            didx = di - (t * stride)
            t1 = strategy1_momentum_reversal(date, self.uni, didx, top_n=top_n,
                                             rebal_days=rebal_days)
            if t1 is None:
                t1 = last_tgt.get("mom", {})
            t5 = strategy5_lowvol_quality(date, self.uni, didx)
            members = self.uni.get_sp500(date)
            tv = strategy_value(self.uni, date, members, top_n=10)
            t3 = strategy3_sector_rotation(date, self.uni, didx)
            if t3 is None:
                t3 = last_tgt.get("s3", {})
            if t5 is None:
                t5 = last_tgt.get("s5", {})
            last_tgt.update({"mom": t1, "val": tv, "s3": t3, "s5": t5})

            nu = self.umd_20d.loc[:date]
            crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < UMD_CRASH_THRESHOLD
            ew = _short_weights(PROD_WEIGHTS_CRASH) if crash else \
                {"mom": mom_w, "val": val_w, "s5": lv_w, "s3": sec_w}
            fdate = self.features_by_date.get(date, {})
            above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
            tot_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
            breadth = above / max(tot_f, 1)
            blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))
            bear = config.get("bear_weights", _short_weights(PROD_WEIGHTS_BEAR))
            blended = {n: ew[n] * blend + bear.get(n, 0) * (1 - blend) for n in ew}

            combined = {}
            for nm, w_ in blended.items():
                for sym, w in last_tgt.get(nm, {}).items():
                    if w > 0:
                        combined[sym] = combined.get(sym, 0) + w * w_

            vs = 1.0
            if vol_scaling and len(recent) >= 20:
                rv = float(np.std(recent)) * np.sqrt(252)
                if rv > 0.01:
                    vs = min(config.get("vol_scale_cap", 1.5), max(0.30, vol_target / rv))
                    combined = {s: w * vs for s, w in combined.items()}

            longs = {s: min(w, cap) for s, w in combined.items() if w > 0}
            g = sum(longs.values())
            if g > 1.0:
                longs = {s: w / g for s, w in longs.items()}
            combined = {s: w for s, w in longs.items() if w >= 0.005}

            derisk = credit_derisk if (credit_pct and crmap.get(date, 0.5) >= credit_pct) else 1.0
            lev_t = leverage * derisk

            # THE POINT OF THIS FILE: the tranche sizes off NAV/K of the CURRENT SHARED NAV
            tnav = nav / K
            tot_w = sum(combined.values())
            base = ({s: w / tot_w for s, w in combined.items()} if tot_w > 0 else dict(combined))
            m = lev_t * vs
            tgt_gross = tnav * lev_t * vs
            target = {}
            for _ in range(4):
                target, gg = {}, 0.0
                for s, w0 in base.items():
                    px = today.get(s)
                    if not px or px <= 0:
                        continue
                    q = float(np.floor(tnav * min(w0 * m, pos_cap) / px))
                    if q > 0:
                        target[s] = q; gg += q * px
                if gg <= 0:
                    break
                m = min(m * (tgt_gross / gg), lev_ceiling)

            b = books[t]
            for s in list(b):                                   # exit names this tranche drops
                if s not in target:
                    px = today.get(s, b[s]["entry_px"])
                    cash += b[s]["shares"] * px * (1 - cost_frac)
                    del b[s]
            for s, q in target.items():                          # resize to integer target
                px = today.get(s)
                if not px or px <= 0:
                    continue
                cur = b[s]["shares"] if s in b else 0.0
                d = q - cur
                if abs(d * px) < max(tnav * 0.003, 1e-9):
                    continue
                cost = abs(d * px) * cost_frac
                cash -= d * px + cost
                if s in b:
                    b[s]["shares"] = q
                    if b[s]["shares"] < 1:
                        del b[s]
                else:
                    b[s] = {"shares": q, "entry_px": px, "peak_px": px}

            nav = cash + sum(mark(today, bb) for bb in books)
            shadow_gross = sum(combined.values()) * shadow_nav
            prev_gross = snapshot()
            self._gross_path.append((date, (nav - cash) / nav if nav > 0 else 0))
            port.append((date, nav))

        vals = pd.Series([v for _, v in port], index=pd.DatetimeIndex([d for d, _ in port]))
        dr = vals.pct_change().dropna()
        yrs = max((vals.index[-1] - vals.index[0]).days / 365.25, 1)
        gp = pd.Series([g for _, g in self._gross_path])
        return {"cagr": (vals.iloc[-1] / vals.iloc[0]) ** (1 / yrs) - 1,
                "sharpe": dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0,
                "max_dd": ((vals - vals.cummax()) / vals.cummax()).min(),
                "daily_values": vals, "fin_paid": self._fin_paid,
                "avg_gross": float(gp.mean()) if len(gp) else 0.0}
