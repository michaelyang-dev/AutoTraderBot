"""
LIVE-MIRROR BACKTESTER — everything the live IBKR book actually does, in one run().

Mirrors live reality that the standard FastBacktester approximates via a return-overlay:
  - start capital $50,000 (config initial_capital)
  - gross notional = LEVERAGE * NAV (1.00x or 1.49x), applied to the dollar targets
  - INTEGER shares, ROUNDED DOWN (floor) — no fractional (IBKR API constraint)
  - real margin: cash goes negative (debit); FINANCING charged daily on the debit at 6.3%/yr
  - CREDIT GATE wired INSIDE the loop: causal expanding-percentile HY-OAS; when in the top
    (1-pct) tail, scale gross leverage by derisk (e.g. 0.5). Acts next day (shift 1).
  - vol-scaling / 40% stops / 20d rebal / bear-weights / UMD crash — all the live stack, run
    JOINTLY (not stacked overlays) so interactions are real.

Only the SIZING/execution/financing/gate is changed; ALL strategy signal code is inherited
verbatim from FastBacktester.run (copied below, edits marked  # LIVEMIRROR).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester, INITIAL_CASH, COST_BPS, SLIPPAGE_BPS  # noqa: E402
from strategies.multi_strategy_engine import (  # noqa: E402
    strategy1_momentum_reversal, strategy3_sector_rotation, strategy5_lowvol_quality,
    strategy_value, PROD_WEIGHTS_BEAR, PROD_WEIGHTS_CRASH, UMD_CRASH_THRESHOLD,
)
from main_production_backtest import _short_weights  # noqa: E402


class LiveMirrorBacktester(FastBacktester):

    def _sector_map(self):
        """ticker -> 2-digit SIC (industry group), most-recent record per ticker. Cached."""
        if getattr(self, "_secmap", None) is not None:
            return self._secmap
        si = pd.read_parquet("data/wrds/crsp_security_info.parquet", columns=["Ticker", "SICCD"])
        si = si.dropna(subset=["Ticker", "SICCD"])
        si["sic2"] = (pd.to_numeric(si["SICCD"], errors="coerce") // 100)
        si = si.dropna(subset=["sic2"])
        self._secmap = {t: int(s) for t, s in zip(si["Ticker"], si["sic2"])}
        return self._secmap

    def _credit_pctile_map(self, trading_dates, col="hy_oas"):
        """date -> causal expanding percentile of credit spread (acts t+1). Used for BOTH
        the down-gate (pctile>=0.95 -> de-risk) and the bull-calm regime (pctile<=calm)."""
        cr = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "research", "_credit_signal.parquet"))[col].dropna()
        idx = pd.DatetimeIndex(trading_dates)
        s = cr.reindex(idx.union(cr.index)).sort_index().ffill().reindex(idx)
        pr = s.expanding(min_periods=252).apply(lambda a: (a[-1] >= a).mean(), raw=True).shift(1)
        return {d: (float(v) if pd.notna(v) else 0.5) for d, v in zip(idx, pr.values)}

    def _stress_pctile_map(self, trading_dates, col, win=None):
        """date -> causal expanding percentile of a _macro_stress column (hy_oas/rates_vol/
        curve), acts t+1. `win` re-derives rates_vol at a custom window from DGS10 if given."""
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")
        ms = pd.read_parquet(os.path.join(base, "_macro_stress.parquet"))
        s = ms[col].dropna()
        idx = pd.DatetimeIndex(trading_dates)
        s = s.reindex(idx.union(s.index)).sort_index().ffill().reindex(idx)
        pr = s.expanding(min_periods=252).apply(lambda a: (a[-1] >= a).mean(), raw=True).shift(1)
        return {d: (float(v) if pd.notna(v) else 0.5) for d, v in zip(idx, pr.values)}

    def run(self, start="2018-01-01", end="2025-12-31", config=None):
        if config is None:
            config = {}
        universe = config.get("universe", "sp1500")
        self.uni.get_sp500 = self._get_sp1500 if universe == "sp1500" else self._original_get_sp500
        trading_dates = [d for d in sorted(self.prices.index)
                         if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        if not trading_dates:
            return None

        # ---- LIVEMIRROR params ----
        capital = float(config.get("initial_capital", INITIAL_CASH))
        leverage = float(config.get("leverage", 1.0))
        integer_shares = bool(config.get("integer_shares", False))
        fin_rate = float(config.get("financing_rate", 0.0))
        credit_pct = config.get("credit_pct", None)      # down-gate percentile (e.g. 0.95)
        credit_derisk = config.get("credit_derisk", 0.5)
        crmap = self._credit_pctile_map(trading_dates, config.get("credit_col", "hy_oas")) \
            if (credit_pct or config.get("bull_weights") or config.get("bull_lever")) else {}
        # composite gate: OR over macro-stress columns (e.g. ["hy_oas","rates_vol"])
        gate_cols = config.get("gate_cols")
        gate_pct = config.get("gate_pct", 0.95)
        gate_derisk = config.get("gate_derisk", 0.5)
        gate_maps = {c: self._stress_pctile_map(trading_dates, c) for c in gate_cols} if gate_cols else {}
        # strong-bull regime knobs
        bull_w = config.get("bull_weights")              # short-key weight dict for strong bull
        bull_breadth = config.get("bull_breadth", 0.60)
        bull_need_trend = config.get("bull_need_trend", True)
        bull_credit_calm = config.get("bull_credit_calm", None)   # pctile <= this = calm
        bull_lever = config.get("bull_lever", 1.0)       # extra leverage mult in strong bull

        cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
        mom_w = config.get("mom_w", 0.60); val_w = config.get("val_w", 0.15)
        lv_w = config.get("lv_w", 0.15); sec_w = config.get("sec_w", 0.10)
        top_n = config.get("top_n", 8); cap = config.get("cap", 0.15)
        use_rp = config.get("use_rp", True); rp_power = config.get("rp_power", 1.0)
        rebal_days = config.get("rebal_days", 10)
        trailing_stop = config.get("trailing_stop", None)
        vol_scaling = config.get("vol_scaling", False)
        vol_target = config.get("vol_target", 0.20); vol_lookback = config.get("vol_lookback", 40)
        recent_rets = []

        cash = capital
        holdings = {}
        port_values = []
        last_targets = {}
        self._fin_paid = 0.0
        self._gross_path = []   # (date, realized gross / NAV) diagnostic
        prev_sh = {}; prev_px = {}; prev_gross = 0.0   # UNLEVERED vol-signal tracking
        shadow_nav = capital; shadow_gross_val = 0.0   # unlevered-book NAV (vol-scale feedback, as live)

        def shr(dollars, px):
            n = dollars / px
            return float(np.floor(n)) if integer_shares else n

        for day_idx, date in enumerate(trading_dates):
            today = {}
            if date in self.prices.index:
                row = self.prices.loc[date]
                for sym in list(holdings.keys()):
                    v = row.get(sym)
                    if v is not None and not np.isnan(v):
                        today[sym] = v
                for sym in row.dropna().index:
                    today[sym] = row[sym]

            # ---- vol signal = UNLEVERED-book NAV return (leverage-independent, with the
            #      vol-scale feedback equilibrium live uses: vol-scale the strategy, THEN
            #      leverage). Matches the validated overlay's avg gross ~1.19 at 1.49x. ----
            if prev_gross > 0:
                pdr = sum(prev_sh[s] * (today.get(s, prev_px[s]) - prev_px[s]) for s in prev_sh) / prev_gross
                shadow_pnl = shadow_gross_val * pdr
                sret = shadow_pnl / shadow_nav if shadow_nav > 0 else 0.0
                recent_rets.append(sret)
                if len(recent_rets) > vol_lookback:
                    recent_rets.pop(0)
                shadow_nav += shadow_pnl
                shadow_gross_val *= (1 + pdr)

            # trailing stop (identical to live)
            if trailing_stop:
                for sym in list(holdings):
                    px = today.get(sym)
                    if px:
                        if "peak_px" not in holdings[sym]:
                            holdings[sym]["peak_px"] = px
                        if px > holdings[sym]["peak_px"]:
                            holdings[sym]["peak_px"] = px
                        dd = (px - holdings[sym]["peak_px"]) / holdings[sym]["peak_px"]
                        if dd < -abs(trailing_stop):
                            cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                            del holdings[sym]

            # ---- LIVEMIRROR: daily financing on the margin debit ----
            if fin_rate and cash < 0:
                chg = (-cash) * (fin_rate / 252.0)
                cash -= chg; self._fin_paid += chg

            nav = cash + sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in holdings.items())

            def _snapshot():
                prev_sh.clear(); prev_px.clear()
                g = 0.0
                for s, h in holdings.items():
                    px = today.get(s, h["entry_px"])
                    prev_sh[s] = h["shares"]; prev_px[s] = px; g += h["shares"] * px
                return g

            if day_idx % rebal_days != 0:
                prev_gross = _snapshot()
                port_values.append((date, nav))
                continue

            # short interest ranks (identical)
            if self._si_months:
                midx = np.searchsorted(self._si_months, date, side="right") - 1
                if midx >= 0:
                    self.uni._short_interest_rank = self._si_ranks_by_month.get(self._si_months[midx], {})
                    self.uni._si_change_rank = self._si_change_ranks_by_month.get(self._si_months[midx], {})
                else:
                    self.uni._short_interest_rank = {}; self.uni._si_change_rank = {}

            # ---- strategy signals (EXACT production code) ----
            mom_qual = config.get("mom_quality_filter")   # e.g. "gp_assets"/"roe": keep top-N by quality
            sector_cap = config.get("sector_cap")         # max names per 2-digit SIC among top_n
            pool = config.get("mom_pool", config.get("mom_quality_pool", 2.0))
            _tn = int(round(top_n * pool)) if (mom_qual or sector_cap) else top_n
            t1 = strategy1_momentum_reversal(date, self.uni, day_idx, top_n=_tn, rebal_days=rebal_days)
            if t1 is None:
                t1 = last_targets.get("mom", {})
            elif mom_qual and len(t1) > top_n:
                qmap = self.uni.get_feature_map(date, mom_qual)
                ranked = sorted(t1.keys(),
                                key=lambda s: (qmap.get(s) if qmap.get(s) is not None else -1e18), reverse=True)
                keep = ranked[:top_n]
                tot = sum(t1[s] for s in keep)
                t1 = {s: t1[s] / tot for s in keep} if tot > 0 else {s: t1[s] for s in keep}
            elif sector_cap and len(t1) > top_n:
                smap = self._sector_map()
                ranked = sorted(t1, key=t1.get, reverse=True)
                picked, cnt = [], {}
                for s in ranked:
                    sec = smap.get(s, -1)
                    if cnt.get(sec, 0) < sector_cap:
                        picked.append(s); cnt[sec] = cnt.get(sec, 0) + 1
                    if len(picked) >= top_n:
                        break
                for s in ranked:                      # backfill if cap left us short
                    if len(picked) >= top_n:
                        break
                    if s not in picked:
                        picked.append(s)
                tot = sum(t1[s] for s in picked)
                t1 = {s: t1[s] / tot for s in picked} if tot > 0 else {s: t1[s] for s in picked}
            members = self.uni.get_sp500(date)
            t_val = strategy_value(self.uni, date, members, top_n=10)
            t3 = strategy3_sector_rotation(date, self.uni, day_idx)
            t5 = strategy5_lowvol_quality(date, self.uni, day_idx)
            if t3 is None:
                t3 = last_targets.get("s3", {})
            if t5 is None:
                t5 = last_targets.get("s5", {})
            last_targets.update({"mom": t1, "val": t_val, "s3": t3, "s5": t5})

            nu = self.umd_20d.loc[:date]
            in_crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < UMD_CRASH_THRESHOLD
            ew = _short_weights(PROD_WEIGHTS_CRASH) if in_crash else {"mom": mom_w, "val": val_w, "s5": lv_w, "s3": sec_w}

            # ---- vol-managed momentum (Barroso): scale the mom sleeve by target/own-vol ----
            vmm = config.get("vol_managed_mom")   # target annualized vol for the momentum book
            if vmm and t1:
                syms = [s for s in t1 if s in self.prices.columns]
                if len(syms) >= 2:
                    px = self.prices[syms].loc[:date].tail(127)
                    pr = px.pct_change().dropna(how="all")
                    if len(pr) >= 60:
                        port = pr.mean(axis=1)   # equal-weight momentum book proxy
                        rv = port.std() * np.sqrt(252)
                        if rv and rv > 0.01:
                            sc = min(config.get("vmm_cap", 1.0), vmm / rv)
                            ew = dict(ew); ew["mom"] = ew["mom"] * sc

            fdate = self.features_by_date.get(date, {})
            above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
            total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
            breadth = above / max(total_f, 1)
            blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))
            bear = config.get("bear_weights", _short_weights(PROD_WEIGHTS_BEAR))
            blended = {n: ew[n] * blend + bear.get(n, 0) * (1 - blend) for n in ew}

            # ---- STRONG-BULL regime: tilt harder into momentum when breadth is high AND
            #      SPY uptrend AND credit calm (and not in a UMD crash). Multi-state mix. ----
            cr_pct = crmap.get(date, 0.5)
            strong_bull = False
            if (bull_w or bull_lever > 1.0) and not in_crash and breadth >= bull_breadth:
                up = True
                if bull_need_trend and "SPY" in self.prices.columns:
                    sh = self.prices["SPY"].loc[:date].dropna()
                    up = len(sh) >= 200 and sh.iloc[-1] > sh.iloc[-200:].mean()
                calm = (bull_credit_calm is None) or (cr_pct <= bull_credit_calm)
                strong_bull = up and calm
            if strong_bull and bull_w:
                blended = dict(bull_w)

            combined = {}
            for name, cap_pct in blended.items():
                tgt = last_targets.get(name, {})
                if use_rp and name in ("mom", "val"):
                    tgt = self._apply_rp(tgt, date, power=rp_power)
                for sym, w in tgt.items():
                    if w > 0:
                        combined[sym] = combined.get(sym, 0) + w * cap_pct

            if vol_scaling and len(recent_rets) >= 20:
                realized_vol = np.std(recent_rets) * np.sqrt(252)
                if realized_vol > 0.01:
                    vol_scale = min(1.5, max(0.3, vol_target / realized_vol))
                    combined = {s: w * vol_scale for s, w in combined.items()}

            longs = {s: w for s, w in combined.items() if w > 0}
            for sym in list(longs):
                if longs[sym] > cap:
                    longs[sym] = cap
            gross = sum(longs.values())
            if gross > 1.0:
                for sym in longs:
                    longs[sym] /= gross
            combined = {s: w for s, w in longs.items() if w >= 0.005}
            shadow_gross_val = sum(combined.values()) * shadow_nav   # redeploy unlevered shadow

            # ---- LIVEMIRROR: leverage * de-risk gate (down) * bull-lever (up) ----
            if gate_cols:
                off = any(gate_maps[c].get(date, 0.5) >= gate_pct for c in gate_cols)
                derisk = gate_derisk if off else 1.0
            else:
                derisk = credit_derisk if (credit_pct and cr_pct >= credit_pct) else 1.0
            lev_t = leverage * derisk
            if strong_bull and bull_lever > 1.0:
                lev_t *= bull_lever
            budget = nav * lev_t
            target_d = {s: w * budget for s, w in combined.items()}

            # exit names no longer targeted (integer -> sell whole position)
            for sym in list(holdings):
                if sym not in target_d:
                    px = today.get(sym, holdings[sym]["entry_px"])
                    cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                    del holdings[sym]

            # rebalance to integer share targets (margin allowed; no cash>=delta block)
            for sym, tgt in target_d.items():
                px = today.get(sym)
                if not px or px <= 0:
                    continue
                cur_sh = holdings[sym]["shares"] if sym in holdings else 0.0
                tgt_sh = shr(tgt, px)
                d_sh = tgt_sh - cur_sh
                d_val = d_sh * px
                if abs(d_val) < max(nav * 0.003, 1e-9):
                    continue
                cost = abs(d_val) * cost_frac
                if d_sh > 0:
                    cash -= d_val + cost
                    if sym in holdings:
                        holdings[sym]["shares"] += d_sh
                    else:
                        holdings[sym] = {"shares": d_sh, "entry_px": px, "peak_px": px}
                else:
                    cash += (-d_val) - cost
                    holdings[sym]["shares"] += d_sh
                    if holdings[sym]["shares"] < (1 if integer_shares else 0.01):
                        cash += 0
                        del holdings[sym]

            nav = cash + sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in holdings.items())
            prev_gross = _snapshot()   # end-of-rebal snapshot for tomorrow's vol signal
            self._gross_path.append((date, prev_gross / nav if nav > 0 else 0))
            port_values.append((date, max(nav, 0)))

        vals = pd.Series([v for _, v in port_values],
                         index=pd.DatetimeIndex([d for d, _ in port_values]))
        years = max((vals.index[-1] - vals.index[0]).days / 365.25, 1)
        dr = vals.pct_change().dropna()
        cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
        sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        peak = vals.cummax(); max_dd = ((vals - peak) / peak).min()
        gp = pd.Series([g for _, g in self._gross_path])
        return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
                "vol": dr.std() * np.sqrt(252), "final": vals.iloc[-1],
                "fin_paid": self._fin_paid, "avg_gross": float(gp.mean()) if len(gp) else 0,
                "daily_values": vals}
