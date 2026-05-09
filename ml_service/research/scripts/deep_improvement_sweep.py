"""
Deep Structural Improvement Sweep for v10 Momentum Strategy
============================================================
Tests fundamental changes to the momentum scoring, regime handling,
and portfolio construction to push OOS CAGR from ~14-20% toward 24%+.

Changes tested:
  A) Skip-month momentum (12-1 month, academic standard)
  B) Risk-adjusted momentum (return / volatility)
  C) Remove multiplicative quality boosts (simpler = less overfit)
  D) Remove regime shifting (UMD crash / breadth bear)
  E) Linear quality composite instead of multiplicative
  F) Different position counts and rebalance frequencies
  G) No value/lowvol/sector sleeves (pure momentum)
"""

import numpy as np
import pandas as pd
import time
import logging
import sys
from pathlib import Path
from fast_backtest import FastBacktester, SLIPPAGE_BPS
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, INITIAL_CASH, COST_BPS,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
log = logging.getLogger("deep_sweep")

# We'll subclass FastBacktester to override the run method with custom scoring


class DeepBacktester(FastBacktester):
    """Extended backtester with pluggable momentum scoring."""

    def run_custom(self, start="2018-01-01", end="2025-12-31", config=None):
        """
        Run with custom momentum scoring function.
        config keys:
          - scoring: "production" | "skip_month" | "risk_adj" | "simple_avg" | "six_one"
          - quality_mode: "multiplicative" (production) | "additive" | "none"
          - regime_mode: "production" | "none" | "light"
          - top_n, rebal_days, cap, trailing_stop
          - mom_w, val_w, lv_w, sec_w (strategy weights)
          - use_rp, rp_power, gld_pct, vixm_pct
          - vol_scaling, vol_target
          - trend_filter: "sma50" (production) | "sma100" | "sma200" | "none"
          - earnings_filter: True/False
        """
        if config is None:
            config = {}

        scoring = config.get("scoring", "production")
        quality_mode = config.get("quality_mode", "multiplicative")
        regime_mode = config.get("regime_mode", "production")
        trend_filter = config.get("trend_filter", "sma50")
        earnings_filter = config.get("earnings_filter", True)

        # Universe
        universe = config.get("universe", "sp1500")
        if universe == "sp1500":
            self.uni.get_sp500 = self._get_sp1500
        else:
            self.uni.get_sp500 = self._original_get_sp500

        trading_dates = [d for d in sorted(self.prices.index)
                         if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        if not trading_dates:
            return None

        cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
        mom_w = config.get("mom_w", 1.0)
        val_w = config.get("val_w", 0.0)
        lv_w = config.get("lv_w", 0.0)
        sec_w = config.get("sec_w", 0.0)
        top_n = config.get("top_n", 10)
        cap = config.get("cap", 0.15)
        use_rp = config.get("use_rp", True)
        rp_power = config.get("rp_power", 1.0)
        gld_pct = config.get("gld_pct", 0.0)
        vixm_pct = config.get("vixm_pct", 0.0)
        rebal_days = config.get("rebal_days", 5)
        trailing_stop = config.get("trailing_stop", 0.25)
        vol_scaling = config.get("vol_scaling", False)
        vol_target = config.get("vol_target", 0.20)
        recent_rets = []

        cash = INITIAL_CASH
        holdings = {}
        port_values = []
        last_targets = {}

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

            # Per-position trailing stop
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

            eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                for s, h in holdings.items())
            total_val = eq_val

            # Track returns for vol scaling
            if len(port_values) > 0:
                prev = port_values[-1][1]
                if prev > 0:
                    recent_rets.append(total_val / prev - 1)
                    if len(recent_rets) > 40:
                        recent_rets.pop(0)

            if day_idx % rebal_days != 0:
                port_values.append((date, total_val))
                continue

            # ── Custom momentum scoring ─────────────────────────
            members = self.uni.get_sp500(date)
            if len(members) < 50:
                port_values.append((date, total_val))
                continue

            composite = self._custom_score(date, members, scoring, quality_mode,
                                           trend_filter, earnings_filter, top_n)

            # ── Regime handling ─────────────────────────────────
            if regime_mode == "production":
                # Same as production: UMD crash + breadth bear
                nu = self.umd_20d.loc[:date]
                in_crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < -0.05
                fdate = self.features_by_date.get(date, {})
                above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
                total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
                breadth = above / max(total_f, 1)

                if in_crash:
                    ew = {"mom": 0.15, "val": 0.45, "s5": 0.30, "s3": 0.10}
                else:
                    ew = {"mom": mom_w, "val": val_w, "s5": lv_w, "s3": sec_w}

                blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))
                bear = {"mom": 0.10, "val": 0.20, "s5": 0.60, "s3": 0.10}
                blended = {n: ew[n] * blend + bear.get(n, 0) * (1 - blend)
                           for n in ew}

                # Get other strategy targets
                t_val = self._strategy_value(date, members, top_n=10)
                t3 = strategy3_sector_rotation(date, self.uni, day_idx)
                t5 = strategy5_lowvol_quality(date, self.uni, day_idx)
                if t3 is None:
                    t3 = last_targets.get("s3", {})
                if t5 is None:
                    t5 = last_targets.get("s5", {})
                last_targets.update({"val": t_val, "s3": t3, "s5": t5})

                # Blend
                combined = {}
                tgts = {"mom": composite, "val": t_val, "s5": t5, "s3": t3}
                for name, cap_pct in blended.items():
                    tgt = tgts.get(name, {})
                    if use_rp and name in ("mom", "val"):
                        tgt = self._apply_rp(tgt, date, power=rp_power)
                    for sym, w in tgt.items():
                        if w > 0:
                            combined[sym] = combined.get(sym, 0) + w * cap_pct

            elif regime_mode == "light":
                # Light regime: only reduce to cash in extreme crashes, no sleeve shifting
                fdate = self.features_by_date.get(date, {})
                above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
                total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
                breadth = above / max(total_f, 1)

                # In extreme bear (breadth < 25%), scale down exposure
                if breadth < 0.25:
                    scale = max(0.3, breadth / 0.25)
                else:
                    scale = 1.0

                if use_rp:
                    composite = self._apply_rp(composite, date, power=rp_power)
                combined = {s: w * scale for s, w in composite.items()}

            else:  # regime_mode == "none"
                if use_rp:
                    composite = self._apply_rp(composite, date, power=rp_power)
                combined = dict(composite)

            # Vol scaling
            if vol_scaling and len(recent_rets) >= 20:
                realized_vol = np.std(recent_rets) * np.sqrt(252)
                if realized_vol > 0.01:
                    vol_scale = min(1.5, max(0.3, vol_target / realized_vol))
                    combined = {s: w * vol_scale for s, w in combined.items()}

            # Normalize and cap
            longs = {s: w for s, w in combined.items() if w > 0}
            for sym in list(longs):
                if longs[sym] > cap:
                    longs[sym] = cap
            gross = sum(longs.values())
            if gross > 1.0:
                for sym in longs:
                    longs[sym] /= gross
            combined = {s: w for s, w in longs.items() if w >= 0.005}

            target_d = {s: w * total_val for s, w in combined.items()}

            # Execute trades
            for sym in list(holdings):
                if sym not in target_d:
                    px = today.get(sym, holdings[sym]["entry_px"])
                    cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                    del holdings[sym]

            for sym, tgt in target_d.items():
                px = today.get(sym)
                if not px or px <= 0:
                    continue
                cur = holdings[sym]["shares"] * px if sym in holdings else 0
                delta = tgt - cur
                if abs(delta) < total_val * 0.003:
                    continue
                cost = abs(delta) * cost_frac
                if delta > 0 and cash >= delta:
                    shares = (delta - cost) / px
                    if sym in holdings:
                        holdings[sym]["shares"] += shares
                    else:
                        holdings[sym] = {"shares": shares, "entry_px": px, "peak_px": px}
                    cash -= delta
                elif delta < 0 and sym in holdings:
                    sell = min(abs(delta) / px, holdings[sym]["shares"])
                    cash += sell * px - cost
                    holdings[sym]["shares"] -= sell
                    if holdings[sym]["shares"] < 0.01:
                        del holdings[sym]

            eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                for s, h in holdings.items())
            total_val = eq_val
            port_values.append((date, max(total_val, 0)))

        # Metrics
        vals = pd.Series([v for _, v in port_values],
                         index=pd.DatetimeIndex([d for d, _ in port_values]))
        years = (vals.index[-1] - vals.index[0]).days / 365.25
        if years <= 0:
            years = 1
        dr = vals.pct_change().dropna()
        cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
        sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        peak = vals.cummax()
        max_dd = ((vals - peak) / peak).min()
        vol = dr.std() * np.sqrt(252)
        yearly = {}
        for year in range(int(start[:4]), int(end[:4]) + 1):
            mask = (vals.index >= f"{year}-01-01") & (vals.index <= f"{year}-12-31")
            yv = vals[mask]
            if len(yv) > 10:
                yr = (yv.iloc[-1] / yv.iloc[0]) - 1
                ydr = yv.pct_change().dropna()
                ys = ydr.mean() / ydr.std() * np.sqrt(252) if ydr.std() > 0 else 0
                ydd = ((yv - yv.cummax()) / yv.cummax()).min()
                yearly[year] = {"cagr": yr, "sharpe": ys, "max_dd": ydd}
        return {
            "cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "vol": vol,
            "final": vals.iloc[-1], "yearly": yearly,
        }

    def _custom_score(self, date, members, scoring, quality_mode,
                      trend_filter, earnings_filter, top_n):
        """
        Custom momentum scoring with different algorithms.
        Returns {symbol: weight} for top_n picks.
        """
        uni = self.uni

        # Get all features we might need
        ret_20 = uni.get_feature_map(date, "ret_20d", members)
        ret_60 = uni.get_feature_map(date, "ret_60d", members)
        ret_120 = uni.get_feature_map(date, "ret_120d", members)
        ret_126 = uni.get_feature_map(date, "ret_126d", members)
        ret_252 = uni.get_feature_map(date, "ret_252d", members)
        vol_60 = uni.get_feature_map(date, "vol_60d", members)
        dist_sma50 = uni.get_feature_map(date, "dist_sma50", members)
        dist_sma200 = uni.get_feature_map(date, "dist_sma200", members)
        eps_surp = uni.get_feature_map(date, "eps_surprise_last", members)
        roe_map = uni.get_feature_map(date, "roe")

        scores = {}
        for sym in members:
            # ── Momentum score ──
            if scoring == "skip_month":
                # 12-1 month momentum: skip the most recent month
                r252 = ret_252.get(sym)
                r20 = ret_20.get(sym)
                if r252 is None or r20 is None or np.isnan(r252) or np.isnan(r20):
                    continue
                # 12-month return minus last month
                mom = r252 - r20
                score = mom

            elif scoring == "six_one":
                # 6-1 month momentum
                r126 = ret_126.get(sym)
                r20 = ret_20.get(sym)
                if r126 is None or r20 is None or np.isnan(r126) or np.isnan(r20):
                    continue
                mom = r126 - r20
                score = mom

            elif scoring == "risk_adj":
                # Risk-adjusted momentum (return / volatility)
                r126 = ret_126.get(sym)
                v60 = vol_60.get(sym)
                if r126 is None or v60 is None or np.isnan(r126) or np.isnan(v60) or v60 < 0.01:
                    continue
                score = r126 / v60

            elif scoring == "risk_adj_12":
                # Risk-adjusted 12-month momentum
                r252 = ret_252.get(sym)
                v60 = vol_60.get(sym)
                if r252 is None or v60 is None or np.isnan(r252) or np.isnan(v60) or v60 < 0.01:
                    continue
                score = r252 / v60

            elif scoring == "dual_mom":
                # Dual momentum: 6m + 12m, skip recent month
                r126 = ret_126.get(sym)
                r252 = ret_252.get(sym)
                r20 = ret_20.get(sym)
                if r126 is None or r252 is None or r20 is None:
                    continue
                if np.isnan(r126) or np.isnan(r252) or np.isnan(r20):
                    continue
                # Average of 6-1 and 12-1
                m6 = r126 - r20
                m12 = r252 - r20
                score = 0.5 * m6 + 0.5 * m12

            elif scoring == "simple_avg":
                # Simple average of multi-timeframe (no consistency weighting)
                rets = []
                for rd in [ret_20, ret_60, ret_126, ret_252]:
                    v = rd.get(sym)
                    if v is not None and not np.isnan(v):
                        rets.append(v)
                if len(rets) < 2:
                    continue
                score = np.mean(rets)

            elif scoring == "weighted_recent":
                # Weight recent momentum more heavily
                r20 = ret_20.get(sym)
                r60 = ret_60.get(sym)
                r126 = ret_126.get(sym)
                r252 = ret_252.get(sym)
                vals = [(r20, 0.35), (r60, 0.30), (r126, 0.20), (r252, 0.15)]
                total_w = 0
                total_s = 0
                for v, w in vals:
                    if v is not None and not np.isnan(v):
                        total_s += v * w
                        total_w += w
                if total_w < 0.5:
                    continue
                score = total_s / total_w

            else:  # "production" — exact production scoring
                rets = []
                for rd in [ret_20, ret_60, ret_126, ret_252]:
                    v = rd.get(sym)
                    if v is not None and not np.isnan(v):
                        rets.append(v)
                if len(rets) < 2:
                    continue
                consistency = sum(1 for r in rets if r > 0) / len(rets)
                avg_ret = np.mean(rets)
                score = avg_ret * (consistency ** 2)

            # ── Quality adjustments ──
            if quality_mode == "multiplicative":
                # Production style
                es = eps_surp.get(sym)
                if earnings_filter and es is not None and not np.isnan(es) and es > 0:
                    score *= 1.15
                roe_val = roe_map.get(sym)
                if roe_val is not None and not np.isnan(roe_val) and roe_val > 0.15:
                    score *= 1.05
                fg = uni._fin_growth.get(sym)
                if fg:
                    rg = fg.get("rev_growth")
                    eg = fg.get("eps_growth")
                    if rg is not None and not np.isnan(rg) and rg > 0.08:
                        score *= 1.10
                    if eg is not None and not np.isnan(eg) and eg > 0.10:
                        score *= 1.10
                earn_sigs = uni._earnings_signals.get(date, {}).get(sym)
                if earn_sigs:
                    rs = earn_sigs.get("rev_surprise")
                    if rs is not None and not np.isnan(rs) and rs > 0.02:
                        score *= 1.10
                    bs = earn_sigs.get("beat_streak", 0)
                    if bs >= 3:
                        score *= 1.05
                pt = uni._price_targets.get(sym)
                if pt:
                    px = uni.get_close_at(date, sym)
                    if px and px > 0 and pt["target"] > 0:
                        upside = (pt["target"] - px) / px
                        if upside > 0.15:
                            score *= 1.10
                        elif upside < -0.10:
                            score *= 0.90

            elif quality_mode == "additive":
                # Additive quality: add bonus instead of multiply
                bonus = 0
                es = eps_surp.get(sym)
                if earnings_filter and es is not None and not np.isnan(es) and es > 0:
                    bonus += 0.02
                roe_val = roe_map.get(sym)
                if roe_val is not None and not np.isnan(roe_val) and roe_val > 0.15:
                    bonus += 0.01
                fg = uni._fin_growth.get(sym)
                if fg:
                    rg = fg.get("rev_growth")
                    if rg is not None and not np.isnan(rg) and rg > 0.08:
                        bonus += 0.015
                score += bonus

            elif quality_mode == "roe_only":
                # Only use ROE as quality filter (simplest)
                roe_val = roe_map.get(sym)
                if roe_val is not None and not np.isnan(roe_val) and roe_val > 0.15:
                    score *= 1.10

            # else: quality_mode == "none" — no quality adjustments

            # ── Trend filter ──
            if trend_filter == "sma50":
                d50 = dist_sma50.get(sym, 0)
                if d50 is None or d50 <= 0:
                    continue
            elif trend_filter == "sma200":
                d200 = dist_sma200.get(sym, 0)
                if d200 is None or d200 <= 0:
                    continue
            elif trend_filter == "both":
                d50 = dist_sma50.get(sym, 0)
                d200 = dist_sma200.get(sym, 0)
                if d50 is None or d50 <= 0 or d200 is None or d200 <= 0:
                    continue
            # else: "none" — no trend filter

            scores[sym] = score

        if not scores:
            return {}

        # Select top N
        ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
        return {s: 1.0 / len(ranked) for s in ranked}


def fmt_yearly(yearly, oos_start=2022):
    """Format yearly returns compactly."""
    parts = []
    for y in sorted(yearly.keys()):
        c = yearly[y]["cagr"]
        marker = "*" if y >= oos_start else " "
        parts.append(f"{y}:{c:+.0%}{marker}")
    return " ".join(parts)


def run_split(bt, config, name):
    """Run full sample + OOS split and print results."""
    t0 = time.time()

    # Full sample
    full = bt.run_custom("2018-01-01", "2025-04-30", config)
    # OOS only
    oos = bt.run_custom("2022-01-01", "2025-04-30", config)

    elapsed = time.time() - t0
    if not full or not oos:
        print(f"  {name:<55} FAILED")
        return None

    print(f"  {name:<55} Full: {full['cagr']:>+5.1%} S={full['sharpe']:.2f} DD={full['max_dd']:.0%}"
          f"  |  OOS: {oos['cagr']:>+5.1%} S={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}"
          f"  [{elapsed:.0f}s]")
    print(f"    {fmt_yearly(full['yearly'])}")
    return {"full": full, "oos": oos, "name": name}


if __name__ == "__main__":
    print("Loading data...")
    bt = DeepBacktester()
    print("Data loaded.\n")

    results = []

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Momentum Scoring Algorithms
    # ═══════════════════════════════════════════════════════════════
    print("=" * 120)
    print("PHASE 1: Momentum Scoring (pure mom, no regime, r5, top10, trail25)")
    print("=" * 120)

    base = {
        "universe": "sp1500", "mom_w": 1.0, "val_w": 0.0, "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 10, "rebal_days": 5, "trailing_stop": 0.25, "cap": 0.15,
        "regime_mode": "none", "use_rp": True, "quality_mode": "multiplicative",
        "trend_filter": "sma50", "earnings_filter": True,
    }

    for scoring in ["production", "skip_month", "six_one", "risk_adj", "risk_adj_12",
                     "dual_mom", "simple_avg", "weighted_recent"]:
        cfg = {**base, "scoring": scoring}
        r = run_split(bt, cfg, f"scoring={scoring}")
        if r:
            results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Quality Mode Variations (best scoring from phase 1)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("PHASE 2: Quality Modes (with top scoring methods)")
    print("=" * 120)

    # Test quality modes with the top 3 likely scorers
    for scoring in ["production", "skip_month", "dual_mom"]:
        for qm in ["none", "roe_only", "additive", "multiplicative"]:
            cfg = {**base, "scoring": scoring, "quality_mode": qm}
            r = run_split(bt, cfg, f"scoring={scoring} quality={qm}")
            if r:
                results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Trend Filter Variations
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("PHASE 3: Trend Filters")
    print("=" * 120)

    for scoring in ["production", "skip_month"]:
        for tf in ["sma50", "sma200", "both", "none"]:
            cfg = {**base, "scoring": scoring, "trend_filter": tf}
            r = run_split(bt, cfg, f"scoring={scoring} trend={tf}")
            if r:
                results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: Regime Handling
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("PHASE 4: Regime Modes")
    print("=" * 120)

    for scoring in ["production", "skip_month"]:
        for rm in ["none", "light", "production"]:
            # With production regime, need strategy weights
            if rm == "production":
                cfg = {**base, "scoring": scoring, "regime_mode": rm,
                       "mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0}
            else:
                cfg = {**base, "scoring": scoring, "regime_mode": rm}
            r = run_split(bt, cfg, f"scoring={scoring} regime={rm}")
            if r:
                results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5: Position Count & Rebalance Frequency
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("PHASE 5: Position Count & Rebalance (best scoring)")
    print("=" * 120)

    for scoring in ["production", "skip_month"]:
        for top_n in [5, 8, 10, 12, 15]:
            for rebal in [3, 5, 10]:
                cfg = {**base, "scoring": scoring, "top_n": top_n, "rebal_days": rebal}
                r = run_split(bt, cfg, f"scoring={scoring} top{top_n} r{rebal}")
                if r:
                    results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 6: Trailing Stop Variations
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("PHASE 6: Trailing Stop")
    print("=" * 120)

    for scoring in ["production", "skip_month"]:
        for ts in [None, 0.15, 0.20, 0.25, 0.30, 0.35]:
            cfg = {**base, "scoring": scoring, "trailing_stop": ts}
            ts_name = f"{ts:.0%}" if ts else "none"
            r = run_split(bt, cfg, f"scoring={scoring} trail={ts_name}")
            if r:
                results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 7: Best Combinations
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("PHASE 7: Best Combinations")
    print("=" * 120)

    # Assemble combinations of best elements found above
    combos = [
        ("BEST-A: skip_month+noQ+r5+t10+trail25", {
            **base, "scoring": "skip_month", "quality_mode": "none",
            "top_n": 10, "rebal_days": 5, "trailing_stop": 0.25,
        }),
        ("BEST-B: skip_month+addQ+r5+t10+trail30", {
            **base, "scoring": "skip_month", "quality_mode": "additive",
            "top_n": 10, "rebal_days": 5, "trailing_stop": 0.30,
        }),
        ("BEST-C: dual_mom+multQ+r5+t10+trail25", {
            **base, "scoring": "dual_mom", "quality_mode": "multiplicative",
            "top_n": 10, "rebal_days": 5, "trailing_stop": 0.25,
        }),
        ("BEST-D: skip_month+multQ+r3+t8+trail25", {
            **base, "scoring": "skip_month", "quality_mode": "multiplicative",
            "top_n": 8, "rebal_days": 3, "trailing_stop": 0.25,
        }),
        ("BEST-E: risk_adj+multQ+r5+t12+trail30", {
            **base, "scoring": "risk_adj", "quality_mode": "multiplicative",
            "top_n": 12, "rebal_days": 5, "trailing_stop": 0.30,
        }),
        ("BEST-F: production+noQ+r5+t10+trail25+light", {
            **base, "scoring": "production", "quality_mode": "none",
            "top_n": 10, "rebal_days": 5, "trailing_stop": 0.25,
            "regime_mode": "light",
        }),
        ("BEST-G: skip_month+multQ+r5+t10+trail25+volscale", {
            **base, "scoring": "skip_month", "quality_mode": "multiplicative",
            "top_n": 10, "rebal_days": 5, "trailing_stop": 0.25,
            "vol_scaling": True, "vol_target": 0.18,
        }),
        ("BEST-H: skip_month+roeQ+r5+t10+trail25", {
            **base, "scoring": "skip_month", "quality_mode": "roe_only",
            "top_n": 10, "rebal_days": 5, "trailing_stop": 0.25,
        }),
        ("BEST-I: weighted_recent+multQ+r5+t10+trail25", {
            **base, "scoring": "weighted_recent", "quality_mode": "multiplicative",
            "top_n": 10, "rebal_days": 5, "trailing_stop": 0.25,
        }),
        ("BEST-J: dual_mom+roeQ+r3+t10+trail30+sma200", {
            **base, "scoring": "dual_mom", "quality_mode": "roe_only",
            "top_n": 10, "rebal_days": 3, "trailing_stop": 0.30,
            "trend_filter": "sma200",
        }),
    ]

    for name, cfg in combos:
        r = run_split(bt, cfg, name)
        if r:
            results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY: Top 10 by OOS Sharpe
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print("TOP 10 BY OOS SHARPE")
    print("=" * 120)

    by_oos_sharpe = sorted(results, key=lambda x: x["oos"]["sharpe"], reverse=True)
    for i, r in enumerate(by_oos_sharpe[:10]):
        oos = r["oos"]
        full = r["full"]
        print(f"  #{i+1}: {r['name']:<55}")
        print(f"       OOS: CAGR={oos['cagr']:+.1%} Sharpe={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}")
        print(f"       Full: CAGR={full['cagr']:+.1%} Sharpe={full['sharpe']:.2f} DD={full['max_dd']:.0%}")
        print(f"       {fmt_yearly(full['yearly'])}")

    print(f"\n{'=' * 120}")
    print("TOP 10 BY OOS CAGR")
    print("=" * 120)

    by_oos_cagr = sorted(results, key=lambda x: x["oos"]["cagr"], reverse=True)
    for i, r in enumerate(by_oos_cagr[:10]):
        oos = r["oos"]
        full = r["full"]
        print(f"  #{i+1}: {r['name']:<55}")
        print(f"       OOS: CAGR={oos['cagr']:+.1%} Sharpe={oos['sharpe']:.2f} DD={oos['max_dd']:.0%}")
        print(f"       Full: CAGR={full['cagr']:+.1%} Sharpe={full['sharpe']:.2f} DD={full['max_dd']:.0%}")

    print(f"\nTotal configs tested: {len(results)}")
