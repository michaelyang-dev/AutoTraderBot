"""
Momentum Scoring Variants for v10 Strategy
============================================
Drop-in replacements for strategy1_momentum_reversal with different
momentum scoring algorithms. All preserve the bear sector tilt,
signal-weighted allocation, and stress detection from production.
"""

import numpy as np


def strategy1_skip_month(date, uni, day_idx, top_n=10, rebal_days=5,
                         trend_filter="sma200", quality_boosts=True):
    """
    Skip-month momentum (12-1): 252d return minus 20d return.
    Academic standard momentum factor. Preserves all production features:
    bear sector tilt, signal-weighted allocation, stress detection.
    """
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)
    if len(members) < 50:
        return {}

    regime = uni.get_regime(date)
    # Stress detection (same as production)
    dist_sma50_all = uni.get_feature_map(date, "dist_sma50")
    if dist_sma50_all:
        mkt_breadth = sum(1 for v in dist_sma50_all.values() if v > 0) / max(len(dist_sma50_all), 1)
    else:
        mkt_breadth = 0.5
    stress = mkt_breadth < 0.30
    n = max(top_n // 2, 5) if stress else top_n

    # Features
    ret_20 = uni.get_feature_map(date, "ret_20d", members)
    ret_252 = uni.get_feature_map(date, "ret_252d", members)
    dist_sma50 = uni.get_feature_map(date, "dist_sma50", members)
    dist_sma200 = uni.get_feature_map(date, "dist_sma200", members)
    eps_surp = uni.get_feature_map(date, "eps_surprise_last", members)
    roe_map = uni.get_feature_map(date, "roe")

    composite = {}
    for sym in members:
        r252 = ret_252.get(sym)
        r20 = ret_20.get(sym)
        if r252 is None or r20 is None or np.isnan(r252) or np.isnan(r20):
            continue

        # Skip-month momentum: 12-month minus last month
        score = r252 - r20

        # Quality boosts (same as production)
        if quality_boosts:
            es = eps_surp.get(sym)
            if es is not None and not np.isnan(es) and es > 0:
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
            else:
                rs = uni._revenue_surprise.get(sym)
                if rs is not None and not np.isnan(rs) and rs > 0.02:
                    score *= 1.10
                bs = uni._beat_streak.get(sym, 0)
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

        # Trend filter
        if trend_filter == "sma50":
            d50 = dist_sma50.get(sym, 0)
            if d50 is None or d50 <= 0:
                continue
        elif trend_filter == "sma200":
            d200 = dist_sma200.get(sym, 0)
            if d200 is None or d200 <= 0:
                continue
        elif trend_filter == "either":
            d50 = dist_sma50.get(sym, 0)
            d200 = dist_sma200.get(sym, 0)
            if (d50 is None or d50 <= 0) and (d200 is None or d200 <= 0):
                continue

        composite[sym] = score

    # Bear sector tilt (same as production)
    spy_bull = regime.get("spy_above_sma200", True)
    if not spy_bull and composite:
        sec_rets = {}
        for etf in ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]:
            close = uni.get_close_series(etf, date, 65)
            if len(close) >= 60:
                sec_rets[etf] = (close.iloc[-1] / close.iloc[-60]) - 1.0
        etf_to_sec = {"XLK": "Technology", "XLF": "Financial Services", "XLE": "Energy",
                      "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Consumer Cyclical",
                      "XLP": "Consumer Defensive", "XLB": "Basic Materials", "XLRE": "Real Estate",
                      "XLU": "Utilities", "XLC": "Communication Services"}
        if sec_rets:
            ranked = sorted(sec_rets, key=sec_rets.get, reverse=True)
            top_secs = {etf_to_sec.get(e, "") for e in ranked[:3]}
            bot_secs = {etf_to_sec.get(e, "") for e in ranked[-3:]}
            boosted = {}
            for sym, sc in composite.items():
                sym_sec = uni.sector_map.get(sym, "")
                if sym_sec in top_secs:
                    boosted[sym] = sc * 2.0
                elif sym_sec in bot_secs:
                    continue
                else:
                    boosted[sym] = sc
            composite = boosted

    # Short interest (Ortex live only)
    for sym in list(composite.keys()):
        ortex = uni._options.get(sym)
        if ortex and ortex.get("si_pct_float") is not None:
            si = ortex["si_pct_float"]
            if si < 2.0:
                composite[sym] *= 1.10
            elif si > 10.0:
                composite[sym] *= 0.85

    if not composite:
        return {}

    # Signal-weighted allocation (same as production)
    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:n]
    scores = [max(composite[s], 0.001) for s in sorted_syms]
    total = sum(scores)
    if total > 0:
        weights = {s: min(sc / total, 2.0 / len(sorted_syms)) for s, sc in zip(sorted_syms, scores)}
        wt = sum(weights.values())
        if wt > 0:
            weights = {s: w / wt for s, w in weights.items()}
        return weights
    return {}


def strategy1_risk_adj(date, uni, day_idx, top_n=10, rebal_days=5,
                       trend_filter="sma200", quality_boosts=True):
    """
    Risk-adjusted momentum: 12-month return / 60-day volatility.
    Same framework as skip_month but normalizes by volatility.
    """
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)
    if len(members) < 50:
        return {}

    regime = uni.get_regime(date)
    dist_sma50_all = uni.get_feature_map(date, "dist_sma50")
    if dist_sma50_all:
        mkt_breadth = sum(1 for v in dist_sma50_all.values() if v > 0) / max(len(dist_sma50_all), 1)
    else:
        mkt_breadth = 0.5
    stress = mkt_breadth < 0.30
    n = max(top_n // 2, 5) if stress else top_n

    ret_252 = uni.get_feature_map(date, "ret_252d", members)
    vol_60 = uni.get_feature_map(date, "vol_60d", members)
    dist_sma50 = uni.get_feature_map(date, "dist_sma50", members)
    dist_sma200 = uni.get_feature_map(date, "dist_sma200", members)
    eps_surp = uni.get_feature_map(date, "eps_surprise_last", members)
    roe_map = uni.get_feature_map(date, "roe")

    composite = {}
    for sym in members:
        r252 = ret_252.get(sym)
        v60 = vol_60.get(sym)
        if r252 is None or v60 is None or np.isnan(r252) or np.isnan(v60) or v60 < 0.05:
            continue

        score = r252 / v60

        if quality_boosts:
            es = eps_surp.get(sym)
            if es is not None and not np.isnan(es) and es > 0:
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

        if trend_filter == "sma50":
            d50 = dist_sma50.get(sym, 0)
            if d50 is None or d50 <= 0:
                continue
        elif trend_filter == "sma200":
            d200 = dist_sma200.get(sym, 0)
            if d200 is None or d200 <= 0:
                continue

        composite[sym] = score

    # Bear sector tilt
    spy_bull = regime.get("spy_above_sma200", True)
    if not spy_bull and composite:
        sec_rets = {}
        for etf in ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]:
            close = uni.get_close_series(etf, date, 65)
            if len(close) >= 60:
                sec_rets[etf] = (close.iloc[-1] / close.iloc[-60]) - 1.0
        etf_to_sec = {"XLK": "Technology", "XLF": "Financial Services", "XLE": "Energy",
                      "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Consumer Cyclical",
                      "XLP": "Consumer Defensive", "XLB": "Basic Materials", "XLRE": "Real Estate",
                      "XLU": "Utilities", "XLC": "Communication Services"}
        if sec_rets:
            ranked = sorted(sec_rets, key=sec_rets.get, reverse=True)
            top_secs = {etf_to_sec.get(e, "") for e in ranked[:3]}
            bot_secs = {etf_to_sec.get(e, "") for e in ranked[-3:]}
            boosted = {}
            for sym, sc in composite.items():
                sym_sec = uni.sector_map.get(sym, "")
                if sym_sec in top_secs:
                    boosted[sym] = sc * 2.0
                elif sym_sec in bot_secs:
                    continue
                else:
                    boosted[sym] = sc
            composite = boosted

    for sym in list(composite.keys()):
        ortex = uni._options.get(sym)
        if ortex and ortex.get("si_pct_float") is not None:
            si = ortex["si_pct_float"]
            if si < 2.0:
                composite[sym] *= 1.10
            elif si > 10.0:
                composite[sym] *= 0.85

    if not composite:
        return {}

    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:n]
    scores = [max(composite[s], 0.001) for s in sorted_syms]
    total = sum(scores)
    if total > 0:
        weights = {s: min(sc / total, 2.0 / len(sorted_syms)) for s, sc in zip(sorted_syms, scores)}
        wt = sum(weights.values())
        if wt > 0:
            weights = {s: w / wt for s, w in weights.items()}
        return weights
    return {}
