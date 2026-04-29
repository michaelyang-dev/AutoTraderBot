"""
Multi-Strategy Engine v9.6 (SP500)
==================================
Core strategy engine for the SP500 portfolio.
Runs momentum (S1), sector rotation (S3), and low-vol quality (S5)
with breadth-based regime blending. Pre-indexes all data for O(1) lookups.

Usage:
    cd ml_service && python3 -m strategies.multi_strategy_engine
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from massive_data_provider import MassiveDataProvider
from sp500_history import get_sp500_on_date, load_sp500_changes
from strategies.portfolio_combiner import PortfolioCombiner
from strategies.s6_bear_short import strategy6_bear_short

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INITIAL_CASH = 100_000.0
COST_BPS = 5
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]


def log(msg):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-indexed data container (fast O(1) lookups)
# ══════════════════════════════════════════════════════════════════════════════

class FastUniverse:
    """Pre-indexed universe data for fast strategy computation."""

    def __init__(self, features_df, prices_df, sector_map, sp500_changes, vix_data,
                 ml_predictions=None, enhanced_data=None):
        log("  Building fast index ...")
        t0 = time.time()

        self.prices = prices_df
        self.sector_map = sector_map
        self.sp500_changes = sp500_changes

        # Pre-index features: {date: {symbol: {feature: value}}}
        self._feat_by_date = {}
        for date, grp in features_df.groupby("date"):
            self._feat_by_date[date] = {}
            for _, row in grp.iterrows():
                self._feat_by_date[date][row["symbol"]] = row.to_dict()

        # Pre-compute close price dict: {symbol: Series}
        self._close = {}
        for col in prices_df.columns:
            self._close[col] = prices_df[col].dropna()

        # Pre-compute daily returns for drift calculation
        self._daily_returns = prices_df.pct_change()

        # SP500 membership cache
        load_sp500_changes()
        self._sp500_cache = {}

        # VIX data
        self._vix = {}
        self._vix_ts = {}
        if vix_data is not None:
            for date in vix_data.index:
                v = vix_data.loc[date].get("^VIX")
                v3m = vix_data.loc[date].get("^VIX3M")
                if v is not None and not np.isnan(v):
                    self._vix[date] = v
                if v and v3m and not np.isnan(v3m) and v > 0:
                    self._vix_ts[date] = v3m / v

        # SP500 additions lookup: {date: [symbols]}
        self._additions = {}
        for c in sp500_changes:
            sym = c.get("symbol", "")
            date_str = c.get("date") or c.get("dateAdded", "")
            if not sym or not date_str:
                continue
            try:
                d = pd.Timestamp(date_str).normalize()
                if d not in self._additions:
                    self._additions[d] = []
                self._additions[d].append(sym)
            except (ValueError, TypeError):
                pass

        # ML predictions index: {date: {symbol: score}}
        self._ml_preds = {}
        if ml_predictions is not None:
            for date, grp in ml_predictions.groupby("date"):
                self._ml_preds[date] = dict(zip(grp["symbol"], grp["prob_ensemble"]))

        # Enhanced data: price targets, DCF, financial growth, crypto/forex
        self._price_targets = {}  # {symbol: {target_consensus, upside_pct}}
        self._dcf = {}            # {symbol: {dcf, upside_pct}}
        self._fin_growth = {}     # {symbol: [{date, revenue_growth, eps_growth, ...}]}
        self._crypto_fx = {}      # {date: {btc_ret_20d, eth_ret_20d, ...}}
        self._ev = {}             # {symbol: [{date, ev_to_revenue, market_cap}]}
        self._profiles = {}       # {symbol: {beta, market_cap, sector, industry}}
        self._insiders = {}       # {symbol: DataFrame with date, is_buy, shares}
        self._sentiment = {}      # {symbol: sentiment_score}
        self._estimates = {}      # {symbol: {eps_avg, revenue_avg}}
        self._options = {}        # {symbol: {pc_ratio, atm_iv}}
        self._short_ratio = {}    # {date: {symbol: short_ratio_20d}} from FINRA

        if enhanced_data:
            # Price targets
            pt = enhanced_data.get("price_targets")
            if pt is not None and len(pt) > 0:
                for _, row in pt.iterrows():
                    sym = row.get("symbol")
                    tc = row.get("target_consensus")
                    if sym and tc and not np.isnan(tc):
                        self._price_targets[sym] = {"target": tc}

            # DCF
            dcf = enhanced_data.get("dcf")
            if dcf is not None and len(dcf) > 0:
                for _, row in dcf.iterrows():
                    sym = row.get("symbol")
                    d = row.get("dcf")
                    if sym and d and not np.isnan(d):
                        self._dcf[sym] = {"dcf": d}

            # Financial growth (use most recent per symbol)
            fg = enhanced_data.get("financial_growth")
            if fg is not None and len(fg) > 0:
                fg = fg.sort_values(["symbol", "date"])
                for sym, grp in fg.groupby("symbol"):
                    latest = grp.iloc[-1]
                    self._fin_growth[sym] = {
                        "rev_growth": latest.get("revenue_growth"),
                        "eps_growth": latest.get("eps_growth"),
                        "fcf_growth": latest.get("free_cf_growth"),
                    }

            # Enterprise values (most recent)
            ev = enhanced_data.get("enterprise_values")
            if ev is not None and len(ev) > 0:
                ev = ev.sort_values(["symbol", "date"])
                for sym, grp in ev.groupby("symbol"):
                    latest = grp.iloc[-1]
                    self._ev[sym] = {
                        "ev_to_rev": latest.get("ev_to_revenue"),
                        "market_cap": latest.get("market_cap"),
                    }

            # Profiles
            prof = enhanced_data.get("profiles")
            if prof is not None and len(prof) > 0:
                for _, row in prof.iterrows():
                    sym = row.get("symbol")
                    if sym:
                        self._profiles[sym] = {
                            "beta": row.get("beta"),
                            "market_cap": row.get("market_cap"),
                            "industry": row.get("industry"),
                        }

            # Crypto/forex (date-level)
            cfx = enhanced_data.get("crypto_forex")
            if cfx is not None and len(cfx) > 0:
                for date in cfx.index:
                    row = cfx.loc[date]
                    self._crypto_fx[date] = {
                        col: row[col] for col in cfx.columns if not np.isnan(row[col])
                    }

            # Insider trades
            ins = enhanced_data.get("insiders")
            if ins is not None and len(ins) > 0:
                ins["date"] = pd.to_datetime(ins["date"])
                for sym, grp in ins.groupby("symbol"):
                    self._insiders[sym] = grp.sort_values("date")

            # Transcript sentiment (most recent per symbol)
            ts = enhanced_data.get("transcript_sentiment")
            if ts is not None and len(ts) > 0:
                ts = ts.sort_values(["symbol", "date"]) if "date" in ts.columns else ts
                for sym, grp in ts.groupby("symbol"):
                    self._sentiment[sym] = grp.iloc[-1].get("sentiment", 0.0)

            # Forward estimates (most recent per symbol)
            est = enhanced_data.get("estimates")
            if est is not None and len(est) > 0:
                est = est.sort_values(["symbol", "date"]) if "date" in est.columns else est
                for sym, grp in est.groupby("symbol"):
                    latest = grp.iloc[-1]
                    eps = latest.get("eps_avg")
                    rev = latest.get("revenue_avg")
                    if eps is not None and not np.isnan(eps):
                        self._estimates[sym] = {"eps_avg": eps, "revenue_avg": rev}

            # Options snapshot (put/call ratio, IV)
            opt = enhanced_data.get("options")
            if opt is not None and len(opt) > 0:
                for _, row in opt.iterrows():
                    sym = row.get("symbol")
                    if sym:
                        self._options[sym] = {
                            "pc_ratio": row.get("pc_ratio"),
                            "atm_iv": row.get("atm_iv"),
                        }

            # Ortex short interest (overrides/supplements _options dict)
            ortex_si = enhanced_data.get("ortex_si")
            if ortex_si is not None and len(ortex_si) > 0:
                for _, row in ortex_si.iterrows():
                    sym = row.get("ticker")
                    if sym:
                        if sym not in self._options:
                            self._options[sym] = {}
                        self._options[sym]["si_pct_float"] = row.get("siPctFreeFloat")
                        self._options[sym]["short_score"] = row.get("shortScore")
                        self._options[sym]["si_shares"] = row.get("siShares")

        # Load FINRA short volume features (2022+) for backtest-able SI signal
        finra_path = DATA_DIR / "finra_short_features.parquet"
        if finra_path.exists():
            finra = pd.read_parquet(finra_path)
            finra["date"] = pd.to_datetime(finra["date"])
            for date, grp in finra.groupby("date"):
                self._short_ratio[date] = dict(zip(grp["symbol"], grp["short_ratio_20d"]))
            log(f"    FINRA short data loaded: {len(self._short_ratio)} dates")

        log(f"    Index built in {time.time()-t0:.1f}s: {len(self._feat_by_date)} dates, "
            f"{len(self._ml_preds)} ML prediction dates, "
            f"{len(self._price_targets)} price targets, {len(self._dcf)} DCF values")

    def get_sp500(self, date):
        if date not in self._sp500_cache:
            self._sp500_cache[date] = get_sp500_on_date(date)
        return self._sp500_cache[date]

    def get_regime(self, date):
        regime = {"vix": 20, "vix_term_structure": 1.0, "spy_above_sma200": True}

        # VIX
        regime["vix"] = self._vix.get(date, 20)
        regime["vix_term_structure"] = self._vix_ts.get(date, 1.0)

        # SPY above 200-day SMA
        if "SPY" in self._close:
            spy = self._close["SPY"].loc[:date]
            if len(spy) >= 200:
                regime["spy_above_sma200"] = spy.iloc[-1] > spy.iloc[-200:].mean()

        return regime

    def get_feature_map(self, date, feature, members=None):
        """Get {symbol: value} for a feature on a date. O(1) per symbol."""
        fdate = self._feat_by_date.get(date, {})
        if members is None:
            return {sym: d[feature] for sym, d in fdate.items()
                    if feature in d and not np.isnan(d[feature])}
        return {sym: fdate[sym][feature] for sym in members
                if sym in fdate and feature in fdate[sym]
                and not np.isnan(fdate[sym][feature])}

    def get_insider_signal(self, symbol, date, lookback_days=90):
        """Net insider buying in last N days. Returns (net_shares, buy_ratio)."""
        df = self._insiders.get(symbol)
        if df is None or len(df) == 0:
            return 0, 0.0
        cutoff = date - pd.Timedelta(days=lookback_days)
        recent = df[(df["date"] >= cutoff) & (df["date"] <= date)]
        if len(recent) == 0:
            return 0, 0.0
        buys = recent[recent["is_buy"] == 1]
        sells = recent[recent["is_buy"] == 0] if "is_sell" not in recent.columns else recent[recent["is_sell"] == 1]
        buy_shares = buys["shares"].sum() if len(buys) > 0 else 0
        sell_shares = sells["shares"].sum() if len(sells) > 0 else 0
        total = buy_shares + sell_shares
        net = buy_shares - sell_shares
        ratio = buy_shares / total if total > 0 else 0.0
        return net, ratio

    def get_close_at(self, date, symbol):
        """Get close price for symbol on date."""
        s = self._close.get(symbol)
        if s is None:
            return None
        if date in s.index:
            v = s.loc[date]
            return v if not np.isnan(v) else None
        # Find nearest prior date
        prior = s.loc[:date]
        return prior.iloc[-1] if len(prior) > 0 else None

    def get_close_series(self, symbol, end_date, lookback):
        """Get close price series ending at date, lookback days."""
        s = self._close.get(symbol)
        if s is None:
            return pd.Series(dtype=float)
        trimmed = s.loc[:end_date]
        return trimmed.iloc[-lookback:] if len(trimmed) >= lookback else trimmed

    def get_drift_pct(self, symbol, date, lookback=63):
        """Get % positive return days over lookback. Vectorized."""
        if symbol not in self._daily_returns.columns:
            return 0.0
        rets = self._daily_returns[symbol].loc[:date].iloc[-lookback:]
        if len(rets) < 30:
            return 0.0
        return (rets > 0).mean()

    def get_ml_scores(self, date, members=None):
        """Get ML model scores for a date. {symbol: score}."""
        preds = self._ml_preds.get(date, {})
        if members:
            return {s: preds[s] for s in members if s in preds}
        return preds

    def get_short_ratios(self, date, members=None):
        """Get {symbol: short_ratio_20d} for a date. FINRA for backtest, Ortex for live."""
        # FINRA daily data (backtest: 2022+)
        sr = self._short_ratio.get(date, {})
        if sr and members:
            return {s: sr[s] for s in members if s in sr}
        if sr:
            return sr
        # Fallback: Ortex snapshot (live only — single date)
        if self._options:
            result = {}
            syms = members if members else self._options.keys()
            for s in syms:
                opt = self._options.get(s)
                if opt and opt.get("si_pct_float") is not None:
                    # Convert SI% to a 0-1 ratio for consistency with FINRA
                    result[s] = opt["si_pct_float"] / 100.0
            return result
        return {}

    def get_additions(self, date, lookback_days=4):
        """Get SP500 additions announced in last N calendar days."""
        result = []
        for lb in range(lookback_days):
            d = date - pd.Timedelta(days=lb)
            result.extend(self._additions.get(d, []))
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  Vectorized Strategy Implementations (no per-row loops)
# ══════════════════════════════════════════════════════════════════════════════

def strategy1_momentum_reversal(date, uni, day_idx, top_n=8, rebal_days=10,
                                ml_ranker=None, ml_blend_weight=0.4,
                                si_blend_weight=0.15):
    """Adaptive Momentum with consistency weighting + sector tilt. Top-8, 10d."""
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)
    if len(members) < 50:
        return {}

    regime = uni.get_regime(date)
    # Use breadth for stress detection (no VIX dependency)
    dist_sma50_all = uni.get_feature_map(date, "dist_sma50")
    if dist_sma50_all:
        mkt_breadth = sum(1 for v in dist_sma50_all.values() if v > 0) / max(len(dist_sma50_all), 1)
    else:
        mkt_breadth = 0.5
    stress = mkt_breadth < 0.30  # fewer than 30% of stocks above 50d SMA
    n = max(top_n // 2, 5) if stress else top_n

    # Multi-timeframe momentum + relative strength
    ret_20 = uni.get_feature_map(date, "ret_20d", members)
    ret_60 = uni.get_feature_map(date, "ret_60d", members)
    ret_126 = uni.get_feature_map(date, "ret_126d", members)
    ret_252 = uni.get_feature_map(date, "ret_252d", members)
    dist_sma50 = uni.get_feature_map(date, "dist_sma50", members)
    eps_surp = uni.get_feature_map(date, "eps_surprise_last", members)

    composite = {}
    for sym in members:
        rets = []
        for rd in [ret_20, ret_60, ret_126, ret_252]:
            v = rd.get(sym)
            if v is not None and not np.isnan(v):
                rets.append(v)
        if len(rets) < 2:
            continue

        # Momentum consistency: fraction of lookbacks positive
        consistency = sum(1 for r in rets if r > 0) / len(rets)
        avg_ret = np.mean(rets)
        score = avg_ret * (consistency ** 2)

        # Earnings surprise boost — strongest fundamental (+0.60% per 10d, IC=+0.052)
        es = eps_surp.get(sym)
        if es is not None and not np.isnan(es) and es > 0:
            score *= 1.15

        # ROE quality boost (+0.26% per 10d, IC=+0.020)
        roe_val = uni.get_feature_map(date, "roe").get(sym)
        if roe_val is not None and not np.isnan(roe_val) and roe_val > 0.15:
            score *= 1.05

        # Revenue/earnings acceleration (IC +0.017 and +0.016)
        fg = uni._fin_growth.get(sym)
        if fg:
            rg = fg.get("rev_growth")
            eg = fg.get("eps_growth")
            if rg is not None and not np.isnan(rg) and rg > 0.08:
                score *= 1.10  # growing revenue >8%
            if eg is not None and not np.isnan(eg) and eg > 0.10:
                score *= 1.10  # growing EPS >10%

        # Analyst target upside boost (from FMP price targets)
        pt = uni._price_targets.get(sym)
        if pt:
            px = uni.get_close_at(date, sym)
            if px and px > 0 and pt["target"] > 0:
                upside = (pt["target"] - px) / px
                if upside > 0.15:      # >15% upside target
                    score *= 1.10
                elif upside < -0.10:   # analysts think it's overvalued
                    score *= 0.90

        # DCF value boost (intrinsic value vs price)
        dcf_data = uni._dcf.get(sym)
        if dcf_data:
            px = uni.get_close_at(date, sym)
            if px and px > 0 and dcf_data["dcf"] > 0:
                dcf_upside = (dcf_data["dcf"] - px) / px
                if dcf_upside > 0.20:  # >20% undervalued by DCF
                    score *= 1.10

        # Trend filter: above 50d SMA
        d50 = dist_sma50.get(sym, 0)
        if d50 is not None and d50 > 0:
            composite[sym] = score

    # Bear: sector tilt
    spy_bull = regime.get("spy_above_sma200", True)
    if not spy_bull and composite:
        sec_rets = {}
        for etf in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLB","XLRE","XLU","XLC"]:
            close = uni.get_close_series(etf, date, 65)
            if len(close) >= 60:
                sec_rets[etf] = (close.iloc[-1] / close.iloc[-60]) - 1.0
        etf_to_sec = {"XLK":"Technology","XLF":"Financial Services","XLE":"Energy",
                      "XLV":"Healthcare","XLI":"Industrials","XLY":"Consumer Cyclical",
                      "XLP":"Consumer Defensive","XLB":"Basic Materials","XLRE":"Real Estate",
                      "XLU":"Utilities","XLC":"Communication Services"}
        if sec_rets:
            ranked = sorted(sec_rets, key=sec_rets.get, reverse=True)
            top_secs = {etf_to_sec.get(e, "") for e in ranked[:3]}
            bot_secs = {etf_to_sec.get(e, "") for e in ranked[-3:]}
            boosted = {}
            for sym, score in composite.items():
                sym_sec = uni.sector_map.get(sym, "")
                if sym_sec in top_secs:
                    boosted[sym] = score * 2.0
                elif sym_sec in bot_secs:
                    continue
                else:
                    boosted[sym] = score
            composite = boosted

    if not composite:
        return {}

    # Short interest (Ortex live only — FINRA backtest showed negligible impact)
    # Keep simple threshold boost for live: low SI → clean momentum, high SI → crash risk
    # This only fires in live production where Ortex data is loaded
    for sym in list(composite.keys()):
        ortex = uni._options.get(sym)
        if ortex and ortex.get("si_pct_float") is not None:
            si = ortex["si_pct_float"]
            if si < 2.0:
                composite[sym] *= 1.10
            elif si > 10.0:
                composite[sym] *= 0.85

    # ML blend: if ranker available, expand candidates and re-rank
    if ml_ranker is not None and ml_ranker.model is not None:
        # Expand to top-20 candidates for ML to re-rank
        candidate_n = min(20, len(composite))
        candidates = sorted(composite, key=composite.get, reverse=True)[:candidate_n]

        ml_scores = ml_ranker.predict(date, uni, candidates)

        if len(ml_scores) >= n:
            # Normalize both score sets to [0, 1] within candidates
            f_vals = [composite[s] for s in candidates if s in ml_scores]
            f_min, f_max = min(f_vals), max(f_vals)
            f_range = f_max - f_min if f_max > f_min else 1.0

            m_vals = list(ml_scores.values())
            m_min, m_max = min(m_vals), max(m_vals)
            m_range = m_max - m_min if m_max > m_min else 1.0

            blended = {}
            for s in candidates:
                if s in ml_scores:
                    f_norm = (composite[s] - f_min) / f_range
                    m_norm = (ml_scores[s] - m_min) / m_range
                    blended[s] = (1 - ml_blend_weight) * f_norm + ml_blend_weight * m_norm

            sorted_syms = sorted(blended, key=blended.get, reverse=True)[:n]
        else:
            sorted_syms = sorted(composite, key=composite.get, reverse=True)[:n]
    else:
        sorted_syms = sorted(composite, key=composite.get, reverse=True)[:n]

    # Signal-weighted: higher score = more capital (but capped at 2x equal weight)
    scores = [max(composite[s], 0.001) for s in sorted_syms]
    total = sum(scores)
    if total > 0:
        weights = {s: min(sc / total, 2.0 / len(sorted_syms)) for s, sc in zip(sorted_syms, scores)}
        # Renormalize
        wt = sum(weights.values())
        if wt > 0:
            weights = {s: w / wt for s, w in weights.items()}
        return weights
    return {s: 1.0 / len(sorted_syms) for s in sorted_syms}


def strategy2_drift_reversal(date, uni, day_idx, top_n=10, rebal_days=5):
    """Drift-Conditional Value-Reversal. Top-10, 5-day rebal."""
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)

    # Drift filter: >60% positive days over 63 days
    qualifiers = [s for s in members if uni.get_drift_pct(s, date, 63) >= 0.60]
    if len(qualifiers) < 5:
        return {}

    # Value: inverse price (z-scored)
    prices = {}
    for sym in qualifiers:
        px = uni.get_close_at(date, sym)
        if px and px > 0:
            prices[sym] = 1.0 / px

    # Reversal: negative of 10d return
    ret10 = uni.get_feature_map(date, "ret_10d")
    rev = {s: -ret10[s] for s in qualifiers if s in ret10}

    def zscore(d):
        if len(d) < 5:
            return {}
        vals = np.array(list(d.values()))
        mu, sig = vals.mean(), vals.std()
        return {s: (v - mu) / sig for s, v in d.items()} if sig > 1e-10 else {}

    z_val = zscore(prices)
    z_rev = zscore(rev)

    composite = {}
    for sym in qualifiers:
        zs = [z for z in [z_val.get(sym), z_rev.get(sym)] if z is not None]
        if zs:
            composite[sym] = np.mean(zs)

    if not composite:
        return {}

    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:top_n]
    w = 1.0 / len(sorted_syms)
    return {s: w for s in sorted_syms}


def strategy3_sector_rotation(date, uni, day_idx, rebal_days=21):
    """Sector Momentum Rotation. Top 4 sectors, monthly."""
    if day_idx % rebal_days != 0:
        return None

    regime = uni.get_regime(date)
    bear = not regime.get("spy_above_sma200", True)

    # Blended 3m + 6m sector momentum for more robust signal
    sector_rets = {}
    for etf in SECTOR_ETFS:
        close = uni.get_close_series(etf, date, 130)
        if len(close) >= 126:
            ret_6m = (close.iloc[-1] / close.iloc[-126]) - 1.0
            ret_3m = (close.iloc[-1] / close.iloc[-63]) - 1.0
            if not np.isnan(ret_6m) and not np.isnan(ret_3m):
                sector_rets[etf] = (ret_6m + ret_3m) / 2

    if len(sector_rets) < 3:
        return {}

    top4 = sorted(sector_rets, key=sector_rets.get, reverse=True)[:4]
    return {etf: 1.0 / len(top4) for etf in top4}


def strategy4_index_inclusion(date, uni, day_idx, active_trades):
    """Index Inclusion Arbitrage. Buy additions, hold 20 days."""
    # Expire old trades
    expired = [s for s, entry in active_trades.items()
               if (date - entry).days >= 30]
    for s in expired:
        del active_trades[s]

    # Check for new additions
    new = uni.get_additions(date, lookback_days=4)
    for sym in new:
        if sym in active_trades or len(active_trades) >= 5:
            continue
        if uni.get_close_at(date, sym) is not None:
            active_trades[sym] = date

    if not active_trades:
        return {}

    w = 1.0 / 5  # fixed 20% per slot
    return {s: w for s in active_trades}


def strategy5_lowvol_quality(date, uni, day_idx, top_n=10, rebal_days=10):
    """Low-Vol Quality + Momentum. Top-10, 10-day rebal. NO trend filter (defensive anchor)."""
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)

    vol60 = uni.get_feature_map(date, "vol_60d", members)
    gm = uni.get_feature_map(date, "gross_margin", members)
    dte = uni.get_feature_map(date, "debt_to_equity", members)
    ret_126 = uni.get_feature_map(date, "ret_126d", members)
    eps = uni.get_feature_map(date, "eps_surprise_last", members)

    inv_vol = {s: -v for s, v in vol60.items() if v > 0}
    inv_dte = {s: -v for s, v in dte.items() if v >= 0}
    mom_6m = {s: v for s, v in ret_126.items() if not np.isnan(v)}

    # EV/Revenue: lower = cheaper (value signal)
    inv_ev_rev = {}
    for s in members:
        ev_data = uni._ev.get(s)
        if ev_data:
            evr = ev_data.get("ev_to_rev")
            if evr is not None and not np.isnan(float(evr)) and float(evr) > 0:
                inv_ev_rev[s] = -float(evr)

    # Forward earnings yield: higher = cheaper vs estimates
    fwd_ey = {}
    for s in members:
        est = uni._estimates.get(s)
        if est and est.get("eps_avg"):
            px = uni.get_close_at(date, s)
            if px and px > 0 and est["eps_avg"] > 0:
                fwd_ey[s] = est["eps_avg"] / px

    def zscore(d):
        if len(d) < 20: return {}
        vals = np.array(list(d.values()))
        mu, sig = vals.mean(), vals.std()
        return {s: (v - mu) / sig for s, v in d.items()} if sig > 1e-10 else {}

    z1 = zscore(inv_vol)
    z2 = zscore(gm)
    z3 = zscore(inv_dte)
    z4 = zscore(mom_6m)
    z5 = zscore(inv_ev_rev)    # EV/Revenue value signal
    z6 = zscore(fwd_ey)        # Forward earnings yield

    composite = {}
    for sym in members:
        # Core factors: vol, quality, leverage, momentum
        # Extended: EV/Revenue, forward earnings yield (if data available)
        zs = [z for z in [z1.get(sym), z2.get(sym), z3.get(sym), z4.get(sym),
                           z5.get(sym), z6.get(sym)] if z is not None]
        if len(zs) >= 2:
            composite[sym] = np.mean(zs)

    if not composite: return {}
    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:top_n]
    return {s: 1.0 / len(sorted_syms) for s in sorted_syms}


# ══════════════════════════════════════════════════════════════════════════════
#  Backtester
# ══════════════════════════════════════════════════════════════════════════════

# Strategy config: (name, capital_pct, function)
# Dynamic allocation: shifts between strategies based on regime
# Bull: heavy momentum + sector
# Bear: heavy low-vol quality + sector (which goes to cash in bear)
STRATEGY_CONFIG_BULL = [
    ("s1_momentum", 0.90),
    ("s2_drift", 0.00),
    ("s3_sector", 0.05),
    ("s4_inclusion", 0.00),
    ("s5_lowvol", 0.05),
    ("s6_short", 0.00),
]

STRATEGY_CONFIG_BEAR = [
    ("s1_momentum", 0.10),   # sector-tilted picks
    ("s2_drift", 0.00),
    ("s3_sector", 0.20),     # sector momentum
    ("s4_inclusion", 0.00),
    ("s5_lowvol", 0.70),     # defensive anchor
    ("s6_short", 0.00),
]

# Use the same names so lookup works
STRATEGY_CONFIG = STRATEGY_CONFIG_BULL  # default, overridden at runtime

VIX_EXTREME = 40


def run_backtest(uni, start_date="2022-01-01", end_date="2025-12-31",
                 ml_ranker=None, ml_blend_weight=0.4, open_prices=None):
    """
    Run multi-strategy backtest.

    Args:
        open_prices: DataFrame with open prices (columns=symbols, index=dates).
                     When provided, new buys use next-day open instead of
                     same-day close (eliminates same-bar execution bias).
    """
    trading_dates = sorted(uni.prices.index)
    trading_dates = [d for d in trading_dates
                     if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

    if not trading_dates:
        return {"cagr": 0, "sharpe": 0, "max_dd": 0}

    # Pre-build next-date map and open price lookup for next-day-open execution
    _next_date = {}
    for idx in range(len(trading_dates) - 1):
        _next_date[idx] = trading_dates[idx + 1]

    _open_lookup = {}
    if open_prices is not None:
        dates_in_open = open_prices.index.intersection(trading_dates)
        if len(dates_in_open) > 0:
            _open_lookup = open_prices.loc[dates_in_open].to_dict(orient="index")

    cost_frac = COST_BPS / 10_000
    cash = INITIAL_CASH
    holdings = {}  # sym -> {shares, entry_px}
    port_values = []
    trade_count = 0
    total_costs = 0.0

    # Strategy state
    last_targets = {name: {} for name, _ in STRATEGY_CONFIG}
    s4_active = {}  # index inclusion active trades

    for day_idx, date in enumerate(trading_dates):
        # Get prices for today
        today_prices = {}
        for sym in set(list(holdings.keys()) + list(uni._close.keys())):
            px = uni.get_close_at(date, sym)
            if px is not None:
                today_prices[sym] = px

        # Compute strategy targets
        regime = uni.get_regime(date)
        vix = regime.get("vix", 20)

        # Pause strategies in extreme VIX
        paused = set()
        if vix > VIX_EXTREME:
            paused = {"s1_momentum", "s2_drift", "s4_inclusion"}

        t1 = strategy1_momentum_reversal(date, uni, day_idx,
                                             ml_ranker=ml_ranker, ml_blend_weight=ml_blend_weight)
        t2 = strategy2_drift_reversal(date, uni, day_idx)
        t3 = strategy3_sector_rotation(date, uni, day_idx)
        t4 = strategy4_index_inclusion(date, uni, day_idx, s4_active)
        t5 = strategy5_lowvol_quality(date, uni, day_idx)
        t6 = strategy6_bear_short(date, uni, day_idx)

        # Update targets (None = no rebalance, keep last)
        if t1 is not None:
            last_targets["s1_momentum"] = t1
        if t2 is not None:
            last_targets["s2_drift"] = t2
        if t3 is not None:
            last_targets["s3_sector"] = t3
        last_targets["s4_inclusion"] = t4 if t4 else {}
        if t5 is not None:
            last_targets["s5_lowvol"] = t5
        if t6 is not None:
            last_targets["s6_short"] = t6

        # Only rebalance when a major strategy triggers
        major_rebal = any(x is not None for x in [t1, t2, t3, t5, t6])
        s4_changed = t4 is not None and t4 != last_targets.get("s4_inclusion_prev", {})
        if t4:
            last_targets["s4_inclusion_prev"] = dict(t4)

        # Daily stop-loss: exit positions down > 15% from entry
        STOP_LOSS = -0.15
        for sym in list(holdings.keys()):
            px = today_prices.get(sym, holdings[sym]["entry_px"])
            ret = (px / holdings[sym]["entry_px"]) - 1.0 if holdings[sym]["entry_px"] > 0 else 0
            if ret < STOP_LOSS:
                cost = abs(holdings[sym]["shares"] * px) * cost_frac
                cash += holdings[sym]["shares"] * px - cost
                total_costs += cost
                trade_count += 1
                del holdings[sym]

        if not major_rebal and not s4_changed:
            # Mark to market only
            equity = cash
            for sym, h in holdings.items():
                px = today_prices.get(sym, h["entry_px"])
                equity += h["shares"] * px
            port_values.append((date, equity))
            continue

        # Dynamic regime: blend bull/bear configs based on breadth
        dist_sma50 = uni.get_feature_map(date, "dist_sma50")
        if dist_sma50:
            breadth = sum(1 for v in dist_sma50.values() if v > 0) / max(len(dist_sma50), 1)
        else:
            breadth = 0.5
        # Blend factor: 0 = full bear config, 1 = full bull config
        blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))

        # Crypto risk-on boost: if BTC trending up strongly, tilt more bullish
        cfx = uni._crypto_fx.get(date, {})
        btc_ret = cfx.get("btc_ret_20d", 0)
        if btc_ret and not np.isnan(btc_ret) and btc_ret > 0.15:
            blend = min(1.0, blend + 0.10)  # slight bull tilt when crypto risk-on

        # Blended capital allocations
        blended_config = {}
        for name, bull_pct in STRATEGY_CONFIG_BULL:
            bear_pct = dict(STRATEGY_CONFIG_BEAR).get(name, 0)
            blended_config[name] = bull_pct * blend + bear_pct * (1 - blend)

        # Combine strategy targets into portfolio
        combined = {}
        for name, cap_pct in blended_config.items():
            targets = last_targets.get(name, {})
            if name in paused:
                continue
            for sym, w in targets.items():
                scaled = w * cap_pct
                combined[sym] = combined.get(sym, 0) + scaled

        # Separate longs and shorts
        longs = {s: w for s, w in combined.items() if w > 0}
        shorts = {s: w for s, w in combined.items() if w < 0}

        # Apply constraints to longs
        for sym in list(longs):
            if longs[sym] > 0.15:
                longs[sym] = 0.15

        sector_totals = {}
        for sym, w in longs.items():
            sec = uni.sector_map.get(sym, "X")
            sector_totals[sec] = sector_totals.get(sec, 0) + w
        for sec, total in sector_totals.items():
            if total > 0.35:
                scale = 0.35 / total
                for sym in list(longs):
                    if uni.sector_map.get(sym, "X") == sec:
                        longs[sym] *= scale

        gross_long = sum(longs.values())
        if gross_long > 1.0:
            for sym in longs:
                longs[sym] /= gross_long

        # Cap short exposure at 20%
        gross_short = sum(abs(w) for w in shorts.values())
        if gross_short > 0.20:
            scale = 0.20 / gross_short
            shorts = {s: w * scale for s, w in shorts.items()}

        # Merge
        combined = {s: w for s, w in longs.items() if w >= 0.005}
        combined.update({s: w for s, w in shorts.items() if abs(w) >= 0.005})

        # Current equity
        equity = cash
        for sym, h in holdings.items():
            px = today_prices.get(sym, h["entry_px"])
            equity += h["shares"] * px

        # Differential rebalance: only trade the changes
        target_dollars = {sym: w * equity for sym, w in combined.items() if w > 0}
        current_dollars = {}
        for sym, h in holdings.items():
            px = today_prices.get(sym, h["entry_px"])
            current_dollars[sym] = h["shares"] * px

        # Sell positions no longer wanted
        for sym in list(holdings.keys()):
            if sym not in target_dollars:
                px = today_prices.get(sym, holdings[sym]["entry_px"])
                proceeds = holdings[sym]["shares"] * px
                cost = abs(proceeds) * cost_frac
                cash += proceeds - cost
                total_costs += cost
                trade_count += 1
                del holdings[sym]

        # Adjust existing and buy new.
        # For NEW buys, use next-day open if available (avoids same-bar
        # execution bias). Sells and size adjustments use today's close.
        next_d = _next_date.get(day_idx)
        next_open = _open_lookup.get(next_d, {}) if next_d else {}

        for sym, target_val in target_dollars.items():
            px = today_prices.get(sym)
            if px is None or px <= 0:
                continue
            current_val = current_dollars.get(sym, 0)
            delta = target_val - current_val
            if abs(delta) < equity * 0.005:
                continue
            cost = abs(delta) * cost_frac
            total_costs += cost
            trade_count += 1
            if delta > 0 and cash >= delta:
                # Use next-day open for new buys if available
                if sym not in holdings and next_open:
                    buy_px = next_open.get(sym, px)
                    if buy_px is None or np.isnan(buy_px) or buy_px <= 0:
                        buy_px = px
                else:
                    buy_px = px
                new_shares = (delta - cost) / buy_px
                if sym in holdings:
                    holdings[sym]["shares"] += new_shares
                else:
                    holdings[sym] = {"shares": new_shares, "entry_px": buy_px}
                cash -= delta
            elif delta < 0 and sym in holdings:
                sell_shares = min(abs(delta) / px, holdings[sym]["shares"])
                cash += sell_shares * px - cost
                holdings[sym]["shares"] -= sell_shares
                if holdings[sym]["shares"] < 0.01:
                    del holdings[sym]

        # Mark to market
        equity = cash
        for sym, h in holdings.items():
            px = today_prices.get(sym, h["entry_px"])
            equity += h["shares"] * px

        port_values.append((date, max(equity, 0)))

    # Metrics
    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0:
        years = 1.0
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd_s = dr[dr < 0].std()
    sortino = dr.mean() / dd_s * np.sqrt(252) if dd_s > 0 else np.nan
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()

    # SPY comparison
    spy = uni.prices["SPY"].reindex(vals.index, method="ffill").dropna()
    spy = spy / spy.iloc[0] * INITIAL_CASH
    spy_cagr = (spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1
    spy_dr = spy.pct_change().dropna()
    spy_sharpe = spy_dr.mean() / spy_dr.std() * np.sqrt(252) if spy_dr.std() > 0 else 0
    spy_dd = ((spy - spy.cummax()) / spy.cummax()).min()

    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "vol": dr.std() * np.sqrt(252),
        "trades": trade_count, "total_costs": total_costs,
        "final": vals.iloc[-1], "years": years,
        "spy_cagr": spy_cagr, "spy_sharpe": spy_sharpe, "spy_dd": spy_dd,
    }


def main():
    t0 = time.perf_counter()
    log("=" * 80)
    log("  MULTI-STRATEGY ENGINE v8 (optimized)")
    log("=" * 80)

    # Load data
    log("\n1. Loading data ...")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])

    provider = MassiveDataProvider(validate_vs_yfinance=False)
    all_syms = sorted(features["symbol"].unique().tolist())
    bars = provider.fetch_bars_batch(list(set(all_syms + ["SPY"])), warmup_days=3800)
    close_frames = {sym: df["close"] for sym, df in bars.items() if len(df) > 0}
    prices = pd.DataFrame(close_frames)
    prices.index = pd.to_datetime(prices.index)

    # Build open prices for next-day-open execution (Fix 1: same-bar bias)
    open_frames = {sym: df["open"] for sym, df in bars.items() if len(df) > 0}
    open_prices = pd.DataFrame(open_frames)
    open_prices.index = pd.to_datetime(open_prices.index)

    sector_file = DATA_DIR / "cache_sectors.json"
    with open(sector_file) as f:
        sector_map = json.load(f)

    changes_file = DATA_DIR / "sp500_changes.json"
    with open(changes_file) as f:
        sp500_changes = json.load(f)

    # Load cached VIX data for deterministic results
    vix_cache = DATA_DIR / "enhanced_data" / "vix_cache.parquet"
    vix_data = None
    if vix_cache.exists():
        vix_data = pd.read_parquet(vix_cache)
        vix_data.index = pd.to_datetime(vix_data.index)
        log(f"  VIX data: {len(vix_data)} rows (cached)")
    else:
        try:
            vix_raw = yf.download(["^VIX", "^VIX3M"], start="2016-01-01",
                                   end="2027-01-01", progress=False, auto_adjust=True)
            vix_data = vix_raw["Close"]
            vix_data.index = pd.to_datetime(vix_data.index).tz_localize(None)
        except Exception:
            pass

    # Load ML predictions if available
    ml_preds = None
    wf_file = DATA_DIR / "walkforward" / "predictions_walkforward_all.parquet"
    if wf_file.exists():
        ml_preds = pd.read_parquet(wf_file)
        ml_preds["date"] = pd.to_datetime(ml_preds["date"])
        log(f"  ML predictions: {len(ml_preds):,} rows")

    # Load enhanced data
    enhanced_data = {}
    enhanced_dir = DATA_DIR / "enhanced_data"
    if enhanced_dir.exists():
        for fname, key in [("price_targets.parquet", "price_targets"),
                            ("dcf_values.parquet", "dcf"),
                            ("financial_growth.parquet", "financial_growth"),
                            ("enterprise_values.parquet", "enterprise_values"),
                            ("company_profiles.parquet", "profiles"),
                            ("crypto_forex_extended.parquet", "crypto_forex")]:
            fpath = enhanced_dir / fname
            if fpath.exists():
                enhanced_data[key] = pd.read_parquet(fpath)
                if "date" in enhanced_data[key].columns:
                    enhanced_data[key]["date"] = pd.to_datetime(enhanced_data[key]["date"])
        if enhanced_data:
            log(f"  Enhanced data: {', '.join(enhanced_data.keys())}")

    uni = FastUniverse(features, prices, sector_map, sp500_changes, vix_data,
                       ml_predictions=ml_preds, enhanced_data=enhanced_data)

    # Full period
    log("\n2. Running full-period backtest (2022-2025) ...")
    full = run_backtest(uni, "2022-01-01", "2025-12-31", open_prices=open_prices)

    log(f"\n{'='*80}")
    log("  FULL PERIOD RESULTS")
    log(f"{'='*80}")
    log(f"  Portfolio: CAGR={full['cagr']:+.1%}  Sharpe={full['sharpe']:.2f}  "
        f"Sortino={full['sortino']:.2f}  DD={full['max_dd']:.1%}  "
        f"Vol={full['vol']:.1%}  Trades={full['trades']}  $100K->${full['final']:,.0f}")
    log(f"  SPY:       CAGR={full['spy_cagr']:+.1%}  Sharpe={full['spy_sharpe']:.2f}  "
        f"DD={full['spy_dd']:.1%}")
    log(f"  Alpha:     {full['cagr'] - full['spy_cagr']:+.1%}")

    # Per-year
    log(f"\n{'='*80}")
    log("  PER-YEAR BREAKDOWN")
    log(f"{'='*80}")
    log(f"  {'Year':<6} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7} {'vs SPY':>8}")
    log(f"  {'─'*6} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*8}")

    yearly = []
    for year in range(2022, 2026):
        m = run_backtest(uni, f"{year}-01-01", f"{year}-12-31", open_prices=open_prices)
        m["year"] = year
        yearly.append(m)
        alpha = m["cagr"] - m["spy_cagr"]
        log(f"  {year:<6} {m['cagr']:>+7.1%} {m['sharpe']:>7.2f} "
            f"{m['max_dd']:>6.1%} {m['trades']:>7} {alpha:>+7.1%}")

    med_cagr = np.median([m["cagr"] for m in yearly])
    med_sharpe = np.median([m["sharpe"] for m in yearly])
    pos = sum(1 for m in yearly if m["cagr"] > 0)
    log(f"\n  Median CAGR: {med_cagr:+.1%}  Median Sharpe: {med_sharpe:.2f}  "
        f"Positive: {pos}/{len(yearly)}")

    elapsed = time.perf_counter() - t0
    log(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
