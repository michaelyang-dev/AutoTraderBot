"""
Multi-Strategy Backtesting Engine v8 (optimized)
=================================================
Runs all 5 strategies combined with realistic costs.
Pre-indexes all data for O(1) lookups instead of DataFrame scans.

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
                 ml_predictions=None):
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

        log(f"    Index built in {time.time()-t0:.1f}s: {len(self._feat_by_date)} dates, "
            f"{len(self._ml_preds)} ML prediction dates")

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

def strategy1_momentum_reversal(date, uni, day_idx, top_n=10, rebal_days=10):
    """Momentum + Reversal + Trend + ML Ensemble. Top-10, 10-day rebal."""
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)
    if len(members) < 50:
        return {}

    regime = uni.get_regime(date)
    vix = regime.get("vix", 20)
    vix_ts = regime.get("vix_term_structure", 1.0)
    spy_bull = regime.get("spy_above_sma200", True)
    # Also check SPY vs 50d SMA for faster bear detection
    dist_sma50_spy = uni.get_feature_map(date, "dist_sma50").get("SPY", 0)
    fast_bear = dist_sma50_spy is not None and dist_sma50_spy < -0.03  # SPY >3% below 50d SMA
    stress = vix > 30 or vix_ts < 0.95
    bear = not spy_bull or fast_bear
    n = max(top_n // 2, 5) if stress else top_n

    # Signals
    ret_252 = uni.get_feature_map(date, "ret_252d", members)
    ret_126 = uni.get_feature_map(date, "ret_126d", members)
    ret_60 = uni.get_feature_map(date, "ret_60d", members)
    ret_20 = uni.get_feature_map(date, "ret_20d", members)
    ret_5d = uni.get_feature_map(date, "ret_5d", members)
    eps_surp = uni.get_feature_map(date, "eps_surprise_last", members)
    gross_m = uni.get_feature_map(date, "gross_margin", members)

    # 12-1 momentum
    mom = {s: ret_252[s] - ret_20[s] for s in members
           if s in ret_252 and s in ret_20}

    # Momentum acceleration: 3m return - 6m return (rising momentum)
    mom_accel = {}
    for s in members:
        r60 = ret_60.get(s)
        r126 = ret_126.get(s)
        if r60 is not None and r126 is not None and not np.isnan(r60) and not np.isnan(r126):
            mom_accel[s] = r60 - (r126 - r60)  # recent 3m vs prior 3m

    # 5d reversal
    rev = {s: -ret_5d[s] for s in ret_5d}

    # Z-score and combine
    def zscore(d):
        if len(d) < 10:
            return {}
        vals = np.array(list(d.values()))
        mu, sig = vals.mean(), vals.std()
        if sig < 1e-10:
            return {}
        return {s: (v - mu) / sig for s, v in d.items()}

    z_mom = zscore(mom)
    z_accel = zscore(mom_accel)
    z_rev = zscore(rev)
    z_earn = zscore(eps_surp)
    z_qual = zscore(gross_m)

    # ML model scores (from LambdaRank walk-forward predictions)
    ml_scores = uni.get_ml_scores(date, members)
    z_ml = zscore(ml_scores)

    # In bear: add sector momentum signal (stocks in top sectors get a boost)
    z_secmom = {}
    if bear:
        sec_rets = {}
        for etf in ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLB","XLRE","XLU","XLC"]:
            close = uni.get_close_series(etf, date, 65)
            if len(close) >= 60:
                sec_rets[etf] = (close.iloc[-1] / close.iloc[-60]) - 1.0
        # Map ETF to sector name for matching
        etf_to_sec = {"XLK":"Technology","XLF":"Financial Services","XLE":"Energy",
                      "XLV":"Healthcare","XLI":"Industrials","XLY":"Consumer Cyclical",
                      "XLP":"Consumer Defensive","XLB":"Basic Materials","XLRE":"Real Estate",
                      "XLU":"Utilities","XLC":"Communication Services"}
        if sec_rets:
            best_etfs = sorted(sec_rets, key=sec_rets.get, reverse=True)[:4]
            best_secs = {etf_to_sec.get(e, "") for e in best_etfs}
            for sym in members:
                sym_sec = uni.sector_map.get(sym, "")
                z_secmom[sym] = 1.0 if sym_sec in best_secs else -0.5

    composite = {}
    for sym in members:
        if bear:
            # Bear: momentum + quality + sector momentum
            zs = [z for z in [z_mom.get(sym), z_qual.get(sym),
                               z_earn.get(sym), z_secmom.get(sym)] if z is not None]
        else:
            # Bull: momentum + reversal + earnings + quality
            zs = [z for z in [z_mom.get(sym), z_rev.get(sym),
                               z_earn.get(sym), z_qual.get(sym)] if z is not None]
        if len(zs) >= 2:
            composite[sym] = np.mean(zs)

    if not composite:
        return {}

    # Trend filter: only include stocks above their 50-day SMA
    if not bear:
        trend_filtered = {}
        dist_sma50 = uni.get_feature_map(date, "dist_sma50", members)
        for sym in composite:
            d50 = dist_sma50.get(sym, 0)
            if d50 is not None and d50 > 0:  # above 50d SMA
                trend_filtered[sym] = composite[sym]
        if len(trend_filtered) >= 5:
            composite = trend_filtered

    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:n]
    w = 1.0 / len(sorted_syms)
    return {s: w for s in sorted_syms}


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
    """Sector Momentum Rotation. Top 3 sectors, monthly."""
    if day_idx % rebal_days != 0:
        return None

    regime = uni.get_regime(date)
    bear = not regime.get("spy_above_sma200", True)

    # In bear: still use momentum ranking but favor defensive sectors
    sector_rets = {}
    for etf in SECTOR_ETFS:
        close = uni.get_close_series(etf, date, 130)
        if len(close) >= 126:
            ret = (close.iloc[-1] / close.iloc[-126]) - 1.0
            if not np.isnan(ret):
                sector_rets[etf] = ret

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


def strategy5_lowvol_quality(date, uni, day_idx, top_n=15, rebal_days=10):
    """Low-Vol Quality + Momentum. Top-15, 10-day rebal."""
    if day_idx % rebal_days != 0:
        return None
    members = uni.get_sp500(date)

    vol60 = uni.get_feature_map(date, "vol_60d", members)
    gm = uni.get_feature_map(date, "gross_margin", members)
    dte = uni.get_feature_map(date, "debt_to_equity", members)

    # Add 6-month momentum to quality screen
    ret_126 = uni.get_feature_map(date, "ret_126d", members)

    inv_vol = {s: -v for s, v in vol60.items() if v > 0}
    inv_dte = {s: -v for s, v in dte.items() if v >= 0}
    mom_6m = {s: v for s, v in ret_126.items() if not np.isnan(v)}

    def zscore(d):
        if len(d) < 20:
            return {}
        vals = np.array(list(d.values()))
        mu, sig = vals.mean(), vals.std()
        return {s: (v - mu) / sig for s, v in d.items()} if sig > 1e-10 else {}

    z1 = zscore(inv_vol)
    z2 = zscore(gm)
    z3 = zscore(inv_dte)
    z4 = zscore(mom_6m)

    composite = {}
    for sym in members:
        zs = [z for z in [z1.get(sym), z2.get(sym), z3.get(sym), z4.get(sym)] if z is not None]
        if len(zs) >= 2:
            composite[sym] = np.mean(zs)

    if not composite:
        return {}

    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:top_n]
    w = 1.0 / len(sorted_syms)
    return {s: w for s in sorted_syms}


# ══════════════════════════════════════════════════════════════════════════════
#  Backtester
# ══════════════════════════════════════════════════════════════════════════════

# Strategy config: (name, capital_pct, function)
# Dynamic allocation: shifts between strategies based on regime
# Bull: heavy momentum + sector
# Bear: heavy low-vol quality + sector (which goes to cash in bear)
STRATEGY_CONFIG_BULL = [
    ("s1_momentum", 0.75),
    ("s2_drift", 0.00),
    ("s3_sector", 0.15),
    ("s4_inclusion", 0.00),
    ("s5_lowvol", 0.10),
]

STRATEGY_CONFIG_BEAR = [
    ("s1_momentum", 0.10),   # minimal, reversal-mode
    ("s2_drift", 0.00),
    ("s3_sector", 0.20),     # defensive sectors in bear
    ("s4_inclusion", 0.00),
    ("s5_lowvol", 0.70),     # defensive anchor
]

# Use the same names so lookup works
STRATEGY_CONFIG = STRATEGY_CONFIG_BULL  # default, overridden at runtime

VIX_EXTREME = 40


def run_backtest(uni, start_date="2022-01-01", end_date="2025-12-31"):
    trading_dates = sorted(uni.prices.index)
    trading_dates = [d for d in trading_dates
                     if pd.Timestamp(start_date) <= d <= pd.Timestamp(end_date)]

    if not trading_dates:
        return {"cagr": 0, "sharpe": 0, "max_dd": 0}

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

        t1 = strategy1_momentum_reversal(date, uni, day_idx)
        t2 = strategy2_drift_reversal(date, uni, day_idx)
        t3 = strategy3_sector_rotation(date, uni, day_idx)
        t4 = strategy4_index_inclusion(date, uni, day_idx, s4_active)
        t5 = strategy5_lowvol_quality(date, uni, day_idx)

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

        # Only rebalance when a major strategy triggers (not S4 daily check)
        major_rebal = any(x is not None for x in [t1, t2, t3, t5])
        s4_changed = t4 is not None and t4 != last_targets.get("s4_inclusion_prev", {})
        if t4:
            last_targets["s4_inclusion_prev"] = dict(t4)

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

        # Apply constraints
        for sym in list(combined):
            if combined[sym] > 0.15:
                combined[sym] = 0.15

        sector_totals = {}
        for sym, w in combined.items():
            sec = uni.sector_map.get(sym, "X")
            sector_totals[sec] = sector_totals.get(sec, 0) + w
        for sec, total in sector_totals.items():
            if total > 0.35:
                scale = 0.35 / total
                for sym in list(combined):
                    if uni.sector_map.get(sym, "X") == sec:
                        combined[sym] *= scale

        gross = sum(combined.values())
        if gross > 1.0:
            for sym in combined:
                combined[sym] /= gross

        combined = {s: w for s, w in combined.items() if w >= 0.005}

        # Current equity
        equity = cash
        for sym, h in holdings.items():
            px = today_prices.get(sym, h["entry_px"])
            equity += h["shares"] * px

        # Differential rebalance: only trade the changes
        target_dollars = {sym: w * equity for sym, w in combined.items()}
        current_dollars = {}
        for sym, h in holdings.items():
            px = today_prices.get(sym, h["entry_px"])
            current_dollars[sym] = h["shares"] * px

        # Sell positions no longer wanted
        for sym in list(holdings.keys()):
            if sym not in target_dollars:
                px = today_prices.get(sym, holdings[sym]["entry_px"])
                proceeds = holdings[sym]["shares"] * px
                cost = proceeds * cost_frac
                cash += proceeds - cost
                total_costs += cost
                trade_count += 1
                del holdings[sym]

        # Adjust existing and buy new
        for sym, target_val in target_dollars.items():
            px = today_prices.get(sym)
            if px is None or px <= 0:
                continue

            current_val = current_dollars.get(sym, 0)
            delta = target_val - current_val

            # Only trade if change is >0.5% of equity (reduce churn)
            if abs(delta) < equity * 0.005:
                continue

            cost = abs(delta) * cost_frac
            total_costs += cost
            trade_count += 1

            if delta > 0:
                # Buy more
                buy_amount = delta - cost
                if buy_amount > 0 and cash >= delta:
                    new_shares = buy_amount / px
                    if sym in holdings:
                        holdings[sym]["shares"] += new_shares
                    else:
                        holdings[sym] = {"shares": new_shares, "entry_px": px}
                    cash -= delta
            else:
                # Sell some
                if sym in holdings:
                    sell_val = abs(delta)
                    sell_shares = min(sell_val / px, holdings[sym]["shares"])
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

    sector_file = DATA_DIR / "cache_sectors.json"
    with open(sector_file) as f:
        sector_map = json.load(f)

    changes_file = DATA_DIR / "sp500_changes.json"
    with open(changes_file) as f:
        sp500_changes = json.load(f)

    vix_data = None
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

    uni = FastUniverse(features, prices, sector_map, sp500_changes, vix_data,
                       ml_predictions=ml_preds)

    # Full period
    log("\n2. Running full-period backtest (2022-2025) ...")
    full = run_backtest(uni, "2022-01-01", "2025-12-31")

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
        m = run_backtest(uni, f"{year}-01-01", f"{year}-12-31")
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
