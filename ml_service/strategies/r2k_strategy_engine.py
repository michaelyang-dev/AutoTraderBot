"""
Russell 2000 Small-Cap Strategy Engine
======================================
Independent factor strategy for small-cap stocks ($300M-$2B market cap).
Runs alongside the SP500 v9.5 strategy on separate capital.

Strategies:
  S1: Adaptive Momentum (70% bull / 20% bear) — consistency-weighted, top-15
  S5: Low-Vol Quality (30% bull / 80% bear) — defensive z-score composite

Key differences from SP500 v9.5:
  - 15 positions (vs 8) — lower concentration for higher single-stock risk
  - 15 bps transaction costs (vs 5 bps) — wider spreads
  - -20% stop-loss (vs -15%) — more volatile stocks
  - 8% single-name cap (vs 15%)
  - $5 minimum price, 200K min daily volume — liquidity filter

Usage:
    cd ml_service && python3 -m strategies.r2k_strategy_engine
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger("r2k")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── R2K-specific parameters ─────────────────────────────────────────────────
INITIAL_CASH = 100_000
COST_BPS = 15           # 15 bps round-trip (wider spreads for small-caps)
STOP_LOSS = -0.20       # -20% stop (small-caps more volatile)
MIN_PRICE = 5.0         # skip penny stocks
MIN_VOLUME_20D = 200_000  # skip illiquid names

R2K_CONFIG_BULL = [
    ("r2k_s1_momentum", 0.70),
    ("r2k_s5_lowvol",   0.30),
]
R2K_CONFIG_BEAR = [
    ("r2k_s1_momentum", 0.20),
    ("r2k_s5_lowvol",   0.80),
]


# ── R2K Universe ─────────────────────────────────────────────────────────────

class R2KUniverse:
    """Pre-indexed small-cap universe data for fast strategy computation."""

    def __init__(self, features_df, prices_df, universe_df, sector_map=None):
        t0 = time.time()
        print("  Building R2K universe index ...", flush=True)

        self.sector_map = sector_map or {}

        # Pre-index features: {date: {symbol: {feature: value}}}
        self._feat_by_date = {}
        for date, grp in features_df.groupby("date"):
            sym_dict = {}
            for _, row in grp.iterrows():
                sym = row["symbol"]
                sym_dict[sym] = {col: row[col] for col in grp.columns
                                 if col not in ("date", "symbol")}
            self._feat_by_date[date] = sym_dict

        # Pre-compute close prices: {symbol: Series}
        self._close = {}
        for col in prices_df.columns:
            s = prices_df[col].dropna()
            if len(s) > 0:
                self._close[col] = s

        # Universe membership: {date: set(symbols)}
        self._universe_by_date = {}
        universe_df["date"] = pd.to_datetime(universe_df["date"])
        for date, grp in universe_df.groupby("date"):
            self._universe_by_date[date] = set(grp["symbol"].tolist())

        elapsed = time.time() - t0
        print(f"    R2K index built in {elapsed:.1f}s: {len(self._feat_by_date)} dates, "
              f"{len(self._close)} price series", flush=True)

    def get_universe(self, date):
        """Get R2K stocks on a given date."""
        # Find nearest date
        if date in self._universe_by_date:
            return self._universe_by_date[date]
        # Find closest prior date
        prior = [d for d in sorted(self._universe_by_date.keys()) if d <= date]
        if prior:
            return self._universe_by_date[prior[-1]]
        return set()

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
        prior = s.loc[:date]
        return prior.iloc[-1] if len(prior) > 0 else None

    def get_close_series(self, symbol, end_date, lookback):
        """Get close price series ending at date, lookback days."""
        s = self._close.get(symbol)
        if s is None:
            return pd.Series(dtype=float)
        return s.loc[:end_date].iloc[-lookback:]

    def get_regime(self, date):
        """Get market regime from SPY (same as SP500 strategy)."""
        regime = {"vix": 20}
        if "SPY" in self._close:
            spy = self._close["SPY"].loc[:date]
            if len(spy) >= 200:
                regime["spy_above_sma200"] = spy.iloc[-1] > spy.iloc[-200:].mean()
        return regime


# ── R2K Strategy 1: Momentum ────────────────────────────────────────────────

def r2k_strategy1_momentum(date, uni, day_idx, top_n=15, rebal_days=10):
    """Adaptive Momentum for small-caps. Top-15, 10-day rebal."""
    if day_idx % rebal_days != 0:
        return None

    members = uni.get_universe(date)
    if len(members) < 50:
        return {}

    regime = uni.get_regime(date)

    # Market breadth for stress detection
    dist_sma50_all = uni.get_feature_map(date, "dist_sma50")
    if dist_sma50_all:
        mkt_breadth = sum(1 for v in dist_sma50_all.values() if v > 0) / max(len(dist_sma50_all), 1)
    else:
        mkt_breadth = 0.5
    stress = mkt_breadth < 0.30
    n = max(top_n // 2, 8) if stress else top_n

    # Feature lookups
    ret_20 = uni.get_feature_map(date, "ret_20d", members)
    ret_60 = uni.get_feature_map(date, "ret_60d", members)
    ret_120 = uni.get_feature_map(date, "ret_120d", members)
    gross_m = uni.get_feature_map(date, "gross_margin", members)
    dist_sma50 = uni.get_feature_map(date, "dist_sma50", members)
    eps_surp = uni.get_feature_map(date, "eps_surprise_last", members)
    dte_map = uni.get_feature_map(date, "debt_to_equity", members)

    # Use ret_120d as proxy for ret_126d (close enough for small-caps)
    composite = {}
    for sym in members:
        # Liquidity filter
        px = uni.get_close_at(date, sym)
        if px is None or px < MIN_PRICE:
            continue

        # Multi-timeframe momentum
        rets = []
        for rd in [ret_20, ret_60, ret_120]:
            v = rd.get(sym)
            if v is not None and not np.isnan(v):
                rets.append(v)
        if len(rets) < 2:
            continue

        consistency = sum(1 for r in rets if r > 0) / len(rets)
        avg_ret = np.mean(rets)
        score = avg_ret * (consistency ** 2)

        # Quality boost
        gm = gross_m.get(sym)
        if gm is not None and not np.isnan(gm) and gm > 0.30:
            score *= 1.15

        # Earnings surprise
        es = eps_surp.get(sym)
        if es is not None and not np.isnan(es) and es > 0:
            score *= 1.15

        # Leverage penalty (tighter for small-caps)
        dte_val = dte_map.get(sym)
        if dte_val is not None and not np.isnan(dte_val) and dte_val > 2.5:
            score *= 0.85

        # Trend filter: above 50d SMA
        d50 = dist_sma50.get(sym, 0)
        if d50 is not None and d50 > 0:
            composite[sym] = score

    if not composite:
        return {}

    # Bear market: sector tilt (if sector data available)
    spy_bull = regime.get("spy_above_sma200", True)
    if not spy_bull and composite:
        # In bear, tighten to only strongest momentum
        n = max(n // 2, 5)

    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:n]

    # Signal-weighted sizing
    scores = [max(composite[s], 0.001) for s in sorted_syms]
    total = sum(scores)
    if total > 0:
        weights = {s: min(sc / total, 2.0 / len(sorted_syms))
                   for s, sc in zip(sorted_syms, scores)}
        wt = sum(weights.values())
        if wt > 0:
            weights = {s: w / wt for s, w in weights.items()}
        return weights
    return {s: 1.0 / len(sorted_syms) for s in sorted_syms}


# ── R2K Strategy 5: Low-Vol Quality ─────────────────────────────────────────

def r2k_strategy5_lowvol(date, uni, day_idx, top_n=10, rebal_days=10):
    """Low-Vol Quality for small-caps. Top-10, 10-day rebal."""
    if day_idx % rebal_days != 0:
        return None

    members = uni.get_universe(date)

    vol60 = uni.get_feature_map(date, "vol_60d", members)
    gm = uni.get_feature_map(date, "gross_margin", members)
    dte = uni.get_feature_map(date, "debt_to_equity", members)
    ret_120 = uni.get_feature_map(date, "ret_120d", members)

    inv_vol = {s: -v for s, v in vol60.items() if v > 0}
    inv_dte = {s: -v for s, v in dte.items() if v >= 0}
    mom = {s: v for s, v in ret_120.items() if not np.isnan(v)}

    def zscore(d):
        if len(d) < 20:
            return {}
        vals = np.array(list(d.values()))
        mu, sig = vals.mean(), vals.std()
        return {s: (v - mu) / sig for s, v in d.items()} if sig > 1e-10 else {}

    z1 = zscore(inv_vol)
    z2 = zscore(gm)
    z3 = zscore(inv_dte)
    z4 = zscore(mom)

    composite = {}
    for sym in members:
        # Liquidity filter
        px = uni.get_close_at(date, sym)
        if px is None or px < MIN_PRICE:
            continue

        zs = [z for z in [z1.get(sym), z2.get(sym), z3.get(sym), z4.get(sym)]
              if z is not None]
        if len(zs) >= 2:
            composite[sym] = np.mean(zs)

    if not composite:
        return {}
    sorted_syms = sorted(composite, key=composite.get, reverse=True)[:top_n]
    return {s: 1.0 / len(sorted_syms) for s in sorted_syms}


# ── R2K Backtester ──────────────────────────────────────────────────────────

def run_r2k_backtest(uni, start, end):
    """Run R2K strategy backtest."""
    trading_dates = [d for d in sorted(uni._feat_by_date.keys())
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

    cost_frac = COST_BPS / 10000
    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    last_targets = {}
    trade_count = 0

    for day_idx, date in enumerate(trading_dates):
        today_prices = {}
        for sym in list(holdings.keys()):
            px = uni.get_close_at(date, sym)
            if px is not None:
                today_prices[sym] = px

        # Get prices for universe
        members = uni.get_universe(date)
        for sym in list(members)[:700]:
            px = uni.get_close_at(date, sym)
            if px is not None:
                today_prices[sym] = px

        # Run strategies
        t1 = r2k_strategy1_momentum(date, uni, day_idx)
        t5 = r2k_strategy5_lowvol(date, uni, day_idx)

        if t1 is not None:
            last_targets["r2k_s1_momentum"] = t1
        if t5 is not None:
            last_targets["r2k_s5_lowvol"] = t5

        major_rebal = any(x is not None for x in [t1, t5])

        # Daily stop-loss
        for sym in list(holdings.keys()):
            px = today_prices.get(sym, holdings[sym]["entry_px"])
            ret = (px / holdings[sym]["entry_px"]) - 1.0 if holdings[sym]["entry_px"] > 0 else 0
            if ret < STOP_LOSS:
                cost = abs(holdings[sym]["shares"] * px) * cost_frac
                cash += holdings[sym]["shares"] * px - cost
                trade_count += 1
                del holdings[sym]

        if not major_rebal:
            equity = cash
            for sym, h in holdings.items():
                equity += h["shares"] * today_prices.get(sym, h["entry_px"])
            port_values.append((date, equity))
            continue

        # Breadth blend
        dist_sma50 = uni.get_feature_map(date, "dist_sma50")
        breadth = sum(1 for v in dist_sma50.values() if v > 0) / max(len(dist_sma50), 1) if dist_sma50 else 0.5
        blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))

        # Combine strategies
        combined = {}
        for name, bull_pct in R2K_CONFIG_BULL:
            bear_pct = dict(R2K_CONFIG_BEAR).get(name, 0)
            cap = bull_pct * blend + bear_pct * (1 - blend)
            for sym, w in last_targets.get(name, {}).items():
                if w > 0:
                    combined[sym] = combined.get(sym, 0) + w * cap

        # Constraints
        for sym in list(combined):
            if combined[sym] > 0.08:
                combined[sym] = 0.08  # 8% single-name cap
        # Sector cap: 30%
        sec_tot = {}
        for sym, w in combined.items():
            sec = uni.sector_map.get(sym, "X")
            sec_tot[sec] = sec_tot.get(sec, 0) + w
        for sec, tot in sec_tot.items():
            if tot > 0.30:
                scale = 0.30 / tot
                for sym in list(combined):
                    if uni.sector_map.get(sym, "X") == sec:
                        combined[sym] *= scale
        # Gross cap: 100%
        gross = sum(combined.values())
        if gross > 1.0:
            for sym in combined:
                combined[sym] /= gross
        combined = {s: w for s, w in combined.items() if w >= 0.003}

        # Equity
        equity = cash
        for sym, h in holdings.items():
            equity += h["shares"] * today_prices.get(sym, h["entry_px"])

        # Rebalance
        target_d = {s: w * equity for s, w in combined.items()}
        for sym in list(holdings):
            if sym not in target_d:
                px = today_prices.get(sym, holdings[sym]["entry_px"])
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                trade_count += 1
                del holdings[sym]

        for sym, tgt in target_d.items():
            px = today_prices.get(sym)
            if not px or px <= 0:
                continue
            cur = holdings[sym]["shares"] * today_prices.get(sym, holdings[sym]["entry_px"]) if sym in holdings else 0
            delta = tgt - cur
            if abs(delta) < equity * 0.003:
                continue
            cost = abs(delta) * cost_frac
            trade_count += 1
            if delta > 0 and cash >= delta:
                shares = (delta - cost) / px
                if sym in holdings:
                    holdings[sym]["shares"] += shares
                else:
                    holdings[sym] = {"shares": shares, "entry_px": px}
                cash -= delta
            elif delta < 0 and sym in holdings:
                sell = min(abs(delta) / px, holdings[sym]["shares"])
                cash += sell * px - cost
                holdings[sym]["shares"] -= sell
                if holdings[sym]["shares"] < 0.01:
                    del holdings[sym]

        equity = cash
        for sym, h in holdings.items():
            equity += h["shares"] * today_prices.get(sym, h["entry_px"])
        port_values.append((date, max(equity, 0)))

    if not port_values:
        return None

    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0:
        years = 1
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd_s = dr[dr < 0].std()
    sortino = dr.mean() / dd_s * np.sqrt(252) if dd_s > 0 else np.nan
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    vol = dr.std() * np.sqrt(252)

    # IWM benchmark
    iwm_cagr = 0
    if "IWM" in uni._close:
        iwm = uni._close["IWM"].reindex(vals.index, method="ffill").dropna()
        if len(iwm) > 10:
            iwm = iwm / iwm.iloc[0] * INITIAL_CASH
            iwm_cagr = (iwm.iloc[-1] / iwm.iloc[0]) ** (1 / years) - 1

    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "vol": vol, "trades": trade_count,
        "final": vals.iloc[-1], "years": years,
        "iwm_cagr": iwm_cagr, "alpha": cagr - iwm_cagr,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    print("=" * 80)
    print("  RUSSELL 2000 SMALL-CAP STRATEGY BACKTEST")
    print("=" * 80)

    # Load data
    print("\n1. Loading data ...")
    features = pd.read_parquet(DATA_DIR / "smallcap" / "features_smallcap.parquet")
    features["date"] = pd.to_datetime(features["date"])
    print(f"   Features: {len(features)} rows, {features['symbol'].nunique()} symbols")

    universe = pd.read_parquet(DATA_DIR / "russell_short" / "russell2000_proxy_universe.parquet")
    universe["date"] = pd.to_datetime(universe["date"])
    print(f"   Universe: {len(universe)} rows, avg {universe.groupby('date')['symbol'].count().mean():.0f}/date")

    prices = pd.read_parquet(DATA_DIR / "russell_short" / "prices_russell.parquet")
    prices.index = pd.to_datetime(prices.index)
    print(f"   Prices: {len(prices)} dates, {len(prices.columns)} symbols")

    # Add IWM for benchmark
    try:
        from massive_data_provider import MassiveDataProvider
        provider = MassiveDataProvider(validate_vs_yfinance=False)
        iwm_bars = provider.fetch_bars_batch(["IWM", "SPY"], warmup_days=3800)
        for sym in ["IWM", "SPY"]:
            if sym in iwm_bars and len(iwm_bars[sym]) > 0:
                prices[sym] = iwm_bars[sym]["close"]
        print(f"   IWM/SPY benchmark loaded")
    except Exception as e:
        print(f"   WARNING: Could not load IWM/SPY: {e}")

    # Build sector map (use cache if available)
    import json
    sector_map = {}
    sector_file = DATA_DIR / "cache_sectors.json"
    if sector_file.exists():
        with open(sector_file) as f:
            sector_map = json.load(f)

    # Build universe
    uni = R2KUniverse(features, prices, universe, sector_map)

    # ── IN-SAMPLE ──
    print("\n" + "=" * 80)
    print("  IN-SAMPLE (2014-2021)")
    print("=" * 80)

    is_full = run_r2k_backtest(uni, "2014-01-01", "2021-12-31")
    if is_full:
        print(f"\n  CAGR:   {is_full['cagr']:+.1%}")
        print(f"  Sharpe: {is_full['sharpe']:.2f}")
        print(f"  Max DD: {is_full['max_dd']:.1%}")
        print(f"  Alpha:  {is_full['alpha']:+.1%} vs IWM")
        print(f"  Trades: {is_full['trades']}")

    print("\n  Per year:")
    for year in range(2014, 2022):
        m = run_r2k_backtest(uni, f"{year}-01-01", f"{year}-12-31")
        if m:
            print(f"    {year}: CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:.2f}  "
                  f"DD={m['max_dd']:.1%}  Alpha={m['alpha']:+.1%}")

    # ── OUT-OF-SAMPLE ──
    print("\n" + "=" * 80)
    print("  OUT-OF-SAMPLE (2022-2025)")
    print("=" * 80)

    oos_full = run_r2k_backtest(uni, "2022-01-01", "2025-12-31")
    if oos_full:
        print(f"\n  CAGR:    {oos_full['cagr']:+.1%}")
        print(f"  Sharpe:  {oos_full['sharpe']:.2f}")
        print(f"  Sortino: {oos_full['sortino']:.2f}")
        print(f"  Max DD:  {oos_full['max_dd']:.1%}")
        print(f"  Vol:     {oos_full['vol']:.1%}")
        print(f"  Alpha:   {oos_full['alpha']:+.1%} vs IWM")
        print(f"  Trades:  {oos_full['trades']}")
        print(f"  $100K -> ${oos_full['final']:,.0f}")

    print("\n  Per year:")
    for year in range(2022, 2026):
        m = run_r2k_backtest(uni, f"{year}-01-01", f"{year}-12-31")
        if m:
            print(f"    {year}: CAGR={m['cagr']:+.1%}  Sharpe={m['sharpe']:.2f}  "
                  f"DD={m['max_dd']:.1%}  Alpha={m['alpha']:+.1%}")

    # ── COMPARISON ──
    if is_full and oos_full:
        print("\n" + "=" * 80)
        print("  COMPARISON")
        print("=" * 80)
        print(f"\n  IS CAGR: {is_full['cagr']:+.1%}  OOS CAGR: {oos_full['cagr']:+.1%}")
        print(f"  IS Sharpe: {is_full['sharpe']:.2f}  OOS Sharpe: {oos_full['sharpe']:.2f}")
        if is_full["sharpe"] > 0:
            ratio = oos_full["sharpe"] / is_full["sharpe"]
            print(f"  Sharpe retention: {ratio:.2f} "
                  f"({'GOOD (>0.5)' if ratio > 0.5 else 'OVERFITTING CONCERN (<0.5)'})")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
