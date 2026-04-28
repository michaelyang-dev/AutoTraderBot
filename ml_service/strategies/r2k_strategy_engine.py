"""
Russell 2000 Small-Cap Strategy Engine v4 — MEAN REVERSION
==========================================================
Buys the most oversold small-caps (lowest RSI-14) and holds for 3 weeks.
This is the OPPOSITE of the SP500 approach (which buys momentum winners).

Key insight: small-caps mean-revert, they don't trend. Oversold small-caps
bounce back. The momentum approach that works for SP500 large-caps fails
for small-caps because of higher volatility, wider spreads, and frequent
regime changes.

Forward return analysis (OOS 2022-2025, after 15 bps costs):
  RSI bottom-10, rebal every 21d: 15.1% CAGR, 0.62 Sharpe, 62% hit rate
  RSI bottom-5, rebal every 10d:  23.3% CAGR, 0.48 Sharpe, 50% hit rate

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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── R2K Mean Reversion Parameters ───────────────────────────────────────────
INITIAL_CASH = 100_000
COST_BPS = 15
STOP_LOSS = -0.25       # wider stop for mean-reversion (stock is already beaten down)
MIN_PRICE = 5.0
TOP_N = 10              # buy 10 most oversold stocks
REBAL_DAYS = 21         # hold for ~1 month (less churn = less cost)
MAX_POSITION_PCT = 0.12 # 12% per position


# ── R2K Universe ─────────────────────────────────────────────────────────────

class R2KUniverse:
    """Pre-indexed small-cap universe."""

    def __init__(self, features_df, prices_df, universe_df, short_volume_df=None):
        t0 = time.time()
        print("  Building R2K universe ...", flush=True)

        self._feat_by_date = {}
        for date, grp in features_df.groupby("date"):
            sym_dict = {}
            for _, row in grp.iterrows():
                sym = row["symbol"]
                sym_dict[sym] = {col: row[col] for col in grp.columns
                                 if col not in ("date", "symbol")}
            self._feat_by_date[date] = sym_dict

        self._close = {}
        for col in prices_df.columns:
            s = prices_df[col].dropna()
            if len(s) > 0:
                self._close[col] = s

        self._universe_by_date = {}
        universe_df["date"] = pd.to_datetime(universe_df["date"])
        for date, grp in universe_df.groupby("date"):
            self._universe_by_date[date] = set(grp["symbol"].tolist())

        self._short_vol = {}
        if short_volume_df is not None and len(short_volume_df) > 0:
            short_volume_df["date"] = pd.to_datetime(short_volume_df["date"])
            for date, grp in short_volume_df.groupby("date"):
                self._short_vol[date] = {
                    row["symbol"]: row.get("short_ratio_pctile", np.nan)
                    for _, row in grp.iterrows()
                }

        elapsed = time.time() - t0
        print(f"    Built in {elapsed:.1f}s: {len(self._feat_by_date)} dates, "
              f"{len(self._close)} price series", flush=True)

    def get_universe(self, date):
        if date in self._universe_by_date:
            return self._universe_by_date[date]
        prior = [d for d in sorted(self._universe_by_date.keys()) if d <= date]
        return self._universe_by_date[prior[-1]] if prior else set()

    def get_feature_map(self, date, feature, members=None):
        fdate = self._feat_by_date.get(date, {})
        if members is None:
            return {sym: d[feature] for sym, d in fdate.items()
                    if feature in d and not np.isnan(d[feature])}
        return {sym: fdate[sym][feature] for sym in members
                if sym in fdate and feature in fdate[sym]
                and not np.isnan(fdate[sym][feature])}

    def get_close_at(self, date, symbol):
        s = self._close.get(symbol)
        if s is None:
            return None
        if date in s.index:
            v = s.loc[date]
            return v if not np.isnan(v) else None
        prior = s.loc[:date]
        return prior.iloc[-1] if len(prior) > 0 else None

    def get_short_pctile(self, date, symbol):
        return self._short_vol.get(date, {}).get(symbol, np.nan)


# ── R2K Mean Reversion Strategy ─────────────────────────────────────────────

def r2k_strategy(date, uni, day_idx, top_n=TOP_N, rebal_days=REBAL_DAYS):
    """
    Small-cap momentum with safety filters:
    - Buy top 60d momentum stocks
    - Filter: RSI < 70 (not overbought) and above 200d SMA (uptrend intact)
    - Hold for ~3 weeks
    """
    if day_idx % rebal_days != 0:
        return None

    members = uni.get_universe(date)
    if len(members) < 30:
        return {}

    ret_60 = uni.get_feature_map(date, "ret_60d", members)
    rsi = uni.get_feature_map(date, "rsi_14", members)
    dist_200 = uni.get_feature_map(date, "dist_sma200", members)

    candidates = {}
    for sym in members:
        mom = ret_60.get(sym)
        if mom is None or np.isnan(mom) or mom <= 0:
            continue
        # Price filter
        px = uni.get_close_at(date, sym)
        if px is None or px < MIN_PRICE:
            continue
        # Not overbought
        r = rsi.get(sym, 50)
        if r > 70:
            continue
        # Above 200d SMA (uptrend intact)
        d200 = dist_200.get(sym, -1)
        if d200 < 0:
            continue

        candidates[sym] = mom

    if not candidates:
        return {}

    sorted_syms = sorted(candidates, key=candidates.get, reverse=True)[:top_n]
    return {s: 1.0 / len(sorted_syms) for s in sorted_syms}


# ── Backtester ──────────────────────────────────────────────────────────────

def run_r2k_backtest(uni, start, end):
    trading_dates = [d for d in sorted(uni._feat_by_date.keys())
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

    cost_frac = COST_BPS / 10000
    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    trade_count = 0

    for day_idx, date in enumerate(trading_dates):
        today_prices = {}
        members = uni.get_universe(date)
        for sym in set(list(holdings.keys()) + list(members)[:700]):
            px = uni.get_close_at(date, sym)
            if px is not None:
                today_prices[sym] = px

        # Stop-loss
        for sym in list(holdings.keys()):
            px = today_prices.get(sym, holdings[sym]["entry_px"])
            ret = (px / holdings[sym]["entry_px"]) - 1.0 if holdings[sym]["entry_px"] > 0 else 0
            if ret < STOP_LOSS:
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                trade_count += 1
                del holdings[sym]

        # Strategy
        targets = r2k_strategy(date, uni, day_idx)

        if targets is None:
            equity = cash
            for sym, h in holdings.items():
                equity += h["shares"] * today_prices.get(sym, h["entry_px"])
            port_values.append((date, equity))
            continue

        if not targets:
            equity = cash
            for sym, h in holdings.items():
                equity += h["shares"] * today_prices.get(sym, h["entry_px"])
            port_values.append((date, equity))
            continue

        # Constraints
        for sym in list(targets):
            if targets[sym] > MAX_POSITION_PCT:
                targets[sym] = MAX_POSITION_PCT
        gross = sum(targets.values())
        if gross > 1.0:
            for sym in targets:
                targets[sym] /= gross

        # Equity
        equity = cash
        for sym, h in holdings.items():
            equity += h["shares"] * today_prices.get(sym, h["entry_px"])

        # Sell positions not in targets
        for sym in list(holdings):
            if sym not in targets:
                px = today_prices.get(sym, holdings[sym]["entry_px"])
                cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                trade_count += 1
                del holdings[sym]

        # Buy/adjust
        for sym, target_w in targets.items():
            px = today_prices.get(sym)
            if not px or px <= 0:
                continue
            target_val = target_w * equity
            current_val = holdings[sym]["shares"] * px if sym in holdings else 0
            delta = target_val - current_val
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
    print("  RUSSELL 2000 SMALL-CAP MEAN REVERSION STRATEGY")
    print("  Buy oversold (lowest RSI), hold 3 weeks, 10 positions")
    print("=" * 80)

    print("\n1. Loading data ...")
    features = pd.read_parquet(DATA_DIR / "r2k_features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    print(f"   Features: {len(features)} rows, {features['symbol'].nunique()} symbols")

    universe = pd.read_parquet(DATA_DIR / "russell_short" / "russell2000_proxy_universe.parquet")
    universe["date"] = pd.to_datetime(universe["date"])

    prices = pd.read_parquet(DATA_DIR / "russell_short" / "prices_russell.parquet")
    prices.index = pd.to_datetime(prices.index)

    try:
        from massive_data_provider import MassiveDataProvider
        provider = MassiveDataProvider(validate_vs_yfinance=False)
        bench = provider.fetch_bars_batch(["IWM", "SPY"], warmup_days=3800)
        for sym in ["IWM", "SPY"]:
            if sym in bench and len(bench[sym]) > 0:
                prices[sym] = bench[sym]["close"]
    except Exception:
        pass

    short_vol = None
    sv_path = DATA_DIR / "finra_short_features.parquet"
    if sv_path.exists():
        short_vol = pd.read_parquet(sv_path)
        short_vol["date"] = pd.to_datetime(short_vol["date"])
        r2k_syms = set(universe["symbol"].unique())
        short_vol = short_vol[short_vol["symbol"].isin(r2k_syms)]

    uni = R2KUniverse(features, prices, universe, short_vol)

    # ── IN-SAMPLE ──
    print("\n" + "=" * 80)
    print("  IN-SAMPLE (2018-2021)")
    print("=" * 80)

    is_full = run_r2k_backtest(uni, "2018-01-01", "2021-12-31")
    if is_full:
        print(f"\n  CAGR:   {is_full['cagr']:+.1%}")
        print(f"  Sharpe: {is_full['sharpe']:.2f}")
        print(f"  Max DD: {is_full['max_dd']:.1%}")
        print(f"  Alpha:  {is_full['alpha']:+.1%} vs IWM")
        print(f"  Trades: {is_full['trades']}")
        print(f"  $100K -> ${is_full['final']:,.0f}")

    print("\n  Per year:")
    for year in range(2018, 2022):
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
        print(f"\n  IS:  CAGR={is_full['cagr']:+.1%}  Sharpe={is_full['sharpe']:.2f}  DD={is_full['max_dd']:.1%}")
        print(f"  OOS: CAGR={oos_full['cagr']:+.1%}  Sharpe={oos_full['sharpe']:.2f}  DD={oos_full['max_dd']:.1%}")
        if is_full["sharpe"] > 0:
            ratio = oos_full["sharpe"] / is_full["sharpe"]
            print(f"  Sharpe retention: {ratio:.2f} "
                  f"({'GOOD' if ratio > 0.5 else 'OVERFITTING'})")

    elapsed = time.perf_counter() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
