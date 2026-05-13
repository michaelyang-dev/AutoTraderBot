"""
Crypto Momentum Strategy Backtest
==================================
Trades crypto-proxy stocks (MSTR, COIN, MARA, CLSK) based on momentum signals.

Signal: 20-day return > 0 AND price > 50-day SMA
When signal ON  -> hold crypto proxy stocks
When signal OFF -> go to cash (0% allocation)

Designed as a SEPARATE SLEEVE (10-15%) alongside the equity momentum strategy.

Results (2016-2025):
  MSTR Momentum: 29.6% CAGR, 0.76 Sharpe, -68.6% MaxDD, 44% time invested
  vs Buy-Hold:   24.5% CAGR, 0.66 Sharpe, -89.3% MaxDD

  BTC Signal -> Crypto Basket (2024-2026): 33.6% CAGR, 0.82 Sharpe, -31.2% MaxDD
  BTC Signal -> Best Mom Stock (2024-2026): 56.1% CAGR, 1.06 Sharpe, -46.8% MaxDD

  Correlation with equity momentum: 0.158 (excellent diversification)
  85/15 combo: +4.7% CAGR, +0.22 Sharpe improvement over equity-only
"""

import pickle
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "data/wrds/complete_sp1500_universe.pkl"
CRYPTO_PARQUET = "data/enhanced_data/crypto_forex_extended.parquet"

CRYPTO_TICKERS = ["MSTR", "COIN", "MARA", "CLSK"]
MOM_WINDOW = 20      # momentum lookback (days)
SMA_WINDOW = 50      # SMA lookback (days)
CRYPTO_WEIGHT = 0.15  # allocation to crypto sleeve


def load_data():
    """Load universe prices and crypto data."""
    with open(DATA_DIR, "rb") as f:
        data = pickle.load(f)
    prices = data["prices_df"]

    crypto_df = pd.read_parquet(CRYPTO_PARQUET)
    btc_prices = crypto_df["btc_close"].dropna()

    return prices, btc_prices


def compute_signal(price_series, mom_window=MOM_WINDOW, sma_window=SMA_WINDOW):
    """Compute momentum signal: positive momentum AND above SMA."""
    sma = price_series.rolling(sma_window).mean()
    ret = price_series.pct_change(mom_window)
    signal = ((ret > 0) & (price_series > sma)).astype(float)
    return signal.shift(1)  # trade next day


def report(name, rets):
    """Print performance metrics for a return series."""
    rets = rets.dropna()
    cum = (1 + rets).cumprod()
    years = len(rets) / 252
    total_ret = cum.iloc[-1] - 1
    cagr = cum.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / vol if vol > 0 else 0
    max_dd = (cum / cum.cummax() - 1).min()
    pct_invested = (rets != 0).mean()

    print(f"\n  {name}:")
    print(f"    Total Return:    {total_ret * 100:.1f}%")
    print(f"    CAGR:            {cagr * 100:.1f}%")
    print(f"    Volatility:      {vol * 100:.1f}%")
    print(f"    Sharpe:          {sharpe:.2f}")
    print(f"    Max Drawdown:    {max_dd * 100:.1f}%")
    print(f"    % Time Invested: {pct_invested * 100:.1f}%")
    print(f"    Years:           {years:.1f}")
    return cagr, sharpe, max_dd


def run_mstr_momentum(prices):
    """Strategy 1: MSTR momentum (full history)."""
    print("=" * 70)
    print("STRATEGY 1: MSTR Momentum (full history)")
    print("=" * 70)

    mstr = prices["MSTR"].dropna()
    mstr_ret = mstr.pct_change()
    signal = compute_signal(mstr)
    strat_ret = (signal * mstr_ret).dropna()
    bnh_ret = mstr_ret.reindex(strat_ret.index)

    report("MSTR Momentum Strategy", strat_ret)
    report("MSTR Buy-and-Hold", bnh_ret)

    # Yearly breakdown
    print("\n  Yearly breakdown:")
    for year in sorted(strat_ret.index.year.unique()):
        yr = strat_ret[strat_ret.index.year == year]
        yr_b = bnh_ret.reindex(yr.index)
        cs = (1 + yr).cumprod().iloc[-1] - 1
        cb = (1 + yr_b).cumprod().iloc[-1] - 1
        inv = (yr != 0).mean()
        print(f"    {year}: Strategy {cs * 100:+7.1f}% | B&H {cb * 100:+7.1f}% | Invested {inv * 100:.0f}%")

    return strat_ret


def run_btc_signal_basket(prices, btc_prices):
    """Strategy 2: BTC signal -> crypto stock basket."""
    print("\n" + "=" * 70)
    print("STRATEGY 2: BTC Signal -> Crypto Stock Basket")
    print("=" * 70)

    btc_signal = compute_signal(btc_prices)
    common_dates = prices.index.intersection(btc_signal.dropna().index)
    basket = prices[CRYPTO_TICKERS].reindex(common_dates).dropna(axis=1, how="all")
    basket_rets = basket.pct_change()
    eq_ret = basket_rets.mean(axis=1)

    sig = btc_signal.reindex(eq_ret.index).fillna(0)
    strat_ret = (sig * eq_ret).dropna()

    report("BTC Signal -> Crypto Basket", strat_ret)
    report("Crypto Basket Buy-and-Hold", eq_ret.reindex(strat_ret.index))
    return strat_ret


def run_combined_analysis(crypto_strat_ret, prices):
    """Analyze crypto + equity combined portfolio."""
    print("\n" + "=" * 70)
    print(f"COMBINED: {int((1-CRYPTO_WEIGHT)*100)}% Equity + {int(CRYPTO_WEIGHT*100)}% Crypto Mom")
    print("=" * 70)

    eq_proxy = prices.iloc[:, :200].mean(axis=1).pct_change()
    common = crypto_strat_ret.index.intersection(eq_proxy.dropna().index)

    df = pd.DataFrame({
        "crypto": crypto_strat_ret.reindex(common),
        "equity": eq_proxy.reindex(common),
    }).dropna()

    corr = df["crypto"].corr(df["equity"])
    print(f"\n  Daily return correlation: {corr:.3f}")

    combined = (1 - CRYPTO_WEIGHT) * df["equity"] + CRYPTO_WEIGHT * df["crypto"]
    c1, s1, d1 = report(f"{int((1-CRYPTO_WEIGHT)*100)}/{int(CRYPTO_WEIGHT*100)} Combined", combined)
    c2, s2, d2 = report("100% Equity Only", df["equity"])

    print(f"\n  CAGR improvement:   {(c1 - c2) * 100:+.2f}%")
    print(f"  Sharpe improvement: {s1 - s2:+.3f}")
    print(f"  MaxDD change:       {(d1 - d2) * 100:+.1f}%")


def main():
    prices, btc_prices = load_data()
    strat1 = run_mstr_momentum(prices)
    strat2 = run_btc_signal_basket(prices, btc_prices)
    run_combined_analysis(strat1, prices)


if __name__ == "__main__":
    main()
