"""
Multi-Asset Trend Following Backtest + Correlation with Momentum
=================================================================
Trend Universe: SPY, QQQ, IWM, GLD, TLT, XLK, XLF, XLE, XLV, XLI, XLP, XLU, XLY, XLC, XLRE, XLB, SDS
Momentum Universe: SP1500 stocks (skip-month momentum, SMA200 filter, top-8, 10d rebal)

Signal:   Price > 200-SMA AND 50-SMA > 200-SMA => BUY
          Price < 200-SMA => CASH for that slot
Weighting: Inverse volatility (risk parity) among passing assets
Rebalance: Every 20 trading days
"""

import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ML_DIR = Path(__file__).resolve().parent.parent

# ── Load data ────────────────────────────────────────────────────────────────
PKL = ML_DIR / "data" / "wrds" / "complete_sp1500_universe.pkl"
with open(PKL, "rb") as f:
    data = pickle.load(f)

prices_df = data["prices_df"]
features_by_date = data["features_by_date"]

# SP1500 membership
sp500_mem = data.get("sp500_mem", {})
sp400_mem = data.get("sp400_mem", {})
sp600_mem = data.get("sp600_mem", {})

ETF_UNIVERSE = [
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLP", "XLU", "XLY", "XLC", "XLRE", "XLB", "SDS",
]

etf_prices = prices_df[ETF_UNIVERSE].copy().dropna(how="all")

print(f"ETF prices: {etf_prices.shape[0]} days, {etf_prices.shape[1]} assets")
print(f"Stock prices: {prices_df.shape[0]} days, {prices_df.shape[1]} symbols")
print(f"Date range: {etf_prices.index.min().date()} to {etf_prices.index.max().date()}")

# ── Compute ETF indicators ──────────────────────────────────────────────────
sma200 = etf_prices.rolling(200).mean()
sma50 = etf_prices.rolling(50).mean()
vol60 = etf_prices.pct_change().rolling(60).std() * np.sqrt(252)

# ── Helper ───────────────────────────────────────────────────────────────────
COST_BPS = 5
cost_frac = COST_BPS / 10_000
INITIAL_CASH = 100_000.0

start_date = pd.Timestamp("2017-01-01")
end_date = pd.Timestamp("2025-12-31")
trading_dates = prices_df.loc[start_date:end_date].index.tolist()
print(f"Backtest: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)\n")


def compute_metrics(returns_series, name="Strategy"):
    """Compute CAGR, Sharpe, Max DD from daily returns."""
    equity = (1 + returns_series).cumprod()
    total_days = (returns_series.index[-1] - returns_series.index[0]).days
    years = total_days / 365.25
    if years <= 0:
        years = 1.0
    cagr = equity.iloc[-1] ** (1 / years) - 1
    sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    yearly = {}
    for yr in range(returns_series.index[0].year, returns_series.index[-1].year + 1):
        yr_ret = returns_series[returns_series.index.year == yr]
        if len(yr_ret) > 0:
            yearly[yr] = (1 + yr_ret).prod() - 1

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  CAGR:       {cagr*100:+.1f}%")
    print(f"  Sharpe:     {sharpe:.2f}")
    print(f"  Max DD:     {max_dd*100:.1f}%")
    print(f"  Ann. Vol:   {returns_series.std()*np.sqrt(252)*100:.1f}%")
    print(f"\n  Yearly Returns:")
    for yr, r in sorted(yearly.items()):
        print(f"    {yr}: {r*100:+.1f}%")

    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "yearly": yearly,
            "returns": returns_series, "equity": equity}


# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY 1: MULTI-ASSET TREND FOLLOWING
# ═════════════════════════════════════════════════════════════════════════════
print("Running TREND FOLLOWING strategy...")

REBAL_TREND = 20
cash = INITIAL_CASH
holdings = {}
trend_port_values = []
day_since_rebal = REBAL_TREND  # trigger on first day

for day_idx, date in enumerate(trading_dates):
    px_today = etf_prices.loc[date] if date in etf_prices.index else pd.Series(dtype=float)

    # Mark-to-market
    port_val = cash
    for sym, h in holdings.items():
        p = px_today.get(sym, np.nan)
        if not np.isnan(p):
            port_val += h["shares"] * p
    trend_port_values.append({"date": date, "value": port_val})

    day_since_rebal += 1
    if day_since_rebal < REBAL_TREND:
        continue
    day_since_rebal = 0

    # Trend filter
    passing = []
    for sym in ETF_UNIVERSE:
        p = px_today.get(sym, np.nan)
        if np.isnan(p):
            continue
        s200 = sma200.at[date, sym] if date in sma200.index else np.nan
        s50 = sma50.at[date, sym] if date in sma50.index else np.nan
        v = vol60.at[date, sym] if date in vol60.index else np.nan
        if np.isnan(s200) or np.isnan(s50) or np.isnan(v):
            continue
        if p > s200 and s50 > s200 and v > 0.01:
            passing.append((sym, v))

    # Inverse-vol weights
    if passing:
        inv_vols = {sym: 1.0 / v for sym, v in passing}
        total_inv = sum(inv_vols.values())
        target_weights = {sym: iv / total_inv for sym, iv in inv_vols.items()}
    else:
        target_weights = {}

    # Liquidate all
    for sym, h in holdings.items():
        p = px_today.get(sym, np.nan)
        if not np.isnan(p):
            proceeds = h["shares"] * p
            cash += proceeds - proceeds * cost_frac
    holdings = {}

    # Buy targets
    if target_weights:
        invest_total = cash  # invest all cash
        for sym, w in target_weights.items():
            p = px_today.get(sym, np.nan)
            if not np.isnan(p) and p > 0:
                alloc = invest_total * w
                shares = int(alloc / p)
                if shares > 0:
                    cost_val = shares * p * cost_frac
                    cash -= shares * p + cost_val
                    holdings[sym] = {"shares": shares, "entry_px": p}

trend_series = pd.DataFrame(trend_port_values).set_index("date")["value"]
trend_returns = trend_series.pct_change().dropna()
trend_metrics = compute_metrics(trend_returns, "MULTI-ASSET TREND FOLLOWING")


# ═════════════════════════════════════════════════════════════════════════════
#  STRATEGY 2: MOMENTUM (simplified v10.1 replica)
# ═════════════════════════════════════════════════════════════════════════════
print("\n\nRunning MOMENTUM strategy (v10.1 replica)...")

REBAL_MOM = 10
TOP_N = 8

def get_sp1500_members(date):
    """Get SP1500 members for a given date from membership dicts."""
    members = set()
    for mem_dict in [sp500_mem, sp400_mem, sp600_mem]:
        for sym, periods in mem_dict.items():
            if isinstance(periods, list):
                for p in periods:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        s, e = pd.Timestamp(p[0]), pd.Timestamp(p[1])
                        if s <= date <= e:
                            members.add(sym)
                            break
    return members

# Fallback: use features_by_date keys as member proxy
def get_members_from_features(date):
    """Use stocks present in features as proxy for investable universe."""
    feats = features_by_date.get(date, {})
    return set(feats.keys()) if isinstance(feats, dict) else set()

cash = INITIAL_CASH
holdings = {}
mom_port_values = []

for day_idx, date in enumerate(trading_dates):
    # Mark-to-market
    port_val = cash
    for sym, h in holdings.items():
        px = prices_df.at[date, sym] if sym in prices_df.columns else np.nan
        if not np.isnan(px):
            port_val += h["shares"] * px
        else:
            port_val += h["shares"] * h["entry_px"]  # stale price
    mom_port_values.append({"date": date, "value": port_val})

    if day_idx % REBAL_MOM != 0:
        continue

    # Get features for today
    feats = features_by_date.get(date, {})
    if not isinstance(feats, dict) or len(feats) < 50:
        continue

    members = set(feats.keys())

    # Breadth-based stress detection
    above_sma50 = sum(1 for s, f in feats.items()
                      if isinstance(f, dict) and f.get("dist_sma50", 0) > 0)
    breadth = above_sma50 / max(len(feats), 1)
    stress = breadth < 0.30
    n = max(TOP_N // 2, 5) if stress else TOP_N

    # Score: skip-month momentum (12-1) + consolidation + quality
    composite = {}
    for sym in members:
        f = feats[sym]
        if not isinstance(f, dict):
            continue
        r252 = f.get("ret_252d")
        r20 = f.get("ret_20d")
        d200 = f.get("dist_sma200")
        if r252 is None or r20 is None or np.isnan(r252) or np.isnan(r20):
            continue
        if d200 is None or np.isnan(d200) or d200 <= 0:
            continue  # trend filter: must be above 200-SMA

        score = r252 - r20  # skip-month momentum

        # Consolidation breakout
        v20 = f.get("vol_20d")
        if v20 is not None and not np.isnan(v20) and v20 < 0.25 and score > 0.20:
            score *= 1.15

        # EPS surprise boost
        eps = f.get("eps_surprise_last")
        if eps is not None and not np.isnan(eps) and eps > 0:
            score *= 1.15

        # ROE boost
        roe = f.get("roe")
        if roe is not None and not np.isnan(roe) and roe > 0.15:
            score *= 1.05

        composite[sym] = score

    if not composite:
        continue

    # Top N
    ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)[:n]
    target_syms = [sym for sym, _ in ranked]

    # Equal weight among top N
    target_w = 1.0 / len(target_syms)

    # Liquidate
    for sym, h in holdings.items():
        px = prices_df.at[date, sym] if sym in prices_df.columns else np.nan
        if not np.isnan(px):
            cash += h["shares"] * px - h["shares"] * px * cost_frac
    holdings = {}

    # Buy
    invest_total = cash
    for sym in target_syms:
        px = prices_df.at[date, sym] if sym in prices_df.columns else np.nan
        if not np.isnan(px) and px > 0:
            alloc = invest_total * target_w
            shares = int(alloc / px)
            if shares > 0:
                cash -= shares * px + shares * px * cost_frac
                holdings[sym] = {"shares": shares, "entry_px": px}

mom_series = pd.DataFrame(mom_port_values).set_index("date")["value"]
mom_returns = mom_series.pct_change().dropna()
mom_metrics = compute_metrics(mom_returns, "MOMENTUM (v10.1 REPLICA)")


# ═════════════════════════════════════════════════════════════════════════════
#  CORRELATION ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  CORRELATION ANALYSIS")
print("="*60)

common_idx = trend_returns.index.intersection(mom_returns.index)
t_aligned = trend_returns.loc[common_idx]
m_aligned = mom_returns.loc[common_idx]

corr = t_aligned.corr(m_aligned)
print(f"  Daily return correlation:      {corr:.3f}")

rolling_corr = t_aligned.rolling(60).corr(m_aligned).dropna()
print(f"  Rolling 60d corr range:        [{rolling_corr.min():.2f}, {rolling_corr.max():.2f}]")
print(f"  Rolling 60d corr mean:         {rolling_corr.mean():.2f}")

# Correlation in down months
monthly_t = t_aligned.resample("ME").sum()
monthly_m = m_aligned.resample("ME").sum()
down_months = monthly_m[monthly_m < 0].index
if len(down_months) > 5:
    down_t = monthly_t.loc[down_months]
    down_m = monthly_m.loc[down_months]
    down_corr = down_t.corr(down_m)
    print(f"  Correlation in DOWN months:    {down_corr:.3f} (n={len(down_months)})")


# ═════════════════════════════════════════════════════════════════════════════
#  COMBINED PORTFOLIO ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  COMBINED PORTFOLIO ANALYSIS")
print("="*60)

allocations = [
    ("100/0  (Momentum only)", 1.0, 0.0),
    ("80/20  Mom/Trend",       0.8, 0.2),
    ("70/30  Mom/Trend",       0.7, 0.3),
    ("60/40  Mom/Trend",       0.6, 0.4),
    ("50/50  Mom/Trend",       0.5, 0.5),
    ("0/100  (Trend only)",    0.0, 1.0),
]

results_table = []
for label, mom_w, trend_w in allocations:
    combined = m_aligned * mom_w + t_aligned * trend_w
    equity = (1 + combined).cumprod()
    total_days = (combined.index[-1] - combined.index[0]).days
    years = total_days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1
    sharpe = combined.mean() / combined.std() * np.sqrt(252) if combined.std() > 0 else 0
    dd = (equity / equity.cummax() - 1).min()
    ann_vol = combined.std() * np.sqrt(252)

    # Yearly
    yearly = {}
    for yr in range(2017, 2026):
        yr_ret = combined[combined.index.year == yr]
        if len(yr_ret) > 0:
            yearly[yr] = (1 + yr_ret).prod() - 1

    results_table.append({
        "Allocation": label,
        "cagr": cagr, "sharpe": sharpe, "dd": dd, "vol": ann_vol,
        "yearly": yearly,
    })

print(f"\n  {'Allocation':<25} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>8} {'Vol':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in results_table:
    print(f"  {r['Allocation']:<25} {r['cagr']*100:+7.1f}% {r['sharpe']:>7.2f} {r['dd']*100:>7.1f}% {r['vol']*100:>7.1f}%")

# Yearly breakdown for key allocations
print(f"\n  YEARLY RETURNS BY ALLOCATION:")
header = f"  {'Year':<6}"
key_allocs = [0, 1, 2, 3, 5]  # 100/0, 80/20, 70/30, 60/40, 0/100
for i in key_allocs:
    header += f" {results_table[i]['Allocation']:<14}"
print(header)
print(f"  {'-'*80}")
for yr in range(2017, 2026):
    row = f"  {yr:<6}"
    for i in key_allocs:
        y = results_table[i]["yearly"].get(yr, 0)
        row += f" {y*100:+13.1f}%"
    print(row)


# ═════════════════════════════════════════════════════════════════════════════
#  VERDICT
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  VERDICT: DOES TREND IMPROVE THE PORTFOLIO?")
print("="*60)

mom_only = results_table[0]
for r in results_table[1:-1]:
    better_cagr = r["cagr"] > mom_only["cagr"]
    better_sharpe = r["sharpe"] > mom_only["sharpe"]
    less_dd = r["dd"] > mom_only["dd"]  # less negative = better
    lower_vol = r["vol"] < mom_only["vol"]
    improvements = []
    if better_cagr:
        improvements.append(f"CAGR +{(r['cagr']-mom_only['cagr'])*100:.1f}pp")
    if better_sharpe:
        improvements.append(f"Sharpe +{r['sharpe']-mom_only['sharpe']:.2f}")
    if less_dd:
        improvements.append(f"DD improved {(r['dd']-mom_only['dd'])*100:+.1f}pp")
    if lower_vol:
        improvements.append(f"Vol -{(mom_only['vol']-r['vol'])*100:.1f}pp")
    degradations = []
    if not better_cagr:
        degradations.append(f"CAGR {(r['cagr']-mom_only['cagr'])*100:+.1f}pp")
    if not better_sharpe:
        degradations.append(f"Sharpe {r['sharpe']-mom_only['sharpe']:+.2f}")

    status = ""
    if improvements:
        status += "  BETTER: " + ", ".join(improvements)
    if degradations:
        status += "  |  WORSE: " + ", ".join(degradations)
    print(f"  {r['Allocation']:<25}{status}")

# ═════════════════════════════════════════════════════════════════════════════
#  VOL-NORMALIZED COMPARISON (fair apples-to-apples)
# ═════════════════════════════════════════════════════════════════════════════
# The momentum replica runs at ~55% vol vs real v10.1's ~22%. Scale to 22% target.
print("\n\n" + "="*60)
print("  VOL-NORMALIZED COMPARISON (target 22% vol)")
print("="*60)

TARGET_VOL = 0.22
mom_scale = TARGET_VOL / (m_aligned.std() * np.sqrt(252))
m_scaled = m_aligned * mom_scale  # scale to 22% vol (rest goes to cash)

scaled_mom_equity = (1 + m_scaled).cumprod()
scaled_years = (m_scaled.index[-1] - m_scaled.index[0]).days / 365.25
scaled_mom_cagr = scaled_mom_equity.iloc[-1] ** (1 / scaled_years) - 1
scaled_mom_sharpe = m_scaled.mean() / m_scaled.std() * np.sqrt(252)
scaled_mom_dd = (scaled_mom_equity / scaled_mom_equity.cummax() - 1).min()
scaled_mom_vol = m_scaled.std() * np.sqrt(252)

print(f"\n  Momentum (vol-scaled):  CAGR {scaled_mom_cagr*100:+.1f}%, Sharpe {scaled_mom_sharpe:.2f}, DD {scaled_mom_dd*100:.1f}%, Vol {scaled_mom_vol*100:.1f}%")

# Now combine at same vol
print(f"\n  {'Allocation':<25} {'CAGR':>8} {'Sharpe':>8} {'Max DD':>8} {'Vol':>8}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

norm_results = []
for label, mom_w, trend_w in allocations:
    combined = m_scaled * mom_w + t_aligned * trend_w
    equity = (1 + combined).cumprod()
    total_days = (combined.index[-1] - combined.index[0]).days
    years = total_days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1
    sharpe = combined.mean() / combined.std() * np.sqrt(252) if combined.std() > 0 else 0
    dd = (equity / equity.cummax() - 1).min()
    ann_vol = combined.std() * np.sqrt(252)
    print(f"  {label:<25} {cagr*100:+7.1f}% {sharpe:>7.2f} {dd*100:>7.1f}% {ann_vol*100:>7.1f}%")
    norm_results.append({"label": label, "cagr": cagr, "sharpe": sharpe, "dd": dd, "vol": ann_vol})

print(f"\n  VERDICT (vol-normalized):")
nr0 = norm_results[0]
for r in norm_results[1:-1]:
    improvements = []
    if r["cagr"] > nr0["cagr"]:
        improvements.append(f"CAGR +{(r['cagr']-nr0['cagr'])*100:.1f}pp")
    if r["sharpe"] > nr0["sharpe"]:
        improvements.append(f"Sharpe +{r['sharpe']-nr0['sharpe']:.2f}")
    if r["dd"] > nr0["dd"]:
        improvements.append(f"DD improved {(r['dd']-nr0['dd'])*100:+.1f}pp")
    degradations = []
    if r["cagr"] <= nr0["cagr"]:
        degradations.append(f"CAGR {(r['cagr']-nr0['cagr'])*100:+.1f}pp")
    if r["sharpe"] <= nr0["sharpe"]:
        degradations.append(f"Sharpe {r['sharpe']-nr0['sharpe']:+.2f}")
    status = ""
    if improvements:
        status += "BETTER: " + ", ".join(improvements)
    if degradations:
        status += "  |  WORSE: " + ", ".join(degradations)
    print(f"    {r['label']:<25} {status}")

print("\nDone.")
