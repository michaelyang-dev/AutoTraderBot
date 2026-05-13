"""
Expanded Universe Momentum Backtest
====================================
Tests whether expanding beyond SP1500 to ALL liquid US stocks improves
the momentum strategy.

Uses CRSP daily data for all stocks, applies liquidity filters,
computes skip-month momentum, and runs a proper portfolio simulation
with trailing stops, transaction costs, and position caps.
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# ─── Configuration ───────────────────────────────────────────────────────────
MIN_PRICE = 10.0
MIN_MCAP = 500_000     # $500M in CRSP DlyCap units (thousands of dollars)
MIN_AVG_VOL = 100_000  # shares
TOP_N = 8
REBAL_DAYS = 20        # ~monthly
TRAILING_STOP = 0.35
POSITION_CAP = 0.125
COST_BPS = 15          # 15bp per trade
INITIAL_CASH = 100_000.0
START = "2017-01-03"
END = "2025-12-31"

print("=" * 70)
print("EXPANDED UNIVERSE MOMENTUM BACKTEST")
print("=" * 70)

# ─── 1. Load CRSP data ──────────────────────────────────────────────────────
t0 = time.time()
print("\n[1] Loading CRSP daily data (post-2016, equity only)...")

cols = ["PERMNO", "Ticker", "DlyCalDt", "DlyPrc", "DlyVol", "DlyCap",
        "DlyRet", "SecurityType", "IssuerType", "ShareType", "PrimaryExch"]

df = pd.read_parquet(str(BASE / "data/wrds/crsp_daily_stock_full.parquet"), columns=cols)
df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
df = df[df["DlyCalDt"] >= "2016-01-01"]  # need 252 lookback from 2017

# Filter to common stocks only (no funds, no preferred)
df = df[df["SecurityType"] == "EQTY"]
df = df[df["IssuerType"].isin(["CORP", "REIT"])]  # corporates + REITs
df = df[df["ShareType"] == "NS"]  # ordinary shares only (no ADRs for now)
df = df[df["PrimaryExch"].isin(["N", "Q", "A"])]  # NYSE, NASDAQ, AMEX

# Use abs price (CRSP uses negative for bid/ask average)
df["DlyPrc"] = df["DlyPrc"].abs()

# Drop rows with missing critical data
df = df.dropna(subset=["DlyPrc", "DlyVol", "Ticker"])
df = df[df["DlyPrc"] > 0]

print(f"  Loaded {len(df):,} rows, {df['Ticker'].nunique():,} unique tickers")
print(f"  Date range: {df['DlyCalDt'].min().date()} to {df['DlyCalDt'].max().date()}")
print(f"  Time: {time.time()-t0:.1f}s")

# ─── 2. Build price/return/volume matrices ──────────────────────────────────
print("\n[2] Building price matrix (pivot by ticker)...")
t1 = time.time()

# Use PERMNO as the identifier to handle ticker changes, then map back
# For each PERMNO-date, keep only one row (latest ticker)
df = df.sort_values(["PERMNO", "DlyCalDt"]).drop_duplicates(
    subset=["PERMNO", "DlyCalDt"], keep="last"
)

# Create a PERMNO -> latest ticker mapping per date
# But for simplicity, use the ticker that was active on each date
# We'll work with PERMNO internally and map to ticker for display

# Pivot price by PERMNO
price_piv = df.pivot_table(index="DlyCalDt", columns="PERMNO", values="DlyPrc", aggfunc="last")
vol_piv = df.pivot_table(index="DlyCalDt", columns="PERMNO", values="DlyVol", aggfunc="last")
cap_piv = df.pivot_table(index="DlyCalDt", columns="PERMNO", values="DlyCap", aggfunc="last")
ret_piv = df.pivot_table(index="DlyCalDt", columns="PERMNO", values="DlyRet", aggfunc="last")

# Build PERMNO -> Ticker mapping (use last known ticker for each PERMNO)
permno_ticker = df.sort_values("DlyCalDt").drop_duplicates("PERMNO", keep="last").set_index("PERMNO")["Ticker"].to_dict()

print(f"  Price matrix: {price_piv.shape[0]} dates x {price_piv.shape[1]} PERMNOs")
print(f"  Time: {time.time()-t1:.1f}s")

# ─── 3. Compute signals ─────────────────────────────────────────────────────
print("\n[3] Computing momentum signals...")
t2 = time.time()

# Adjust prices for splits/dividends using cumulative returns
# CRSP DlyRet already accounts for dividends, so we use it to build adjusted prices
# Forward fill to handle missing days, then compute returns from adjusted prices

# Compute cumulative return factor per PERMNO
cum_ret = (1 + ret_piv.fillna(0)).cumprod()

# 252-day return (1-year)
ret_252d = cum_ret / cum_ret.shift(252) - 1

# 20-day return (1-month)
ret_20d = cum_ret / cum_ret.shift(20) - 1

# Skip-month momentum: 12-month return minus last month
skip_mom = ret_252d - ret_20d

# SMA200 filter: price above 200-day SMA
sma200 = price_piv.rolling(200, min_periods=180).mean()
above_sma200 = price_piv > sma200

# Average volume over last 60 days
avg_vol_60 = vol_piv.rolling(60, min_periods=40).mean()

# Recent market cap
recent_cap = cap_piv.rolling(5, min_periods=1).mean()

# 20-day realized volatility (for consolidation detection)
vol_20d = ret_piv.rolling(20, min_periods=15).std() * np.sqrt(252)

print(f"  Signals computed. Time: {time.time()-t2:.1f}s")

# ─── 4. Load SP1500 membership for comparison ───────────────────────────────
print("\n[4] Loading SP1500 membership...")
import pickle
try:
    with open(str(BASE / "data/wrds/complete_sp1500_universe.pkl"), "rb") as f:
        sp_data = pickle.load(f)
    sp500_mem = sp_data.get("sp500_mem", {})
    sp400_mem = sp_data.get("sp400_mem", {})
    sp600_mem = sp_data.get("sp600_mem", {})

    # Build SP1500 ticker set per date
    def get_sp1500_tickers(date):
        members = set()
        for mem in [sp500_mem, sp400_mem, sp600_mem]:
            prior = [d for d in mem.keys() if d <= date]
            if prior:
                members.update(mem[max(prior)])
        return members

    # Map SP1500 tickers to PERMNOs
    ticker_permno = {}
    for permno, ticker in permno_ticker.items():
        ticker_permno.setdefault(ticker, []).append(permno)

    has_sp1500 = True
    print(f"  SP1500 membership loaded")
except Exception as e:
    print(f"  WARNING: Could not load SP1500: {e}")
    has_sp1500 = False

# ─── 5. Backtest function ───────────────────────────────────────────────────
def run_backtest(name, universe_filter="all", top_n=TOP_N):
    """
    Run a proper portfolio simulation.

    universe_filter: "all" for expanded, "sp1500" for SP1500 only
    """
    print(f"\n{'─'*60}")
    print(f"  Running backtest: {name}")
    print(f"  Universe: {universe_filter}, Top-N: {top_n}")
    print(f"  Trailing stop: {TRAILING_STOP*100:.0f}%, Position cap: {POSITION_CAP*100:.1f}%")
    print(f"  Cost: {COST_BPS}bp, Rebal: {REBAL_DAYS}d")
    print(f"{'─'*60}")

    trading_dates = price_piv.index[
        (price_piv.index >= pd.Timestamp(START)) &
        (price_piv.index <= pd.Timestamp(END))
    ]

    cost_frac = COST_BPS / 10000
    cash = INITIAL_CASH
    holdings = {}  # PERMNO -> {shares, entry_px, peak_px}
    port_values = []
    trade_count = 0
    stocks_in_expanded_not_sp1500 = 0
    total_picks = 0

    for day_idx, date in enumerate(trading_dates):
        prices_today = price_piv.loc[date].dropna()

        # ── Trailing stop check ──
        for permno in list(holdings):
            px = prices_today.get(permno)
            if px and px > 0:
                if px > holdings[permno]["peak_px"]:
                    holdings[permno]["peak_px"] = px
                dd = (px - holdings[permno]["peak_px"]) / holdings[permno]["peak_px"]
                if dd < -TRAILING_STOP:
                    cash += holdings[permno]["shares"] * px * (1 - cost_frac)
                    del holdings[permno]
                    trade_count += 1

        # ── Portfolio value ──
        eq_val = cash
        for permno, h in holdings.items():
            px = prices_today.get(permno, h["entry_px"])
            eq_val += h["shares"] * px
        total_val = eq_val

        # ── Rebalance ──
        if day_idx % REBAL_DAYS == 0:
            # Filter universe
            valid_permnos = set(prices_today.index)

            # Liquidity filters
            caps = recent_cap.loc[date].reindex(list(valid_permnos))
            vols = avg_vol_60.loc[date].reindex(list(valid_permnos))
            prcs = prices_today.reindex(list(valid_permnos))

            mask = (prcs >= MIN_PRICE) & (caps >= MIN_MCAP) & (vols >= MIN_AVG_VOL)
            mask = mask.fillna(False)
            eligible = set(mask[mask].index)

            # SP1500 filter if needed
            if universe_filter == "sp1500" and has_sp1500:
                sp1500_tickers = get_sp1500_tickers(date)
                sp1500_permnos = set()
                for t in sp1500_tickers:
                    for p in ticker_permno.get(t, []):
                        sp1500_permnos.add(p)
                eligible = eligible & sp1500_permnos

            # Higher market cap filter ($2B+)
            if universe_filter == "all_large":
                large_caps = caps[caps >= 2_000_000].index  # $2B in CRSP units
                eligible = eligible & set(large_caps)

            # Momentum cap for "capped" variant
            mom_cap = 3.0 if universe_filter == "all_capped" else None

            # Get momentum scores
            mom_scores = skip_mom.loc[date].reindex(list(eligible)).dropna()
            sma_filter = above_sma200.loc[date].reindex(list(eligible))

            # Apply SMA200 filter
            above = sma_filter[sma_filter == True].index
            mom_scores = mom_scores.reindex(above).dropna()

            # Consolidation boost: high momentum + low vol = coiled spring
            v20 = vol_20d.loc[date].reindex(mom_scores.index)
            for p in mom_scores.index:
                v = v20.get(p)
                if v is not None and not np.isnan(v) and v < 0.25 and mom_scores[p] > 0.20:
                    mom_scores[p] *= 1.15

            # Cap extreme momentum if requested
            if mom_cap is not None:
                mom_scores = mom_scores[mom_scores <= mom_cap]

            # Pick top N
            if len(mom_scores) < 5:
                port_values.append((date, total_val))
                continue

            picks = mom_scores.nlargest(top_n).index.tolist()

            # Track how many picks are outside SP1500
            if has_sp1500 and universe_filter == "all":
                sp1500_tickers_now = get_sp1500_tickers(date)
                sp1500_permnos_now = set()
                for t in sp1500_tickers_now:
                    for p in ticker_permno.get(t, []):
                        sp1500_permnos_now.add(p)
                non_sp1500 = sum(1 for p in picks if p not in sp1500_permnos_now)
                stocks_in_expanded_not_sp1500 += non_sp1500
                total_picks += len(picks)

            # Equal weight with position cap
            weights = {p: 1.0 / len(picks) for p in picks}
            for p in weights:
                if weights[p] > POSITION_CAP:
                    weights[p] = POSITION_CAP
            total_w = sum(weights.values())
            if total_w > 1.0:
                weights = {p: w / total_w for p, w in weights.items()}

            target_vals = {p: w * total_val for p, w in weights.items()}

            # Sell positions not in target
            for permno in list(holdings):
                if permno not in target_vals:
                    px = prices_today.get(permno, holdings[permno]["entry_px"])
                    cash += holdings[permno]["shares"] * px * (1 - cost_frac)
                    del holdings[permno]
                    trade_count += 1

            # Rebalance existing and buy new
            for permno, tgt in target_vals.items():
                px = prices_today.get(permno)
                if not px or px <= 0:
                    continue
                cur_val = holdings[permno]["shares"] * px if permno in holdings else 0
                delta = tgt - cur_val
                if abs(delta) < total_val * 0.003:
                    continue  # skip small rebalances
                cost = abs(delta) * cost_frac
                if delta > 0 and cash >= delta:
                    shares = (delta - cost) / px
                    if permno in holdings:
                        holdings[permno]["shares"] += shares
                    else:
                        holdings[permno] = {"shares": shares, "entry_px": px, "peak_px": px}
                    cash -= delta
                    trade_count += 1
                elif delta < 0 and permno in holdings:
                    sell = min(abs(delta) / px, holdings[permno]["shares"])
                    cash += sell * px - cost
                    holdings[permno]["shares"] -= sell
                    if holdings[permno]["shares"] < 0.01:
                        del holdings[permno]
                    trade_count += 1

        port_values.append((date, max(total_val, 0)))

    # ── Compute metrics ──
    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    sd = dr[dr < 0].std()
    sortino = dr.mean() / sd * np.sqrt(252) if sd > 0 else 0
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    vol = dr.std() * np.sqrt(252)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Yearly breakdown
    yearly = {}
    for year in range(2017, 2026):
        mask = (vals.index >= f"{year}-01-01") & (vals.index <= f"{year}-12-31")
        yv = vals[mask]
        if len(yv) > 10:
            yr = (yv.iloc[-1] / yv.iloc[0]) - 1
            ydr = yv.pct_change().dropna()
            ys = ydr.mean() / ydr.std() * np.sqrt(252) if ydr.std() > 0 else 0
            ydd = ((yv - yv.cummax()) / yv.cummax()).min()
            yearly[year] = {"ret": yr, "sharpe": ys, "max_dd": ydd}

    # Print results
    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │ {name:^31} │")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │ CAGR:      {cagr:>8.1%}             │")
    print(f"  │ Sharpe:    {sharpe:>8.2f}             │")
    print(f"  │ Sortino:   {sortino:>8.2f}             │")
    print(f"  │ Max DD:    {max_dd:>8.1%}             │")
    print(f"  │ Vol:       {vol:>8.1%}             │")
    print(f"  │ Calmar:    {calmar:>8.2f}             │")
    print(f"  │ Trades:    {trade_count:>8,}             │")
    print(f"  │ Final:     ${vals.iloc[-1]:>10,.0f}          │")
    print(f"  └─────────────────────────────────┘")

    print(f"\n  Year-by-year:")
    print(f"  {'Year':>4}  {'Return':>8}  {'Sharpe':>7}  {'Max DD':>8}")
    for year in sorted(yearly):
        y = yearly[year]
        print(f"  {year:>4}  {y['ret']:>8.1%}  {y['sharpe']:>7.2f}  {y['max_dd']:>8.1%}")

    if universe_filter == "all" and total_picks > 0:
        pct_outside = stocks_in_expanded_not_sp1500 / total_picks * 100
        print(f"\n  Picks outside SP1500: {stocks_in_expanded_not_sp1500}/{total_picks} ({pct_outside:.1f}%)")

    return {
        "name": name, "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "vol": vol, "calmar": calmar, "trades": trade_count,
        "final": vals.iloc[-1], "yearly": yearly, "values": vals,
    }


# ─── 6. Run both backtests ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RUNNING BACKTESTS")
print("=" * 70)

r_sp1500 = run_backtest("SP1500 Momentum (Baseline)", universe_filter="sp1500", top_n=TOP_N)
r_expanded = run_backtest("ALL Stocks Expanded", universe_filter="all", top_n=TOP_N)

# Also test with more picks from expanded universe
r_expanded_12 = run_backtest("Expanded Top-12", universe_filter="all", top_n=12)

# Test with higher market cap filter ($2B+) - "SP1500-adjacent"
r_large = run_backtest("Expanded $2B+ Top-8", universe_filter="all_large", top_n=TOP_N)

# Test with momentum cap (exclude >300% to avoid blow-up names)
r_capped = run_backtest("Expanded MomCap Top-8", universe_filter="all_capped", top_n=TOP_N)

# ─── 7. Comparison ──────────────────────────────────────────────────────────
all_results = [r_sp1500, r_expanded, r_expanded_12, r_large, r_capped]

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

headers = [r["name"][:18] for r in all_results]
print(f"\n  {'Metric':<15}", "  ".join(f"{h:>18}" for h in headers))
print(f"  {'─'*15}", "  ".join("─"*18 for _ in headers))
for metric, key, fmt in [("CAGR","cagr",".1%"),("Sharpe","sharpe",".2f"),
                          ("Sortino","sortino",".2f"),("Max DD","max_dd",".1%"),
                          ("Vol","vol",".1%"),("Calmar","calmar",".2f"),
                          ("Trades","trades",",")]:
    vals = []
    for r in all_results:
        v = r[key]
        if fmt == ",":
            vals.append(f"{v:>18,}")
        elif fmt == ".1%":
            vals.append(f"{v:>18.1%}")
        else:
            vals.append(f"{v:>18{fmt}}")
    print(f"  {metric:<15}", "  ".join(vals))

# Year-by-year comparison
print(f"\n  Year-by-year returns:")
print(f"  {'Year':>4}", "  ".join(f"{r['name'][:14]:>14}" for r in all_results))
for year in range(2017, 2026):
    vals = []
    for r in all_results:
        y = r["yearly"].get(year, {})
        ret = y.get("ret", 0)
        vals.append(f"{ret:>14.1%}")
    print(f"  {year:>4}", "  ".join(vals))

# ─── 8. Sample picks comparison ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("SAMPLE: Top momentum stocks on last rebalance date")
print("=" * 70)

# Show what stocks the expanded universe picks that SP1500 doesn't
sample_date = trading_dates = price_piv.index[
    (price_piv.index >= pd.Timestamp(START)) &
    (price_piv.index <= pd.Timestamp(END))
][-1]  # Last date

prcs = price_piv.loc[sample_date].dropna()
caps = recent_cap.loc[sample_date].reindex(prcs.index)
vols = avg_vol_60.loc[sample_date].reindex(prcs.index)
mask = (prcs >= MIN_PRICE) & (caps >= MIN_MCAP) & (vols >= MIN_AVG_VOL)
mask = mask.fillna(False)
eligible_all = set(mask[mask].index)

mom = skip_mom.loc[sample_date].reindex(list(eligible_all)).dropna()
sma_ok = above_sma200.loc[sample_date].reindex(list(eligible_all))
above = sma_ok[sma_ok == True].index
mom = mom.reindex(above).dropna()
top20 = mom.nlargest(20)

if has_sp1500:
    sp1500_t = get_sp1500_tickers(sample_date)
    sp1500_p = set()
    for t in sp1500_t:
        for p in ticker_permno.get(t, []):
            sp1500_p.add(p)

print(f"\n  Top 20 momentum stocks on {sample_date.date()}:")
print(f"  {'Rank':>4}  {'Ticker':<8}  {'Skip-Mom':>10}  {'MCap ($M)':>10}  {'In SP1500':>10}")
for i, (permno, score) in enumerate(top20.items(), 1):
    ticker = permno_ticker.get(permno, f"P{permno}")
    mcap = caps.get(permno, 0) / 1e6
    in_sp = "YES" if (has_sp1500 and permno in sp1500_p) else "NO"
    print(f"  {i:>4}  {ticker:<8}  {score:>10.1%}  {mcap:>10,.0f}  {in_sp:>10}")

print(f"\n  SP1500 universe size: ~{len(sp1500_p):,} PERMNOs")
print(f"  Expanded eligible: {len(eligible_all):,} PERMNOs")
print(f"  Difference: {len(eligible_all) - len(sp1500_p):,} additional stocks")

print(f"\n\nTotal runtime: {time.time()-t0:.1f}s")
