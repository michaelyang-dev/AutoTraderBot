"""
ML-Enhanced Momentum Test
=========================
Compares:
  (a) Standard v10.1 momentum top-8
  (b) ML-filtered: momentum top-16, re-ranked by ML score, select top-8

Uses the exact FastBacktester infrastructure but overrides the momentum
strategy to inject ML re-ranking.
"""

import numpy as np
import pandas as pd
import pickle
import time
import logging
from pathlib import Path

from wrds_universe import WRDSUniverse
from wrds_data_provider import SP500Membership, WRDSDataProvider
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, INITIAL_CASH, COST_BPS,
)
from fast_backtest import FastBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ml_enhance")

SLIPPAGE_BPS = 5

# ── Load ML predictions ─────────────────────────────────────────────────────
ml_preds = pd.read_parquet("/tmp/ml_v2_preds.parquet")
ml_preds["date"] = pd.to_datetime(ml_preds["date"])
# Build lookup: {date -> {symbol -> ml_score}}
ml_by_date = {}
for dt, grp in ml_preds.groupby("date"):
    ml_by_date[dt] = dict(zip(grp["symbol"], grp["ml_score"]))
log.info(f"ML predictions loaded: {len(ml_by_date)} dates, {len(ml_preds)} rows")

# Find nearest ML date for a given rebalance date
ml_dates_sorted = sorted(ml_by_date.keys())

def get_ml_scores(date):
    """Get ML scores for the nearest date <= rebalance date."""
    idx = np.searchsorted(ml_dates_sorted, date, side="right") - 1
    if idx < 0:
        return {}
    ml_date = ml_dates_sorted[idx]
    # Only use if within 30 days
    if (date - ml_date).days > 30:
        return {}
    return ml_by_date[ml_date]


def run_backtest(bt, start, end, config, ml_filter=False):
    """
    Run backtest with optional ML filtering.
    If ml_filter=True: momentum gets top_n=16, then ML re-ranks to top 8.
    """
    universe = config.get("universe", "sp1500")
    if universe == "sp1500":
        bt.uni.get_sp500 = bt._get_sp1500
    else:
        bt.uni.get_sp500 = bt._original_get_sp500

    trading_dates = [d for d in sorted(bt.prices.index)
                     if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_dates:
        return None

    cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
    mom_w = config.get("mom_w", 0.85)
    val_w = config.get("val_w", 0.15)
    lv_w = config.get("lv_w", 0.0)
    sec_w = config.get("sec_w", 0.0)
    top_n = config.get("top_n", 8)
    cap = config.get("cap", 0.15)
    use_rp = config.get("use_rp", True)
    rp_power = config.get("rp_power", 1.0)
    gld_pct = config.get("gld_pct", 0.02)
    vixm_pct = config.get("vixm_pct", 0.0)
    rebal_days = config.get("rebal_days", 10)
    trailing_stop = config.get("trailing_stop", 0.25)
    vol_scaling = config.get("vol_scaling", False)
    vol_target = config.get("vol_target", 0.20)
    recent_rets = []

    # For ML filter: request 16 from momentum, then filter to 8
    mom_top_n = 16 if ml_filter else top_n

    cash = INITIAL_CASH
    holdings = {}
    port_values = []
    last_targets = {}
    gld_shares = 0
    vixm_shares = 0

    if vixm_pct > 0 and trading_dates[0] in bt.etf_df.index and "VIXM" in bt.etf_df.columns:
        vp = bt.etf_df.loc[trading_dates[0], "VIXM"]
        if pd.notna(vp) and vp > 0:
            vixm_shares = (INITIAL_CASH * vixm_pct) / vp
            cash -= INITIAL_CASH * vixm_pct

    ml_hit_count = 0
    ml_miss_count = 0

    for day_idx, date in enumerate(trading_dates):
        today = {}
        if date in bt.prices.index:
            row = bt.prices.loc[date]
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
        gld_val = 0
        if gld_shares > 0 and date in bt.etf_df.index and "GLD" in bt.etf_df.columns:
            gp = bt.etf_df.loc[date, "GLD"]
            if pd.notna(gp):
                gld_val = gld_shares * gp
        vixm_val = 0
        if vixm_shares > 0 and date in bt.etf_df.index and "VIXM" in bt.etf_df.columns:
            vp = bt.etf_df.loc[date, "VIXM"]
            if pd.notna(vp):
                vixm_val = vixm_shares * vp
        total_val = eq_val + gld_val + vixm_val

        if len(port_values) > 0:
            prev = port_values[-1][1]
            if prev > 0:
                recent_rets.append(total_val / prev - 1)
                if len(recent_rets) > 40:
                    recent_rets.pop(0)

        if day_idx % rebal_days != 0:
            port_values.append((date, total_val))
            continue

        # Update short interest ranks
        if bt._si_months:
            midx = np.searchsorted(bt._si_months, date, side="right") - 1
            if midx >= 0:
                bt.uni._short_interest_rank = bt._si_ranks_by_month.get(
                    bt._si_months[midx], {})
                bt.uni._si_change_rank = bt._si_change_ranks_by_month.get(
                    bt._si_months[midx], {})
            else:
                bt.uni._short_interest_rank = {}
                bt.uni._si_change_rank = {}

        # ── Strategy signals ─────────────────────────────────────────
        t1 = strategy1_momentum_reversal(date, bt.uni, day_idx,
                                         top_n=mom_top_n, rebal_days=rebal_days)
        if t1 is None:
            t1 = last_targets.get("mom", {})

        # ML filtering: re-rank the 16 momentum picks by ML score, take top 8
        if ml_filter and t1 and len(t1) > top_n:
            ml_scores = get_ml_scores(date)
            if ml_scores:
                # Score each pick
                scored = []
                for sym, w in t1.items():
                    ms = ml_scores.get(sym)
                    if ms is not None:
                        scored.append((sym, ms))
                    else:
                        # No ML score: use median as neutral
                        scored.append((sym, 0.0))

                # Sort by ML score descending, take top_n
                scored.sort(key=lambda x: x[1], reverse=True)
                kept = [s for s, _ in scored[:top_n]]

                # Re-weight equally (then RP will adjust)
                t1 = {s: 1.0 / len(kept) for s in kept}
                ml_hit_count += 1
            else:
                # No ML data for this date: just take top_n by momentum score
                sorted_syms = sorted(t1, key=t1.get, reverse=True)[:top_n]
                t1 = {s: 1.0 / len(sorted_syms) for s in sorted_syms}
                ml_miss_count += 1

        members = bt.uni.get_sp500(date)
        t_val = bt._strategy_value(date, members, top_n=10)
        t3 = strategy3_sector_rotation(date, bt.uni, day_idx)
        t5 = strategy5_lowvol_quality(date, bt.uni, day_idx)
        if t3 is None:
            t3 = last_targets.get("s3", {})
        if t5 is None:
            t5 = last_targets.get("s5", {})
        last_targets.update({"mom": t1, "val": t_val, "s3": t3, "s5": t5})

        # UMD regime
        nu = bt.umd_20d.loc[:date]
        in_crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < -0.05
        if in_crash:
            ew = {"mom": 0.15, "val": 0.45, "s5": 0.30, "s3": 0.10}
        else:
            ew = {"mom": mom_w, "val": val_w, "s5": lv_w, "s3": sec_w}

        # Breadth
        fdate = bt.features_by_date.get(date, {})
        above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
        total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
        breadth = above / max(total_f, 1)
        blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))
        bear = {"mom": 0.10, "val": 0.20, "s5": 0.60, "s3": 0.10}
        blended = {n: ew[n] * blend + bear.get(n, 0) * (1 - blend) for n in ew}

        combined = {}
        for name, cap_pct in blended.items():
            tgt = last_targets.get(name, {})
            if use_rp and name in ("mom", "val"):
                tgt = bt._apply_rp(tgt, date, power=rp_power)
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

        eq_pct = 1.0 - vixm_pct - (gld_pct if gld_pct > 0 else 0)
        target_d = {s: w * total_val * eq_pct for s, w in combined.items()}

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

        # GLD trend
        buy_gld = False
        if gld_pct > 0 and "GLD" in bt.etf_df.columns:
            gld_px = bt.etf_df["GLD"].loc[:date].dropna()
            if len(gld_px) >= 252:
                buy_gld = gld_px.iloc[-1] > gld_px.iloc[-252:].mean()
        if gld_shares > 0 and not buy_gld:
            if date in bt.etf_df.index:
                gp = bt.etf_df.loc[date, "GLD"]
                if pd.notna(gp) and gp > 0:
                    cash += gld_shares * gp * (1 - cost_frac)
            gld_shares = 0
        if buy_gld and gld_shares == 0:
            gld_amt = total_val * gld_pct
            if date in bt.etf_df.index and cash >= gld_amt:
                gp = bt.etf_df.loc[date, "GLD"]
                if pd.notna(gp) and gp > 0:
                    gld_shares = (gld_amt - gld_amt * cost_frac) / gp
                    cash -= gld_amt

        eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                            for s, h in holdings.items())
        gld_val = 0
        if gld_shares > 0 and date in bt.etf_df.index and "GLD" in bt.etf_df.columns:
            gp = bt.etf_df.loc[date, "GLD"]
            if pd.notna(gp):
                gld_val = gld_shares * gp
        total_val = eq_val + gld_val + vixm_val
        port_values.append((date, max(total_val, 0)))

    if ml_filter:
        log.info(f"ML filter applied on {ml_hit_count} rebalances, missed {ml_miss_count}")

    # Metrics
    vals = pd.Series([v for _, v in port_values],
                     index=pd.DatetimeIndex([d for d, _ in port_values]))
    years = (vals.index[-1] - vals.index[0]).days / 365.25
    if years <= 0:
        years = 1
    dr = vals.pct_change().dropna()
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    sd = dr[dr < 0].std()
    sortino = dr.mean() / sd * np.sqrt(252) if sd > 0 else 0
    peak = vals.cummax()
    max_dd = ((vals - peak) / peak).min()
    vol = dr.std() * np.sqrt(252)

    spy = bt.prices["SPY"].reindex(vals.index, method="ffill").dropna()
    spy = spy / spy.iloc[0] * INITIAL_CASH
    spy_cagr = (spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1

    yearly = {}
    for year in range(2019, 2026):
        mask = (vals.index >= f"{year}-01-01") & (vals.index <= f"{year}-12-31")
        yv = vals[mask]
        if len(yv) > 10:
            yr = (yv.iloc[-1] / yv.iloc[0]) - 1
            ydr = yv.pct_change().dropna()
            ys = ydr.mean() / ydr.std() * np.sqrt(252) if ydr.std() > 0 else 0
            ydd = ((yv - yv.cummax()) / yv.cummax()).min()
            yearly[year] = {"ret": yr, "sharpe": ys, "max_dd": ydd}

    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd, "vol": vol, "alpha": cagr - spy_cagr,
        "final": vals.iloc[-1], "yearly": yearly,
    }


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester()
    log.info(f"Backtester loaded in {time.time() - t0:.1f}s")

    cfg = {
        "universe": "sp1500",
        "mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0,
        "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25,
        "gld_pct": 0.02, "vixm_pct": 0.0,
        "use_rp": True, "rp_power": 1.0, "cap": 0.15,
    }

    # (a) Standard momentum top-8
    log.info("Running STANDARD momentum top-8...")
    t1 = time.time()
    res_std = run_backtest(bt, "2019-01-01", "2025-12-31", cfg, ml_filter=False)
    log.info(f"Standard done in {time.time() - t1:.1f}s")

    # (b) ML-filtered momentum top-8 (from top-16)
    log.info("Running ML-FILTERED momentum top-8 (from top-16)...")
    t2 = time.time()
    res_ml = run_backtest(bt, "2019-01-01", "2025-12-31", cfg, ml_filter=True)
    log.info(f"ML-filtered done in {time.time() - t2:.1f}s")

    # Report
    print("\n" + "=" * 75)
    print("ML-ENHANCED MOMENTUM TEST: 2019-2025")
    print("=" * 75)

    print(f"\n{'Metric':<25} {'Standard Top-8':>18} {'ML-Filtered Top-8':>20}")
    print("-" * 65)
    print(f"{'CAGR':<25} {res_std['cagr']:>+17.1%} {res_ml['cagr']:>+19.1%}")
    print(f"{'Sharpe':<25} {res_std['sharpe']:>18.2f} {res_ml['sharpe']:>20.2f}")
    print(f"{'Sortino':<25} {res_std['sortino']:>18.2f} {res_ml['sortino']:>20.2f}")
    print(f"{'Max Drawdown':<25} {res_std['max_dd']:>17.1%} {res_ml['max_dd']:>19.1%}")
    print(f"{'Volatility':<25} {res_std['vol']:>17.1%} {res_ml['vol']:>19.1%}")
    print(f"{'Alpha vs SPY':<25} {res_std['alpha']:>+17.1%} {res_ml['alpha']:>+19.1%}")
    print(f"{'Final Value ($100k)':<25} {res_std['final']:>17,.0f} {res_ml['final']:>19,.0f}")

    print(f"\n{'Year':<10} {'Std Return':>12} {'Std Sharpe':>12} {'ML Return':>12} {'ML Sharpe':>12} {'Delta':>8}")
    print("-" * 68)
    for year in range(2019, 2026):
        ys = res_std["yearly"].get(year, {})
        ym = res_ml["yearly"].get(year, {})
        sr = ys.get("ret", 0)
        ss = ys.get("sharpe", 0)
        mr = ym.get("ret", 0)
        ms = ym.get("sharpe", 0)
        delta = mr - sr
        print(f"  {year:<8} {sr:>+11.1%} {ss:>11.2f} {mr:>+11.1%} {ms:>11.2f} {delta:>+7.1%}")

    print("\n" + "=" * 75)
    cagr_diff = res_ml["cagr"] - res_std["cagr"]
    sharpe_diff = res_ml["sharpe"] - res_std["sharpe"]
    print(f"CAGR improvement:   {cagr_diff:+.1%}")
    print(f"Sharpe improvement: {sharpe_diff:+.2f}")
    if cagr_diff > 0.01 and sharpe_diff > 0.05:
        print("VERDICT: ML filtering IMPROVES risk-adjusted returns.")
    elif cagr_diff > 0.01:
        print("VERDICT: ML filtering improves returns but not risk-adjusted.")
    elif sharpe_diff > 0.05:
        print("VERDICT: ML filtering improves Sharpe but not raw returns.")
    else:
        print("VERDICT: ML filtering does NOT meaningfully improve results.")
    print("=" * 75)
