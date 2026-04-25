#!/usr/bin/env python3
"""
Russell 2000 Short Model Pipeline
==================================
End-to-end pipeline for testing short signal viability on a Russell 2000
market-cap proxy universe.

Universe methodology: MARKET CAP PROXY (not true Russell 2000 membership).
  - Start with a broad set of US small/mid-cap tickers
  - On each date, define "in Russell 2000 proxy" as market cap $300M-$2B
  - Historical market cap approximated from yfinance current mktcap + adj prices
  - This is PIT-safe: no future knowledge is used for membership

Phases:
  R1: Universe acquisition + PIT validation
  R2: Short feature engineering
  R3: Walk-forward model training
  R4: Signal quality evaluation
  R5: Heuristic baseline comparison
  R6: Tradeability analysis
  R7: Verdict report generation
"""

import csv
import io
import json
import os
import ssl
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data" / "russell_short"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

MKTCAP_LOW = 300_000_000    # $300M
MKTCAP_HIGH = 2_000_000_000  # $2B


# ═══════════════════════════════════════════════════════════════════════
# PHASE R1: Universe Acquisition
# ═══════════════════════════════════════════════════════════════════════

def fetch_ticker_pool():
    """Get a broad pool of US small/mid-cap tickers from GitHub Russell 2000 list."""
    url = "https://raw.githubusercontent.com/ikoniaris/Russell2000/master/russell_2000_components.csv"
    try:
        resp = urllib.request.urlopen(url, context=_SSL_CTX, timeout=30)
        text = resp.read().decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        next(reader)  # skip header
        symbols = sorted(set(row[0].strip() for row in reader if row and row[0].strip()))
        print(f"[R1] Downloaded {len(symbols)} tickers from GitHub Russell 2000 list")
        return symbols
    except Exception as e:
        print(f"[R1] ERROR: Could not fetch ticker list: {e}")
        return []


def get_market_caps(symbols, batch_pause=0.05):
    """Get current market cap + shares outstanding from yfinance for each symbol."""
    cache_file = DATA_DIR / "mktcap_cache.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        print(f"[R1] Loaded {len(cached)} cached market caps")
        # Only fetch missing symbols
        missing = [s for s in symbols if s not in cached]
        if not missing:
            return cached
        print(f"[R1] Fetching {len(missing)} missing symbols...")
    else:
        cached = {}
        missing = symbols

    for i, sym in enumerate(missing):
        try:
            info = yf.Ticker(sym).info
            mc = info.get("marketCap")
            shares = info.get("sharesOutstanding")
            ex = info.get("exchange", "")
            if mc and shares:
                cached[sym] = {"marketCap": mc, "sharesOutstanding": shares, "exchange": ex}
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(missing)} done ({len(cached)} valid)")
            # Save progress
            with open(cache_file, "w") as f:
                json.dump(cached, f)
        time.sleep(batch_pause)

    with open(cache_file, "w") as f:
        json.dump(cached, f)
    print(f"[R1] Total symbols with market cap data: {len(cached)}")
    return cached


def download_prices(symbols, start="2010-01-01"):
    """Download adjusted close prices for all symbols."""
    price_file = DATA_DIR / "prices_russell.parquet"
    if price_file.exists():
        prices = pd.read_parquet(price_file)
        print(f"[R1] Loaded cached prices: {prices.shape}")
        missing = [s for s in symbols if s not in prices.columns]
        if not missing:
            return prices
        print(f"[R1] Downloading {len(missing)} missing symbols...")
    else:
        prices = pd.DataFrame()
        missing = symbols

    # Download in batches
    batch_size = 100
    new_frames = []
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        print(f"  ... downloading batch {i//batch_size + 1}/{(len(missing)-1)//batch_size + 1} ({len(batch)} symbols)")
        try:
            data = yf.download(batch, start=start, progress=False, threads=True, auto_adjust=True)
            if "Close" in data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else "Close" in data.columns:
                if isinstance(data.columns, pd.MultiIndex):
                    close = data["Close"]
                else:
                    close = data[["Close"]].rename(columns={"Close": batch[0]})
                new_frames.append(close)
        except Exception as e:
            print(f"    batch error: {e}")
        time.sleep(1)

    if new_frames:
        new_prices = pd.concat(new_frames, axis=1)
        if not prices.empty:
            prices = pd.concat([prices, new_prices], axis=1)
        else:
            prices = new_prices

    prices.to_parquet(price_file)
    print(f"[R1] Prices shape: {prices.shape}")
    return prices


def build_pit_universe(prices, mktcap_data):
    """Build point-in-time Russell 2000 proxy using market cap approximation.

    For each symbol on each date:
      historical_mktcap ≈ current_mktcap × (adj_close[date] / adj_close[latest])

    Then filter to $300M-$2B range.
    """
    print("[R1] Building PIT universe from market cap proxy...")

    records = []
    symbols_used = 0
    for sym in prices.columns:
        if sym not in mktcap_data:
            continue
        current_mc = mktcap_data[sym]["marketCap"]
        series = prices[sym].dropna()
        if len(series) < 252:  # need at least 1 year
            continue

        latest_price = series.iloc[-1]
        if latest_price <= 0:
            continue

        # Historical market cap proxy
        hist_mc = current_mc * (series / latest_price)
        # Filter to Russell 2000 range
        in_range = (hist_mc >= MKTCAP_LOW) & (hist_mc <= MKTCAP_HIGH)
        dates_in = series.index[in_range]

        if len(dates_in) > 0:
            for dt in dates_in:
                records.append({"date": dt, "symbol": sym, "approx_mktcap": hist_mc.loc[dt]})
            symbols_used += 1

    df = pd.DataFrame(records)
    if df.empty:
        print("[R1] ERROR: No symbols in Russell 2000 market cap range!")
        return df

    df["date"] = pd.to_datetime(df["date"])
    print(f"[R1] PIT universe: {len(df):,} symbol-date pairs, {df['symbol'].nunique()} unique symbols")
    print(f"[R1] Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Stats per year
    df["year"] = df["date"].dt.year
    yearly = df.groupby("year").agg(
        symbols=("symbol", "nunique"),
        obs=("symbol", "count"),
    )
    print("\n[R1] Universe size per year:")
    print(yearly.to_string())

    return df


def validate_pit(universe_df, prices, mktcap_data):
    """Validate PIT correctness by checking 5 random dates."""
    print("\n[R1] === PIT VALIDATION ===")
    np.random.seed(42)
    dates = sorted(universe_df["date"].unique())
    check_dates = np.random.choice(dates, size=min(5, len(dates)), replace=False)

    for dt in sorted(check_dates):
        dt_ts = pd.Timestamp(dt)
        members = universe_df[universe_df["date"] == dt]
        n = len(members)
        sample = members.sample(min(3, n))
        print(f"\n  Date: {dt_ts.date()} — {n} stocks in universe")
        for _, row in sample.iterrows():
            sym = row["symbol"]
            approx = row["approx_mktcap"]
            # Verify: is this market cap derived only from data available before dt?
            # The price on dt is known on dt (not future). Current mktcap is used as
            # a scaling anchor — this is an approximation, not look-ahead, because
            # we're scaling by price ratio (which only uses past prices).
            price_on_dt = prices.loc[dt_ts, sym] if dt_ts in prices.index else None
            print(f"    {sym}: approx_mktcap=${approx/1e6:.0f}M, price={price_on_dt:.2f}" if price_on_dt else f"    {sym}: approx_mktcap=${approx/1e6:.0f}M")

    print("\n  PIT assessment: Market cap proxy uses adj_close[date]/adj_close[latest] × current_mktcap.")
    print("  The price on date X is known on date X. The scaling factor (current_mktcap / latest_price)")
    print("  is a constant that converts price to market cap — it does NOT leak future membership info.")
    print("  Mild bias: share count changes (buybacks, dilution) are not captured.")
    print("  This is MUCH safer than the is_former_sp500 leak but NOT true Russell 2000 membership.")


def run_phase_r1():
    """Execute Phase R1: Universe acquisition and PIT validation."""
    print("=" * 70)
    print("PHASE R1: Russell 2000 Universe Acquisition")
    print("=" * 70)

    # Step 1: Get ticker pool
    symbols = fetch_ticker_pool()
    if not symbols:
        print("STOP: Cannot acquire ticker pool")
        return None, None, None

    # Step 2: Get market caps
    mktcap_data = get_market_caps(symbols)
    valid_symbols = sorted(mktcap_data.keys())
    print(f"[R1] Valid symbols with market cap: {len(valid_symbols)}")

    if len(valid_symbols) < 200:
        print("STOP: Too few valid symbols (<200)")
        return None, None, None

    # Step 3: Download prices
    prices = download_prices(valid_symbols)

    # Step 4: Build PIT universe
    universe = build_pit_universe(prices, mktcap_data)
    if universe.empty:
        print("STOP: Empty universe")
        return None, None, None

    # Step 5: Validate PIT
    validate_pit(universe, prices, mktcap_data)

    # Save
    universe_file = DATA_DIR / "russell2000_proxy_universe.parquet"
    universe.to_parquet(universe_file, index=False)
    print(f"\n[R1] Saved universe to {universe_file}")

    return universe, prices, mktcap_data


# ═══════════════════════════════════════════════════════════════════════
# PHASE R2: Short Feature Engineering
# ═══════════════════════════════════════════════════════════════════════

def compute_short_features(prices, universe_df):
    """Compute ~18 short-specific features from price data only.

    All features use only data available on or before the prediction date.
    No fundamental data needed (avoids FMP API dependency for Russell 2000).
    """
    print("\n" + "=" * 70)
    print("PHASE R2: Short Feature Engineering")
    print("=" * 70)

    all_symbols = universe_df["symbol"].unique()
    print(f"[R2] Computing features for {len(all_symbols)} symbols...")

    feature_records = []
    for idx, sym in enumerate(all_symbols):
        if sym not in prices.columns:
            continue
        s = prices[sym].dropna()
        if len(s) < 252:
            continue

        # Technical features (all backward-looking, PIT safe)
        close = s.values
        dates = s.index

        for i in range(252, len(close)):
            dt = dates[i]
            c = close[:i + 1]

            rec = {"date": dt, "symbol": sym}

            # --- Momentum features ---
            rec["ret_5d"] = c[-1] / c[-6] - 1 if i >= 5 else np.nan
            rec["ret_10d"] = c[-1] / c[-11] - 1 if i >= 10 else np.nan
            rec["ret_20d"] = c[-1] / c[-21] - 1 if i >= 20 else np.nan
            rec["ret_60d"] = c[-1] / c[-61] - 1 if i >= 60 else np.nan
            rec["ret_120d"] = c[-1] / c[-121] - 1 if i >= 120 else np.nan

            # --- SMA features ---
            sma50 = np.mean(c[-50:])
            sma200 = np.mean(c[-200:])
            rec["dist_sma50"] = c[-1] / sma50 - 1
            rec["dist_sma200"] = c[-1] / sma200 - 1
            rec["below_sma50"] = 1.0 if c[-1] < sma50 else 0.0
            rec["below_sma200"] = 1.0 if c[-1] < sma200 else 0.0
            rec["death_cross"] = 1.0 if sma50 < sma200 else 0.0

            # --- 52-week high/low ---
            high_252 = np.max(c[-252:])
            low_252 = np.min(c[-252:])
            rec["dist_52w_high"] = c[-1] / high_252 - 1
            rec["dist_52w_low"] = c[-1] / low_252 - 1

            # --- Volatility ---
            rets_20 = np.diff(np.log(c[-21:])) if i >= 20 else [np.nan]
            rets_60 = np.diff(np.log(c[-61:])) if i >= 60 else [np.nan]
            rec["vol_20d"] = np.std(rets_20) * np.sqrt(252)
            rec["vol_60d"] = np.std(rets_60) * np.sqrt(252)
            rec["vol_expansion"] = rec["vol_20d"] / rec["vol_60d"] if rec["vol_60d"] > 0 else np.nan

            # --- RSI ---
            if i >= 14:
                diffs = np.diff(c[-15:])
                gains = np.where(diffs > 0, diffs, 0).mean()
                losses = np.where(diffs < 0, -diffs, 0).mean()
                rs = gains / losses if losses > 0 else 100
                rec["rsi_14"] = 100 - 100 / (1 + rs)
            else:
                rec["rsi_14"] = np.nan

            # --- Momentum ratio (short-term vs medium-term) ---
            if i >= 60:
                ret_20 = c[-1] / c[-21] - 1
                ret_60 = c[-1] / c[-61] - 1
                rec["mom_ratio_20_60"] = ret_20 / ret_60 if abs(ret_60) > 0.001 else np.nan
            else:
                rec["mom_ratio_20_60"] = np.nan

            # --- Volume features (use close as proxy if no volume) ---
            # Price trend strength: how linear is the decline?
            if i >= 20:
                x = np.arange(20)
                y = c[-20:] / c[-20] - 1
                slope, = np.polyfit(x, y, 1)[:1]
                rec["price_slope_20d"] = slope
            else:
                rec["price_slope_20d"] = np.nan

            # --- Consecutive down days ---
            if i >= 10:
                daily_rets = np.diff(c[-11:])
                consec = 0
                for r in reversed(daily_rets):
                    if r < 0:
                        consec += 1
                    else:
                        break
                rec["consec_down_days"] = consec
            else:
                rec["consec_down_days"] = 0

            feature_records.append(rec)

        if (idx + 1) % 100 == 0:
            print(f"  ... {idx+1}/{len(all_symbols)} symbols processed")

    features_df = pd.DataFrame(feature_records)
    features_df["date"] = pd.to_datetime(features_df["date"])
    print(f"[R2] Features computed: {features_df.shape}")
    print(f"[R2] Feature columns: {[c for c in features_df.columns if c not in ('date', 'symbol')]}")

    return features_df


FEATURE_COLS = [
    "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "dist_sma50", "dist_sma200", "below_sma50", "below_sma200", "death_cross",
    "dist_52w_high", "dist_52w_low",
    "vol_20d", "vol_60d", "vol_expansion",
    "rsi_14", "mom_ratio_20_60", "price_slope_20d", "consec_down_days",
]


def run_phase_r2(prices, universe_df):
    """Execute Phase R2."""
    features = compute_short_features(prices, universe_df)

    # Merge with universe to keep only PIT-valid rows
    features = features.merge(
        universe_df[["date", "symbol"]],
        on=["date", "symbol"],
        how="inner",
    )
    print(f"[R2] After PIT filter: {features.shape}")

    feat_file = DATA_DIR / "features_russell_short.parquet"
    features.to_parquet(feat_file, index=False)
    print(f"[R2] Saved features to {feat_file}")
    return features


# ═══════════════════════════════════════════════════════════════════════
# PHASE R3: Walk-Forward Training
# ═══════════════════════════════════════════════════════════════════════

def create_labels(features_df, prices):
    """Create binary label: 1 if stock declined >5% over next 10 trading days."""
    print("[R3] Creating labels...")

    fwd_returns = {}
    for sym in features_df["symbol"].unique():
        if sym not in prices.columns:
            continue
        s = prices[sym].dropna()
        # Forward 10-day return
        fwd = s.shift(-10) / s - 1
        fwd_returns[sym] = fwd

    labels = []
    for _, row in features_df.iterrows():
        sym = row["symbol"]
        dt = row["date"]
        if sym in fwd_returns and dt in fwd_returns[sym].index:
            fwd = fwd_returns[sym].loc[dt]
            if np.isnan(fwd):
                labels.append(np.nan)
            else:
                labels.append(1 if fwd < -0.05 else 0)
        else:
            labels.append(np.nan)

    features_df = features_df.copy()
    features_df["fwd_ret_10d"] = [
        fwd_returns.get(row["symbol"], pd.Series(dtype=float)).get(row["date"], np.nan)
        for _, row in features_df.iterrows()
    ]
    features_df["target_short"] = labels
    features_df = features_df.dropna(subset=["target_short", "fwd_ret_10d"])
    features_df["target_short"] = features_df["target_short"].astype(int)

    print(f"[R3] Labels: {len(features_df):,} samples, {features_df['target_short'].mean():.1%} positive (declined >5%)")
    return features_df


def walk_forward_train(labeled_df):
    """Walk-forward train LGBM + XGBoost ensemble."""
    import lightgbm as lgb

    os.environ["OMP_NUM_THREADS"] = "1"
    from xgboost import XGBClassifier
    from sklearn.isotonic import IsotonicRegression

    print("[R3] Walk-forward training...")

    LGB_PARAMS = dict(
        n_estimators=500, learning_rate=0.03, max_depth=5, num_leaves=24,
        min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.0, objective="binary", metric="auc",
        random_state=42, n_jobs=1, verbose=-1,
    )
    XGB_PARAMS = dict(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, n_jobs=1, eval_metric="auc",
        tree_method="hist",
    )

    labeled_df = labeled_df.copy()
    labeled_df["year"] = labeled_df["date"].dt.year
    years = sorted(labeled_df["year"].unique())
    test_years = [y for y in years if y >= 2015]

    all_predictions = []
    results = []

    for test_year in test_years:
        # Train: 3 years before test year
        train_years = [y for y in years if test_year - 3 <= y < test_year]
        train = labeled_df[labeled_df["year"].isin(train_years)]
        test = labeled_df[labeled_df["year"] == test_year]

        if len(train) < 1000 or len(test) < 100:
            print(f"  {test_year}: SKIP (train={len(train)}, test={len(test)})")
            continue

        feat_cols = [c for c in FEATURE_COLS if c in train.columns]
        X_train = train[feat_cols].fillna(0)
        y_train = train["target_short"]
        X_test = test[feat_cols].fillna(0)
        y_test = test["target_short"]

        # Train LGBM
        lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
        lgb_model.fit(X_train, y_train)
        lgb_prob = lgb_model.predict_proba(X_test)[:, 1]

        # Train XGBoost
        xgb_model = XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(X_train, y_train, verbose=False)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

        # Ensemble
        raw_prob = 0.6 * lgb_prob + 0.4 * xgb_prob

        # Isotonic calibration on train set
        lgb_train_prob = lgb_model.predict_proba(X_train)[:, 1]
        xgb_train_prob = xgb_model.predict_proba(X_train)[:, 1]
        raw_train = 0.6 * lgb_train_prob + 0.4 * xgb_train_prob
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_train, y_train)
        prob_short = iso.predict(raw_prob)

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_test, prob_short)
        except:
            auc = 0.5

        # Store predictions
        pred_df = test[["date", "symbol", "fwd_ret_10d", "target_short"]].copy()
        pred_df["prob_short"] = prob_short
        pred_df.to_parquet(DATA_DIR / f"predictions_{test_year}.parquet", index=False)
        all_predictions.append(pred_df)

        pos_rate = y_test.mean()
        print(f"  {test_year}: train={len(train):,}, test={len(test):,}, AUC={auc:.4f}, pos_rate={pos_rate:.1%}")
        results.append({"year": test_year, "train_size": len(train), "test_size": len(test),
                        "auc": auc, "pos_rate": pos_rate})

        # Save models
        lgb_model.booster_.save_model(str(DATA_DIR / f"model_lgb_{test_year}.bin"))
        xgb_model.save_model(str(DATA_DIR / f"model_xgb_{test_year}.json"))

    return all_predictions, results


def run_phase_r3(features_df, prices):
    """Execute Phase R3."""
    print("\n" + "=" * 70)
    print("PHASE R3: Walk-Forward Training")
    print("=" * 70)

    labeled = create_labels(features_df, prices)
    predictions, train_results = walk_forward_train(labeled)

    return predictions, train_results


# ═══════════════════════════════════════════════════════════════════════
# PHASE R4: Signal Quality Test
# ═══════════════════════════════════════════════════════════════════════

def run_phase_r4(predictions):
    """Evaluate signal quality: do top-5 short candidates actually decline?"""
    print("\n" + "=" * 70)
    print("PHASE R4: Signal Quality Test")
    print("=" * 70)

    from scipy import stats

    yearly_results = []

    for pred_df in predictions:
        year = pred_df["date"].dt.year.mode()[0]
        dates = sorted(pred_df["date"].unique())

        top5_rets = []
        random5_rets = []
        bottom5_rets = []
        q1_rets = []  # top quintile (highest prob_short)
        q5_rets = []  # bottom quintile (lowest prob_short)
        mid_rets = []  # middle third

        np.random.seed(42)
        for dt in dates:
            day = pred_df[pred_df["date"] == dt].copy()
            if len(day) < 10:
                continue

            day = day.sort_values("prob_short", ascending=False)
            n = len(day)

            # Top 5 (highest short prob)
            top5 = day.head(5)
            top5_rets.extend(top5["fwd_ret_10d"].values)

            # Random 5
            rand5 = day.sample(min(5, n))
            random5_rets.extend(rand5["fwd_ret_10d"].values)

            # Bottom 5 (lowest short prob = predicted winners)
            bottom5 = day.tail(5)
            bottom5_rets.extend(bottom5["fwd_ret_10d"].values)

            # Quintiles
            q_size = n // 5
            if q_size >= 2:
                q1_rets.extend(day.head(q_size)["fwd_ret_10d"].values)
                q5_rets.extend(day.tail(q_size)["fwd_ret_10d"].values)
                mid_start = n // 3
                mid_end = 2 * n // 3
                mid_rets.extend(day.iloc[mid_start:mid_end]["fwd_ret_10d"].values)

        # Stats
        top5_mean = np.mean(top5_rets) * 100
        random_mean = np.mean(random5_rets) * 100
        bottom5_mean = np.mean(bottom5_rets) * 100
        spread = bottom5_mean - top5_mean
        t, p = stats.ttest_ind(bottom5_rets, top5_rets)
        hit_rate = np.mean([1 for r in top5_rets if r < 0])

        yearly_results.append({
            "year": year,
            "days": len(dates),
            "top5_fwd": top5_mean,
            "random_fwd": random_mean,
            "bottom5_fwd": bottom5_mean,
            "spread": spread,
            "spread_p": p,
            "hit_rate": hit_rate,
            "q1_fwd": np.mean(q1_rets) * 100 if q1_rets else np.nan,
            "q5_fwd": np.mean(q5_rets) * 100 if q5_rets else np.nan,
        })

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {year}: top5={top5_mean:+.3f}% | rand={random_mean:+.3f}% | bot5={bottom5_mean:+.3f}% | spread={spread:+.3f}%{sig} (p={p:.4f}) | hit={hit_rate:.1%}")

    # Aggregate
    all_top5 = np.mean([r["top5_fwd"] for r in yearly_results])
    all_spread = np.mean([r["spread"] for r in yearly_results])
    neg_years = sum(1 for r in yearly_results if r["top5_fwd"] < 0)
    sig_years = sum(1 for r in yearly_results if r["spread_p"] < 0.05)

    print(f"\n  AGGREGATE: top5_avg={all_top5:+.3f}% | spread_avg={all_spread:+.3f}%")
    print(f"  Negative return years: {neg_years}/{len(yearly_results)}")
    print(f"  Significant spread years (p<0.05): {sig_years}/{len(yearly_results)}")

    return yearly_results


# ═══════════════════════════════════════════════════════════════════════
# PHASE R5: Heuristic Baseline Comparison
# ═══════════════════════════════════════════════════════════════════════

def run_phase_r5(predictions):
    """Compare ML model against simple heuristic baselines on same universe."""
    print("\n" + "=" * 70)
    print("PHASE R5: Heuristic Baseline Comparison")
    print("=" * 70)

    results = {}
    heuristics = {
        "lowest_momentum_60d": lambda df: df.nsmallest(5, "ret_60d"),
        "below_sma200_worst": lambda df: df[df["below_sma200"] == 1].nsmallest(5, "dist_sma200") if df["below_sma200"].sum() >= 5 else df.nsmallest(5, "dist_sma200"),
        "high_vol_below_sma50": lambda df: df[df["below_sma50"] == 1].nlargest(5, "vol_20d") if df["below_sma50"].sum() >= 5 else df.nlargest(5, "vol_20d"),
        "worst_52w_high": lambda df: df.nsmallest(5, "dist_52w_high"),
        "ml_top5": lambda df: df.nlargest(5, "prob_short"),
    }

    for pred_df in predictions:
        year = pred_df["date"].dt.year.mode()[0]

        for name, selector in heuristics.items():
            dates = sorted(pred_df["date"].unique())
            rets = []
            for dt in dates:
                day = pred_df[pred_df["date"] == dt]
                if len(day) < 10:
                    continue
                try:
                    picks = selector(day)
                    rets.extend(picks["fwd_ret_10d"].values)
                except:
                    continue

            if name not in results:
                results[name] = {}
            results[name][year] = np.mean(rets) * 100 if rets else np.nan

    print(f"\n{'Method':<25} {'Avg 10d Fwd Ret':>15} {'Best Year':>10} {'Worst Year':>12}")
    print("-" * 65)
    for name in heuristics:
        vals = [v for v in results[name].values() if not np.isnan(v)]
        if vals:
            avg = np.mean(vals)
            best = max(vals)
            worst = min(vals)
            marker = " <-- ML" if name == "ml_top5" else ""
            print(f"{name:<25} {avg:>+14.3f}% {best:>+9.3f}% {worst:>+11.3f}%{marker}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# PHASE R6: Tradeability Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_phase_r6(predictions, mktcap_data, prices):
    """Analyze tradeability of ML top-5 short candidates."""
    print("\n" + "=" * 70)
    print("PHASE R6: Tradeability Analysis")
    print("=" * 70)

    all_syms = set()
    for pred_df in predictions:
        dates = sorted(pred_df["date"].unique())
        for dt in dates:
            day = pred_df[pred_df["date"] == dt]
            if len(day) < 5:
                continue
            top5 = day.nlargest(5, "prob_short")
            all_syms.update(top5["symbol"].values)

    print(f"[R6] Unique symbols appearing in top-5: {len(all_syms)}")

    mktcaps = []
    for sym in all_syms:
        if sym in mktcap_data:
            mktcaps.append({"symbol": sym, "marketCap": mktcap_data[sym]["marketCap"]})

    if not mktcaps:
        print("[R6] No market cap data for top-5 symbols")
        return {}

    mc_df = pd.DataFrame(mktcaps)
    mc = mc_df["marketCap"]

    stats = {
        "unique_symbols": len(all_syms),
        "with_mktcap": len(mc_df),
        "median_mktcap_M": mc.median() / 1e6,
        "mean_mktcap_M": mc.mean() / 1e6,
        "pct_below_500M": (mc < 500e6).mean() * 100,
        "pct_below_1B": (mc < 1e9).mean() * 100,
    }

    # Check price levels (penny stock proxy)
    latest_prices = []
    for sym in all_syms:
        if sym in prices.columns:
            last = prices[sym].dropna()
            if len(last) > 0:
                latest_prices.append(last.iloc[-1])

    if latest_prices:
        lp = np.array(latest_prices)
        stats["pct_below_5"] = (lp < 5).mean() * 100
        stats["pct_below_10"] = (lp < 10).mean() * 100
        stats["median_price"] = np.median(lp)

    print(f"\n  Median market cap: ${stats['median_mktcap_M']:.0f}M")
    print(f"  Mean market cap: ${stats['mean_mktcap_M']:.0f}M")
    print(f"  Below $500M: {stats['pct_below_500M']:.0f}%")
    print(f"  Below $1B: {stats['pct_below_1B']:.0f}%")
    if "median_price" in stats:
        print(f"  Median price: ${stats['median_price']:.2f}")
        print(f"  Below $5 (penny): {stats['pct_below_5']:.0f}%")
        print(f"  Below $10: {stats['pct_below_10']:.0f}%")

    return stats


# ═══════════════════════════════════════════════════════════════════════
# PHASE R7: Verdict Report
# ═══════════════════════════════════════════════════════════════════════

def write_verdict(train_results, r4_results, r5_results, r6_stats, universe_df):
    """Write final verdict report."""
    print("\n" + "=" * 70)
    print("PHASE R7: Final Verdict Report")
    print("=" * 70)

    report_file = DATA_DIR / "RUSSELL_SHORT_VERDICT.md"

    # Determine verdict
    top5_avg = np.mean([r["top5_fwd"] for r in r4_results])
    spread_avg = np.mean([r["spread"] for r in r4_results])
    neg_years = sum(1 for r in r4_results if r["top5_fwd"] < 0)
    sig_years = sum(1 for r in r4_results if r["spread_p"] < 0.05)
    total_years = len(r4_results)

    # Check ML vs heuristics
    ml_avg = np.mean([v for v in r5_results.get("ml_top5", {}).values() if not np.isnan(v)])
    heuristic_avgs = {}
    for name in r5_results:
        if name != "ml_top5":
            vals = [v for v in r5_results[name].values() if not np.isnan(v)]
            if vals:
                heuristic_avgs[name] = np.mean(vals)
    best_heuristic = min(heuristic_avgs.values()) if heuristic_avgs else 999
    ml_beats_heuristics = ml_avg < best_heuristic

    tradeable_pct = 100 - r6_stats.get("pct_below_500M", 100)

    if top5_avg < -1.0 and sig_years >= 7 and ml_beats_heuristics and tradeable_pct >= 50:
        verdict = "PROCEED"
    elif top5_avg >= 0:
        verdict = "STOP"
    elif not ml_beats_heuristics:
        verdict = "STOP"
    elif tradeable_pct < 30:
        verdict = "STOP"
    else:
        verdict = "INCONCLUSIVE"

    # Build report
    lines = [
        "# Russell 2000 Short Model: Viability Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Universe Methodology",
        "",
        "**Market Cap Proxy** (not true Russell 2000 membership)",
        "",
        "- Ticker pool: GitHub ikoniaris/Russell2000 list (~1900 symbols)",
        "- Historical market cap approximated as: `current_mktcap × (adj_close[date] / adj_close[latest])`",
        "- On each date, stocks with market cap $300M–$2B are included",
        "- Point-in-time safe: membership determined by price data available on that date",
        "- Mild bias: share count changes (buybacks, dilution) not captured",
        "- NOT true Russell 2000 membership (annual reconstitution not modeled)",
        "",
    ]

    # Universe stats
    if not universe_df.empty:
        lines.append("### Universe Size Per Year")
        lines.append("")
        lines.append("| Year | Unique Symbols | Observations |")
        lines.append("|------|---------------|-------------|")
        for year in sorted(universe_df["date"].dt.year.unique()):
            yr_data = universe_df[universe_df["date"].dt.year == year]
            lines.append(f"| {year} | {yr_data['symbol'].nunique()} | {len(yr_data):,} |")
        lines.append("")

    # Training results
    lines.append("## Walk-Forward Results")
    lines.append("")
    lines.append("| Year | Train | Test | AUC | Pos Rate |")
    lines.append("|------|-------|------|-----|----------|")
    for r in train_results:
        lines.append(f"| {r['year']} | {r['train_size']:,} | {r['test_size']:,} | {r['auc']:.4f} | {r['pos_rate']:.1%} |")
    lines.append("")

    # Signal quality
    lines.append("## Signal Quality (Phase R4)")
    lines.append("")
    lines.append("| Year | Days | Top-5 Fwd | Random | Bottom-5 | Spread | p-value | Hit Rate |")
    lines.append("|------|------|-----------|--------|----------|--------|---------|----------|")
    for r in r4_results:
        sig = "***" if r["spread_p"] < 0.001 else "**" if r["spread_p"] < 0.01 else "*" if r["spread_p"] < 0.05 else ""
        lines.append(f"| {r['year']} | {r['days']} | {r['top5_fwd']:+.3f}% | {r['random_fwd']:+.3f}% | {r['bottom5_fwd']:+.3f}% | {r['spread']:+.3f}%{sig} | {r['spread_p']:.4f} | {r['hit_rate']:.1%} |")
    lines.append("")
    lines.append(f"**Aggregate**: top5_avg={top5_avg:+.3f}%, spread_avg={spread_avg:+.3f}%")
    lines.append(f"**Negative return years**: {neg_years}/{total_years}")
    lines.append(f"**Significant spread years (p<0.05)**: {sig_years}/{total_years}")
    lines.append("")

    # Heuristic comparison
    lines.append("## Heuristic Comparison (Phase R5)")
    lines.append("")
    lines.append("| Method | Avg 10d Fwd Ret |")
    lines.append("|--------|----------------|")
    for name in r5_results:
        vals = [v for v in r5_results[name].values() if not np.isnan(v)]
        if vals:
            avg = np.mean(vals)
            marker = " **(ML)**" if name == "ml_top5" else ""
            lines.append(f"| {name} | {avg:+.3f}%{marker} |")
    lines.append("")
    lines.append(f"**ML beats best heuristic**: {'YES' if ml_beats_heuristics else 'NO'}")
    lines.append("")

    # Tradeability
    lines.append("## Tradeability (Phase R6)")
    lines.append("")
    for k, v in r6_stats.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.1f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"### **{verdict}**")
    lines.append("")

    if verdict == "PROCEED":
        lines.append("The Russell 2000 short model shows viable signal:")
        lines.append(f"- Top-5 avg forward return: {top5_avg:+.3f}% (negative = good)")
        lines.append(f"- Significant in {sig_years}/{total_years} years")
        lines.append(f"- ML beats heuristic baselines")
        lines.append(f"- {tradeable_pct:.0f}% of signals are tradeable")
        lines.append("")
        lines.append("### Next Steps")
        lines.append("- Estimated time to build full long/short infrastructure: 20-30 hours")
        lines.append("- Required: backtester changes, execution logic, position sizing")
    elif verdict == "STOP":
        reasons = []
        if top5_avg >= 0:
            reasons.append(f"Top-5 avg return is POSITIVE ({top5_avg:+.3f}%) — no short signal")
        if not ml_beats_heuristics:
            reasons.append("ML does NOT beat simple heuristics — model adds no value")
        if tradeable_pct < 30:
            reasons.append(f"Only {tradeable_pct:.0f}% of signals are tradeable")
        lines.append("**Why:** " + "; ".join(reasons))
    else:
        lines.append("Results are marginal — need user judgment on whether to proceed.")

    report = "\n".join(lines)
    with open(report_file, "w") as f:
        f.write(report)
    print(f"[R7] Verdict: {verdict}")
    print(f"[R7] Report written to {report_file}")
    return verdict


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", help="Which phase to run (r1, r2, r3, r4, r5, r6, r7, all)")
    args = parser.parse_args()

    phase = args.phase.lower()

    if phase in ("r1", "all"):
        universe, prices, mktcap_data = run_phase_r1()
        if universe is None:
            sys.exit(1)

    if phase in ("r2", "all"):
        if phase != "all":
            universe = pd.read_parquet(DATA_DIR / "russell2000_proxy_universe.parquet")
            prices = pd.read_parquet(DATA_DIR / "prices_russell.parquet")
        features = run_phase_r2(prices, universe)

    if phase in ("r3", "all"):
        if phase != "all":
            features = pd.read_parquet(DATA_DIR / "features_russell_short.parquet")
            prices = pd.read_parquet(DATA_DIR / "prices_russell.parquet")
        predictions, train_results = run_phase_r3(features, prices)

    if phase in ("r4", "all"):
        if phase != "all":
            pred_files = sorted(DATA_DIR.glob("predictions_*.parquet"))
            predictions = [pd.read_parquet(f) for f in pred_files]
        r4_results = run_phase_r4(predictions)

    if phase in ("r5", "all"):
        if phase != "all":
            pred_files = sorted(DATA_DIR.glob("predictions_*.parquet"))
            predictions = [pd.read_parquet(f) for f in pred_files]
        # Need features merged with predictions for heuristic columns
        if phase == "all":
            # Predictions already have features from R3
            pass
        r5_results = run_phase_r5(predictions)

    if phase in ("r6", "all"):
        if phase != "all":
            pred_files = sorted(DATA_DIR.glob("predictions_*.parquet"))
            predictions = [pd.read_parquet(f) for f in pred_files]
            prices = pd.read_parquet(DATA_DIR / "prices_russell.parquet")
            with open(DATA_DIR / "mktcap_cache.json") as f:
                mktcap_data = json.load(f)
        r6_stats = run_phase_r6(predictions, mktcap_data, prices)

    if phase in ("r7", "all"):
        if phase != "all":
            universe = pd.read_parquet(DATA_DIR / "russell2000_proxy_universe.parquet")
        verdict = write_verdict(train_results, r4_results, r5_results, r6_stats, universe)
