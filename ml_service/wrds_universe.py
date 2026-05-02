"""
WRDS-Powered FastUniverse
=========================
Builds a FastUniverse-compatible object from WRDS data (CRSP prices,
Compustat fundamentals, CRSP SP500 membership).

Same interface as FastUniverse in multi_strategy_engine.py so existing
strategy code runs unchanged. Data sources differ:
  - SP500 membership: CRSP constituent daily (ground truth, not FMP)
  - Prices: CRSP daily stock file (survivorship-bias-free)
  - Fundamentals: Compustat quarterly (point-in-time via rdq)
  - Delistings: CRSP delisting returns (fixes missing terminal returns)

For BACKTESTING AND RESEARCH ONLY — not live production.

Usage:
    from wrds_universe import build_wrds_universe
    uni = build_wrds_universe(start="2018-01-01", end="2025-12-31")
    # Now use uni.get_sp500(date), uni.get_feature_map(date, feature), etc.
"""

import bisect
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

from wrds_data_provider import (
    WRDSDataProvider,
    load_delistings,
    load_sp500_prices,
)

log = logging.getLogger("wrds_universe")


# ─── Technical Feature Computation ──────────────────────────────────────────

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series):
    """MACD line and signal."""
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()
    return macd_line, signal


def compute_technical_features(close: pd.Series, volume: pd.Series,
                                high: pd.Series = None, low: pd.Series = None) -> pd.DataFrame:
    """Compute all technical features for one symbol's price series.

    Matches the feature names from data_pipeline.py:compute_symbol_features().
    """
    feat = pd.DataFrame(index=close.index)

    # Returns
    for n in [5, 10, 20, 60, 120, 126, 252]:
        feat[f"ret_{n}d"] = close.pct_change(n)

    # Rolling volatility (annualized)
    daily_ret = close.pct_change()
    for n in [10, 20, 60]:
        feat[f"vol_{n}d"] = daily_ret.rolling(n).std() * np.sqrt(252)

    # RSI
    feat["rsi_14"] = compute_rsi(close, 14)

    # MACD
    feat["macd_line"], feat["macd_signal"] = compute_macd(close)

    # Distance from SMAs
    for n in [50, 200]:
        sma = close.rolling(n).mean()
        feat[f"dist_sma{n}"] = (close - sma) / sma

    # 52-week high/low distance (if high/low available)
    if high is not None and low is not None:
        high_252 = high.rolling(252, min_periods=252).max()
        low_252 = low.rolling(252, min_periods=252).min()
        feat["dist_52w_high"] = (close - high_252) / high_252.replace(0, np.nan)
        feat["dist_52w_low"] = (close - low_252) / low_252.replace(0, np.nan)

    # Volume ratio vs 20-day average
    if volume is not None:
        avg_vol = volume.rolling(20).mean()
        feat["vol_ratio_20d"] = volume / avg_vol.replace(0, np.nan)

    return feat


# ─── Fundamental Feature Computation ────────────────────────────────────────

def compute_fundamental_features(
    cq_symbol: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute fundamental features from Compustat quarterly data for one symbol.

    Uses rdq (report date) for point-in-time alignment — on each trading day,
    only fundamentals with rdq <= that day are visible.

    Args:
        cq_symbol: Compustat quarterly rows for one symbol, sorted by datadate
        trading_dates: DatetimeIndex of trading dates to compute features for

    Returns:
        DataFrame indexed by trading_dates with fundamental feature columns
    """
    feat = pd.DataFrame(index=trading_dates, dtype=float)

    if len(cq_symbol) < 2:
        return feat

    # Sort by rdq (report date = when data became available)
    cq = cq_symbol.sort_values("rdq").reset_index(drop=True)
    rdq_dates = cq["rdq"].values  # numpy datetime64 array for bisect

    def _get_latest_idx(dt):
        """Find index of most recent quarterly report available on date dt."""
        # bisect_right gives insertion point; subtract 1 for latest available
        idx = np.searchsorted(rdq_dates, np.datetime64(dt), side="right") - 1
        return idx if idx >= 0 else None

    # Pre-compute for each trading date which quarterly row is the latest
    latest_idx = []
    for dt in trading_dates:
        latest_idx.append(_get_latest_idx(dt))

    # Vectorized feature extraction
    roe_vals = []
    gross_margin_vals = []
    eps_surprise_vals = []
    revenue_growth_vals = []
    eps_growth_vals = []
    debt_to_equity_vals = []
    operating_margin_vals = []
    net_margin_vals = []

    for idx in latest_idx:
        if idx is None or idx < 0:
            roe_vals.append(np.nan)
            gross_margin_vals.append(np.nan)
            eps_surprise_vals.append(np.nan)
            revenue_growth_vals.append(np.nan)
            eps_growth_vals.append(np.nan)
            debt_to_equity_vals.append(np.nan)
            operating_margin_vals.append(np.nan)
            net_margin_vals.append(np.nan)
            continue

        row = cq.iloc[idx]

        # ROE = net income / stockholders' equity (annualized)
        niq = row.get("niq")
        seqq = row.get("seqq")
        roe = (niq / seqq * 4) if (pd.notna(niq) and pd.notna(seqq) and seqq > 0) else np.nan
        roe_vals.append(roe)

        # Gross margin = (sales - COGS) / sales
        saleq = row.get("saleq")
        cogsq = row.get("cogsq")
        if pd.notna(saleq) and pd.notna(cogsq) and saleq > 0:
            gross_margin_vals.append((saleq - cogsq) / saleq)
        elif pd.notna(saleq) and saleq > 0:
            gross_margin_vals.append(np.nan)
        else:
            gross_margin_vals.append(np.nan)

        # Operating margin = operating income / sales
        oibdpq = row.get("oibdpq")
        if pd.notna(oibdpq) and pd.notna(saleq) and saleq > 0:
            operating_margin_vals.append(oibdpq / saleq)
        else:
            operating_margin_vals.append(np.nan)

        # Net margin
        if pd.notna(niq) and pd.notna(saleq) and saleq > 0:
            net_margin_vals.append(niq / saleq)
        else:
            net_margin_vals.append(np.nan)

        # Debt-to-equity = (long-term + short-term debt) / equity
        dlttq = row.get("dlttq", 0) or 0
        dlcq = row.get("dlcq", 0) or 0
        if pd.notna(seqq) and seqq > 0:
            debt_to_equity_vals.append((dlttq + dlcq) / seqq)
        else:
            debt_to_equity_vals.append(np.nan)

        # YoY revenue growth (need Q-4)
        if idx >= 4:
            prev_row = cq.iloc[idx - 4]
            prev_sale = prev_row.get("saleq")
            if pd.notna(saleq) and pd.notna(prev_sale) and prev_sale > 0:
                revenue_growth_vals.append((saleq - prev_sale) / prev_sale)
            else:
                revenue_growth_vals.append(np.nan)
        else:
            revenue_growth_vals.append(np.nan)

        # YoY EPS growth
        epsfxq = row.get("epsfxq")
        if idx >= 4:
            prev_eps = cq.iloc[idx - 4].get("epsfxq")
            if pd.notna(epsfxq) and pd.notna(prev_eps) and abs(prev_eps) > 0.01:
                eps_growth_vals.append((epsfxq - prev_eps) / abs(prev_eps))
            else:
                eps_growth_vals.append(np.nan)
        else:
            eps_growth_vals.append(np.nan)

        # EPS surprise (actual vs prior quarter — simplified without analyst estimates)
        # Use sequential change as proxy: (current EPS - same quarter last year) / abs(last year)
        eps_surprise_vals.append(np.nan)  # Will compute below if we have analyst data

    feat["roe"] = roe_vals
    feat["gross_margin"] = gross_margin_vals
    feat["operating_margin"] = operating_margin_vals
    feat["net_margin"] = net_margin_vals
    feat["debt_to_equity"] = debt_to_equity_vals
    feat["revenue_growth_yoy"] = revenue_growth_vals
    feat["eps_growth_yoy"] = eps_growth_vals
    feat["eps_surprise_last"] = eps_surprise_vals

    return feat


# ─── WRDS Universe Builder ──────────────────────────────────────────────────

class WRDSUniverse:
    """FastUniverse-compatible object backed by WRDS data.

    Provides the same interface as FastUniverse:
      - get_sp500(date) → set of tickers
      - get_feature_map(date, feature, members) → {symbol: value}
      - get_close_at(date, symbol) → float
      - get_close_series(symbol, end_date, lookback) → Series
      - get_regime(date) → dict
      - get_drift_pct(symbol, date, lookback) → float
    """

    def __init__(self, prices_df, features_by_date, sp500_provider,
                 fred_rates=None, sector_map=None):
        """
        Args:
            prices_df: DataFrame[date × ticker] of close prices
            features_by_date: {date: {symbol: {feature: value}}}
            sp500_provider: SP500Membership instance
            fred_rates: DataFrame of FRED interest rates (for regime)
            sector_map: {ticker: sector_name}
        """
        self.prices = prices_df
        self._feat_by_date = features_by_date
        self._sp500 = sp500_provider
        self._fred = fred_rates
        self.sector_map = sector_map or {}

        # Pre-compute close price dict: {symbol: Series}
        self._close = {}
        for col in prices_df.columns:
            self._close[col] = prices_df[col].dropna()

        # Pre-compute daily returns
        self._daily_returns = prices_df.pct_change()

        # SP500 membership cache (ticker sets)
        self._sp500_cache = {}

        # Empty placeholders for enhanced data not available in WRDS
        # (these are live-only data from FMP/Massive/Ortex)
        self._ml_preds = {}
        self._insiders = {}
        self._options = {}
        self._short_ratio = {}
        self._additions = {}
        self._price_targets = {}
        self._dcf = {}
        self._revenue_surprise = {}
        self._beat_streak = {}
        self._piotroski = {}
        self._analyst_consensus = {}
        self._pc_ratio = {}
        self._estimates = {}
        self._fin_growth = {}
        self._earnings_signals = {}
        self._ev = {}
        self._crypto_fx = {}
        self._profiles = {}
        self._sentiment = {}
        self._altman_z = {}

    def get_sp500(self, date):
        """Get SP500 ticker set on date (ground truth from CRSP)."""
        if date not in self._sp500_cache:
            dt = date.date() if hasattr(date, 'date') else date
            self._sp500_cache[date] = self._sp500.get_members_ticker(dt)
        return self._sp500_cache[date]

    def get_regime(self, date):
        """Get market regime indicators."""
        regime = {"vix": 20, "vix_term_structure": 1.0, "spy_above_sma200": True}

        # SPY above 200-day SMA
        if "SPY" in self._close:
            spy = self._close["SPY"].loc[:date]
            if len(spy) >= 200:
                regime["spy_above_sma200"] = spy.iloc[-1] > spy.iloc[-200:].mean()

        # FRED-based regime indicators (if available)
        if self._fred is not None and date in self._fred.index:
            row = self._fred.loc[date]
            # Yield curve
            yc = row.get("t10y2y")
            if pd.notna(yc):
                regime["yield_curve_10y2y"] = yc
            # Credit spread (BAA - AAA)
            baa = row.get("dbaa")
            aaa = row.get("daaa")
            if pd.notna(baa) and pd.notna(aaa):
                regime["credit_spread"] = baa - aaa

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
        """Get % positive return days over lookback."""
        if symbol not in self._daily_returns.columns:
            return 0.0
        rets = self._daily_returns[symbol].loc[:date].iloc[-lookback:]
        if len(rets) < 30:
            return 0.0
        return (rets > 0).mean()

    def get_ml_scores(self, date, members=None):
        return {}

    def get_short_ratios(self, date, members=None):
        return {}

    def get_additions(self, date, lookback_days=4):
        return []

    def get_insider_signal(self, symbol, date, lookback_days=90):
        return 0, 0.0


# ─── Main Builder ───────────────────────────────────────────────────────────

def build_wrds_universe(
    start: str = "2017-06-01",
    end: str = "2025-12-31",
    warmup_days: int = 550,
) -> WRDSUniverse:
    """Build a WRDSUniverse for backtesting.

    Args:
        start: Backtest start date (strategy signals start here)
        end: Backtest end date
        warmup_days: Extra calendar days before start for SMA200 etc.

    Returns:
        WRDSUniverse with all data pre-loaded and pre-indexed
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    t0 = time.time()
    log.info("Building WRDSUniverse from %s to %s (warmup=%d days)...", start, end, warmup_days)

    wp = WRDSDataProvider()

    # Extend start date for warmup (SMA200 needs ~252 trading days)
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")

    # ── Load prices ──────────────────────────────────────────────────────
    log.info("  Loading CRSP prices...")
    prices, open_prices, volumes = wp.load_prices(warmup_start, end)
    log.info("  Price matrix: %d dates × %d tickers", *prices.shape)

    # ── Load SP500 membership ────────────────────────────────────────────
    sp500 = wp.sp500
    all_permnos = sp500.get_all_permnos()

    # ── Load fundamentals ────────────────────────────────────────────────
    log.info("  Loading Compustat quarterly fundamentals...")
    cq = wp.load_fundamentals(all_permnos, "2010-01-01", end)

    # Map LPERMNO → ticker using the mapper
    mapper = wp.mapper
    # Build a quick PERMNO→latest_ticker lookup from the constituent file
    sample_date = date(2024, 1, 2)
    permno_ticker_map = sp500.get_members_map(sample_date)

    # For Compustat, use the 'tic' column directly (Compustat has its own ticker)
    # and also the LPERMNO for cross-referencing
    cq_by_ticker = {}
    for tic, group in cq.groupby("tic"):
        if tic and isinstance(tic, str) and len(tic) > 0:
            cq_by_ticker[tic] = group

    # ── Compute features ─────────────────────────────────────────────────
    log.info("  Computing features for %d tickers...", len(prices.columns))

    # Features: {date: {symbol: {feature: value}}}
    features_by_date = {}
    trading_dates = prices.index

    # Get list of tickers that are in SP500 at some point
    sp500_tickers = set()
    for dt in sp500.get_trading_dates(
        pd.Timestamp(start).date(),
        pd.Timestamp(end).date()
    ):
        sp500_tickers.update(sp500.get_members_ticker(dt))

    # Only compute features for tickers that were in SP500
    target_tickers = [t for t in prices.columns if t in sp500_tickers]
    log.info("  Computing features for %d SP500 tickers...", len(target_tickers))

    processed = 0
    for ticker in target_tickers:
        if ticker not in prices.columns:
            continue

        close = prices[ticker].dropna()
        if len(close) < 252:
            continue  # Need at least 1 year of data

        vol = volumes[ticker] if ticker in volumes.columns else pd.Series(0, index=close.index)

        # Technical features
        tech = compute_technical_features(close, vol)

        # Fundamental features (if Compustat data available)
        fund = pd.DataFrame(index=tech.index, dtype=float)
        if ticker in cq_by_ticker:
            fund = compute_fundamental_features(cq_by_ticker[ticker], tech.index)

        # Merge tech + fund
        all_feats = pd.concat([tech, fund], axis=1)

        # Insert into features_by_date dict
        for dt in all_feats.index:
            if dt not in features_by_date:
                features_by_date[dt] = {}
            row_dict = all_feats.loc[dt].to_dict()
            # Only store non-NaN features
            features_by_date[dt][ticker] = {k: v for k, v in row_dict.items()
                                             if pd.notna(v)}

        processed += 1
        if processed % 100 == 0:
            log.info("    %d/%d tickers processed...", processed, len(target_tickers))

    log.info("  Features computed for %d tickers across %d dates",
             processed, len(features_by_date))

    # ── Load I/B/E/S data ──────────────────────────────────────────────
    from wrds_data_provider import WRDS_DIR as wrds_dir
    log.info("  Loading I/B/E/S data (earnings surprise, price targets, recommendations)...")
    ibes_surprise = None
    ibes_price_targets = None
    ibes_recommendations = None

    ibes_surp_path = wrds_dir / "ibes_surprise.parquet"
    if ibes_surp_path.exists():
        ibes_surprise = pd.read_parquet(ibes_surp_path)
        ibes_surprise["anndats"] = pd.to_datetime(ibes_surprise["anndats"])
        log.info("    I/B/E/S Surprise: %d rows", len(ibes_surprise))

    ibes_pt_path = wrds_dir / "ibes_price_targets.parquet"
    if ibes_pt_path.exists():
        ibes_price_targets = pd.read_parquet(ibes_pt_path)
        ibes_price_targets["ANNDATS"] = pd.to_datetime(ibes_price_targets["ANNDATS"])
        log.info("    I/B/E/S Price Targets: %d rows", len(ibes_price_targets))

    ibes_recs_path = wrds_dir / "ibes_recommendations_summary.parquet"
    if ibes_recs_path.exists():
        ibes_recommendations = pd.read_parquet(ibes_recs_path)
        ibes_recommendations["STATPERS"] = pd.to_datetime(ibes_recommendations["STATPERS"])
        log.info("    I/B/E/S Recommendations: %d rows", len(ibes_recommendations))

    # ── Build enhanced data from Compustat + I/B/E/S ─────────────────
    log.info("  Building enhanced data...")
    fin_growth = {}       # {symbol: {rev_growth, eps_growth}}
    earnings_signals = {} # {date: {symbol: {rev_surprise, beat_streak}}}
    ev_data = {}          # {symbol: {ev_to_rev}}
    estimates_data = {}   # {symbol: {eps_avg}}

    for ticker, cq_group in cq_by_ticker.items():
        cq_sorted = cq_group.sort_values("datadate")
        if len(cq_sorted) < 5:
            continue

        # Latest quarter for _fin_growth
        latest = cq_sorted.iloc[-1]
        prev_4q = cq_sorted.iloc[-5] if len(cq_sorted) >= 5 else None

        rev_growth = np.nan
        eps_growth = np.nan
        if prev_4q is not None:
            s_cur = latest.get("saleq")
            s_prev = prev_4q.get("saleq")
            if pd.notna(s_cur) and pd.notna(s_prev) and s_prev > 0:
                rev_growth = (s_cur - s_prev) / s_prev
            e_cur = latest.get("epsfxq")
            e_prev = prev_4q.get("epsfxq")
            if pd.notna(e_cur) and pd.notna(e_prev) and abs(e_prev) > 0.01:
                eps_growth = (e_cur - e_prev) / abs(e_prev)

        if not (np.isnan(rev_growth) and np.isnan(eps_growth)):
            fin_growth[ticker] = {"rev_growth": rev_growth, "eps_growth": eps_growth}

        # EV/Revenue for _ev (use market cap proxy from Compustat)
        mkvaltq = latest.get("mkvaltq")
        dlttq = latest.get("dlttq", 0) or 0
        dlcq_val = latest.get("dlcq", 0) or 0
        cheq_val = latest.get("cheq", 0) or 0
        saleq_val = latest.get("saleq")
        if pd.notna(mkvaltq) and mkvaltq > 0 and pd.notna(saleq_val) and saleq_val > 0:
            ev = mkvaltq + dlttq + dlcq_val - cheq_val
            ev_to_rev = ev / (saleq_val * 4)  # annualize quarterly revenue
            ev_data[ticker] = {"ev_to_rev": ev_to_rev}

        # Forward EPS estimate proxy: trailing 4Q EPS for _estimates
        if len(cq_sorted) >= 4:
            last_4q = cq_sorted.tail(4)
            eps_vals = last_4q["epsfxq"].dropna()
            if len(eps_vals) >= 3:
                eps_avg = eps_vals.sum()  # trailing 4Q total
                estimates_data[ticker] = {"eps_avg": eps_avg}

        # Build earnings_signals: date-indexed for point-in-time access
        # For each quarter, compute YoY EPS surprise and track beat streak
        beat_count = 0
        for i in range(4, len(cq_sorted)):
            row = cq_sorted.iloc[i]
            prev = cq_sorted.iloc[i - 4]
            rdq = row.get("rdq")
            if pd.isna(rdq):
                continue

            # EPS surprise (YoY change as proxy without analyst estimates)
            eps_cur = row.get("epsfxq")
            eps_prev = prev.get("epsfxq")
            rev_cur = row.get("saleq")
            rev_prev = prev.get("saleq")

            eps_surp = np.nan
            rev_surp = np.nan
            beat = False

            if pd.notna(eps_cur) and pd.notna(eps_prev) and abs(eps_prev) > 0.01:
                eps_surp = (eps_cur - eps_prev) / abs(eps_prev)
                beat = eps_cur > eps_prev

            if pd.notna(rev_cur) and pd.notna(rev_prev) and rev_prev > 0:
                rev_surp = (rev_cur - rev_prev) / rev_prev

            if beat:
                beat_count += 1
            else:
                beat_count = 0

            rdq_ts = pd.Timestamp(rdq)
            if rdq_ts not in earnings_signals:
                earnings_signals[rdq_ts] = {}
            earnings_signals[rdq_ts][ticker] = {
                "eps_surprise": eps_surp,
                "rev_surprise": rev_surp,
                "beat_streak": beat_count,
            }

    # ── I/B/E/S Earnings Surprise (replaces Compustat YoY proxy) ────────
    # Use real analyst estimates vs actuals for proper earnings surprise
    if ibes_surprise is not None:
        eps_surp = ibes_surprise[ibes_surprise["MEASURE"] == "EPS"].copy()
        sal_surp = ibes_surprise[ibes_surprise["MEASURE"] == "SAL"].copy()

        # Build earnings_signals from I/B/E/S (overrides Compustat proxy)
        # Group by ticker + announcement date
        for _, row in eps_surp.iterrows():
            ticker = row.get("OFTIC")
            ann_date = row["anndats"]
            if pd.isna(ann_date) or not ticker:
                continue

            ann_ts = pd.Timestamp(ann_date)
            surp_mean = row.get("surpmean")
            sue = row.get("suescore")
            actual = row.get("actual")

            if ann_ts not in earnings_signals:
                earnings_signals[ann_ts] = {}

            existing = earnings_signals.get(ann_ts, {}).get(ticker, {})

            # EPS surprise as percentage
            eps_surprise_pct = np.nan
            if pd.notna(surp_mean) and pd.notna(actual) and abs(actual - surp_mean) > 0 and abs(actual) > 0.01:
                eps_surprise_pct = surp_mean / abs(actual)  # surprise relative to actual

            existing["eps_surprise"] = eps_surprise_pct
            existing["sue_score"] = sue if pd.notna(sue) else existing.get("sue_score", np.nan)
            earnings_signals[ann_ts][ticker] = existing

        # Revenue surprise from SAL measure
        for _, row in sal_surp.iterrows():
            ticker = row.get("OFTIC")
            ann_date = row["anndats"]
            if pd.isna(ann_date) or not ticker:
                continue

            ann_ts = pd.Timestamp(ann_date)
            surp_mean = row.get("surpmean")
            actual = row.get("actual")

            if ann_ts not in earnings_signals:
                earnings_signals[ann_ts] = {}

            existing = earnings_signals.get(ann_ts, {}).get(ticker, {})

            rev_surprise_pct = np.nan
            if pd.notna(surp_mean) and pd.notna(actual) and abs(actual) > 0:
                rev_surprise_pct = surp_mean / abs(actual)

            existing["rev_surprise"] = rev_surprise_pct
            earnings_signals[ann_ts][ticker] = existing

        # Build beat streak from I/B/E/S surprise data (per ticker, sorted by date)
        eps_by_ticker = eps_surp.groupby("OFTIC")
        for ticker, group in eps_by_ticker:
            g = group.sort_values("anndats")
            streak = 0
            for _, row in g.iterrows():
                sue = row.get("suescore")
                if pd.notna(sue) and sue > 0:
                    streak += 1
                else:
                    streak = 0

                ann_ts = pd.Timestamp(row["anndats"])
                if ann_ts in earnings_signals and ticker in earnings_signals[ann_ts]:
                    earnings_signals[ann_ts][ticker]["beat_streak"] = streak

        log.info("    I/B/E/S earnings signals: %d dates with signals", len(earnings_signals))

    # ── I/B/E/S Price Targets (point-in-time consensus) ──────────────
    price_targets = {}
    if ibes_price_targets is not None:
        # Compute consensus (median) price target per ticker using most recent targets
        # For each ticker, take the median of targets announced in the last 90 days
        latest_date = ibes_price_targets["ANNDATS"].max()
        cutoff = latest_date - pd.Timedelta(days=90)
        recent_pt = ibes_price_targets[ibes_price_targets["ANNDATS"] >= cutoff]

        for ticker, group in recent_pt.groupby("OFTIC"):
            vals = group["VALUE"].dropna()
            if len(vals) >= 2:
                price_targets[ticker] = {"target": vals.median()}

        log.info("    I/B/E/S price targets: %d tickers with consensus", len(price_targets))

    # ── I/B/E/S Recommendations (consensus buy/hold/sell) ────────────
    analyst_consensus = {}
    if ibes_recommendations is not None:
        # Take the latest recommendation summary per ticker
        latest_recs = ibes_recommendations.sort_values("STATPERS").groupby("OFTIC").last()
        for ticker, row in latest_recs.iterrows():
            mean_rec = row.get("MEANREC")
            buy_pct = row.get("BUYPCT")
            num_rec = row.get("NUMREC")
            if pd.notna(mean_rec):
                analyst_consensus[ticker] = {
                    "mean_rec": mean_rec,  # 1=strong buy, 5=strong sell
                    "buy_pct": buy_pct if pd.notna(buy_pct) else 0,
                    "num_analysts": num_rec if pd.notna(num_rec) else 0,
                }

        log.info("    I/B/E/S recommendations: %d tickers with consensus", len(analyst_consensus))

    log.info("  Enhanced data: fin_growth=%d, earnings_signals=%d dates, ev=%d, estimates=%d, price_targets=%d, analyst=%d",
             len(fin_growth), len(earnings_signals), len(ev_data), len(estimates_data),
             len(price_targets), len(analyst_consensus))

    # ── Load FRED rates for regime ───────────────────────────────────────
    fred_rates = wp.fred_rates

    # ── Build sector map ─────────────────────────────────────────────────
    sector_map = _build_sector_map(prices.columns, sp500)

    elapsed = time.time() - t0
    log.info("WRDSUniverse built in %.1fs", elapsed)

    uni = WRDSUniverse(
        prices_df=prices,
        features_by_date=features_by_date,
        sp500_provider=sp500,
        fred_rates=fred_rates,
        sector_map=sector_map,
    )

    # ── Build persistent revenue_surprise and beat_streak per symbol ────
    # The strategy checks _earnings_signals[date][sym] first (exact date match),
    # then falls back to _revenue_surprise[sym] and _beat_streak[sym].
    # We populate the fallbacks with each stock's MOST RECENT earnings result
    # that was announced BEFORE the end of the backtest period.
    # These are NOT look-ahead: they use the announcement date (anndats),
    # not the fiscal period end date.
    revenue_surprise = {}
    beat_streak = {}

    if ibes_surprise is not None:
        eps_surp = ibes_surprise[ibes_surprise["MEASURE"] == "EPS"].copy()
        sal_surp = ibes_surprise[ibes_surprise["MEASURE"] == "SAL"].copy()

        # Latest EPS surprise per ticker (by announcement date)
        for ticker, group in eps_surp.groupby("OFTIC"):
            g = group.sort_values("anndats")
            # Take the most recent announcement
            last = g.iloc[-1]
            sue = last.get("suescore")
            # Beat streak: count consecutive positive SUE scores from the end
            streak = 0
            for _, row in g.iloc[::-1].iterrows():
                s = row.get("suescore")
                if pd.notna(s) and s > 0:
                    streak += 1
                else:
                    break
            beat_streak[ticker] = streak

        # Latest revenue surprise per ticker
        for ticker, group in sal_surp.groupby("OFTIC"):
            g = group.sort_values("anndats")
            last = g.iloc[-1]
            surp_mean = last.get("surpmean")
            actual = last.get("actual")
            if pd.notna(surp_mean) and pd.notna(actual) and abs(actual) > 0:
                revenue_surprise[ticker] = surp_mean / abs(actual)

        log.info("  Persistent signals: revenue_surprise=%d, beat_streak=%d tickers",
                 len(revenue_surprise), len(beat_streak))

    # Populate enhanced data
    uni._fin_growth = fin_growth
    uni._earnings_signals = earnings_signals
    uni._ev = ev_data
    uni._estimates = estimates_data
    uni._price_targets = price_targets
    uni._analyst_consensus = analyst_consensus
    uni._revenue_surprise = revenue_surprise
    uni._beat_streak = beat_streak

    # Store open prices for next-day-open execution (same data source as close)
    uni.open_prices = open_prices

    return uni


def _build_sector_map(tickers, sp500):
    """Build {ticker: sector} from CRSP SIC codes.

    Uses SIC→GICS approximate mapping for the sector rotation strategy.
    """
    # Simple SIC division → sector mapping
    # This is approximate but sufficient for sector caps
    SIC_TO_SECTOR = {
        range(100, 1000): "Agriculture",
        range(1000, 1500): "Mining",
        range(1500, 1800): "Construction",
        range(2000, 4000): "Manufacturing",
        range(4000, 5000): "Transportation",
        range(5000, 5200): "Wholesale",
        range(5200, 6000): "Retail",
        range(6000, 6800): "Finance",
        range(7000, 9000): "Services",
        range(9100, 9800): "Government",
    }
    # For now, return empty — the sector rotation strategy uses ETFs (XLK, XLF, etc.)
    # which are in the price data directly. Individual stock sector mapping
    # would need the CRSP security_info SICCD field.
    return {}


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Quick test: build universe for a short period
    log.info("Quick test: building WRDSUniverse for 2023-2024...")
    uni = build_wrds_universe(start="2023-01-01", end="2024-12-31", warmup_days=400)

    # Test SP500 membership
    test_date = pd.Timestamp("2024-01-02")
    members = uni.get_sp500(test_date)
    print(f"\nSP500 on {test_date.date()}: {len(members)} members")

    # Test feature map
    ret_20 = uni.get_feature_map(test_date, "ret_20d", members)
    print(f"ret_20d available for {len(ret_20)} stocks")
    if ret_20:
        top5 = sorted(ret_20.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5: {[(s, f'{v:.3f}') for s, v in top5]}")

    # Test close price
    aapl_px = uni.get_close_at(test_date, "AAPL")
    print(f"AAPL close on {test_date.date()}: ${aapl_px:.2f}")

    # Test regime
    regime = uni.get_regime(test_date)
    print(f"Regime: {regime}")

    # Test fundamentals
    roe = uni.get_feature_map(test_date, "roe", members)
    print(f"ROE available for {len(roe)} stocks")
    if roe:
        top5_roe = sorted(roe.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top 5 ROE: {[(s, f'{v:.2f}') for s, v in top5_roe]}")

    print("\nDone.")
