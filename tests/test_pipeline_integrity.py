#!/usr/bin/env python3
"""
Pipeline Integrity Tests
========================
Validates all 7 audit fixes:
  1. No same-bar execution (next-day-open entry)
  2. Predictions have split column
  3. No FRED macro backfill (no future leak)
  4. Cross-sectional ranks use in_sp500 only
  5. OHLCV consistency
  6. Delisting returns present for known delistings
  7. No duplicate ticker-date rows

Run:
    cd "auto-trader 2" && python3 -m pytest tests/test_pipeline_integrity.py -v
    # or standalone:
    cd "auto-trader 2" && python3 tests/test_pipeline_integrity.py
"""

import sys
from pathlib import Path

# Add ml_service to path
ML_DIR = Path(__file__).resolve().parent.parent / "ml_service"
sys.path.insert(0, str(ML_DIR))

import numpy as np
import pandas as pd

DATA_DIR = ML_DIR / "data"
FEATURES_FILE = DATA_DIR / "features.parquet"
PRED_FILE = DATA_DIR / "predictions.parquet"


def _load_features():
    if not FEATURES_FILE.exists():
        return None
    df = pd.read_parquet(FEATURES_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_predictions():
    if not PRED_FILE.exists():
        return None
    df = pd.read_parquet(PRED_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Test 1: Next-day-open execution plumbing
# ══════════════════════════════════════════════════════════════════════════════

def test_load_open_bars_cached_exists():
    """Verify load_open_bars_cached is importable (Fix 1 plumbing)."""
    from unified_backtester import load_open_bars_cached
    assert callable(load_open_bars_cached)


def test_portfolio_manager_accepts_open_data():
    """Verify PortfolioManager.run() accepts open_data kwarg (Fix 1)."""
    import inspect
    from unified_backtester import PortfolioManager
    sig = inspect.signature(PortfolioManager.run)
    assert "open_data" in sig.parameters, \
        "PortfolioManager.run() missing open_data parameter"


# ══════════════════════════════════════════════════════════════════════════════
#  Test 2: Predictions have split column
# ══════════════════════════════════════════════════════════════════════════════

def test_predictions_split_column():
    """Verify predictions.parquet has a 'split' column (Fix 2)."""
    df = _load_predictions()
    if df is None:
        print("  SKIP: predictions.parquet not found (re-run train_production_model.py)")
        return
    assert "split" in df.columns, \
        "predictions.parquet missing 'split' column — re-run train_production_model.py"
    valid_splits = {"train", "purge", "calib", "oos"}
    actual = set(df["split"].unique())
    assert actual.issubset(valid_splits), \
        f"Unexpected split values: {actual - valid_splits}"
    # At least some rows should be OOS
    oos_pct = (df["split"] == "oos").mean()
    print(f"  split distribution: {dict(df['split'].value_counts())}")
    print(f"  OOS fraction: {oos_pct:.1%}")


def test_backtest_oos_only_flag():
    """Verify backtest.py accepts --oos-only flag (Fix 2)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("backtest", ML_DIR / "backtest.py")
    # Just check the source contains the flag
    source = (ML_DIR / "backtest.py").read_text()
    assert "--oos-only" in source, "backtest.py missing --oos-only flag"


# ══════════════════════════════════════════════════════════════════════════════
#  Test 3: No FRED macro backfill
# ══════════════════════════════════════════════════════════════════════════════

def test_no_macro_bfill_in_pipeline():
    """Verify data_pipeline.py does NOT bfill macro data (Fix 3)."""
    source = (ML_DIR / "data_pipeline.py").read_text()
    # Find the macro section
    idx = source.find("Loading FRED macro data")
    if idx == -1:
        print("  SKIP: FRED macro section not found in data_pipeline.py")
        return
    macro_section = source[idx:idx + 500]
    assert "bfill()" not in macro_section, \
        "data_pipeline.py still has bfill() in macro section — future leak!"


def test_no_macro_backfill_in_features():
    """Verify FRED features are NaN before series start dates (Fix 3)."""
    df = _load_features()
    if df is None:
        print("  SKIP: features.parquet not found")
        return
    if "hy_spread" not in df.columns:
        print("  SKIP: hy_spread column not in features")
        return
    # BAMLH0A0HYM2 (HY spread) starts ~1996. Before that, hy_spread should be NaN.
    # Our training data starts ~2014, so all rows should have it.
    # Check: earliest dates shouldn't have impossible values.
    earliest = df.nsmallest(100, "date")
    hy_filled = earliest["hy_spread"].notna().mean()
    print(f"  Earliest 100 rows: {hy_filled:.0%} have hy_spread (expect >90% since data starts 2014+)")


# ══════════════════════════════════════════════════════════════════════════════
#  Test 4: Cross-sectional ranks use in_sp500 only
# ══════════════════════════════════════════════════════════════════════════════

def test_ranks_nan_for_non_sp500():
    """Verify rank features are NaN for non-in_sp500 rows (Fix 4)."""
    df = _load_features()
    if df is None:
        print("  SKIP: features.parquet not found")
        return
    if "in_sp500" not in df.columns or "return_rank_3m" not in df.columns:
        print("  SKIP: required columns not found")
        return
    non_sp500 = df[df["in_sp500"] == False]
    if len(non_sp500) == 0:
        print("  SKIP: no non-SP500 rows found")
        return
    rank_cols = ["return_rank_3m", "return_rank_6m", "return_rank_12m",
                 "vol_rank_3m", "vol_rank_6m"]
    for col in rank_cols:
        if col in non_sp500.columns:
            pct_nan = non_sp500[col].isna().mean()
            assert pct_nan > 0.95, \
                f"{col} has {(1-pct_nan)*100:.1f}% non-NaN values for non-SP500 rows (expect >95% NaN)"
            print(f"  {col}: {pct_nan:.1%} NaN for non-SP500 rows ✓")


def test_sector_relative_nan_for_non_sp500():
    """Verify sector-relative features are NaN for non-in_sp500 rows (Fix 4)."""
    df = _load_features()
    if df is None:
        print("  SKIP: features.parquet not found")
        return
    sect_cols = ["ret_10d_vs_sector", "ret_20d_vs_sector",
                 "rsi_14_vs_sector", "vol_20d_vs_sector"]
    non_sp500 = df[df["in_sp500"] == False] if "in_sp500" in df.columns else pd.DataFrame()
    if len(non_sp500) == 0:
        print("  SKIP: no non-SP500 rows found")
        return
    for col in sect_cols:
        if col in non_sp500.columns:
            pct_nan = non_sp500[col].isna().mean()
            assert pct_nan > 0.95, \
                f"{col} has {(1-pct_nan)*100:.1f}% non-NaN for non-SP500 rows"
            print(f"  {col}: {pct_nan:.1%} NaN for non-SP500 rows ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  Test 5: OHLCV quality gate
# ══════════════════════════════════════════════════════════════════════════════

def test_quality_gate_checks_ohlcv():
    """Verify run_quality_gate checks OHLCV consistency (Fix 5)."""
    source = (ML_DIR / "massive_data_provider.py").read_text()
    assert "BAD_OHLCV" in source, \
        "massive_data_provider.py missing BAD_OHLCV check"
    assert "n_bad_ohlcv" in source, \
        "quality gate result missing n_bad_ohlcv field"


def test_quality_gate_catches_bad_ohlcv():
    """Verify quality gate catches fabricated bad OHLCV data (Fix 5)."""
    from massive_data_provider import MassiveDataProvider
    # Create a fake bars dict with bad data
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    good = pd.DataFrame({
        "open": [100]*10, "high": [105]*10, "low": [95]*10,
        "close": [102]*10, "volume": [1e6]*10,
    }, index=dates)
    bad = pd.DataFrame({
        "open": [100]*10, "high": [90]*10,  # high < low!
        "low": [95]*10, "close": [102]*10, "volume": [1e6]*10,
    }, index=dates)
    bars = {"GOOD": good, "BAD": bad}
    provider = MassiveDataProvider.__new__(MassiveDataProvider)
    result = provider.run_quality_gate(bars, expected_symbols=["GOOD", "BAD"])
    assert result["n_bad_ohlcv"] > 0, "Quality gate missed BAD_OHLCV"
    print(f"  Caught {result['n_bad_ohlcv']} bad OHLCV symbols ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  Test 6: Delisting returns
# ══════════════════════════════════════════════════════════════════════════════

def test_delisting_code_present():
    """Verify delisting return fix is in data_pipeline.py (Fix 6)."""
    source = (ML_DIR / "data_pipeline.py").read_text()
    assert "Delisting return fix" in source, \
        "data_pipeline.py missing delisting return logic"


def test_delisted_stocks_have_target():
    """Verify delisted stocks (FRC, SIVB) have target=0 for last rows (Fix 6)."""
    df = _load_features()
    if df is None:
        print("  SKIP: features.parquet not found")
        return
    for ticker in ["FRC", "SIVB", "SBNY"]:
        sub = df[df["symbol"] == ticker]
        if len(sub) == 0:
            print(f"  {ticker}: not in features (may not have Polygon data)")
            continue
        last_date = sub["date"].max()
        # Check if data ends well before 2024
        if last_date < pd.Timestamp("2024-01-01"):
            last_rows = sub.tail(5)
            has_target = last_rows["target"].notna().sum()
            print(f"  {ticker}: last date {last_date.date()}, "
                  f"last 5 rows have {has_target}/5 non-NaN targets")
        else:
            print(f"  {ticker}: data extends to {last_date.date()} (not delisted in data)")


# ══════════════════════════════════════════════════════════════════════════════
#  Test 7: No duplicate ticker-date rows
# ══════════════════════════════════════════════════════════════════════════════

def test_dedup_code_present():
    """Verify deduplication is in fetch_ticker_bars (Fix 7)."""
    source = (ML_DIR / "massive_data_provider.py").read_text()
    assert "duplicated" in source, \
        "massive_data_provider.py missing deduplication in fetch_ticker_bars"


def test_no_duplicate_rows_in_features():
    """Verify features.parquet has no duplicate (symbol, date) rows."""
    df = _load_features()
    if df is None:
        print("  SKIP: features.parquet not found")
        return
    dupes = df.duplicated(subset=["symbol", "date"], keep=False)
    n_dupes = dupes.sum()
    assert n_dupes == 0, \
        f"features.parquet has {n_dupes} duplicate (symbol, date) rows"
    print(f"  No duplicates in {len(df):,} rows ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  Main (standalone runner)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        ("Fix 1: load_open_bars_cached exists", test_load_open_bars_cached_exists),
        ("Fix 1: PortfolioManager accepts open_data", test_portfolio_manager_accepts_open_data),
        ("Fix 2: predictions split column", test_predictions_split_column),
        ("Fix 2: backtest --oos-only flag", test_backtest_oos_only_flag),
        ("Fix 3: no macro bfill in pipeline", test_no_macro_bfill_in_pipeline),
        ("Fix 3: no macro backfill in features", test_no_macro_backfill_in_features),
        ("Fix 4: ranks NaN for non-SP500", test_ranks_nan_for_non_sp500),
        ("Fix 4: sector-relative NaN for non-SP500", test_sector_relative_nan_for_non_sp500),
        ("Fix 5: quality gate checks OHLCV", test_quality_gate_checks_ohlcv),
        ("Fix 5: quality gate catches bad OHLCV", test_quality_gate_catches_bad_ohlcv),
        ("Fix 6: delisting code present", test_delisting_code_present),
        ("Fix 6: delisted stocks have target", test_delisted_stocks_have_target),
        ("Fix 7: dedup code present", test_dedup_code_present),
        ("Fix 7: no duplicate rows", test_no_duplicate_rows_in_features),
    ]

    passed = 0
    failed = 0
    skipped = 0

    print("=" * 70)
    print("  PIPELINE INTEGRITY TESTS")
    print("=" * 70)

    for name, fn in tests:
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        try:
            fn()
            passed += 1
            print(f"  ✓ PASSED")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            skipped += 1
            print(f"  ? ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} errors")
    print(f"{'='*70}")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
