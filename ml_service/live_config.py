"""
CANONICAL LIVE CONFIG — the single source of truth for "what is actually deployed".

Import this in ALL research/backtests that claim to test the live strategy:

    from live_config import V12_LIVE_BACKTEST_CONFIG, LIVE_LEVERAGE, LIVE_FINANCING_RATE

Provenance (verified 2026-07-10 parity audit; see docs/LIVE_SYSTEM.md for the ledger):
- Signal construction is SHARED CODE (strategies/multi_strategy_engine.py) between the
  backtest and the live signal server — picks are identical by construction.
- The live weight scheme DIFFERS from the old "validated" backtest config in two ways,
  both audited, quantified, and KEPT (they backtest BETTER: +22.8%/0.94 vs +20.9%/0.89
  at 1x, 2018-25):
    * use_rp=False  — live signal_builder does NOT apply risk-parity inside sleeves
    * cap=0.10      — live caps at 15% of NAV on a ~1.49x book ≈ 10% of the invested book
      (backtest `cap` is a fraction of the book, so 0.10 models the live cap at full lev)
- Vol-scaling: live policy (VOL_TARGET_1X=0.15, lookback 40, floor 0.30, de-risk-only
  cap 1.0) — validated best-of-5 policies on 8yr+26yr, at 1x AND at leverage (2026-07-09/10).
- Leverage: closed-loop 1.49x effective (ibkr_engine._calibrate_quantities), financed at
  IBKR margin (~6.3%/yr on the borrowed portion) — model with the overlay in
  research/leverage_financing_test.py, NOT by multiplying CAGR.

CANONICAL EXPECTATION NUMBERS (research/final_live_config_test.py, 2026-07-10):
  2018-2025 (3-start): +28.3% CAGR / 0.95 Sharpe / -33.9% MaxDD  (levered, financed)
  2001-2025 (2-start): +20.7% CAGR / 0.77 Sharpe / -63.8% MaxDD  (no margin-call modeling)
"""

# Backtest config that reproduces the LIVE system (pass to FastBacktester.run):
V12_LIVE_BACKTEST_CONFIG = {
    "universe": "sp1500",
    # 2026-09-08 (FINAL @1.49x): sleeves 50/35/15 -> 70/21/9; rebalance is now TRANCHED
    # (4 sub-books, one rebuilt every 5 sessions) and the credit gate depth is 0.00 (flat
    # book while gated). FastBacktester has no tranche path: to reproduce live use the
    # clean-room engine (research/VERIFY2_cleanroom.py, cfg tranches=4, tranche_stride=5,
    # credit_derisk=0.0). The 20-day single-book numbers below are the pre-tranche baseline.
    "mom_w": 0.70, "val_w": 0.21, "lv_w": 0.09, "sec_w": 0.0,
    "top_n": 5,
    "rebal_days": 20,          # per book; 4 books staggered 5 sessions apart in live
    "trailing_stop": 0.40,
    "cap": 0.10,          # live-effective relative cap (15% NAV at 1.49x gross)
    "use_rp": False,      # live signal_builder applies NO risk-parity
    # bear_weights DELIBERATELY OMITTED. main_production_backtest falls back to
    # _short_weights(PROD_WEIGHTS_BEAR), so this config tracks production automatically.
    # It used to hardcode {"mom":.10,"val":.30,"s5":.50,"s3":.10}, which went stale when
    # 59860bb zeroed s3 (real BEAR is now .1111/.3333/.5556/0.00) — anything importing
    # this "single source of truth" then silently stopped reproducing live. Don't re-add it.
    "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40,
    # NOTE: stock backtester up-scales to 1.5x; live caps at 1.0 (de-risk only).
    # Use the research fork's vol_scale_cap=1.0 for exact parity (see
    # research/vol_scaling_ab_test.py); difference is ~nil at target 0.15.
}

# The OLD "validated" config (pre-2026-07-10 canonical) — kept for reproducing legacy
# numbers (+25.4% 1x headline). Do NOT use for live expectations.
V12_LEGACY_VALIDATED_CONFIG = {
    "universe": "sp1500",
    "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
    "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.15,
    "bear_weights": {"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
}

LIVE_LEVERAGE = 1.49          # closed-loop measured-gross target (x vol_scale)
LIVE_FINANCING_RATE = 0.063   # IBKR small-account margin, borrowed portion only
LIVE_VOL_SCALE_CAP = 1.0      # de-risk only — never levers above LIVE_LEVERAGE
LIVE_VOL_SCALE_FLOOR = 0.30
LIVE_TRANCHES = 4             # 2026-09-08: 4 virtual sub-books of NAV/4 (ibkr_engine.TRANCHES)
LIVE_TRANCHE_STRIDE = 5       # one sub-book rebuilt every 5 trading days
LIVE_CREDIT_DERISK = 0.0      # credit_gate.DERISK: the rebuilding book goes flat while HY-OAS >= p95
# EXP-059 package (research/FRONTIER_059.md, 2026-09-17) — built behind flags, ALL OFF, owner's call:
LIVE_OVERLAY_DOWN = True      # 2026-09-23 SHIPPED — ibkr_engine.TRANCHE_OVERLAY_DOWN (env IBKR_OVERLAY_DOWN=1): de-risk-only trims of the other books on tranche days
LIVE_MOM_EQUAL_WEIGHT = True  # 2026-09-23 SHIPPED — multi_strategy_engine.MOM_EQUAL_WEIGHT (env MOM_EQUAL_WEIGHT=1): equal-weight momentum picks
LIVE_BULL_WEIGHTS_OVERRIDE = "0.80,0.15,0.05"  # 2026-09-23 SHIPPED — multi_strategy_engine PROD_BULL_WEIGHTS env ("0.80,0.15,0.05" = the package's split); None = 70/21/9
# DATA LAYER (2026-09-30 — docs/LIVE_SYSTEM.md "Live data verification 2026-09-30"):
LIVE_SESSION_SETTLE_HOUR_ET = 17  # a daily bar is final from 17:00 ET: massive_data_provider.SETTLE_HOUR_ET == signal_builder.SESSION_SETTLE_HOUR_ET
LIVE_PRICE_CACHE_RULE = ("reuse a Massive file only if <18h old AND written after the last settled close; "
                         "refetch any cached symbol whose last bar lags the session most symbols end on")  # LIVE 2026-09-30 01:39
LIVE_PRICE_BASIS = "split-adjusted price-only (backtest: total-return; A/B 2026-09-29: identical BUY list)"
LIVE_VOL_WINDOW = "40 close-to-close NAV returns ending at the last COMPLETED close"  # eea223f — deploy pending (after 2026-09-30 close)
LIVE_FUNDAMENTALS_ROW = "whole newest Compustat quarter per ticker (backtest parity)"   # eea223f — deploy pending
