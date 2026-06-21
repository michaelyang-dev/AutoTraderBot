"""
Is crypto momentum truly dead, or did naive volume-ranking just buy meme blow-off tops?
Tests sensible-universe variants on the SAME survivorship-complete Binance data:
  - min_age: exclude fresh listings (the pump-and-dump traps)
  - skip: "12-1" momentum (skip last 7d → don't buy the immediate parabolic top)
  - tighter liquidity (top-30/50 established perps, no meme garbage)
  - long-only vs market-neutral
If 2023+ stays negative across all of these, momentum is genuinely a 2021-mania artifact.
Run:  python crypto/backtest/momentum_diag.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from momentum_binance import load, run, stats

close, qv, fund = load()
print("=" * 90)
print("MOMENTUM DIAGNOSTIC — sensible universes, survivorship-complete | %d coins" % close.shape[1])
print("=" * 90)
print(f"  {'variant':<46}{'2023+ CAGR':>11}{'Sharpe23':>10}{'MaxDD23':>10}{'FULL':>8}")


def row(label, **kw):
    d = run(close, qv, fund, **kw)
    c23, _, s23, m23 = stats(d, "2023-01-01")
    c, _, _, _ = stats(d)
    print(f"  {label:<46}{c23*100:>10.1f}%{s23:>10.2f}{m23*100:>9.1f}%{c*100:>7.0f}%", flush=True)


print("  -- LONG-SHORT market-neutral + funding --")
row("baseline (naive)", long_short=True, use_funding=True)
row("+ min_age 90d (no fresh listings)", long_short=True, use_funding=True, min_age=90)
row("+ skip-7 (12-1 momentum)", long_short=True, use_funding=True, skip=7)
row("+ min_age90 + skip7", long_short=True, use_funding=True, min_age=90, skip=7)
row("+ min_age90 + skip7, top-30 only", long_short=True, use_funding=True, min_age=90, skip=7, top_uni=30)
row("+ min_age90 + skip7, top-50, lb60", long_short=True, use_funding=True, min_age=90, skip=7, top_uni=50, lookback=60)
print("  -- LONG-ONLY (top-quintile momentum) --")
row("long-only baseline", long_short=False, use_funding=False)
row("long-only min_age90 + skip7", long_short=False, use_funding=False, min_age=90, skip=7)
row("long-only min_age90 + skip7, top-30", long_short=False, use_funding=False, min_age=90, skip=7, top_uni=30)
row("long-only min_age180 + skip7, top-30, lb90", long_short=False, use_funding=False, min_age=180, skip=7, top_uni=30, lookback=90)
print("  -- SHORT-ONLY (is the short leg the problem? short worst momentum) --")
# short-only ≈ long_short minus the long leg; approximate by inspecting separately not built-in;
# use market-neutral with tiny long via top_frac to emphasize — skip, covered by books above.
print("\n  Read: if every sensible variant is still negative in 2023+, momentum is a mania artifact.")
print("  If long-only on established coins (min_age+skip, top-30) turns positive, there's a real")
print("  trend signal once you stop buying meme blow-off tops — worth pursuing as long-biased.")
