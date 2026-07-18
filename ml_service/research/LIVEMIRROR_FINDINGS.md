# Live-mirror full-stack backtest — FINDINGS

**2026-07-15. Harness: `research/livemirror_backtest.py` + `livemirror_run.py`. Research-only.**
Everything enabled JOINTLY (not overlays): $50k start, INTEGER shares floored (no fractional,
IBKR constraint), real margin debit + 6.3% financing, vol-scaling (de-risk-only, unlevered-book
feedback), 40% stops, 20d rebal, bear-weights, UMD crash, and the p95 HY-OAS credit gate wired
causally INSIDE the loop. Multi-start (3 short / 2 long).

## Fidelity check (passed)
1.49x integer avgGross **1.14** ≈ validated 1.19; DD −33.2% ≈ validated −33.9%; CAGR 28.9% ≈
validated 28.3% (8yr). Vol-scaling now reads the UNLEVERED book vol (shadow-NAV feedback), as
live does, not the levered NAV. Numbers reproduce the deployed system.

## Results (multi-start avg, $50k, integer floor shares)
**SHORT 2018-25:**
| variant | CAGR | Sharpe | MaxDD | avgGross | fin/yr |
|---|---|---|---|---|---|
| 1.00x integer (unlevered) | +20.6% | 1.02 | −22.6% | 0.76 | 0 |
| 1.00x + credit p95 gate | +20.3% | 1.00 | −22.4% | 0.75 | 0 |
| **1.49x integer (LIVE)** | **+28.9%** | 0.98 | **−33.2%** | 1.14 | $1,731 |
| **1.49x + credit p95 gate** | **+28.4%** | 0.97 | **−32.6%** | 1.14 | $1,674 |
| 1.49x FRACTIONAL (ref) | +28.8% | 0.97 | −34.1% | 1.16 | $1,805 |

**LONG 2001-25:**
| variant | CAGR | Sharpe | MaxDD | avgGross | fin/yr |
|---|---|---|---|---|---|
| 1.00x integer (unlevered) | +15.6% | 0.82 | −47.3% | 0.76 | 0 |
| 1.00x + credit p95 gate | +15.7% | 0.83 | **−41.1%** | 0.75 | 0 |
| **1.49x integer (LIVE)** | **+20.8%** | 0.78 | **−63.5%** | 1.14 | $8,353 |
| **1.49x + credit p95 gate** | **+21.1%** | 0.79 | **−56.6%** | 1.12 | $8,715 |
| 1.49x FRACTIONAL (ref) | +20.8% | 0.77 | −63.7% | 1.14 | $8,389 |

## Verdict (full-stack, everything on)
1. **Credit gate confirmed inside the complete live engine:** LONG −63.5% → −56.6% MaxDD
   (**+6.9pp**) at **+0.3pp CAGR** (free). Even unlevered: −47.3% → −41.1% (+6.2pp). SHORT:
   ~free (−0.5pp CAGR) but no DD help — 2018-25 has no credit crisis, as expected.
   The full-stack DD gain (+6.9pp) is SMALLER than the isolated overlay (+9.6pp) because
   vol-scaling already catches part of the 2008 vol spike — the interaction the joint run
   reveals that stacked overlays miss. Still a real, free tail hedge.
2. **Integer rounding at $50k costs ~NOTHING:** integer 28.9% vs fractional 28.8% (short),
   20.8% vs 20.8% (long). The ~5-position momentum book holds $2-5k positions (plenty of
   shares even at $50k) and the account compounds up fast. IBKR no-fractional is NOT a drag.
3. **Leverage tradeoff (known):** 1.49x adds ~8pp CAGR (20.6→28.9 short, 15.6→20.8 long) and
   deepens DD (−22.6→−33.2, −47.3→−63.5) — Kelly. Financing $1.7k/yr (short) to $8.4k/yr
   (long), already netted.

## Caveat
These are in-sample-config point estimates (walk-forward OOS is ~2.5pp lower on the long book,
see WALKFORWARD_OOS_FINDINGS.md). The RELATIVE effects isolated here — credit gate benefit,
integer-vs-fractional, 1x-vs-1.49x — are the clean takeaways.
