# Thread S3 — sector-sleeve ETF live-parity divergence — FINDINGS

**2026-07-26. Status: REAL divergence, MEASURED BENIGN, no live change made.**
Harness: `research/threadS3_sector_parity.py` (+ `crash_weights` config hook in livemirror).

## The divergence
In bear/crash regimes the sector sleeve (s3) receives 10% of the book and returns sector
**ETFs** (XLK/XLE/XLI/XLRE). The BACKTEST buys them (they are in its price matrix and enter
`combined`). LIVE DROPS them: `signal_builder` emits only SP1500 **members**, and ETFs are not
members, so zero XL* names ever reach the engine (verified on the live server: 24 names with
weight > 0, no ETFs). The engine's closed-loop sizing then redistributes that 10% across the
remaining stocks -> live runs more single-stock-concentrated than the validated backtest, in
exactly the defensive regime where sector diversification was meant to help.

Only bites in bear/crash. **It is live TODAY**: the UMD crash detector is firing (20d UMD
= -0.079 vs -0.05 threshold), so s3 = 0.10 is in the active weight set.

## A/B (full live-mirror, 1.49x integer $50k + financing + vol-scaling + credit gate, both periods)
| variant | 8yr CAGR / Sharpe / MaxDD | 26yr CAGR / Sharpe / MaxDD |
|---|---|---|
| A backtest — holds sector ETFs | +28.1% / 0.96 / -32.6% | +21.1% / 0.79 / -56.5% |
| B live replica — ETFs dropped, renormalized | +28.2% / 0.96 / -32.9% | +21.3% / 0.79 / -56.7% |

Deltas (B - A): **+0.1pp / +0.2pp CAGR, 0.00 Sharpe both periods, -0.3pp / -0.2pp MaxDD.**

## Verdict
**BENIGN — do NOT change live.** Every delta is inside the measured ~0.6pp-per-start noise
floor, Sharpe is identical to two decimals in both periods, and the tiny DD cost is not
distinguishable from noise. Restoring parity would mean routing sector-ETF orders through a
live path that has never traded ETFs, for zero measurable benefit — strictly added operational
risk. Documented and closed.

Caveat retained: this says the 10% sector allocation is *fungible* with more stock exposure at
the book level. It does NOT say sector rotation is worthless in general — only that, at this
book's construction, dropping it and renormalizing costs nothing measurable.
