# Thread C — Per-stock / conditional hold times — FINDINGS

**Status: COMPLETE — NULL (backtest-backed), research-only.** 2026-07-15.
Harness: `research/threadC_holdtime_diag.py` (C1 gate) + `research/threadC_stop_validate.py`.

## Headline
**No per-stock or conditional hold-time rule graduates. The uniform 40% trailing stop +
20d cadence is at the frontier.** Confirms prior "aging flat/inverted" + "cadence optimal
at 20d". The one signal that graduates (rollover) says HOLD, not sell.

## C1 mandatory diagnostic gate (event study, held top-5 momentum, both periods)
Feature graduates only if it separates held-name forward-20d return MONOTONICALLY in the
SAME direction in BOTH periods. Results (8yr / 26yr direction):

| Feature | 8yr | 26yr | Verdict |
|---|---|---|---|
| **rollover (1m<0)** | +2.12pp | +1.10pp (fwd60 +4.60pp) | **GRADUATES** — but rolled-over names OUTPERFORM |
| vol_20d | flat | flat | DROP — **kills C3 (vol-scaled stop has no support)** |
| dist_52w_high | flat | MONO− | DROP (sign not both-period) |
| rsi_14 | flat | MONO− | DROP |
| dist_sma50 / sma200 | MONO− / MONO− | flat / flat | DROP |
| max_dd_6m | flat | MONO− | DROP |
| sma200_slope | flat | MONO+ | DROP |
| ret_60d | flat | flat | DROP |

**Only rollover graduates**, and its sign is "hold the dip": a held winner that pulled back
1-month has HIGHER forward return (short-term reversal within winners). That is a reason to
HOLD, not to add an exit — so it gives C2 no valid exit trigger, and it's already respected
by the deliberately-deep 40% stop.

## Stop-width validation (full run(), 1.49x flat, both periods, multi-start)
| trailing_stop | 8yr CAGR / Sharpe / MaxDD | 26yr CAGR / Sharpe / MaxDD |
|---|---|---|
| 25% | +24.8% / 0.81 / −42.4% | +19.5% / 0.68 / −72.0% |
| **40% (LIVE)** | **+28.7% / 0.86 / −45.4%** | +20.9% / 0.69 / −74.1% |
| 55% | +28.2% / 0.83 / −47.0% | +22.5% / 0.72 / −73.0% |
| NONE | +28.5% / 0.84 / −47.5% | +23.1% / 0.73 / −72.5% |

- **Tighter (25%) hurts both periods** — confirms C1's hold-the-dip (a tight stop cuts
  bounce-prone pullbacks). 8yr: stop40 is best (Sharpe 0.86).
- 26yr: looser/no-stop is marginally better (+2.2pp CAGR, ~equal DD) — the stop drags return
  without buying drawdown there — but this does NOT replicate on 8yr, so it is not actionable.
  stop40 is the defensible both-period compromise; no setting beats it consistently.

## Verdict on each sub-thread
- **C3 (vol/regime-scaled stop): DEAD** — vol_20d flat both periods; no per-name feature
  supports scaling stop width. Regime-tightening folds into Thread B (exposure cut) and is a
  noisier version of the same lever — adds nothing orthogonal.
- **C2 (conditional interim exit): DEAD** — only graduated feature (rollover) says hold.
- **C6 (regime cadence): not built** — cadence proven optimal at 20d; only working regime
  lever is exposure reduction = Thread B.
- **C4 (earnings-aware holds): not built** — entry/exit-timing already null (Different-Jobs
  program, thread2_entry_timing) and thread gate gives no graduation to justify the build.

Thread C closes as an honest NULL, consistent with the plan's stated "Disappoint: age/accel/
per-stock-length (inside noise)."
