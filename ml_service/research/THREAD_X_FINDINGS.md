# Thread X — mid-cycle signal-exit rule — FINDINGS (user idea, 2026-07-22)

**Status: TESTED & REJECTED (both periods).** Harness: `threadX_signal_exit.py` +
`signal_exit_every`/`signal_exit_grace` in `livemirror_backtest.py`.

**Idea:** sell a holding mid-cycle when it drops out of the current signals (asymmetric
cadence: fast exits, 20d entries), motivated by 7/22 (−3.1% day where the losers were
exactly the rotated-out-of-signals names while current picks were green).

**Result (full live-mirror 1.49x integer $50k):** costs **−3.7 to −6.6pp CAGR/yr** with
Sharpe flat-to-worse (26yr −0.05) at ~146-210 exits/yr. The DD reduction (−33→−28 8yr,
−63→−52 26yr) is NOT timing skill — Sharpe doesn't improve — it's de-facto deleveraging
via idle cash, purchasable far cheaper by lowering leverage (which keeps full Sharpe) or
by vol-scaling (already live). Grace (mom top-15 tolerance) softens but doesn't save it.

**Why it loses:** signal drop ≈ recent-month reversal, and C1 showed those names BOUNCE
(+2.1pp fwd20 8yr / +1.1pp 26yr — hold-the-dip); plus ~150-210 trades/yr costs; plus cash
drag waiting for rebalance (recycle separately tested-dead). Consistent with: 20d cadence
beat 10d; Different-Jobs exit threads null. The 20d fixed hold has now survived its most
direct challenge, tested in the exact proposed form. Do not re-litigate without new
mechanism; single-day anecdotes (7/22) are the visible half of an asymmetric coin.
