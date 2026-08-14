# Dynamic leverage — findings (2026-08-13/14)

**Research only. Nothing here is deployed.** Scripts: `threadDYN{,2,3,4,5,6,7,8,9}*.py`.

## The method that makes these numbers mean anything

Every variant is scored against a **constant-leverage curve interpolated at that variant's own
realised `avg_gross`**. Without this control, any rule that de-risks looks brilliant — it is
simply running less exposure. The reported `dSharpe`/`dMaxDD` are residuals *after* removing
the level effect.

Bar for a claim: **positive on BOTH the 8yr (2018-25) and 26yr (2001-25) horizons.**

---

## 1. Dead — do not re-test

| idea | 8yr | 26yr |
|---|---|---|
| EWMA vol | +0.013 | −0.006 |
| asymmetric vol windows | −0.042 | +0.000 |
| eqtrend (own equity curve) | −0.017 | −0.093 |
| ddstate (distance from peak) | −0.169 | −0.138 |
| recovery ramp | −0.091 | −0.082 |
| vol-of-vol | +0.018 | +0.005 (marginal) |
| `umd_crash` | **+0.026** | **−0.020** |
| `mkt_vol` | +0.019 | −0.010 |
| `term_inv` | −0.015 | −0.086 |

`umd_crash` and `mkt_vol` are the cautionary cases: both would have **passed a single-period
test and are wrong**. Anything validated on one horizon here is not validated.

**Levering UP hurts.** Raising `vol_scale_cap` 1.0 → 1.15 → 1.30 → 1.50 monotonically worsens
Sharpe (26yr: −0.010 → −0.020 → −0.024 → −0.026) and CAGR (10.28% → 9.74%).

**The deployed inverse-vol overlay has ~no timing skill** (+0.018 / +0.001, negative with more
starts). Its value is a *level* effect, not knowing when to be levered.

## 2. The finding

`ibkr_engine` applies **both** `vol_scale` **and** the credit gate only inside the rebalance
block → up to **20 sessions late**. Credit spreads blow out in days.

Applying the same gate promptly (5-day re-check, deadband 0.10, `derisk 0.30`,
`OR(hy_oas, baa_aaa)`):

| | 8yr | 26yr |
|---|---|---|
| ΔSharpe (matched exposure) | +0.046 | +0.065 |
| ΔMaxDD | +3.88pp | +19.69pp |

**Walk-forward OOS** (annual re-selection, selection cost fully paid):

| vs deployed | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| 26yr (22 re-selections) | +0.36pp | +0.029 | **−57.51% → −40.02%** |
| 8yr (6 re-selections) | +0.20pp | +0.026 | −0.69pp |

### Honest limits
- **In-sample headline (+1.97pp CAGR) roughly HALVES out-of-sample.** Do not quote it.
- **This is drawdown insurance, not alpha.**
- Trailing-Sharpe selection chose "no gate" for 2020 having seen only calm 2019 — **performance
  selection drops insurance right before it is needed.** Prefer a FIXED config off the plateau.
- At 2× modelled cost: 26yr break-even (+0.002, DD still +16.75pp); **8yr −0.011 to −0.019**.
  2× modelled ≈ 3.3× the 6.10 bps measured live, but the 8yr margin is thin.

### Why it is credible
Sensitivity plateau (every `gate_pct` 0.90-0.97 and `derisk` 0.30-0.70 positive on both);
beats deployed in **all three** sub-periods (2001-08: CAGR −2.38% → **+3.28%**); walk-forward
**beat** the in-sample config on drawdown (+7.11pp) and independently converged on `derisk 0.3`
in all 22 OOS years.

## 3. Promptness does NOT generalise to the book

Shortening the *rebalance* cadence during stress is catastrophic:

| | 8yr ΔSharpe | 26yr ΔSharpe | 8yr CAGR |
|---|---|---|---|
| 20d + off-cadence leverage | **+0.046** | **+0.065** | +21.13% |
| stress 10d | −0.446 | −0.180 | +3.99% |
| stress 5d | −0.418 | −0.175 | +5.14% |
| stress 5d, *no* leverage change | −0.450 | — | +4.25% |

**Why:** a leverage adjustment *rescales* existing positions (small deltas on names already
held). A rebalance *re-picks* the book (near-total turnover). The finding is therefore narrower
than "apply rules promptly" — it is that **cheap adjustments should be prompt, expensive ones
should not.**

## 4. Harness bugs found (both produced convincing fake results)

1. **`live_sizing` renormalised `vol_scale` out of existence.** Weights were scaled then
   renormalised to sum 1, dividing the overlay back out. Every live-sizing arm ran unscaled.
   Caught because *every* policy reported `avgGross ≈ 1.4900`. This invalidated an earlier
   "live over-invests 1.33×, real MaxDD −53.8%/−76.0%" conclusion — actual figures are
   **1.03-1.06×** and **−39.9%/−65.0%**.
2. **`_rv()` rejected windows < 20**, so short-window arms ran unscaled while reporting deltas
   of exactly `0.000`. The first fix then over-corrected (demanded 30 for the default 40-day
   window), silently moving the baseline +28.3410% → +28.0694%. DYN2-DYN8 ran on that
   threshold: **deltas unaffected** (shared within each study), absolute CAGRs ~0.27pp off.

Both fixed and parity-verified: default path and counter path now both give exactly
`+28.3410% / avgGross 1.1258`.

## 5. Blockers before anything could ship

1. `DBAA`/`DAAA` must be wired from FRED (the pickle's `fred_rates` ends 2025-02). The
   credit-gate cron already pulls FRED, so the path exists.
2. **`ibkr_engine` has no off-cadence path at all** — this is new live code, not a config flag.
3. Cost margin on the 8yr is thin; re-audit at the measured 6.10 bps rather than the modelled 10.
