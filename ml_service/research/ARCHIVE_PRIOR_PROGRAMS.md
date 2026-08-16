# ARCHIVE — prior research programs (2026-06 → 2026-08-13)

**This file is the verbatim concatenation of 14 separate findings documents that used to sit
loose in `research/`.** They were merged because they overlapped heavily and it was not obvious
which was current. Nothing was edited; only headers were added.

## ⚠️ BEFORE TRUSTING ANY NUMBER BELOW

Everything here predates three corrections that invalidate large parts of it:

1. **Both universe pickles were rebuilt 2026-07-28** after 8 defects (look-ahead membership,
   survivorship via suffixed tickers, spliced ticker histories, deleted delisting losses...).
   26yr CAGR was overstated by **10.4pp**. Anything dated before 2026-07-28 is suspect.
2. **The "0.6pp/start noise floor" understated reality ~4×** (true 8yr per-start sigma 7.07pp).
   Most verdicts below came from 3-4 start comparisons, which cannot resolve a 2pp effect.
   That combination has already produced a **wrong sign** and a **retracted win**.
3. **The research baseline itself was wrong** (BUGS A5/A9): flat 6.3% financing overcharging
   3.05pp/yr, and a missing credit gate. Honest 26yr baseline is **+12.72% / 0.549 / -55.9%**.

⇒ Treat "dead" verdicts here as **"probably dead, cheap to re-check"**, not settled.
See `RESEARCH_INDEX.md` for the current master list and `LOG.md` for the audited replacements.

---


====================================================================================================
# PROGRAM 2 — DYNAMIC LEVERAGE (2026-08-13/14)
### (was `DYNAMIC_LEVERAGE_FINDINGS.md`)
====================================================================================================

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


---

# ADDENDUM (2026-08-14) — the many-start standard, and two corrections it forced

`threadCANON` measured the per-start sigma of the DEPLOYED config over 12 monthly starts:
**8yr CAGR sigma 7.07pp** (range +15.03%..+34.55%), 26yr 2.75pp. `threadCORE5` measured a
DELTA's per-start sigma at ~2.30pp. The repo's documented "~0.6pp/start noise floor"
understates both by roughly 4x.

**Consequence: a 3-4 start A/B cannot distinguish ~2pp of edge from calendar luck.**
Sign-consistency across >=12 starts is the only test with power. It has already separated a real
effect from an artefact in both directions:

| result | starts | verdict |
|---|---|---|
| value-weight change (CORE2/3/4) | 6/12 positive on Sharpe, range -3.91..+3.70pp | **RETRACTED — noise** |
| off-cadence credit gate (DYN4-8) | **23/24** positive on Sharpe | **survives** |
| gross_margin bound (threadGM) | 8yr 8/12, 26yr 11/12 positive | **helps — earlier -1.50pp was wrong** |

**Correction to the gross_margin A/B.** threadGM reported the bound costing -1.50pp CAGR on the
8yr from 3 starts. Over 12 starts it is **+1.17pp (median +1.37, 8/12 positive)** on the 8yr and
**+0.48pp (median +0.54, 11/12 positive, sigma 0.35pp)** on the 26yr. The sign was wrong. The fix
improves performance as well as correctness.

**Honest canonical numbers:** 8yr **+22.75% +/-4.00pp**, Sharpe 0.79 +/-0.09; 26yr
**+11.26% +/-1.56pp**, Sharpe 0.51 +/-0.05. Not biased — the old point estimates sit on the
12-start means — but far less precise than two decimals imply.


====================================================================================================
# FRONTIER RESEARCH SUMMARY (2026-07)
### (was `FRONTIER_RESEARCH_SUMMARY.md`)
====================================================================================================

# Frontier Research Program — MASTER SUMMARY (plan: foamy-hopping-piglet)

Research-only, deployment-grade rigor, **NOTHING deployed live**. 2026-07-15.
Both periods (2018-25 ×3 starts, 2001-25 ×2 starts), deployed data OFF, real costs, 6.3%
financing on borrowed, causal signals, matched-control tests. No inflated numbers.

## One-line verdict
The equity book is at its frontier on its own axes, exactly as expected — but **two
orthogonal, deployable de-risk layers exist**, and they cover DIFFERENT crisis types so they
nearly ADD: **Thread B credit de-risk (free 2008 insurance)** + **Thread A micro-futures
(modest trend-crisis diversifier)**. Thread C (hold times) is a clean null.

## Thread B — credit/macro leverage signal → **STRONG DD-FIRST WIN** (the prize)
- **Deep-tail (p95) HY-OAS / BAA-AAA credit de-risk gate**: when credit spread is in the top
  ~5% of its expanding history, cut gross leverage ~50%. Cuts a 2008-style drawdown by
  **~10-20pp (26yr MaxDD −63% → −44/−50%) at EQUAL CAGR**, beating a matched-average-gross
  constant-leverage book by +12-19pp → the de-risk is **timed, not just smaller**.
- **Near-zero carry** when unneeded (p95: 8yr CAGR 33.8→33.5%, 26yr +0.3pp). Orthogonal to
  the book's own vol-scaling (which handles equity-vol crises; credit handles credit crises).
- Not a both-period Pareto win only because 2018-25 has no credit crisis to protect against —
  a state-contingent insurance, and it does NOT hurt 2018-25. Lever-up is Kelly-flat (dead).
  **DATA FIX:** HY OAS is available from 1996 (WRDS `fred_interest_rates_spreads_daily`), not
  2010 — the plan's "PC1 is the only both-period signal" was wrong. `_credit_signal.parquet`.
- Files: `threadB_signal_leverage.py`, `threadB_diag.py`, `THREAD_B_FINDINGS.md`.

## Thread C — per-stock / conditional hold times → **NULL (backtest-backed)**
- C1 mandatory gate: of rollover/vol_20d/dist_sma50-200/rsi_14/dist_52w_high/max_dd_6m/
  sma200_slope/ret_60d, **only rollover graduates** (monotone both periods) — and it says
  "hold the dip" (pulled-back winners OUTPERFORM), giving no exit trigger.
- **C3 dead**: vol_20d is flat both periods → a vol/regime-scaled stop has no support.
- Stop-width sweep: tighter (25%) hurts both periods (confirms hold-the-dip); uniform 40%/20d
  is the defensible both-period frontier. Matches prior "aging flat/inverted".
- Files: `threadC_holdtime_diag.py`, `threadC_stop_validate.py`, `THREAD_C_FINDINGS.md`.

## Thread A — micro-futures + equity → **QUALIFIED GO (reverses expected NO-GO)**
- Micro-ONLY book (12 micro-capable markets) is tradeable at $50k: **net Sharpe ~0.4-0.5**,
  holds ~5 markets. Prior −0.34 was an artifact of including unaffordable no-micro markets.
- Crisis convexity **partial**: 2020 +7%, 2022 +13%, 2018Q4 +7% (trend); **no 2008/2011 hedge**
  (bonds have no micro). corr to equity 0.25 (not 0). Sharpe gain to the book small (+0.02-0.05).
- DD-first realloc (w=0.2-0.3): −8 to −13pp DD for −3.6 to −5.6pp CAGR. Caveats: micros only
  exist since ~2019 (pre-2019 counterfactual); real-world roll/margin/execution cost not in sim.
- Files: `threadA_micro_gate.py`, `threadA_integrate.py`, `THREAD_A_FINDINGS.md`.

## Capstone — layered de-risk stack (26yr, `threadCAP_synergy.py`)
| stack | CAGR | Sharpe | MaxDD | 2008 | 2020 | 2022 |
|---|---|---|---|---|---|---|
| Base 1.49x vol-scaled | +20.2% | 0.78 | −62.9% | −46% | −27% | −7% |
| +B credit-p95 (free) | +20.3% | 0.79 | −53.3% | −36% | −28% | −7% |
| +A micro w0.2 | +17.8% | 0.82 | −53.7% | −39%* | −21% | −3% |
| **+B+A JOINT** | +17.9% | **0.83** | **−44.6%** | −30%* | −21% | −3% |
- Joint DD gain +18.3pp ≈ sum of parts (9.6+9.2): **near-additive because B and A hedge
  different crises** (credit vs trend) — better than the plan's expected sub-additive. Neither
  helps 2011. Best Sharpe (0.83 vs 0.78). (*2008 micro counterfactual; B's 2008 cut is real.)

## Recommended, deployable outcome (for user decision — NOT deployed)
1. **Add the p95 credit de-risk gate to the live leverage policy** — near-free 2008-style tail
   insurance, orthogonal to vol-scaling. Strongest single result. Re-validate in full run() +
   wire a daily causal HY-OAS/BAA-AAA expanding-percentile signal before any deploy.
2. **Micro-futures sleeve is optional/marginal** at $50k — real but thin edge, operationally
   heavy; better revisited at higher AUM where bonds (the 2008 hedge) become affordable.
3. **Leave hold-times / stops as-is** — Thread C confirms 40%/20d is at the frontier.


====================================================================================================
# WALK-FORWARD OOS METHOD
### (was `WALKFORWARD_OOS_FINDINGS.md`)
====================================================================================================

# Walk-forward OOS validation — FINDINGS

**2026-07-15. Harness: `research/threadWF_validation.py`. Research-only.**
Answers: (1) is the ~28% in-sample CAGR achievable OOS? (2) does the credit p95 gate survive OOS?

## Method
Rules strategy -> walk-forward = re-SELECT config each year from past data, score held-out
next year, roll. Grid = top_n{3,5,8} x rebal{15,20,30} x stop{.30,.40,.55} (27), weights fixed
live. Selection metric = trailing expanding-window Sharpe. All at 1.49x flat + 6.3% financing.
Then apply the FROZEN credit p95 gate (causal expanding-pctile) to the walk-forward book.

## Results
**LONG 2001-25 (22 held-out yrs — reliable):**
| variant | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| Live config in-sample (OOS window) | +23.8% | 0.75 | −73.6% |
| **Walk-forward selected (TRUE OOS)** | **+21.3%** | 0.68 | −73.6% |
| **WF + credit p95 gate (OOS)** | **+23.1%** | 0.73 | **−57.0%** |
| Oracle best (ex-post upper bound) | +23.8% | 0.76 | −71.9% |

**SHORT 2018-25 (6 held-out yrs, 2020-25 — noisy):**
| variant | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| Live config in-sample (OOS window) | +38.0% | 1.02 | −39.3% |
| **Walk-forward selected (TRUE OOS)** | **+24.9%** | 0.81 | −38.6% |
| WF + credit p95 gate (OOS) | +24.9% | 0.81 | −38.6% |

## Verdict
1. **28% is IN-SAMPLE. Honest OOS haircut:** long ~2.5pp (23.8→21.3, reliable, 22 folds);
   short ~13pp (38→25) but INFLATED — 2020-25 is a monster-momentum window and a 2y-burn-in WF
   selection is noisy (missed the ex-post-best live config in 4/6 yrs). Long is the trustworthy
   estimate: **expect ~21% bare / ~23% with credit gate OOS, Sharpe ~0.68-0.73** (1.49x flat).
2. **Credit p95 gate CONFIRMED OUT-OF-SAMPLE:** on the re-selected walk-forward long book it cut
   MaxDD −73.6%→−57.0% (+16.6pp) AND +1.8pp CAGR (recovers the WF haircut), Sharpe 0.68→0.73.
   2008 de-risk used only pre-2008 data -> genuine OOS, not curve-fit. Added nothing on 2020-25
   (no credit crisis there) — consistent.
3. **Live config is well-chosen** — its ex-post Sharpe (0.75) ~= oracle (0.76); not overfit to a
   lucky corner. But a mechanical WF drifts to WIDER stops (0.55) and never re-picks the live 40%,
   landing ~2.5pp lower — the backtested 28%/24% carries config-selection hindsight.

Caveats: selection = trailing Sharpe over a fixed 27-config grid (other metric/grid shifts the
number); 1.49x flat, no vol-scaling. OOS CAGR is a range, not a point.


====================================================================================================
# LIVE-MIRROR HARNESS FINDINGS
### (was `LIVEMIRROR_FINDINGS.md`)
====================================================================================================

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


====================================================================================================
# THREAD A
### (was `THREAD_A_FINDINGS.md`)
====================================================================================================

# Thread A — Micro-futures + equity — FINDINGS

**Status: COMPLETE — QUALIFIED GO (reverses the expected NO-GO), research-only.** 2026-07-15.
Harness: `research/threadA_micro_gate.py` (A1/A3) + `research/threadA_integrate.py` (A2/A4).

## Headline
**A MICRO-ONLY futures book IS tradeable at $50K** (net Sharpe ~0.4-0.5), reversing the
prior "futures untradeable at $50K / −0.34 Sharpe" — that verdict was an artifact of
including no-micro markets (bonds, global equity, ~$100k notionals) a small account can't
afford. Restricting to the 12 micro-capable markets that FIT the account flips the sign.
The benefit to the equity book is real but **modest** and comes with hard caveats.

## A1 feasibility gate (2010-26 sim, integer micro contracts, tiered costs)
Micro-capable markets: ES,NQ,YM,RTY (equity), 6E,6J,6B,6A (FX), CL,GC,SI,HG (commodity).
| AUM | net Sharpe | mkts held | CAGR / MaxDD |
|---|---|---|---|
| $30k | 0.28 | 3.2 | +2.4% / −20% |
| **$50k** | **0.51** | 4.7 | +5.0% / −18% |
| $100k | 0.35 | 6.3 | +3.2% / −27% |
| $1M | 0.34 | 8.7 | +3.1% / −29% |
(IDEAL uncapacitated Sharpe 0.37 — so the underlying edge is ~0.35-0.40; $50k realizes 0.51
partly by favorable rounding. **PASS: net Sharpe > 0 at $50k.**)

## A3 crisis convexity (micro book at $50k, NO bonds — no micro exists)
2020 COVID **+7.3%**, 2022 bear **+12.6%**, 2018 Q4 **+7.1%** (trend-driven convexity survives),
but **2008 GFC −3.6%, 2011 EU −7.7%** (dropping bonds guts the credit/rates crisis hedge, as
the plan feared). It's a return diversifier with trend-crisis convexity, NOT a 2008 hedge.

## A2 overlay / A4 realloc with equity v12 (2010-25, 3-start)
Equity-only baseline: CAGR +28.0%, Sharpe 0.83, MaxDD −49.8%. **corr(eq, micro) = 0.25**
(NOT ~0 — the book is 1/3 equity-index micros).
| variant | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| A2 overlay w=0.3 (CAGR-first) | +29.6% | 0.85 | −48.8% |
| A2 overlay w=0.5 | +30.6% | 0.86 | −48.3% |
| A4 realloc w=0.3 (DD-first) | +22.4% | 0.86 | −36.6% |
| A4 realloc w=0.5 | +18.0% | 0.88 | −27.2% |
Overlay = small Pareto improvement (stacks margin); realloc = clean DD-first trade.

## Honest verdict — QUALIFIED GO with caveats
- Sharpe gain is small (+0.02-0.05); corr 0.25 not 0 (weaker diversifier than a bond book).
- **Micro contracts only launched ~2019** — the 2010-2018 sim is counterfactual; only 2020/2022
  are real micro-era crisis tests (and those are exactly where it helped). Forward-valid
  (micros exist now), but no real 2008 test and no 2008 hedge.
- Operational cost of running futures (rolls, overnight margin, execution) in a $50k IBKR
  account is real and not in the sim — likely erodes the modest edge.
- **Recommendation:** viable as a small DD-first sleeve (realloc w=0.2-0.3: −8 to −13pp DD for
  −3.6 to −5.6pp CAGR, Sharpe +0.02-0.03), but the edge is thin enough that the complexity may
  not be worth it at $50k. Revisit at larger AUM where bonds become affordable (restores the
  2008 hedge). NOT the honest NO-GO expected — a genuine, if marginal, GO.


====================================================================================================
# THREAD B
### (was `THREAD_B_FINDINGS.md`)
====================================================================================================

# Thread B — Futures/macro/credit as a leverage signal — FINDINGS

**Status: COMPLETE & VALIDATED (research-only, not deployed).** 2026-07-15.
Harness: `research/threadB_signal_leverage.py` (suite) + `research/threadB_diag.py` (attribution).
Method: leverage overlay on 1x book returns (`gross_t·r_1x − financing`), 6.3% on borrowed;
signals causal (expanding-percentile, act next day); both periods, multi-start.

## Headline
**Credit-spread de-risk is a validated crisis-convexity insurance overlay — a genuine
DD-first result, but STATE-CONTINGENT, not a continuous both-period Pareto win.**

## Data correction to the plan
Plan assumed HY OAS is 2010→ (local `macro_fred.parquet`, actually only populated 2023→).
Real source found: **WRDS `fred_interest_rates_spreads_daily.parquet` has `bamlh0a0hym2`
from 1996** + `dbaa`/`daaa` (BAA-AAA) from 1986. Spliced with local tail →
`research/_credit_signal.parquet` (hy_oas 1996→2026, baa_aaa 1986→2025). **Credit IS
both-period testable** — the plan's "PC1 is the only both-period signal" was wrong.

## Results (3-start 8yr, 2-start 26yr; incumbent = LIVE 1.49x vol-scaled)

| Policy | 8yr CAGR / MaxDD | 26yr CAGR / MaxDD | vs matched-gross (26yr) |
|---|---|---|---|
| **Incumbent 1.49x** | +27.9% / **−32.9%** | +20.2% / **−62.9%** | — |
| B2 credit p80 ×0.5 | +24.2% / −33.0% | +20.1% / **−50.0%** | **+12.7pp (timed)** |
| B2 **baa-aaa p80 ×0.5** | +25.1% / −32.9% | +20.7% / **−44.1%** | **+17.0pp (timed)** |
| B2 credit **p95** ×0.5 | +27.8% / −32.4% (≈0 cost) | +22.1% / **−52.7%** | +10.9pp (timed) |
| B4 credit GATE p80 | +26.6% / −32.9% | +20.5% / −59.6% | +4.1pp |

(Full suite incl. PC1, credit∧PC1, blend, lever-up, VIX-term in the harness output.)

## Why it works — crisis-window attribution (`threadB_diag.py`)
- **26yr incumbent MaxDD = GFC (2007-10→2009-03, −63.6%).** Credit de-levered INTO it:
  2008 window gross 1.04→0.52-0.65, window return **−47% → −26 to −29%**. Both starts:
  +12-13pp (credit p80), +18-19pp (baa-aaa) DD gain, CAGR preserved/better. Beats a
  constant-leverage book at the SAME avg gross by +12-19pp → the de-risk is **TIMED, not
  just smaller leverage** (this was the key control).
- **8yr incumbent MaxDD = Q4-2018 (−32.4%)** — a rate/growth equity selloff, NO credit
  blowout → credit can't help it (DD flat, every start). Credit DID cushion COVID
  (−13.7%→−7.0%) and 2022, but those weren't the 8yr max DD.
- **p95 deep tail = "cheap insurance":** near-ZERO CAGR cost when unneeded (8yr
  33.8→33.5%, 26yr +0.3pp) while still delivering +8-11pp DD protection in 2008.

## Honest verdict
- Credit de-risk is **orthogonal** to the book's own vol-scaling: vol-scaling handles
  equity-vol crises (2020 speed, 2022, Q4-18); **credit handles credit-led crises (2008)**
  that vol-scaling is slower on. Its benefit is only *observable* in a sample containing a
  credit crisis → invisible in 2018-25, huge in 2001-25. NOT a data artifact; the economics
  are coherent (spreads lead the 2008 equity drawdown by weeks).
- Fails the STRICT both-period Pareto/DD-first gate **only because 2018-25 has no credit
  crisis to protect against** — and it does NOT materially hurt 2018-25 at p95 (~0 cost).
- **Lever-up (B5b/B1up) is Kelly-flat as predicted** — no CAGR gain; not worth the risk.
- **PC1 alone** (B1): weak/inconsistent (26yr p80 −2.3pp DD, p90 +1.0pp) — credit dominates.
- **VIX-term (B3):** 8yr-only, modest; can't clear both-period bar by construction.

## Recommendation (for user decision — NOT deployed)
Add a **deep-tail (p95) HY-OAS or BAA-AAA credit de-risk gate** to the live leverage policy:
when the credit spread is in the top ~5% of its expanding history, cut gross leverage ~50%
(or to the vol-scale floor). Near-zero carry in calm regimes; cuts a 2008-style drawdown by
~10-20pp. It is crisis insurance, not a continuous alpha — deploy it as such. baa-aaa p80 is
the strongest protector (−44%); credit p95 is the cheapest-carry. Next step before any deploy:
re-validate the chosen variant inside the full `run()` and wire the same causal expanding-
percentile signal the live engine can compute daily.


====================================================================================================
# THREAD C
### (was `THREAD_C_FINDINGS.md`)
====================================================================================================

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


====================================================================================================
# THREAD R
### (was `THREAD_R_FINDINGS.md`)
====================================================================================================

# Thread R — lever-up, richer regimes, quality — FINDINGS ("push harder" round)

**2026-07-15. Research-only.** User pushed: lever UP in good times (not just de-risk down),
richer regime *mixes* (not just leverage), and add quality. Harnesses: `threadB2_leverup.py`,
`threadR_regime_quality.py`, quality-filtered momentum wired into `livemirror_backtest.py`.
All full-stack (1.49x integer $50k, vol-scaling + stops) unless noted. Both periods, multi-start.

## Verdict: everything null-to-negative except one CAGR-first hint. The credit DOWN-gate stays
the only robust addition. The de-risk-only asymmetry is CORRECT, not a gap (see bottom).

### 1. Lever UP in confirmed-good regimes (threadB2 overlay + full-stack)
- Pure credit-calm lever-up (to ~1.7-1.9x): **Kelly-flat/negative** — adds CAGR but LOSES
  Sharpe and deepens DD (short 1.49x→1.9x: DD −32.9%→−36.1%, Sharpe 1.00→0.96).
- credit-calm + SPY-uptrend lever-up: overlay hinted +2.2pp CAGR at equal DD on LONG (beat
  matched-gross), BUT in the full stack (with vol-scaling interaction) it's −0.2pp CAGR /
  −0.05 Sharpe on long, +1.5pp CAGR / −0.03 Sharpe / deeper DD on short. NOT robust.

### 2. Richer regime MIX — strong-bull momentum tilt (mom 0.50→0.70 when breadth high + uptrend + credit-calm)
- **NULL: +0.1pp CAGR both periods, ~0 Sharpe.** Reason: in a strong bull, momentum/value/
  quality all rise together, so tilting the MIX toward momentum barely changes the outcome.
  Sleeves converge exactly when you'd want them to diverge; the tilt only bites at turning
  points, where it's too late. Adding a lever-up to the bull state didn't help either.

### 3. Quality
- **More quality SLEEVE (s5 15%→25%): HURTS** — short −2.4pp CAGR/−0.05 Sharpe; long neutral.
- **Quality-FILTERED momentum (keep top-5 by gp_assets/roe from a 2-3x momentum pool): HURTS
  badly** — short −9.7pp CAGR/−0.25 Sharpe (gp_assets), −11.6pp (roe); long −1.2 to −3.1pp.
  In SP1500 (already quality-screened) a quality filter removes the high-octane momentum
  leaders (2018-25 winners were "expensive" growth) that drive the return. Confirms prior
  "all Compustat factors HURT momentum" — as a sleeve AND as a filter.

### 4. One directional hint (within noise)
- Slightly MORE base momentum (50→60% mom): +1.0pp CAGR short, +0.3pp long, ~0 Sharpe change.
  Marginal, inside the ~0.6pp/start noise floor — not actionable, but direction says the book
  is momentum-hungry, not quality-hungry.

## Why up-risking is not a free lunch (the asymmetry is a feature)
De-risking in a crisis is CONVEX for geometric growth — it avoids the deep drawdowns that
destroy compounding (a −60% needs +150% to recover), so cutting exposure in the left tail
adds long-run CAGR. Levering UP at the Kelly peak is CONCAVE — extra leverage in good times
adds variance faster than return (you're already capturing the upside), so it's Kelly-flat and
raises ruin risk. That is exactly why the credit DOWN-gate works (free, +DD protection) and
every lever-UP / more-aggressive-in-bull variant doesn't. The current de-risk-only design is
optimal on this axis, not an oversight.

## Net
The book is at its efficient frontier on the mix / leverage / quality axes. The only validated
additions from the whole program remain: **credit DOWN-gate (free tail insurance)** and,
optionally, the marginal **micro-futures sleeve** (Thread A). Everything the "push harder"
round tested is null or negative — reported honestly.


====================================================================================================
# THREAD S
### (was `THREAD_S_FINDINGS.md`)
====================================================================================================

# Thread S — sector caps + vol-managed momentum — FINDINGS (deeper push)

**2026-07-15. Research-only.** Two theory-backed, previously-untested momentum risk controls.
Harness: `threadS_sector_volmom.py` + sector-cap/vol-managed-mom wired into livemirror. Full
stack (1.49x integer $50k), both periods, multi-start.

## Verdict: both null-to-negative on the both-period bar. Same structural pattern as Thread R.

### Sector-concentration cap (max K of top-5 momentum per 2-digit SIC — the book had NO sector control)
- max 2/SIC: **null** (short −0.1pp, long −0.0pp) — rarely binds; top-5 momentum seldom has
  >2 in one 2-digit SIC.
- max 1/SIC (force 5 different sectors): **mixed** — long +0.8pp CAGR/+0.02 Sharpe, but short
  −1.9pp CAGR/−0.04 Sharpe. Forcing diversification dilutes the concentrated momentum that IS
  the alpha in recent regimes (2020 tech). Not a both-period win.

### Vol-managed momentum (Barroso: scale mom sleeve by target/own-vol)
- t0.25: short −4.3pp CAGR (over-de-risks), long −0.8pp/+0.02 Sharpe.
- t0.35: short −2.3pp CAGR, long +0.1pp/+0.02 Sharpe.
- Marginal Sharpe help on LONG, consistent CAGR/Sharpe cost on SHORT. Not a both-period win.
- **Why it doesn't help here (it's a documented winner elsewhere):** the book ALREADY has TWO
  momentum-crash protections — the price-UMD crash detector (switches to defensive weights) and
  portfolio vol-scaling. Sleeve-specific vol-management is largely REDUNDANT with those, so its
  extra de-risking just costs CAGR in normal times.

## The structural pattern (across Threads B, C, R, S)
Every INTERNAL lever — reweighting sleeves, regime mix, quality (sleeve/filter), sector caps,
vol-managed momentum, hold-times, lever-up — is **null-to-negative on the both-period bar**.
The ONLY additions that work are signals ORTHOGONAL to what the equity book already sees:
the **credit de-risk gate** (external macro the book is blind to). The equity cross-section's
alpha (momentum, its regime timing, its crash protection) is already fully extracted; you
can't squeeze more by rearranging it. Gains require an orthogonal EXTERNAL signal.

Bonus (modeling): the credit gate is slightly BETTER live than backtested — the backtest
credits de-levered cash 0%, but the live IBKR account earns ~4-5% on idle cash during the
gate's de-risked periods. So real-world the gate's carry is positive, not zero.

## Net
Confirms the book is at its efficient frontier on all internal axes. The credit down-gate
remains the single validated, deployable addition. The next orthogonal frontier (if pursued)
is OTHER external macro-stress signals (funding/TED, treasury-vol, dollar) — but Thread B
already showed credit dominates and two-signal stacking (credit∧PC1) didn't beat credit alone,
so expected marginal.


====================================================================================================
# THREAD S3
### (was `THREAD_S3_FINDINGS.md`)
====================================================================================================

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


====================================================================================================
# THREAD T
### (was `THREAD_T_FINDINGS.md`)
====================================================================================================

# Thread T — macro-stress composite de-risk — FINDINGS

**2026-07-15. Research-only.** Does a composite (credit + rates-vol + curve) beat credit-alone
as the leverage de-risk gate? Harness: `threadT_macro_composite.py`. Data:
`research/_macro_stress.parquet` (hy_oas 1996+, rates_vol=21d realized vol of DGS10 daily
changes [MOVE proxy] 1962+, curve=T10Y2Y). Overlay, both periods, multi-start, matched-gross.

## Headline: YES — modestly. Rates-vol is a slightly BETTER signal than credit; OR(credit,
## rates-vol) gives the deepest drawdown. Curve is a bad trigger (drop it).

| policy (LONG 2001-25) | CAGR | Sharpe | MaxDD | timed(match dDD) |
|---|---|---|---|---|
| baseline 1.49x | +20.2% | 0.78 | −62.9% | — |
| credit only (p95) | +20.3% | 0.79 | −53.3% | +10.2p |
| **rates-vol only (p95)** | **+20.8%** | **0.81** | −53.8% | +9.5p |
| **OR(credit, rates-vol)** | +20.4% | 0.80 | **−52.0%** | +11.1p |
| curve only | +18.3% | 0.75 | −62.9% | −1.9p (HURTS) |
| OR all three | +18.8% | 0.78 | −52.0% | (curve drags CAGR) |
| blend avg-pctile | +20.0% | 0.78 | −62.9% | (too weak to fire) |

SHORT 2018-25: rates-vol best Sharpe (1.03 vs credit 1.00); no policy cuts the Q4-2018 MaxDD.

## Why rates-vol adds value (crisis attribution)
Credit de-risks 2008 (gross 1.04→0.75) but NOT 2022 (1.07→1.07). **Rates-vol de-risks 2022**
(the OR composite 1.07→0.97) — 2022 was a RATES/duration crisis, mild in credit but a huge
MOVE spike. hy↔rates-vol corr is only 0.57, so genuine orthogonality. Rates-vol fires on BOTH
2008 and 2022 → higher CAGR/Sharpe than credit alone in both periods.

## Verdict
- **Rates-vol (MOVE proxy) is the single best de-risk trigger** — beats credit on CAGR AND
  Sharpe in both periods, similar DD. Economically sound (systemic-stress gauge, well-known).
- **OR(credit, rates-vol) is the best for drawdown** — catches credit crises (2008) AND rates
  crises (2022); deepest DD protection (−52.0%), timed. Recommend this as the deploy candidate.
- **Curve/inversion: DROP** — de-risks too early (inverts ~1yr before crashes), misses the
  final rally, hurts CAGR. Averaging-blend too weak.
- Gains over credit-alone are MARGINAL (+0.5pp CAGR, ~1pp DD) but CONSISTENT in both periods,
  timed vs matched-gross, and add real 2022 coverage. Multiple-comparison caveat noted (6
  policies tested); the both-period consistency + economic logic mitigate.

## FULL-ENGINE VALIDATION (threadT_validate.py — de-inflated vs overlay)
The overlay OVERSTATED rates-vol. Real full live-mirror (1.49x integer $50k, LONG 2001-25):
| variant | CAGR | Sharpe | MaxDD | dMaxDD |
|---|---|---|---|---|
| baseline | +20.8% | 0.78 | −63.5% | — |
| credit only p95 | +21.1% | 0.79 | −56.6% | +6.9p |
| rates-vol only p95 | +20.4% | 0.77 | −60.3% | +3.2p |
| OR(credit,rates-vol) p95 | +20.7% | 0.78 | −54.7% | +8.9p |
- rates-vol ALONE is weak in the full engine (+3.2pp, not the overlay's ~+9pp) — it OVERLAPS
  vol-scaling (which already de-risks the equity-vol spikes that accompany rate shocks). Credit
  is more orthogonal to vol-scaling.
- OR composite: +8.9pp DD at −0.1pp CAGR — ~2pp better than credit-alone, real but modest.
- ROBUST, not a knife-edge: p90/p95/p97 give a SMOOTH monotonic tradeoff (p90 −49.1%/−0.8pp
  CAGR, p95 −54.7%/−0.1pp, p97 −58.5%/−0.3pp). p95 is the sweet spot.
- No period-fitted params (expanding percentile = walk-forward by construction). Not inflated.

## Deploy recommendation (updated, post-validation)
**Primary: credit-alone p95** — simplest (one FRED series), captures ~78% of the DD benefit
(+6.9 of +8.9pp), best CAGR (+0.3pp). **Optional enhancement: add rates-vol (OR)** for ~2pp
more 2022-type protection at ~flat CAGR. Curve DROPPED.

## (original overlay recommendation below — superseded by full-engine validation)
Prefer the **OR(credit-p95, rates-vol-p95)** de-risk gate over credit-alone: same free-tail-
insurance property, wider crisis coverage. Data: FRED `BAMLH0A0HYM2` (credit) + `DGS10`
(compute rates-vol locally) — both free, daily. See live-wiring notes.


====================================================================================================
# THREAD X
### (was `THREAD_X_FINDINGS.md`)
====================================================================================================

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


====================================================================================================
# FUTURES / NORGATE GLOBAL MACRO
### (was `FUTURES_FINDINGS.md`)
====================================================================================================

# Futures deep-dive — findings (Norgate gold-standard panel)

Data: Norgate continuous panel, **98 liquid global-macro markets, 1977–2026**, plus
27,261 individual contracts (term structure) + cash commodities. Return construction
validated: `ccb.diff()/nonadj.shift(1)` (difference back-adjusted; ES → 19.4% vol /
8.2% drift, exactly right). One consistent methodology across every test
(`futures_lib.py`): per-market vol-targeting, 1-day execution lag, 1.5bps costs,
12% portfolio-vol overlay.

## Sleeve results (vol-scaled to 12%)

| Sleeve | Full Sharpe | Recent (2015–26) | Crisis-alpha | Verdict |
|---|---|---|---|---|
| **Trend (TSMOM 1/3/12m)** | 1.18 | **0.06** | 2008 +49%, 2020 +20%, 2022 +10% | Convex crisis hedge; standalone DECAYED |
| Trend (diversifying only) | 1.18 | 0.22 | same | Dropping equity-index trend helps recent |
| **Cross-sec momentum** | 1.12 | **0.43** | 2000 +85%, 2022 +21% | Best recent; decayed least |
| **Carry (term structure)** | 0.92 | 0.19 | 2008 +22%, **2020 −6%, 2022 −5%** | Distinct (corr 0.14); earns in calm, NOT a crash hedge |
| Value (5y reversal) | −0.67 | −0.99 | — | **DEAD** (just anti-momentum). Don't use. |

## The prize — combined multi-strat book
**Trend_div + xsmom + carry (equal risk):**
- Full Sharpe **1.44**, MaxDD −24.5% (vs trend-alone 1.18 / −31.5%)
- **Recent (2015–26) Sharpe 0.43; 2020s Sharpe 0.55** — diversification largely fixes
  trend's calm-market bleed (carry is the MVP diversifier, corr 0.14 to trend)
- Crisis convexity PRESERVED: 2000 +137%, 2008 +48%, 2020 +14%, 2022 +10%; corr −0.14

## Integration with the live v12 equity book (2018–2025)
- corr(v12, futures book) ≈ **0.00** — genuinely diversifying
- **Overlay** (capital-efficient, futures on margin): modest — Sharpe 1.12→1.13,
  MaxDD −24%→−21% at 30–40% overlay; 2020 −24%→−21%, 2022 −20%→−19%
- **Reallocation** (move 25% capital to futures): MaxDD −24%→−16%, 2020 −16%, 2022
  −14%, but CAGR 21.6%→16.8% (you give up equity in a bull)
- HONEST: benefit is modest over 2018–25 because that window is a huge equity bull
  AND the futures book's weakest era, with no 2008-style crash. The value rests on
  TAIL protection (2008 +48% vs equity −45%), which the short window can't exercise.

## Dead ends (don't re-test)
- **Value (5y reversal):** dead.
- **Cross-asset regime signal to time the equity de-risk:** does NOT beat the simple
  SPY<200d rule (Sharpe 0.37 vs 0.43, worse DD). The live de-risk is already at frontier.

## Bottom line
The new data genuinely unlocks **one new thing**: a fundable diversified global-macro
futures book (trend+xsmom+carry), recent Sharpe ~0.4–0.55, real crisis convexity,
~0 correlation to the equity book. It's a legitimate diversifying return stream / tail
hedge — NOT a high-return engine (all premia decayed from their 80s–90s peaks).
Caveats: idealized (1.5bps, no market impact, daily rebal, market-selection
survivorship) — net recent Sharpe likely ~0.3–0.45. At $30K, the modest diversification
benefit is marginal vs ongoing futures data/exchange/commission costs; the case
strengthens with AUM. Live execution feasible via IBKR micros; live continuous series
buildable from IBKR and verifiable against this Norgate snapshot.

---

# Deeper dive — first-principles, not the factor zoo

**Pattern that emerged: you CANNOT use the futures panel to predict/time equities
(efficient market), but you CAN harvest it as a diversifier — and conviction-weighting
makes the book meaningfully better.**

- **Macro PCA (2000–26):** PC1 (16%) = risk axis (equities vs bonds), PC2 (10%) =
  dollar/liquidity. Real & economically interpretable — but no equity-predictive edge.
- **Crash-timing via cross-asset stress** (dollar + flight-to-quality + VIX): **DEAD.**
  "200d OR stress" is byte-identical to 200d alone; stress fires *later* than the 200d
  (2020: Feb-17 vs Feb-3). SPY<200d is at the frontier — don't complicate it.
- **Factor timing (momentum vs value) via macro state: DEAD.** IC≈0 (commod +0.03,
  rates −0.01, dollar −0.03, combined −0.007); regime-tilted Sharpe 0.32 < fixed 0.56.
  Factor timing fails OOS, as the literature warns.
- **Cross-asset lead-lag: DEAD** at the tradeable horizon. fwd-1d ICs all < 0.04;
  fwd-5d "notables" have inconsistent signs + overlap artifacts. Equities efficient.
- **★ IMPROVED BOOK — the real win.** Conviction-weighting (risk-adjusted trend
  *strength* via tanh, not just sign) + dropping equity-index trend:
  recent (2015–26) Sharpe **0.32→0.60**, 2020s **0.39→0.77**, MaxDD −25%→−23%, crisis
  convexity kept (2008 +46%, 2020 +9%, 2022 +9%). OI trend-quality filter ≈ neutral.
  (Haircut for in-sample variant selection → realistic net ~0.45–0.5.)
- **Integration with the improved book:** capital-efficient overlay (futures on margin)
  at 50% → v12 **Sharpe 1.04→1.20, CAGR 19.9%→24.2%**, drawdown flat-to-better, corr
  −0.07. Now adds *return AND Sharpe AND* crisis protection — not just modest DD.

**REVISED VERDICT:** the actionable output is the **conviction-weighted diversified
futures book as a capital-efficient overlay** on the equity strategy. It crossed from
"marginal" (v1) to "worth seriously considering": ~+0.16 Sharpe + crisis convexity at
zero capital cost (margin). Remaining honest caveats: idealized costs, $30K
capacity/contract-granularity, ongoing data/exchange fees. Case strengthens with AUM.
The equity strategy itself can't be improved by this data (timing & tilts are efficient);
the value is purely the diversifying overlay.

---

# GO/NO-GO: realistic costs + $30K capacity (futures_capacity.py)

Pushed book construction further (futures_book_v3.py): more trend horizons / risk-parity
sleeve weighting = **diminishing returns** (recent Sharpe 0.59→0.62 but worse DD; risk-
parity ≈ neutral). Conviction-weighting was the real win; we're at the construction
ceiling (~recent Sharpe 0.6, idealized).

Then the decisive stress test — integer micro/full contracts at real notionals + tiered
realistic costs (2–10bps/side by liquidity), STIR excluded, 7x lev cap, weekly rebal:

| | Sharpe (2010–26) | markets held | gross |
|---|---|---|---|
| IDEAL gross (no cost) | 0.90 | — | — |
| IDEAL net (realistic costs) | **0.34** | — | 2.6x |
| **AUM $30k** | **−0.34** | **0** | 0.05x |
| AUM $100k | −0.17 | 2 | 0.19x |
| AUM $300k | 0.17 | 7 | 0.38x |
| AUM $1M | 0.13 | 22 | 1.26x |
| AUM $3M | 0.45 | 43 | 1.63x |
| AUM $10M | 0.49 | 58 | 1.76x |

**Two decisive findings:**
1. **Realistic costs roughly HALVE the edge** (idealized 0.90 → net 0.34). The 1.5bps
   numbers overstate by ~2x. (Weekly rebal is conservative; monthly recovers some.)
2. **At $30K it is a hard NO-GO.** 0 of 41 targeted markets can hold even ONE contract;
   **no bonds** (the key crisis hedge has no micro, ~$100k notional); gross 0.05x = 95%
   uninvested. The diversified book is *physically unbuildable* at this size. It needs
   **~$2–3M+** to approximate the ideal (Sharpe ~0.45 at $3M); below ~$300k it's
   nonfunctional.

**FINAL:** the attractive overlay (v12 Sharpe 1.04→1.20) is real ON PAPER but
**unimplementable at $30K** — it requires ~$2–3M of capital to hold the contracts. Plus
realistic costs halve the standalone edge. **Decision: do NOT build at current size.**
Shelve as a validated, ready sleeve; revisit at ~$2–3M+ AUM. Norgate snapshot + scripts
are the reusable asset for that future build.


====================================================================================================
# VRP PLAN (never executed)
### (was `vrp_plan.md`)
====================================================================================================

# VRP Strategy Plan (Saved for Future Implementation)
# Date: 2026-04-29
# Status: PLANNED — Not started
# See full plan at: .claude/plans/pure-honking-waterfall.md

## Quick Summary
- Strategy: Put credit spreads on SPY (defined risk)
- Expected: 5-8% CAGR standalone, Sharpe 0.5-0.8
- Combined with momentum: Sharpe 1.5-1.8, DD -14-18% (vs -21.4% now)
- Timeline: ~12 months to full deployment
- First step: Verify Alpaca options support

## Key Decisions Needed Before Starting
1. Broker for options (Alpaca vs TastyTrade/IBKR)
2. Buy ORATS data for backtesting ($99-200/month) or skip
3. Separate or same account for equity + options
4. Manual paper trades first (10-20 trades)

## Capital Ramp: 0% → 5% → 10% → 15% → 20% over 12 months
