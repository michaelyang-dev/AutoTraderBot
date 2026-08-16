# BUGS — every leak, accounting error and data defect found in this project

Re-check every new experiment against this list. The same bug reappears constantly.
Newest first within each section.

---

## A. HARNESS AUDIT 2026-08-14 (this research program, cycle 0)

Scripts: `research/AUDIT01_harness_leaks.py`, `research/AUDIT02_falsification.py`.
Universe: `complete_sp1500_universe.pkl` (8yr), harness `LiveMirrorBacktester`.

### A0. VERDICT: the harness is not leaking. Numbers produced by it can be believed.

| test | result | reading |
|---|---|---|
| **shift +1 bar** (every signal lagged one full session: features, closes-for-signals, UMD, breadth; fills still at date-t close) | **−0.35pp CAGR, −0.010 Sharpe** | **GRACEFUL.** A look-ahead bug dies or flips sign here. It does neither. |
| **forward-IC sweep** (rank-IC of all 29 panel features vs forward 20d return, 37 quarterly cross-sections) | max \|IC\| = **0.050** (`vol_60d`, negative) | No feature is contaminated. 0.05 is the low-vol anomaly, not a leak. A leaked target shows \|IC\| > 0.3. |
| **cost ×2 / ×3 / ×5** | +20.98% / +19.16% / **+15.66%** | Edge survives 5× modelled cost. Not a cost-accounting artefact. |
| **shuffle** (sleeves → random draws from the same PIT membership) | see `_audit02_8yr.out` | — |
| non-positive prices | **0** | fixed in the 2026-07-28 rebuild (defect #8) |
| \|1-day return\| > 100% | 170 of 6.0M obs (**0.0028%**) | tolerable; none are systematic |
| interior NaN gaps | 45 symbols, 4,002 day-cells | held at `entry_px` through the gap — see A2 |

### A1. ⚠️ ZERO-LATENCY CLOSE-TO-CLOSE EXECUTION (known, accepted, documented)

`ret_20d` on date D matches the trailing return **including D's close** exactly
(corr 1.000000, MAD 0.000000). The harness then fills at **D's close**. So the model is:
observe close, trade at that same close.

- This is **not** a look-ahead in the strict sense, and it is **faithful to live** — the live
  signal server computes at ~15:50 ET and `ibkr_engine` submits MOC/near-close.
- The shift test above prices the assumption: **0.35pp of CAGR / 0.010 of Sharpe.**
- **Rule for this program:** any new signal must be computable from data available at 15:50 ET
  on the decision day. Anything using the settle/official close of day D to trade day D is not
  implementable and must be shift-tested before it is believed.

### A2. ⚠️ MISSING-PRICE FALLBACK VALUES HOLDINGS AT `entry_px`

`today.get(sym, h["entry_px"])` — a held name with no price on a date is marked at its **entry
price**, and at the next rebalance is *sold* at entry price.

- Direction measured (`UNIVERSE_MANIFEST.md`): entry_px **+26.55%** vs carry-forward
  **+27.21%**. The shipped behaviour is **conservative by ~0.7pp**, because ~95% of delistings
  in this universe are acquisitions marked at cost instead of takeout value.
- Stale marks occur on **1.07% of position-days**.
- **Not fixed** — fixing it would raise reported returns, and an understatement is the safe
  side. Recorded so nobody "fixes" it and books 0.7pp of fake improvement.

### A3. ⚠️ SHARPE IS COMPUTED WITHOUT THE RISK-FREE RATE

`dr.mean()/dr.std()*√252` everywhere. Overstated by ≈ `rf/vol` (≈0.10–0.15 at current rates).
Consistent across arms, so **deltas are fine**; absolute levels are not comparable to published
Sharpe ratios. Do not "fix" mid-program — it would invalidate every logged comparison.

### A4. ⚠️ THE OUTLIER TEST NEEDS A BENCHMARK TO MEAN ANYTHING

Dropping the best 5 / 10 / 20 days takes 8yr CAGR **+22.85% → +14.42% / +9.33% / +1.45%**.
That looks alarming and is **not, on its own, evidence of anything** — a levered long-only
equity book inherits the well-known "miss the best days" property of the index itself.

**The correct form of the test, and the one this program uses, is event-concentration of the
EXCESS return versus the baseline** (`threadCONC`), not of the total return. Applied that way it
has already killed a finding that passed everything else (off-cadence credit gate: 99.6% of its
26yr excess came from 5 days). Any raw "drop best N days" number logged without a benchmark
comparison is uninformative and should be labelled as such.


### A5. 🔴 FLAT 6.3% FINANCING IS WRONG HISTORICALLY AND BIASES EVERY LEVERAGE CONCLUSION

Every backtest in this repo charges a **flat 6.3%/yr** on the margin debit across the entire
sample. That is roughly right *today* (IBKR Pro small-balance = benchmark + ~1.5pp, fed funds
~4.3%) and badly wrong historically.

Measured (`research/build_financing_curve.py`, DFF + 1.5pp):

| period | actual mean | flat assumption | overcharge |
|---|---|---|---|
| **2001-2025** | **3.25%** | 6.30% | **+3.05pp/yr** |
| 2001-2007 | 4.53% | 6.30% | +1.77pp/yr |
| **2009-2015** | **1.63%** | 6.30% | **+4.67pp/yr** |
| 2020-03 → 2022-02 | 1.60% | 6.30% | +4.70pp/yr |
| 2018-2025 | 3.86% | 6.30% | +2.44pp/yr |
| 2024-2025 | 6.56% | 6.30% | −0.26pp/yr |

**Consequence:** the levered arm borrows ~0.40 of NAV more than the unlevered one, so ~3pp/yr
of excess charge hands roughly **1.2pp/yr of CAGR to the unlevered arm for free**. EXP-005
measured 1.49× leverage costing −0.044 Sharpe over 26 years *under the flat rate*; that number
cannot be trusted until re-run. → EXP-010.

**Fixed by** `financing_curve: True` in `LiveMirrorBacktester` (DFF is published same-day, so
charging financing at date t using DFF at date t is not look-ahead). Default remains the flat
rate so every previously logged number stays reproducible.

**Note the direction.** This one biases results *conservatively* for the deployed config (which
IS levered) — the strategy's true historical returns were higher than reported. It is still a
defect, because it silently distorts every leverage *comparison*.

### A9. 🔴 MY RESEARCH BASELINE OMITS THE CREDIT GATE THAT THE LIVE ENGINE RUNS

`evalkit.DEPLOYED` has no `credit_pct` key, so every A/B in this program (cycles 1-19) compared
challengers against a baseline **without** the HY-OAS credit gate. The live engine **does** run
it: `ibkr_engine.compute_credit_derisk()` reads `credit_gate.gate_status()` and halves the
gross-leverage target while HY-OAS sits at or above its p95 expanding percentile
(`credit_gate.PCT = 0.95`, `DERISK = 0.5`, deployed 2026-07-18).

**Why this matters, concretely.** Per `UNIVERSE_MANIFEST.md`, the gate is worth roughly
+0.5pp CAGR / +0.02 Sharpe / **+7.7pp MaxDD** on the 26yr. So:
- the 26yr baseline drawdown I have been quoting (−64.1%) should be nearer **−58%** with the
  gate on;
- **EXP-017/EXP-018's headline is the one at risk.** Its central claim is that removing the vol
  overlay and running constant leverage improves drawdown 12/12 on both horizons. The credit
  gate ALSO buys crisis drawdown protection, so a meaningful part of what constant-lower-leverage
  appears to add may already be supplied by a control the real system has and my baseline lacks.
  The two could be substitutes.

**This is the same class of error as the `clear_deployed()` footgun** (section C): the research
baseline silently diverging from the deployed configuration. It is exactly the failure this
program was set up to catch, and I introduced it myself by building `DEPLOYED` from
`main_production_backtest`'s config defaults rather than from the live engine.

**Not yet corrected, deliberately.** Changing `DEPLOYED` mid-program would silently invalidate
every logged comparison. Plan: measure the gate's effect on the baseline, then re-run the two
audited candidates against a gate-ON baseline, and only then decide whether to switch the
canonical baseline. Until that is done, **treat the EXP-017/018 drawdown claim as provisional**,
and note that EXP-014 (tranching) is far less exposed — tranching is a variance-reduction result
that does not compete with the gate for the same job.

### A8. ⚠️ TWO DEFECTS IN MY OWN MEASUREMENT TOOLING (found in EXP-017, both mine)

**A8a — `event_concentration` is meaningless when total excess ≈ 0.**
The metric is `top-5-day excess / TOTAL excess`. When the two arms are nearly identical the
denominator approaches zero and the ratio explodes: EXP-017 printed **−1525%, +195%, +256%** for
arms whose true excess was a rounding error. Those numbers look alarming and mean nothing.

*Fix:* only report the share when `|total excess|` is materially non-zero (say > 2% cumulative
log-return over the horizon); otherwise print `n/a — excess ≈ 0`. A near-zero excess is not
"concentrated", it is *absent*, and the two are opposite conclusions.

*Consequence:* the concentration figures quoted for EXP-014 (11-28%) are valid — that arm has a
large, clearly non-zero excess. The EXP-017 figures are not, and must not be read.

**A8b — cost sensitivity must compare arms at the SAME cost multiplier.**
EXP-017 compared "candidate at cost ×2" against "deployed at cost ×1" and reported dSharpe
−0.014, which reads as "the candidate dies at 2× cost". It shows nothing of the kind — it
measures the cost increase itself, which hits *both* arms. The correct test is
`candidate@×2 − deployed@×2`.

*Fix:* pair every cost multiplier against the baseline at that same multiplier. EXP-014 did this
correctly (its K=1 baseline was re-run at each cost level); EXP-017 did not.

### A7. 🔴 THE 8yr WINDOW CANNOT EVALUATE THE DEPLOYED PARAMETERS — IT IS THE SAMPLE THEY WERE CHOSEN FROM

Measured across three independent 12-start sweeps on the 8yr (2018-2025):

| parameter | swept | 8yr optimum |
|---|---|---|
| `top_n` (EXP-012) | 3 / **5** / 8 / 12 / 20 / 30 | Sharpe 0.725 / **0.793** / 0.727 / 0.721 / 0.666 / 0.599 → **5 = deployed** |
| `trailing_stop` (EXP-009) | 30% / **40%** / 50% / none | Sharpe 0.763 / **0.793** / 0.720 / 0.730 → **40% = deployed** |
| `rebal_days` (EXP-007 control) | 5 / **20** | 0.785 / **0.793** → 20 ≈ deployed |

**Every deployed parameter sits at or near a local maximum of the 8yr window.** That is the
signature of in-sample optimisation. It does not prove the parameters are wrong — but it does
mean **the 8yr backtest has no power to evaluate them**, because it is the sample they were
selected from. Reporting "we swept it and the deployed value won" on the 8yr is circular.

**Rule:** parameter questions are decided on the **26yr** horizon only. The 8yr may be used to
check that a 26yr-chosen value is not catastrophic there, never to choose.

### A6. ⚠️ CROSS-SECTIONAL CONFOUNDING — a trap I fell into and nearly reported

EXP-002 compared forward returns of names with earnings inside the window against those without,
pooled across all dates, and got **t = +20.2**. The share of names reporting swings from **2% to
85% across the calendar**, so the "in" group was dominated by earnings-season dates and the "out"
group by non-earnings dates. Forward 20d return is mostly market beta, so the test had silently
become *"were earnings-season months good months?"*

**Corrected by comparing only names observed on the SAME date** (per-date difference of means,
then a t-stat across dates): **t = +20.2 → t = +0.73.** The entire effect was the confound.

**Standing rule:** any claim of the form "names with property X behave differently" must be
measured **within cross-section**. If the prevalence of X varies over time — and it almost always
does — the pooled comparison measures the calendar, not the property.

---

## B. LIVE-vs-BACKTEST DIVERGENCES (found 2026-07/08, all fixed)

These are the class that backtests structurally cannot catch, because they only exist in the
live code path. Every one shipped undetected for weeks-to-months.

| # | bug | impact | fix |
|---|---|---|---|
| B1 | **Momentum + lowvol sleeves selected from SP500 only** (503 names) while every validated backtest used SP1500. 65% of the book drawn from 1/3 of the validated universe. Undetected since launch. | **−15.7pp CAGR / −0.38 Sharpe** on a clean-PIT 8yr A/B | `c8a097a` |
| B2 | `gross_margin` **unbounded** — VIR reported gm = 4,561 (vs worst-in-backtest ~12), z-scoring collapsed, live served **15 names instead of 25** | book size halved on affected days | `74d57cb` `_sane_gross_margin`, `[-1,1]` bound |
| B3 | **Partial-session bug** — an incomplete current-day bar entered the price matrix, so `rolling(200)` went NaN and SNDK (4× top momentum score) was silently dropped | silent single-name drops | `_is_partial_session()`, single source of truth for "today" |
| B4 | rev-surprise ×1.10 / beat-streak ×1.05 boosts **active live, zero in every validated run** | untested multipliers on the live score | zeroed |
| B5 | `roe` computed with `seqq ≤ 0` → 10 spurious value-eligible names (incl. SABR) | bad value picks | `seqq > 0` guard |
| B6 | **`ibkr_engine` applied vol_scale AND the credit gate only inside the rebalance block** → up to **20 sessions late** | 26yr WF MaxDD −57.5% vs −40.0% if prompt | **NOT FIXED** — engine has no off-cadence path; and the finding failed event-concentration (see LOG) |
| B7 | market-open gate trusted a server flag that could wedge | trading outside hours | clock-first, flag-can-only-veto |
| B8 | reconnect loop could spin forever | silent disconnection | `RECONNECT_FAIL_LIMIT=5` → `os._exit(1)` for pm2 |

**Standing rule from this class:** a backtest A/B is necessary but never sufficient. Before any
strategy change is believed, instrument the LIVE path at runtime and confirm the same values
flow through it. "The constant is set" ≠ "the constant is used" — B1 and B4 both passed a
source-code read and failed a runtime probe.

---

## C. BACKTEST-ONLY ALPHA (the `clear_deployed` footgun) — structurally fixed

The backtest universe pickle populates 11 maps that are **empty in live**: `_options`,
`_price_targets`, `_fin_growth`, `_ev`, `_estimates`, `_insiders`, `_earnings_signals`,
`_revenue_surprise`, `_beat_streak`, `_short_interest_rank`, `_si_change_rank`.
10 of the 11 are read by production sleeves.

Measured: leaving short interest loaded applies **19,765 score multipliers over 2018-21 alone**
(325 × 1.10, 19,440 × 0.80) that live cannot reproduce.

Was a convention (`clear_deployed()`); **now the default** (`deployed_parity=True`,
commit `8ac11df`), and `_si_months` is zeroed too because `run()` re-applies SI per date from
that source. Forgetting is now harmless.

---

## D. UNIVERSE / DATA BUILD DEFECTS (2026-07-26 → 07-28, all fixed, universes rebuilt)

Eight defects, each found only after fixing the previous one. Full detail in
`data/wrds/UNIVERSE_MANIFEST.md`.

1. **Look-ahead membership** — WRDS `Index Constituent` flag ignored → 3,510 names at 2001-01-02
   instead of 1,500 (2.3×). The backtest could hold tomorrow's index members today.
2. **Survivorship via suffixed tickers** — WRDS suffixes delisted names (`AAMRQ-201312`);
   **59.1% of symbols carry a suffix** and none matched CRSP → every delisted name dropped.
3. **Bankruptcy Q-ticker, price side** (`SIVBQ` vs CRSP `SIVB`).
4. **Bankruptcy Q-ticker, membership side** — 91 names had prices `get_sp1500()` never returned.
5. **Ticker-keyed prices spliced two companies** — 940/3,642 series (25.8%), e.g. AA
   Alcoa→Arconic. Now PERMNO-keyed.
6. **Delisting losses deleted** — CRSP leaves `DlyRet` NaN on the final row; the loop read it as
   0%. SVB's $39.37→$0.40 vanished; a held bankruptcy realised −85% instead of −99.9%.
7. **Fundamentals ticker-joined** — roe coverage 66%(2001)→88%(2025); the value sleeve is 35% of
   the book, so early value picks came from a survivor-skewed subset. Now joined on `LPERMNO`.
8. **Zero prices** (introduced by fix #6) — CRSP writes `DlyPrc = 0` for "no valid price";
   back-adjustment anchored on a 0 zeroed whole columns (666,293 cells).

**Cost of these:** 26yr CAGR was overstated by **10.4pp** and Sharpe by **0.30**; 8yr CAGR by
3.5pp and drawdown understated by 6.3pp. **Every figure in this repo predating 2026-07-28 is
suspect.**

**Residual known gap:** 399/4,305 membership symbols (9.3%) never resolve to a CRSP ticker.
A name-based bridge was tested and **rejected** — it matched LEHMQ→Lehigh Valley RR and
WAMUQ→Wampler Longacre, wrong 3 of 4 times. Clean fix is a WRDS re-download with PERMNO/GVKEY
(~Sept 2026).

---

## E. RESEARCH-HARNESS BUGS THAT PRODUCED CONVINCING FAKE RESULTS

Every one of these ran without error and produced a plausible equity curve.

| # | bug | how it was caught | what it invalidated |
|---|---|---|---|
| E1 | **`live_sizing` renormalised `vol_scale` out of existence** — weights were scaled by vol_scale then renormalised to sum 1, dividing the overlay back out. Every live-sizing arm ran unscaled. | *Every* policy reported `avgGross ≈ 1.4900` regardless of vol | Killed the "live over-invests 1.33×, real MaxDD −53.8%/−76.0%" conclusion. Actual: **1.03–1.06×, −39.9%/−65.0%** |
| E2 | **`_rv()` rejected windows < 20** → short-window arms silently ran *unscaled* while reporting deltas of exactly `0.000` | deltas of exactly zero are a bug report | asymmetric-vol arms |
| E3 | the fix to E2 **over-corrected** (demanded 30 obs for the default 40-day window), silently moving the baseline +28.3410% → +28.0694% | parity check against the known baseline | absolute CAGRs in DYN2–DYN8 off by ~0.27pp (deltas unaffected) |
| E4 | **variable-cadence counter off-by-one** — `_dsr` only incremented on non-rebalance days, so it fired on day 21 not day 20 | +5.71% CAGR appeared in a parity check that should have been 0.00% | all variable-cadence arms |
| E5 | **SI runtime probe measured nothing** — reported "0 strategy1 calls" because `livemirror` imports sleeves *by name*, so patching `M.strategy1...` never intercepted | the answer was suspiciously round | the probe itself; fixed by patching all namespaces + hard-fail if none patched |
| E6 | **partial-session guard produced an EMPTY BOOK** — the first cut trimmed the price matrix while `build_signals_v9` still took `today` from SPY bars independently | efficacy test before restart | caught pre-deploy |

**Pattern:** in 4 of 6 cases the tell was a number that was *too clean* — exactly 0.000, exactly
1.4900, exactly 0. **Suspicious roundness is the highest-yield bug detector in this codebase.**

---

## F. STATISTICAL INFLATION MECHANISMS FOUND IN THIS REPO'S OWN CONCLUSIONS

| # | mechanism | evidence |
|---|---|---|
| F1 | **The documented "~0.6pp/start noise floor" understated reality by ~4×.** True 8yr per-start σ = **7.07pp** (range +15.03%…+34.55% on entry month alone); a *delta* carries σ ≈ 2.30pp. | `threadCANON_confidence.py`, `threadCORE5` |
| F2 | **A sensitivity plateau measured on the same few starts inherits their bias.** CORE4's "10 of 10 perturbations beat deployed" looked airtight and was worthless — all ten shared the same 4 biased starts. | `threadCORE4/5` |
| F3 | **Start-consistency and walk-forward share a blind spot.** 24 starts on a 26yr horizon are 24 views of *one* 2008, not 24 samples of "does this help in a crisis". WF re-selects annually but still only ever sees one 2008. | `threadCONC` |
| F4 | **A finding passed 23/24 start-consistency, walk-forward OOS, a sensitivity plateau and all three sub-periods — and 99.6% of its 26yr excess came from 5 days.** It helped 2008 (+10.9pp) and 2020 (+7.3pp) and hurt 2001 (−10.2pp) and 2022 (−8.4pp): two wins and two losses on the one thing it exists for. | `threadCONC` |
| F5 | **A 3-start A/B reported the WRONG SIGN.** The gross-margin bound was logged as −1.50pp on the 8yr; over 12 starts it is **+1.17pp (8/12 positive)**. | `threadGM` vs 12-start rerun |
| F6 | **Many "dead" verdicts in this repo were reached on 3–4 starts AND on the pre-audit poisoned universes.** They are **not safely dead.** | — |

---

## G. OPEN / UNRESOLVED

- **A2** missing-price → `entry_px` fallback: conservative, deliberately not fixed.
- **A3** Sharpe excludes rf: deliberately not fixed mid-program.
- **B6** off-cadence gate application: real defect, but the fix failed event-concentration.
  Not deployed. `ibkr_engine` still has no off-cadence path.
- **D-residual** 9.3% membership join gap; blocks EDGAR go-forward reconciliation too.
- **Live over-invests vs backtest** — magnitude now measured at 1.03–1.06×, was believed 1.33×.
- **cap-then-renormalize defeats the position cap** — the 10% cap is applied, then gross is
  renormalised to 1.0, which can push a name back above the cap. Documented in
  `multi_strategy_engine` docstring; not fixed; effect not yet measured.
- **VIX stale-but-inert** in the live path.
