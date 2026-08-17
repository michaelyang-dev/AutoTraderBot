# BASELINE — current audited champion

**Last updated 2026-08-14.** Update ONLY after a result clears the full audit gate in `LOG.md`.

## 🔴 2026-08-14 — `DEPLOYED` OMITS THE CREDIT GATE. Measured, corrected, RESOLVED (BUGS A9).

`evalkit.DEPLOYED` has no `credit_pct`, but the live `ibkr_engine` halves gross leverage while
HY-OAS ≥ its p95 expanding percentile (`credit_gate.PCT=0.95`, `DERISK=0.5`, live since
2026-07-18). Cycles 1-19 therefore compared everything against a baseline weaker than the real
system. **Measured (EXP-022, 26yr, 12 starts, live_sizing, real financing): the gate is worth
+0.61pp CAGR / +0.021 Sharpe / +8.54pp MaxDD, 12/12 on all three.**

**HONEST 26yr baseline (gate ON, real financing, live sizing): +12.72% / 0.549 / −55.9%**
(worst −64.2%). NOT the +11.26% / 0.509 / −64.8% quoted through cycles 1-19.

`DEPLOYED` is deliberately left unchanged so the 19 prior cycles stay reproducible. **From here
on pass `credit_pct: 0.95, credit_derisk: 0.5` explicitly**, and quote the gate-ON figures.

**The risk I flagged did not materialise.** I expected the gate and constant-lower-leverage to be
substitutes, shrinking the EXP-017/018 finding. Measured substitution is **−1.86pp on MaxDD** —
they are **complements**. Removing the overlay helps *more* with the gate on (+4.11pp of drawdown
vs +2.25pp without).

## Champion config (= what is deployed live, v12)

```python
{"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
 "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
 "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
 "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
 "financing_rate": 0.063}
```

Long-only US equity, SP1500 point-in-time membership. Three sleeves blended by market breadth:
momentum (12-1 skip-month, top-5, SMA200 filter), value, low-vol-quality. 20-session rebalance,
40% per-name trailing stop, 10%-of-book position cap, inverse-vol gross scaling to a 15%×1.49
target, 1.49× closed-loop leverage financed at 6.3%, integer shares, credit gate (HY-OAS ≥ p95
expanding → halve leverage) applied at rebalance.

Research harness: `research/livemirror_backtest.py` (`LiveMirrorBacktester`), which is the only
harness that models integer shares, margin debit, financing and the credit gate jointly.
`deployed_parity=True` is the default — backtest-only inputs (short interest, price targets,
analyst estimates, EV, earnings-surprise maps) are zeroed so the backtest cannot use data live
does not have.

## Audited metrics — quote these, with the error bars

Measured over **12 monthly starts** per horizon (`research/threadCANON_confidence.py`).
Never quote a point estimate from fewer than 12 starts.

| horizon | CAGR | σ per start | 95% CI on mean | Sharpe | MaxDD |
|---|---|---|---|---|---|
| **8yr 2018-2025** | **+22.75%** | 7.07pp | **±4.00pp** | **0.79 ±0.09** | −38.2% |
| **26yr 2001-2025** | **+11.26%** | 2.75pp | **±1.56pp** | **0.51 ±0.05** | −64.8% |

8yr range across entry month alone: **+15.03% … +34.55%**. Prefer the 26yr number when the two
horizons disagree — it is ~2.5× tighter and spans dot-com + GFC + COVID + 2022.

### 🔴 CORRECTION 2026-08-14 (b) — the strategy BEATS passive over 26yr; cycle 3 was wrong

Cycle 3 (EXP-004) concluded the strategy was "at parity with passive EW SP1500 and behind on
risk". That used a baseline missing **both** the credit gate and honest financing. Corrected:

| | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| passive EW SP1500, quarterly buy&hold, **cost-free** | +11.17% | **0.590** | −58.5% |
| **deployed as actually run** | **+12.72%** | 0.549 | **−55.9%** |
| deployed minus the vol overlay (const 1.10×) | **+12.93%** | 0.584 | **−51.8%** |
| const 1.00×, no overlay | +12.27% | **0.592** | **−48.1%** |

Deployed beats passive by **+1.55pp CAGR and +2.6pp drawdown**, losing 0.04 of Sharpe — against a
benchmark paying zero costs and holding 1,500 names. Removing the overlay beats it outright.

### 🔴 CORRECTION 2026-08-14 (a) — the 26yr numbers above UNDERSTATE the strategy

Every figure in this repo charges a **flat 6.3%/yr** on the margin debit. Actual broker financing
(DFF + 1.5pp = IBKR Pro small-balance tier) averaged **3.25%** over 2001-2025 and **1.63%** over
2009-2015 (BUGS A5). Re-measured with the real curve (`financing_curve: True`, EXP-010, 12 starts):

| 26yr, 1.49× | flat 6.3% | **REAL DFF+1.5pp** | correction |
|---|---|---|---|
| CAGR | +11.26% | **+11.97%** | **+0.71pp** |
| Sharpe | 0.509 | **0.531** | +0.021 |
| MaxDD | −64.2% | −64.1% | ~0 |

The flat rate is retained as the DEFAULT so every previously logged comparison stays
reproducible, but **+11.97% / 0.531 is the honest 26yr figure.** The 8yr correction is smaller
(actual mean 3.86% vs 6.30%) and has not yet been re-measured.

### Leverage is a preference, not a free parameter (EXP-010, 26yr, REAL financing, 12 starts)

| leverage | CAGR | Sharpe | Sortino | MaxDD | worst | avgGross |
|---|---|---|---|---|---|---|
| 1.00× | +9.37% | **0.553** | **0.776** | **−48.2%** | −54.8% | 0.797 |
| 1.25× | +10.95% | 0.544 | 0.763 | −56.8% | −63.8% | 1.000 |
| **1.49× (deployed)** | **+11.97%** | 0.531 | 0.743 | −64.1% | −71.0% | 1.194 |
| 1.75× | +12.67% | 0.517 | 0.722 | −70.8% | −77.2% | 1.404 |

Paired vs 1.00×: 1.49× buys **+2.60pp CAGR** and costs **−0.022 Sharpe (0/12 starts) and
−15.87pp MaxDD**. Monotone in leverage on every axis. This is a genuine risk/return trade the
owner must choose, not a bug — but it is not a *free* parameter, and "1.49× ≈ optimal" (repo
canon, established pre-audit on few starts) is **not supported**: Sharpe and Sortino are
maximised at the LOWEST leverage tested, on 12/12 starts.

Sharpe is computed **without subtracting the risk-free rate** (`dr.mean()/dr.std()*√252`).
It is therefore overstated by roughly `rf/vol` ≈ 0.10–0.15 versus a textbook Sharpe. This is
consistent across all arms, so *deltas* are unaffected; absolute levels are not comparable to
published Sharpes.

## Falsification status of the baseline itself

See `research/AUDIT02_falsification.py` and `BUGS.md`. Summary (8yr, 4 starts):

| test | result |
|---|---|
| cost ×2 | +20.98% / 0.76 (−1.87pp CAGR) |
| cost ×3 | +19.16% / 0.71 (−3.69pp CAGR) |
| cost ×5 | see `_audit02_8yr.out` |
| shift +1 bar | see `_audit02_8yr.out` |
| shuffle (random picks) | see `_audit02_8yr.out` |
| drop best 5/10/20 days | see `_audit02_8yr.out` |

---

## ✅ AUDITED CANDIDATE — REBALANCE-PHASE TRANCHING (not deployed)

**First result in this program to pass the full audit gate.** `research/EXP001*.py`,
`EXP014_tranching_audit_gate.py`.

**What it is.** Split capital into K sub-books, each running the identical strategy but
rebalancing on a different phase of the 20-session cycle (K=4 → phases 0, 5, 10, 15). No new
signal, no new data, no forecast.

**Why it works.** The rebalance date carries zero information — it is a pure nuisance parameter.
Measured with start date AND capital held constant, the choice of phase moves 8yr CAGR by
**σ = 7.92pp** (26yr: 1.94pp), and the deployed book stakes 100% of capital on one arbitrary
phase. No phase is systematically better (per-phase means span 4pp against a 2.04pp SEM), which
is the required confirmation that this is a nuisance and not something to optimise.

| | dCAGR | dSharpe | dMaxDD | worst DD | **sd ratio** (theory 1/√4 = 0.500) |
|---|---|---|---|---|---|
| **26yr, $50k** | **+0.52pp** | **+0.031** | +0.28pp | **−71.1% → −64.5%** | **0.409** |
| 8yr, $50k | +2.87pp | +0.094 | +1.04pp | −45.5% → −44.5% | 0.571 |
| 8yr, **$33k real size** | +2.57pp | +0.086 | +0.2pp | −45.2% → −45.2% | 0.560 |
| 8yr, $33k, **cost ×3** | +2.48pp | +0.081 | — | — | 0.569 |

**Quote the 26yr.** The 8yr gain is ~4× larger only because phase risk is loudest there. Of the
8yr's +2.87pp, ~+1.1pp is arithmetic (recovered variance drag: the CAGR of an average path
exceeds the average of the paths' CAGRs) and the rest is within noise.

**Audit gate — all passed:**
- parity: `rebal_phase=0` ≡ untouched path, bit-for-bit
- capital control: effect is PHASE, not book size (−0.35pp default / **+0.51pp** live sizing)
- live-sizing control: holds under the closed-loop quantity calibration the engine uses
- **real account size**: dSharpe +0.086 at $33k vs +0.094 at $50k — live is now ~$50k, the better case
- **cost ×2 / ×3**: dSharpe +0.084 / **+0.081** — almost flat, as expected mechanically
  (tranching changes *when* dollars trade, not how many)
- **event concentration: 11-28%** of excess from top-5 days (reject threshold 50%)
- both horizons positive

**Sign consistency is 8/12, and that is expected** — base is one draw from the phase
distribution, K=4 is its average, so base wins ~half the starts by construction. The
pre-registered statistic was the sd ratio and it hit its theoretical value on both horizons.

**🔴 BLOCKER — engineering, not research.** `ibkr_engine` holds ONE book. K=4 requires four
target books netted into one IBKR account with per-tranche holdings tracked. The cheap
single-book approximation (partial adjustment) was tested and **destroys the strategy**
(EXP-007: −7 to −10pp CAGR), so this cost cannot be avoided. **K=2 captures ~half the benefit
(dSharpe +0.041 at $33k) for half the complexity** and is the sensible first step.

---

## ★★ FINAL RECOMMENDATION (2026-08-17) — OPTION 2 + HARDER CREDIT GATE

`vol_scaling: False` · `tranches: 4` (5-day stride) · `credit_derisk: ≤0.20` (0.00 = the limit)
· leverage chosen by risk appetite. Gate stays at p95.

**At IDENTICAL exposure to the deployed book** (gross 1.211 vs 1.221), 26yr, 12 starts:

| | LIVE | candidate | delta |
|---|---|---|---|
| CAGR | +12.72% | **+15.11%** | **+2.39pp** |
| Sharpe | 0.549 | **0.624** | **+0.075** |
| MaxDD | −55.9% | **−49.4%** | **+6.5pp** |
| worst-start DD | −64.2% | **−53.9%** | **+10.3pp** |
| crisis-year DD | −33.1% | **−32.0%** | **+1.1pp** |
| sd CAGR | 3.05pp | **1.80pp** | −41% |

10/12 starts on Sharpe. **Better on every axis with no leverage sleight-of-hand.**

**Three things this rests on, each independently validated:**
1. **Tranching K=4** — sd CAGR falls monotonically 1.31→1.16→1.04→1.01 across K=2/4/5/10, then
   saturates. K=4 is the build; K=10 is pure operational cost.
2. **No vol overlay** — negative timing skill on both horizons (−0.024 / −0.017 Sharpe at matched
   exposure); its value was a level effect all along.
3. **Credit gate cut to ~0** — monotone improvement on every axis to the boundary of the tested
   range. NOT a tuned parameter; the finding is "go flat in credit stress".

**⚠️ Limits, stated:**
- 8yr Sharpe sign-consistency is 7-8/12 vs the ≥9/12 bar (26yr is 10/12). Marginal on the
  recent horizon.
- The harder gate costs a trivial −0.45pp of 8yr CAGR (no credit crisis in 2018-25 for it to
  earn on) while still improving 8yr drawdown.
- At ~1.00× leverage every variant is a RISK product only: drawdown better in 24-25/25 years,
  **no return edge** (the year-by-year killed all of them on return).
- The momentum tilt is validated ONLY at 1.10-1.25×. At 1.00× it FAILED the year-by-year twice.

**Blocker:** four target books netted in one IBKR account. The cheap single-book approximation
was tested and destroys the strategy (EXP-007, −7 to −10pp).

---

## ★ AUDITED CANDIDATE 3 (STRONGEST) — TRANCHING + CONSTANT LEVERAGE, NO VOL OVERLAY

**= candidates 1 and 2 combined.** `research/EXP019_stack_candidates.py`,
`EXP027_combined_audit.py`. K=4 phase sub-books · `vol_scaling: False` · `leverage: 1.00`
(or 1.10 for the CAGR-preferring version). Gate ON, honest financing, live sizing.

**They stack additively** (EXP-019 interaction: CAGR +0.024, Sharpe +0.001) — one fixes a
*timing* nuisance, the other a *sizing* nuisance. Independent problems.

| horizon | dCAGR | dSharpe | dMaxDD | **MATCHED dSharpe** | sd ratio |
|---|---|---|---|---|---|
| **26yr** | +0.08 → +0.90pp | **+0.074 (10/12)** | +7.74pp (12/12) | **+0.032 … +0.035** | 0.397 |
| **8yr** | +0.31 → +1.02pp | **+0.096 (8/12)** | +4.76pp (9/12) | **+0.076 … +0.084** | 0.555 |

Absolute (26yr, $50k): deployed +12.72% / 0.549 / −55.9% → **arm E +12.79% / 0.623 / −48.2%**
(worst −64.2% → **−51.4%**).

**Audit gate — 12 cells (2 horizons × 2 capitals × 3 cost levels), all passed:**
- **account size is immaterial**: $33k ≡ $50k to 0.1pp in all 12 cells. The live account is now **~$50k** (2026-08-15), i.e. the *easier* of the two tested — the constraint most likely to kill this does not bite at either size
- **cost:** dCAGR **improves** with cost, dSharpe flat — it trades less
- **matched exposure:** positive in all 12 cells (EXP-008 died here at −0.099)
- **stability:** sd ratio 0.397-0.399 / 0.537-0.556 across every cell

**⚠️ Misses one bar: 8yr Sharpe sign consistency is 8/12 vs the ≥9/12 rule** (26yr 10/12).
Marginal-to-strong, **not a clean pass**. Rule not relaxed.

**⚠️ The drawdown gain is a LEVEL EFFECT.** Matched dMaxDD ≈ 0 (−0.37 to +0.36pp) — it is what
any book at ~0.99× gross instead of ~1.22× would get. **Exposure-independent** are: the Sharpe
residual (+0.03 to +0.08) and the ~60% dispersion cut.

**The precise claim:** *the drawdown of running ~1.0× leverage at NO CAGR cost* (plain
de-levering costs −2.60pp, EXP-010), plus ~+0.03-0.08 of exposure-independent Sharpe.

**Blocker:** needs four target books netted in one IBKR account. The cheap single-book
approximation was tested and **destroys** the strategy (EXP-007, −7 to −10pp). Candidate 2 alone
is two config values and captures the sizing half.

---

## ✅ AUDITED CANDIDATE 2 — REMOVE THE INVERSE-VOL OVERLAY (not deployed)

`research/EXP015_overlay_vs_constant.py`, `EXP017_constant_vs_deployed.py`,
`EXP018_constant_cost_audit.py`.

**The change:** `vol_scaling: False`, `leverage: 1.00` or `1.10`. A config change that **removes
code** — it deletes the 40-day vol estimator, the leverage churn it causes, and the stale-vol
failure mode. No new signal, data, or live machinery.

**Why:** at MATCHED average gross the overlay has **negative** timing skill on both horizons —
it costs −0.91pp CAGR / −0.024 Sharpe (26yr) and −0.99pp / −0.017 (8yr), and buys +2.89pp of
MaxDD on the 26yr but only +0.59pp on the 8yr, where it makes worst-case drawdown 1.24pp *worse*.
It is a ~1pp-per-year option that only pays in a genuine credit crisis.

**Paired per-start, 12 starts, REAL time-varying financing (which favours the deployed arm):**

| | 26yr dCAGR | 26yr dSharpe | 26yr dMaxDD | 8yr dCAGR | 8yr dSharpe | 8yr dMaxDD |
|---|---|---|---|---|---|---|
| **B const 1.00×** | −0.58pp | **+0.038 (12/12)** | **+6.40pp (12/12)** | −1.78pp | +0.035 (7/12) | **+5.60pp (12/12)** |
| **C const 1.10×** | **+0.14pp** | **+0.034 (12/12)** | **+2.49pp (12/12)** | −0.20pp | +0.026 (8/12) | +2.15pp (8/12) |

Absolute, 26yr: deployed +11.97% / 0.531 / −64.1% → **C: +12.11% / 0.564 / −61.6%**;
**B: +11.38% / 0.569 / −57.7%**.

**Audit — passed:** cost sensitivity (C's dSharpe **grows** +0.026→+0.028→+0.030 and dCAGR
improves −0.20→−0.00→+0.19pp as cost triples, confirming the gain is a turnover reduction);
event concentration (**28%**, threshold 50%); honest financing; not a knife-edge (1.00×/1.10×/
1.20× all positive, all 12/12 on 26yr Sharpe).

**⚠️ Does NOT fully clear the ≥9/12-on-both bar.** Decisive on the 26yr (12/12 Sharpe and
drawdown), **marginal at 8/12 on the 8yr**. Never negative on either horizon. Recorded as
marginal-to-strong, not a clean pass.

**The most sign-consistent result in the program:** drawdown improves in **12/12 starts on BOTH
horizons** (+6.40pp / +5.60pp at 1.00×). EXP-004 established drawdown is the axis where this
strategy is genuinely behind passive.

**Trade-off is the owner's:** 1.00× = maximum drawdown protection at −0.6 to −1.8pp CAGR;
1.10× = CAGR-neutral with smaller protection.

---

## Bar a challenger must clear

1. **≥12 monthly starts** per horizon, judged on **sign-consistency**, not the mean.
   6/12 = coin flip = dead. Target ≥9/12 on Sharpe and ≥8/12 on CAGR.
2. **Positive on BOTH horizons** (8yr and 26yr). One-horizon results have been wrong before
   (`umd_crash`: +0.026 8yr / −0.020 26yr).
3. **Matched-exposure control** for anything that changes gross exposure — score against a
   constant-leverage curve interpolated at the variant's own realised `avg_gross`, otherwise
   any de-risking rule looks like genius purely by running less.
4. **Event-concentration test** — what share of total excess comes from the top 5/10/20 days?
   >50% from 5 days = rejected. Start-consistency and walk-forward share a blind spot: every
   start window inside a horizon contains the *same* crisis.
5. **Walk-forward OOS** with annual re-selection, selection cost fully paid.
6. **Cost sensitivity** — must survive 2× modelled cost (modelled = 10 bps round trip;
   measured live = 6.10 bps, so 2× modelled ≈ 3.3× reality).
7. **Live-implementability** — it must be reproducible by `ibkr_engine` + `signal_builder`
   with data the live box actually has. Anything needing backtest-only maps is dead on arrival.
8. **Multiple-testing deflation** — report how many cells/configs were examined. A t of −2.18
   from 6 cells is p ≈ 0.18 after Bonferroni, not 0.03 (EXP-006).
9. **Hit rate inside the selected book, not pool-wide IC** — the momentum sleeve takes the top 5
   of a ~1,100-name eligible pool. A screen with a real pool-wide effect can still be worth
   nothing because it almost never touches a name that reaches the book (EXP-006). Evaluate
   screens on how often they change an actual pick.
