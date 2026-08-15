# LOG — one entry per experiment, wins and kills alike

Format: hypothesis → change → IS/OOS metrics → audit result → verdict → *why* it failed.
Kills are logged in as much detail as wins; the failure reasons are what generate the next
hypotheses.

**Configurations tested to date: 734 (this program) + ~60 inherited (see "Inherited verdicts").**

---

## Cycle 0 — HARNESS AUDIT (no strategy change)

**Hypothesis:** before any result can be believed, the harness itself must be shown not to leak.

**What was run:** `research/AUDIT01_harness_leaks.py` (data layer),
`research/AUDIT02_falsification.py` (falsification battery), 8yr universe, 4 starts,
`LiveMirrorBacktester` with `deployed_parity=True`.

**Baseline reproduced:** +22.85% / 0.81 / −39.6% (4 starts) against the canonical 12-start
+22.75% / 0.79 / −38.2%. The harness is self-consistent.

| test | result | verdict |
|---|---|---|
| **shift +1 bar** — every signal lagged one full session (features, closes-for-signals, UMD, breadth); fills still at date-t close | **+22.51% / 0.80**, ΔCAGR **−0.35pp**, ΔSharpe **−0.010** | **PASS — graceful.** A look-ahead bug dies or flips sign here. |
| **shuffle** — every sleeve replaced by a random draw from the same PIT membership, same count, equal weight, 3 reps | **+2.53% / 0.23 / −50.8%**, ΔCAGR **−20.32pp**, ΔSharpe **−0.576** | **PASS — decisive.** Selection is doing the work; the accounting is not manufacturing return. |
| **cost ×2 / ×3 / ×5** | +20.98% / +19.16% / **+15.66%** | **PASS.** Edge survives 5× modelled cost. (Modelled = 10bps round trip; measured live = 6.10bps.) |
| **forward-IC sweep** — rank-IC of all 29 panel features vs forward 20d return, 37 quarterly cross-sections | max \|IC\| = **0.050** (`vol_60d`, negative) | **PASS.** No contamination. A leaked target shows \|IC\| > 0.3. |
| **outlier** — drop best 5/10/20 days | +14.42% / +9.33% / +1.45% | **UNINFORMATIVE as run** — see below. |
| non-positive prices / absurd ticks | 0 / 170 of 6.0M (0.0028%) | PASS |

**Correction to my own test design.** The raw "drop best N days" check is not interpretable for
a levered long-only equity book — the index itself has the same property, so the number measures
beta, not fragility. The correct form is **event-concentration of the EXCESS return versus the
baseline**, which is the test already built in `threadCONC` and which has already killed a
finding. Logged in `BUGS.md` A4 so the uninformative number is never quoted as a red flag.

**Incidental finding, promoted to IDEAS I-11:** in the forward-IC sweep the strongest feature in
the entire 29-feature panel is `vol_60d` at **−0.050** — stronger than any return feature
(`ret_252d` +0.039, `ret_20d` +0.013). The low-vol anomaly is visible in our own data and the
momentum sleeve harvests it only via a soft ×1.15 nudge.

**VERDICT: harness cleared.** Numbers produced by `LiveMirrorBacktester` under
`deployed_parity=True` can be believed to the precision of the error bars in `BASELINE.md`.
Known, accepted, documented deviations: zero-latency close-to-close fills (priced at 0.35pp),
`entry_px` mark for missing prices (conservative by 0.7pp), Sharpe excludes rf (consistent).

---

## Inherited verdicts — carried forward, and how far to trust each

These predate this program. **BUGS F6 applies to all of them:** many were reached on 3–4 starts
and/or on the pre-audit poisoned universes, and that combination has already produced a
**wrong sign** (gross-margin bound) and a **retracted win** (value weights). Treat "dead" as
"probably dead, cheap to re-check" unless the entry says otherwise.

### Believed solid (12-start standard, post-audit universes)

| result | evidence | verdict |
|---|---|---|
| gross_margin `[-1,1]` bound | 8yr 8/12, 26yr 11/12 positive | **WON — deployed** (`74d57cb`) |
| value-weight change | 6/12, range −3.91…+3.70pp | **RETRACTED — was noise** |
| off-cadence credit-gate application | 23/24 starts, WF-OOS, sensitivity plateau, all 3 sub-periods — then **99.6% of 26yr excess from 5 days**; +10.9pp in 2008 and +7.3pp in 2020, **−10.2pp in 2001 and −8.4pp in 2022** | **KILLED by event-concentration.** Two wins and two losses on the one thing it exists for. |
| levering UP (`vol_scale_cap` 1.0→1.5) | monotonically worse Sharpe (26yr −0.010→−0.026) and CAGR (10.28%→9.74%) | **DEAD — do not re-test** |
| shortening *rebalance* cadence under stress | 8yr ΔSharpe −0.45; CAGR +22% → +4% | **DEAD.** Cheap adjustments (leverage) can be prompt; expensive ones (re-picking the book) cannot. |

### Dynamic-leverage timing family — all negative on ≥1 horizon

EWMA vol · asymmetric vol windows · equity-curve trend · distance-from-peak · recovery ramp ·
vol-of-vol · market vol · curve inversion · `umd_crash`.
`umd_crash` (+0.026 8yr / **−0.020 26yr**) and `mkt_vol` (+0.019 / −0.010) would both have
**passed a single-period test and are wrong** — this is why the two-horizon bar exists.

### Alpha-source families reported exhausted (pre-date the many-start standard)

Cross-sectional rank axis · ML ranking (orth IC 0.008) · news sentiment (FNSPID + FinBERT,
IC≈0) · weather · attention/Wikipedia · short strategies (27+ rounds) · crypto (momentum =
survivorship mirage) · futures global-macro (real diversifier, decayed, marginal at $30k) ·
aging · dispersion · book-crowding · opportunity-set · residual momentum · hysteresis ·
entry/exit-timing jobs · cluster caps · cash parking · valuation scoring (regime bet: −8pp 8yr,
+2.7pp 26yr).

---

## Cycle 0b — the 26yr shuffle result, which is more important than it looks

`AUDIT02` on the 26yr universe (4 starts):

| arm | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| deployed | +10.87% | 0.50 | −64.1% |
| **random picks** from the same PIT membership, identical chassis | **+6.36%** | **0.37** | −58.8% |
| cost ×2 / ×3 / ×5 | +9.08% / +7.37% / (see out) | | |
| shift +1 bar | see `_audit02_26yr.out` | | |

**The harness still passes** — random is clearly worse, so the accounting is not manufacturing
returns. But note the size of the gap by horizon:

| horizon | selection adds (CAGR) | selection adds (Sharpe) |
|---|---|---|
| 8yr 2018-25 | **+20.3pp** | +0.576 |
| **26yr 2001-25** | **+4.5pp** | **+0.127** |

Through a full cycle, **most of the return comes from the chassis** — leverage, vol-scaling,
trailing stops, trend filters, breadth-blended weights — applied to *any* basket of SP1500
names, not from picking the right names. The 8yr window flatters the stock selection by ~4×.

Caveat before over-reading: the random arm is 3 reps × 4 starts = 12 high-variance paths, and
"selection" here bundles the sleeves' trend/quality **filters** with their **ranking**.
Separating those two is now the highest-value question in the program → new hypothesis **I-21**.

---

## Cycle 1 — EXP-001 · rebalance-phase tranching (I-01)

**Hypothesis.** The rebalance date carries zero information, yet the deployed book stakes 100%
of its capital on one arbitrary phase of the 20-session cycle. Averaging K phases should shrink
that dispersion by ~√K with no assumption that any signal works better.

**Pre-registered predictions** (written into the script before running, so they could falsify):
(1) mean CAGR ~unchanged, |Δ| < 1pp — *a large positive delta is a bug report*;
(2) σ of CAGR across starts down by ~√K; (3) Sharpe up slightly; (4) MaxDD modestly better.

**Implementation.** New `rebal_phase` config in `livemirror_backtest`. The sleeves gate
internally on `day_idx % their_own_rebal_days`, so they are handed a phase-adjusted index —
without that, a phase of 5 makes `strategy5_lowvol_quality` (rebal_days=10) return `None`
forever and the lowvol sleeve silently empties. **Parity gate: `rebal_phase=0` reproduces the
untouched path bit-for-bit (300,475.89 vs 300,475.89).**

**Result — 8yr, 12 monthly starts, $50,000 total, integer shares, costs fully charged:**

| arm | mean CAGR | **sd CAGR** | Sharpe | sd Sharpe | mean MaxDD | worst MaxDD |
|---|---|---|---|---|---|---|
| base (1 phase, $50k) | +22.75% | **7.07pp** | 0.793 | 0.164 | −39.4% | −46.2% |
| K=2 ($25k × 2) | +23.42% | **3.99pp** | 0.842 | 0.088 | −37.9% | −42.7% |
| **K=4 ($12.5k × 4)** | **+24.79%** | **3.42pp** | **0.900** | 0.080 | **−36.2%** | **−38.8%** |

| | σ ratio achieved | **theoretical 1/√K** |
|---|---|---|
| K=2 | 0.565 | 0.707 |
| K=4 | **0.484** | **0.500** |

**Prediction (2) hit its theoretical value.** That is the result.

**Direct measurement of phase risk, with start date AND capital held constant:**
cross-phase σ of 8yr CAGR = **7.92pp** (max 8.27pp). Per-phase means p0 +22.40%, p5 +24.24%,
p10 +22.05%, p15 +26.04% — a 4pp spread against a per-phase SEM of 2.04pp, i.e. **no phase is
systematically better.** That is the required confirmation that this is a pure nuisance
parameter and not something to be optimised.

**Where the mean gain comes from (honest decomposition).** Total +2.04pp = phase +2.39pp
(K4 @12.5k vs p0 @12.5k) − capital −0.35pp (12.5k vs 50k, same phase). Of the +2.39pp,
about **+1.11pp is the arithmetic of compounding** — the CAGR of an average path exceeds the
average of the paths' CAGRs whenever they differ, i.e. recovered variance drag — and the
remaining ~+1.28pp is phase 0 sitting below the phase average, which is **noise** (SEM 2.04pp).
**So: mean CAGR unchanged within noise; the structural gains are ~1.1pp of variance drag and a
halving of dispersion.** Prediction (1) is upheld once decomposed, and the raw +2.04pp headline
should not be quoted on its own.

**Sign consistency: 6/12 CAGR, 8/12 Sharpe — which by the house rule is "marginal", not
"SURVIVES".** This is *not* a rule being loosened: sign-consistency tests whether the **mean**
improves, and this hypothesis is about **variance**. The base is one draw from the phase
distribution and K=4 is its average, so base must win ~half the starts by construction — 6/12
is precisely what the hypothesis predicts. The pre-registered statistic was the σ ratio, and it
landed on 0.484 against a theoretical 0.500. Recorded explicitly so nobody later reads 6/12 as
a pass that was waved through.

**AUDIT — outstanding before this can be believed:**
- [x] parity gate (phase 0 ≡ untouched path)
- [x] no phase systematically superior (would be a fitting trap)
- [ ] **26yr confirmation** — running
- [ ] **capital control at proper pairing + `live_sizing=True`** (EXP-001b) — running.
      This matters: the default harness path floors quantities and never recovers the drag,
      while live closed-loops a multiplier over 4 passes until realised gross hits target. The
      default path therefore *overstates* the cost of running smaller books — the exact cost
      this experiment turns on.
- [ ] **at the real account size ($33k, not $50k)** — 4 tranches of $8.25k at 1.49× is ~$535
      per position, and names above ~$500/share become unbuyable. Not yet priced.
- [ ] **live implementability** — `ibkr_engine` has one book. Four phases means four target
      books netted into one account: substantial new live code, not a config change.

**Status: PRELIMINARY — UNAUDITED.** Not promoted to BASELINE.md.

---

## Cycle 2 — EXP-002 · earnings-window variance avoidance (I-07) — **KILLED**

**Hypothesis.** We deliberately zeroed every earnings-surprise boost (BUGS B4), i.e. we assert
no ability to predict the print. A holding whose `rdq` falls inside the coming 20 sessions
therefore takes extra variance at zero extra mean. Removing it should raise Sharpe.

**Premise check first, on the cross-section alone, before building any backtest arm.**
Causal flag only: next announcement *estimated* as prior `rdq` + 91 days, which is information
available at decision time. (Compustat's `rdq` is the *realised* date; using it directly is a
mild look-ahead, so it is computed alongside only to bound what the estimate loses.)

**First pass was confounded and I nearly reported it.** Raw in/out comparison gave
+1.07pp forward-return advantage at **t = +20.2** — an apparently enormous earnings premium.
The flaw: the share of names reporting swings from **2% to 85% across the calendar**, so the
"in" group is dominated by earnings-season dates and the "out" group by non-earnings dates.
Forward 20d return is mostly market beta, so the test had silently become *"were earnings-season
months good months?"* — a time-series question wearing a cross-sectional costume.

**Corrected, comparing only names observed on the SAME day:**

| | within-date vol ratio | within-date mean diff | t | dates positive |
|---|---|---|---|---|
| true `rdq` (upper bound, uses future) | 1.296 | +0.267pp | +1.98 | 56% |
| **estimated `rdq` (causal)** | **1.177** | **+0.099pp** | **+0.73** | 52% |

**t = +20.2 → t = +0.73.** The entire effect was the confound.

**Verdict: the premise is CONFIRMED but too small to act on.** A vol ratio of 1.177 is a
variance ratio of 1.386 — which is exactly what one announcement day at ~3× normal vol inside a
20-day window predicts (19v² + 9v² = 28v² vs 20v² → 1.40). The measurement is clean and the
mean effect is indistinguishable from zero, so the extra variance *is* uncompensated.

But it is idiosyncratic variance in a ~23-name book with staggered announcements: at any time
~28% of holdings are in-window, so the portfolio-level variance increase is roughly
0.28 × 0.386 × (idiosyncratic share of portfolio variance) ≈ 4%, i.e. ~2% of portfolio vol, i.e.
**≈ +0.016 Sharpe.** Against a per-start Sharpe σ of 0.164 over 12 starts (SEM 0.047), that is
**a third of one standard error — unmeasurable with the tools this program has.** Screening out
28% of the eligible pool at every rebalance would also cost real signal.

**KILLED: correct mechanism, real but ~0.016 Sharpe, below our detection floor.** Deploying an
effect we cannot measure is how unvalidated complexity enters a live system.

**What it generated:** BUGS entry on cross-sectional confounding (now A5), and the note that the
*tail* channel is untested — an earnings gap can trip the 40% trailing stop in a levered book,
which is a drawdown question rather than a variance question → **I-22**.

---

## Cycle 3 — EXP-004 · benchmark context (I-24) — **the most consequential number so far**

**Hypothesis.** Nowhere in this repo is the deployed strategy compared to a passive alternative
on matched terms. Without that, the whole program could be sophisticated beta.

**Method.** Same price matrix the backtests use, same calendar, same PIT membership dicts, same
1.49× / 6.3% financing convention. `research/EXP004_benchmark_context.py`.

**I caught and fixed an inflated benchmark mid-experiment.** The first version computed
equal-weight SP1500 as a plain `nanmean` of daily returns — which is a **daily-rebalanced** index.
Daily rebalancing across 1,500 names harvests a large volatility/rebalancing premium and is
completely unimplementable. Corrected to reconstitute monthly/quarterly and **buy-and-hold** in
between, so weights drift as they would in a real account. Effect of the correction: 8yr EW
+10.23% → **+9.63%**, 26yr +12.10% → **+11.07%**. The inflated version would have made the
strategy look strictly dominated; it is not.

### 8yr 2018-2025 — the strategy wins clearly

| | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| **STRATEGY (12-start mean)** | **+22.75%** | **0.79** | **−38.2%** |
| SPY buy & hold | +13.99% | 0.77 | −33.7% |
| SPY 1.49× financed | +16.23% | 0.67 | −47.2% |
| SPY 1.49× + our inverse-vol overlay | +14.11% | 0.66 | −35.7% |
| SP1500 EW, quarterly, buy&hold | +9.79% | 0.52 | −41.5% |

### 26yr 2001-2025 — the strategy does **not** clearly win

| | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| **STRATEGY (12-start mean)** | **+11.26% ±1.56pp** | **0.51 ±0.05** | **−64.8%** |
| SPY buy & hold | +8.69% | 0.53 | −55.2% |
| SPY 1.49× financed | +8.32% | 0.42 | −73.2% |
| **SP1500 EW, quarterly, buy&hold** | **+11.17%** | **0.59** | **−58.5%** |
| SP1500 EW quarterly 1.49× financed | +11.52% | 0.50 | −76.7% |

**Read it carefully.** Against unlevered quarterly equal-weight SP1500, the strategy is
**+0.09pp on CAGR (indistinguishable), −0.08 on Sharpe, and 6.3pp WORSE on drawdown** — while
running 1.49× leverage and paying 6.3% financing to get there.

**Caveats that cut against my own conclusion, stated because they matter:**
1. The passive figure is **one path from one start date**; the strategy figure is a 12-start
   mean with a 95% CI of ±1.56pp. Those overlap heavily. "Level with passive" is a difference
   we **cannot resolve**, not a measured underperformance.
2. The EW benchmark pays **zero transaction costs** and assumes you can hold ~1,500 names. At
   $33k that is $22/position — impossible directly, though roughly reachable via an ETF.
3. My EW construction holds a name flat when its price disappears and drops it at the next
   reconstitution, which **understates bankruptcy losses**. AUDIT01 measured terminal-bar
   returns on early-ending series at mean −3.55% (3.65% drop >50%), so the bias is real but
   modest — and it favours the benchmark, not the strategy.
4. Not risk-matched: 1.49× vs 1.0×. → **EXP-005 running.**

**Verdict: no strategy change, but the program's priorities are re-ordered.** The demonstrated
edge is a 2018-2025 phenomenon, consistent with cycle 0b (selection worth +20pp on the 8yr,
+4.5pp on the 26yr). Through a full cycle we are at parity with beta on return and behind it on
risk. **That makes drawdown the target, not CAGR** — CAGR is where we are already at parity, and
−64.8% at 1.49× is the number that makes this product hard to hold.

**Generated:** I-24 (this), I-25 (below), and a re-prioritisation of I-03/I-04/I-05 (the risk and
construction tier) above the signal tier.

### I-25 · Is 1.49× leverage actually earning its keep? · **RUNNING (EXP-005)**
26yr, levered SPY is **worse than unlevered SPY on every axis** (+8.32% vs +8.69%, 0.42 vs 0.53,
−73.2% vs −55.2%) — at 6.3% financing, leverage on beta is value-destroying through a full cycle.
The same may be true of leverage on this strategy. Repo canon says "1.49× ≈ optimal", but that was
established on pre-audit universes and few starts (BUGS F6).

---

## Cycle 4 — EXP-006 · index-addition reversal (I-06) — **VOID, not actionable**

**Hypothesis.** Index funds are forced buyers into an addition's effective date. That demand is
mechanical and temporary, so the pop should reverse. Our momentum sleeve is structurally
attracted to exactly these names — a stock that just popped on index demand scores well on 12-1
momentum for a reason carrying no future return. Excluding recent adds should be free.
**Who loses:** index funds, by construction.

**Method.** Additions read from the same PIT membership dicts the backtest trades on (a name is
"added on d" the first date it appears in a membership set — no forward knowledge of index
changes). All comparisons **within-date**, per EXP-002's lesson: additions cluster at quarterly
rebalance dates, so the calendar-confounding trap is present here too.

**Result (8yr, 420 cross-sections, ~1,503 names/date, 706 addition events):**

| added within | forward | mean diff | t | dates negative |
|---|---|---|---|---|
| **20d** | **20d** | **−0.623pp** | **−2.18** | 55% |
| 20d | 60d | −0.553pp | −1.04 | 54% |
| 60d | 20d | −0.270pp | −1.53 | 51% |
| 60d | 60d | −0.120pp | −0.36 | 55% |
| 120d | 20d | −0.146pp | −1.09 | 52% |
| 120d | 60d | −0.134pp | −0.52 | 54% |

**VOID.** The sign is right in all six cells and the effect decays monotonically in both the
lookback window and the forward horizon — which is what a real, short-lived reversal looks like.
But:

1. **I tested 6 cells and am quoting the best one.** Bonferroni on t = −2.18 gives an adjusted
   p ≈ 0.18. This is precisely the "best of N configurations without deflating for N" trap in
   the mission brief. It does not clear.
2. **55% of dates negative is a coin flip** with a slight tilt.
3. **The portfolio-level magnitude is negligible even if real.** ~6.4 flagged names out of
   ~1,503 per date. The momentum sleeve takes the top 5 of a ~1,100-name eligible pool, so a
   recent addition rarely lands in it. −0.62pp per 20d on a small fraction of a 50%-weighted
   sleeve rounds to nothing on the book.

**Not built.** No backtest arm. The mechanism is real in the literature and the data leans the
right way, but the effect is both statistically unresolved here and economically irrelevant at
our concentration. Re-testable on the 26yr, where the index effect was documented to be
stronger before ~2010 — queued, low priority.

**What it generated:** the observation that our *concentration* (top-5 from ~1,100) makes almost
any broad cross-sectional screen irrelevant at the book level. A screen only matters if it hits
names that actually reach the top 5. Future screen ideas must be evaluated on **hit rate inside
the selected book**, not on pool-wide IC → recorded as a standing rule in the audit checklist.

---

## Cycle 5 — EXP-007 · partial-adjustment rebalancing (I-23) — **KILLED, decisively**

**Hypothesis.** EXP-001 showed rebalance phase is worth 7.92pp of uncompensated 8yr CAGR
dispersion, and that averaging 4 phases removes about half. Tranching needs 4 target books
netted inside one IBKR account — substantial new live code. The cheap approximation: keep ONE
book, rebalance every 5 or 10 sessions, move only a fraction of the way to each new target.

**Parity gate passed** (`move_frac=1.0` ≡ untouched path, bit-for-bit, 300,475.89).

**Result (8yr, first 4 of 12 starts — the effect sizes are far outside the noise floor):**

| start | base 20d×100% | 5d×25% | 5d×40% | 10d×50% | 10d×35% | CONTROL 5d×100% |
|---|---|---|---|---|---|---|
| 2018-01 | **+25.18%** | +5.71% | +9.03% | +8.30% | +6.94% | +15.94% |
| 2018-03 | **+21.97%** | +8.28% | +11.86% | +9.39% | +7.70% | +22.07% |
| 2018-05 | **+28.69%** | +9.47% | +13.33% | +13.73% | +11.45% | +21.59% |
| 2018-07 | **+26.05%** | +8.04% | +11.03% | +14.33% | +12.29% | +18.33% |

**−13 to −20pp of CAGR.** Per-start σ is 7.07pp; these are 2-3σ per start and consistent in
sign across every start and every arm. This is not noise.

**Why it fails, and this is the useful part.** The CONTROL arm isolates it: 5-day cadence with a
*full* move costs −4 to −7pp on its own (consistent with the inherited finding that shortening
the *rebalance* cadence is expensive). Partial adjustment then costs a further −8 to −13pp on
top. The mechanism: with `move_frac < 1` a name is never fully exited, so the book accumulates a
long tail of decaying, half-sized legacy positions. It stops being a concentrated top-5 momentum
book and drifts toward a diluted quasi-index — and the returns drift with it, toward the
equal-weight SP1500 numbers EXP-004 measured.

**The structural lesson, which is worth more than the experiment:**
**tranching and smoothing are not the same thing.** Tranching keeps each sub-book *concentrated
and fresh* and merely staggers *when* each is refreshed. Partial adjustment produces one
*diluted and stale* book. The strategy's return depends on concentration in fresh top-5 momentum
names — EXP-003 is measuring exactly how much of that is ranking versus filtering.

⇒ If the EXP-001 phase-diversification benefit is to be captured at all, it has to be **genuine
tranching** (K independent sub-books), and the live engineering cost cannot be dodged this way.

**Generated:** I-26 — since dilution moves the book toward the index, the *reverse* is testable:
does INCREASING concentration (top-3 instead of top-5) improve the 26yr, where we are currently
at parity with passive? The gradient measured here says return rises with concentration; the
open question is what it does to drawdown, which is the axis EXP-004 says we actually need.

---

## Cycle 6 — EXP-001b · the capital control EXP-001 was missing — **phase effect CONFIRMED**

EXP-001 compared K tranches of $12,500 against one book of $50,000, so it confounded PHASE
(4 rebalance phases vs 1) with CAPITAL ($12.5k books vs a $50k book). At $12.5k, integer-share
truncation drops more target positions, which is a different portfolio, not the same portfolio on
a different day. EXP-001b runs both, paired per start, under **both** sizing regimes.

The live-sizing arm matters more here than anywhere else: `ibkr_engine._calibrate_quantities`
closed-loops a multiplier over 4 passes until realised gross *after* whole-share truncation hits
target, i.e. it **recovers** rounding drag. The default harness path does not, so it *overstates*
the cost of running smaller books — the exact cost this experiment turns on.

**8yr, 12 starts, $50,000 total, K=4 phases [0,5,10,15]:**

| arm | mean CAGR | **sd CAGR** | Sharpe | sd Sharpe | mean MaxDD | worst MaxDD |
|---|---|---|---|---|---|---|
| base $50k | +22.75% | 7.07pp | 0.793 | 0.164 | −39.4% | −46.2% |
| base $12.5k | +22.40% | 7.40pp | 0.805 | 0.180 | −37.8% | −43.9% |
| **K4 $12.5k×4** | **+24.79%** | **3.42pp** | **0.900** | 0.080 | **−36.2%** | **−38.8%** |
| base $50k live-sizing | +23.40% | 6.85pp | 0.787 | 0.150 | −40.8% | −45.5% |
| base $12.5k live-sizing | +23.91% | 7.05pp | 0.797 | 0.152 | −40.5% | −46.0% |
| **K4 live-sizing** | **+26.27%** | **3.91pp** | **0.881** | 0.087 | −39.7% | −44.5% |

**Decomposition (paired per start):**

| | dCAGR | dSharpe | dMaxDD | sign (Sharpe) | **sd ratio** |
|---|---|---|---|---|---|
| capital (12.5k − 50k) | −0.35pp | +0.012 | +1.54pp | 7/12 | 1.047 |
| **PHASE (K4 − 12.5k)** | **+2.39pp** | **+0.095** | +1.63pp | 7/12 | **0.462** |
| total (K4 − 50k) | +2.04pp | +0.107 | +3.18pp | 8/12 | 0.484 |
| [live] capital | +0.51pp | +0.009 | +0.23pp | 10/12 | 1.030 |
| **[live] PHASE** | **+2.36pp** | **+0.084** | +0.81pp | 8/12 | **0.554** |
| [live] total | +2.87pp | +0.094 | +1.04pp | 8/12 | 0.571 |

**My capital-confound worry is refuted.** The capital effect is −0.35pp under default sizing and
**+0.51pp** under live sizing — negligible, and if anything favourable, because the closed loop
recovers the rounding drag. Essentially the entire effect is PHASE, under both regimes.

**σ ratio 0.462 / 0.554 against a theoretical 0.500.** The pre-registered statistic lands on its
predicted value under both sizing models.

**Phase risk measured with start AND capital held constant: σ = 7.92pp** (max 8.27pp), with
per-phase means p0 +22.40%, p5 +24.24%, p10 +22.05%, p15 +26.04% — a 4pp spread against a
per-phase SEM of 2.04pp, i.e. **no phase is systematically better.** That is the required
confirmation that this is a nuisance parameter and not something to be optimised.

**AUDIT STATUS — still PRELIMINARY. Outstanding:**
- [x] parity gate (phase 0 ≡ untouched, bit-for-bit)
- [x] no phase systematically superior
- [x] **capital control** — effect is phase, not book size
- [x] **live-sizing control** — holds under the closed-loop quantity calibration live uses
- [ ] **26yr confirmation** — running; first 2 starts mixed (−0.90pp, +0.97pp)
- [ ] **real account size ($33k)** — 4 tranches of $8.25k at 1.49× ≈ $535/position; names above
      ~$500/share become unbuyable. Not yet priced.
- [ ] **live implementability** — `ibkr_engine` has one book. EXP-007 proved the cheap
      single-book approximation (partial adjustment) **destroys** the strategy, so this
      engineering cost cannot be dodged.

**Not promoted to BASELINE.md.**

---

## Cycle 7 — EXP-003 · FILTER vs RANKING (I-21) — **the ranking is everything, the filters are nothing**

**Question.** Cycle 0b showed all of "stock selection" is worth +4.5pp CAGR / +0.13 Sharpe over
26 years. But selection is two machines: cheap mechanical **filters** (`dist_sma200 > 0`, valid
12-1 momentum, bear-regime sector exclusion, the value sleeve's quality screens) and the
**ranking** that orders the survivors. Which one earns it?

**Design.** Three arms; the random ones are **sticky** (keep names still eligible, replace only
drop-outs), because a naive random re-pick turns over ~100% every rebalance and a plain
A-vs-random gap would silently price *turnover* rather than ranking. Eligible pools come from
calling the REAL sleeves with `top_n=100000`, so every filter, boost-driven exclusion and regime
rule is preserved exactly — nothing is reimplemented.

Pool sizes @2013-01-02: members 1,494 · mom-eligible **1,105** · lowvol-eligible 1,394 ·
value-eligible 913.

**Result — 26yr, 12 starts, 24 paths per random arm:**

| arm | n | mean CAGR | sd CAGR | Sharpe | mean MaxDD |
|---|---|---|---|---|---|
| A DEPLOYED (filter + rank) | 12 | **+11.26%** | 2.75pp | **0.509** | −64.2% |
| B FILTER only (sticky random inside pool) | 24 | +8.13% | 1.79pp | 0.444 | −58.7% |
| C NO FILTER (sticky random from membership) | 24 | +8.19% | 1.79pp | 0.445 | −59.6% |

| decomposition | dCAGR | SEM | dSharpe | SEM |
|---|---|---|---|---|
| **RANKING (A − B)** | **+3.13pp** | 0.87 | **+0.065** | 0.027 |
| **FILTER (B − C)** | **−0.06pp** | 0.52 | **−0.001** | 0.020 |
| total selection (A − C) | +3.08pp | 0.87 | +0.064 | 0.027 |

**102% of the selection edge is the RANKING. The filters contribute −2%, i.e. nothing.**
Ranking: t ≈ 3.6 on CAGR, 2.4 on Sharpe — solid. Filters: indistinguishable from zero on both.

**Design validation:** the sticky-random arms return +8.13% versus AUDIT02's non-sticky shuffle
at +6.36%. Stickiness is worth +1.8pp — confirming turnover *was* confounding the naive shuffle
and that the control change was necessary, not cosmetic.

**This inverts my working assumption.** I had expected the trend/quality screens to be doing the
work and the ranking to be rediscovering them. The opposite is true. Signal work on the composite
score is the productive direction; the screens are inert.

**But note the ceiling.** All of the ranking is worth **+0.065 Sharpe**. Doubling the quality of
the ranking would buy another +0.065. That bounds what any future signal research can deliver.

**The uncomfortable synthesis with EXP-004 and EXP-005:**

| | 26yr Sharpe |
|---|---|
| passive quarterly EW SP1500, unlevered | **0.590** |
| deployed strategy, 1.00× | 0.553 |
| deployed strategy, 1.49× | 0.509 |
| our chassis with **random** picks, 1.49× | 0.444 |

Our own chassis — 1.49× leverage, vol-scaling, 40% stops, ~23 concentrated names, real costs and
financing — turns a 0.59 passive Sharpe into 0.444 with random picks. The ranking then claws back
+0.065 to 0.509. **The concentration and leverage cost more Sharpe than the security selection
creates.** That is the central finding of this program so far.

**Generated → EXP-012 (running):** if the ranking has real information but concentration is
eating it, hold MORE ranked names. Sweep `top_n` 3→40 with the position cap scaled so it cannot
silently re-concentrate. This does **not** contradict EXP-007 (partial adjustment), which diluted
with *stale decaying* positions; this dilutes with *fresh top-ranked* ones.

---

## Cycle 8 — EXP-008 · ex-ante holdings-based vol estimate (I-03) — **KILLED**

**Hypothesis.** `vol_scale` divides the target by the trailing 40d **realised** vol of the book,
so after a rebalance rotates into five new names the estimate describes the OLD book for up to
40 sessions. Replace it with an ex-ante estimate from the *current* holdings:
`sigma_p^2 = rho*(Σ w_i sigma_i)^2 + (1−rho)*Σ (w_i sigma_i)^2`, per-name sigma from the panel's
causal `vol_60d`, rho fixed (correlation is slow, composition is fast).

Explicitly **not** a timing rule — it predicts nothing, it measures the book you actually hold.

**Result — 8yr, 12 starts, with the mandatory matched-exposure control:**

| arm | CAGR | Sharpe | MaxDD | avgGross | dCAGR | dSharpe | **matched dSharpe** |
|---|---|---|---|---|---|---|---|
| exante rho=0.35 | +14.19% | 0.738 | −29.7% | 0.699 | −8.56pp | −0.055 | **−0.092 (0/12)** |
| exante rho=0.25 | +15.71% | 0.737 | −33.7% | 0.794 | −7.03pp | −0.055 | **−0.096 (0/12)** |
| exante rho=0.45 | +12.96% | 0.728 | −27.1% | 0.631 | −9.79pp | −0.064 | **−0.099 (0/12)** |

**Dead at every rho, 0/12 on the matched-exposure residual.** The estimator systematically
over-predicts book vol (avgGross collapses 1.111 → 0.63-0.79), and after correcting for that
lower exposure it is still worse.

**Why — and this is the reusable lesson.** The 40d realised estimate is *smooth*; the ex-ante
estimate jumps every rebalance as names change, so it churns leverage for no informational gain.
More fundamentally, `DYNAMIC_LEVERAGE_FINDINGS` already established the deployed overlay has
**~no timing skill** — its value is a pure LEVEL effect. Making a skill-free estimator *more
accurate* cannot help, because there was no signal being mis-measured. I mistook an estimator
problem for what is actually a "there is nothing here to estimate" problem.

**The matched-exposure control did its job.** On raw numbers this arm shows MaxDD improving by
up to +12.29pp, which looks like a drawdown win — exactly the axis EXP-004 says we need. It is
entirely the level effect of running 0.63× gross instead of 1.11×. Without that control I would
have reported a fake drawdown improvement.

---

## Cycle 9 — EXP-005 · risk-matched leverage sweep (I-25) — **PRELIMINARY, superseded by EXP-010**

**26yr, 12 starts, flat 6.3% financing:**

| arm | CAGR | sd | Sharpe | Sortino | MaxDD | worst | avgGross |
|---|---|---|---|---|---|---|---|
| 1.00× | +9.37% | 1.78pp | **0.553** | 0.776 | **−48.2%** | −54.8% | 0.797 |
| 1.25× | +10.74% | 2.27pp | 0.537 | 0.752 | −56.8% | −63.9% | 1.000 |
| 1.49× | +11.26% | 2.75pp | 0.509 | 0.713 | −64.2% | −71.1% | 1.193 |

Paired: **1.49× vs 1.00× = +1.89pp CAGR, −0.044 Sharpe, −16.01pp MaxDD, Sharpe better at 1.00×
in 12/12 starts.** The most sign-consistent result this program has produced.

**Do not act on it yet — I found the assumption that manufactures it.** All of the above charges
a **flat 6.3%/yr** on the debit across 2001-2025. Actual (DFF + 1.5pp, IBKR Pro small-balance
tier) averaged **3.25%** — the flat rate **overcharges by 3.05pp/yr**, and by 4.67pp/yr through
2009-2015. The levered arm borrows ~0.40 of NAV more, so this hands roughly **1.2pp/yr of CAGR
to the unlevered arm for free**. Logged as BUGS A5; re-run as EXP-010 under both assumptions.

Also worth stating plainly: leverage **must** lower Sharpe at positive financing cost —
`Sharpe(L) = mu/sigma − (L−1)·rf/(L·sigma)`. Observing that is arithmetic, not research. The real
question is whether the CAGR bought is worth the drawdown paid.

---

## Cycle 10 — EXP-009 · vol-normalised trailing stop (I-05) — **KILLED, and the controls say something bigger**

**Hypothesis.** A flat 40% stop is a ~2σ annual event on a 20%-vol name and ordinary noise on a
60%-vol one, so it applies a different confidence level to every position. Replace with
`clamp(k × annualised name vol, lo, hi)`.

**Result — 8yr, 12 starts:**

| arm | CAGR | Sharpe | Sortino | MaxDD | dCAGR | dSharpe | dMaxDD | +Shrp |
|---|---|---|---|---|---|---|---|---|
| **base flat 40%** | **+22.75%** | **0.793** | **1.125** | **−39.4%** | — | — | — | — |
| vol k=1.0 | +21.51% | 0.748 | 1.029 | −40.7% | −1.24pp | −0.045 | −1.34pp | 3/12 |
| vol k=1.2 | +21.50% | 0.747 | 1.026 | −41.0% | −1.25pp | −0.046 | −1.66pp | 3/12 |
| vol k=1.5 | +21.44% | 0.745 | 1.024 | −41.1% | −1.31pp | −0.047 | −1.72pp | 3/12 |
| vol k=1.2 [.30,.70] | +21.37% | 0.742 | 1.015 | −41.6% | −1.38pp | −0.051 | −2.21pp | 3/12 |
| vol k=1.2 on vol_20d | +21.40% | 0.745 | 1.022 | −41.2% | −1.34pp | −0.048 | −1.78pp | 3/12 |
| flat 30% (control) | +20.95% | 0.763 | 1.094 | −37.6% | −1.80pp | −0.029 | +1.80pp | 3/12 |
| flat 50% (control) | +20.31% | 0.720 | 0.990 | −40.3% | −2.44pp | −0.073 | −0.95pp | 3/12 |
| **no stop (control)** | +20.85% | 0.730 | 0.999 | −43.4% | −1.90pp | −0.062 | −4.04pp | 3/12 |

**Dead on every arm, 3/12 in every case.** Vol-normalisation makes both return AND drawdown
worse — it is not a return/risk trade, it is simply worse. The mechanism argument was sound and
the data rejected it: equalising the false-positive rate across names is not what a stop is for
here. The flat stop's virtue appears to be that it is *loose enough on quiet names to never
fire*, so it functions as a pure tail cut on the volatile winners rather than as a per-name
confidence rule.

**⚠️ THE CONTROLS ARE THE REAL FINDING.** Flat 30% and flat 50% are BOTH worse than 40%, on both
CAGR and Sharpe. Combined with EXP-012's 8yr breadth sweep (top_n = 3/5/8/12/20/30 →
0.725 / **0.793** / 0.727 / 0.721 / 0.666 / 0.599) and EXP-007's cadence control (20d ≈ 5d),
**every deployed parameter sits at or near a local optimum of the 8yr window** — the window it
was originally tuned on.

That is the signature of in-sample optimisation, not of a robust configuration. It does not mean
the parameters are wrong; it means the 8yr backtest **cannot** be used to evaluate them, because
it is the sample they were chosen from. The 26yr is the only horizon with any power here, which
is why EXP-012 is now running on it. Recorded in BUGS as a standing caution.

---

## Cycle 11 — EXP-001 26yr · phase tranching CONFIRMED on the second horizon

| arm | mean CAGR | **sd CAGR** | Sharpe | sd Sharpe | mean MaxDD | **worst MaxDD** |
|---|---|---|---|---|---|---|
| base (1 phase) | +11.26% | 2.75pp | 0.509 | 0.081 | −64.2% | −71.1% |
| K=2 | +11.67% | 1.48pp | 0.531 | 0.046 | −64.6% | −67.2% |
| **K=4** | **+11.78%** | **1.13pp** | **0.540** | 0.035 | −64.0% | **−64.5%** |

| | dCAGR | dSharpe | dMaxDD | +CAGR | +Sharpe | **sd ratio** (target 1/√K) |
|---|---|---|---|---|---|---|
| K=2 | +0.41pp | +0.022 | −0.33pp | 7/12 | 8/12 | **0.539** (0.707) |
| K=4 | +0.52pp | +0.031 | +0.28pp | 8/12 | 8/12 | **0.409** (0.500) |

**Positive on both horizons, and the σ-reduction beat its theoretical target on both**
(8yr 0.484, 26yr 0.409 vs 0.500). Worst-case drawdown improves on both: 8yr −46.2% → −38.8%,
26yr **−71.1% → −64.5%**. That is a within-path tail improvement, not a cross-start statistic.

Phase risk on the 26yr is σ = 1.94pp versus 7.92pp on the 8yr — as expected, the phase lottery
averages out over a longer horizon, which is exactly why the 8yr gain (+2.04pp) is ~4× the 26yr
gain (+0.52pp). **Quote the 26yr number.** The 8yr headline is the same effect measured on a
horizon where the nuisance is loudest.

**Audit gate now running (EXP-014):** real $33k account size, cost ×1/×2/×3, and
event-concentration of the excess. The $33k test is the most likely killer — K=4 there is
$8,250/tranche, ~$535/position, and names above ~$500/share become unbuyable.

---

## Cycle 12 — EXP-013 · alternative momentum rankings (I-09/I-10) — **KILLED**

EXP-003 showed the ranking carries 100% of the selection edge, so the ranking is the only thing
worth improving. Four alternative orderings, each applied by **re-ordering the pool the real
`strategy1` already produced** — every filter, boost and regime rule preserved bit-for-bit,
nothing reimplemented. `mom_pool=4` refines the top 20; `mom_pool=100` re-ranks the whole
eligible pool.

**I found and fixed a bug in my own test design mid-experiment.** The first version let variants
*select* by the new score but inherit *weights* from the original momentum score. With a wide
pool, a name chosen by 52-week-high proximity can carry a near-zero momentum score — so it was
selected and then given ~no capital, silently collapsing the book to 3-4 names. Fixed to equal
weight (`mom_rescore_weight`, default `equal`). The fix moved `multi pool=4` from −0.148 to
−0.108 Sharpe: material, and it did not change the verdict.

**Result — 8yr, 12 starts, base +22.75% / 0.793 / −39.4%:**

| arm | CAGR | Sharpe | MaxDD | dCAGR | dSharpe | +Sharpe |
|---|---|---|---|---|---|---|
| multi (12-1/6-1/3-1 rank avg) pool=4 | +20.61% | 0.685 | −46.1% | −2.14pp | −0.108 | 3/12 |
| multi pool=100 | +15.19% | 0.579 | −48.3% | −7.55pp | −0.214 | 2/12 |
| hi52 (52w-high proximity) pool=4 | +15.38% | 0.586 | −46.0% | −7.37pp | −0.206 | 2/12 |
| hi52 pool=100 | +13.76% | 0.544 | −45.1% | −8.99pp | −0.248 | 1/12 |

**All dead, and dead by margins far larger than the total value of the ranking itself.**
EXP-003 bounds ALL existing ranking at +0.065 Sharpe; these lose 0.11-0.25. Note also that the
wider the pool (i.e. the more the variant *replaces* rather than *polishes* the deployed
ordering), the worse it gets — monotonically, for every variant. That is a strong signal that
the deployed 12-1 composite is genuinely well-specified, not merely lucky.

**Caveat honoured:** BUGS A7 says the 8yr cannot adjudicate parameters chosen on it. The
composite *scoring formula* was also developed on this window, so some of this gap is in-sample
advantage. But an in-sample edge of ~0.03-0.05 Sharpe cannot explain gaps of 0.11-0.25. The kill
stands; a 26yr rerun would only sharpen it.

**Generated:** the `multi` ensemble losing to its own single member falsifies its stated
mechanism (rank-averaging as estimator variance reduction). Momentum horizons are not
interchangeable noisy estimates of one quantity — 12-1 and 3-1 are *different signals*, and
averaging them dilutes the one that works. Recorded so the "ensembling is free variance
reduction" argument is not reused uncritically elsewhere.

---

## Cycle 13 — EXP-010 · leverage under HONEST financing — **corrects the canonical numbers**

EXP-005 measured 1.49× leverage costing −0.044 Sharpe over 26 years, 12/12 sign-consistent. I
then found the assumption that manufactures it: a **flat 6.3%/yr** financing charge applied
across 2001-2025, when the real rate (DFF + 1.5pp) averaged **3.25%** and just **1.63%** through
2009-2015 (BUGS A5). Re-run under both.

**26yr, 12 starts, REAL financing:**

| leverage | CAGR | Sharpe | Sortino | MaxDD | worst | avgGross |
|---|---|---|---|---|---|---|
| 1.00× | +9.37% | **0.553** | **0.776** | **−48.2%** | −54.8% | 0.797 |
| 1.25× | +10.95% | 0.544 | 0.763 | −56.8% | −63.8% | 1.000 |
| **1.49× deployed** | **+11.97%** | 0.531 | 0.743 | −64.1% | −71.0% | 1.194 |
| 1.75× | +12.67% | 0.517 | 0.722 | −70.8% | −77.2% | 1.404 |

**What the financing assumption alone was worth:**

| leverage | dCAGR | dSharpe |
|---|---|---|
| 1.00× | +0.00pp | +0.000 |
| 1.25× | +0.20pp | +0.008 |
| **1.49×** | **+0.71pp** | **+0.021** |
| 1.75× | +1.34pp | +0.035 |

**I over-estimated this effect when I flagged it.** I predicted ~1.2pp/yr at 1.49×, assuming the
levered arm borrows ~0.40 of NAV more than the unlevered one. It actually borrows ~0.19 — because
`avg_gross` at 1.49× is 1.194, not 1.49, once the vol overlay and cash drag are accounted for.
Real effect **+0.71pp**. The direction was right, the magnitude was 1.7× too large.
The 1.00× arm is exactly unchanged (avg_gross 0.797 < 1 → no debit → nothing to finance), which
is a clean internal sanity check that the curve is wired correctly.

**Verdict on leverage.** The Sharpe penalty **halves** under honest financing (−0.044 → −0.022)
but does not vanish, and remains **0/12** sign-consistent. Paired vs 1.00×, 1.49× buys +2.60pp
CAGR and costs −0.022 Sharpe and **−15.87pp MaxDD**, monotone on every axis and at every step.

- Repo canon "1.49× ≈ optimal" is **not supported.** Sharpe and Sortino are maximised at the
  lowest leverage tested, in 12/12 starts.
- But this is a **preference trade, not a free win**, and the mission asks for CAGR up *and*
  drawdown down *and* Sharpe up. De-levering delivers two of three and gives up 2.6pp of CAGR.
  **Not deployed. Owner's call**, now quantified.
- Leverage *must* lower Sharpe at positive financing cost — `Sharpe(L) = μ/σ − (L−1)rf/(Lσ)`.
  The research content here is the **size** of the trade, not its sign.

**BASELINE.md corrected:** honest 26yr figure is **+11.97% / 0.531 / −64.1%**, not
+11.26% / 0.509. The flat rate stays the default so prior comparisons remain reproducible.

**Effect on the passive comparison (EXP-004).** Against cost-free quarterly EW SP1500
(+11.17% / 0.59 / −58.5%), the strategy at 1.49× now **wins on CAGR (+0.80pp)** and is still
behind on Sharpe (−0.06) and drawdown (−5.6pp). The "strategy loses to the index" reading from
cycle 3 was partly an artefact of the financing assumption and should be softened to: *roughly a
wash on return, still behind on risk* — with the benchmark still enjoying zero costs and a
1,500-name book nobody could hold directly.

---

## Cycle 14 — EXP-012 · book breadth (I-26/I-17) — **hypothesis FALSIFIED, cleanly**

**Hypothesis.** EXP-003 showed the ranking carries all the selection edge; EXP-004/005 showed
concentration costs more Sharpe than the ranking creates. So holding MORE ranked names should
harvest the same edge on a better-diversified base: CAGR falls slowly, vol falls faster, Sharpe
and MaxDD improve to an interior optimum.

**Result — Sharpe by `top_n`, 12 starts, position cap scaled so it cannot silently re-concentrate:**

| top_n | 3 | **5** | 8 | 12 | 20 | 30 |
|---|---|---|---|---|---|---|
| **8yr** | 0.725 | **0.793** | 0.727 | 0.721 | 0.666 | 0.599 |
| **26yr** | 0.490 | **0.509** | 0.469 | 0.464 | 0.451 | — |

**Falsified on both horizons.** Sharpe peaks at 5 and declines monotonically in both directions.
Diversifying into lower-ranked names loses more return than it saves in variance — so the
ranking's information is **concentrated in the very top names**, not spread across the pool.

**This also partly retracts my own BUGS A7 warning.** I flagged that every deployed parameter
sitting at an 8yr optimum was the signature of in-sample fitting. For `top_n` that is now
disproven: 5 wins on the **independent 26yr horizon** too. A7 stands as a caution about *method*
(the 8yr cannot adjudicate parameters chosen on it) but `top_n=5` is genuinely well-chosen, not
an artefact. `trailing_stop=40%` has not yet had the same 26yr test and remains unverified.

**Synthesis with EXP-007.** Diluting with *stale* positions (partial adjustment) cost −7 to
−10pp; diluting with *fresh top-ranked* ones costs −1.4pp of CAGR and −0.04 Sharpe per step.
Both dilute, both hurt — the stale version far more. **Concentration is load-bearing.**

---

## Cycle 15 — EXP-016 · MARGINAL VALUE — **the first result that improves all three axes**

### THIS RESULT MAY BE INFLATED. Auditing before treating it as a finding.

**The question no prior work in this repo asks.** Everything here compares the strategy to a
benchmark as a **replacement**. That is the wrong comparison for an owner who could hold both.
The right one is **marginal**: starting from a passive index portfolio, does adding some of this
strategy improve it? A strategy can be worse standalone and still be valuable at the margin,
purely through imperfect correlation.

**Method.** Blend the strategy's daily path with a passive path, **rebalanced monthly** (drift is
not free), sweeping w = 0…1. Two passive legs: SPY, and quarterly-reconstituted buy-and-hold EW
SP1500. Strategy leg uses honest time-varying financing. 26yr, 12 starts.

**Result — EW SP1500 core (the better of the two):**

| w in strategy | CAGR | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|
| 0.0 (pure passive) | +11.35% | 0.598 | 0.787 | −58.5% |
| 0.2 | +11.84% | 0.615 | 0.823 | −58.7% |
| **0.3** | **+12.01%** | **0.615** | **0.832** | −59.2% |
| 0.4 | +12.14% | 0.611 | 0.834 | −59.7% |
| 0.5 | +12.22% | 0.602 | 0.829 | −60.3% |
| 1.0 (pure strategy) | +11.97% | 0.531 | 0.743 | −64.1% |

SPY core has the identical shape (max Sharpe 0.613 at w=0.3, plateau 0.2-0.4).
**daily corr(strategy, EW) = 0.703, corr(strategy, SPY) = 0.670.**

**A 30/70 strategy/EW blend strictly dominates the standalone strategy:** same CAGR (+12.01% vs
+11.97%), **+0.084 Sharpe**, **+0.089 Sortino**, **+4.9pp less drawdown**. It also beats the pure
passive core on all three: +0.66pp CAGR, +0.017 Sharpe, DD flat.

### Audit

**1. Is it just DE-LEVERING?** This is the serious objection. At w=0.3 the effective gross is
0.3 × 1.194 + 0.7 × 1.0 = **1.058**, *below* the strategy's own 1.194 — and Sharpe rises as
leverage falls (EXP-010). So part of the gain could be a pure level effect with no
diversification content.

Checked against EXP-010's own leverage curve (REAL financing, same 12 starts): the pure strategy
at avg_gross 1.000 has Sharpe 0.544 and at 1.194 has 0.531, so at 1.058 it interpolates to
**≈0.540**. The blend delivers **0.615**. **Diversification adds ≈ +0.075 beyond the level
effect.** The control passes decisively. *(Now automated inside the script rather than computed
by hand — re-running to produce it as auditable output.)*

**2. Is this a discovered anomaly?** No, and it should not be sold as one. It is textbook
mean-variance: blending two positively-but-imperfectly-correlated assets (corr 0.70) where one
has higher return and much higher vol. There is no new edge here. **The finding is that the
current 100% allocation ignores it** — a portfolio-construction error, not a missing signal.
That is exactly the kind of result least likely to be a backtest artefact.

**3. Is the optimum knife-edge?** No. Sharpe is 0.615 / 0.615 / 0.611 / 0.602 across
w = 0.2 / 0.3 / 0.4 / 0.5. A broad flat plateau. Any w in 0.2-0.4 captures it.

**4. Where the benchmark is FAVOURED (so the finding is a lower bound):**
   - the EW leg pays **zero transaction costs and zero expense ratio**. Real implementation
     (RSP + EW mid/small ETFs) costs ~0.20-0.40%/yr → roughly −0.25pp on the 70% leg.
   - my EW construction holds a name flat when its price vanishes, **understating bankruptcy
     losses**.
   Both make the passive leg look better than it is, so they bias *against* the blend, not for it.

**5. Statistical status.** Strategy leg = 12-start mean; passive leg = one path. Stated, not
hidden. The blend is computed per-start against that start's own passive window, so every blended
number is a real path.

**Outstanding before this could be promoted:** 8yr confirmation (running — expect a much higher
optimal w, since the strategy dominates passive on that window), and the automated
exposure-matched control.

### 🔴 CORRECTION — the 8yr result flips the headline. My first write-up above was too strong.

I logged this as "improves all three axes". **That is true on the 26yr only.** The 8yr:

| w in strategy | CAGR | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|
| 0.0 (pure passive EW) | +10.63% | 0.545 | 0.718 | −41.5% |
| 0.3 | +14.89% | 0.698 | 0.932 | −39.4% |
| 0.5 | +17.48% | 0.758 | 1.030 | −38.7% |
| 0.8 | +20.98% | 0.796 | 1.112 | **−38.4%** |
| **1.0 (pure strategy)** | **+23.05%** | **0.800** | **1.135** | −39.1% |

**On the 8yr, Sharpe is maximised at w = 1.0.** corr(strategy, EW) = 0.630.

So the two horizons disagree about the *direction*, and by my own pre-registered two-horizon
bar, **"dilute the strategy with passive" FAILS.** w = 0.3 costs −0.102 Sharpe on the 8yr. I am
not loosening the rule to let it through.

**The correct decomposition, which IS two-horizon robust:**

| claim | 8yr | 26yr | verdict |
|---|---|---|---|
| adding the STRATEGY to a passive core helps | +0.255 Sharpe at best w | +0.017 at best w | **holds on both** |
| adding PASSIVE to the strategy helps | **−0.102** at w=0.3 | +0.084 at w=0.3 | **fails — horizon-dependent** |

⇒ **The strategy has genuine marginal value over passive on both horizons.** That is the durable
finding, and it is the first thing in this program to survive the two-horizon bar on the return
axis. ⇒ **Whether to dilute the deployed book with an index core is a regime bet**, not a
structural improvement: it wins big if the next decade resembles 2001-2025 and loses if it
resembles 2018-2025.

Worth noting the 8yr's own shape: Sharpe is essentially flat from w = 0.8 to 1.0
(0.796 / 0.799 / 0.800) while MaxDD is *best* at w = 0.7-0.8 (−38.4% vs −39.1%). So even on the
favourable horizon, holding ~20% passive costs ~0.004 of Sharpe and buys 0.7pp of drawdown. A
w ≈ 0.7-0.8 is the only mix that is not clearly harmful on either horizon — but that is me
reading a compromise off two in-sample frontiers, and it should be labelled as such rather than
recommended.

**Status: PRELIMINARY. The marginal-value claim survives; the dilution claim does NOT. Nothing
promoted to BASELINE.md.**

**What it means if it holds.** The strategy's problem is not that it lacks edge — EXP-003 showed
the ranking is worth a real +3.13pp CAGR / +0.065 Sharpe. The problem is that **100% of capital
is allocated to a concentrated, levered expression of a modest edge.** Sizing it at 20-40% of a
passive core keeps the edge and discards most of the idiosyncratic risk.

---

## Cycle 16 — EXP-014 · TRANCHING AUDIT GATE — **PASSES. First fully-audited win.**

All three remaining attacks, run at `live_sizing=True` (the closed-loop quantity calibration the
engine actually uses), 8yr, 12 starts.

### At the REAL account size ($33,000) — the attack most likely to kill it

| arm | mean CAGR | sd CAGR | Sharpe | mean MaxDD | worst DD | dCAGR | dSharpe | +Shrp | sd ratio | **top-5 day share of excess** |
|---|---|---|---|---|---|---|---|---|---|---|
| K=1 cost×1 | +23.66% | 7.12pp | 0.792 | −40.4% | −45.2% | — | — | — | 1.000 | — |
| K=2 cost×1 | +24.51% | 4.36pp | 0.832 | −40.5% | −46.4% | +0.85pp | +0.041 | 6/12 | 0.613 | 13% |
| **K=4 cost×1** | **+26.23%** | **3.99pp** | **0.878** | −40.2% | −45.2% | **+2.57pp** | **+0.086** | 8/12 | **0.560** | **28%** |
| K=4 cost×2 | +24.24% | 3.94pp | 0.827 | −41.2% | −46.7% | +2.55pp | +0.084 | 8/12 | 0.558 | 27% |
| **K=4 cost×3** | +22.28% | 3.96pp | 0.777 | −42.0% | −48.3% | **+2.48pp** | **+0.081** | 8/12 | 0.569 | 20% |

**1. Real account size: PASSES.** I expected this to be the killer — K=4 at $33k is $8,250 per
tranche, ~$535/position, and names above ~$500/share become unbuyable. Effect is essentially
unchanged: dSharpe +0.086 at $33k vs +0.094 at $50k. The closed-loop sizing recovers the
rounding drag, as EXP-001b predicted.

**2. Cost sensitivity: PASSES.** ×1 → ×2 → ×3 moves dSharpe +0.086 → +0.084 → **+0.081** and
dCAGR +2.57 → +2.55 → **+2.48pp**. Almost perfectly flat, which is exactly right *mechanically*:
tranching does not change turnover per dollar, it only changes *when* each dollar trades. A
result that was secretly a turnover artefact would decay steeply here. It does not.

**3. Event concentration: PASSES.** Top-5 days contribute **11-28%** of the excess across every
cell, against a >50% reject threshold. For comparison, the off-cadence credit gate — which passed
23/24 start-consistency, walk-forward OOS, a sensitivity plateau and all three sub-periods — was
killed by this test at **99.6%**. Tranching is a process, not an event.

**4. sd ratio stable at 0.56-0.62** across every capital and cost cell.

### The honest number

| | dCAGR | dSharpe | dMaxDD | worst DD | sd ratio |
|---|---|---|---|---|---|
| 8yr, $50k | +2.87pp | +0.094 | +1.04pp | −45.5% → −44.5% | 0.571 |
| 8yr, $33k | +2.57pp | +0.086 | +0.2pp | −45.2% → −45.2% | 0.560 |
| **26yr, $50k** | **+0.52pp** | **+0.031** | **+0.28pp** | **−71.1% → −64.5%** | **0.409** |

**Quote the 26yr.** The 8yr gain is ~4× larger because phase risk is loudest there (σ 7.92pp vs
1.94pp) — it is the same effect measured on a noisier horizon, not a bigger effect. And of the
8yr's +2.87pp, roughly **+1.1pp is arithmetic** (the CAGR of an average path exceeds the average
of the paths' CAGRs — recovered variance drag) and the remainder is phase 0 sitting below the
phase average, which is noise at a 2.04pp SEM.

**The claim, stated precisely:** K=4 phase tranching reduces outcome dispersion by ~45% (sd ratio
0.41-0.57 against a theoretical 0.50), improves Sharpe by **+0.031 (26yr) to +0.086 (8yr)**,
improves worst-case drawdown by **6.6pp on the 26yr** (−71.1% → −64.5%), and costs nothing in
CAGR. It requires **no new signal, no new data, and no forecast** — it removes an uncompensated
nuisance the current implementation takes for free.

**Sign consistency is 8/12, not 11/12, and that is expected, not a failure.** Base is one draw
from the phase distribution; K=4 is its average; base must win ~half the starts by construction.
The pre-registered statistic was the sd ratio and it landed on its theoretical value. I am
recording this rather than quietly reporting 8/12 as a pass.

**REMAINING BLOCKER — engineering, not research.** `ibkr_engine` holds one book. K=4 means four
target books netted into one IBKR account, with per-tranche holdings tracked so each rebalances
on its own phase. EXP-007 proved the cheap single-book approximation (partial adjustment)
**destroys** the strategy (−7 to −10pp CAGR), so this cost cannot be dodged. **K=2 captures about
half the benefit (dSharpe +0.041 at $33k) for half the complexity** and is the sensible first
step.

**PROMOTED to BASELINE.md as an audited candidate, flagged NOT DEPLOYED.**

---

## Cycle 17 — EXP-015 · is the deployed vol overlay doing anything? (I-20) — **it is HARMFUL**

**Claim on file** (`DYNAMIC_LEVERAGE_FINDINGS`): the deployed inverse-vol overlay has "~no timing
skill" and its value is purely a LEVEL effect. If true it is replaceable by a constant, which
would remove a 40-day estimator, leverage churn, and a stale-vol failure mode from the live
engine. Testing it also tests my own reasoning in EXP-008, where I killed a better vol
*estimator* partly on the argument that "there is no timing skill to improve."

**26yr, 12 starts. A = deployed (1.49× + overlay, avg_gross 1.1932). B = constant leverage with
`vol_scaling` OFF, interpolated to the SAME avg_gross from a 6-point grid:**

| lev | avgGross | CAGR | Sharpe | Sortino | MaxDD | worst |
|---|---|---|---|---|---|---|
| 0.90 | 0.8595 | +10.43% | 0.567 | 0.806 | −53.6% | −61.3% |
| 1.00 | 0.9567 | +11.38% | **0.569** | 0.807 | −57.7% | −65.3% |
| 1.10 | 1.0531 | +11.93% | 0.558 | 0.792 | −61.7% | −69.3% |
| 1.20 | 1.1495 | +12.13% | 0.540 | 0.766 | −65.6% | −73.0% |
| 1.30 | 1.2458 | +12.23% | 0.524 | 0.743 | −69.0% | −76.3% |
| **A: deployed + overlay** | **1.1932** | **+11.26%** | **0.509** | 0.713 | −64.2% | −71.1% |

**A − B at matched exposure:**

| dCAGR | dSharpe | dSortino | dMaxDD | dWorstDD |
|---|---|---|---|---|
| **−0.91pp** | **−0.0241** | **−0.0428** | **+2.89pp** | **+3.41pp** |

**The overlay is not decorative — it is mildly harmful on return and risk-adjusted return, and it
buys real drawdown protection.** Versus simply running constant leverage at the same average
gross, it costs 0.91pp of CAGR and 0.024 of Sharpe, and buys 2.89pp of MaxDD / 3.41pp of
worst-case DD. That refines the repo's "no timing skill" claim: timing skill is *negative*, but
the overlay does reshape the drawdown, which is not nothing.

**⚠️ The stronger implication, which needs its own audit.** Constant leverage **1.10×** returns
+11.93% / 0.558 / −61.7% versus the deployed +11.26% / 0.509 / −64.2% — **better on all three
axes simultaneously** (+0.67pp CAGR, +0.049 Sharpe, +2.5pp MaxDD). Constant 1.00× is better still
on risk (+11.38% / 0.569 / −57.7%, i.e. **6.5pp less drawdown** than deployed at higher CAGR and
higher Sharpe).

Normally de-levering trades CAGR for drawdown (EXP-010). Here CAGR goes **up** as well, because
the overlay's drag exceeds what the extra leverage earns.

**Before believing that:** these grid runs use the flat 6.3% financing. Under honest financing
the higher-gross deployed arm gains more (+0.71pp CAGR / +0.021 Sharpe at avg_gross 1.19, versus
~+0.2pp / +0.008 at 1.05), which narrows but does not close the gap — deployed → ~+11.97%/0.530,
constant 1.10× → ~+12.13%/0.566. **Still better on all three.** 8yr confirmation is running; the
8yr is momentum-friendly and is where the overlay should look best, so that is the real test.

**8yr confirmation — the overlay's cost is CONSISTENT, its benefit is NOT:**

| A − B at matched exposure | dCAGR | dSharpe | dSortino | dMaxDD | dWorstDD |
|---|---|---|---|---|---|
| **26yr** | −0.91pp | −0.0241 | −0.0428 | **+2.89pp** | **+3.41pp** |
| **8yr** | −0.99pp | −0.0168 | −0.0351 | +0.59pp | **−1.24pp** |

The overlay costs **~1pp of CAGR and ~0.02 of Sharpe on both horizons** — remarkably stable. What
it buys is not stable: +2.89pp of MaxDD on the 26yr (which contains 2008), but only +0.59pp on
the 8yr, where it makes the **worst-case drawdown 1.24pp WORSE**.

**Verdict: the deployed inverse-vol overlay has negative timing skill on both horizons.** It is a
drawdown-purchasing device whose purchase only pays in a genuine credit crisis, and it charges
~1pp of CAGR every year for that option. The repo's "no timing skill" note was right in
direction and understated in magnitude.

**Status: the matched-exposure comparison is CONCLUSIVE on both horizons. The stronger claim —
that constant leverage should REPLACE the overlay — needs a paired per-start test → EXP-017.**

---

## Cycle 18 — EXP-017 · constant leverage vs deployed, PAIRED — **marginal, does NOT clear the bar**

EXP-015 compared **means**. That is not sufficient and this repo has been burned by exactly that
(the value-weight finding, retracted at 6/12). EXP-017 runs the paired per-start test.

**8yr, 12 starts, REAL financing (which FAVOURS the deployed arm — it runs more gross so gains
more from the corrected rate):**

| arm | CAGR | sdCAGR | Sharpe | Sortino | MaxDD | avgGross | dCAGR | dSharpe | +CAGR | **+Sharpe** | +DD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A deployed 1.49×+overlay | +23.03% | 7.08pp | 0.800 | 1.135 | −39.2% | 1.112 | — | — | — | — | — |
| B const 1.00× | +21.25% | 5.27pp | **0.835** | **1.198** | **−33.6%** | 0.930 | −1.78pp | +0.035 | 2/12 | 7/12 | **12/12** |
| **C const 1.10×** | +22.83% | 5.89pp | 0.826 | 1.184 | −37.0% | 1.027 | −0.20pp | **+0.026** | 8/12 | **8/12** | 8/12 |
| D const 1.20× | +24.20% | 6.56pp | 0.815 | 1.168 | −40.2% | 1.124 | +1.16pp | +0.015 | 9/12 | 8/12 | 5/12 |

**C reaches 8/12 on Sharpe. My pre-registered bar was ≥9/12. It does not clear.**
By evalkit's own rule (`ns≥9 and nc≥8`) this is **marginal**, not SURVIVES. I am not lowering the
bar. The EXP-015 "strict domination on all three axes" was a comparison of *means*, and the
paired test does not support promoting it.

**What IS strongly sign-consistent:** B (constant 1.00×, no overlay) improves drawdown in
**12/12 starts**, by +5.60pp on average, for −1.78pp of CAGR. That is the same CAGR-for-drawdown
trade EXP-010 already quantified for leverage — a real, robust, owner's-preference option, not a
free win.

**Two defects in my own tooling, found here and fixed (BUGS A8):**
- `event_concentration` divides by total excess; when two arms are nearly identical that
  denominator → 0 and the metric printed **−1525%, +333%, +256%** for arms whose excess was a
  rounding error. Now returns NaN below a 2% cumulative-excess threshold. **The EXP-014
  concentration figures (11-28%) remain valid** — that arm has a large, clearly non-zero excess.
  The EXP-017 ones must not be read.
- The "C cost ×2" row compared the candidate at 2× cost against the **deployed at 1× cost**,
  which measures the cost increase, not the difference between arms. Meaningless as written;
  needs `candidate@×2 − deployed@×2`. EXP-014 did this correctly, EXP-017 did not.

**26yr still running.** Verdict will be recorded there; the 8yr already establishes that this is
at best marginal rather than the domination EXP-015's means suggested.

---

## Cycle 19 — EXP-017/018 · REMOVING THE VOL OVERLAY — **second audited candidate**

**The change:** delete the inverse-vol overlay and run constant leverage. `vol_scaling: False`,
`leverage: 1.00` (arm B) or `1.10` (arm C). This is a **config change that removes code** — no
new signal, no new data, no new live machinery. It deletes the 40-day estimator, the leverage
churn it causes, and the stale-vol failure mode.

**Why it should work (EXP-015, matched exposure, both horizons):** at the *same* average gross,
the overlay costs **−0.91pp CAGR / −0.024 Sharpe (26yr)** and **−0.99pp / −0.017 (8yr)** and buys
+2.89pp of MaxDD on the 26yr but only +0.59pp on the 8yr, where it makes the *worst-case*
drawdown 1.24pp **worse**. Its timing skill is negative on both horizons; it is a
drawdown-purchasing option costing ~1pp of CAGR a year that only pays in a genuine credit crisis.

### Paired per-start results, REAL time-varying financing (which FAVOURS the deployed arm)

**26yr, 12 starts:**

| arm | CAGR | Sharpe | Sortino | MaxDD | dCAGR | dSharpe | dMaxDD | +CAGR | **+Sharpe** | **+DD** | top-5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A deployed 1.49×+overlay | +11.97% | 0.531 | 0.743 | −64.1% | — | — | — | — | — | — | — |
| B const 1.00× | +11.38% | **0.569** | **0.807** | **−57.7%** | −0.58pp | +0.038 | **+6.40pp** | 3/12 | **12/12** | **12/12** | n/a |
| **C const 1.10×** | **+12.11%** | **0.564** | 0.801 | **−61.6%** | **+0.14pp** | **+0.034** | **+2.49pp** | 7/12 | **12/12** | **12/12** | **28%** |

**8yr, 12 starts, corrected cost pairing (EXP-018, each cost level paired against the deployed
arm at the SAME cost — fixing BUGS A8b):**

| cost | arm | dCAGR | dSharpe | dMaxDD | +CAGR | +Sharpe | +DD |
|---|---|---|---|---|---|---|---|
| ×1 | B const 1.00× | −1.78pp | +0.035 | +5.60pp | 2/12 | 7/12 | **12/12** |
| ×1 | C const 1.10× | −0.20pp | +0.026 | +2.15pp | 8/12 | 8/12 | 8/12 |
| ×2 | B | −1.50pp | +0.034 | +5.52pp | 3/12 | 8/12 | **12/12** |
| ×2 | C | −0.00pp | **+0.028** | +2.08pp | 8/12 | 8/12 | 8/12 |
| ×3 | B | −1.20pp | +0.034 | +5.60pp | 4/12 | 8/12 | **12/12** |
| ×3 | C | **+0.19pp** | **+0.030** | +2.32pp | 8/12 | 8/12 | 7/12 |

### Audit

**Cost sensitivity — PASSES, and confirms the mechanism.** Pre-registered: *"dSharpe should hold
or grow with cost, because removing the overlay removes leverage churn so the candidate trades
less. If dSharpe shrinks with cost, the gain is not where I think it is."*
Observed: C's dSharpe **grows** +0.026 → +0.028 → +0.030 and its dCAGR **improves** −0.20 → −0.00
→ +0.19pp as cost triples. B's dSharpe holds flat at +0.034-0.035. The advantage is a
turnover reduction, exactly as predicted.

**Event concentration — PASSES.** C on the 26yr: **28%** of excess from the top-5 days, against a
50% reject threshold, on a clearly meaningful net excess.

**Honest financing — PASSES.** Everything above uses the real DFF+1.5pp curve, which *helps the
deployed arm* (it runs more gross, so it gains more from the corrected rate: +0.71pp CAGR /
+0.021 Sharpe). The candidate wins anyway.

**Robustness — PASSES.** Not a knife-edge: 1.00×, 1.10× and 1.20× are all positive on 26yr
Sharpe (+0.038 / +0.034 / +0.031, all 12/12).

### Verdict — stated precisely, without inflation

| | 26yr | 8yr |
|---|---|---|
| C const 1.10× dSharpe | **+0.034, 12/12** | +0.026, **8/12** |
| C const 1.10× dMaxDD | **+2.49pp, 12/12** | +2.15pp, 8/12 |
| C const 1.10× dCAGR | +0.14pp | −0.20pp |
| B const 1.00× dMaxDD | **+6.40pp, 12/12** | **+5.60pp, 12/12** |

**This does NOT fully clear my pre-registered bar of ≥9/12 on BOTH horizons.** It clears
decisively on the 26yr (12/12 on Sharpe and drawdown) and is **marginal at 8/12 on the 8yr**. It
is never negative on either horizon on Sharpe or drawdown. I am recording it as *marginal-to-
strong*, not as a clean pass, and not lowering the bar to make it one.

**The single most sign-consistent finding in this entire program:** removing the overlay improves
**drawdown in 12/12 starts on BOTH horizons** — +6.40pp (26yr) and +5.60pp (8yr) at constant
1.00×, and +2.49pp / +2.15pp at 1.10×. Given EXP-004 established that drawdown is the axis where
this strategy is genuinely behind passive, that is the relevant axis.

**Deployability:** trivial. Two config values, and it *deletes* a code path rather than adding
one — the opposite of the tranching candidate, which needs real new engine work.

**NOT DEPLOYED.** Recorded in BASELINE.md as the second audited candidate. The choice between
1.00× (max drawdown protection, −0.6 to −1.8pp CAGR) and 1.10× (CAGR-neutral, smaller
protection) is the owner's.

---

## Next

Running: EXP-001 26yr · EXP-001b (capital + live-sizing control) · EXP-003 (I-21 filter vs
ranking) · EXP-005 (I-25 risk-matched leverage sweep).

Queued: EXP-006 (I-23 partial-adjustment rebalancing — the deployable form of the one
preliminary win), then the risk/construction tier I-03 / I-04 / I-05, which cycle 3 promoted
above the signal tier.
