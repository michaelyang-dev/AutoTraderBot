# LOG — one entry per experiment, wins and kills alike

Format: hypothesis → change → IS/OOS metrics → audit result → verdict → *why* it failed.
Kills are logged in as much detail as wins; the failure reasons are what generate the next
hypotheses.

**Configurations tested to date: 4,442 (this program) + ~60 inherited (see "Inherited verdicts").**

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

## Cycle 20 — EXP-022 · CREDIT-GATE BASELINE FIX (BUGS A9) — **corrects my own record**

**The defect I found in my own program.** `evalkit.DEPLOYED` had no `credit_pct`, so cycles 1-19
compared every challenger against a baseline **without** the HY-OAS credit gate — while the live
`ibkr_engine.compute_credit_derisk()` halves gross leverage whenever HY-OAS sits at or above its
p95 expanding percentile (`credit_gate.PCT=0.95`, `DERISK=0.5`, live since 2026-07-18). I built
`DEPLOYED` from the backtester's config defaults instead of from the live engine. Same class as
the `clear_deployed()` footgun.

**26yr, 12 starts, live_sizing, REAL time-varying financing:**

| arm | CAGR | Sharpe | Sortino | MaxDD | worst DD |
|---|---|---|---|---|---|
| A0 baseline **as I had been using it** (gate OFF) | +12.10% | 0.528 | 0.742 | −64.5% | −71.4% |
| **A1 baseline AS ACTUALLY DEPLOYED (gate ON)** | **+12.72%** | **0.549** | **0.769** | **−55.9%** | −64.2% |
| C0 const 1.10× no-overlay, gate OFF | +12.25% | 0.555 | 0.791 | −62.2% | −70.1% |
| **C1 const 1.10× no-overlay, gate ON** | **+12.93%** | **0.584** | **0.833** | **−51.8%** | −63.4% |
| **B1 const 1.00× no-overlay, gate ON** | +12.27% | **0.592** | **0.845** | **−48.1%** | −59.7% |
| T1 tranching K=4, gate ON | +13.24% | 0.577 | 0.814 | −55.6% | **−57.3%** |

| paired delta | dCAGR | dSharpe | dMaxDD | +CAGR | +Sharpe | +DD |
|---|---|---|---|---|---|---|
| **GATE effect on baseline (A1−A0)** | **+0.61pp** | **+0.021** | **+8.54pp** | **12/12** | **12/12** | **12/12** |
| C const 1.10× vs HONEST base | +0.21pp | **+0.035** | **+4.11pp** | 7/12 | **12/12** | **12/12** |
| B const 1.00× vs HONEST base | −0.45pp | **+0.043** | **+7.84pp** | 4/12 | **12/12** | **12/12** |
| T tranching K=4 vs HONEST base | +0.52pp | +0.028 | +0.32pp | 7/12 | 9/12 | 5/12 |
| C vs the OLD (wrong) base | +0.15pp | +0.027 | +2.25pp | 7/12 | 12/12 | 12/12 |

### 1. The risk I flagged did NOT materialise — and it went the other way

I predicted the credit gate and constant-lower-leverage might be **substitutes**, which would
have shrunk or killed the EXP-017/018 finding. Measured substitution `(C0−A0) − (C1−A1)`:
**−0.06pp CAGR, −0.008 Sharpe, −1.86pp MaxDD.** Negative — they are **complements**.
Removing the overlay helps *more* when the gate is present (+4.11pp of drawdown vs +2.25pp
without it), because with a genuine crisis control already in place the overlay's contribution is
even more redundant while its ~1pp/yr CAGR drag is unchanged. **The EXP-017/018 result is
strengthened, not weakened.**

### 2. 🔴 CORRECTION to cycle 3 (EXP-004): the strategy does NOT lose to passive over 26yr

Cycle 3 concluded the deployed strategy was "at parity with passive EW SP1500 on return and
behind on risk". That comparison used a baseline missing **both** the credit gate **and** honest
financing. Corrected:

| | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| passive EW SP1500, quarterly, cost-free | +11.17% | **0.590** | −58.5% |
| **deployed as actually run (A1)** | **+12.72%** | 0.549 | **−55.9%** |
| **C1 (deployed minus the vol overlay)** | **+12.93%** | 0.584 | **−51.8%** |
| **B1 (const 1.00×, no overlay)** | +12.27% | **0.592** | **−48.1%** |

The deployed system **beats passive on CAGR by +1.55pp and on drawdown by +2.6pp**, losing only
0.04 of Sharpe — against a benchmark that pays zero costs and holds 1,500 names nobody could hold
directly. And **B1 beats it on all three axes.** My cycle-3 framing was wrong and is corrected
here rather than left to stand.

### 3. Tranching's drawdown contribution mostly disappears — as I predicted it would

Against the honest gate-ON baseline, tranching gives **+0.32pp mean MaxDD at 5/12** — essentially
nothing — while keeping **+0.52pp CAGR and +0.028 Sharpe (9/12)**. That is the expected result and
I said so before running it: tranching is variance reduction across rebalance phase, and does not
compete with a credit gate for tail protection.

One thing survives: **worst-case drawdown across starts −64.2% → −57.3% (+6.9pp)** even though
the *mean* is flat. Tranching compresses the dispersion of drawdown outcomes rather than lowering
the average, which is exactly what a variance-reduction mechanism should do.

**Net effect on the two candidates:**
- **Remove-the-overlay: STRENGTHENED.** vs the honest baseline C is +0.035 Sharpe / +4.11pp DD,
  both **12/12**; B is +0.043 / +7.84pp, both **12/12**.
- **Tranching: NARROWED.** Still +0.028 Sharpe (9/12) and +0.52pp CAGR, but its drawdown claim
  against a correctly-specified baseline is ~zero. Given it needs substantial new engine code
  while the overlay change is two config values, **its priority drops below the overlay change.**

### 4. 8yr confirmation — and it narrows BOTH candidates

| arm | CAGR | Sharpe | MaxDD | worst |
|---|---|---|---|---|
| A0 base gate-OFF | +23.85% | 0.799 | −40.4% | −45.5% |
| **A1 base gate-ON (honest)** | **+23.88%** | **0.802** | **−38.5%** | −45.5% |
| C1 const 1.10× no-overlay, gate ON | +23.45% | 0.812 | −37.3% | −41.5% |
| B1 const 1.00× no-overlay, gate ON | +22.09% | **0.825** | **−34.2%** | **−37.1%** |
| T1 tranching K=4, gate ON | **+26.56%** | **0.892** | −37.6% | −42.4% |

| paired vs the HONEST 8yr base | dCAGR | dSharpe | dMaxDD | +CAGR | +Sharpe | +DD |
|---|---|---|---|---|---|---|
| gate effect (A1−A0) | +0.02pp | +0.003 | +1.87pp | 3/12 | 4/12 | 6/12 |
| C const 1.10× | −0.43pp | +0.010 | +1.25pp | 6/12 | 8/12 | 6/12 |
| **B const 1.00×** | −1.78pp | +0.023 | **+4.32pp** | 2/12 | 8/12 | **11/12** |
| T tranching K=4 | **+2.68pp** | **+0.090** | +0.91pp | 7/12 | 7/12 | 6/12 |

**The gate does essentially nothing on the 8yr** (+0.02pp CAGR, 3/12) — 2018-2025 contains no
genuine credit crisis. That is exactly as documented, and it means the 8yr baseline barely moved,
so the 8yr verdicts below are a real re-test rather than an artefact of the correction.

### Both candidates, re-scored against the HONEST baseline on both horizons

| | 26yr dSharpe | 8yr dSharpe | 26yr dMaxDD | 8yr dMaxDD |
|---|---|---|---|---|
| **B const 1.00×, no overlay** | **+0.043 (12/12)** | +0.023 (8/12) | **+7.84pp (12/12)** | **+4.32pp (11/12)** |
| C const 1.10×, no overlay | **+0.035 (12/12)** | +0.010 (8/12) | **+4.11pp (12/12)** | +1.25pp (6/12) |
| T tranching K=4 | +0.028 (9/12) | +0.090 (7/12) | +0.32pp (5/12) | +0.91pp (6/12) |

**Neither candidate clears ≥9/12 on Sharpe on BOTH horizons.** B and C reach 12/12 on the 26yr
and 8/12 on the 8yr; tranching reaches 9/12 and 7/12. I am not relaxing the bar — both stay
**marginal-to-strong, not clean passes.**

**One claim DOES clear cleanly on both horizons: B's drawdown.** +7.84pp at 12/12 (26yr) and
+4.32pp at 11/12 (8yr). Running constant 1.00× with no vol overlay is a genuine, two-horizon,
sign-consistent drawdown improvement, costing −0.45pp (26yr) / −1.78pp (8yr) of CAGR. That is
the most robust deployable result this program has produced, and it is a *config change that
deletes code*.

**Tranching is narrowed and reprioritised.** Against a correctly-specified baseline its drawdown
claim is ~zero (5/12, 6/12); what survives is CAGR/Sharpe (+0.52pp/+0.028 on 26yr, +2.68pp/+0.090
on 8yr) at 7-9/12. Since it needs substantial new engine code (four books netted in one IBKR
account) while the overlay change is two config values, **its priority drops below the overlay
change.**

**BASELINE.md corrected.** `DEPLOYED` is left unchanged so the 19 prior cycles stay reproducible;
the gate-ON numbers are recorded alongside as the honest reference, and `credit_pct`/`credit_derisk`
must be passed explicitly from here on.

---

## Cycle 21 — EXP-020 · portfolio construction (I-04 sleeve risk parity, I-13 overlap) — **both DEAD**

Escalation-ladder level 5. Neither adds a signal; both change how existing signals are combined.
Parity gate passed (hooks off ≡ untouched path, bit-for-bit).

**8yr, 12 starts, REAL financing. Base +23.03% / 0.800 / −39.2%, avgGross 1.1117.**

| arm | CAGR | Sharpe | MaxDD | avgGross | dCAGR | dSharpe | +Sharpe | verdict |
|---|---|---|---|---|---|---|---|---|
| sleeveRP p=0.5 | +21.89% | 0.784 | −39.3% | 1.1428 | −1.14pp | −0.015 | 2/12 | dead |
| sleeveRP p=1.0 | +20.79% | 0.768 | −39.2% | 1.1723 | −2.25pp | −0.032 | 1/12 | dead |
| sleeveRP p=1.5 | +19.55% | 0.747 | −38.9% | 1.1956 | −3.48pp | −0.053 | 0/12 | dead |
| overlap flat (no doubling) | +22.85% | 0.808 | −39.8% | 1.1232 | −0.19pp | +0.008 | 8/12 | marginal |
| overlap boost ×1.25 | +23.25% | 0.799 | −38.5% | 1.0973 | +0.21pp | −0.000 | 6/12 | dead |
| overlap boost ×1.50 | +23.33% | 0.799 | −38.0% | 1.0822 | +0.30pp | −0.001 | 7/12 | marginal |
| sleeveRP + flat | +20.35% | 0.772 | −39.8% | 1.1829 | −2.68pp | −0.028 | 4/12 | dead |

### I-04 sleeve risk parity — DEAD, and the monotonicity is the finding

Sharpe degrades **monotonically in the parity power**: −0.015 → −0.032 → −0.053 at p = 0.5 → 1.0
→ 1.5, with 2/12 → 1/12 → **0/12** sign consistency. A clean dose-response, not noise.

**And it loses while running MORE exposure.** avgGross rises 1.1117 → 1.1956 as the tilt
strengthens (shifting capital toward the 10-name lowvol sleeve flattens the weight distribution,
so the 10% cap binds less often and realised gross goes up). More gross normally *raises* CAGR;
this loses 3.48pp anyway. The direction is unambiguous despite the exposure mismatch.

**Why: momentum's outsized risk contribution is EARNED, not accidental.** EXP-003 showed the
ranking carries 100% of the selection edge, and momentum is the ranked sleeve. Equalising risk
across sleeves deliberately moves capital *away* from the only sleeve that has an edge and toward
sleeves that are, on that evidence, close to random draws from a filtered pool. The fixed
50/35/15 capital split is better precisely because it is *not* risk-balanced.

**⚠️ Caveat on my own implementation, stated rather than buried.** I tried to make this
gross-neutral by renormalising the sleeve weights to a constant sum, and it did not work — the
10% position cap and the 0.5% minimum-weight filter both act *after* that rescale, so changing
the weight *distribution* changes realised gross even when the pre-cap sum is held fixed. The
arms are still interpretable (they lose while running more exposure), but a clean version would
have to rescale post-cap or be scored through `matched_exposure_curve`. Flagged in the output as
"GROSS MOVED, discard" rather than quietly reported.

### I-13 cross-sleeve overlap — DEAD, and that is a useful answer

All three arms sit within ±0.008 of base Sharpe. Boosting duplicates does nothing (−0.000,
−0.001); removing the doubling entirely does nothing (+0.008, 8/12).

**So cross-sleeve agreement carries no information — and the current double-counting is not
harmful either.** That closes the open question: no risk control is needed there, and the
sleeves' shared inputs (momentum and lowvol both read `roe`; value and lowvol both read
`gross_margin`) are not silently concentrating the book in any way that costs money. This is the
outcome I flagged as worth knowing regardless of sign, and it is.

**Generated:** the sleeve-RP dose-response says the book is *under*-weighted to momentum on a
risk basis, not over. The opposite tilt — deliberately raising the momentum sleeve's share — is
the untested direction that dose-response implies → **I-27**.

### 🔴 CORRECTION — the 26yr does NOT reproduce the 8yr dose-response. My write-up above was
### drawn from one horizon and overstated.

26yr, 12 starts (base +11.97% / 0.531 / −64.1%, gross 1.1939):

| arm | CAGR | Sharpe | MaxDD | gross | dCAGR | **dSharpe** | **dMaxDD** | +Sharpe |
|---|---|---|---|---|---|---|---|---|
| sleeveRP p=0.5 | +11.85% | 0.532 | −62.9% | 1.2209 | −0.12pp | **+0.001** | **+1.19pp** | 9/12 |
| sleeveRP p=1.0 | +11.68% | 0.532 | −61.6% | 1.2442 | −0.29pp | **+0.002** | **+2.53pp** | 9/12 |
| sleeveRP p=1.5 | +11.48% | 0.531 | −59.9% | 1.2627 | −0.49pp | **+0.001** | **+4.16pp** | 9/12 |
| overlap flat | +11.75% | 0.527 | −63.3% | 1.2028 | −0.22pp | −0.004 | +0.81pp | 6/12 |
| overlap boost ×1.25 | +12.12% | 0.534 | −64.4% | 1.1815 | +0.15pp | +0.004 | −0.27pp | 6/12 |
| overlap boost ×1.50 | +12.26% | 0.539 | −64.6% | 1.1688 | +0.29pp | +0.008 | −0.45pp | 10/12 |
| sleeveRP + flat | +11.57% | 0.533 | −60.2% | 1.2534 | −0.40pp | +0.002 | +3.85pp | 7/12 |

**On the 26yr, sleeve risk parity is Sharpe-NEUTRAL (+0.001 to +0.002, 9/12), not harmful.** The
monotone dose-response I reported is real but it lives in **drawdown**, not Sharpe:
**+1.19 → +2.53 → +4.16pp of MaxDD** as the tilt strengthens, at −0.12 → −0.29 → −0.49pp of CAGR.

**So the honest verdict changes: sleeve RP is not "dead with a clean dose-response against it".
It is a CAGR-for-drawdown trade that is roughly free on the 26yr and costly on the 8yr**
(8yr: −0.015 → −0.032 → −0.053 Sharpe, 0-2/12). The two horizons disagree on Sharpe, so it still
**fails the two-horizon bar** and is not promoted — but my stated reason was wrong and the
mechanism story I wrote ("momentum's risk share is earned") is supported only by the 8yr.

**Also note it improves drawdown while running MORE gross** (1.1939 → 1.2627). That is not a
contradiction: shifting capital toward the 10-name lowvol sleeve genuinely lowers risk per
dollar, and flattening the weight distribution makes the 10% cap bind less often so realised
gross rises. More dollars in lower-risk names. It does mean the arms are not exposure-matched
and the flag says so.

**Overlap remains dead on both horizons** (|dSharpe| ≤ 0.008 everywhere; boost ×1.50 reaches
10/12 on the 26yr but −0.001/7 out of 12 on the 8yr, and +0.008 is inside the noise floor).
The conclusion that agreement between sleeves carries no information stands.

---

## Cycle 22 — EXP-021 · REVIVE BATCH (I-19) — **one idea comes back from the dead**

BUGS F6: most "dead" verdicts in this repo were reached on 3-4 starts AND on the pre-audit
poisoned universes — a combination that has already produced a wrong sign once and a retracted
win. Every idea here has an existing config hook, so **no new modelling risk is introduced; only
the measurement standard changes.**

**26yr, 12 starts, honest financing. Base +11.97% / 0.531 / −64.1%, avgGross 1.1939.**
(Note: gate OFF here — this batch was queued before the BUGS A9 correction landed. Deltas are
still valid, since base and variants share the same baseline.)

| arm | CAGR | Sharpe | MaxDD | gross | dCAGR | dSharpe | dMaxDD | +CAGR | +Shrp | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| vol-managed mom 0.20 | +10.69% | 0.522 | −60.2% | 1.126 | −1.28pp | −0.009 | +3.90pp | 0/12 | 6/12 | dead |
| vol-managed mom 0.30 | +11.26% | 0.522 | −62.7% | 1.171 | −0.71pp | −0.008 | +1.44pp | 0/12 | 6/12 | dead |
| risk parity in-sleeve | +11.82% | 0.538 | −62.5% | 1.213 | −0.15pp | +0.007 | +1.59pp | 5/12 | 7/12 | marginal |
| sector cap 2/sleeve | +11.72% | 0.522 | −64.0% | 1.196 | −0.25pp | −0.008 | +0.11pp | 2/12 | 2/12 | dead |
| mom quality **gp_assets** | +9.30% | 0.458 | −63.7% | 1.213 | **−2.67pp** | **−0.072** | +0.45pp | **0/12** | **0/12** | dead |
| **mom quality `roe`** | **+12.24%** | **0.551** | **−60.3%** | 1.212 | **+0.27pp** | **+0.021** | **+3.81pp** | **9/12** | **10/12** | **SURVIVES** |
| **signal-exit every 5d** | +10.83% | 0.550 | **−52.7%** | 1.146 | −1.14pp | +0.020 | **+11.39pp** | 5/12 | 8/12 | marginal |
| park idle cash IEF | +11.97% | 0.531 | −64.1% | 1.194 | +0.00pp | +0.000 | +0.00pp | 0/12 | 0/12 | **VOID — see below** |

### `park idle cash IEF` tested NOTHING — reporting it as "dead" would be wrong

Every metric returned **exactly 0.000**. That is the suspicious-roundness tell from BUGS section
E (4 of 6 harness bugs in this repo announced themselves as a number that was too clean). Traced
it: `park_target = max(0.0, nav_now * (1.0 - lev_t))` — the hook only funds a bond position when
the leverage target drops **below 1.0×**, which at `leverage=1.49` with `vol_scale_cap=1.0` and
`derisk=0.5` never happens (floor is 0.745×… and even then `1−lev_t` is negative). The arm is
**inert by construction**. Logged as VOID, not dead: "we tested it and it did nothing" and "the
code path never executed" are different claims and only one of them is true.

### The revival: `mom_quality_filter="roe"` — take top-15 by momentum, keep top-5 by ROE

**+0.27pp CAGR, +0.021 Sharpe, +3.81pp MaxDD, 9/12 and 10/12.** Previously logged dead.
It improves drawdown **while running MORE gross** (1.2115 vs 1.1939) — exposure normally makes
drawdown worse, so the sign of that is encouraging rather than suspicious.

**Mechanism:** quality-momentum interaction (Novy-Marx). Among names that have already run, the
profitable ones are the ones whose run reflects improving fundamentals; the unprofitable ones
reverse. It refines the **ordering**, which EXP-003 showed is the only part of selection carrying
any edge — so it acts on the right axis rather than bolting on a new one.

**THE STRONGEST ARGUMENT AGAINST IT, stated plainly:** `gp_assets` — the *other* quality metric,
same mechanism, same hook, same pool size — was **catastrophic** (−2.67pp, −0.072 Sharpe, 0/12
and 0/12). If quality-momentum were a robust mechanism, a closely-related profitability measure
should not be that bad. That asymmetry is real evidence this is a lucky draw. Also: `roe` is one
arm of nine here, and Bonferroni on 9 tests turns a nominal p≈0.03 into ≈0.24.

**And `roe` is a FUNDAMENTAL**, so unlike every price-based arm it is exposed to period-vs-release
timestamping. **The shift test is the one that matters** and it leads the audit gate.

**EXP-024 queued:** 8yr confirmation, shift test, pool sweep (2/3/4/6 — a real effect should not
need exactly 15 candidates), cost pairing at matched multipliers, guarded event concentration,
matched-exposure residual. **Status: PRELIMINARY — UNAUDITED.**

### Also worth a follow-up: mid-cycle signal exit

`signal_exit_every=5` (sell a holding when no sleeve still wants it, without waiting out the
20-day cadence) gives **+11.39pp of MaxDD** (−64.1% → −52.7%) and +0.020 Sharpe (8/12) for
−1.14pp of CAGR. Marginal on the sign test, but that is the largest single drawdown improvement
any arm has produced — on the axis EXP-004/022 identified as the one that matters. → **I-29**.

**Everything else stays dead**, now on the better standard rather than on 3-4 starts: vol-managed
momentum, in-sleeve risk parity, sector caps, gp_assets quality. That is cheap negative knowledge
and it stops these being re-proposed.

---

## Cycle 23 — EXP-019 · DO THE TWO CANDIDATES STACK? — **yes, additively. Best combined arm yet.**

**The specific worry.** Both candidates reduce variance from a nuisance: tranching averages away
the rebalance-phase lottery, removing the overlay stops leverage lurching on a stale 40-day
estimate. If a large part of what tranching fixes *is* the overlay's staleness — four sub-books
each carrying their own vol estimate, so averaging them smooths exactly the noise the overlay
injects — then doing both should deliver much less than the sum, and the right advice would be
"do the cheap config change, skip the expensive engine work". Worth knowing **before** anyone
builds four-book netting into `ibkr_engine`.

**26yr, 12 starts, REAL financing, live_sizing:**

| arm | CAGR | **sd CAGR** | Sharpe | Sortino | MaxDD | **worst DD** |
|---|---|---|---|---|---|---|
| A deployed | +12.10% | 2.75pp | 0.528 | 0.742 | −64.5% | −71.4% |
| B tranching K=4 only | +12.67% | 1.17pp | 0.557 | 0.787 | −64.7% | −65.3% |
| C const 1.10× only | +12.25% | 2.30pp | 0.555 | 0.791 | −62.2% | −70.1% |
| **D BOTH @1.10×** | **+12.84%** | **1.06pp** | **0.584** | **0.830** | −62.5% | −63.2% |
| **E BOTH @1.00×** | +12.21% | **0.96pp** | **0.594** | **0.845** | **−58.5%** | **−59.1%** |

| paired vs A | dCAGR | dSharpe | dMaxDD | +CAGR | +Sharpe | +DD | sd ratio |
|---|---|---|---|---|---|---|---|
| B tranching | +0.57pp | +0.028 | −0.20pp | 8/12 | 8/12 | 5/12 | 0.427 |
| C const 1.10× | +0.15pp | +0.027 | +2.25pp | 7/12 | **12/12** | **12/12** | 0.836 |
| **D BOTH @1.10×** | **+0.74pp** | **+0.056** | +1.99pp | 8/12 | **10/12** | 8/12 | **0.383** |
| **E BOTH @1.00×** | +0.11pp | **+0.066** | **+5.99pp** | 8/12 | **10/12** | **12/12** | **0.350** |

### Stacking: `(D−A) − [(B−A)+(C−A)]`

| | additive prediction | actual | **interaction** |
|---|---|---|---|
| D CAGR | +0.713pp | +0.738pp | **+0.024** |
| D Sharpe | +0.055 | +0.056 | **+0.001** |
| D MaxDD | +2.051pp | +1.993pp | **−0.058** |

**Almost perfectly additive.** The interaction terms are ~1-4% of the effects. **My overlap
worry was wrong: the two candidates fix genuinely independent problems** — one is a *timing*
nuisance (which day of the cycle you trade), the other a *sizing* nuisance (a stale vol estimate
moving leverage). Doing both gets both.

**And at 1.00× the drawdown interaction is strongly SUPER-additive:**

| E BOTH @1.00× | additive | actual | interaction |
|---|---|---|---|
| CAGR | +0.713pp | +0.111pp | −0.602 |
| Sharpe | +0.055 | +0.066 | +0.011 |
| **MaxDD** | **+2.051pp** | **+5.994pp** | **+3.942** |

Combining tranching with constant 1.00× buys **~4pp MORE drawdown protection than the sum of the
parts**, at the cost of most of the CAGR gain. Plausible mechanism: staggered sub-books enter a
crisis at four different exposure points, and without the overlay chasing a stale estimate none
of them is caught over-levered at the wrong moment — the tail improvement compounds rather than
overlapping. That is a hypothesis fitted after the fact and is labelled as such.

### The best arms, against the deployed system

**E (both, 1.00×) improves ALL THREE axes:** CAGR +0.11pp, **Sharpe +0.066 (10/12)**,
**MaxDD +5.99pp (12/12)**, worst-case DD **−71.4% → −59.1% (+12.3pp)**, and cuts outcome
dispersion by 65% (sd ratio 0.350).
**D (both, 1.10×)** is the CAGR-preferring version: +0.74pp CAGR, +0.056 Sharpe (10/12).

**Status: PRELIMINARY — UNAUDITED.** Outstanding: 8yr confirmation (queued), cost sensitivity,
and a re-run against the gate-ON baseline (BUGS A9 — this used gate OFF, so A is the *old*
baseline; EXP-022 showed the gate and constant leverage are complements, which should preserve
these deltas, but "should" is not "measured").

**Concentration reads `n/a(0/12)`** — the guarded metric (BUGS A8a) correctly refuses to report a
share when the net excess is small relative to the gross daily movement. That is the guard
working, not a pass. It means these arms differ from the baseline slowly and steadily rather
than through a few events — which is what a variance-reduction result should look like — but it
is not the same as passing a concentration test, and I am not recording it as one.

---

## Cycle 24 — EXP-019 8yr, gate-ON · the combined arm confirmed on the second horizon

Re-run with the **honest gate-ON baseline** (BUGS A9). Earlier I wrote that the gate-OFF deltas
"should hold, but 'should' is not 'measured'." Now measured.

**8yr, 12 starts, gate ON, REAL financing, live_sizing:**

| arm | CAGR | sd CAGR | Sharpe | Sortino | MaxDD | worst DD |
|---|---|---|---|---|---|---|
| A deployed | +23.88% | 7.05pp | 0.802 | 1.153 | −38.5% | −45.5% |
| B tranching K=4 | **+26.56%** | 4.05pp | 0.892 | 1.306 | −37.6% | −42.4% |
| C const 1.10× | +23.45% | 6.67pp | 0.812 | 1.193 | −37.3% | −41.5% |
| D BOTH @1.10× | +25.92% | 4.42pp | 0.888 | 1.311 | −37.1% | −40.9% |
| **E BOTH @1.00×** | +24.19% | **3.92pp** | **0.898** | **1.325** | **−33.8%** | **−37.1%** |

| paired vs A | dCAGR | dSharpe | dMaxDD | +Sharpe | +DD | sd ratio |
|---|---|---|---|---|---|---|
| B tranching | +2.68pp | +0.090 | +0.91pp | 7/12 | 6/12 | 0.574 |
| C const 1.10× | −0.43pp | +0.010 | +1.25pp | 8/12 | 6/12 | 0.946 |
| D BOTH @1.10× | +2.05pp | +0.086 | +1.45pp | 8/12 | 6/12 | 0.626 |
| **E BOTH @1.00×** | **+0.31pp** | **+0.096** | **+4.76pp** | 8/12 | **9/12** | 0.555 |

### Arm E across BOTH horizons, versus the deployed system

| | dCAGR | dSharpe | dMaxDD | worst DD | sd ratio |
|---|---|---|---|---|---|
| **26yr** | **+0.11pp** | **+0.066 (10/12)** | **+5.99pp (12/12)** | −71.4% → **−59.1%** | 0.350 |
| **8yr** | **+0.31pp** | **+0.096 (8/12)** | **+4.76pp (9/12)** | −45.5% → **−37.1%** | 0.555 |

**Positive on all three axes on both horizons.** Drawdown clears the ≥9/12 bar on both (12/12,
9/12). **Sharpe clears on the 26yr (10/12) and MISSES on the 8yr (8/12)** — stated plainly rather
than smoothed over. CAGR is positive on both but at 6-8/12, i.e. incidental rather than a claim.

### The drawdown super-additivity replicates

| interaction `(E−A) − [(B−A)+(C−A)]` | 26yr | 8yr |
|---|---|---|
| CAGR | −0.602pp | −1.944pp |
| Sharpe | +0.011 | −0.003 |
| **MaxDD** | **+3.942pp** | **+2.598pp** |

Sharpe stacking is essentially additive on both horizons (|interaction| ≤ 0.013). The **drawdown
interaction is super-additive on both** — combining tranching with constant 1.00× protects the
tail by 2.6-3.9pp *more* than the sum of the parts, while giving up most of the CAGR gain.
A replicated interaction across two independent horizons is much harder to dismiss than the
single-horizon version I flagged as "fitted after the fact" last cycle, though the mechanism
(four staggered sub-books entering a crisis at four exposure points, none caught over-levered by
a stale estimate) remains a story told after seeing the number.

**Status: PRELIMINARY — UNAUDITED.** EXP-027 now running the full gate: matched-exposure control
(the decisive one — arm E runs ~1.00× against the deployed ~1.19×, and de-levering improves
drawdown by arithmetic alone; this is exactly how EXP-008 died), cost ×1/×2/×3 paired at matched
multipliers, real $33k account size, and guarded event concentration.

---

## Cycle 25 — EXP-024 · ROE-QUALITY AUDIT GATE — **no leak, real, but a SUBSTITUTE not an addition**

26yr, 12 starts, honest gate-ON baseline (+12.57% / 0.552 / −55.5%, gross 1.1821).

### B. SHIFT TEST — **PASSES**, and this was the one that mattered

`roe` is the first FUNDAMENTAL-driven arm in this program, so unlike every price-based result it
is exposed to period-vs-release timestamping. Lagging every signal one full bar:

| | dCAGR | dSharpe | dMaxDD | +Sharpe |
|---|---|---|---|---|
| unshifted | +0.39pp | +0.026 | +6.47pp | 10/12 |
| **SHIFTED +1 bar** | **+0.73pp** | **+0.035** | +4.95pp | **10/12** |

The shifted arm is if anything **stronger** on Sharpe. A period-vs-release leak dies or flips
here. **No look-ahead.**

### C. POOL SWEEP — positive everywhere, but NON-MONOTONE. The headline must not be the peak.

| pool | top-N by momentum | dSharpe | dMaxDD | +Sharpe |
|---|---|---|---|---|
| 2 | 10 | +0.020 | +5.95pp | 7/12 |
| 3 | 15 | +0.026 | +6.47pp | 10/12 |
| **4** | **20** | **+0.080** | **+6.74pp** | **11/12** |
| 6 | 30 | +0.020 | +4.29pp | 9/12 |

**pool=4 is a spike, not a plateau** — 3× its neighbours on Sharpe. That is the signature of a
fitted parameter, and quoting +0.080 would be exactly the "best of N configurations without
deflating for N" trap. **The honest number is the plateau: ~+0.020 to +0.026.** Recorded here so
the +0.080 is never lifted out of context.

What *is* encouraging: the sign is positive at **every** pool size, and the **drawdown gain is
large and stable across all four** (+4.29 to +6.74pp). The drawdown effect looks like the robust
part; the Sharpe effect is small and noisy.

### D. COST (paired at matched multipliers, BUGS A8b) — survives, degrading gracefully

| cost | dCAGR | dSharpe | dMaxDD | +Sharpe |
|---|---|---|---|---|
| ×1 | +0.39pp | +0.026 | +6.47pp | 10/12 |
| ×2 | +0.27pp | +0.020 | +6.37pp | 10/12 |
| ×3 | +0.16pp | +0.014 | +6.14pp | 10/12 |

Sharpe decays with cost (a turnover-adding change, unlike the overlay removal whose advantage
*grew*), but stays positive and **10/12 at every level**, and the drawdown gain barely moves.

### E. gp_assets — still catastrophic: −2.73pp CAGR, −0.073 Sharpe, **0/12**

The contradiction is unresolved and remains the strongest argument that `roe` is a lucky draw.
EXP-025 (three more profitability metrics) settles it.

### F. 🔴 MATCHED EXPOSURE — **dSharpe −0.003 (8/12), dCAGR −0.75pp.** The important caveat.

Against a constant-leverage curve at its own realised gross, the ROE arm's Sharpe advantage is
**zero**. Reading it correctly: the ROE arm still carries the vol overlay, and EXP-015 showed the
overlay is worth about **−0.024 Sharpe** versus constant leverage. So ROE's +0.026 over the
deployed baseline is **the same size as simply deleting the overlay**, and it does not beat that
alternative.

**⇒ ROE quality and overlay-removal look like SUBSTITUTES, not complements.** Both deliver
~+0.02-0.03 Sharpe and several pp of drawdown; neither obviously adds on top of the other. That
is a materially weaker claim than "a new independent edge", and it is what the control was for.

**VERDICT: PRELIMINARY, real but redundant.** Clean on leakage, consistent in sign across pools
and costs, with a robust drawdown effect — but it buys roughly what the far simpler config change
already buys, and its Sharpe magnitude was overstated by the pool=4 spike.

**Not promoted.** Two things decide it → **EXP-025** (is the mechanism real, or is `roe` one
lucky draw among profitability metrics?) and **EXP-028** (does ROE add anything ON TOP of
overlay-removal, or are they redundant?).

---

## Cycle 26 — EXP-027 · COMBINED-ARM AUDIT — matched exposure PASSES, and separates two claims
## I had been conflating

**26yr, $50k, cost ×1, gate ON. Base +12.72% / 0.549 / −55.9% (worst −64.2%), gross 1.2206.**

| arm | CAGR | Sharpe | MaxDD | worst | gross | dCAGR | dSharpe | dMaxDD | +Shrp | +DD | sd ratio | **MATCHED dSharpe / dMaxDD** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E both @1.00×** | +12.79% | **0.623** | **−48.2%** | **−51.4%** | 0.987 | +0.08pp | **+0.074** | **+7.74pp** | 10/12 | **12/12** | **0.397** | **+0.035 / −0.11pp** |
| D both @1.10× | **+13.48%** | 0.612 | −52.0% | −55.4% | 1.086 | +0.77pp | +0.063 | +3.94pp | 10/12 | **12/12** | 0.443 | **+0.029 / −0.20pp** |

### The matched-exposure test — the one that killed EXP-008 — PASSES

Arm E runs 0.987× gross against the deployed 1.221×, so a large part of its drawdown gain could
be pure de-levering. Scored against a constant-leverage curve at its **own** realised gross:

**MATCHED dSharpe = +0.035 (E), +0.029 (D).** For contrast, EXP-008's ex-ante vol estimator
showed raw dMaxDD up to +12.29pp and a matched dSharpe of **−0.099** — it was entirely a level
effect. This is not.

### 🔴 But the same control splits the result in two, and I had been reporting them as one

| | matched dSharpe | matched dMaxDD |
|---|---|---|
| E both @1.00× | **+0.035** | **−0.11pp** |
| D both @1.10× | **+0.029** | **−0.20pp** |

**The Sharpe gain is real. The DRAWDOWN gain is almost entirely a level effect.** Matched dMaxDD
is ≈0 (slightly negative), meaning the +7.74pp of drawdown improvement is what *any* book running
0.99× instead of 1.22× would get. It is not a tail-protection skill.

That matters for how this is described. Through cycles 23-24 I reported "+5.99pp / +4.76pp of
drawdown, 12/12 on both horizons" as if it sat alongside the Sharpe gain as independent evidence.
It does not. The honest split:

- **Genuine, exposure-independent:** +0.029 to +0.035 Sharpe, from phase diversification plus
  removing a demonstrably harmful overlay. Also the **sd-ratio 0.397** — outcome dispersion cut
  ~60% — which is a real, exposure-independent property.
- **A level effect anyone can have:** the drawdown improvement. Available just as cheaply by
  running lower leverage, and already quantified as such in EXP-010.

Neither is worthless — de-levering is a legitimate choice and this arm delivers it at **no CAGR
cost** (+0.08pp), which plain de-levering does not (EXP-010: 1.49×→1.00× costs −2.60pp). That
combination is the actual finding: **the same drawdown as de-levering, without the CAGR bill,
plus ~+0.03 of exposure-independent Sharpe.**

**Still outstanding:** $33k account size, cost ×2/×3, and the 8yr. Running.

---

## Cycle 27 — EXP-025 · PROFITABILITY FAMILY — **KILLS the ROE result. Pre-registration did its job.**

EXP-024 had cleared ROE on leakage (shift test came back *stronger*), on cost (10/12 at ×1/×2/×3)
and on pool robustness (positive at every pool size). This is the test that kills it anyway.

**Pre-registered criterion, written before running:** *"≥3/5 positive with a clean
income-vs-grossprofit split → MECHANISM, roe is real. Scattered, ~half positive, noise-sized →
LUCKY DRAW, do not believe EXP-024."*

**26yr, 12 starts, gate ON:**

| metric | numerator | dCAGR | dSharpe | dMaxDD | +Sharpe | verdict |
|---|---|---|---|---|---|---|
| **roe** | income/equity | +0.39pp | **+0.026** | +6.47pp | 10/12 | survives |
| net_margin | income/sales | −1.70pp | **−0.040** | +0.88pp | 1/12 | dead |
| operating_margin | opinc/sales | −1.16pp | **−0.021** | +2.08pp | 7/12 | marginal |
| gp_assets | grossprofit/assets | −2.73pp | **−0.073** | −0.18pp | 0/12 | dead |
| gross_margin | grossprofit/sales | −2.46pp | **−0.064** | −0.70pp | 1/12 | dead |

income-based mean dSharpe **−0.012**; gross-profit mean **−0.069**. **1/5 positive.**

**8yr, 12 starts — worse still:**

| metric | dCAGR | dSharpe | +Sharpe |
|---|---|---|---|
| roe | **−7.40pp** | **−0.160** | **2/12** |
| net_margin | −9.14pp | −0.211 | 0/12 |
| operating_margin | −10.64pp | −0.257 | 0/12 |
| gp_assets | −9.44pp | −0.221 | 0/12 |
| gross_margin | −9.45pp | −0.221 | 0/12 |

**0/5 positive.** income-based mean −0.210, gross-profit mean −0.221.

### VERDICT: ROE QUALITY IS DEAD. Killed on three independent grounds.

1. **Family test (the pre-registered one): 1/5 on the 26yr, 0/5 on the 8yr, no income-vs-gross
   split, family mean negative on both horizons.** By the criterion I wrote in advance, that is
   **LUCKY DRAW**, not mechanism.
2. **Two-horizon bar: −0.160 Sharpe at 2/12 on the 8yr.** Not marginal — catastrophic.
3. **Matched exposure: −0.003 (26yr) and −0.175 at 0/12 (8yr).** It never beat constant leverage.

**This is the most instructive kill of the program.** ROE passed a clean shift test, survived
3× costs, held its sign across four pool sizes, and reached 10/12 on the 26yr. **A signal can be
leak-free, cost-robust, sign-consistent on one horizon, and still be noise.** Nine arms were
tested in EXP-021; one came back positive; four sibling metrics measuring the same construct
came back negative. That is what one lucky draw out of nine looks like, and only the family test
and the second horizon could see it.

**IDEAS I-30 answered: (a), the lucky draw.** No follow-up. `roe` stays out of the momentum pool.
EXP-028 (does ROE add on top of overlay-removal?) is now moot and is **removed from the queue**
rather than run — there is nothing left to add.

---

## Cycle 28 — EXP-027 · COMBINED-ARM AUDIT GATE, both horizons — **PASSES on everything but one bar**

Arm **E = tranching K=4 + `vol_scaling` OFF + constant 1.00×**, vs the deployed system, gate ON,
honest financing, live sizing. Six cells per horizon (capital × cost).

| horizon | cell | dCAGR | dSharpe | dMaxDD | +Shrp | +DD | sd ratio | **MATCHED dSharpe** | matched dMaxDD |
|---|---|---|---|---|---|---|---|---|---|
| 26yr | $50k ×1 | +0.08pp | +0.074 | +7.74pp | 10/12 | 12/12 | 0.397 | **+0.035** | −0.11pp |
| 26yr | $50k ×3 | +0.87pp | +0.075 | +8.35pp | 10/12 | 12/12 | 0.398 | **+0.032** | +0.00pp |
| 26yr | **$33k ×1** | +0.12pp | +0.074 | +7.86pp | 10/12 | 12/12 | 0.398 | **+0.035** | +0.03pp |
| 26yr | $33k ×3 | +0.90pp | +0.076 | +8.46pp | 10/12 | 12/12 | 0.399 | **+0.033** | +0.21pp |
| 8yr | $50k ×1 | +0.31pp | +0.096 | +4.76pp | 8/12 | 9/12 | 0.555 | **+0.079** | +0.36pp |
| 8yr | $50k ×3 | +0.89pp | +0.097 | +4.67pp | 8/12 | 9/12 | 0.552 | **+0.076** | +0.04pp |
| 8yr | **$33k ×1** | +0.41pp | +0.097 | +3.95pp | 8/12 | 8/12 | 0.537 | **+0.084** | −0.23pp |
| 8yr | $33k ×3 | +1.02pp | +0.098 | +4.16pp | 8/12 | 8/12 | 0.556 | **+0.081** | −0.37pp |

**Every attack passed:**
- **Real account size:** $33k ≡ $50k to two decimals in every cell. The constraint I most
  expected to kill this does not bite.
- **Cost:** dCAGR **improves** with cost (26yr +0.08 → +0.87pp; 8yr +0.31 → +0.89pp) and dSharpe
  is flat. The combined arm trades less, so higher costs favour it — the same mechanism that
  validated overlay-removal, now confirmed on the combination.
- **Matched exposure — the decisive one:** **+0.032 to +0.035 (26yr), +0.076 to +0.084 (8yr).**
  Positive in all twelve cells. EXP-008 died here at −0.099; this does not.
- **Stability:** sd ratio 0.397-0.399 (26yr) and 0.537-0.556 (8yr) across every capital and cost.

**The one bar it misses: Sharpe sign consistency on the 8yr is 8/12, against my ≥9/12 rule.**
(26yr is 10/12.) I am not relaxing the rule. It is **marginal-to-strong, not a clean pass.**

**And the honest decomposition stands:** matched dMaxDD is ≈0 in every cell (−0.37 to +0.36pp),
so the headline drawdown improvement remains a **level effect** — what any book at ~0.99× gross
instead of ~1.22× would get. What is exposure-independent is the **Sharpe residual and the
dispersion collapse**.

**The claim, stated precisely:** this arm delivers the drawdown of running ~1.0× leverage **at no
CAGR cost** (+0.08 to +1.02pp, improving with cost), whereas plain de-levering costs −2.60pp
(EXP-010) — plus **+0.03 to +0.08 of exposure-independent Sharpe** and a **~60% cut in outcome
dispersion**. **Concentration reads n/a**: the guard (BUGS A8a) refuses to report a share when
net excess is small relative to gross daily movement — the arms diverge slowly and steadily,
which is consistent with variance reduction, but it is **not a passed concentration test.**

---

## Cycle 29 — EXP-026 · MID-CYCLE SIGNAL EXIT (I-29) — **a real drawdown instrument that costs Sharpe**

`signal_exit_every=N`: sell a holding as soon as no sleeve still wants it, instead of waiting out
the 20-session cadence. EXP-021 had flagged it with the largest single drawdown improvement in
the program (+11.39pp). This runs the cadence sweep with the matched-exposure control.

**26yr, 12 starts, gate ON. Base +12.57% / 0.552 / −55.5% (worst −63.7%), gross 1.1821.**

| arm | CAGR | Sharpe | MaxDD | worst | gross | dCAGR | dSharpe | dMaxDD | +Shrp | +DD | **MATCHED dSharpe / dMaxDD** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exit 3d | +10.28% | 0.550 | **−46.1%** | −54.2% | 1.118 | −2.29pp | −0.001 | **+9.49pp** | 6/12 | 11/12 | **−0.037 / +7.84pp** |
| exit 5d | +11.23% | 0.568 | −48.2% | −56.3% | 1.134 | −1.34pp | +0.016 | **+7.39pp** | 8/12 | 11/12 | **−0.018 / +6.31pp** |
| exit 10d | +12.07% | 0.568 | −49.7% | −57.5% | 1.155 | −0.50pp | +0.017 | **+5.89pp** | 7/12 | 10/12 | **−0.016 / +5.56pp** |
| exit 5d grace10 | +11.11% | 0.553 | −48.3% | −55.9% | 1.136 | −1.46pp | +0.001 | +7.23pp | 8/12 | 11/12 | −0.033 / +6.22pp |

### This is the OPPOSITE profile to the combined arm, and the contrast is the finding

For the combined arm (EXP-027) the matched control said: **Sharpe gain real, drawdown gain a
level effect.** Here it says exactly the reverse:

- **matched dMaxDD = +5.56 to +7.84pp** — the drawdown gain **SURVIVES exposure matching**. It is
  not merely "holding less". Exiting a name the model has abandoned genuinely protects the tail.
- **matched dSharpe = −0.016 to −0.037** — negative at every cadence. On a risk-adjusted basis,
  exposure-matched, it **loses**.

So signal-exit is a *genuine* tail-risk instrument that is *paid for* in Sharpe and CAGR
(−0.50 to −2.29pp). A clean monotone trade: shorter cadence → more drawdown protection → more
CAGR given up (3d: +9.49pp DD for −2.29pp CAGR; 10d: +5.89pp for −0.50pp).

**Verdict: NOT PROMOTED, but not dead either — it is a preference instrument, not an improvement.**
It fails the Sharpe bar (6-8/12, negative matched) so it cannot be called a win. But it is the
only mechanism found in this program whose drawdown benefit is *exposure-independent*, which
makes it the right tool if a drawdown-first mandate is ever chosen — and the wrong tool for
maximising risk-adjusted return.

**Mechanism confirmed as stated in I-29:** EXP-007 showed re-picking the book mid-cycle is
catastrophic (−7 to −10pp) while merely *exiting* is cheap. That distinction holds: exit-only
costs 0.5-2.3pp, an order of magnitude less than re-picking, and buys real tail protection.

**The grace parameter adds nothing** (grace10 ≈ plain 5d on drawdown, worse on Sharpe), so
tolerating names still inside the momentum top-K is not the useful part.

---

## Cycle 30 — EXP-030 · CAP MECHANICS + MOMENTUM TILT (I-02, I-27) — one clean dose-response

**8yr, 12 starts, gate ON. Base +23.02% / 0.802 / −37.7%, gross 1.1077.**

| arm | CAGR | Sharpe | MaxDD | gross | dCAGR | dSharpe | dMaxDD | +CAGR | +Shrp |
|---|---|---|---|---|---|---|---|---|---|
| cap waterfill | +23.41% | 0.808 | −37.9% | **1.1332** | +0.39pp | +0.005 | −0.15pp | 10/12 | 7/12 |
| cap 0.15 (looser) | +24.48% | 0.818 | −40.1% | 1.1034 | +1.45pp | +0.016 | −2.36pp | 10/12 | 9/12 |
| cap 0.07 (tighter) | +21.79% | 0.800 | −34.8% | 1.0984 | −1.24pp | −0.003 | +2.89pp | 2/12 | 5/12 |
| tilt 40/42/18 (**inverse**) | +21.86% | 0.784 | −37.4% | 1.1366 | **−1.17pp** | **−0.018** | +0.34pp | 1/12 | 2/12 |
| *base 50/35/15* | *+23.02%* | *0.802* | *−37.7%* | *1.1077* | — | — | — | — | — |
| tilt 60/28/12 | +23.93% | 0.815 | −37.1% | 1.0739 | +0.91pp | +0.013 | +0.61pp | 11/12 | 8/12 |
| tilt 70/21/9 | +24.70% | 0.826 | −36.3% | 1.0350 | +1.67pp | +0.023 | +1.39pp | 10/12 | 8/12 |
| tilt 80/14/6 | +24.91% | 0.827 | −35.6% | 0.9882 | **+1.89pp** | **+0.024** | +2.12pp | 8/12 | 7/12 |

### I-02 CAP MECHANICS — the docstring caveat is REAL but immaterial

`cap_mode="waterfill"` raises realised gross **1.1077 → 1.1332** (+2.3%), which proves the 10%
cap **does** bind and the deployed code really does discard the truncated excess, leaving the
book under-invested on those days. Worth **+0.39pp CAGR / +0.005 Sharpe** to fix — real, and too
small to matter. **The long-standing docstring caveat can be closed as measured-and-negligible**
rather than left as an open unknown. (It is also a live-vs-backtest parity item: the live engine's
closed-loop sizing effectively redistributes, so the *backtest* was the under-invested one.)

### I-27 MOMENTUM TILT — a clean monotone dose-response THROUGH the baseline

CAGR runs −1.17 → *0* → +0.91 → +1.67 → +1.89pp and Sharpe −0.018 → *0* → +0.013 → +0.023 →
+0.024 as capital moves from value/lowvol into momentum. **Monotone in both directions**, with
the inverse tilt clearly worse. That is a dose-response, not a lucky cell, and it independently
confirms EXP-003: **the ranking carries 100% of the selection edge and momentum is the ranked
sleeve** — value and lowvol behave close to random draws from a filtered pool, so capital in them
is capital not earning the one edge the system has.

**My pre-registered fear was wrong.** I predicted "momentum is the highest-vol sleeve, so every
tilt raises drawdown — expect this to fail the drawdown test." Drawdown **improves** monotonically
too (+0.61 → +1.39 → +2.12pp). But the reason is mundane: **avgGross falls monotonically**
(1.1077 → 1.0350 → 0.9882) because the momentum sleeve holds only 5 names, so the 10% cap binds
harder and the book ends up less invested. The drawdown gain is largely that level effect again.
What is NOT a level effect is the CAGR: it **rises while gross falls**, which leverage cannot
explain.

**NOT PROMOTED — 7-8/12 on Sharpe, below the ≥9/12 bar, and 26yr not yet in.** Sleeve-weight
changes have been retracted as noise in this repo before (CORE2/3/4 at 6/12), which is exactly
why the dose-response matters more here than any single cell. 26yr queued.

---

## Cycle 31 — EXP-023 · MOMENTUM UNIVERSE TIER (I-28) — **DEAD, and the size story is inverted**

**8yr, 12 starts, gate ON. Base (SP1500) +23.02% / 0.802 / −37.7%.**

| momentum-sleeve universe | CAGR | Sharpe | MaxDD | dCAGR | dSharpe | +CAGR | +Shrp |
|---|---|---|---|---|---|---|---|
| sp500 only (**correctness check**) | +13.78% | 0.616 | −34.9% | **−9.24pp** | **−0.187** | 1/12 | 1/12 |
| sp500+sp400 | +14.86% | 0.635 | −37.3% | −8.16pp | −0.167 | 1/12 | 1/12 |
| sp400 only (mid) | +8.47% | 0.431 | −41.0% | **−14.56pp** | **−0.372** | 0/12 | 0/12 |
| sp600 only (small) | **+25.58%** | 0.786 | −39.8% | **+2.56pp** | **−0.016** | 7/12 | 5/12 |
| sp400+sp600 (mid+small) | +18.43% | 0.675 | −40.0% | −4.59pp | −0.127 | 0/12 | 0/12 |

**The built-in correctness check passes.** sp500-only returns −9.24pp CAGR / −0.187 Sharpe,
reproducing the known B1 live bug (measured −15.7pp / −0.38 on a clean 8yr A/B against a
different baseline config that also restricted the lowvol sleeve). Same sign, same order of
magnitude — the hook is wired correctly, so the rest of the table is trustworthy.

**Every restriction loses.** Small-cap-only is the only arm with positive CAGR (+2.56pp) and even
it is **negative on Sharpe (−0.016) at 5/12** — the extra return is fully paid for in volatility
and 2.05pp more drawdown.

**The mechanism I proposed is not supported.** I argued momentum's premium should be larger in
smaller, less-covered names, and that our $33k size uniquely lets us harvest it. The data says
the **breadth of the selection pool matters more than its size tier**: SP1500 beats every subset,
including combinations. Mid-caps alone are catastrophic (−14.56pp), which no size-premium story
predicts.

**Reading it correctly:** the momentum ranking works by finding the best 5 names out of ~1,100
eligible. Any restriction shrinks that pool and the top-5 gets worse — and that cost exceeds
whatever size premium exists. Consistent with EXP-012 (concentration is load-bearing) and with
B1's original −15.7pp. **The pool should be as wide as the data allows.**

**Generated → I-32:** the one direction not yet tested is a pool WIDER than SP1500 (Russell 3000).
The gradient says wider is better, and the repo has `expanded_r3000_universe.pkl` — but that file
is **not in the audited set** (`UNIVERSE_MANIFEST` lists it research-only, built pre-audit), so
using it would reintroduce exactly the survivorship and look-ahead defects the 2026-07 rebuild
fixed. **Blocked on a clean R3000 build, not on compute.**

---

## Cycle 32 — EXP-029 · ALTERNATIVE CREDIT-GATE SIGNALS (I-31) — **all dead; the deployed gate is not improvable from this data**

Escalation-ladder level 9. The deployed gate uses HY-OAS *level* ≥ p95 expanding percentile and
EXP-022 measured it at +0.61pp CAGR / +0.021 Sharpe / +8.54pp MaxDD, 12/12 on all three — the
single most valuable risk control in the system. The question: is the aggregate spread level the
best available credit signal, or merely the first one tried?

**Implementability was checked BEFORE building**, which saved testing a signal that could never
ship:
- `tedrate` (LIBOR−T-bill funding stress): only **0.348** correlated with hy_oas — genuinely
  independent — but the series **ends 2022-01-21 because LIBOR was discontinued.** Not
  computable live. Included as a research-only bound.
- `ccc_bb` (CCC-and-lower yield − BB yield, the quality spread *inside* high yield): **0.779**
  correlated, so ~22% new information, and both legs are still published daily by FRED.
  Mechanism: aggregate HY-OAS rises both for benign broad repricing and for distress
  *concentrating* in the worst credits; `ccc_bb` isolates the second, which is what precedes
  defaults and forced selling.

**8yr, 12 starts, vs the DEPLOYED hy_oas p95 gate:**

| arm | CAGR | Sharpe | MaxDD | dCAGR | dSharpe | +Shrp | **matched dSharpe** |
|---|---|---|---|---|---|---|---|
| ccc_bb p95 | +22.02% | 0.779 | −40.0% | −1.01pp | −0.024 | 1/12 | −0.044 |
| ccc_bb p90 | +22.04% | 0.781 | −39.9% | −0.98pp | −0.022 | 2/12 | −0.042 |
| OR(hy_oas, ccc_bb) | +22.31% | 0.789 | −38.7% | −0.71pp | −0.014 | 3/12 | −0.034 |
| tedrate p95 *[not deployable]* | +22.69% | 0.791 | −39.2% | −0.33pp | −0.011 | 0/12 | −0.028 |
| OR(hy, ted) *[not deployable]* | +22.96% | 0.801 | −37.9% | −0.06pp | −0.002 | 0/12 | −0.019 |

**Every arm is negative, at 0-3/12, with negative matched-exposure residuals.** Neither replacing
the deployed signal nor OR-ing a second one with it helps. Even the genuinely-independent funding
signal (tedrate, corr 0.35) adds nothing — which is the more informative result, because it was
the best case: if an orthogonal stress measure cannot improve on HY-OAS, the gate is not
signal-starved.

**Caveat, stated: the 8yr contains no genuine credit crisis** (2020 was brief), so this is the
weak horizon for a crisis gate and the 26yr is queued. But the direction is uniform across five
arms and two independent signals.

**Verdict: the deployed HY-OAS p95 gate is not improvable from the credit data available
locally.** That is worth knowing — it closes level 9 for credit and says the gate should be left
alone rather than tuned.

---

## STILL RUNNING at hand-off (serial queue, one job at a time under the memory guard)

`EXP030_cap_and_tilt.py 26yr` · `EXP023_universe_tier.py 26yr` · `EXP029_new_stress_gates.py 26yr`
· `EXP026_signal_exit.py 8yr`. Each is a second-horizon confirmation of a result already logged
above; none can promote a candidate on its own, and none of the conclusions in BASELINE.md
depends on them.

---

## Cycle 33 — EXP-031 · THE LEVERAGE FRONTIER — **Option 2 shifts the whole frontier**

Options 1 and 2 had each been tested at one or two leverage settings chosen to roughly match the
deployed book's realised gross. That answered "is the change good at the exposure we already
run", not "what exposure should we run it at". This runs the full curve.

**Critically, the arms run DIFFERENT gross at the same nominal leverage** — removing the overlay
*raises* gross (no more de-risking), tranching *lowers* it slightly. So comparing at equal
nominal leverage is meaningless; the honest comparison is **CAGR at matched DRAWDOWN**.

**8yr, 12 starts, $33,000 (the real account), gate ON, real financing, live sizing:**

| arm | lev | avgGross | CAGR | sd CAGR | Sharpe | MaxDD |
|---|---|---|---|---|---|---|
| A baseline | 1.00 | 0.781 | +18.04% | 4.94pp | 0.832 | −27.3% |
| A baseline | **1.49 (deployed)** | 1.174 | **+24.12%** | 7.34pp | **0.805** | **−38.4%** |
| A baseline | 2.00 | 1.489 | +26.43% | 9.08pp | 0.766 | −47.5% |
| B option 1 | 1.00 | 0.994 | +22.29% | 6.10pp | 0.827 | −34.1% |
| B option 1 | 1.49 | 1.481 | +28.17% | 8.74pp | 0.791 | −48.4% |
| B option 1 | 2.00 | **1.690** | +30.13% | 9.32pp | 0.789 | −53.3% |
| **C option 2** | 1.00 | 0.995 | +24.53% | **3.94pp** | **0.902** | −34.4% |
| **C option 2** | 1.10 | 1.093 | +26.24% | 4.32pp | 0.893 | −37.4% |
| **C option 2** | 1.25 | 1.241 | +28.83% | 4.98pp | 0.885 | −41.7% |
| **C option 2** | 1.75 | 1.594 | **+33.27%** | 6.11pp | 0.871 | −50.2% |

### ISO-DRAWDOWN — the only fair comparison

| target MaxDD | A baseline | B option 1 | **C option 2** |
|---|---|---|---|
| −35% | +22.34% | +22.65% | **+24.85%** |
| −40% | +24.59% | +24.86% | **+27.79%** |
| −45% | +25.92% | +26.92% | **+30.27%** |
| −50% | *(off curve)* | +28.79% | **+33.15%** |

**At every drawdown level, Option 2 delivers +2.5 to +4.4pp more CAGR than the baseline**, and
Option 1 delivers +0.3 to +1.0pp more. The frontier moves, it does not merely slide along itself.
That is a stronger statement than any single-point comparison and it is what justifies calling
these improvements rather than repackaged leverage.

### Two incidental findings

**1. Gross SATURATES for Option 1 at ~1.69×.** Nominal 1.75 → avgGross 1.6834; nominal 2.00 →
1.6900, and CAGR barely moves (+29.94% → +30.13%). The 10% position cap binds hard enough that
raising nominal leverage past ~1.75 buys almost nothing. **Anyone tempted to "just add leverage"
should know it stops working there**, and the sd of outcomes keeps rising while return does not.

**2. Option 2's sd CAGR is roughly HALF the baseline's at every leverage** (3.94-6.11pp vs
4.94-9.08pp). The dispersion reduction is not a low-leverage artefact — it holds across the
whole curve, which is what a genuine variance-reduction mechanism should do.

**26yr frontier queued.** The 8yr is the momentum-friendly horizon and will flatter high
leverage; the 26yr (dot-com + GFC + COVID + 2022) is the one to decide on.

---

## Cycle 34 — EXP-031 26yr · THE LEVERAGE FRONTIER, full-cycle horizon — **the decision table**

$50,000 (the live account is now ~$50k, up from $33k), 12 starts, gate ON, real financing,
live sizing. EXP-027 had already measured $33k ≡ $50k to 0.1pp, and the 8yr rerun confirmed it
(+24.19% vs +24.53% at 1.00×), so account size is not a driver.

| arm | lev | avgGross | CAGR | sd CAGR | Sharpe | MaxDD | worst DD |
|---|---|---|---|---|---|---|---|
| A baseline | 1.00 | 0.817 | +9.93% | 1.97pp | **0.569** | **−40.7%** | −48.8% |
| A baseline | 1.25 | 1.021 | +11.52% | 2.55pp | 0.558 | −49.1% | −57.3% |
| A baseline | **1.49 (deployed)** | 1.221 | **+12.72%** | 3.05pp | **0.549** | **−55.9%** | **−64.2%** |
| A baseline | 2.00 | 1.548 | +14.41% | 3.97pp | 0.548 | −65.4% | −73.5% |
| B option 1 | 1.00 | 0.988 | +12.27% | 2.26pp | 0.592 | −48.1% | −59.7% |
| B option 1 | 1.25 | 1.235 | +13.75% | 2.88pp | 0.572 | −57.1% | −68.9% |
| B option 1 | 1.75 | 1.686 | **+15.66%** | 3.88pp | 0.560 | −69.8% | −80.6% |
| B option 1 | 2.00 | 1.700 | +15.61% | 3.89pp | 0.558 | −71.9% | −81.8% |
| **C option 2** | **1.00** | 0.987 | +12.79% | **1.21pp** | **0.623** | −48.2% | **−51.4%** |
| **C option 2** | **1.10** | 1.086 | +13.48% | 1.35pp | 0.612 | −52.0% | −55.4% |
| **C option 2** | **1.25** | 1.234 | +14.39% | 1.59pp | 0.599 | −57.3% | −60.7% |
| **C option 2** | 1.49 | 1.471 | +15.53% | 2.01pp | 0.586 | −64.2% | −67.7% |
| **C option 2** | 1.75 | 1.673 | **+16.43%** | 2.35pp | 0.583 | −69.7% | −73.4% |

### ISO-DRAWDOWN — CAGR delivered at the SAME drawdown (26yr)

| target MaxDD | A baseline | B option 1 | **C option 2** | O2 − base |
|---|---|---|---|---|
| −45% | +10.80% | *(off curve)* | *(off curve)* | — |
| −50% | +11.68% | +12.60% | **+13.12%** | **+1.44pp** |
| −55% | +12.56% | +13.42% | **+14.00%** | **+1.44pp** |
| −60% | +13.37% | +14.19% | **+14.84%** | **+1.47pp** |

**Option 2 delivers +1.44 to +1.47pp more CAGR at every drawdown level, and the gap is almost
perfectly constant** — the frontier is translated upward, not rotated. Option 1 captures about
60% of that (+0.86 to +0.92pp). The 8yr showed the same shape at larger magnitude
(+2.5 to +4.4pp), consistent with the 8yr being the momentum-friendly window.

### 🔴 THE OVERLAY BUYS SOMETHING CONSTANT LEVERAGE CANNOT — a real caveat against my own conclusion

At the **−45%** row only the BASELINE is on-curve. Options 1 and 2 cannot reach −45% **at any
leverage tested**, because their minimum realised gross is ~0.99× (leverage 1.00 with no
overlay), which floors their drawdown at −48.1%/−48.2%.

The deployed vol overlay scales gross **below** 1.0× in high-vol regimes (clamp floor 0.30), so
it can reach exposure territory a constant-leverage book cannot. **That is a genuine capability
of the overlay that every "the overlay is harmful" statement in this log understates**: it is
harmful *at matched exposure*, but it is the only mechanism tested that accesses sub-1.0× gross.

**This is a gap in my testing, not a defence of the overlay.** The obvious fix was never run:
Options 1 and 2 at leverage **below** 1.00 (0.70 / 0.85). If constant 0.85× reaches −45% with
better CAGR than the baseline does, the caveat dissolves; if it cannot, the overlay has a real
role at low target risk. **Recorded as I-33, untested.**

### Leverage saturation, confirmed on both horizons

Option 1 nominal 1.75 → 2.00 makes CAGR **fall** (+15.66% → +15.61%) while drawdown worsens
−69.8% → −71.9%. The 10% position cap binds and extra borrowing only adds financing cost.
**Above ~1.75× nominal you pay drawdown for nothing.**

### Sharpe still declines with leverage on every arm

0.623 → 0.583 for Option 2, 0.592 → 0.558 for Option 1, 0.569 → 0.548 for the baseline.
Sharpe-optimal is **1.00× on all three**. Choosing higher leverage is a deliberate
CAGR-for-drawdown trade, not an optimisation — and the worst-DD column is the one to read when
making it (Option 2 at 1.49× has a worst start of −67.7%).

---

## Cycle 35 — EXP-032 · JOINT-TRANCHE FIDELITY — **the published tranching numbers are slightly
## OPTIMISTIC. Found before writing live code, not after.**

**The gap I found while designing the engine.** Every tranching number in this program
(EXP-001, -014, -019, -027, -031) came from running K sub-books as SEPARATE backtests at
capital/K and summing the curves. Those sub-books never share capital — a lucky phase stays big
for 26 years. **The live engine cannot work that way:** all K sub-books sit in one margin
account, so a rebalancing tranche necessarily sizes off NAV/K of the *current total* NAV. That
is continuous reallocation between tranches.

Cross-phase 26yr CAGR σ is 1.94pp, compounding to ~1.6× between best and worst tranche, so the
two designs are genuinely different products. Quoting one while shipping the other is the
backtest-vs-live divergence class in BUGS section B — the one that cost −15.7pp when the
momentum sleeve was silently SP500-only.

**Built `research/livemirror_tranche.py`** — K virtual ledgers with per-tranche trailing-stop
peaks (entry dates differ, so stops fire at different times), one shared cash balance and margin
debit, one tranche rebalancing every `rebal_days/K` sessions, each sizing off shared-NAV/K
through the same closed-loop integer-share calibration the live engine runs.

### Two parity gates, and the first version of gate 2 was a BAD TEST

- **Gate 1** — `tranches=1` delegates to the validated parent: **IDENTICAL** (236,634.08). Zero
  regression risk. But this gate never enters the new code, so it validates nothing about it.
- **Gate 2, first attempt** — K=4 with `stride = rebal_days`, expecting "one book split 4 ways":
  came back **−53%** and looked like a machinery bug. It was a **bad gate**: with stride=20 and
  K=4, each tranche fires every K×stride = **80** sessions, not 20, so the book is 4× stale. The
  comparison was meaningless. Recorded because a wrong test produced a number that looked like a
  catastrophic bug.
- **Gate 2, corrected** — added `force_joint` so K=1 runs *through* the joint loop instead of
  delegating; with stride=20 that is by construction one book on the deployed cadence, so it must
  reproduce the parent. **236,858.59 vs 236,634.08 = +0.09%.** The reimplementation is faithful.

### Result — 8yr, 12 starts

| arm | mean CAGR | sd CAGR | Sharpe | Sortino | MaxDD | worst DD |
|---|---|---|---|---|---|---|
| **A summed-independent** (what I published) | **+24.19%** | 3.92pp | **0.898** | 1.325 | −33.8% | −37.1% |
| **B joint-reallocated** (what the engine does) | **+23.64%** | 4.20pp | **0.887** | 1.303 | −33.9% | −37.4% |

**B − A: −0.54pp CAGR, −0.0110 Sharpe, −0.19pp MaxDD, at 2/12 and 3/12** — i.e. the joint design
is worse in 10/12 starts on CAGR and 9/12 on Sharpe. Small, but **sign-consistent, not noise**.

My pre-registered threshold was |dSharpe| < 0.010 = faithful. **It came in at −0.0110 — just
over.** I am reporting it as a miss rather than rounding it to "close enough".

**My hypothesis was wrong in direction.** I wrote that reallocation "may add a small rebalancing
bonus" by moving capital from lucky phases to unlucky ones. It does not — it costs. Most likely
extra turnover: a rebalancing tranche sized off the *current* shared NAV must trade further than
one compounding its own capital, and that trades against a drifted book every cycle.

### 🔴 CONSEQUENCE — the 8yr headline changes sign on CAGR

Deployed 8yr baseline is **+23.88%**. The summed version of Option 2 showed **+24.19% (+0.31pp)**.
The live-faithful version is **+23.64% (−0.24pp)**.

**So on the 8yr, Option 2 as it can actually be built gives slightly LESS CAGR than the deployed
system, not more.** What survives on that horizon is the risk side: Sharpe 0.887 vs 0.802
(**+0.085**) and MaxDD −33.9% vs −38.5% (**+4.6pp**).

Every Option-2 CAGR figure previously quoted should be read **−0.54pp**, and every Sharpe
**−0.011**. That does not overturn the finding — the Sharpe and drawdown gains dominate the
correction — but it does remove the "free CAGR" framing on the 8yr.

### 26yr — the correction does NOT carry, and the decision horizon says FAITHFUL

Gate 1 IDENTICAL (758,728.67). Gate 2 joint-as-one-book **+0.15%**.

| arm | mean CAGR | sd CAGR | Sharpe | Sortino | MaxDD | worst DD |
|---|---|---|---|---|---|---|
| A summed-independent | +12.79% | 1.21pp | 0.623 | 0.888 | −48.2% | −51.4% |
| **B joint-reallocated** | **+12.83%** | **1.16pp** | **0.625** | 0.890 | −48.3% | −51.8% |

**B − A: +0.04pp CAGR, +0.0020 Sharpe, −0.15pp MaxDD, 9/12 and 9/12.**
**|dSharpe| = 0.002, well inside the pre-registered 0.010 threshold → FAITHFUL.**

**So the correction is horizon-specific, and it does not affect the horizon that matters:**

| | dCAGR (B−A) | dSharpe (B−A) | verdict |
|---|---|---|---|
| 8yr | **−0.54pp** | **−0.0110** | just outside threshold |
| **26yr** | **+0.04pp** | **+0.0020** | **faithful** |

On the full-cycle horizon the live design is if anything *marginally better* than the summed
version (9/12 on both), and its dispersion is slightly lower (1.16pp vs 1.21pp). **The 26yr
tranching numbers stand as published.** Only the 8yr figures need the −0.54pp / −0.011 haircut.

**Why the difference is credible rather than convenient:** the reallocation drag is a turnover
cost, and turnover cost is roughly constant per year while the 8yr horizon has ~3× fewer years
to amortise the compounding benefit of a lower-dispersion path. Over 26 years the variance
reduction (sd 1.16 vs 1.21) compounds enough to offset the extra trading. That is a story fitted
after the fact and is labelled as such — but the *measurement* is 9/12 sign-consistent on both
metrics, which is not.

**Net effect on the recommendation:** Option 2's 26yr case is unchanged. Its 8yr CAGR edge is
gone (−0.24pp vs baseline) while its 8yr Sharpe (+0.085) and drawdown (+4.6pp) edges remain.
**Option 2 is a risk improvement on the 8yr and a both-axes improvement on the 26yr.**

---

## Cycles 36-38 — IMPROVING OPTION 2 (EXP-034/035/036/037)

### The gap the year-by-year exposed, and what was aimed at it
EXP-033: Option 2 @1.00× has drawdown better in **25/25 years** but **no return edge** (13/25,
arithmetic sum −20.4%); its equal CAGR comes from compounding a smoother path. At 1.25× the
return appears but crisis drawdown goes worse (2008 −4.7pp, 2020 −2.9pp). So the leverage dial
trades away the one thing it is good at. Target: **the smoothness of 1.00× with the return of
1.25×.**

### Idea 1 — MORE TRANCHES: monotone but SATURATING

| K | sd CAGR | Sharpe |
|---|---|---|
| LIVE | 3.05% | 0.549 |
| 2 | 1.31% | 0.618 |
| **4** | **1.16%** | **0.625** |
| 5 | 1.04% | 0.622 |
| 10 | 1.01% | 0.623 |

The falsifiable shape held — sd falls monotonically, confirming the variance mechanism is real
rather than a K=4 fluke. **But it saturates:** K=5→10 buys 0.03pp of sd for 2.5× the operational
complexity, and Sharpe is flat throughout. **K=4 is the right build.** Useful negative: no reason
to engineer 10 sub-books.

### Idea 2 — HARDER CREDIT GATE: the actual win

Replace the vol overlay's permanent ~1pp/yr tax with a conditional one that only pays when credit
is stressed. Monotone in derisk on the 26yr:

| derisk @1.25× | CAGR | Sharpe | MaxDD | crisis ddn |
|---|---|---|---|---|
| 0.50 | +14.46% | 0.602 | −57.4% | −34.1% |
| 0.30 | +14.74% | 0.612 | −52.8% | −33.3% |
| **0.20** | **+14.87%** | **0.616** | **−51.2%** | **−32.8%** |

Cutting *harder* raises return AND lowers drawdown — avoiding more of a crash compounds better.
Not a fresh fit: the DYN walk-forward independently chose derisk 0.30 in all 22 OOS years.

### Idea A — MOMENTUM TILT: aggregate looked great, YEAR-BY-YEAR KILLED IT at 1.00×

| tilt @1.00× | CAGR | Sharpe |
|---|---|---|
| 50/35/15 | +12.83% | 0.625 |
| 60/28/12 | +13.11% | 0.622 |
| 70/21/9 | +13.44% | 0.619 |
| 80/14/6 | +13.82% | 0.615 |

Clean monotone dose-response across four levels, matching the independent 8yr result in EXP-030.
**And `tilt80 @1.00` FAILED the year-by-year: 11/25 years won, and the excess DIES on dropping
its best year (−1.6%) and collapses on the best two (−12.3%).** Carried by 2020 (+16.0%) and
2005 (+10.7%).

**⚠️ Lesson recorded: a monotone dose-response is NOT protection against a one-year artefact.**
I had been treating that shape as strong evidence. It is evidence the *parameter* behaves
sensibly, not that the *edge* is broad. Only the year-by-year separated them.

`tilt70 @1.25` DID pass (15/25 years, +25.5% after dropping the best two), so the tilt is not
dead — it is only validated at the leverage where it was tested.

### Idea B — signal exit: VOID, never executed (my bug)

`signal_exit_every=5` with `tranche_stride=5` gates on the same days, so the exit could **never
fire**: 0 of 200 days. All three arms returned numbers **byte-identical** to the no-exit
reference (+14.46% / 0.602 / −57.4%). Caught by the suspicious-roundness heuristic — the same
tell that caught `park idle cash IEF` in EXP-021. Fixed with a guard that now RAISES if the exit
cadence is coarser than the stride, instead of silently doing nothing.

### EXP-037 — the matched-exposure result

Two arms landed at the SAME realised gross as LIVE (1.221), making these true like-for-like
comparisons rather than leverage re-picks:

| arm | gross | CAGR | Sharpe | MaxDD | crisis |
|---|---|---|---|---|---|
| **LIVE** | **1.221** | **+12.72%** | **0.549** | **−55.9%** | **−33.1%** |
| O2 @1.25 d0.20 | **1.218** | +14.87% | 0.616 | −51.2% | −32.8% |
| O2+T70 @1.25 d0.20 | **1.212** | **+15.47%** | 0.618 | −52.7% | −33.0% |

**+2.15pp / +2.75pp of CAGR at identical exposure, with better Sharpe and better drawdown.**

**Best arm on BOTH horizons — `O2 + tilt70 + d0.20 @1.00×`:**

| | dCAGR | dSharpe | dMaxDD | crisis | gross vs LIVE |
|---|---|---|---|---|---|
| 26yr | +1.05pp | +0.083 | +11.6pp | −27.1% vs −33.1% | 0.974 vs 1.221 |
| 8yr | +2.26pp | +0.107 | +5.4pp | −24.8% vs −30.3% | 0.979 vs 1.175 |

Better on all four, both horizons, at ~20% LESS exposure.

**Horizon disagreement noted:** the harder gate alone helps on the 26yr (+12.83→+13.17%) and
hurts on the 8yr (+23.64→+23.37%) — 2018-25 has no genuine credit crisis, so cutting harder is
pure cost there. The combination with the tilt is positive on both.

### Outstanding — NOT yet validated
- **d0.10 / d0.00 edge check:** on the 8yr all three are identical (+27.27 / +27.25 / +27.18),
  but that horizon **cannot discriminate** — the gate barely fires, so every derisk converges.
  The 26yr is the discriminating run and is queued.
### 🔴 EXP-036b — the year-by-year KILLED the arm that looked best in aggregate

| candidate | yrs won | ddn won | sum diff | drop best | **drop 2** | crisis avg | verdict |
|---|---|---|---|---|---|---|---|
| **O2+T70 @1.00 d0.20** | 12/25 | **24/25** | +7.4% | **−8.2%** | **−17.0%** | +7.3% | **FAIL** |
| **O2 @1.00 d0.20** | 15/25 | **25/25** | −12.8% | −27.3% | −37.4% | +7.0% | **FAIL** |
| O2+T70 @1.10 d0.20 | 13/25 | 22/25 | +38.5% | +21.6% | **+5.7%** | +6.0% | PASS |
| **O2+T70 @1.25 d0.20** | 15/25 | 14/25 | **+77.2%** | +52.8% | **+35.4%** | +4.4% | **PASS** |

**`O2+T70 @1.00 d0.20` was the best arm on BOTH horizons in aggregate** (+1.05pp / +2.26pp CAGR,
+0.083 / +0.107 Sharpe, better drawdown, at 20% LESS gross). **It failed.** Its return edge dies
on dropping the best year (−8.2%) and collapses on the best two (−17.0%), with only 12/25 years
won. Identical failure mode to `tilt80 @1.00`.

**This is the second time an arm with a clean aggregate AND a clean dose-response died here.**
The pre-registered flag ("this is exactly the profile that fooled me once") was correct, and
running the test rather than trusting the aggregate is what prevented shipping it.

### The pattern is now consistent across every variant tested

**At LOW leverage (~1.00×) these are RISK products:** drawdown better in **24-25 of 25 years**,
crisis-year drawdown **+7pp** better — and **no return edge at all** (sum diff +7.4% / −12.8%,
dying on the drop-best test). Their equal-or-better CAGR comes from compounding a smoother path,
exactly as EXP-033 found for plain Option 2.

**At HIGHER leverage (1.10-1.25×) the return edge is real and survives** (+35.4% after removing
the two best years, 15/25 years) **but the drawdown advantage shrinks** from 24/25 years to 14/25.

**You cannot have both from the leverage dial.** Every attempt to buy return with leverage spends
the drawdown edge, and every attempt to bank the drawdown edge gives up the return. The harder
credit gate improved BOTH ends of that trade-off, but did not abolish it.

### EXP-038 — the edge check: derisk is MONOTONE TO THE BOUNDARY. 0.20 is NOT a tuned optimum.

**26yr, O2 @1.25×, 12 starts:**

| derisk | gross | CAGR | Sharpe | MaxDD | worst | crisis | dCAGR | dSharpe | +Shrp |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 1.227 | +14.46% | 0.602 | −57.4% | −61.2% | −34.1% | +1.74pp | +0.053 | 10/12 |
| 0.30 | 1.221 | +14.74% | 0.612 | −52.8% | −57.7% | −33.3% | +2.02pp | +0.063 | 10/12 |
| 0.20 | 1.218 | +14.87% | 0.616 | −51.2% | −55.9% | −32.8% | +2.15pp | +0.067 | 10/12 |
| 0.10 | 1.214 | +15.02% | 0.621 | −50.1% | −54.8% | −32.4% | +2.30pp | +0.072 | 10/12 |
| **0.00** | 1.211 | **+15.11%** | **0.624** | **−49.4%** | **−53.9%** | **−32.0%** | **+2.39pp** | **+0.075** | 10/12 |

**It improves monotonically all the way to 0.00 on every single axis.** So the answer is the
FIRST of the three pre-registered outcomes: **the finding is "go FLAT while credit is stressed",
not "0.20 is special".** 0.20 was simply the edge of the range I happened to test, and quoting
it as a tuned parameter would have been a fitted artefact. **Quote a range (≤0.20), or quote
0.00 as the limit — never 0.20 as an optimum.**

**8yr, same sweep:**

| derisk | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| 0.50 | +27.63% | 0.866 | −41.8% |
| 0.30 | +27.46% | 0.865 | −41.0% |
| 0.20 | +27.27% | 0.862 | −40.6% |
| 0.10 | +27.25% | 0.862 | −40.3% |
| 0.00 | +27.18% | 0.861 | −40.2% |

**The horizons disagree in DIRECTION on CAGR** — 26yr rises +14.46→+15.11%, 8yr falls
+27.63→+27.18%. But note the magnitudes: the entire 8yr range is **−0.45pp** while the 26yr range
is **+0.65pp**, and 2018-25 contains no genuine credit crisis so the gate barely fires there.
**Drawdown improves monotonically on BOTH horizons** (8yr −41.8→−40.2%, 26yr −57.4→−49.4%).

**Honest reading:** cutting to zero in credit stress is unambiguously right over a full cycle and
costs a trivial amount in a crisis-free decade. The drawdown benefit is two-horizon consistent.
This is the one parameter change in the entire program that improves every axis monotonically
to the boundary of its range.

---

## Cycles 39-41 — THE FINAL SEARCH. Winner: `REC + tilt70 @1.25×`

### Year-by-year breadth (the test that killed 2 of 2 prior look-alikes)

**26yr, 25 calendar years:**

| candidate | yrs won | ddn won | sum | drop best | **drop 2** | crisis | verdict |
|---|---|---|---|---|---|---|---|
| REC+BOTH @1.75 | 13/25 | 21/25 | +35.2% | +19.5% | +3.9% | +6.3% | PASS |
| REC+sigexit @1.75 | 15/25 | 20/25 | +34.7% | +17.4% | +1.4% | +5.2% | PASS |
| **REC+tilt @1.25** | 15/25 | 14/25 | **+82.7%** | **+58.1%** | **+41.1%** | +4.7% | **PASS** |
| REC @1.10 | 16/25 | **25/25** | +20.3% | +6.1% | **−5.0%** | +6.0% | PASS |

**8yr, 8 calendar years:**

| candidate | yrs won | ddn won | sum | drop best | drop 2 | crisis | verdict |
|---|---|---|---|---|---|---|---|
| REC+BOTH @1.75 | 4/8 | 6/8 | +12.2% | **−7.3%** | −17.0% | +14.6% | **FAIL** |
| REC+sigexit @1.75 | 5/8 | 7/8 | +15.5% | −2.4% | −9.6% | +12.6% | marginal |
| **REC+tilt @1.25** | 5/8 | 6/8 | **+29.8%** | **+14.0%** | −0.3% | +6.9% | **PASS** |
| REC @1.10 | 3/8 | 7/8 | −6.2% | −17.4% | −20.8% | +5.4% | **FAIL** |

### 🔴 COST SENSITIVITY KILLED THE SIGNAL-EXIT ARMS

dSharpe vs LIVE, paired at each multiplier (BUGS A8b):

| 26yr | ×1 | ×2 | ×3 |
|---|---|---|---|
| REC+BOTH @1.75 | +0.073 | +0.039 | **+0.005** |
| REC+sigexit @1.75 | +0.059 | +0.019 | **−0.019** |
| **REC+tilt @1.25** | +0.076 | +0.079 | **+0.081** |
| **REC @1.10** | +0.088 | +0.089 | **+0.090** |

**The signal-exit arms collapse to zero (or negative) at 3× cost** — they sell and re-buy, so
their entire advantage is a turnover subsidy. At the modelled 10bps that looks fine; at 3× it is
gone. The tilt and plain-REC arms are **flat or IMPROVING** with cost because they trade LESS
than the deployed book. Same pattern on the 8yr (+0.114 → +0.056 for BOTH; +0.078 → +0.079 for
the tilt).

**This is why EXP-040's "best all-round arm" (REC+BOTH @1.75) is NOT the answer** despite winning
on aggregates on both horizons. It failed the 8yr year-by-year AND collapsed under cost. Third
time an arm that looked best in aggregate died in validation.

### ⇒ WINNER: `REC + tilt70 @1.25×`

vol_scaling OFF · 4 tranches (5-day stride) · credit gate → 0 · sleeves 70/21/9 · leverage 1.25×

| | gross | CAGR | Sharpe | MaxDD | crisis | +Shrp |
|---|---|---|---|---|---|---|
| LIVE (26yr) | 1.221 | +12.72% | 0.549 | −55.9% | −33.1% | — |
| **WINNER (26yr)** | **1.205** | **+15.72%** | **0.625** | **−50.8%** | **−32.3%** | **10/12** |
| LIVE (8yr) | 1.175 | +23.88% | 0.802 | −38.5% | −30.3% | — |
| **WINNER (8yr)** | 1.210 | **+28.95%** | **0.880** | −39.9% | −29.9% | 7/12 |

**26yr +3.00pp CAGR / +0.076 Sharpe / +5.16pp MaxDD at MATCHED gross (1.205 vs 1.221).**
**8yr +5.07pp / +0.078 / −1.42pp.**

Why it wins where the others failed:
- **only arm to PASS year-by-year on BOTH horizons**
- **strongest breadth by far**: +41.1% still positive after removing its two best years (next
  best +3.9%)
- **cost-ROBUST**: dSharpe rises with cost on both horizons
- matched exposure on the 26yr, so not a leverage re-pick

**Weaknesses, stated:** 8yr MaxDD is −1.42pp WORSE than LIVE; 8yr drop-2 is −0.3% (marginal);
8yr sign-consistency 7/12 is below the ≥9/12 bar.

### Runner-up, and the honest alternative: `REC @1.10×` — the RISK product

26yr **+0.088 Sharpe at 11/12** (best sign-consistency in the program), **+11.58pp MaxDD**,
drawdown better in **25 of 25 years**, cost-flat. But **FAILS the 8yr year-by-year on return**
(3/8 years, −20.8% after dropping two). Pick it if the mandate is drawdown, not return.

---

## Cycle 42 — VERIFY2 · CLEAN-ROOM INDEPENDENT VERIFICATION + UNTOUCHED HOLDOUT

Every number in this program came from ONE simulator I wrote. A systematic error in it would be
inherited by every result, and no amount of start-consistency / cost / year-by-year testing would
reveal it — those all run THROUGH the same engine. The only detector is a second implementation
that shares no code.

**Built `research/VERIFY2_cleanroom.py`**: independent portfolio simulator with five deliberate
implementation differences so a shared bug cannot survive in both — NAV rebuilt from scratch each
day (never accumulated), costs charged as a separate deduction, **financing accrued on the
PREVIOUS day's debit** (a different, slightly more conservative convention), stops from an
explicit peak dict, and an up-front tranche schedule rather than a modulo. The production sleeves
are NOT reimplemented — they are what live calls, so rewriting them would test my copy of the
strategy rather than the strategy.

### 🔴 The verifier itself was broken first, and it "refuted" the result

First run: reference arm returned **+1.36% CAGR / −70.3% MaxDD** against the original's +23.88%.
A 22pp gap that looked like a devastating refutation.

**It was my new code.** `prc.get(s, 0.0)` marked any holding without a print that day to **zero**
— and 577 of 2,740 symbols lack a print on a typical day, so held names were being written to nil
and sold for nil. Fixed to carry the last observed price. Reference then returned **+29.29%**
against the original's +29.85% on a comparable config — agreement.

**Recorded because an independent verifier that is itself broken is worse than no verifier: it
would have killed a good result with false confidence.** This is the fourth time in this program
that a "finding" was my own tooling.

### Results — independent engine, both samples

| horizon / sample | arm | CAGR | Sharpe | MaxDD |
|---|---|---|---|---|
| **8yr UNTOUCHED (even months)** | ref @1.49 | +32.00% | 0.855 | −46.9% |
| | **WINNER @1.25** | +32.07% | **0.943** | **−38.8%** |
| | **DELTA** | **+0.07pp** | **+0.088** | **+8.05pp** (7/12) |
| 8yr previously-used (odd) | ref @1.49 | +30.15% | 0.829 | −46.9% |
| | **WINNER @1.25** | +30.66% | **0.918** | **−38.6%** |
| | **DELTA** | **+0.51pp** | **+0.089** | **+8.37pp** (7/12) |
| **26yr UNTOUCHED (even months)** | ref @1.49 | +16.76% | 0.607 | −62.3% |
| | **WINNER @1.25** | +17.02% | **0.666** | **−50.0%** |
| | **DELTA** | **+0.27pp** | **+0.060** | **+12.32pp** (8/12) |
| 26yr previously-used (odd) | ref @1.49 | +16.08% | 0.592 | −63.3% |
| | **WINNER @1.25** | +16.94% | **0.660** | **−49.9%** |
| | **DELTA** | **+0.87pp** | **+0.068** | **+13.43pp** (10/12) |

### What this DOES establish

**1. The edge is not a product of the start dates I picked.** Every experiment in this program used
ODD-month starts. The UNTOUCHED even-month sample gives **+0.088 vs +0.089 Sharpe (8yr)** and
**+0.060 vs +0.068 (26yr)** — essentially identical. That is a genuine out-of-sample check on
entry timing and it passes cleanly.

**2. The Sharpe and drawdown improvements replicate in independent code**, under a different
financing convention and different NAV accounting. Sign, rough magnitude and 10/12 consistency
on the 26yr all reproduce.

### ⚠️ What it does NOT establish — a limitation of my own design

**The clean-room reference is NOT the deployed book.** It is "deployed minus the vol overlay" at
1.49×, because I did not implement the overlay in the clean room. So its DELTA measures only
tranching + tilt + harder gate — it **excludes** the overlay removal, which is the single largest
component of the original's improvement.

That is exactly why the CAGR deltas differ: original **+3.00pp** (26yr) vs clean-room **+0.87pp**.
The gap (~2.1pp) is the overlay removal, which the clean-room reference already has. Read that
way the two engines are **consistent**, not contradictory — but the CAGR claim is verified only
in part, and the overlay-removal component has NOT been independently re-derived.

Corroborating: the clean-room "reference" returns +16.08% where the real LIVE returns +12.72% on
the same starts — a +3.4pp gap that is itself independent evidence the overlay is costly.

**Status: the Sharpe and drawdown improvements are INDEPENDENTLY CONFIRMED and survive an
untouched holdout. The CAGR improvement is partially confirmed; its largest component (deleting
the overlay) still rests on a single engine.**

---

## Cycle 43 — VERIFY3 · FULL INDEPENDENT VERIFICATION vs the TRUE deployed book

The first clean-room pass had a gap I flagged: its reference lacked the vol overlay, so
overlay-removal — the largest component of the CAGR gain — was never independently derived.
Closed by implementing the overlay in the clean room **to the LIVE engine's own definition**
(`ibkr_engine.compute_vol_scale`: clamp(0.15×1.49 / 40-day realised NAV vol, 0.30, 1.00)),
which is a genuinely different construction from the original harness's synthetic unlevered
"shadow book".

### 26yr — independent engine, TRUE live reference

| sample | arm | CAGR | Sharpe | MaxDD | dCAGR | dSharpe | dMaxDD | +Shrp |
|---|---|---|---|---|---|---|---|---|
| **UNTOUCHED** | LIVE @1.49+ovl | +14.17% | 0.596 | −55.3% | — | — | — | — |
| | WIN @1.00 | +14.96% | 0.682 | −41.7% | +0.79pp | +0.085 | +13.65pp | 10/12 |
| | **WIN @1.10** | +15.87% | 0.673 | −45.3% | **+1.70pp** | **+0.076** | **+10.04pp** | 10/12 |
| | WIN @1.25 | +17.02% | 0.666 | −50.0% | +2.85pp | +0.070 | +5.32pp | 10/12 |
| | WIN @1.49 | +18.36% | 0.661 | −56.1% | +4.19pp | +0.064 | −0.75pp | 9/12 |
| | **noOvl ONLY @1.49** | +16.76% | 0.607 | −62.3% | +2.59pp | **+0.010** | **−7.00pp** | 8/12 |
| used | LIVE @1.49+ovl | +13.68% | 0.581 | −56.6% | — | — | — | — |
| | WIN @1.00 | +14.97% | 0.676 | −41.5% | +1.29pp | +0.095 | +15.05pp | 11/12 |
| | **WIN @1.10** | +15.89% | 0.668 | −45.2% | **+2.21pp** | **+0.086** | **+11.41pp** | 11/12 |
| | WIN @1.25 | +16.94% | 0.660 | −49.9% | +3.27pp | +0.079 | +6.71pp | 11/12 |
| | **noOvl ONLY @1.49** | +16.08% | 0.592 | −63.3% | +2.40pp | **+0.011** | **−6.72pp** | 7/12 |

### 8yr — independent engine, TRUE live reference

| sample | arm | dCAGR | dSharpe | dMaxDD | +Shrp |
|---|---|---|---|---|---|
| **UNTOUCHED** | WIN @1.00 | +1.32pp | +0.086 | +5.61pp | 7/12 |
| | **WIN @1.10** | **+3.36pp** | **+0.080** | **+2.78pp** | 8/12 |
| | WIN @1.25 | +4.75pp | +0.058 | **−0.89pp** | 7/12 |
| | **noOvl ONLY @1.49** | +4.68pp | **−0.029** | **−8.94pp** | **4/12** |
| used | WIN @1.00 | +1.01pp | +0.078 | +5.91pp | 7/12 |
| | **WIN @1.10** | **+3.03pp** | **+0.073** | **+3.10pp** | 7/12 |
| | WIN @1.25 | +4.28pp | +0.049 | −0.51pp | 7/12 |
| | **noOvl ONLY @1.49** | +3.77pp | **−0.040** | **−8.87pp** | **2/12** |

### 1. CONFIRMED — the untouched holdout matches the used sample at EVERY leverage

26yr dSharpe holdout vs used: +0.085/+0.095 (1.00×), +0.076/+0.086 (1.10×), +0.070/+0.079
(1.25×). 8yr: +0.086/+0.078, +0.080/+0.073, +0.058/+0.049. **The edge is not a product of the
start dates chosen** — this is a clean out-of-sample result on entry timing, in independent code.

### 2. 🔴 CORRECTION — "just delete the overlay" is NOT a win on its own

`noOvl ONLY @1.49` (delete the overlay, leave leverage at 1.49×) is:
- 26yr: **+0.010 / +0.011 Sharpe** — indistinguishable from zero — and **−7pp drawdown**
- 8yr: **−0.029 / −0.040 Sharpe — NEGATIVE** — and **−9pp drawdown**, at **2-4/12**

**I previously called this "Option 1, the easy config win" and quoted +0.9pp CAGR.** That figure
came from an ISO-DRAWDOWN comparison — i.e. leverage adjusted downward to match. At UNCHANGED
1.49× nominal, deleting the overlay simply raises exposure (nominal ≈ realised once the scaler is
gone), buying CAGR with drawdown and losing Sharpe.

**The overlay removal and the leverage reduction are ONE change, not two.** Doing the first
without the second is worse than doing nothing. That correction matters because it was the piece
I described as deployable today.

### 3. The all-positive setting is 1.10×, not 1.25×

`WIN @1.10` is positive on CAGR **and** Sharpe **and** drawdown in **all four cells** (2 horizons
× 2 samples). `WIN @1.25` gives more CAGR but goes slightly negative on 8yr drawdown
(−0.89 / −0.51pp). The original engine had favoured 1.25×; the independent engine with the true
live reference favours **1.10×**.

**FINAL: WIN @1.10 — tranching K=4 + no overlay + credit gate→0 + sleeves 70/21/9 + constant
1.10× leverage.** Confirmed in two independent engines, on two horizons, on a start sample never
previously examined.

---

## Cycle 44 — VERIFY4 · YEAR-BY-YEAR, and the CREDIT-GATE / DRAWDOWN COUPLING

User chose 1.25x and asked to keep the live credit gate (derisk 0.50) rather than harden it to
0.00, plus a year-by-year breakdown. Ran three arms x 24 starts (12 used + 12 untouched) on the
clean-room engine. VERIFY2_cleanroom.py left BYTE-IDENTICAL on disk; the one line needed (return
the NAV curve) is patched in memory at import. Every calendar year is averaged over all 24 starts
-- a single start lets the 20-session rebalance phase drive the per-year numbers (sigma 7.92pp).

### Headline, 24 starts

| horizon | arm | CAGR | Sharpe | MaxDD | dCAGR | dSharpe | dMaxDD | +Shrp |
|---|---|---|---|---|---|---|---|---|
| 26yr | LIVE | +13.93% | 0.589 | −55.9% | — | — | — | — |
| | REC@1.25 **gate0.50** | +16.37% | 0.643 | **−57.0%** | +2.44pp | +0.054 | **−1.12pp** | 18/24 |
| | REC@1.25 gate0.00 | +16.98% | 0.663 | **−49.9%** | +3.06pp | +0.074 | **+6.01pp** | 21/24 |
| 8yr | LIVE | +26.85% | 0.877 | −38.0% | — | — | — | — |
| | REC@1.25 **gate0.50** | +31.89% | 0.936 | −38.7% | +5.04pp | +0.059 | −0.73pp | 14/24 |
| | REC@1.25 gate0.00 | +31.36% | 0.930 | −38.7% | +4.52pp | +0.054 | −0.70pp | 14/24 |

Internal consistency check PASSES: these 24-start means reproduce the cycle-43 12-start-per-sample
figures almost exactly (26yr gate0.00: +16.98/0.663/−49.9 here vs +17.02/0.666/−50.0 holdout and
+16.94/0.660/−49.9 used; LIVE +13.93/0.589/−55.9 vs the +14.17/+13.68 pair). Two runs, same answer.

### 1. 🔴 THE DRAWDOWN BENEFIT WAS THE GATE, NOT THE CONFIG

At 1.25x, **derisk 0.50 gives NO drawdown improvement — it is slightly WORSE than live on both
horizons** (−1.12pp on 26yr, −0.73pp on 8yr). derisk 0.00 at the identical leverage gives
**+6.01pp on 26yr**. A 7.1pp MaxDD swing from the gate multiplier alone.

It traces almost entirely to **2008**: LIVE −29.63%, gate0.50 **−35.52%** (−5.89pp), gate0.00
−30.23% (−0.61pp). And 2009 recovery: LIVE +24.23%, gate0.50 +27.48%, gate0.00 **+37.29%**.
Removing the vol overlay and running 1.25x raises crisis exposure; only the FULL gate buys it back.
Half a gate on an unbraked book is not half the protection — in 2008 it was almost none.

**This means the package I recommended cannot be decomposed the way I implied.** I already
corrected "overlay removal is standalone" in cycle 43. Same error one level up: **overlay-removal
and the HARD gate are also one unit.** The config has three coupled dials (overlay / leverage /
gate depth), and drawdown improvement requires at least two of the three to move together.

### 2. The gain is NOT broad-based year to year

| | 26yr gate0.50 | 8yr gate0.50 |
|---|---|---|
| years REC beats LIVE | **13/24** | **4/7** |
| median year diff | **+0.71pp** | **+0.18pp** |
| mean year diff | +2.47pp | +4.45pp |
| drop the single best year (2021) | +1.52pp, **12/23** | +2.25pp, **3/6** |

13/24 is a coin flip. The mean is carried by **2020 (+21.5pp) and 2021 (+24.4pp)**; the median
year is worth +0.71pp. Worst years 2008 (−5.89pp), 2024 (−4.85pp), 2004 (−4.49pp).

Mechanism is plausible, not a bug: 2020 = the overlay clamps after the March crash and stays
clamped through the recovery (its documented negative timing skill); 2021 = 70% momentum sleeve in
a momentum year. But that makes the edge **REGIME-SPECIFIC — it pays in V-shaped recoveries — not
a steady per-year process edge.** Reporting the +2.44pp mean without this is the "one good year"
inflation the user explicitly asked me to guard against, and on the mean-vs-median split it very
nearly is one.

### 3. Verdict on the user's requested config

1.25x + gate 0.50 = **more CAGR, modestly better Sharpe, NO drawdown improvement.** That is a
return-seeking change, not the risk improvement the program was aimed at. If drawdown matters,
either the gate goes to 0.00 or leverage comes down; at 1.25x with a half gate, both dials are
spent. 2008 at −35.5% is the number to look at before choosing.

---

## Cycle 45 — EXP-042/043/044 · THE EDGE HAS DECAYED. Reported against my own recommendation.

Opened a new phase by attacking my OWN results before chasing new ideas. Three findings, and the
third is the one that matters.

### EXP-042 — universe integrity: 3 defects (BUGS D9/D10/D11)
See BUGS.md. Headline: the 26yr file is missing ~10% of the modern investable universe (9.9% by
2025, 179 real SP1500 names with zero prices from 2017 on), and D5's PERMNO fix is incomplete
(WW/WTW carries a 15,925.6% one-day splice in BOTH files). **The A/B DELTA survives** -- same
start, only the file differs, REC-minus-LIVE agrees to 0.07pp in 2024 and 0.13pp in 2025 while the
LEVELS differ by up to 10.55pp. Conclusions are deltas, so they stand; absolute 26yr recent-year
figures must not be quoted.

### EXP-044 — does the splice contamination reach the book? MOSTLY NO
Membership-gated share of top-5 momentum slots taken by a likely artefact:
**8yr 0.010% (1 slot of 10,050). 26yr 0.856% (86 slots).** Ungated it was 7.07%/5.60%, so PIT
membership removes ~99%/85% of it. Nearly every flagged name (WW, TBHC, DXLG, ACY, ACIC, SEZL,
CLSK, MARA, RGC, KOPN) was **never an index member**.

**The one real hit is WTW on the 26yr file, and the asymmetry is instructive.** The same company is
`WW` in the 8yr file (never a member -> harmless) and `WTW` in the 26yr file, where it is a member
on 85 of 85 absurd days. The 26yr build married a genuine member's membership record to a corrupted
spliced price series. Small (0.86% of slots) but real; EXP-045 measures the return impact directly.

### 🔴 EXP-043 — THE EDGE HAS DECAYED. This cuts against my own recommendation.

I expected D9 to explain the 2023-2025 weakness -- the 26yr file is missing modern high-momentum
names, so the clean 8yr file should have looked better. **It does not. It looks worse.**

8yr sub-periods (clean file, 24 starts), REC@1.25 gate0.50 vs LIVE:

| period | dCAGR | dSharpe | dMaxDD | sign |
|---|---|---|---|---|
| 2018-2020 | **+10.72pp** | **+0.247** | +1.32pp | 16/24 |
| 2021-2022 | +5.16pp | +0.077 | −5.08pp | 12/24 |
| **2023-2025** | **+0.67pp** | **−0.053** | **−4.51pp** | **9/24** |

Rolling 3-year CAGR delta, 8yr: +10.72 / +13.01 / +9.74 / +3.78 / −0.94 / +0.67.
**Trend slope −2.80pp per year.** 26yr agrees at the tail: 2022-2024 −1.93pp and 2023-2025 −2.43pp
are the **4th percentile of all 23 windows**.

**Honest reading:** in the most recent three years the proposed config delivers no excess CAGR, a
NEGATIVE Sharpe delta, WORSE drawdown, and sign consistency of 9/24 -- below a coin flip. The
+2.44pp/+5.04pp headline I reported is earned almost entirely in 2018-2022.

**Statistical caveat, stated so I do not overclaim in the other direction:** the 8yr trend is fit
on 6 heavily overlapping windows from 8 years -- they are nowhere near independent, and the script's
own percentile call ("inside the historical spread") is meaningless on 6 points. The 26yr tail
(4th percentile of 23) is the stronger evidence. This is a serious warning, not a proof of death.

**Open confound, and it is a real one (-> EXP-046).** LIVE = 1.49x x vol overlay; in a calm bull
market the overlay sits at its 1.00 cap so LIVE runs the full 1.49x, while REC runs 1.25x flat.
2023-2025 was exactly that regime, so a pure LEVERAGE gap predicts the same rolling pattern as
decay with no change in signal quality. The two are separated only by holding leverage fixed:
(REC@1.25 - LIVE@1.25) is the strategy term, (LIVE@1.49 - LIVE@1.25) is the leverage term.
**Until that decomposition lands, "the edge decayed" is the leading hypothesis, not the verdict.**

---

## Cycle 46 — EXP-046 · 🔴 RETRACTION: THERE IS NO DECAY. It was the leverage term.

Cycle 45 reported "the edge has decayed" off the COMBINED rolling delta (REC@1.25 vs LIVE@1.49).
That comparison confounds two changes at once -- the strategy AND a 1.49x->1.25x leverage cut --
and I flagged the confound but led with the alarming reading. **The decomposition says the
confound was the whole story. I was wrong.**

Holding leverage fixed:

| term | 26yr mean | 26yr last3 | 8yr mean | 8yr last3 | pos windows |
|---|---|---|---|---|---|
| **STRATEGY** (REC@1.25 − LIVE@1.25) | +3.54pp | **+3.88pp** | +8.95pp | **+4.71pp** | **20/23 · 6/6** |
| LEVERAGE (LIVE@1.49 − LIVE@1.25) | +0.81pp | **+3.50pp** | +2.78pp | **+3.54pp** | 18/23 · 6/6 |
| COMBINED (REC@1.25 − LIVE@1.49) | +2.73pp | +0.38pp | +6.17pp | +1.17pp | 17/23 · 5/6 |
| STRATEGY@1.49 (REC@1.49 − LIVE@1.49) | +4.29pp | +2.69pp | +8.91pp | +3.67pp | 19/23 · 6/6 |

### 1. The strategy term is NOT decaying
26yr: recent **+3.88pp vs its own long-run +3.54pp** -- slightly ABOVE average, 20/23 windows
positive. 8yr: positive in **6/6** windows, recent +4.71pp.

### 2. The 8yr "decline" is REVERSION TO THE 26yr NORM, not decay
The 8yr strategy term ran +11 to +16pp in 2018-2021 -- three to five times the 26-year average of
+3.5pp. That window is the COVID crash and V-recovery, the single most favourable regime for
removing a vol overlay that clamps down after a crash and stays clamped. Recent +4.71pp sits right
on the 26yr long-run mean. **An exceptional period ending is not an edge dying**, and reading the
8yr's own early windows as the baseline is exactly the mistake the two-horizon rule exists to catch.

### 3. What ACTUALLY changed is the LEVERAGE term
26yr: +0.81pp historically -> **+3.50pp in the last three windows** (2022-2024 +3.28, 2023-2025
+4.51). 2022-2025 has been a strong low-vol bull market, which is precisely when running 1.49x
instead of 1.25x pays. **The combined delta shrank because I recommended giving up leverage, not
because the signal weakened.**

### Consequence
The strategy package (overlay removal + sleeves 70/21/9 + 4 tranches) is robust and regime-stable
at BOTH leverage levels -- 19-20/23 on the 26yr, 6/6 on the 8yr. The leverage level is a pure
risk-preference choice with a known price, not a signal question. Cycle 45's headline is retracted;
the sub-period Sharpe weakness in 2023-2025 remains real and is the leverage term showing up in a
bull market.

**Process note.** The confound was written down in EXP-046's docstring BEFORE the result, with both
outcomes pre-specified. That is the only reason this got caught rather than shipped as a scary
finding. Cycle 45 should have led with the confound instead of the headline.

---

## Cycle 47 — EXP-045/047 · 🔴 SECOND RETRACTION: REMOVING THE OVERLAY WAS THE WRONG CALL

### EXP-045 — the splice guard, with a working positive control
`splice300` returned **exactly 0.000** on every 8yr metric. That is the suspicious-roundness
pattern that caught EXP-035, so I verified the guard rather than accepting it: it fires on 1.36
names/day but **intersects SP1500 membership zero times** on the 8yr file, so the no-op is correct.
The 26yr is the positive control (GPOR intersects membership on 171 date-cells) and it **does**
move: +0.06pp CAGR / +0.002 Sharpe. Guard works; **D10's real cost is ~0.06-0.08pp CAGR.**
REC−LIVE goes +2.44pp → +2.46pp. **D10 is resolved as immaterial.**

`splice100` (which also removes genuine squeezes) COSTS −2.43/−4.46pp CAGR: the strategy really
does harvest real squeezes, and they are worth money.

### 🔴 EXP-047 — the 24-config factorial. THE OVERLAY SHOULD STAY.

MAIN EFFECTS (each averaged over 12 configs — far more robust than any config's rank):

| factor | 8yr FULL | 26yr FULL | 8yr 2023-25 | 26yr 2023-25 |
|---|---|---|---|---|
| **tranche t4 − t1** | **+0.065** | **+0.033** | +0.069 | −0.023 |
| **leverage 1.10 − 1.49** | **+0.018** | **+0.021** | +0.050 | +0.053 |
| overlay ON − OFF | **+0.027** | **−0.011** | **+0.118** | **+0.102** |
| sleeves 70/21/9 − 50/35/15 | +0.023 | +0.002 | −0.035 | −0.017 |

**On the 8yr, all five configs beating LIVE on Sharpe in EVERY sub-period have the overlay ON and
4 tranches. Zero no-overlay configs pass.** The no-overlay arms show the giveaway shape: the
highest ΔCAGR in the grid (+7.03pp for `noovl_s702109_t4_L1.49`) alongside the WORST recent
sub-period (−0.113) and worse drawdown (−5.19pp). They buy CAGR with risk.

**My entire overlay-removal thesis was a 2018-2022 artefact.** That window is the COVID crash plus
V-recovery — the single most favourable regime possible for deleting a brake that clamps after a
crash and stays clamped. Both horizons now agree the overlay is BETTER in 2023-2025 (+0.118/+0.102).

### The contaminated cell — why the horizons disagree on the full-sample overlay effect
`ovl_s503515_t4_L1.49` (LIVE + tranching only) has 2023-2025 dSharpe **−0.049 on the 26yr file and
+0.052 on the 8yr file** — opposite signs, same config, same calendar window, different universe
file. BUGS D9: the 26yr file is missing ~10% of the modern universe in exactly that window.
**The 26yr 2023-2025 column is the least trustworthy cell in the table** and must be down-weighted
against the 8yr's, which is clean there.

### What is actually established
Two-horizon agreement, which is the bar: **tranching helps (+0.065/+0.033)** — the largest single
effect and the one with the cleanest mechanism (variance reduction, no alpha claim, therefore
regime-independent) — and **lower leverage raises Sharpe monotonically** on both. Sleeve tilt is
noise on the 26yr (+0.002) and NEGATIVE recently on both; it should probably be dropped too.

### NOT selecting on the leaderboard
Only 2 of 24 configs went 4/4 on the 26yr, by margins of +0.006 and +0.007, while strictly better
configs missed by −0.018. That is a knife-edge, not a finding, and 5-of-24 passing 3/3 on the 8yr
is barely above the ~3 expected by chance. What makes the result credible is that survivors
CLUSTER on one factor combination instead of scattering. Finalists are therefore chosen from the
MAIN EFFECTS, not the ranking → EXP-051 at the full 24-start standard with the untouched holdout.

**Leading candidate: overlay ON + 4 tranches + leverage 1.10.** Expect roughly FLAT CAGR with
+0.056/+0.103 Sharpe and **+6.7 to +8.4pp of drawdown**. That is a drawdown-reduction package, not
the CAGR package I pitched. Less exciting and much more likely to be real.

---

## Cycle 48 — EXP-048 · IDIOSYNCRATIC-VOL SCREEN (I-11): **KILLED**, and not marginally

`vol_60d` is the single strongest feature in the AUDIT01 forward-IC sweep (IC −0.050, low vol →
high forward return), stronger than any return feature, and the book harvests it only through a
soft ×1.15 nudge. The low-vol anomaly is among the most replicated results in the literature. This
was the best-motivated untested idea in IDEAS.md. It fails hard.

Dose-response, dSharpe vs the unscreened base (12 starts):

| screen | 8yr dCAGR | 8yr dSharpe | 26yr dCAGR | 26yr dSharpe |
|---|---|---|---|---|
| p80 (drop top vol quintile) | −16.50pp | **−0.279** | −4.52pp | −0.061 |
| p67 (drop top tercile, = I-11) | −22.07pp | **−0.441** | −7.76pp | −0.155 |
| p50 (drop top half) | −22.17pp | −0.395 | −7.67pp | −0.109 |

Negative at every dose, on both horizons, in almost every sub-period. Also NON-MONOTONE (p67 worse
than p50), which by the criterion I set before running means the fine structure is noise — but the
direction is not in doubt.

**Why it fails, and why the premise was still not silly.** A cross-sectional IC says low-vol names
beat high-vol names *on average across the whole cross-section*. It does NOT say the momentum
sleeve's top-5 are improved by deleting the high-vol half of the pool — momentum winners ARE the
high-vol names, so a book-wide vol screen removes the sleeve's entire hunting ground. The feature
is real; harvesting it this way destroys the thing it is bolted onto. A signal being predictive
standalone and being additive to an existing book are different claims, and I conflated them.

**Scope honesty:** this screened the WHOLE book, which is a stronger intervention than I-11
proposed (momentum pool only) -- flagged in the docstring before running, not after. A pool-only
version remains formally untested, but at −0.28 to −0.44 Sharpe the narrower variant is very
unlikely to rescue it. Downranked to LOW, not reopened.

One notable side-result: p50 on the 26yr buys **+14.99pp of MaxDD** for −7.67pp CAGR. A large
drawdown lever exists here — it is simply a bad *trade* (Sharpe −0.109), not an absent effect.

---

## Cycle 49 — EXP-052/053 · K=4 CONFIRMED; and the credit gate should go to 0.00

### EXP-052 — tranche count was ASSUMED, never tested. It survives.
K=4 was picked at the start of this program and never validated. Sweep (K=1/2/4/5/10, leverage
1.10 and 1.49, 24 starts, both horizons):

| K | 8yr dSharpe @1.10 | 26yr dSharpe @1.10 | sigma ratio 8yr | vs sqrt(K) |
|---|---|---|---|---|
| 2 | +0.025 | +0.021 | 0.643 | 0.707 |
| **4** | **+0.062** | **+0.027** | 0.602 | 0.500 |
| 5 | +0.057 | +0.026 | 0.518 | 0.447 |
| 10 | +0.060 | +0.024 | 0.473 | 0.316 |

**K=4 peaks on BOTH horizons and is the ONLY K positive in every 8yr sub-period at BOTH leverage
levels** — K=2/5/10 all go negative in 2018-2020. Per-start sigma falls 40% (8yr) / 55% (26yr).

**Against myself:** I-01's falsification criterion said sigma must fall by ~sqrt(K) "or the idea is
void." **It does not** (0.602 at K=4 vs 0.500 predicted; 0.473 at K=10 vs 0.316). Phases are more
correlated than the mechanism assumed because sub-books hold overlapping names. Mechanism PARTIALLY
confirmed; the idealised version is wrong. Recording the failed criterion rather than dropping it.

### 🔴 EXP-053 — gate depth on the WINNING (overlay-ON) base: 0.00 beats 0.50

EXP-049 measured gate depth on the overlay-OFF base, which EXP-047 superseded. Re-run on the
actual candidate, 24 starts, both samples. At leverage 1.10, UNTOUCHED HOLDOUT:

| horizon | gate | CAGR | Sharpe | MaxDD | dSharpe | dMaxDD | won |
|---|---|---|---|---|---|---|---|
| 26yr | **0.00** | **+14.33%** | **0.666** | **−43.1%** | **+0.070** | **+12.25pp** | 4/4 |
| 26yr | 0.25 | +14.13% | 0.658 | −44.2% | +0.062 | +11.06pp | 4/4 |
| 26yr | 0.50 | +13.88% | 0.647 | −48.1% | +0.051 | +7.24pp | 4/4 |
| 8yr | 0.00 | +27.43% | 0.995 | −29.9% | +0.111 | +8.01pp | 3/3 |
| 8yr | 0.50 | +27.46% | 0.992 | −29.9% | +0.107 | +8.02pp | 3/3 |

**26yr: MONOTONE across 0.00 → 0.25 → 0.50 on CAGR, Sharpe AND drawdown** — the pre-registered
dose-response criterion, met on the axis that matters. g0.00 dominates g0.50 by +0.019 Sharpe and
**+5.0pp of drawdown** while also delivering MORE CAGR (+0.45pp), because a shallower drawdown
compounds better. 8yr: gate depth is irrelevant (+0.004 spread) — no cost to taking the deeper gate.

**Why this reverses the earlier 'keep 0.50' call, and it is not a contradiction.** VERIFY4 found
g0.00 overshot in 2020 (+16.99% vs g0.50's +20.64%) on the overlay-OFF base. With the overlay ON
that penalty VANISHES: 2018-2020 dSharpe is +0.105 (g0.00) vs +0.101 (g0.50), a tie. The overlay
has already de-grossed by the time the gate fires, so the gate is no longer doing the de-risking
alone and its depth costs nothing in a V-recovery. The two mechanisms are complements, and 'keep
the gate shallow' was only correct on a base that no longer exists.

**Note g0.25 is NOT selected** despite sitting between: it was never pre-registered, and it is
dominated by g0.00 on every axis anyway, so nothing turns on it.

**FINAL: overlay ON + 4 tranches (5-day stride) + sleeves 70/21/9 + leverage 1.10 + credit gate
derisk 0.00.** 26yr holdout +0.070 Sharpe / **+12.25pp MaxDD**; 8yr holdout +0.111 / +8.01pp;
CAGR flat on both. 4/4 and 3/3 sub-periods.

---

## Cycle 50 — EXP-054 · THE PARAMETER FRONTIER IS REACHED. Nothing promoted.

Seven parameters that had never been swept — all hardcoded in the clean-room engine, which is
precisely why nothing had ever touched them. Star design, 3 levels each (base in the middle so
monotonicity is testable), 15 configs x 2 horizons.

**Instrumentation parity, twice confirmed.** The knobs were made config-driven by an in-memory
source patch (VERIFY2_cleanroom.py untouched on disk). With every knob at its hardcoded default
the patched BASE reproduces the UNPATCHED engine exactly: 8yr +26.46%/0.974/−29.6% vs EXP-053's
+26.46%/0.974/−29.6%; 26yr +14.15%/0.655/−42.8% vs EXP-053's +14.15%/0.655/−42.8%. The patch is
behaviour-neutral, so the sweep measures the parameters and not my instrumentation.

**A pre-flight assertion caught a real defect before any compute was spent:** the anchor
`VOL_TARGET = 0.15 * 1.49` appears TWICE (once in an explanatory comment). A silent partial patch
would have left the vol target hardcoded while the label said otherwise, producing plausible wrong
numbers across the whole sweep. Fixed with a line-anchored replacement and a full uniqueness check.

### RESULT: nothing clears the pre-registered bar. The base is at a local optimum.

| parameter | 8yr coherence | 26yr coherence | verdict |
|---|---|---|---|
| **top_n** | both worse (−0.151/−0.110) | both worse (−0.038/−0.034) | **5 CONFIRMED optimal** |
| **trailing_stop** | both worse (−0.020/−0.096) | both worse (−0.016/−0.036) | **40% CONFIRMED optimal** |
| **credit_pct** | both worse (−0.017/−0.008) | gradient, fails sub-period | **p95 CONFIRMED** |
| vol_target | gradient → 0.85x, all sub-periods | **+0.000/+0.000, base in a hole** | rejected |
| vol_floor | gradient → 0.40, all sub-periods | gradient but fails a sub-period | rejected |
| cap_pos | gradient → 0.20, all sub-periods | fails a sub-period | rejected |
| vol_lookback | gradient → 20, fails a sub-period | gradient → 20, fails a sub-period | rejected (see below) |

Three variants passed on the 8yr (vol_target 0.85x +0.029, vol_floor 0.40 +0.007, cap_pos 0.20
+0.003). **None survives on the 26yr** — they collapse to +0.000, +0.004 and −0.000 respectively.
Classic single-horizon artefacts, caught by the two-horizon rule exactly as designed.

### Three parameters are now CONFIRMED rather than assumed
`top_n=5`, `trailing_stop=0.40` and `credit_pct=0.95` are each WORSE on both sides on BOTH
horizons — a genuine local optimum, not an untested default. `credit_pct=0.98` is the standout
warning: on the 26yr it costs **−12.11pp of drawdown**, because raising the trigger means the gate
almost never fires and the crisis protection disappears.

### The one near-miss, documented and NOT promoted
`vol_lookback=20` is the only variant positive on BOTH horizons on full-sample Sharpe
(**+0.033 8yr, +0.018 26yr**) AND on drawdown (+0.33pp, +0.64pp). It fails only the every-
sub-period test (8yr 2021-2022 −0.068; 26yr 2001-2008 −0.017).

**It is rejected.** Rule (b) was written before the run and says positive in EVERY sub-period on
BOTH horizons. Promoting this would mean loosening a validation rule to let a result through,
which is the one thing this program forbids. Recorded as the single most promising direction if it
is ever revisited under a fresh pre-registration — not as a pending win.

**Consequence:** escalation-ladder rung 1 (parameters) is now exhausted. The recommended config is
unchanged and every one of its parameters is either measured or confirmed at a local optimum.

---

## Cycle 50 — FINAL-CONFIG FREE AUDIT · it is a RISK package, and it LOSES most calendar years

Final config = overlay ON, K=4 / 5-day stride, sleeves 70/21/9, leverage 1.10, credit gate
p95 → derisk 0.00. Audit pieces computed from the EXP-053 cached curves (24 starts = 12 used +
12 untouched, both horizons), clean-room engine:

| | 8yr | 26yr |
|---|---|---|
| dCAGR | +0.10pp | +0.31pp |
| dSharpe | **+0.108** (16/24) | **+0.072** (20/24) |
| dMaxDD | **+8.22pp (24/24)** | **+12.98pp (24/24)** |
| worst-start MaxDD | −46.0% → **−31.3%** | −67.7% → **−49.7%** |
| per-start CAGR σ ratio | 0.48 | 0.45 |

**Drawdown is shallower in every one of 48 start dates.** Start-date dispersion is halved. That
part is as solid as anything in this program.

### But by the user's own criterion — "better in most years, not one good year" — it FAILS
Year-by-year (mean over 24 starts, wins pairwise): **FINAL beats LIVE in 2/7 years (8yr) and
10/24 (26yr); median year −2.74pp / −0.65pp.** It wins big in the bad years (2008 +6.1pp, 2022
+7.1pp, 2020 +4.7pp, 2015 +6.0pp, 2005 +7.6pp) and gives up ground in strong bull years (2003
−11.9pp, 2021 −15.8pp/−7.7pp, 2023 −8.6pp, 2013 −6.8pp). Mean yearly diff is NEGATIVE (−0.95pp
on 26yr) while CAGR diff is positive (+0.31pp): shallower drawdowns compound better — the
arithmetic/geometric gap is the whole return story.

**That is exactly what a 1.10× book with a hard gate should do against a 1.49× book.** It is not
a signal improvement showing up every year; it is less gross plus better crash handling. Event
concentration is N/A for the same reason: net excess is ≈0, so there is nothing to concentrate
(BUGS A8a guard fires: excess not meaningful in any start).

### Consequence: the Sharpe gain must be SPENT explicitly, and I have only measured one way
- Spent on risk (1.10×): same CAGR, 8-13pp less drawdown, loses most bull years.
- Spent on return (run the final base at LIVE's own ~1.49×): CAGR read at LIVE's own drawdown —
  the only leverage-fair answer to "is it better most years". EXP-053's ladder stopped at 1.25×
  (26yr −46.2%, still 10pp shallower than LIVE). → **EXP-056 extends to 1.40× / 1.49×**, queued.

Both are legitimate. Which one is "the improved strategy" is the user's risk preference, and the
write-up must present both rather than pick silently.

---

## Cycle 51 — EXP-055 · PAIRED COST SENSITIVITY on the final base: PASS, and the delta WIDENS

Final base (overlay ON, K=4/5, 70/21/9, L1.10, gate p95→0.00) vs LIVE, both arms at the same cost
multiplier (BUGS A8b), 12 starts, clean-room engine:

| cost × | 8yr dSharpe | 8yr dCAGR | 26yr dSharpe | 26yr dCAGR | 26yr dMaxDD | sign (Shrp) |
|---|---|---|---|---|---|---|
| 1 | +0.105 | +0.09pp | +0.074 | +0.48pp | +13.72pp | 7/12 · 11/12 |
| 2 | +0.108 | +0.47pp | +0.076 | +0.83pp | +13.73pp | 8/12 · 11/12 |
| 3 | +0.109 | +0.78pp | +0.078 | +1.17pp | +13.74pp | 8/12 · 11/12 |
| 5 | **+0.115** | **+1.53pp** | **+0.084** | **+1.86pp** | +13.70pp | 8/12 · 11/12 |

The edge is not turnover-fragile; it is turnover-FAVOURED. FINAL runs 1.10× gross against LIVE's
1.49× and the hard gate goes flat instead of half-size in stress, so its dollar turnover is lower
and every multiplier taxes LIVE more. (The earlier 2×/3×/5× pass was on the retired overlay-OFF
package; this is the first on the actual candidate.) Audit gate for the structure is now complete
except Option B's leverage ladder (EXP-056, running).

---

## Cycle 52 — EXP-056 · the ladder to 1.49×: the structure DOMINATES LIVE at every leverage. PROGRAM CLOSED.

Final base at 1.40× and 1.49× added to EXP-053's ladder (24 starts, both horizons). The
"iso-drawdown" question dissolves: **even at LIVE's own 1.49× the structure's MaxDD is 6.5pp (26yr)
/ 3.6pp (8yr) shallower than LIVE's** — LIVE's drawdown is off the bottom of the curve. Every rung
gives more CAGR, more Sharpe and less drawdown than LIVE simultaneously.

At 1.49× (26yr): CAGR +15.53% vs +13.93% (+1.60pp, 18/24), Sharpe +0.052 (19/24), MaxDD −49.5% vs
−55.9% (+6.5pp, 20/24), worst-start −58.0% vs −67.7%, **better in 16/24 calendar years, median
+1.17pp, +0.96pp with the best year dropped (15/23).** This is the first configuration in the
program to pass the user's "better in most years" bar on the crisis-bearing horizon. 8yr: 3/7
years — the strongest momentum years go to LIVE by 1-3pp; too few years to decide.

**Verdict — leverage is a pure risk dial; recommendation 1.25×** (highest leverage at which MaxDD
is shallower in all 48 starts; +0.9/+1.4pp CAGR; +0.06/+0.10 Sharpe; the user's stated
preference). 1.49× is the "change only the structure" option with the most CAGR. 1.10× is
drawdown-first. No setting is worse than LIVE on any axis.

Audit gate for the final structure is complete. Remaining work is ENGINEERING (tranche path in
ibkr_engine, staggered transition) and DATA (WRDS re-download with PERMNO/GVKEY, then rebuild both
universes and re-verify). ~4,452 configurations, 56 experiments, 3 retractions, 1 live incident
found and fixed. Closing the research program here.

---

## Cycle 53 — v2 UNIVERSE REBUILD + FULL RE-VERIFICATION (2026-09-07/08): the recommendation stands on corrected data

**Ask.** Rebuild both universes from the refreshed WRDS pull (PERMNO-keyed, 2026 prices, fundamentals to
2026-08), verify them, re-run LIVE vs the recommended structure on both horizons through 2026-08-31, and prove
nothing is inflated. Deliverable: `FINAL_RECOMMENDATION.md` section 0 (side-by-side, per-year, 24-row
PASS/FAIL table).

**Result (24 starts each, clean-room engine).** 8yr LIVE +29.52% / 0.956 / −38.6% vs FINAL@1.25 +31.03% /
1.053 / −33.5% (Sharpe better 17/24, MaxDD 17/24); 26yr LIVE +16.34% / 0.658 / −54.1% vs +17.26% / 0.730 /
−43.0% (Sharpe 20/24, MaxDD **24/24**). Cost 2× leaves the delta intact (+0.098 / +0.073). Same ordering at
1.10× and 1.49×. Verdict unchanged: 1.25× recommended; it is a risk package (loses 2023-26 by −0.029 Sharpe,
wins every bad year: 2008, 2015, 2018, 2020, 2022).

**Four defects found by the gate and fixed BEFORE any number was kept (BUGS.md D12a-d).** (a) all 286,782 new
fundamentals rows dropped by a NaN link column; (b) `V.END` override applied after the globals copy → every
curve ended 2025-12-31 under a header saying Aug 2026 (caught by the curve-span check); (c) ETFs are
`tpci="%"` in Compustat → zero 2026 SPY/sector prices; (d) membership symbols are Compustat SECURITY tics, not
point-in-time — the date-aware CRSP-era resolver mapped 197 renamed symbols (2.7% of member-days) to the wrong
company and collapsed twins; resolver v4 (security semantics + CUSIP bridge through Security Monthly, which
recovers AMR/Kodak/Frontier/Chesapeake/Delta/Delphi/Lear/Peabody) leaves 3 overlap PERMNO-days in 26 years.
Pipeline runs 1-3 (voided) are archived under `research/_v2_BUGGY_*`.

**Independent checks that PASSED.** Universe checklist 19/19 on both horizons (row counts vs an explicit
calendar, duplicates, NaN/inf per feature, gaps = 9/11 / Ford / Sandy only, count stability, spot checks,
features recomputed from prices ≤T, roe recomputed from rdq-dated rows). Ledger audit 44/44 on both horizons
(data-access recorder: 0 of ~92k accesses beyond the decision date; membership key ≤ date; 15% cap; gross ≤
leverage bound; ledger reconstructs cash to the cent; CAGR/MaxDD recomputed by a second method to 2e-16;
Sharpe daily vs monthly within 0.07; win rates 49.8-53.2%). 2026 prices vs Polygon: corr 0.9993 on 1,450
names; 2017-25 vs the live system's cached Polygon closes: corr 0.94-0.999.

**Flag investigated.** 2024 (+66-74%) and 2026 YTD (+68-70%) are huge years. Prices confirmed against an
independent vendor; the 2026 cross-section (EW +15.7%, p90 +58.8%) supports a top-5 momentum book at 1.49×
doing +68%. Could NOT run: live-account reconciliation — live NAV Jun 15→Aug 31 includes a $16.8k deposit
(Jul 17, +47.5% one-day jump); deposit-netted ≈ +24% vs backtest +0.7% over the same window; live ran
SP500-only sleeves until Jul 25 and degraded-coverage signals in Aug, so this is open tracking work.

**Caveats.** Membership known only to 2026-05-01; 19 symbols (0.2%) unresolvable; two CRSP/Compustat
survivor disagreements (USB, BDC) collapse (0.02%); 2026 is a different vendor chained on CRSP.

Configs tested this cycle: 6 arms × 24 starts × 2 horizons (no new search). Program total ≈4,452.

## Cycle 54 — LEAKAGE / INFLATION AUDIT of the v2 result (user checklist 1-17): nothing found; the "too good" is 2026

Treated the v2 numbers as wrong until proven otherwise. Scripts: `AUDIT_leakage_v2.py` (features' source
timestamps for 10 random rows, one trade's exact rows, membership turnover, top-20 held-name daily returns vs raw
CRSP/Polygon, per-fill costs, equity rebuilt from the fill log by a separate loop, quarterly attribution,
truncation to 2025, untouched starts only, EW buy-and-hold), `AUDIT_harness_v2.py` (next-close fills; random
signal through the identical harness), `AUDIT_origengine_v2.py` + `AUDIT_enginegap_v2.py` (the original live-mirror
engine on the same v2 data). Results table in `FINAL_RECOMMENDATION.md` §0.5.

Findings: (1) no leakage signature anywhere — 10/10 rows, 40/40 daily returns, 0 accesses beyond T, fill log
reconstructs to 2e-15; (2) random signal through the same harness: 8yr +4.5%/0.30, 26yr +7.4%/0.40 (real +31.5%,
+13.6%) — the harness does not manufacture returns; (3) next-close fills cost 0.1-0.5pp, delta intact; (4) the
plausibility question is answered by the WINDOW: through 2025 the v2 clean room gives 8yr +23.26% (canon +23.58%)
and 26yr +14.27%; the headline +29.52%/+16.34% is Jan-Aug 2026 (+68% YTD, prices verified against Polygon);
(5) the two engines agree to 0.2pp on the 26yr; on the 8yr the clean room is +3.1pp higher, decomposed to the
position cap (mirror config 0.10 vs live/clean-room 0.15 = `ibkr_engine.POSITION_CAP`) 2-3.8pp + financing 0.4pp.

Correction logged: cycle 53's "levels moved <1pp" compared different windows; on the same window the rebuild
LOWERED the 8yr by 3.6pp. Least-confident check: the 8yr headline rests on one 8-month period, and ~1pp of
engine disagreement (marking/fill conventions) is not decomposed.

## Cycle 55 — DEPLOYMENT DAY AUDIT (2026-09-08 evening): universe -> backtest -> live parity, one live bug found and fixed before the first tranche day

Owner asked whether the numbers are real, end to end. New tests (not repeats): membership file vs public
index-change effective dates — 12/12 exact (TSLA 2020-12-21, META 2013-12-23, GOOGL 2006-04-03, UBER, SMCI,
PLTR, DELL, CRWD, ABNB, NFLX adds; LEH 2008-09-16, ENRN 2001-11-29 removals; TWTR/FRC/SIVB last day = last
trading day before removal). Split/dividend continuity vs an independent adjusted source: 7/7 to 0.00-0.03%.
SHIFT test (decide on T-1, execute at T close): LIVE +1.06pp / +0.024 Sharpe, FINAL_1.49 +0.33pp / +0.011 —
a leak would die; nothing does. Outlier dependence: best 5 of ~2,170 days = 25% (LIVE) / 22% (FINAL) of the
8yr log-return; with those days removed from BOTH arms the structure's Sharpe edge is +0.101 (8yr) / +0.076
(26yr). Live vs clean room since the 2026-08-11 rebalance (no deposits): live +7.05% vs backtest +5.46%,
daily corr 0.72, holdings 21/25 in common. Credit-gate percentile parity: live 0.0214 vs recomputed 0.0210.

**Cost realism:** with $61k split into 4 books the median order is ~$600 and IBKR's $1 minimum makes commissions
~16 bp of notional (+~5 bp slippage) ≈ 2x the modelled 10 bp. At 2x costs (8 starts, 8yr): FINAL_1.49 +28.39% /
0.933 / -39.0% vs LIVE +26.52% / 0.887 / -39.7% (delta intact: +1.87pp, +0.045). 26yr in `_audit_cost2_149_26yr.out`.

**BUG FOUND LIVE (the point of the exercise):** same-day signal parity on 2026-09-04 was 17/25 and the live
momentum sleeve ranked BNY #2. The vendor's per-TICKER history for "BNY" begins with the ~$10.5 closed-end fund
that held the ticker before BNY Mellon adopted it on 2026-05-21 -> +1,263% one-day jump -> fake +1,608%
momentum. The PERMNO-keyed backtest is immune. Fix: `signal_builder._splice_guard` (one-day ratio > 4x =
splice; keep only the new security's bars), tests 7/7, deployed 23:10 (e0ac1bd). After the fix: BUY overlap
21/23, momentum top-5 identical (SNDK, MU, LITE, WDC, STX). Also fixed today: partial-session guard data rule
(alarm #3), freshness-check false alarms (EDGAR patched-count, WRDS "next upload").

Verdict: universe (PIT membership, total-return prices, PIT fundamentals), backtest (no leak signature under
shift/delay/recorder/random-null, ledger-exact, two engines agree once the cap is matched), and live (same
sleeves, same constants, 21/23 same-day picks after the splice fix) are consistent. Honest expectation for the
live account is the cost-2x line, and the strategy's return is fat-tailed (a quarter of it comes from ~5 days
per 8 years).

## Cycle 56 — "ARE YOU SURE IT IS BETTER?" (2026-09-13): statistical strength, decomposition, fuller risk. Verdict TONED DOWN.

Owner asked for deep research that the deployed structure beats the old one. New evidence, all on the v2 caches
at the deployed 1.49x (`research/AUDIT_decompose_v2.py` + inline bootstrap/risk script):

**1. Statistical strength (the part earlier cycles overstated).** The 24 starts overlap almost entirely, so
"17/24 starts" is not 17 independent wins. Block bootstrap (60-day blocks, 2,000 draws) of the paired daily
difference FINAL_1.49 - LIVE: 8yr Sharpe diff +0.075, 90% CI [-0.195, +0.392], P(>0)=63%; 26yr +0.059,
[-0.081, +0.209], P(>0)=75%. Annualised return diff 8yr +2.4pp [-7.8, +13.9], 26yr +1.4pp [-3.3, +6.0].
=> the return/Sharpe advantage is plausible but NOT statistically decisive.

**2. Decomposition at 1.49x (8yr 8 starts / 26yr 6 starts).** Tranches alone: Sharpe +0.038 / +0.033 (4/8, 5/6),
MaxDD ~0 on BOTH horizons. Gate 0.50->0.00: Sharpe +0.003 / +0.016, MaxDD +0.0pp / **+8.5pp (6/6)**. Sleeves
70/21/9: Sharpe +0.004 / +0.009, CAGR +1.3pp / +0.6pp, MaxDD +0.4pp / -1.1pp. => the headline "MaxDD shallower
24/24 starts" is the CREDIT GATE at 0.00, i.e. the 2008 (and 2020) episodes, not tranching. The gate is
insurance priced off a handful of crises; tranching is a small, plausible Sharpe gain plus a large reduction in
start-date luck; the sleeve change is a marginal return tilt that gives back a little drawdown.

**3. What IS robust.** Outcome dispersion across start dates falls by a third to a half (26yr CAGR sd 3.1->1.6pp,
Sharpe sd 0.089->0.048; 8yr 5.5->4.0pp). 26yr: better on all three metrics in 18/24 starts, worse on all three
in 0/24; worst-start CAGR +11.1->+15.2%, worst-start MaxDD -58.9->-51.6%, worst month -25.1->-20.9%, longest
time under water 1,176->969 days, Calmar 0.30->0.40, 2008 episode -52->-45%. 8yr: worst month -20.7->-16.5%,
but all-three-better 11/24 vs all-three-worse 9/24 (a coin flip at 1.49x on that horizon). Rolling 3-year
windows favour FINAL 56-63% of the time. Cost-2x, next-close and shift tests: delta intact (cycles 54-55).

**Verdict (replaces "dominates").** The deployed structure is a RISK-SHAPING change with a modest, unproven
return edge: same signals, same leverage, less dependence on one rebalance date, a credit-stress cutoff that
made the worst historical episodes ~7pp shallower, and no scenario in which it is systematically worse. It is
not a demonstrated alpha improvement and should not be described as one. Where it will lag: momentum melt-up
years (2003, 2013, 2021, 2023, 2025) and any period where the gate flattens a book that then rallies.

## Cycle 57 — EXP-059 · FRONTIER SEARCH opened (2026-09-13): owner asks for more return, higher Sharpe, lower drawdown, consistently

Base = the DEPLOYED structure (4 tranches / 5-session stride, gate 0.00, sleeves 70/21/9, 1.49x) on the v2
universes: 8yr +32.42% / 1.026 / -36.7%, 26yr +18.16% / 0.719 / -46.6% (24 starts). Everything below is measured
as a DELTA to that base. Promotion gate (`research/GATE059.py`): 24 starts x 2 horizons, block-bootstrap 90% CI on
dSharpe with P(>0) >= 70% on both horizons, better in >= 50% of calendar years with no single year > 40% of the
positive delta, every sub-period dSharpe >= -0.02, untouched even-month starts agree in sign, and the edge survives
removing the best 5 days from both arms. Stage 1 = 8 starts (every third of the 24) per horizon; stage 2 = 24.

Candidates (all switches in `EXP059_frontier.py`, default OFF = deployed engine): I-34 exit_all (mid-cycle signal
exit for every book on tranche days), I-35 overlay_all (prompt vol x gate rescale of every book on tranche days),
I-02 waterfill cap, I-36 min-trade band 1%/2%, I-37 slow value/lowvol refresh, I-05 vol-normalised stop (1.5x, 2x),
I-06 exclude recent index additions (60/120 sessions), and a K re-sweep (2/5/8/10 books) on the corrected data.
Dead families are NOT re-tested (memory: timing rules, lever-up, ML ranking, sentiment, alt-data, idio-vol screen,
sleeve risk parity, hysteresis, aging, dispersion, book-crowding, residual momentum).

### Stage 1, 8yr (8 starts) — PRELIMINARY, UNAUDITED. Base +30.46% / 0.983 / -38.0%.

| arm | dCAGR | dSharpe (+) | dMaxDD (+) | years better |
|---|---|---|---|---|
| exit_all (I-34) | -0.64pp | +0.062 (7/8) | **+8.43pp (8/8)** | 4/8 |
| overlay_all (I-35) | +2.74pp | +0.083 (7/8) | +5.61pp (8/8) | 4/8 |
| **exit + overlay** | **+6.20pp** | **+0.108 (7/8)** | +5.37pp (8/8) | 5/8 |
| slow_vl_2 (I-37) | +2.21pp | +0.050 (8/8) | +1.23pp (6/8) | 6/8 |
| waterfill (I-02) | -0.26pp | +0.015 (8/8) | -0.06pp | 6/8 |
| K5_s4 | +1.13pp | +0.028 (5/8) | +3.17pp (6/8) | 5/8 |
| min_trade 1%/2% (I-36) | ~0 | 0.000 (4/8) | ~0 | 3/8 |
| exante_vol / exante_max (I-03) | -0.1 / -1.7pp | -0.013 / -0.010 (1/8) | -1.9 / +0.8pp | 4/8, 3/8 |
| vol_stop 1.5 / 2.0 (I-05) | -2.2pp | **-0.070 (0/8)** | -0.2pp | 3-4/8 |
| excl_adds 60 / 120 (I-06) | +0.8 / -1.6pp | +0.023 (7/8) / -0.026 (2/8) | -1.4 / -3.1pp | 4/8 |
| K2 / K8 / K10 | -0.8 / -1.1 / +0.4pp | -0.030 / -0.018 / +0.008 | mixed | 2-5/8 |

Read: the two "prompt adjustment" switches (exit names no sleeve wants; rescale every book to today's vol x gate)
are the only large effects and they stack; both REDUCE exposure in stress, so the +6.2pp CAGR of the pair is
exactly the kind of number that must be vol-matched and bootstrapped before it means anything. slow_vl_2 is a
turnover reduction that helps at 1x cost already (it will help more at 2x). Water-filling is tiny but consistent.
I-03 (ex-ante vol), I-05 (vol-normalised stop) and I-06 (exclude additions) are dead on the 8yr; K=4 stands
(K=5 is noise-level). Stage 2 (24 starts x 2 horizons) queued for the survivors and their combinations.

### Stage 1, 26yr (8 starts) — the 8yr's star COLLAPSES; two switches survive on both horizons. Base +18.11% / 0.717 / -47.1%.

| arm | dCAGR | dSharpe (+) | dMaxDD (+) | years better |
|---|---|---|---|---|
| exit_all (I-34) | -0.94pp | +0.036 (8/8) | **+5.99pp (8/8)** | 10/25 |
| overlay_all (I-35) | +0.87pp | +0.031 (8/8) | **+6.23pp (8/8)** | 13/25 |
| **exit + overlay** | +0.26pp | **-0.011 (3/8)** | +0.33pp (4/8) | 14/25 |
| exante+overlay | +0.05pp | +0.024 (8/8) | +5.88pp (8/8) | 10/25 |
| slow_vl_2 / waterfill / min_trade / exante / K2 / K5 / K10 | ~0 | -0.007 … +0.003 | -0.7 … +0.5pp | 10-16/25 |
| K8_s2 | -2.62pp | -0.076 (3/8) | -7.09pp | 9/25 |
| vol_stop / excl_adds_120 / vix_gate_0.5 | -0.5 … -0.6pp | -0.024 / -0.009 / -0.007 (0/8) | mixed | — |
| **mom_ens (I-09) / mom_52wh (I-10)** | **-3.8 / -4.6pp** | **-0.125 / -0.113 (0/8)** | -2.9 / -1.7pp | 9-10/25 |

Read: the +6.2pp / +0.108 8yr result for exit+overlay is HORIZON-SPECIFIC and dead — mechanism identified in the
code: after exit_all empties part of a book, overlay_all re-levers the REMAINING names of that book back to the
target every week (f2 > 1), i.e. it concentrates and churns; that paid in the 2018-26 momentum melt-ups and cost
over 26 years. The two switches ALONE survive both horizons with the same profile each time: exit_all = drawdown
instrument (-0.6/-0.9pp CAGR, +0.04-0.06 Sharpe, +6-8pp MaxDD); overlay_all = both (+0.9/+2.7pp, +0.03-0.08,
+5.6-6.2pp). Signal-construction changes (I-09, I-10) are badly negative on the long horizon — the cross-sectional
axis is exhausted, as the memory said. I-14 VIX gate dead. K=4 confirmed again (K=8 clearly worse).

Redesign for stage 2: `overlay_down` = the prompt overlay applied DE-RISK ONLY (scale other books down when
vol x gate calls for it; never lever them back up between rebuilds — the deployed overlay's own convention). Lean
stage 2 (24 starts x 2 horizons): base, exit_all, overlay_all, overlay_down, exit+overlay_down, their 1.65x
exposure-matched versions, and cost-2x versions of each. The 8yr-only combos are dropped.

8yr results for the three signal/timing arms confirm the 26yr: mom_ens -0.193 (0/8), mom_52wh -0.290 (0/8),
vix_gate_0.5 -0.010 (2/8; +2.6pp MaxDD but better in 1/8 years). The sleeve's own composite momentum score beats
both textbook alternatives by a wide margin on both horizons — the signal is not the place to look.

### Stage 2 (24 starts x 2 horizons) + GATE059 — PRELIMINARY, UNAUDITED (run 2026-09-14..16). Base 8yr +32.42/1.026/-36.7, 26yr +18.16/0.719/-46.6.

| arm | 8yr dCAGR / dSharpe (+) / dMaxDD (+) | 8yr yrs better · top-yr share | 26yr dCAGR / dSharpe (+) / dMaxDD (+) | 26yr yrs better · top-yr share | GATE 8yr / 26yr |
|---|---|---|---|---|---|
| overlay_all (I-35) | +2.58pp / +0.079 (23/24) / +4.6pp (24/24) | 4/8 · 2020 = 63% | +0.70pp / +0.026 (23/24) / +5.0pp (21/24) | 13/25 · 21% | FAIL (conc) / PASS |
| **overlay_down (I-35 de-risk-only)** | +1.60pp / +0.070 (23/24) / +4.9pp (24/24) | 3/8 · 2020 = 76% | **+0.77pp / +0.036 (24/24) / +5.5pp (21/24)** | 14/25 · 34% | FAIL (yrs, conc) / **PASS** |
| overlay_down_L1.65 (exposure-matched) | +2.49pp / +0.057 (22/24) / +3.5pp (24/24) | 5/8 · 59% | +1.32pp / +0.032 (24/24) / +3.4pp (20/24) | **17/25** · 20% | FAIL (conc) / PASS |
| overlay_down_gate0.25 | +1.54pp / +0.067 (24/24) / +4.9pp | 3/8 · 75% | +0.44pp / +0.025 (22/24) / +3.3pp | 15/25 · 24% | FAIL / PASS |
| overlay_down_gate0.50 | ~base | — | -3.5pp MaxDD vs overlay_down | — | not gated (worse) |
| exit_all (I-34) | -1.76pp / +0.035 (18/24) / +8.1pp (24/24) | 3/8 | -1.14pp / +0.030 (24/24) / +6.0pp (24/24) | 10/25 | FAIL / FAIL (yrs, sub-period 2001-08 -0.036) |
| exit+overlay_down | -0.72pp / +0.068 (19/24) / +8.6pp (24/24) | 3/8 | -0.76pp / +0.044 (24/24) / +8.6pp (24/24) | 11/25 | FAIL / FAIL (yrs) |
| exit+overlay_down_L1.65 | +0.38pp / +0.065 (19/24) / +7.5pp | 5/8 | -0.12pp / +0.045 (24/24) / +6.8pp | 12/25 | PASS / FAIL (yrs 48%) |

Cost 2x (arm_cost2 vs base_cost2, 24 starts; base_cost2 = 8yr +30.34/0.977/-37.6, 26yr +16.27/0.664/-47.8):
overlay_down +1.50pp/+0.066 (23/24)/+5.4pp and +0.69pp/+0.032 (24/24)/+5.1pp — the edge is INTACT at the
account's real cost level; overlay_all +2.30/+0.072/+5.1 and +0.50/+0.020/+4.7; exit_all +0.020 (14/24) and
+0.007 (13/24) — the exit rule's Sharpe edge is gone at 2x cost (it is pure turnover); exit+overlay_down
+0.053 (19/24) / +0.021 (17/24) with -1.1 / -1.2pp CAGR.

Batch 3 (stage 1, 8 starts, both horizons): mom_equal +0.010/+0.009 Sharpe (8/8 both; 19/25 years on the
26yr — a small, unusually even edge from equal-weighting the 5 momentum picks); overlay_down_lb20 +0.037/+0.025
(8/8 both), +5.4/+5.3pp MaxDD (a faster vol estimate helps the overlay); overlay_down_lb60 weaker than lb40;
min_weight_2pct ~0; exit_all_stop50 negative. Mechanism diagnostic (`AUDIT_overlay_mech.py`): the overlay's
gains sit in the crisis years by construction (2008 +8..10pp, 2020 +14..16pp) and it costs -1..-6pp in calm
bull years (2023 -3.1, 2024 -5.6 on the 8yr); exit_all's calm-year cost is far larger (2003 -16pp, 2024 -12pp).

### Batch 4 (stage 2, 24 starts) — overlay refinements + equal-weight momentum. PRELIMINARY, UNAUDITED (2026-09-16).

| arm | 8yr dCAGR / dSharpe (+) / dMaxDD | 8yr yrs · top-yr | 26yr dCAGR / dSharpe (+) / dMaxDD | 26yr yrs · top-yr | GATE 8yr / 26yr |
|---|---|---|---|---|---|
| mom_equal (I-38: equal-weight the 5 momentum picks) | +0.52pp / +0.012 (23/24) / +0.6pp | 6/8 · 49% | +0.32pp / +0.009 (24/24) / +0.2pp | **19/25** · 23% | FAIL (conc 49% on a +0.6pp mean) / **PASS, CI_lo > 0** |
| overlay_down_thr0.90 (I-40) | +2.08pp / +0.076 (23/24) / +4.8pp | 4/8 · 77% | +0.85pp / +0.036 (24/24) / +5.4pp | 16/25 · 35% | FAIL (conc) / PASS |
| overlay_down_thr0.85 | +2.61pp / +0.085 (24/24) / +4.7pp | 3/8 · 72% | +0.88pp / +0.035 (24/24) / +4.9pp | 15/25 · 37% | FAIL (yrs, conc) / PASS |
| **overlay_down + mom_equal** | +2.08pp / +0.081 (24/24) / +5.1pp | 4/8 · 72% | **+1.08pp / +0.045 (24/24) / +5.6pp** | 14/25 · 30% | FAIL (conc) / **PASS, CI_lo +0.005, P 97%** |
| overlay_down_lb20 (I-39) | -0.18pp / +0.040 (22/24) / +4.6pp | 3/8 | -0.01pp / +0.021 (24/24) / +4.3pp | 10/25 | dead: weaker than lb40 on both |
| overlay_down_lb20 + mom_equal | +0.17pp / +0.050 / +4.9pp | 3/8 | +0.24pp / +0.028 / +4.6pp | 10/25 | dead (lb20) |
| overlay_down_lb20_cost2 | (vs base at 1x) -2.4pp / -0.015 | — | -2.0pp / -0.042 | — | dead |

Read: (a) equal-weighting the momentum sleeve's five picks is the most CONSISTENT thing found in EXP-059 — tiny
(+0.3-0.5pp CAGR, +0.01 Sharpe) but positive on 24/24 starts on both horizons, better in 19/25 and 6/8 years, and
the 26yr bootstrap CI excludes zero; it fails the 8yr only on the year-concentration check applied to a +0.6pp mean
(2024 = +3.0pp of +6.2pp positive delta), which is the check being brittle on a small effect, not a red flag.
Mechanism: score-weighting concentrates the sleeve into the top-ranked name, whose rank is the noisiest; equal
weight is a shrinkage. Still needs the 2x-cost check (queued) and a look at whether the live sleeve's weights
match the backtest's before it means anything live. (b) The overlay's trigger threshold barely matters on the 26yr
(0.90 vs 0.95: +0.85 vs +0.77pp, 16 vs 14 years) — the churn premium is not where the calm-year cost comes from;
the cost is the de-risking itself being early/wrong in V-shaped dips (2023, 2024). (c) overlay + equal-weight
stacks additively (+0.045 = +0.036 + +0.009). (d') Threshold 0.85 re-run cleanly in 747s (24 x 28s) — the 2h20m stall of the first 26yr batch was NOT the arm (BUGS E-059b: unexplained one-off stall at 100% CPU, not reproducible; per-start timing now printed so a repeat is caught in minutes). (d) A 20-day vol lookback is worse than 40 on both horizons — the
stage-1 +0.037 (8 starts) did not survive 24 starts; another reminder of the many-start rule.

### Batch 5 (stage 2, 24 starts) — cost 2x, threshold sweep, exposure-matched, combos. PRELIMINARY, UNAUDITED (2026-09-16 evening).

| arm | 8yr dCAGR / dSharpe (+) / dMaxDD | 8yr yrs · top-yr | 26yr dCAGR / dSharpe (+) / dMaxDD | 26yr yrs · top-yr | GATE 8yr / 26yr |
|---|---|---|---|---|---|
| overlay_down_thr0.80 / 0.75 | +2.7 / +2.6pp · +0.083 / +0.076 (24/24) · +4.8 / +4.5pp | 4/8 | +1.0 / +1.1pp · +0.036 / +0.037 (24/24) · +4.9 / +4.7pp | 15 / 13 of 25 | threshold irrelevant (0.75..0.95 identical) |
| overlay_down_thr0.85_L1.65 (exposure-matched) | +3.45pp / +0.069 (24/24) / +3.4pp | 5/8 · 54% | +1.51pp / +0.033 (24/24) / +2.9pp | **19/25 · 19%** | FAIL (conc) / PASS |
| **overlay_down_thr0.85 + mom_equal** | +3.25pp / +0.100 (24/24) / +5.1pp · CI_lo +0.006 | 5/8 · 67% | **+1.19pp / +0.044 (24/24) / +5.3pp · CI_lo +0.006, P 98%** | 17/25 · 31% | FAIL (conc only) / PASS |
| overlay_down_thr0.90 + mom_equal | +2.62pp / +0.089 (24/24) / +5.1pp | 4/8 · 73% | +1.14pp / +0.045 (24/24) / +5.5pp | 16/25 · 31% | FAIL (conc) / PASS |

Cost 2x (arm_cost2 vs base_cost2, 24 starts) — all three survivors INTACT: mom_equal +0.52pp/+0.013 (23/24) and
+0.29pp/+0.009 (24/24, 19/25 years); overlay_down+mom_equal +1.99pp/+0.078/+5.7pp and +0.99pp/+0.042/+5.5pp;
overlay_down_thr0.85 +2.58pp/+0.084/+5.3pp and +0.84pp/+0.033/+4.6pp.

**Ex-crisis test** (calendar 2020 removed on the 8yr; 2008 + 2020 removed on the 26yr, from BOTH arms; 24 starts).
This is the direct answer to "is the gain just one year":

| arm | 8yr dSharpe ex-2020 (+) · d ann. return | 26yr dSharpe ex-2008/2020 (+) · d ann. return |
|---|---|---|
| mom_equal | +0.013 (24/24) · +0.42pp | +0.009 (24/24) · +0.25pp |
| overlay_down | +0.011 (18/24) · -0.74pp | +0.010 (20/24) · -0.25pp |
| overlay_down + mom_equal | +0.024 (24/24) · -0.31pp | +0.020 (24/24) · +0.00pp |
| overlay_down_thr0.85 + mom_equal | +0.041 (24/24) · +0.62pp | +0.019 (24/24) · +0.17pp |
| exit_all | +0.039 (22/24) · -2.14pp | +0.040 (22/24) · -2.06pp |

Read: (1) equal-weight momentum's edge has NOTHING to do with crises — identical inside and outside them (24/24 on
both horizons either way), and it survives 2x cost; it is a small, honest shrinkage improvement (+0.3-0.5pp,
+0.01 Sharpe). (2) The de-risk-only overlay outside the crisis years is Sharpe-flat to slightly positive
(+0.01) at a return cost of -0.25..-0.75pp/yr — i.e. CHEAP insurance, not free, and not a return engine; inside
2008/2020 it pays +10..+15pp. The 8yr gate's year-concentration failure is the arithmetic of one crash in eight
years, not a red flag; the 26yr passes every check including the year test (15-17/25) and, combined with
mom_equal, the bootstrap CI excludes zero on BOTH horizons. (3) The exposure-matched version (1.65x) is the
most even on the 26yr (19/25 years, top-year 19%) but gives back most of the MaxDD gain (+2.9pp) — it converts
the insurance into return, which is a leverage choice, not evidence. (4) exit_all keeps a Sharpe edge ex-crisis
only by cutting return -2pp/yr; dead.

**Standing verdict (2026-09-16, end of batch 5):** two candidates deserve the deployment conversation, neither
as a return claim: (A) equal-weight momentum picks — tiny, consistent, cost-proof, crisis-independent; (B) the
de-risk-only prompt overlay (any threshold 0.85-0.95; +mom_equal) — drawdown insurance worth ~+5pp MaxDD on
both horizons and ~+1pp CAGR on the 26yr, costing ~0.5pp/yr in calm years; the 8yr CAGR gain (+2-3pp) is 2020
and must not be quoted as expected return. NOT deployed; owner's call. Next batch: value/lowvol sleeves equal
weight, momentum count under equal weight (4/6/7), and a DAILY (not weekly) de-risk overlay at 2x cost.

### Batch 6 (stage 2, 24 starts) — daily overlay, sleeve equal weights, momentum count. PRELIMINARY, UNAUDITED (2026-09-16 night).

| arm | 8yr dCAGR / dSharpe (+) / dMaxDD · yrs | 26yr dCAGR / dSharpe (+) / dMaxDD · yrs | GATE 8yr / 26yr | read |
|---|---|---|---|---|
| overlay_daily (every session, not just tranche days) | +1.11pp / +0.069 (24/24) / +5.0pp · 3/8 | +0.61pp / +0.037 (24/24) / +4.6pp · 11/25 | FAIL / FAIL (yrs 44%) | no better than weekly (+0.070/+0.036), more calm-year cost (2024 -8.2pp), 2x-cost +0.032. Cadence beyond 5 sessions adds nothing — DEAD |
| overlay_daily + mom_equal | +1.63pp / +0.082 / +5.2pp · 3/8 | +0.91pp / +0.046 / +5.0pp · 13/25 | — | same |
| vl_equal (value + lowvol sleeves equal-weighted) | +0.07pp / +0.006 (22/24) / +0.6pp | +0.09pp / +0.004 (23/24) / -0.3pp · 16/25 | — | ~0 — DEAD |
| all_equal (mom + value + lowvol equal) | +0.60pp / +0.019 (23/24) / +1.2pp · 5/8 | +0.36pp / +0.013 (24/24) / -0.1pp · 15/25 | PASS / PASS (CI_lo -0.001) | = mom_equal + ~0.004; keep the simpler mom_equal |
| mom_equal_n4 | -3.51pp / **-0.073 (0/24)** | -0.69pp / -0.012 (2/24) | — | fewer names = clearly worse |
| mom_equal_n6 | +1.73pp / +0.026 (22/24) · 6/8 | -0.21pp / -0.012 (2/24) · 15/25 | PASS / FAIL | 8yr-only mirage; 5 names stands |
| mom_equal_n7 | +1.04pp / +0.012 (15/24) | -0.78pp / -0.030 (0/24) | — | dead |

Read: the tranche-day (weekly) cadence is the right one for the overlay — daily de-risking trades more and helps
less; the value/lowvol sleeves' score-weighting is harmless; the momentum sleeve's count of 5 is confirmed on the
long horizon (n=6's 8yr gain is another one-horizon result). Batch 7 (running): the LIVE-PARITY cap. The deployed
signal server caps a name at 15% of the book; the clean room's combiner caps at 10% (accepted deviation,
AUDIT_enginegap_v2). Under a 15% cap an equal-weighted 14% momentum name is NOT capped while a score-weighted top
name IS, so mom_equal's mechanism may differ live. Arms: cap0.15 base, +mom_equal, +overlay_down, all three,
and cost-2x versions — every candidate must be re-measured against the cap-0.15 base before any live claim.

### Batch 7 (stage 2, 24 starts) — LIVE-PARITY combiner cap 0.15. PRELIMINARY, UNAUDITED (2026-09-17 00:38).

The cap itself is inert: cap0.15 vs the clean-room base = -0.006 / +0.003 Sharpe (8yr / 26yr), so the clean-room's
10% convention and the live 15% cap describe the same book. Every candidate re-measured AGAINST the cap-0.15 base:

| arm (all with cap 0.15) | 8yr dCAGR / dSharpe (+) / dMaxDD · yrs · top-yr | 26yr dCAGR / dSharpe (+) / dMaxDD · yrs · top-yr |
|---|---|---|
| + mom_equal | +0.58pp / +0.013 (22/24) / +0.8pp · 6/8 · 47% | +0.30pp / +0.009 (24/24) / +0.1pp · **18/25** · 32% |
| + mom_equal, cost 2x (vs cap0.15_cost2) | +0.57pp / +0.013 (23/24) / +0.7pp · 6/8 | +0.30pp / +0.009 (24/24) / +0.1pp · 18/25 |
| + overlay_down | +1.80pp / +0.072 (24/24) / +4.7pp · 3/8 · 76% | +0.81pp / +0.036 (24/24) / +5.6pp · 15/25 · 36% |
| + overlay_down + mom_equal | +2.51pp / +0.088 (24/24) / +5.1pp · 5/8 · 71% | **+1.15pp / +0.047 (24/24) / +5.8pp · 17/25 · 30%** |

Read: identical to the cap-0.10 measurements to the second decimal — the equal-weight edge does not depend on the
cap binding (so the mechanism is the rank-noise shrinkage, not the cap interaction I worried about), and the
overlay's insurance profile is unchanged. Both candidates are now measured under the book the live engine
actually runs. Live implementation notes: mom_equal = equal weights in `strategy1_momentum_reversal`'s output
(currently score-weighted with a 2/N cap); overlay_down = on every tranche day, for each NON-rebuilding book,
if (NAV/4 x 1.49 x vol_scale x gate) / book_gross < 0.95, trim every position by that factor (whole shares,
0.3%-of-book min trade) — never scale up. Neither deployed; owner's call.

### Batch 8 (stage 2, 24 starts) — risk parameters on top of OM = overlay_down + mom_equal. PRELIMINARY, UNAUDITED (2026-09-17 02:40).

OM levels: 8yr +34.50% / 1.108 / -31.6% (vol 31.2%), 26yr +19.24% / 0.764 / -41.0% (vol 28.3%). Deltas vs OM:

| arm | 8yr dCAGR / dSharpe (+) / dMaxDD | 26yr dCAGR / dSharpe (+) / dMaxDD · yrs | read |
|---|---|---|---|
| stop 35% | -0.43pp / -0.006 (9/24) / 0.0 | -0.23pp / -0.005 (7/24) / -0.4pp · 13/25 | worse |
| stop 45% | -0.59pp / -0.025 (5/24) / +0.5pp | -0.20pp / -0.009 (5/24) / +0.3pp · 10/25 | worse |
| stop 50% | -1.53pp / -0.048 (0/24) / +0.2pp | -0.47pp / -0.017 (0/24) / +0.2pp · 12/25 | clearly worse — **40% stands** |
| vol-clamp floor 0.20 / 0.40 | 0.000 / +0.004 | 0.000 / +0.002 | the floor never binds — irrelevant |
| vol target 17% (1x) | +1.15pp / -0.013 / -1.6pp, vol 32.8% | +0.69pp / -0.003 / -2.4pp, vol 29.7%; vol-matched CAGR -0.3pp | a leverage dial, not an edge |
| vol target 13% (1x) | -0.88pp / +0.029 (24/24) / +1.9pp, vol 29.3% | -0.83pp / +0.005 / +2.4pp, vol 26.6%; vol-matched CAGR +0.3pp | same dial, other direction |

Read: every risk parameter of the deployed book is at its frontier on the v2 data — the trailing stop at 40%,
the vol-scale floor (inert), and the 15% x 1.49 vol target (moving it only trades CAGR for MaxDD along the
leverage line; vol-matched CAGR changes by ±0.3pp, i.e. nothing). No promotion. Batch 9 (running): the sleeve
split 70/21/9 re-checked under OM (60/30/10, 80/15/5, 50/35/15) and inverse-vol weighting inside the momentum
sleeve as the next shrinkage step after equal weight.

### Batch 9 (stage 2, 24 starts) — sleeve split under OM; inverse-vol inside the momentum sleeve. PRELIMINARY, UNAUDITED (2026-09-17 04:00).

Pairwise gate (`GATE059_pair.py`, same checks as GATE059 but against any base; adds the ex-crisis dSharpe):

| arm vs base | 8yr dCAGR / dSharpe (+) · yrs · conc · ex-crisis | 26yr dCAGR / dSharpe (+) · yrs · conc · ex-crisis | gate |
|---|---|---|---|
| **OM_s80_15_5 vs OM** (sleeves 80/15/5) | +1.04pp / +0.014 (23/24) · 7/8 · 33% · +0.007 (20/24) | +0.47pp / +0.014 (24/24) · 17/25 · 20% · +0.016 (24/24) | **PASS / PASS** (P 82% / 91%) |
| OM_s60_30_10 vs OM | -0.16pp / +0.013 (24/24) · 4/8 | -0.30pp / -0.003 (3/24) · 11/25 | FAIL / FAIL |
| OM_s50_35_15 vs OM | -0.88pp / +0.009 (20/24) · 2/8 | -0.77pp / -0.012 (0/24) · 9/25 | dead |
| mom_ivol vs base (inverse-vol weights in the momentum sleeve) | -1.14pp / -0.008 (5/24) · 3/8 | see gate line above | dead — worse than equal weight |
| O_ivol vs overlay_down | -1.29pp / -0.013 (5/24) | see gate line above | dead |

Read: inverse-vol inside the sleeve is the wrong shrinkage (it under-weights exactly the high-vol winners momentum
lives on); equal weight stays. The sleeve split is NOT at its frontier under equal-weight momentum: shifting
70/21/9 -> 80/15/5 adds +0.014 Sharpe on both horizons with the right consistency (17/25 years, 20% top-year
share, ex-crisis positive 24/24) and the same realized vol. CAUTION: the direction (more momentum) is the one the
8yr always rewards, and 70/21/9 itself was fitted on this data in cycle 5x; batch 10 (running) checks whether the
gain keeps growing toward pure momentum (90/7/3, 100/0/0 — if it does, the value/lowvol sleeves are just a
drag and the finding is "less diversification", which is a different and riskier claim), isolates the split
from the overlay and from equal weight, and adds the 2x-cost arm.

### Batch 10 (stage 2, 24 starts) — momentum-share sweep; the split isolated; cost 2x. PRELIMINARY, UNAUDITED (2026-09-17 05:30).

| arm vs base | 8yr dCAGR / dSharpe (+) · yrs · sub 2023-26 | 26yr dCAGR / dSharpe (+) · yrs · sub 2023-26 | pairwise gate |
|---|---|---|---|
| OM_s80_15_5 vs OM | +1.04pp / +0.014 (23/24) · 7/8 · -0.006 | +0.47pp / +0.014 (24/24) · 17/25 · -0.011 | PASS / PASS |
| OM_s90_7_3 vs OM | +1.26pp / +0.016 (20/24) · 5/8 · **-0.042** | +0.66pp / +0.028 (24/24) · 15/25 · **-0.046** | FAIL / FAIL (recent sub-period) |
| OM_s100 (pure momentum) vs OM | +0.58pp / -0.001 (10/24) · 4/8 · **-0.101** | +0.17pp / +0.023 (24/24) · 9/25 · **-0.107** | FAIL / FAIL |
| s80_15_5 vs base (split alone) | +0.63pp / +0.008 (18/24) · 5/8 | +0.31pp / +0.011 (24/24) · 12/25 | FAIL / FAIL (weak) |
| M_s80_15_5 vs mom_equal (split + equal, no overlay) | +0.57pp / +0.006 (18/24) | +0.28pp / +0.010 (24/24) · 12/25 | FAIL / FAIL |
| OM_s80_15_5_cost2 vs OM_cost2 | +1.11pp / +0.017 (24/24) · 7/8 | +0.54pp / +0.017 (24/24) · 17/25 | PASS / PASS — intact at 2x cost |

Read: (1) the momentum share is NOT monotonic — the sleeve gain plateaus at 80-90 and pure momentum gives it
back; the value/lowvol sleeves earn their place. (2) Every step beyond 80 fails on the SAME check: the 2023-26
sub-period goes negative and gets worse the heavier the tilt (-0.006 -> -0.042 -> -0.101 on the 8yr, the 26yr
identical). The momentum tilt is fading in the most recent regime; 80/15/5 is the last point the evidence
supports, and even it is slightly negative there. (3) The split needs the overlay: alone (or with equal weight
only) it is +0.008-0.011 on 12/25 years and fails; under the de-risk overlay it is +0.014 on 17/25. Mechanism:
the overlay handles the drawdown control the value/lowvol sleeves used to provide, so the book can carry more
momentum — a re-allocation of the same risk budget, not new alpha.

### THE PACKAGE (2026-09-17): overlay_down + mom_equal + sleeves 80/15/5, vs the DEPLOYED book (24 starts)

| horizon | levels base -> package | dCAGR | dSharpe (+) · bootstrap 90% CI | dMaxDD | yrs · top-yr | ex-crisis dSharpe | GATE059 |
|---|---|---|---|---|---|---|---|
| 26yr | +18.16%/0.719/-46.6% -> +19.71%/0.779/-41.2% | +1.55pp | +0.060 (24/24) · [+0.019, +0.110], P 99% | +5.4pp (21/24) | 17/25 · 23% | +0.035 (24/24) | **PASS** (every check) |
| 8yr | +32.42%/1.026/-36.7% -> +35.54%/1.122/-31.4% | +3.12pp | +0.096 (24/24) · [-0.006, +0.264], P 93% | +5.3pp (24/24) | 5/8 · 67% (2020) | +0.031 (23/24) | FAIL on year-concentration only |
| 26yr at 2x cost | +16.27%/0.664/-47.8% -> +17.80%/0.722/-42.6% | +1.53pp | +0.058 (24/24) · [+0.018, +0.108] | +5.2pp | 16/25 · 23% | +0.034 (24/24) | PASS |

Honest framing for the owner: the 26yr number (+1.5pp CAGR, +0.06 Sharpe, +5pp MaxDD, 17/25 years, CI
excluding zero, intact at 2x cost, positive ex-crisis) is the expectation to quote. The 8yr +3.1pp is 2020-heavy
(67%) and is NOT an expected-return figure. Each component is small and honest: equal weight = rank-noise
shrinkage (+0.01), overlay = cheap crisis insurance (+0.035, costs ~0.3-0.7pp/yr in calm years), split =
re-allocating the risk budget the overlay frees (+0.014, fading in 2023-26). Not deployed — owner's call.
Live changes if chosen: (a) equal weights in strategy1's output, (b) PROD_WEIGHTS_BULL 0.80/0.15/0.05 with the
signal server's 15% cap, (c) a tranche-day de-risk trim of the non-rebuilding books in ibkr_engine (never up).
Caveat under test (batch 11): at $60k the value/lowvol names at 1.5%/0.5% of a $15k book are $225/$75 — whole
shares truncate most of them to 0; the backtest starts at $50k and compounds, so it only sees this early.

### Batch 11 (stage 2, 24 starts) — CAPITAL SCALE / whole-share truncation. PRELIMINARY, UNAUDITED (2026-09-17 07:10).

The clean room sizes whole shares from a starting NAV (base $50k) and compounds. Same configs re-run from $15k
(books of $3.75k = a quarter of the live account), $60k (= live: four $15k books) and $1M:

| config | 8yr $15k | 8yr $60k | 8yr $1M | 26yr $60k | 26yr $1M |
|---|---|---|---|---|---|
| deployed base | +31.58% / 1.007 / -38.6% | +32.42% / 1.027 / -36.6% | +32.55% / 1.033 / -36.0% | +18.16% / 0.719 / -46.5% | +18.09% / 0.717 / -46.4% |
| package (overlay + equal + 80/15/5) | +34.49% / 1.106 / -31.5% | +35.61% / 1.124 / -31.4% | +35.83% / 1.127 / -31.4% | +19.70% / 0.778 / -41.3% | +19.62% / 0.775 / -41.5% |
| package - base | +2.91pp / +0.099 (24/24) / +7.1pp | +3.19pp / +0.097 / +5.2pp | +3.28pp / +0.094 / +4.6pp | +1.54pp / +0.059 / +5.3pp | +1.53pp / +0.058 / +4.9pp |

Truncation cost (8yr, $15k vs $1M): base -0.97pp CAGR / -0.026 Sharpe; package -1.34pp / -0.021. 26yr $15k: base +18.17%/0.720, package +19.79%/0.783 (package - base +0.063, 24/24). At $60k vs $1M:
base -0.13pp, package -0.22pp. The 26yr is scale-free (compounds out of the small-book regime within years).
Read: at the live size the steady-state whole-share cost is bounded at roughly -0.2 .. -1.3pp/yr (the $60k run
leaves the small regime quickly; the $15k run is a quarter of live size), it hits base and package alike, and the
package's edge over the base survives at every scale (24/24 starts even at $15k). Not a reason to prefer either.
Practical note for the live book: at 80/15/5 the value/lowvol names are 1.5% / 0.5% of a $15k book ($225 / $75)
and most round to 0-1 shares — the sleeves' diversification is partly nominal at this account size regardless of
the split (batch 12 tests fewer, larger value names for exactly this reason).

### Batch 12 (stage 2, 24 starts) — value / lowvol name counts and a 2% minimum weight under the package. PRELIMINARY, UNAUDITED (2026-09-17 09:05).

| arm vs package | 8yr dCAGR / dSharpe (+) / dMaxDD · yrs | 26yr dCAGR / dSharpe (+) / dMaxDD · yrs | read |
|---|---|---|---|
| value top 5 | +0.85pp / +0.015 (19/24) / **-2.0pp (0/24)** · 6/8 | -0.37pp / -0.015 (0/24) / -2.1pp · 14/25 | FAIL both (8yr Sharpe gain bought with drawdown; 26yr negative) |
| value top 7 | +0.59pp / +0.012 (18/24) / -0.3pp | -0.09pp / -0.004 (1/24) / -0.9pp · 10/25 | dead |
| value top 15 | -1.22pp / -0.028 (0/24) | -0.60pp / -0.017 (0/24) | dead |
| lowvol top 5 | -0.45pp / -0.018 (7/24) | -0.23pp / -0.012 (6/24) · 8/25 | dead |
| value 5 + lowvol 5 | +0.40pp / -0.003 | -0.51pp / -0.024 (0/24) / -2.8pp | dead |
| min weight 2% | -0.28pp / -0.005 (7/24) / +0.5pp | -0.49pp / +0.007 (21/24) / +1.0pp · 10/25 | inert-to-negative; not worth the live simplification |

Read: 10 value + 10 lowvol names stand; the diversifying sleeves want breadth, not concentration (fewer names =
more drawdown on every start). The live truncation of 1.5%/0.5% names to 0-1 shares is therefore a cost the small
account pays, not something to engineer away by concentrating the sleeves. Frontier search of the package's
internals is complete: cadence, exits, overlay form/threshold/lookback, gate depth, sleeve split, momentum
count/weighting, value/lowvol counts, stop, vol floor/target, cap, cost, capital scale — all measured. Batch 13
(running): the package at 1.25x and 1.65x, so the owner's decision can be made on the leverage line.

### Batch 13 (stage 2, 24 starts) — the package on the leverage line. PRELIMINARY, UNAUDITED (2026-09-17 09:51).

| horizon | book | CAGR | Sharpe | MaxDD | realized vol |
|---|---|---|---|---|---|
| 8yr | deployed structure @1.25x | +31.03% | 1.053 | -33.5% | 30.0% |
| 8yr | deployed @1.49x (live) | +32.42% | 1.026 | -36.7% | 32.5% |
| 8yr | package @1.25x | +33.77% | 1.139 | -29.2% | 29.4% |
| 8yr | package @1.49x | +35.54% | 1.122 | -31.4% | 31.6% |
| 8yr | package @1.65x | +36.55% | 1.114 | -32.7% | 32.9% |
| 26yr | deployed structure @1.25x | +17.26% | 0.730 | -43.0% | 26.8% |
| 26yr | deployed @1.49x (live) | +18.16% | 0.719 | -46.6% | 29.2% |
| 26yr | package @1.25x | +18.48% | 0.776 | -38.5% | 26.4% |
| 26yr | package @1.49x | +19.71% | 0.779 | -41.2% | 28.3% |
| 26yr | package @1.65x | +20.33% | 0.778 | -43.2% | 29.4% |

Read: Sharpe is flat along the line (leverage is a dial); the package at 1.25x ≈ the deployed CAGR with ~10pp less MaxDD
on the 26yr; at 1.65x ≈ the deployed MaxDD with ~+3pp CAGR. Owner's risk choice; the research recommendation stays
'package at the current 1.49x' because it changes nothing about the leverage the owner already accepted.

Verdict so far (honest): NOTHING passes the gate on both horizons. The de-risk-only prompt overlay is the one
candidate with a consistent, cost-robust profile — 24/24 starts better Sharpe on the 26yr, 21-24/24 better
MaxDD on both, intact at 2x cost, gate 26yr PASS — but on the 8yr it fails the year-consistency (better in 3/8
years) and year-concentration (2020 = 76% of the positive delta) checks: it is DRAWDOWN INSURANCE whose premium
(-1..-6pp in calm bull years) is repaid in 2008/2020, not a return engine. That is exactly the kind of
one-year-driven result the owner asked us to flag; it is NOT promoted on the return claim. The 26yr number
(+0.77pp CAGR, +0.036 Sharpe, +5.5pp MaxDD, 14/25 years) is the honest expectation. Exit rules (I-34) are
dead at real costs. Batch 4 (stage 2, running): overlay_down_lb20, mom_equal, their combos, and an overlay
trigger threshold (0.90/0.85) to cut the overlay's calm-year churn — the only remaining lever on its premium.


## Cycle 58 — EXP-060 · A SECOND ENGINE (opened 2026-09-17): owner asks for an improvement that shows up in MOST years, from a different return source

Owner (09-17): the EXP-059 package is good and saved, but its return gain is 2020-heavy; wants a change that is better in
most years, and a fundamentally different or additional strategy rather than more momentum tuning. Keep going until told
to stop. Base for every comparison = the PACKAGE (overlay_down + mom_equal + 80/15/5 @1.49x; cached as `P` in
`_v2_*/exp060/`), PRIMARY metric = calendar years better on BOTH horizons, then Sharpe with bootstrap CI, at 2x cost.

Data not yet used by any sleeve: IBES (WRDS) — quarterly EPS surprise scores with announcement dates (1992-2026),
monthly analyst estimate summaries (mean FY1 EPS, #up/#down), recommendation summaries; 13F holdings; Audit Analytics.
`EXP060_ibes_features.py` builds a point-in-time table (asof = IBES statistical period, surprise joined as-of <= 120d):
924,891 ticker-months, 3,877 tickers, coverage sue 70% / rev3m 97% / rec1m 76%. Ticker join (IBES OFTIC = CRSP ticker);
mismatches through renames are accepted (lost coverage, no look-ahead).

Harness `EXP060_engine.py` (EXP-059 harness + an analyst sleeve `sleeve_x`: top-10 equal-weight by one signal among
members with >= 3 estimates, positive score only; bull-sleeve weights scaled by (1 - x_w)). Batch 1 (stage 2, 24 starts):
each signal ALONE at 100% (raw sleeve quality) and the package + 15% sleeve for sue / rev3m / updown / rec1m / combo.
Live feasibility if anything works: IBES is a quarterly WRDS upload (stale); FMP has earnings-surprise and analyst-estimate
endpoints (reference_api_docs) — the live path would be FMP, to be parity-checked against IBES before any deployment.
Honesty note: earnings surprise / revisions are PUBLISHED anomalies (PEAD, Chan-Jegadeesh-Lakonishok); the owner's original
directive preferred first-principles ideas, but the 09-17 ask is explicitly for another fundamental strategy, so they are
tested as such and reported with that label.


### EXP-060 batch 1 — ANALYST SLEEVE (IBES, PIT): DEAD on the 8yr; 26yr not run. (2026-09-18, 24 starts, PRELIMINARY)

Sleeve = top-10 equal-weight among members with >= 3 estimates by one signal, positive score only. Package 8yr = +35.54% / 1.122 / -31.4%.

| signal | ALONE (100% of book, same stops/overlay/gate/tranches) | corr with package | years > deployed base | as 15% sleeve in the package: dCAGR / dSharpe (+) · years better |
|---|---|---|---|---|
| earnings surprise (SUE, <= 90d) | +9.5% / 0.466 / -45% | 0.72 | 1/8 | -0.74pp / -0.010 (5/24) · 2/8 |
| FY1 estimate revision (3m) | +14.7% / 0.609 / -55% | 0.74 | 1/8 | -0.29pp / -0.007 (7/24) · 2/8 |
| up-minus-down estimates | +8.2% / 0.423 / -51% | 0.76 | 0/8 | -1.91pp / -0.040 (0/24) · 2/8 |
| recommendation upgrades | +13.1% / 0.582 / -44% | 0.72 | 1/8 | -0.59pp / -0.008 (5/24) · 5/8 |
| rank combo (sue + rev + updown) | +14.2% / 0.604 / -51% | 0.73 | 2/8 | -0.30pp / -0.003 (9/24) · 3/8 |

Read: none of the analyst signals is a second engine. Alone they earn 8-15%/yr at Sharpe 0.4-0.6 in the same
chassis where the package earns 35% at 1.12; they are 0.72-0.76 correlated with the package's daily returns (so
not a diversifier either); and inside the package every one of them lowers Sharpe and CAGR. The recommendation
signal's 5/8 years is the only non-negative year count and it comes with -0.6pp CAGR. Killed on the 8yr without
running the 26yr (a sleeve this far below the package on one horizon cannot be rescued by the other; the
horizon rule guards promotions, not kills). Consistent with the 2018-2026 literature: PEAD and revision drift
in large/mid caps are largely arbitraged. Data and harness stay (E-060a bridge is reusable for any IBES idea).

Infrastructure note (E-059b confirmed): starts on this batch took 40s..16,000s at 99% CPU with the 26yr universe
resident (RSS 35 GB, macOS compressor 1.2B compressions, 8 GB compressed); the stalls are memory-compressor
thrash, not the code. Batch 2 (price-based engines) runs the 8yr first and the 26yr only for anything alive.


### EXP-060 batch 2 — PRICE-BASED second engines, 8yr (24 starts, PRELIMINARY). Package = +35.54% / 1.122 / -31.4%.

| engine | ALONE (100%) | corr w/ package | yrs > deployed base | as 15% sleeve in the package: dCAGR / dSharpe (+) / dMaxDD · years better |
|---|---|---|---|---|
| short-term reversal (buy 20d losers above SMA200) | +8.8% / 0.434 / -53% | 0.78 | 3/8 | -1.74pp / -0.043 (0/24) / -1.1pp · 3/8 — dead |
| calendar seasonality (same-month 10y mean) | +10.1% / 0.471 / -40% | 0.71 | 1/8 | -1.78pp / -0.041 (0/24) / +0.1pp · 3/8 — dead |
| **long-term reversal (months 13-60 losers)** | **+25.3% / 0.807 / -37%** | **0.68** | 2/8 | **+1.28pp / +0.019 (23/24) / +0.0pp · 6/8** (2019 +0.7, 2020 +0.3, 2021 +7.2, 2022 +0.9, 2023 -1.4, 2024 +6.7, 2025 +1.7, 2026 -5.2) |

Read: the two fast mean-reversion engines are dead in this chassis (they need daily rebalancing and pay the
tranche cadence's 5-20 session lag). Long-term reversal is different in kind: it is the LOWEST-correlated
sleeve found so far (0.68 vs 0.72-0.78 for everything else) and it is the first candidate that meets the owner's
09-17 criterion on the 8yr — better in 6 of 8 years with no dominant year (largest share 2021 +7.2 of +17.5pp
positive). CAUTIONS before anything more is said: (1) 8yr only; the 26yr (2001-2026, which contains the 2000-02
and 2008-09 value/reversal regimes AND the 2010s when reversal was flat) is queued; (2) reversal sleeves are
value-like and crash-prone (alone MaxDD -37%), so the 2x-cost and n/weight/uptrend sweep (batch 4) must hold;
(3) 15% at top-10 equal weight means 1.5% names = whole-share truncation at $60k (batch 11 of EXP-059 showed
this costs base and package alike, but it must be re-checked with this sleeve because its names are beaten-down,
lower-priced stocks). 26yr + sweep queued behind batches 3/3b.


### EXP-060 batches 3 / 3b — return decomposition, institutional breadth, and a DIFFERENT ASSET as the sleeve. 8yr, 24 starts, PRELIMINARY.

| sleeve | ALONE (100%) | corr | yrs > base | as sleeve in the package: dCAGR / dSharpe (+) / dMaxDD · years better |
|---|---|---|---|---|
| overnight-minus-intraday 12m momentum (Lou-Polk-Skouras) | +1.8% / 0.225 / **-70%** | 0.55 | 2/8 | 15%: -4.27pp / -0.113 (0/24) / -4.5pp · 2/8 — dead |
| overnight 12m sum alone | +21.9% / 0.759 / -51% | 0.72 | 3/8 | not run (close-to-close momentum in disguise) |
| 13F breadth: q/q change in # holders (Chen-Hong-Stein) | +19.2% / 0.733 / -40% | 0.77 | 1/8 | 15%: +0.12pp / -0.000 (13/24) · 4/8 — a zero |
| 13F change in aggregate shares | +18.9% / 0.748 / -38% | 0.71 | 1/8 | not run |
| **gold (GLD) 10%, always** | — | — | — | **+0.40pp / +0.030 (24/24) / +0.7pp · 5/8** (2019 +1.9, 2020 +0.5, 2021 -3.9, 2022 +1.7, 2023 -0.6, 2024 +2.3, 2025 +4.4, 2026 -4.3) |
| gold 10% / 20% above its SMA200 else cash | — | — | — | +0.12 / +0.13pp, +0.017 / +0.019 (24/24) · 5/8 |
| sector-ETF trend (top-3 of 11 by 6m, above SMA200) 15% | — | — | — | -0.65pp / +0.003 (16/24) · 4/8 — dead |

Read: (1) the two information-based candidates (overnight, 13F) are not second engines — the overnight signal is
a levered intraday short in disguise and breadth is momentum with a 45-day lag. (2) The first thing that behaves
like a genuine diversifier is not a stock signal at all: a 10% permanent gold sleeve raises Sharpe on every one of
24 starts, is better in 5 of 8 years, and its year pattern is gold's own (2021 -3.9, 2026 -4.3 are gold's flat
years). Timing gold with its SMA200 halves the benefit. CAUTIONS: 2019-2026 is a gold bull market (GLD roughly
+150%), the series starts 2004-11 so the 26yr test covers 21 years and the sleeve is empty before that, and the
benefit is a diversification effect (corr of GLD to the book ~0), not stock-picking — honest framing is "a 10%
strategic gold allocation inside the leverage budget", which is a portfolio-construction choice the owner may or
may not want. 26yr queued (chain5) plus 15%, cost-2x, and the reversal + gold combination.


### EXP-060 batch 4 — LONG-TERM REVERSAL sleeve sweep, 8yr (24 starts, PRELIMINARY). All vs the package.

| variant | dCAGR | dSharpe (+) | dMaxDD | years better | note |
|---|---|---|---|---|---|
| 10% | +0.89pp | +0.015 (24/24) | 0.0 | 5/8 | |
| 15% | +1.28pp | +0.019 (23/24) | 0.0 | 6/8 | |
| 20% | +1.48pp | +0.028 (24/24) | 0.0 | 6/8 | 2026 cost -7.3pp |
| 25% | +1.80pp | +0.034 (24/24) | +0.1 | 6/8 | 2026 cost -8.8pp |
| 15%, top-5 | +0.77pp | +0.013 (21/24) | +0.1 | 4/8 | concentration hurts |
| 15%, top-15 | +1.69pp | +0.025 (24/24) | 0.0 | 6/8 | breadth helps |
| **15%, only names above SMA200** | **+2.61pp** | **+0.056 (24/24)** | 0.0 | **6/8** | the "recovering loser" filter |
| 15% at 2x cost (vs package at 2x) | +1.13pp | +0.021 (24/24) | +0.1 | 6/8 | cost-robust (quarterly-ish turnover) |

Read: the sleeve is monotonic in weight with the melt-up-year cost growing with it (2026 -3.5 -> -8.8pp); breadth
beats concentration; the edge survives 2x cost. The uptrend filter (buy the 3-5 year losers that have already
turned, i.e. above their 200-day average) roughly triples the Sharpe gain — economically that is "value with a
catalyst", the classic momentum-of-value fix for the falling-knife problem, and it is exactly the kind of
combination that can be fitted on 8 years, so it is treated as UNPROVEN until the 26yr (chain 7). Order of 26yr
runs: ltr alone + 15% (running), gold, ltr 20/25, ltr-trend.


### EXP-060 — long-term reversal sleeve on the 26yr: DEAD. (2026-09-18 19:52, 24 starts)

| | 8yr | 26yr |
|---|---|---|
| LTR alone (100%) | +25.3% / 0.807 / -37%, corr 0.68, 2/8 yrs > base | **+6.9% / 0.369 / -80%**, corr 0.70, 10/25 yrs > base |
| package + 15% LTR vs package | +1.28pp / +0.019 (23/24) · 6/8 yrs · ex-crisis +0.022 | **-1.06pp / -0.041 (0/24) / -4.0pp MaxDD · 11/25 yrs** · sub-periods 2001-08 -0.045, 2009-16 -0.027, 2017-22 -0.075, 2023-26 +0.010 |

26yr per-year delta: the sleeve adds in 2006, 2010, 2012-13, 2016, 2018, 2021, 2024 and costs 5-13pp in 2005, 2007,
2011, 2014-15, 2017, 2019 — the classic value/reversal cycle, net negative over 26 years and negative in three of
four sub-periods. The 8yr result was the 2021 and 2024 value rallies. Verdict: the plain 13-60-month reversal
sleeve is a regime bet, not a second engine; it fails the owner's most-years criterion on the long horizon and the
pairwise gate on both. The 20%/25% 26yr runs are cancelled (monotonically worse). Still open: the uptrend-FILTERED
reversal (batch 7 + its 26yr), which is a different rule (losers that have already turned), and gold (chain 5).


### EXP-060 batch 5 — gold sleeve refinements, 8yr (24 starts, PRELIMINARY). Vs the package.

| arm | dCAGR | dSharpe (+) | dMaxDD | years better | vol |
|---|---|---|---|---|---|
| gold 10% always | +0.40pp | +0.030 (24/24) | +0.7pp | 5/8 | 30.8% vs 31.6% |
| gold 15% always | +0.51pp | +0.035 (24/24) | +0.8pp | 5/8 | 30.8% |
| gold 10% at 2x cost (vs package at 2x) | +0.29pp | +0.026 (24/24) | +0.7pp | 5/8 | — |
| reversal 15% + gold 10% | +1.45pp | +0.054 (24/24) | +0.7pp | 6/8 | additive; the reversal half is dead on the 26yr |

Read: a small permanent gold allocation lowers the book's vol by ~0.8pp and raises Sharpe on every start at 1x and
2x cost; the weight barely matters between 10 and 15. The 26yr (2004-11 onward for the sleeve, i.e. gold's 2005-11
bull, 2012-15 bear, 2016-19 flat, 2020-26 bull) decides whether "5/8 years" is gold's 2019-26 run or a property
of the diversification. Running.


### EXP-060 — GOLD SLEEVE on the 26yr: PASSES the pairwise gate. (2026-09-18 20:48, 24 starts, PRELIMINARY)

| arm vs package | 26yr dCAGR / dSharpe (+) / dMaxDD | years better | top-yr share | sub-periods (01-08 / 09-16 / 17-22 / 23-26) | ex-crisis | gate |
|---|---|---|---|---|---|---|
| **gold 10% always** | **+0.28pp / +0.020 (24/24) / +0.8pp** | **14/25** (14/22 since the sleeve exists) | 18% | +0.034 / +0.009 / +0.002 / +0.054 | +0.022 (24/24) | **PASS** (8yr: +0.030, 5/8 yrs, fails only on 41% concentration of a +0.4pp mean) |
| gold 10% above SMA200 else cash | +0.30pp / +0.017 (24/24) / +0.5pp | 12/25 | 23% | +0.025 / +0.022 / -0.003 / +0.037 | +0.020 | FAIL (years) — timing gold hurts |

Levels 26yr: package +19.71% / 0.779 / -41.2% -> +19.99% / 0.799 / -40.4%. Per-year delta: gold adds 1-5pp in
2005-07, 2010-11, 2014, 2019, 2022, 2024-25 and costs in 2013 (-9.1, gold's crash year), 2021 (-3.6), 2026 (-4.2),
2018 (-1.7). Read: this is the first candidate in EXP-060 that meets the owner's criterion on BOTH horizons — better
in most years, no dominant year, positive in every sub-period, intact at 2x cost (8yr) — and it is exactly what it
looks like: a 10% strategic allocation to an uncorrelated asset inside the leverage budget (the sleeve's names get the
same 40% trailing stop and the same de-risk overlay). The SIZE of the effect is small (+0.02 Sharpe, +0.3pp CAGR,
+0.8pp MaxDD on the 26yr; the 8yr's +0.03/+0.4pp/+0.7pp is gold's 2019-26 run and should not be quoted). It does not
change the character of the book. Owner's call; live implementation would be a fixed 10% GLD line in every book (2-5
whole shares per $15k book at today's price), rebuilt on the book's own tranche day. Queued: 15% weight and 2x-cost
on the 26yr (chain 9). Not a reason to stop looking for a real second engine.


### EXP-060 batch 7 — UPTREND-FILTERED long-term reversal (13-60m losers that are above their SMA200), 8yr. PRELIMINARY.

| arm vs package | starts | dCAGR | dSharpe (+) | dMaxDD | years better | per-year delta |
|---|---|---|---|---|---|---|
| 15%, top-10 | 24 | +2.61pp | +0.056 (24/24) | 0.0 | 6/8 | 2019 +0.7, **2020 +9.0**, 2021 +5.4, 2022 +2.8, 2023 -3.9, 2024 +6.5, 2025 +2.4, 2026 -3.7 |
| 15%, top-15 | 24 | +2.34pp | +0.040 (24/24) | 0.0 | 6/8 | 2020 +6.3, 2021 +11.9 |
| 20%, top-10 | 24 | +3.20pp | +0.069 (24/24) | 0.0 | 6/8 | 2020 +11.9 |
| 25%, top-10 | 8 | +3.70pp | +0.083 (8/8) | +0.2 | 6/8 | |
| 15% at 2x cost (vs package at 2x) | 8 | +2.34pp | +0.051 (8/8) | +0.2 | 6/8 | cost-robust |
| sleeve ALONE (100%) | 8 | +30.6% / 0.948 / -35.8% | corr 0.71 | | 3/8 yrs > base | the best standalone sleeve in EXP-060 |

Read: on the 8yr this is the strongest thing the search has produced — every start, 6/8 years, no drawdown cost,
cost-robust, and the sleeve alone is a respectable strategy (the plain version alone was 25%/0.81; the filter adds
5pp and 0.14). But its biggest year is 2020 (+9pp of +19pp positive), i.e. the post-crash recovery of beaten-down
names that had turned up — exactly the regime bet that killed the plain sleeve on the 26yr (-0.041). The filter
may or may not fix the 2005/2007/2011/2015/2017/2019 losses. The 26yr (24 starts) is running and is the verdict;
nothing is claimed until it lands. Infrastructure: stalls resolved (battery + nice), throughput back to normal.


### EXP-060 batch 8 — VIX-futures tail hedge (VIXM, from 2011) as a sleeve, 8yr stage 1 (8 starts). Vs the package.

| arm | dCAGR | dSharpe (+) | dMaxDD | years better | 2020 | 2023 |
|---|---|---|---|---|---|---|
| VIXM 3% always | -0.00pp | +0.015 (7/8) | +0.6pp | 5/8 | +1.6 | -2.4 |
| VIXM 5% always | -0.22pp | +0.018 (7/8) | +0.9pp | 5/8 | +1.9 | -4.2 |
| VIXM 5% only when below its SMA50 (contango proxy) | -0.01pp | +0.017 (7/8) | +0.8pp | 5/8 | +2.5 | -4.0 |

Read: a permanent VIX-futures line behaves like gold's weaker cousin — a small Sharpe/MaxDD gain paid for by
negative carry in calm years (2023). It adds nothing gold does not add more cheaply, and VIXM's history starts in
2011 (no 26yr test possible before then). Dead as a candidate; not promoted to the 26yr.


### EXP-060 — uptrend-filtered reversal on the 26yr: DEAD. (2026-09-20 15:50, 24 starts)

| vs package | 8yr | 26yr |
|---|---|---|
| 15%, top-10, above SMA200 | +2.61pp / +0.056 (24/24) / 0.0 · 6/8 yrs · gate PASS | **-0.49pp / -0.024 (0/24) / -3.2pp MaxDD · 11/25 yrs** · subs -0.023 / -0.017 / -0.042 / +0.010 · ex-crisis -0.034 (0/24) · gate FAIL |

26yr per-year delta: +7.4 (2006), +7.5 (2012), +11.7 (2013), +4.1 (2020), +5.3 (2021), +5.8 (2024) against -5.1
(2007), -7.6 (2011), -8.8 (2015), -8.5 (2017), -9.3 (2019), -4.2 (2023): the identical value-cycle signature as the
plain sleeve, the filter only trims the losses by ~1pp. The 8yr result (2018-26) is one favourable half-cycle.
Verdict: the reversal family is closed — a regime bet, not a second engine; the many-start / two-horizon rule did
its job again (the 8yr had 24/24 starts and a gate PASS). Remaining from this search: the gold line (passes) and
the two candidates below.


### EXP-060 — gold sleeve, 26yr completion (2026-09-20 17:02, 24 starts). Vs the package (or package at 2x cost).

| arm | dCAGR | dSharpe (+) | dMaxDD | years better | sub-periods | ex-crisis | gate |
|---|---|---|---|---|---|---|---|
| gold 10% always | +0.28pp | +0.020 (24/24) | +0.8pp | 14/25 | all > 0 | +0.022 | PASS |
| gold 15% always | +0.32pp | +0.023 (24/24) | +0.9pp | 15/25 | all > 0 | +0.026 | PASS |
| gold 10% at 2x cost (vs package at 2x) | +0.26pp | +0.018 (24/24) | +0.8pp | 14/25 | all > 0 | +0.021 | PASS |

Read: the gold line is the one addition from the second-engine search that satisfies the owner's criterion on
both horizons; 15% is marginally better than 10% everywhere and the effect is monotone and small. Candidate for
the owner's decision alongside the EXP-059 package: a fixed 10-15% GLD line per book.


### EXP-060 batch 9 — net share issuance and a 10y Treasury line, 8yr stage 1 (8 starts). Vs the package.

| arm | dCAGR | dSharpe (+) | dMaxDD | years better | read |
|---|---|---|---|---|---|
| net repurchasers alone (100%) | +19.0% / 0.748 / -50%, corr 0.74 | | | 3/8 vs base | a weak value-like sleeve |
| repurchasers 15% | +0.64pp | +0.015 (8/8) | -0.9pp | 3/8 | dead on the years criterion |
| 10y Treasury 10% / 20% (synthetic TR from FRED DGS10) | -0.05 / -0.10pp | +0.011 / +0.013 (6/8) | +0.2 / +0.3pp | 4/8 | ~zero; 2022 shows the bond/equity correlation flip |
| Treasury 10% + gold 10% | +0.30pp | +0.038 (8/8) | +1.0pp | 5/8 | = the gold half; the Treasury half adds nothing |

Read: buybacks are priced (and correlated 0.74 with the book); Treasuries are a zero at this leverage and
horizon — the diversification benefit gold provides is not available from duration. Both dead.


### EXP-060 batch 10 — ACCOUNTING-RISK EXCLUSION (Audit Analytics, PIT by filing date), 8yr stage 1 (8 starts). Vs the package. INTERIM.

Screen: names with an ineffective SOX-404 ICFR opinion, an adverse restatement, or a fraud/SEC-investigation restatement
filed inside the trailing window are dropped from every sleeve's candidate list BEFORE truncation (sleeves pick top_n+3,
drop, truncate, renormalise — so the filter never shrinks or concentrates a sleeve). Flags 3-8% of the S&P 500 at any date.

| window | dCAGR | dSharpe (+) | dMaxDD | years better | per-year delta |
|---|---|---|---|---|---|
| 365 days | +2.14pp | +0.050 (8/8) | 0.0 | **7/8** | 2019 +2.4, 2020 +2.3, 2021 +6.5, 2022 +0.4, 2023 +2.3, 2024 -4.9, 2025 +7.8, 2026 +0.2 |
| 730 days | -4.12pp | -0.080 (0/8) | -0.5pp | 4/8 | **2021 -34.9** |
| 365 days, adverse/fraud restatements only | +1.48pp | +0.035 (8/8) | -0.2pp | 7/8 | 2020 +2.7, 2021 +7.6, 2025 +2.4; worst year -0.3 |
| 365 days, ineffective ICFR only | +1.59pp | +0.037 (8/8) | +0.1pp | 6/8 | 2024 -5.0, 2025 +7.8 |

Diagnostic of the 2021 swing: the 730-day window (but not the 365-day one) excluded PDC Energy (PERMNO 62341), which
returned +137% in 2021 while the 5-name momentum sleeve held it at ~24% of NAV — one name, -33pp. Read: with a 5-name
momentum sleeve every exclusion rule is partly a lottery on single names; the 365-day window's 7/8 years is exactly
the owner's profile but on 8 starts it can be the same lottery in the other direction. Not claimed. The two components are each positive on 8/8 starts and roughly additive (+0.035 + +0.037 ≈ +0.050), which
is mild evidence the effect is not one name — but 8 starts is 8 starts. Queued: 24 starts on the 8yr and the 26yr
(AC-gated) for the combined 365-day window (chain 13).


### EXP-060 batch 11 — investment + gross-profitability sleeve (in-universe PIT fundamentals), 8yr stage 1 (8 starts). Vs the package.

| arm | ALONE (100%) | corr | yrs > base | in package: dCAGR / dSharpe (+) / dMaxDD · years better |
|---|---|---|---|---|
| composite (low asset growth + high GP/assets) | +16.0% / 0.648 / -53% | 0.68 | 1/8 | 15%: +0.42pp / +0.016 (7/8) / -1.6pp · 3/8 |
| composite, above SMA200 only | — | — | — | 15%: +0.81pp / +0.028 (8/8) / -1.4pp · 5/8 (2026 -6.0) |
| gross profitability alone | +2.7% / 0.235 / -49% | 0.68 | 1/8 | 15%: -2.62pp / -0.057 (0/8) · 2/8 |
| low investment alone | +4.7% / 0.301 / -57% | 0.65 | 1/8 | not run |

Read: the classic investment/profitability axes are dead in this universe and chassis — alone they are worse than the
market, and inside the package they add drawdown for a Sharpe crumb in fewer than most years. Dead.


### EXP-060 — accounting-risk exclusion (365d) on 24 starts, 8yr: GATE PASS. (2026-09-20 22:43)

+1.48pp CAGR / +0.034 Sharpe (19/24 starts) / +0.2pp MaxDD vs the package; better in 7/8 calendar years (2019 +2.5, 2020 +2.3,
2021 +4.1, 2022 +0.5, 2023 +2.9, **2024 -7.4**, 2025 +5.4, 2026 +0.1); top-year share 30%; sub-periods +0.062 / +0.043 /
+0.015; odd-month (used) starts +0.039 vs even-month (untouched) +0.030; ex-2020 +0.028 (18/24); bootstrap P 79%, CI
[-0.041, +0.112]. Per start, 14/24 starts are better in >= 6 of 8 years and none in fewer than 4. Honest read: weaker
than the 8-start preview (19/24 not 8/8) and the CI spans zero, but the shape is the one the owner asked for — small,
spread across years, not one crash. It is a NEGATIVE screen (avoid accounting-risk names), so its mechanism is
plausible ex ante and it costs nothing in turnover or exposure. 26yr (24 starts, AC power) running — the screen only
exists from 2004 (ICFR) / 1995 (restatements), so the 2001-03 years are unaffected by construction.
Diagnostic of the 2024 loss (-7.4pp): the screen excluded several 2024 momentum winners whose flags were benign in
hindsight — Sprouts (+166%, adverse restatement), Axon (+130%, ICFR failure), Modine (+91%), GoDaddy (+86%), Deckers
(+80%), NRG (+78%); the median flagged >$2B name returned +9.5% in 2024. So the screen's cost is concentrated in
melt-up years where a flagged name happens to be the leader, and its benefit is spread across the other years. That
is the expected shape of a negative screen and the reason the 26yr matters: it must show the same spread across
2004-2017, where the 8yr has no information.


### EXP-060 batch 12 — in-universe negative screens (drop the top decile by net issuance / by asset growth), 8yr stage 1. DEAD.

| screen | dCAGR | dSharpe (+) | dMaxDD | years better | 2024 |
|---|---|---|---|---|---|
| heaviest 10% issuers excluded | -9.68pp | -0.212 (0/8) | -6.7pp | 3/8 | -36.5 |
| highest 10% asset growth excluded | -10.94pp | -0.250 (0/8) | -6.6pp | 3/8 | -53.6 |
| both | -10.53pp | -0.228 (0/8) | -5.5pp | 3/8 | -48.9 |
| issuers excluded + accounting screen (vs accounting screen) | -9.15pp | -0.192 (0/8) | -5.2pp | 3/8 | -32.6 |

Read: the momentum leaders ARE the heavy issuers and fast asset growers (the 2024 winners raised capital and grew
assets), so a fundamental "quality" screen anti-selects the engine and costs 10pp/yr. The contrast with the
accounting-risk screen is the point: accounting flags (restatements, control failures) are ~orthogonal to the
momentum rank, which is why that screen can remove risk without removing the return. Dead; not re-tested.


### EXP-060 — accounting-risk exclusion (365d) on the 26yr, 24 starts: GATE FAIL. (2026-09-20 23:02)

+0.35pp CAGR / +0.013 Sharpe (24/24 starts, all positive) / +0.8pp MaxDD; better in 14/25 years (2001-04 0/3, 2005-26
14/22); top-year share 18%; sub-periods +0.080 / **-0.029** / -0.006 / +0.009; even-month +0.015; ex-crisis -0.005 (3/24);
bootstrap P 67%, CI [-0.041, +0.069]. Per-year: strong 2006 +8.5, 2008 +8.5, 2011 +4.7, then a run of losses 2013 -7.2,
2014 -3.1, 2015 -7.5, 2017 -7.5 (the same "flagged names were the winners" mechanism as 2024), then positive again
2019-2025. Read: the screen is sign-consistent across starts (a real, small effect) but it is not "better in most years"
on the long horizon and its edge disappears outside the crisis years. Verdict: NOT a second engine; a
momentum-orthogonal negative screen with a small positive expectation and a bull-market cost. Not promoted. Components
(adverse-only, ICFR-only) and 2x-cost at 24 starts on the 8yr are running for the record; the adverse-only 26yr is
queued because its 8yr worst year was only -0.3pp.

24-start 8yr follow-ups: at 2x cost the screen keeps its full edge (+1.46pp / +0.034, 20/24, vs the package at 2x —
as expected, a screen adds no turnover); adverse-restatements-only +1.26pp / +0.029 on 24/24 starts, 6/8 years (fails
only the 47% concentration test on a small mean); ICFR-only +1.04pp / +0.024 (18/24), 6/8 years, PASS. Both halves
carry the effect on the 8yr; the adverse-only 26yr (running) decides whether the narrower screen escapes the 2013-17 run.


### EXP-060 batch 13 — more event screens (officer changes, dividend cuts), 8yr stage 1 (8 starts). DEAD.

| screen (365d) | flags (S&P 500) | dCAGR | dSharpe (+) | dMaxDD | years better | note |
|---|---|---|---|---|---|---|
| CFO change | 10-18% | +0.88pp | +0.020 (6/8) | -3.2pp | 4/8 | 2019 -17.2, 2021 +28.6, 2024 -11.1 — single-name lottery |
| CEO change | ~10% | -3.24pp | -0.072 (0/8) | -1.6pp | 4/8 | |
| dividend cut | 4-7% | -1.70pp | -0.038 (0/8) | 0.0 | 3/8 | 2026 -14.9 |
| accounting screen + CFO change (vs accounting screen) | | -0.47pp | -0.010 (1/8) | -2.3pp | 3/8 | worsens it |

Read: turnover of officers and dividend cuts are too common (10-18% of names) and too weakly tied to future returns to
work as screens; at that flag rate the screen's effect is which momentum winners it happens to remove. The event-screen
class is exhausted: only the narrow accounting-risk flags (3-8%) carry information, and even they are small.


### EXP-060 — adverse-restatements-only screen on the 26yr, 24 starts: FAIL; accounting-screen family CLOSED. (2026-09-20 23:38)

-0.35pp / -0.009 Sharpe (1/24) / +0.5pp MaxDD; 13/25 years; 2009-16 sub-period -0.050; ex-crisis -0.029 (0/24). The
2013-17 losses (-7.3, -7.5, -7.4, -8.5) are identical to the combined screen's — narrowing the flag set does not
escape them. The 8yr's 7/8 years was the 2018-26 window; on the long horizon accounting-risk exclusion is a coin flip
with a crisis-year skew. Closed.

### STATE OF THE SECOND-ENGINE SEARCH (2026-09-20, end of EXP-060 batches 1-13)

Owner's criterion: better than the EXP-059 package in MOST calendar years on both horizons, from a different return
source. Tested at 8-24 starts on the 8yr and 24 starts on the 26yr where anything survived:

| family | best result vs package | verdict |
|---|---|---|
| analyst data (IBES surprise / revisions / up-down / recommendations / combo) | alone Sharpe 0.4-0.6, corr 0.7+, negative inside | dead |
| fast mean reversion (short-term reversal, calendar seasonality) | -0.04 Sharpe, 3/8 yrs | dead |
| return decomposition (overnight momentum), 13F breadth | -0.11 / 0.00 | dead |
| long-term reversal, plain and uptrend-filtered | 8yr +0.019..+0.056 (6/8 yrs) but 26yr -0.041 / -0.024 on 0/24 starts | dead (value-cycle regime bet) |
| ETF assets: sector trend, VIX futures, 10y Treasury | ~0 or negative | dead |
| **gold, 10-15% permanent line** | **26yr +0.020..+0.023 (24/24), 14-15/25 yrs, all sub-periods > 0, 2x cost intact; 8yr +0.030 (24/24), 5/8 yrs** | **PASSES — small diversification effect, owner's call** |
| net share issuance, investment/profitability sleeves, in-universe quality screens | dead; the quality screens remove the momentum leaders (-10pp/yr) | dead |
| accounting-risk exclusion (restatements / ICFR failures) | 8yr +0.034 (19/24), 7/8 yrs, cost-free; 26yr +0.013 (24/24) but 14/25 yrs, 2013-17 negative | sign-consistent, small, not an engine |
| officer-change and dividend-cut screens | lottery on single names | dead |

Conclusion: within the data on disk there is no second engine that is better in most years on both horizons. The one
robust addition is the gold line (+0.3pp CAGR / +0.02 Sharpe / +0.8pp MaxDD on the 26yr, better in 14-15 of 25 years),
which is diversification, not alpha. The accounting screen is real but too small and too crisis-skewed to promote.
Remaining unexplored data: `wrds_financial_ratios` (PERMNO-keyed, public_date = PIT) as a replacement VALUE engine
(the deployed value sleeve is a different return source already in the book; a better one would count). Running next.


### EXP-060 batch 14 — composite VALUE engine from wrds_financial_ratios (PIT), 8yr stage 1 (8 starts). DEAD.

| arm | ALONE | corr | yrs > base | in package: dCAGR / dSharpe (+) / dMaxDD · years better · 2019, 2020 |
|---|---|---|---|---|
| composite value (bm, EV/EBITDA, P/CF, P/S, div yield; roa > 0) | +21.5% / 0.742 / -61% | **0.56** (lowest found) | 3/8 | — |
| same, above SMA200 only | +15.0% / 0.601 / -63% | 0.63 | 3/8 | — |
| replacing the deployed value sleeve (15%) | | | | +0.77pp / -0.003 (4/8) / **-4.8pp** · 5/8 · -12.3, -8.5 |
| added as a 15% sleeve | | | | -0.28pp / -0.006 (3/8) / -3.7pp · 5/8 · -10.6, -7.1 |
| replacing, top-20 | | | | -0.05pp / -0.020 (1/8) / -3.9pp · 5/8 |

Read: a textbook value composite is the most diversifying sleeve in the search (corr 0.56) and still does not help — it
carries value's 2019-20 collapse and a 60% standalone drawdown; the deployed value sleeve's quality/trend guards are why
it survives at 15%. The ratio table is now used; no unused data source remains on disk.

### SECOND-ENGINE SEARCH — COMPLETE for the data on disk (2026-09-20 23:55)
Fourteen batches, every WRDS table and every ETF in the panel tested. Result: no second engine that is better than the
package in most years on both horizons. Robust addition: a 10-15% gold line (diversification, small). Real but not
promotable: the accounting-risk screen. Everything else dead. Further gains require new data (non-public) or a
different execution horizon, as the memory from earlier programs already said. Owner decisions outstanding: the EXP-059
package (built behind flags) and the gold line.


### EXP-060 batch 15 — gold line on the DEPLOYED book (no package), 8yr 24 starts. Vs the deployed base.

| arm | dCAGR | dSharpe (+) | dMaxDD | years better · top-yr | even-month | ex-crisis | gate |
|---|---|---|---|---|---|---|---|
| gold 10% | +0.39pp | +0.028 (24/24) | +1.1pp | 4/8 · 48% (2025) | +0.027 | +0.037 (24/24) | FAIL (years) |
| gold 15% | +0.50pp | +0.032 (24/24) | +1.3pp | 4/8 · 55% | +0.031 | +0.042 (24/24) | FAIL (years) |

Read: same character as on the package — every start better, slightly lower drawdown, but on the 8yr the gain sits in
gold's 2025 and it is better in only half the years. The 26yr (running, AC power) is the test that matters for the
owner's decision; on the package it passed (14-15/25 years, all sub-periods > 0).

26yr, 24 starts, gold 10% on the DEPLOYED book vs the deployed base: **PASS** — +0.43pp CAGR / +0.023 Sharpe (24/24) /
+1.4pp MaxDD; 14/25 years, top-year share 21%; sub-periods +0.040 / +0.023 / -0.006 / +0.055; even-month +0.024;
ex-crisis +0.026 (24/24); bootstrap P 95%, CI [+0.000, +0.050]. Read: the gold line's value does not depend on the
package; it is the same small, spread-out diversification benefit on either book. Owner's decision is independent of
the package decision. Gold 15% on the deployed book, 26yr: PASS — +0.52pp / +0.027 (24/24) / +1.6pp MaxDD; 15/25
years; CI [+0.002, +0.056]; even-month +0.027; ex-crisis +0.030 (24/24). Batch 15 complete.

Gold decision table (24 starts; 26yr is the expectation, 8yr shown for completeness):

| book | gold | 26yr dCAGR / dSharpe (+) / dMaxDD · years | 8yr dCAGR / dSharpe (+) / dMaxDD · years |
|---|---|---|---|
| deployed | 10% | +0.43pp / +0.023 (24/24) / +1.4pp · 14/25 | +0.39pp / +0.028 (24/24) / +1.1pp · 4/8 |
| deployed | 15% | +0.52pp / +0.027 (24/24) / +1.6pp · 15/25 | +0.50pp / +0.032 (24/24) / +1.3pp · 4/8 |
| package | 10% | +0.28pp / +0.020 (24/24) / +0.8pp · 14/25 | +0.40pp / +0.030 (24/24) / +0.7pp · 5/8 |
| package | 15% | +0.32pp / +0.023 (24/24) / +0.9pp · 15/25 | +0.51pp / +0.035 (24/24) / +0.8pp · 5/8 |


### EXP-060 — audit of the GOLD finding before the owner decides (2026-09-21). Verdict: PASS.

1. Ticker reuse hazard: in raw CRSP the ticker GLD maps to four PERMNOs back to 1962. The v2 builder keys each ETF to the
   single PERMNO whose era ends latest (the SPDR Gold Trust); the panel's GLD column starts 2004-11-18 with ZERO values
   before it and matches that PERMNO's return-chained series to 0.004%. No phantom pre-2004 holdings; the small 2002-04
   deltas in the gold arms come from the 10% cap binding differently when sleeve weights are scaled by 0.9, not from GLD.
2. External cross-check (Polygon daily adjusted closes, 2018-2025, fetched via curl after a local-cert failure): n=2011 median rel diff +0.0004%, p99 abs 0.001%, max abs 0.001%, daily-return corr 1.00000 -> PASS
3. Not just lower exposure: on the 26yr the deployed base scaled to the gold arm's realized vol gives +17.68% CAGR vs the
   gold arm's +18.68% (8yr: +31.53% vs +32.91%) — the gain survives vol-matching, i.e. it is diversification, not de-risking.
4. Mechanics: GLD is held as an ordinary position (same 40% trailing stop, same de-risk overlay, whole shares — ~5 shares
   per $15k book at today's price); GLD pays no distributions so price return = total return; the 2x-cost arm holds.


### Deployment note (2026-09-22): the EXP-059 package is LIVE; the bull split is dormant while breadth < 35%
Shipped 21:50-21:57 ET (docs/LIVE_SYSTEM.md has the full record, the two start-up traps, the book-2 dry run and the
breadth verification). Live breadth 25.9% -> blend 0 -> bear weights 11/33/56 for every rebuild until breadth recovers;
verified against a fresh full-universe recomputation (25.8%), the same formula in the backtest, a negligible pool
difference (mean 0.009) and the backtest's own history (blend 0 on 18% of days since 2018). Series saved:
`research/_breadth_series_8yr.csv`. First live rebuild under the package: 2026-09-23 (book 2), compared to the dry run.


## Cycle 59 — EXP-061 · the breadth-blend rule itself (opened 2026-09-22 late, owner's request after seeing blend = 0 live)

Question: v12 blends the bull sleeve mix toward the bear mix (11/33/56 mom/val/lowvol) linearly as the share of names
above their 50-day SMA falls from 60% to 35%. It was part of the validated v12 and never a target of EXP-059/060. Is the
rule earning its keep, and are its two knobs (the ramp and the bear mix) at the frontier? Base = the LIVE package at the
live 15% cap (`cap0.15_package`, 24 starts: 8yr +35.8/1.13; 26yr +19.7/0.78). Arms: no blend (always bull), ramp 25-50%,
ramp 45-70%, ramp 35-85% (wider), bear mix 40/35/25, bear mix 50/30/20, bear mix value-heavy (11/56/33). Stage 1 (8
starts) both horizons, then 24 starts and the pairwise gate for survivors, years-better first. Hypothesis to beat:
"no blend" — if always-bull is not worse in most years on the 26yr, the rule is drag; if it is, the rule is doing
its job and only the knobs are in question. Nothing changes live until this is answered.

## Next

Running: EXP-001 26yr · EXP-001b (capital + live-sizing control) · EXP-003 (I-21 filter vs
ranking) · EXP-005 (I-25 risk-matched leverage sweep).

Queued: EXP-006 (I-23 partial-adjustment rebalancing — the deployable form of the one
preliminary win), then the risk/construction tier I-03 / I-04 / I-05, which cycle 3 promoted
above the signal tier.
