# IDEAS — ranked backlog of untested hypotheses

Never let this fall below 15 entries. Each entry must state **why the edge would exist**
(who is on the other side, or what structural inefficiency is being harvested). An idea with
no mechanism gets downranked — it is almost certainly a fitting artefact.

Status: `OPEN` | `RUNNING` | `KILLED` | `WON` (see `LOG.md` for the verdict and evidence).

---

## Tier 1 — structural / variance-reduction. No alpha claim required, so no alpha risk.

These are the highest-EV entries precisely because they do not need a new edge to exist.
They remove uncompensated risk that the current implementation takes for free.

### I-01 · Rebalance-phase tranching · **PRELIMINARY WIN, audit incomplete (EXP-001)** · EV: HIGH
Split capital into K sub-books, each running the identical strategy but rebalancing on a
different phase of the 20-session cycle (phase 0, 4, 8, 12, 16).

**Mechanism:** the rebalance date carries **zero information**. It is a pure nuisance
parameter. Yet we measured 8yr CAGR σ = **7.07pp across entry month alone**
(+15.03%…+34.55%) — that entire spread is phase risk, taken uncompensated. Averaging K
weakly-correlated phases should shrink it by roughly √K and raise Sharpe with **no assumption
that any signal works better**. This is the same logic as diversifying across names, applied
to the time axis.

**Predicted:** ΔCAGR ≈ 0 (maybe slightly +, from avoiding phase-specific bad luck), Δσ_start
large and negative, ΔSharpe small and positive, ΔMaxDD positive.
**Live constraint:** at $33k, 5 tranches × ~23 names = ~$290/position and integer shares bite
hard. Must be tested WITH integer-share sizing at real capital. K=2 or K=4 may be the
implementable answer.
**Falsification:** if per-start σ does *not* fall by ≈√K, the phases are far more correlated
than assumed and the idea is void.

### I-02 · Fix cap-then-renormalize · **STAGE 2 (EXP-059: +0.015 Sharpe 8/8 on 8yr stage 1, tiny)** · EV: LOW
The 10%-of-book position cap is applied, then gross is renormalised to 1.0 — which pushes
capped names straight back above the cap. Documented in `multi_strategy_engine`, never
measured. Replace with iterative water-filling (cap, redistribute the excess to uncapped
names, repeat to convergence).

**Mechanism:** not alpha — the risk control we believe is in place is not in place. Expect
lower single-name concentration, therefore lower idiosyncratic variance, therefore Sharpe up
slightly and tail-DD down. If the measured effect is zero, the cap never binds and we can stop
worrying about it.

### I-03 · Ex-ante (holdings-based) portfolio-vol estimate instead of 40d realised · **KILLED on 8yr stage 1 (EXP-059: dSharpe -0.013 / -0.010, 1/8 starts)** · EV: was HIGH
Current `vol_scale` divides the vol target by the **trailing 40-day realised vol of the book**.
After a rebalance rotates into five different names, that estimate still describes the *old*
book for up to 40 sessions.

Replace with an ex-ante estimate from **current** holdings: individual name vols (already in
the panel as `vol_20d`/`vol_60d`) plus a shrunk correlation estimate → predicted portfolio vol
today.

**Mechanism:** this is **not** a timing rule — every timing rule in `DYNAMIC_LEVERAGE_FINDINGS`
is dead and must stay dead. This is an *estimator lag* fix: measuring the risk of the book you
actually hold rather than the one you used to hold. The DYN program only ever varied the
*window* on portfolio returns; it never changed *what is being estimated*.
**Falsification:** must clear the matched-exposure control (score against a constant-leverage
curve at its own realised avg_gross) or it is just running less exposure.

### I-04 · Sleeve-level risk parity · **FAILS two-horizon bar, but NOT for the reason first logged** · EV: LOW
26yr: Sharpe-neutral (+0.001/+0.002, 9/12) with a monotone **drawdown** gain (+1.19 / +2.53 /
+4.16pp at parity power 0.5 / 1.0 / 1.5) for −0.12 to −0.49pp of CAGR.
8yr: clearly harmful (−0.015 / −0.032 / −0.053 Sharpe, 0-2/12).
Horizons disagree on Sharpe ⇒ not promoted. But it is a **CAGR-for-drawdown trade**, not a
mistake, and it belongs in the same family as lower constant leverage. Re-open only if a
drawdown-first mandate is ever chosen; then test it exposure-matched (the arms above run 4-6%
more gross).
Sleeve weights are fixed at 50/35/15 in *capital*. Momentum is structurally the highest-vol
sleeve, so the book's *risk* is far more than 50% momentum, and that share swings with regime.
Target constant *risk* contribution per sleeve instead.

**Mechanism:** a fixed capital split delivers a time-varying risk split — an unintended,
uncompensated regime bet. Note `use_rp` is inverse-vol *within* mom/val (a different thing) and
tested dead; sleeve-level parity is untested.
**Caution:** this WILL change avg_gross → matched-exposure control mandatory.

### I-05 · Vol-normalised trailing stop · **KILLED (EXP-059 stage 1: dSharpe -0.070, 0/8 starts, both k)** · EV: was MED
The stop is a flat 40% for every name. On a 20%-vol name that is a ~2σ annual event; on a
60%-vol name it is ordinary noise. Replace with `k × annualised vol`, floored and capped
(e.g. 25%…55%).

**Mechanism:** a fixed stop applies a *different confidence level* to every name, so it fires
on high-vol names for no informational reason and never protects low-vol ones. Normalising
equalises the false-positive rate. Expect fewer whipsaw exits and lower turnover.

---

## Tier 2 — structural flows. There is an identifiable forced participant on the other side.

### I-06 · Exclude recent index ADDITIONS from the momentum sleeve · **KILLED (EXP-059 stage 1: N=60 +0.023 Sharpe but -1.4pp MaxDD; N=120 -0.026 / -3.1pp)** · EV: was MED-HIGH
Do not buy a name that entered SP500/400/600 within the last N sessions (test N = 20/60/120).

**Mechanism:** index funds are **forced buyers** into the effective date. The pop is mechanical
and reverses; the classic addition effect has decayed but the *reversal* is the durable half.
Our momentum sleeve is structurally attracted to exactly these names — a name that just popped
on index demand scores well on 12-1 momentum for a reason that carries no future return.
We hold PIT membership dicts, so additions are observable **at the time**, no new data needed.
**Who loses:** index funds, by construction — they must buy at the deadline regardless of price.
**Falsification:** if excluded names' forward 20d returns match the rest of the pool, void.

### I-07 · Earnings-date variance avoidance · **KILLED (EXP-002)** · see LOG cycle 2
Down-weight (or skip) a name whose `rdq` announcement falls inside the coming holding window.
`compustat_quarterly` has `rdq` + `LPERMNO`, so this is PIT-honest and needs no new feed.

**Mechanism:** we deliberately zeroed every earnings-surprise boost (B4) — meaning we assert
**no edge in predicting the print**. A holding therefore takes a ±8% idiosyncratic gamble at
zero expected return: pure uncompensated variance. Removing it should raise Sharpe with a small
CAGR cost.
**Falsification:** if names reporting inside the window have the *same* realised variance as
those that do not, the premise is wrong. Check that first — it is one cheap cross-section.
**Leak risk:** `rdq` is a *reported* field and can be revised. Must confirm it is never in the
future relative to the decision date, and lag it by 1 day defensively.

### I-08 · Turn-of-month rebalance anchoring · **OPEN** · EV: LOW-MED
Anchor the rebalance to a calendar day-of-month rather than a rolling 20-session counter.

**Mechanism:** 401(k)/pension inflows cluster at month start; index rebalances at quarter end.
Trading *into* known flow is adverse selection; trading with it is not.
**Interacts strongly with I-01** — if phase tranching wins, this is subsumed. Test after I-01.

---

## Tier 3 — signal construction. Same anomaly, less estimation noise.

### I-09 · Multi-horizon momentum rank ensemble · **KILLED (EXP-059: 26yr dSharpe -0.125, 0/8; 8yr pending but irrelevant)** · EV: was MED
Score = average of *ranks* on 12-1, 6-1 and 9-1 momentum instead of 12-1 alone.

**Mechanism:** the same argument as I-01, on the signal axis. Any single lookback is an
arbitrary choice, and the sampling error in a single-horizon ranking is large. Averaging ranks
across horizons is variance reduction on the *estimator*, not a new bet.
**Not** a parameter sweep — the claim is that the ensemble beats *every* member, which is a
falsifiable structural claim rather than a best-of-N pick.

### I-10 · 52-week-high proximity as the momentum functional form · **KILLED (EXP-059: 26yr dSharpe -0.113, 0/8)** · EV: was MED
Replace / blend `ret_252d − ret_20d` with `price / 52w-high`.

**Mechanism:** George–Hwang. Anchoring on the 52-week high is a documented behavioural
mechanism, and the nearness measure is far more robust to a single outlier month than a raw
12-month return. Different functional form, same underlying anomaly — so it is not a new edge
claim, it is a less noisy estimator of the one we already trade.

### I-11 · Harder idiosyncratic-vol screen · **KILLED (EXP-048)** · see LOG cycle 48
Book-wide screen is harmful at every dose on both horizons: dSharpe −0.279/−0.441/−0.395 (8yr) and
−0.061/−0.155/−0.109 (26yr) at p80/p67/p50, costing 4.5-22pp of CAGR. Momentum winners ARE the
high-vol names, so screening them removes the sleeve's hunting ground. A standalone-predictive
feature is not the same claim as an additive one. Pool-only variant formally untested but not worth
reopening at this magnitude.
The AUDIT01 forward-IC sweep found `vol_60d` the **single strongest** feature in the panel
(IC −0.050, i.e. low vol → high forward return) — stronger than any return feature. The
momentum sleeve currently gives only a soft ×1.15 nudge for `vol_20d < 0.25`.
Test a hard screen: rank the momentum pool, drop the top vol tercile, take top-5 of the rest.

**Mechanism:** the low-vol anomaly (leverage-constrained investors bid up high-beta names) is
one of the most replicated effects in the literature and it is *already visible in our own data*
without being properly harvested. Momentum and low-vol are known to combine well.

### I-12 · Residual (beta/sector-adjusted) momentum · **OPEN** · EV: MED
Rank on momentum of the residual after removing market and sector returns.

**Mechanism:** raw 12-1 momentum is partly a bet on whichever sector ran; residual momentum
isolates the firm-specific continuation and is documented to have a higher information ratio.
Previously logged "GFC hedge not worth cost" — but on **3–4 starts and the pre-audit poisoned
universe** (see BUGS F6). Not safely dead.

### I-13 · Cross-sleeve overlap · **KILLED (EXP-020)** — neither boosting nor flattening moves Sharpe by more than 0.008; agreement carries no information and the current doubling is harmless. Question closed.
A name selected by two sleeves currently receives the sum of both weights, then gets capped.
Test three arms: as-is / boost overlap / neutralise overlap to a single sleeve's weight.

**Mechanism:** if the sleeves are genuinely different views, agreement is information and should
be overweighted. If they share inputs (both read `roe`, `gross_margin`), agreement is just
correlated error and the current behaviour silently concentrates the book. We do not know which,
and the answer determines whether a risk control is needed.

---

## Tier 4 — exposure / external data. Highest prior of failure; every neighbour is dead.

### I-14 · VIX term structure (VIX3M/VIX) as a gross-exposure gate · **KILLED (EXP-059: 26yr -0.007, 0/8; 8yr pending)** · EV: was LOW-MED
Backwardation (VIX > VIX3M) → de-gross.

**Mechanism:** the only genuinely **forward-looking** risk measure available free. Everything
killed in the DYN program was *backward*-looking (realised vol, equity curve, distance from
peak, vol-of-vol) or slow (credit spreads). Options-implied term structure prices *expected*
near-term stress and inverts days before realised vol moves.
**Prior of failure: high** — the whole family is dead. Must clear matched-exposure AND
event-concentration. VIX3M starts ~2004, so the 26yr test is truncated, which weakens the
strongest horizon. Downranked for that reason.

### I-15 · Financing-spread-aware leverage · **OPEN** · EV: LOW-MED
Scale 1.49× by where `(SP500 earnings yield − 3M T-bill)` sits in its own expanding percentile.

**Mechanism:** we borrow at 6.3% to hold equities. Leverage is worth taking only when the
expected equity risk premium exceeds the borrow. Holding leverage *constant* while the spread
you earn on it varies 400bp is an unintended bet. This is a slow-moving valuation quantity, not
a vol-timing rule, so it is not obviously in the dead family.
**Caution:** valuation scoring already failed the both-period bar (`threadV`) — it was a regime
bet. High risk of the same outcome. Data: `fred_interest_rates_spreads_daily`, Compustat.

### I-16 · Convex tail overlay funded from the equity book · **OPEN** · EV: LOW
A small permanent allocation (1-3%) to a convex payoff (VIXM or a put spread), continuously
funded from the equity sleeve.

**Mechanism:** the real problem with this strategy is **MaxDD −64.8% on the 26yr at 1.49×** —
near-ruinous, and the reason leverage cannot be raised. A convex overlay reshapes drawdown
*geometry* rather than trying to time it, so it does not depend on any prediction. Cost is the
whole question: variance risk premium means you pay ~1-3%/yr for it.
`gld_pct`/`vixm_pct` hooks already exist in the harness — check whether they were ever tested
at the 12-start standard before spending on this.

---

## Tier 5 — parameters. Lowest value. Do not spend time here unless something above points at it.

### I-17 · `top_n` concentration sweep (3/5/8/10) · **OPEN** · EV: LOW
Only interesting for its *interaction* with the 10% cap and integer-share granularity at $33k,
not as a standalone tweak.

### I-18 · `rebal_days` re-confirmation at the 12-start standard · **OPEN** · EV: LOW
"20d optimal" was established pre-audit on few starts (BUGS F6). Cheap to re-run; mostly a
sanity check that the deployed cadence is not a fitted artefact.

---

## Tier 6 — re-tests forced by BUGS F6 (verdicts reached on ≤4 starts and/or poisoned universes)

### I-19 · Re-run `threadREVIVE_dead_ideas.py` batch at the 12-start, two-horizon standard · **OPEN** · EV: MED
Vol-managed momentum, risk parity, sector cap, momentum quality filters, signal-exit cadence,
cash parking, bull-lever. All have config hooks, so **no new modelling risk is introduced —
only the measurement standard changes.** Cheapest possible way to convert "probably dead" into
"known dead", and it has already flipped one sign in this repo.

### I-20 · Re-test the deployed inverse-vol overlay itself · **OPEN** · EV: MED
`DYNAMIC_LEVERAGE_FINDINGS` says the deployed overlay has **~no timing skill** (+0.018 8yr /
+0.001 26yr, negative with more starts) and that its value is a pure *level* effect.
If true, the overlay could be replaced by simply running lower constant leverage — identical
risk, less turnover, fewer moving parts. **Simplification is a legitimate win.** Test:
constant-leverage curve at the overlay's realised avg_gross vs the overlay, 12 starts, both
horizons.

---

---

## Tier 0 — promoted above everything else by what cycles 0-2 measured

### I-21 · Decompose the sleeve edge into FILTER vs RANKING · **RUNNING (EXP-003)** · EV: VERY HIGH
The 26yr shuffle says all of "stock selection" is worth **+4.5pp CAGR / +0.13 Sharpe** through a
full cycle (it is worth +20pp / +0.58 on the 8yr — the recent window flatters it ~4×). But
"selection" is two different machines bolted together: cheap mechanical **filters**
(`dist_sma200 > 0`, valid 12-1 momentum, bear-regime sector exclusion, the value sleeve's quality
screens) and the **ranking** that orders the survivors.

**Mechanism:** none required — this is a measurement, not a bet. It decides where every remaining
hour of this program goes. If the ranking is worth ~nothing over a *sticky random draw from the
filtered pool*, then years of cross-sectional scoring work has been rediscovering a trend filter,
and the productive direction is filtering / risk / construction. If the ranking carries it, the
opposite.

**Design note that makes it fair:** the random arm must be **sticky** (keep names still eligible,
replace only drop-outs). A naive random re-pick turns over ~100% every rebalance, so a plain
A-vs-random gap silently prices *turnover*, not ranking.

### I-23 · Partial-adjustment rebalancing as the deployable form of I-01 · **KILLED (EXP-007)** · see LOG cycle 5
EXP-001 shows phase risk is **7.92pp of 8yr CAGR** and that averaging 4 phases removes ~half of
it. But 4 phases means 4 target books netted inside one IBKR account — substantial new live code.

Cheaper approximation with the same intent: keep **one** book, rebalance every 5 sessions, and
move only ~25% of the way toward the new target each time.

**Mechanism:** identical — it spreads execution across the cycle so no single arbitrary day
determines the book. It is *not* mathematically the same as tranching (it smooths toward a moving
target rather than running independent sub-books), so it must be measured, not assumed. If it
captures most of the variance reduction it is far and away the better thing to deploy.

### I-24 · Is the strategy actually beating a passive alternative over 26yr? · **RUNNING (EXP-004)** · EV: VERY HIGH
Nowhere in this repo is the deployed strategy compared to levered SPY or equal-weight SP1500 on
matched terms — same leverage, same 6.3% financing, same period, same price matrix.

**Mechanism:** none — again a measurement, and the one that determines whether this program should
be improving the strategy or replacing its core thesis (escalation ladder level 10). A first pass
put 26yr strategy (+11.26% / 0.51 / −64.8%) *behind* buy-and-hold equal-weight SP1500. That first
pass used a **daily-rebalanced** equal-weight index, which harvests an unimplementable rebalancing
premium; it is being recomputed as monthly/quarterly reconstitution with buy-and-hold in between
before any conclusion is drawn.

---

## Tier 3b — generated by the cycles above

### I-26 · Concentration gradient: does top-3 beat top-5 on the 26yr? · **OPEN** · EV: MED
EXP-007 showed that *diluting* the book (partial adjustment → a long tail of stale half-positions)
drags returns toward the equal-weight index. The gradient therefore runs the other way too:
more concentration → more of whatever edge the selection has.

**Mechanism:** if the momentum ranking carries real information, its top name is better than its
fifth, and holding 5 dilutes it. If EXP-003 finds the ranking is worth little, this should do
nothing — so the two experiments cross-validate each other, which is why this is worth running
regardless of outcome.
**The catch, and why it is only MED:** EXP-004 says the axis we actually need is **drawdown**,
and concentration almost certainly makes drawdown worse. Judge this on Sharpe and MaxDD, not
CAGR. A CAGR-only win here is not a win.

### I-22 · Earnings gaps as a DRAWDOWN channel, not a variance channel · **OPEN** · EV: LOW-MED
EXP-002 killed the mean-variance version of earnings avoidance (real effect ≈ +0.016 Sharpe,
below our detection floor). But it only tested the *average*. Untested: an earnings gap can trip
the **40% trailing stop** in a levered book, forcing a realised loss and a re-entry.

**Mechanism:** stops convert a temporary idiosyncratic gap into a permanent loss. The question is
not "does earnings add variance" (yes, 1.39× — too small to matter in a 23-name book) but "what
share of trailing-stop exits are earnings gaps, and do those exits lose money relative to holding
through". That is a tail question and needs `log_stops` + `rdq`, not a Sharpe A/B.

---

### I-27 · Tilt HARDER into momentum (the inverse of sleeve risk parity) · **OPEN** · EV: MED
EXP-020 found a clean dose-response: the more the book is tilted toward equal RISK across
sleeves, the worse it gets (−0.015 / −0.032 / −0.053 Sharpe at parity power 0.5 / 1.0 / 1.5,
0-2/12 sign). The gradient therefore points the other way.

**Mechanism:** EXP-003 established the ranking carries 100% of the selection edge, and momentum
is the ranked sleeve — value and lowvol behave close to random draws from a filtered pool. If
that is right, capital should be tilted *toward* momentum, not away.
**Why only MED, and the specific danger:** sleeve-weight changes have been RETRACTED as noise in
this repo before (CORE2/3/4 at 6/12), and momentum is the highest-vol sleeve so any tilt raises
drawdown — the axis EXP-004/022 says matters. Judge on Sharpe AND MaxDD, ≥9/12 on both horizons,
and expect this to fail the drawdown test even if Sharpe improves.

---

### I-29 · Mid-cycle signal exit as a DRAWDOWN instrument · **OPEN** · EV: MED-HIGH
`signal_exit_every=5`: sell a holding as soon as no sleeve still wants it, instead of waiting out
the 20-session cadence. EXP-021 (26yr, 12 starts): **MaxDD −64.1% → −52.7% (+11.39pp)**,
Sharpe +0.020 (8/12), CAGR −1.14pp.

**Mechanism:** the deployed book holds a name for up to 20 sessions after the signal that bought
it has gone. In a fast decline that is 20 sessions of holding something the model no longer
wants. Exiting promptly is cheap (a sale, not a re-pick) — and EXP-007 established exactly this
distinction: *cheap adjustments can be prompt, expensive ones cannot*. Re-picking the book
mid-cycle was catastrophic (−7 to −10pp); merely *exiting* is a different operation.
**Why not higher EV:** −1.14pp of CAGR is a real cost, sign consistency is only 8/12, and it
needs new live code (a daily exit check between rebalances). Test the exit cadence (3/5/10) and
the grace parameter, on both horizons, before believing it.

### I-30 · Why does `roe` work when `gp_assets` is catastrophic? · **ANSWERED — (a) LUCKY DRAW (EXP-025)**
Family test: 1/5 positive on 26yr, **0/5 on 8yr**; family mean −0.012 / −0.210; no income-vs-gross
split. `roe` is one lucky draw out of nine EXP-021 arms. See LOG cycle 27. Closed.
EXP-021: same hook, same pool, same mechanism — `roe` +0.021 Sharpe (10/12), `gp_assets` −0.072
(0/12). Both are profitability measures. That asymmetry is either (a) the tell that `roe` is a
lucky draw, or (b) informative about *which* profitability signal matters.

**Why it is worth its own experiment:** it is a cheap, decisive test of the ROE result. If a
third and fourth profitability metric (`net_margin`, `operating_margin` — both already in the
panel) side with `roe`, the mechanism is real and `gp_assets` is the outlier. If they side with
`gp_assets`, `roe` is noise and EXP-024 should not be believed no matter what its own audit says.
This is the multiple-testing question asked constructively rather than just deflating a p-value.

---

### I-33 · Options 1 and 2 at leverage BELOW 1.00× · **OPEN** · EV: MED-HIGH
EXP-031's 26yr iso-drawdown table has a hole: at a −45% target **only the baseline is on-curve**.
Options 1/2 cannot reach −45% at any leverage tested because their minimum realised gross is
~0.99× (leverage 1.00, no overlay), flooring drawdown at −48%.

**Mechanism:** the deployed vol overlay scales gross *below* 1.0× in high-vol regimes (clamp
floor 0.30), so it reaches exposure territory constant leverage cannot. Every "the overlay is
harmful" claim in LOG.md is true *at matched exposure* and understates this.
**The test:** run Options 1/2 at leverage 0.70 / 0.85. If constant 0.85× reaches −45% with more
CAGR than the baseline's +10.80%, the caveat dissolves and the overlay has no remaining role.
If it cannot, the overlay is genuinely useful at low target risk and should be kept for anyone
wanting drawdown below ~−48%.
**Why it matters:** it is the last open question separating "delete the overlay" from "delete the
overlay unless you want low-risk operation", and it is one cheap sweep.

---

## Parked — mechanism understood, blocked on something external

- **Off-cadence credit-gate application** — real defect (BUGS B6), but KILLED by
  event-concentration (99.6% of 26yr excess from 5 days). Would need a mechanism argument that
  survives the 2001/2022 losses. Also needs new `ibkr_engine` code.
- **WRDS membership re-download with PERMNO/GVKEY** (~Sept 2026) — closes the 9.3% join gap and
  un-blinds EDGAR go-forward reconciliation. Blocks nothing here but improves every 26yr number.

## Tier 7 — post-deployment frontier (opened 2026-09-13, EXP-059). Base = deployed FINAL @1.49x on the v2 universes.

### I-34 · Mid-cycle signal exit for ALL books on every tranche day · **DEAD at real costs (EXP-059 stage 2: +0.035/+0.030 Sharpe at 1x but +0.020 (14/24) / +0.007 (13/24) at 2x cost; -1.8/-1.1pp CAGR; better in 3/8 and 10/25 years; gate FAIL both horizons). Pure-turnover drawdown instrument.** · EV: DEAD
I-29 re-cast for the tranche structure: the stride is 5, so "exit a name no sleeve wants" is now a cheap weekly
adjustment for every book, not a re-pick. EXP-026 (old universe, single book): +5.6-7.8pp matched MaxDD for -0.02 Sharpe.
Question: does the tranche structure change that trade-off?

### I-35 · Prompt overlay: rescale ALL books to today's vol_scale x gate on every tranche day · **STAGE 2 DONE — de-risk-only form is the sole survivor but NOT promoted on return: 26yr +0.77pp/+0.036 (24/24)/+5.5pp MaxDD, gate PASS, intact at 2x cost; 8yr +1.6pp/+0.070/+4.9pp but better in only 3/8 years with 2020 = 76% of the gain (gate FAIL = crisis insurance). Batch 4 refining (lb20, threshold 0.90/0.85, +mom_equal)** · EV: MED-HIGH (drawdown)
Cycle 13 found the overlay/gate were applied up to 20 sessions late; the deployed tranches cut that to <=5 sessions for
the rebuilding book only. This applies it to every book weekly (small proportional trades). Costs turnover.

### I-36 · No-trade band for resizes (1-2% of book instead of 0.3%) · **DEAD (EXP-059: ~0 at 1x cost)** · EV: LOW-MED (cost)
Pure cost lever: fewer one-share trims. Must be evaluated at cost 2x, where it matters.

### I-44 · Sleeve split 80/15/5 under the overlay + equal-weight pair · **PASSES pairwise gate both horizons (+0.014, 17/25 yrs, cost-2x intact) but fading in 2023-26 (-0.01); 90/7/3 and pure momentum FAIL on the recent sub-period (-0.04 / -0.10). Needs the overlay to work (alone +0.008-0.011, fails). 80 is the limit.** · EV: LOW-MED
### I-45 · Momentum inverse-vol weights · **DEAD (batch 9: -0.008/-0.004, under-weights the high-vol winners)**
### I-46 · THE PACKAGE = I-35 overlay_down + I-38 mom_equal + I-44 split · **26yr +1.55pp / +0.060 (24/24, CI > 0) / +5.4pp MaxDD, 17/25 yrs, ex-crisis +0.035, cost-2x intact — GATE PASS; 8yr fails only on 2020 concentration. Capital-scale (whole-share) check running (batch 11). NOT deployed.**

### I-47 · 10% permanent gold (GLD) sleeve inside the package · **PASSES pairwise gate both-horizon logic (EXP-060: 26yr +0.020 Sharpe 24/24, 14/25 yrs, all sub-periods > 0; 8yr +0.030 24/24, 5/8 yrs). Small: +0.3pp CAGR / +0.8pp MaxDD. Diversification, not alpha. Timing it with SMA200 hurts.** · EV: LOW-MED (consistency)
### I-48 · Second-engine sleeves tested and DEAD in EXP-060: IBES surprise / revisions / up-down / recommendations; short-term reversal; calendar seasonality; overnight-minus-intraday momentum; 13F breadth; sector-ETF trend; plain long-term reversal (13-60m losers: 8yr 6/8 yrs but 26yr -0.041 0/24 = value-cycle regime bet). uptrend-filtered reversal DEAD on the 26yr (-0.024, 0/24, 11/25 yrs — same value cycle, filter trims losses ~1pp); VIXM tail hedge dead (gold's weaker cousin). Next: net share issuance (financing-decision signal, CRSP), 10y Treasury diversifier sleeve.

### I-49 · Breadth-blend rule (v12: bull mix -> bear mix 11/33/56 as breadth falls 60%->35%) · **TESTED EXP-061 (2026-09-22): always-bull worse on 8/8 starts and 12/25 years on the 26yr; ramp shifts (25-50, 45-70, 35-85) and milder bear mixes (40/35/25, 50/30/20) all worse on 8/8; value-heavy bear noise-level. RULE, THRESHOLDS AND BEAR MIX AT THE FRONTIER — do not re-test.** · EV: closed

### I-37 · Slow value/lowvol refresh (every 2nd rebuild) · **DEAD (EXP-059: +0.05 on the 8yr, ~0 on the 26yr — horizon-specific)** · EV: LOW-MED (cost)
Value and quality signals are slow; refreshing them every 40 sessions per book halves their turnover.

(I-02 water-filling, I-05 vol-normalised stop, I-06 exclude recent index additions, I-09 multi-horizon momentum
ensemble, I-10 52-week-high proximity, I-14 VIX term-structure gate (8yr only, data from 2016), and I-03 ex-ante
holdings vol (two variants + combined with the prompt overlay): ALL DEAD in EXP-059 stage 1 on the 26yr (I-09 -0.125, I-10 -0.113 Sharpe; I-14 ~0; I-03/I-05/I-06 <= 0). New in batch 3-5: **I-38 equal-weight momentum picks — SURVIVOR (stage 2: +0.012/+0.009 Sharpe, 24/24 starts both horizons, 6/8 and 19/25 years, 26yr CI excludes 0, intact at 2x cost, identical ex-crisis) — tiny (+0.3-0.5pp) but the most consistent thing EXP-059 found; live = one-line change in strategy1's score-weighting — CONFIRMED under the live 15% combiner cap (batch 7: +0.013/+0.009 vs cap0.15 base, 22-24/24, 6/8 + 18/25 years, cost-2x intact) — DEPLOYABLE candidate, owner's call**; I-41 daily overlay DEAD (no better than weekly, 11/25 yrs); I-42 value/lowvol equal weight ~0; I-43 momentum count 4/6/7 under equal weight — 5 stands (n6 8yr-only); I-39 overlay lookback 20d DEAD (stage-1 +0.037 fell to +0.021 at 24 starts, worse than lb40 on both); I-40 overlay trigger threshold — IRRELEVANT (0.75..0.95 identical). I-35 de-risk-only overlay + I-38: 26yr +1.19pp/+0.044 (24/24)/+5.3pp, CI_lo > 0 on both horizons, ex-crisis Sharpe-flat at -0.3..-0.7pp/yr return = cheap drawdown insurance; NOT a return claim.)
