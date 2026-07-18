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
