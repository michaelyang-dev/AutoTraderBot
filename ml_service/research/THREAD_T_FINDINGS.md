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
