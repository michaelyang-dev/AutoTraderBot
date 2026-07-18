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
