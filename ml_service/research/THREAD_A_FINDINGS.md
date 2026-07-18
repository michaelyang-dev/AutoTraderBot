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
