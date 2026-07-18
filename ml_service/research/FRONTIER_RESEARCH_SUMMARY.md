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
