# Walk-forward OOS validation — FINDINGS

**2026-07-15. Harness: `research/threadWF_validation.py`. Research-only.**
Answers: (1) is the ~28% in-sample CAGR achievable OOS? (2) does the credit p95 gate survive OOS?

## Method
Rules strategy -> walk-forward = re-SELECT config each year from past data, score held-out
next year, roll. Grid = top_n{3,5,8} x rebal{15,20,30} x stop{.30,.40,.55} (27), weights fixed
live. Selection metric = trailing expanding-window Sharpe. All at 1.49x flat + 6.3% financing.
Then apply the FROZEN credit p95 gate (causal expanding-pctile) to the walk-forward book.

## Results
**LONG 2001-25 (22 held-out yrs — reliable):**
| variant | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| Live config in-sample (OOS window) | +23.8% | 0.75 | −73.6% |
| **Walk-forward selected (TRUE OOS)** | **+21.3%** | 0.68 | −73.6% |
| **WF + credit p95 gate (OOS)** | **+23.1%** | 0.73 | **−57.0%** |
| Oracle best (ex-post upper bound) | +23.8% | 0.76 | −71.9% |

**SHORT 2018-25 (6 held-out yrs, 2020-25 — noisy):**
| variant | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| Live config in-sample (OOS window) | +38.0% | 1.02 | −39.3% |
| **Walk-forward selected (TRUE OOS)** | **+24.9%** | 0.81 | −38.6% |
| WF + credit p95 gate (OOS) | +24.9% | 0.81 | −38.6% |

## Verdict
1. **28% is IN-SAMPLE. Honest OOS haircut:** long ~2.5pp (23.8→21.3, reliable, 22 folds);
   short ~13pp (38→25) but INFLATED — 2020-25 is a monster-momentum window and a 2y-burn-in WF
   selection is noisy (missed the ex-post-best live config in 4/6 yrs). Long is the trustworthy
   estimate: **expect ~21% bare / ~23% with credit gate OOS, Sharpe ~0.68-0.73** (1.49x flat).
2. **Credit p95 gate CONFIRMED OUT-OF-SAMPLE:** on the re-selected walk-forward long book it cut
   MaxDD −73.6%→−57.0% (+16.6pp) AND +1.8pp CAGR (recovers the WF haircut), Sharpe 0.68→0.73.
   2008 de-risk used only pre-2008 data -> genuine OOS, not curve-fit. Added nothing on 2020-25
   (no credit crisis there) — consistent.
3. **Live config is well-chosen** — its ex-post Sharpe (0.75) ~= oracle (0.76); not overfit to a
   lucky corner. But a mechanical WF drifts to WIDER stops (0.55) and never re-picks the live 40%,
   landing ~2.5pp lower — the backtested 28%/24% carries config-selection hindsight.

Caveats: selection = trailing Sharpe over a fixed 27-config grid (other metric/grid shifts the
number); 1.49x flat, no vol-scaling. OOS CAGR is a range, not a point.
