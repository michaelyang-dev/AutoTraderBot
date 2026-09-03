# FINAL RECOMMENDATION — the tranche strategy vs what is live

**Status 2026-09-03: structure settled and audited; two audit items pending (marked ⏳).**
Everything below is from the clean-room independent engine, 24 monthly starts per horizon
(12 previously-used + 12 untouched even-month starts never examined during research), two
horizons (8yr 2018-25 on the clean universe file; 26yr 2001-25), $50k account, integer shares,
real financing curve. Full trail: `LOG.md` cycles 43-50, `RESEARCH_INDEX.md`.

## 1. What is live today (v12)

| | |
|---|---|
| Universe / sleeves | SP1500 PIT · momentum 0.50 / value 0.35 / lowvol 0.15 |
| Rebalance | **one book, every 20 sessions** |
| Leverage | **1.49× nominal × inverse-vol overlay** (clamp 0.30-1.00, target 0.15×1.49, 40d) |
| Credit gate | HY-OAS ≥ p95 expanding → **×0.50** |
| Stops / caps | 40% trailing · 15% position cap · top-5 momentum |

## 2. The recommended structure — three changes, two things deliberately kept

| | LIVE | **RECOMMENDED** | evidence |
|---|---|---|---|
| Rebalance | 1 book / 20d | **4 books, 5-session stride** | EXP-047/051/052: largest robust effect (+0.065/+0.033 Sharpe main effect); K=4 peaks on both horizons and is the only K positive in every sub-period; per-start σ halved |
| Credit gate depth | ×0.50 | **×0.00 (fully flat)** | EXP-053: monotone 0.00>0.25>0.50 on CAGR, Sharpe AND MaxDD on the 26yr; +5pp MaxDD vs 0.50; costs nothing on the 8yr |
| Sleeves | 50/35/15 | 70/21/9 | weakest component: +0.023 (8yr), +0.002 (26yr). Drop first if you want fewer moving parts |
| **Vol overlay** | ON | **ON — kept** | EXP-047: all five every-sub-period winners have it ON; my earlier "delete it" was a 2018-22 artefact (retracted, cycle 47) |
| Formula / stops / caps / top_n / vol params | — | **unchanged** | EXP-054 star: top_n, stop, gate pctile confirmed local optima; nothing else promoted |

Leverage is NOT part of the structure. It is the dial that decides how the improvement is spent
(section 3). Everything in the table above holds at every leverage tested.

## 3. Spending the improvement: two honest options

The structure buys Sharpe. Sharpe can be spent on **less risk** or on **more return**. They are
different products and the choice is yours, not mine.

### Option A — RISK spend: run it at 1.10× (fully measured)

| 24 starts | LIVE | **A @1.10×** | Δ | sign |
|---|---|---|---|---|
| 8yr CAGR | +26.85% | +26.95% | +0.10pp | 11/24 |
| 8yr Sharpe | 0.877 | **0.985** | **+0.108** | 16/24 |
| 8yr MaxDD | −38.0% | **−29.8%** | **+8.22pp** | **24/24** |
| 8yr worst-start MaxDD | −46.0% | **−31.3%** | | |
| 26yr CAGR | +13.93% | +14.24% | +0.31pp | 15/24 |
| 26yr Sharpe | 0.589 | **0.661** | **+0.072** | 20/24 |
| 26yr MaxDD | −55.9% | **−42.9%** | **+12.98pp** | **24/24** |
| 26yr worst-start MaxDD | −67.7% | **−49.7%** | | |
| per-start CAGR σ | 6.48 / 3.20pp | **3.09 / 1.45pp** | halved | |

Untouched-holdout vs used-sample agreement on ΔSharpe: within 0.005 on both horizons.

**What it costs, said plainly: Option A loses most calendar years by return.** 2/7 (8yr), 10/24
(26yr); median year −0.65pp on the 26yr. It wins the bad years big (2008 +6.1pp, 2022 +7.1pp,
2020 +4.7pp, 2015 +6.0pp) and gives up ground in bull years (2003 −11.9pp, 2021 −15.8pp,
2023 −8.6pp). That is what a 1.10× book with a hard gate does against a 1.49× book: less gross,
better crash handling. CAGR ends flat only because shallower drawdowns compound better. **If
"better in most years" is the bar, Option A does not pass it and I will not pretend otherwise.**

### Option B — RETURN spend: run the same structure at LIVE's own leverage ⏳ EXP-056

Same CAGR-at-same-drawdown comparison, leverage-fair. Ladder so far (26yr, holdout, gate 0.00):
1.00× −40.4% / +13.75% · 1.10× −43.1% / +14.33% · 1.25× −46.2% / +14.93%. LIVE sits at
−55.3% / +14.17%. Extrapolated, matching LIVE's drawdown lands near 1.45-1.5× at roughly
+16.5-17%; **that is an extrapolation and is being measured, not assumed.** Year-by-year at
1.40× and 1.49× will be reported the same way as Option A — including the losing years.

## 4. Audit gate for the final structure

| test | result |
|---|---|
| ≥12 starts, sign-consistency | 24 starts; MaxDD 48/48, Sharpe 16/24 & 20/24 |
| two horizons | both |
| untouched holdout | even-month starts, never examined; agrees with used sample to 0.005 Sharpe |
| independent engine | clean-room simulator with 5 deliberate implementation differences |
| sub-period consistency | 3/3 (8yr), 4/4 (26yr holdout) — 26yr 2023-25 cell is D9-contaminated (see caveats) |
| event concentration | N/A — net excess ≈0 (guard fires); the gain is the risk profile, not a few days |
| parameter star | nothing promoted; base confirmed local optimum on top_n / stop / gate pctile |
| paired cost sensitivity 1/2/3/5× | ⏳ EXP-055 |

## 5. Caveats you should weigh

1. **26yr universe file is missing ~10% of the modern universe** (BUGS D9). Deltas verified robust
   (same start, only the file differs → agreement to 0.07pp), but no absolute 26yr post-2017 figure
   should be quoted, and the 26yr 2023-25 cell is the least trustworthy number in the program.
2. **Sleeve tilt is noise on the long horizon.** 70/21/9 is carried because it never hurt, not
   because it is proven. Reverting to 50/35/15 changes little.
3. **Not deployable as-is.** Four books netted in one IBKR account, per-tranche stop peaks, one
   shared margin debit, staggered 20-session transition. Real engine work.
4. **Integer shares at $50k:** ~20 positions at ~$2,750 each at 1.10×. Workable; argues against
   >4 tranches or leverage below 1.0×.
5. **Live incident, fixed 09-02:** vendor date holes were silently dropping up to 80% of the
   universe from the momentum sleeve for hours at a time (root cause, fix, engine gate in
   `docs/LIVE_SYSTEM.md`). Unrelated to this research; relevant because the live book has
   occasionally been running on a fraction of its universe.

## 6. What I got wrong on the way here (all logged with evidence)
- Recommended deleting the vol overlay → retracted (2018-22 artefact).
- Reported "the edge has decayed" → retracted (leverage term, not signal).
- Recommended keeping the gate at 0.50 → reversed on the correct base (0.00 wins monotonically).
- Shipped a cache-depth "fix" for the coverage incident → refuted by its own telemetry; real cause
  found two weeks later.
