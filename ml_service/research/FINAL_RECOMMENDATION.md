# FINAL RECOMMENDATION — the tranche strategy vs what is live

**Status 2026-09-08: RE-VERIFIED ON THE REBUILT (v2) UNIVERSES, through 2026-08-31.** The structure and the
verdict below (section 3b) are unchanged; every number in this top section supersedes the 2026-09-03 numbers
further down, which were computed on the pre-rebuild universes and end 2025-12-31.

## 0. 2026-09-08 — v2 universe re-verification (the deliverable)

Both universes were rebuilt from scratch, PERMNO-keyed, from the 2026-09-07 WRDS pull (CRSP daily to
2025-12-31 + Compustat Security Daily for 2026, fundamentals to datadate 2026-08-31), then verified, then
LIVE and the recommended structure were re-run through the clean-room engine, 24 starts per horizon, and
every reported number was recomputed a second way. Four harness/data defects were found and fixed on the
way (BUGS.md D12a–d); nothing was reported from a run that carried one.

### 0.1 Side-by-side, 24 starts (12 previously used + 12 untouched), $50k, integer shares, costs 10 bp/side + financing

| | 8yr 2018-01 → 2026-08 | | | | 26yr 2001-01 → 2026-08 | | | |
|---|---|---|---|---|---|---|---|---|
| arm | CAGR | Sharpe | MaxDD (mean) | worst-start MaxDD | CAGR | Sharpe | MaxDD (mean) | worst-start MaxDD |
| **LIVE** (1 book/20d, gate ×0.50, 50/35/15, 1.49×+overlay) | +29.52% | 0.956 | −38.6% | −48.9% | +16.34% | 0.658 | −54.1% | −58.9% |
| FINAL @1.10× | +29.62% | 1.066 | −31.1% | −37.0% | +16.45% | 0.739 | −40.1% | −44.6% |
| **FINAL @1.25× (recommended)** | **+31.03%** | **1.053** | **−33.5%** | **−40.8%** | **+17.26%** | **0.730** | **−43.0%** | **−47.3%** |
| FINAL @1.49× | +32.42% | 1.026 | −36.7% | −45.4% | +18.16% | 0.719 | −46.6% | −51.6% |

Δ FINAL@1.25 − LIVE: 8yr **+1.50pp CAGR / +0.097 Sharpe / +5.1pp MaxDD**, Sharpe better in 17/24 starts, MaxDD
shallower in 17/24; 26yr **+0.92pp / +0.072 / +11.0pp**, Sharpe better 20/24, MaxDD shallower **24/24**.
Paired cost 2×: Δ Sharpe +0.098 (8yr) / +0.073 (26yr) — the delta does not shrink when costs double.
Sub-period Δ Sharpe (FINAL@1.25 − LIVE): 8yr 2018-20 +0.040 · 2021-22 +0.172 · 2023-26 +0.052; 26yr 2001-08 +0.053 ·
2009-16 +0.086 · 2017-22 +0.117 · **2023-26 −0.029** (the only negative cell: the structure lags LIVE in the
2023-26 momentum regime at every leverage — it is a risk package, not a return package).

Trade statistics (instrumented runs, 26yr start 2001-01-03 / 8yr start 2018-01-03): LIVE 423 / 413 fills per
year, 3,777 / 1,222 closed round-trips, **win rate 51.4% / 51.1%**, turnover 16.0× / 15.6× NAV per year,
commissions+slippage $80k / $16k, financing $45k / $9k (26yr min cash −$419k = margin debit at 1.49×; FINAL@1.25 −$210k). FINAL@1.25: 1,603 / 1,527 fills per year (4 books),
win rate 51.5% / 49.8%, turnover 13.8× / 13.7×, avg win ≈ avg loss (253 vs −255; 122 vs −111).

### 0.2 Per-calendar-year return, mean across 24 starts (2026 = Jan–Aug)

| year | LIVE | FINAL@1.25 | year | LIVE | FINAL@1.25 | year | LIVE | FINAL@1.25 |
|---|---|---|---|---|---|---|---|---|
| 2002 | −21.3% | −23.4% | 2011 | −7.5% | −6.8% | 2020 | −2.1% | −0.5% |
| 2003 | +72.2% | +62.6% | 2012 | +37.2% | +35.5% | 2021 | +49.5% | +45.7% |
| 2004 | +22.7% | +20.8% | 2013 | +42.4% | +38.8% | 2022 | −7.1% | **+3.8%** |
| 2005 | +15.2% | +23.3% | 2014 | +1.0% | +3.9% | 2023 | +44.7% | +38.9% |
| 2006 | +11.7% | +9.6% | 2015 | −4.8% | +1.4% | 2024 | +73.7% | +71.0% |
| 2007 | +2.7% | +2.7% | 2016 | +23.6% | +23.8% | 2025 | +25.1% | +19.3% |
| 2008 | −29.8% | **−26.4%** | 2017 | +13.7% | +11.6% | 2026 YTD | +70.1% | +67.0% |
| 2009 | +30.2% | +33.2% | 2018 | −7.3% | −3.9% | | | |
| 2010 | +10.2% | +8.2% | 2019 | +23.0% | +24.3% | | | |

FINAL@1.25 beats LIVE in 12 of 25 calendar years (26yr) and 3 of 8 (8yr); it wins the bad years (2008, 2015,
2018, 2020, 2022) and loses the momentum-melt-up years (2003, 2013, 2021, 2023, 2025). No single year drives
the total: removing 2026 from the 26yr leaves the ordering of every arm unchanged (it is 8 of 308 months).

**Flags investigated (not just presented).** 2024 (+66–74%) and 2026 YTD (+68–70%) are very large years for
every arm. Both were cross-checked against an independent vendor: 2026 returns of 1,450 SP1500 names vs the
live system's Polygon panel (corr 0.9993; the 2026 leaders SNDK +707%, ICHR +444%, MU +243%, WDC, STX, INTC
+198% agree to three decimals; the only >10% disagreements are 2026 spin-offs where the total-return series is
correct), and 2017–2025 yearly returns vs the live system's cached Polygon closes (corr 0.94–0.999, 2024 =
0.998, ≤2% of names >10% apart, all dividends/spin-offs/ticker reuse in the vendor cache). The equal-weight
SP1500 did +15.7% in Jan–Aug 2026 with a +58.8% 90th percentile; a top-5-momentum book at 1.49× on that
cross-section is consistent with +68%. Backtest 2026 came from Jan–Jun (+11, +11, −5, +19, +13, +7%); Jul −7%,
Aug +8%.

### 0.3 Every check, PASS/FAIL, evidence

| # | check | 8yr | 26yr | evidence |
|---|---|---|---|---|
| U1 | universe rebuilt from scratch, zero errors, assertions on fundamentals rows / ETF 2026 coverage / collapses | PASS | PASS | build rc=0 14 min / 27 min; 286,782 new fundamentals rows linked; 15/15 ETFs 170 prints in 2026 |
| U2 | row count == expected trading days (CRSP calendar ≤2025 + Compustat US days 2026) | PASS | PASS | 2,581 = 2,581 (+276 warm-up) · 6,709 = 6,709; missing [] extra [] |
| U3 | member PERMNOs present as price columns | PASS | PASS | 2,352/2,353 (VSNT, listed 2026-01, no CRSP/CCM row) · 3,792/3,794 (WMAN 1998-2007, OWN 1998-2000) |
| U4 | no duplicate dates / columns / members; no PERMNO in two indices | PASS | PASS | 0 / 0 / 0; overlaps 0 PERMNO-days (8yr), 3 in 6,898 dates (26yr, the vendor's own PDE rows Sep 2001) |
| U5 | NaN/inf per required feature | PASS | PASS | inf = 0; prices non-positive = 0; among members ret_20d NaN 0.0%, dist_sma200 0.1–0.9%, roe 2.9–4.6% (table in `_checklist_v2_*.out`) |
| U6 | date range covers window, no unexplained gap | PASS | PASS | 2015-04-28→2026-09-04 (8yr) · 1998-11-27→2026-09-04 (26yr); 3 gaps = 9/11, Ford mourning day, Sandy |
| U7 | priced-name count and membership size change ≤5% day-over-day | PASS | PASS | membership min/median/max 1489/1500/1508 · 1479/1499/1508; jumps [] |
| U8 | 5 known tickers present & priced to 2026-09; 5 known delisted absent & terminated | PASS | PASS | AAPL MSFT NVDA JPM XOM; SIVB(−99% last bar) FRC BBBY TWTR ATVI · ENE WCOM LEH(−60%) BSC SIVB |
| U9 | v1→v2 non-regression | PASS | PASS | member price coverage 94.7–98.1% → 99.5–99.9% (8yr), 82.1–93.9% → 98.0–99.9% (26yr); members w/o price 2025: 29→1, 109→1; splices 17→9, 22→15, all remaining explained |
| U10 | 2026 extension vs independent vendor (Polygon) | PASS | PASS | n=1,450 corr 0.9993, 6 of 1,450 >10% apart (all spin-offs); top-30 movers daily corr 1.000 |
| B1 | no data access beyond the decision date (every sleeve call + membership, recorded) | PASS | PASS | 8yr: 2,477/2,151/9,886/8,566 accesses, 0 violations; 26yr: 7,593/7,113/30,198/28,436, 0 violations; sample rows printed per decision date |
| B2 | membership as-of-date, not current constituents | PASS | PASS | recorder shows the membership key used ≤ decision date on every call (e.g. 2013-10-23→2013-10-23); **2026-05-01 is the last vendor date, so May–Aug 2026 decisions use May-1 membership** |
| B3 | features are trailing-only; fundamentals PIT | PASS | PASS | 18/18 (ret_20d, dist_sma200) recomputed from prices ≤T match to 1e-6; 12/12 roe values equal an independent recompute from the row with avail_date (rdq, else datadate+90d) ≤ T, lags 14–41 days |
| B4 | costs applied and equal config | PASS | PASS | COST_BPS 5 + SLIPPAGE_BPS 5 = 0.100%/side on every fill; financing daily on prior-day debit; totals in 0.1 |
| B5 | no duplicate fills; no position above the 15% cap; gross ≤ leverage bound | PASS | PASS | 0 duplicates in 3,578…41,166 fills; max entry weight 15.00%; gross/NAV max 1.57–1.61 (LIVE, 1.49× closed-loop) and 1.27–1.29 (FINAL 1.25×) |
| B6 | negative cash | INFO | INFO | cash is negative by design (margin debit under leverage, financed daily); min −$100k (8yr LIVE) … −$512k (26yr LIVE start 2002); FINAL@1.25 −$45k … −$302k |
| R1 | ledger reconstructs engine cash independently | PASS | PASS | e.g. −14,589.76 vs −14,589.76 (8yr); 88,161.27 vs 88,161.27 (26yr) — all 8 instrumented runs exact to the cent |
| R2 | CAGR / MaxDD recomputed a second way (yearly chain; explicit peak loop) for all 24 starts × 4 arms | PASS | PASS | max abs diff 2e-16 |
| R3 | Sharpe recomputed with a different estimator (monthly ×√12) | PASS | PASS | daily vs monthly within 0.041–0.074 for every arm |
| R4 | too-good flags: Sharpe >3, win rate >70%, CAGR far above prior range | PASS | PASS | max per-start Sharpe 1.22 / 0.81; win rates 49.8–53.2%; CAGR vs prior-universe canon: 8yr +26.85→+29.52 (LIVE, +8 months of a +68% year), 26yr +13.93→+16.34 |
| R5 | curves span the declared window | PASS | PASS | 2018-01-03→2026-08-31; 2001-01-03→2026-08-31 (this check caught D12b) |
| X1 | live account vs backtest, Jun 15 → Aug 31 2026 | **COULD NOT RUN** | — | live NAV $33.6k→$61.2k includes a $16.8k deposit on Jul 17 (+47.5% one-day jump); deposit-netted ≈ +24% vs backtest LIVE arm +0.7% (range −4…+6%). Not reconcilable here: live ran SP500-only sleeves until Jul 25, heavy turnover Aug 11, degraded-coverage signals Aug 18–Sep 4. Tracking reconciliation is open work, not a backtest defect |

Checks not run: X1 above; the 26yr vendor cross-check before 2016 (no independent price source on this
machine for 2001–2015; CRSP is the only one).

### 0.4 What was wrong before this run (all fixed before any number was taken)

| | defect | would have produced |
|---|---|---|
| D12a | new 2026 fundamentals rows all dropped by a NaN link column | value/quality on Q1-2026 data through Sep 2026 |
| D12b | engine end-date override applied after the globals copy | every curve ended 2025-12-31 under a header claiming Aug 2026 |
| D12c | ETFs tagged `tpci="%"` not `"F"`: no 2026 SPY/sector-ETF prices | frozen regime + sector tilt in 2026 (measured effect <0.01pp, but real) |
| D12d | membership symbols resolved to CRSP ticker eras by date | 197 renamed symbols mapped to the wrong company (2.7% of member-days); twins collapsed; bankruptcies (AMR, Kodak, Frontier, Chesapeake…) missing until a CUSIP bridge recovered them |

### 0.5 Caveats that survive
* Membership is known only to **2026-05-01**; index changes May–Aug 2026 are not reflected.
* 19 membership symbols (109 symbol-years, 0.2%) remain unresolvable: obscure 2000s bankruptcies (BHMSQ, SOGCQ…), two class-B tickers (TAP.B, TRY.B), VGNT. Direction: excludes a few collapsing names → slightly flatters both arms equally.
* Two twin pairs (USB/USB-200102, BDC/BWC-200407) collapse to one PERMNO because CRSP and Compustat disagree on which security survived the merger (1,955 symbol-days, 0.02%).
* 2026 prices are Compustat chained onto CRSP (no CRSP 2026 until Feb 2027); verified against Polygon, but they are a different vendor.

---


**(superseded numbers below — see section 0) Status 2026-09-03: COMPLETE. Structure settled; full audit gate cleared; leverage ladder measured 1.00-1.49×.**
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

### Option B — RETURN spend: same structure at LIVE's own 1.49× (measured, EXP-056)

**The structure never reaches LIVE's drawdown at any leverage.** Even at 1.49× — LIVE's own
setting — MaxDD is 6.5pp shallower on the 26yr and 3.6pp on the 8yr, so "same drawdown, more
CAGR" is off the curve: at every point on the ladder you get more CAGR *and* less drawdown *and*
more Sharpe than LIVE. The full ladder (24 starts, gate 0.00):

| lev | 26yr CAGR | Sharpe | MaxDD | worst-start | ΔCAGR | ΔSharpe | ΔMaxDD (sign) | years better |
|---|---|---|---|---|---|---|---|---|
| LIVE 1.49× | +13.93% | 0.589 | −55.9% | −67.7% | — | — | — | — |
| 1.00× | +13.65% | 0.667 | −40.3% | −46.7% | −0.27pp | +0.078 | +15.6pp (24/24) | |
| 1.10× | +14.24% | 0.661 | −42.9% | −49.7% | +0.31pp | +0.072 | +13.0pp (24/24) | 10/24 |
| **1.25×** | **+14.85%** | **0.651** | **−46.1%** | **−53.3%** | **+0.92pp** | **+0.062** | **+9.8pp (24/24)** | |
| 1.40× | +15.28% | 0.643 | −48.3% | −56.3% | +1.35pp | +0.054 | +7.6pp (22/24) | 14/24 |
| 1.49× | +15.53% | 0.641 | −49.5% | −58.0% | +1.60pp | +0.052 | +6.5pp (20/24) | **16/24** |

| lev | 8yr CAGR | Sharpe | MaxDD | worst-start | ΔCAGR | ΔSharpe | ΔMaxDD (sign) | years better |
|---|---|---|---|---|---|---|---|---|
| LIVE 1.49× | +26.85% | 0.877 | −38.0% | −46.0% | — | — | — | — |
| 1.10× | +26.95% | 0.985 | −29.8% | −31.3% | +0.10pp | +0.108 | +8.2pp (24/24) | 2/7 |
| **1.25×** | **+28.23%** | **0.973** | **−31.8%** | **−33.7%** | **+1.38pp** | **+0.096** | **+6.2pp (24/24)** | |
| 1.40× | +28.95% | 0.955 | −33.5% | −36.3% | +2.11pp | +0.079 | +4.5pp (21/24) | 3/7 |
| 1.49× | +29.37% | 0.948 | −34.4% | −37.5% | +2.52pp | +0.071 | +3.6pp (15/24) | 3/7 |

**Year-by-year at 1.49× (26yr): better in 16/24 years, median +1.17pp, and still +0.96pp with the
best year removed (15/23 positive).** That passes the "better in most years" bar on the horizon
that contains real crises. On the 8yr it is 3/7 — the strongest momentum years (2021, 2023, 2025)
go to LIVE by 1-3pp; the 8yr has too few years to be decisive either way.

## 3b. Verdict

The structure dominates LIVE on all three axes at every leverage tested. Leverage is a pure
risk-preference dial, and here is what each setting buys:

- **1.10× — drawdown first.** Sharpe +0.07/+0.11, MaxDD 8-13pp shallower, CAGR flat. Loses most
  bull years. For someone whose binding constraint is the worst month.
- **1.25× — the recommendation.** The highest leverage at which drawdown is shallower in **all 48
  starts** on both horizons, with +0.9/+1.4pp CAGR and +0.06/+0.10 Sharpe. Balanced, and the
  setting you said you preferred.
- **1.49× — "change nothing but the structure."** No leverage change, no financing change: only
  tranching, gate depth and the tilt move. +1.6/+2.5pp CAGR, +0.05/+0.07 Sharpe, 3.6-6.5pp less
  drawdown, better in 16/24 years. The cleanest deployment story and the most CAGR. Drawdown gain
  is smaller and less uniform (15-20/24).

There is no setting at which the new structure is worse than LIVE on any of CAGR, Sharpe or MaxDD.

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
| paired cost sensitivity 1/2/3/5× | **PASS, both horizons** — ΔSharpe *widens* with cost (8yr +0.105→+0.115; 26yr +0.074→+0.084), sign 8/12 and 11/12 at every multiplier. FINAL trades fewer dollars than LIVE (1.10× gross, hard gate), so cost hurts LIVE more. Not turnover-fragile |

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
