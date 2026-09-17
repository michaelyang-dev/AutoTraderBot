# EXP-059 frontier search — result for the owner (2026-09-17, PRELIMINARY — research numbers, not audited to the live-parity standard of §0 of FINAL_RECOMMENDATION.md)

**Question asked (2026-09-13):** can the deployed tranche book be made better — more return, higher Sharpe, lower
drawdown — consistently, without the gain being inflated or carried by one year?

**Answer:** yes, modestly, by three small changes that each pass the promotion gate on the 26-year horizon and hold at
2x cost. Together ("the package") they add about **+1.5pp CAGR, +0.06 Sharpe and +5pp of MaxDD** on the 26yr, better in
17 of 25 calendar years, bootstrap 90% CI on the Sharpe edge [+0.019, +0.110], with the edge still +0.035 Sharpe when
2008 and 2020 are removed from both arms. On the 8yr the package shows +3.1pp CAGR, but 67% of that is 2020 and it must
not be quoted as expected return. Nothing is deployed; each item is the owner's call.

## The package (all measured on 24 monthly starts x 2 horizons, v2 universes, deployed 1.49x, whole shares)

| change | what it is | 26yr (vs deployed) | 8yr (vs deployed) | robustness |
|---|---|---|---|---|
| **A. Equal-weight momentum picks** | the momentum sleeve's 5 names get 1/5 each instead of score-weights (capped at 2/N) | +0.30pp / +0.009 Sharpe (24/24 starts) / MaxDD ~0 · 19/25 years | +0.52pp / +0.012 (23/24) · 6/8 years | identical at 2x cost; identical with crisis years removed; 26yr CI > 0; 5 names confirmed (4 and 6-7 worse); inverse-vol worse |
| **B. De-risk-only prompt overlay** | on every tranche day, each NON-rebuilding book is trimmed to today's vol-scale x credit-gate target if it is >5% above it; never levered back up | +0.77pp / +0.036 (24/24) / **+5.5pp MaxDD** · 14/25 years | +1.60pp / +0.070 (23/24) / +4.9pp · 3/8 years, 2020 = 76% of gain | intact at 2x cost; threshold (0.75-0.95), lookback (40 best), daily vs weekly (weekly best) all tested; **ex-crisis it is Sharpe-flat at -0.3..-0.7pp/yr = cheap insurance, not return** |
| **C. Sleeve split 80/15/5** (from 70/21/9) | more momentum, which the overlay's drawdown control makes affordable | +0.47pp / +0.014 (24/24) vs A+B · 17/25 years | +1.04pp / +0.014 (23/24) · 7/8 years | passes pairwise gate both horizons, intact at 2x cost; needs B (alone: fails); **90/7/3 and pure momentum FAIL — 2023-26 sub-period turns negative (-0.04 / -0.10); 80 is the limit and is itself slightly negative there** |
| **A+B+C** | | **+1.55pp / +0.060 (24/24, CI [+0.019,+0.110]) / +5.4pp MaxDD · 17/25 yrs · top year 23% · ex-crisis +0.035 (24/24) · 2x cost +0.058** | +3.12pp / +0.096 (24/24) / +5.3pp · 5/8 yrs · **top year 67% (2020)** · ex-crisis +0.031 (23/24) | GATE059 26yr PASS every check; 8yr FAIL on year-concentration only |

Levels: 26yr deployed +18.16% / 0.719 / -46.6% -> package +19.71% / 0.779 / -41.2% (realized vol 28.3% vs 29.2%).
8yr deployed +32.42% / 1.026 / -36.7% -> package +35.54% / 1.122 / -31.4% (vol 31.6% vs 32.5%).

Per-year (mean over 24 starts, deployed -> package), 26yr: 2002 -25.5 -> -26.4 · 2003 +71.1 -> +72.8 · 2004 +22.2 -> +22.8 ·
2005 +22.9 -> +27.0 · 2006 +10.0 -> +8.2 · 2007 +2.0 -> +4.1 · **2008 -29.1 -> -19.7** · 2009 +37.1 -> +35.4 · 2010 +10.5 -> +11.2 ·
2011 -7.5 -> -4.2 · 2012 +36.6 -> +38.0 · 2013 +40.6 -> +39.1 · 2014 +4.3 -> +4.3 · 2015 +1.7 -> +4.1 · 2016 +24.6 -> +24.4 ·
2017 +14.3 -> +15.3 · 2018 -4.8 -> -4.2 · 2019 +25.3 -> +26.2 · **2020 -1.4 -> +4.6** · 2021 +49.7 -> +52.2 · 2022 +3.2 -> +2.9 ·
2023 +41.1 -> +40.6 · 2024 +75.5 -> +72.9 · 2025 +19.2 -> +20.6 · 2026 +74.1 -> +77.1. The pattern is the honest one for a
de-risk rule: the big gains are 2008 and 2020, most other years are within +-2pp, and the strongest bull years (2013, 2024)
give up 1.5-2.6pp.

## What did NOT work (so nobody re-tests it)
Weekly signal exits for all books (drawdown instrument, no Sharpe edge at 2x cost); the overlay applied daily (no better than
weekly, more calm-year cost); overlay lookback 20/60; overlay threshold; gate depth 0.25/0.50 under the overlay; K = 2/5/8/10 books;
water-filling cap; min-trade band; slow value/lowvol refresh; vol-normalised stops; exclude index additions; momentum ensemble;
52-week-high; VIX term gate; ex-ante holdings vol; momentum count 4/6/7; inverse-vol inside the momentum sleeve; value/lowvol
sleeves equal-weighted; value top 5/7/15; lowvol top 5; min weight 2%; stop 35/45/50; vol-clamp floor; vol target (a leverage dial
only); the 15% live cap vs the clean room's 10% (inert); starting capital $15k-$1M (whole-share truncation costs base and package
alike, -0.2..-1.3pp/yr at live size).

## Caveats the owner should hold onto
1. The 8yr CAGR gain is 2020. Expected improvement is the 26yr figure: ~+1.5pp CAGR, ~+0.06 Sharpe, ~+5pp MaxDD.
2. B is insurance: it costs ~0.3-0.7pp/yr in calm years and pays in crashes. Over 26 years it nets positive; over any 3-4 calm
   years it will look like a small drag. That is the design, not a defect.
3. C is a tilt toward the factor that did best in-sample, and its 2023-26 contribution is already slightly negative. 80/15/5 is
   the last point the evidence supports; do not go further.
4. All numbers are clean-room research numbers at 10 bp + 10 bp cost (2x = 40 bp also shown). The live book's real cost after the
   Tiered switch is ~8 bp commission + slippage; the "2x" rows are the conservative bound.
5. Taxes are not modelled; B's trims realize gains (moot while the ~$200k loss carryforward lasts).
6. This is a 24-start, two-horizon, gate-tested result, but every switch was chosen on the same data. The odd/even start split
   and the sub-period checks are the out-of-sample evidence available; a true holdout does not exist for a 26-year sample.

## If deployed — the three live changes (not done; owner's go required)
1. `strategies/multi_strategy_engine.py` strategy1: return equal weights for the top-5 (replace score/total with 1/N).
2. `PROD_WEIGHTS_BULL` (multi_strategy_engine.py + live_config.py + signal_server 15% cap unchanged): 0.80 / 0.15 / 0.05.
3. `ibkr_engine.py` rebalance_tranche(): after the rebuilding book is done, for each other book compute target = NAV/4 x 1.49 x
   vol_scale x gate; if book_gross / target > 1/0.95, sell int(q x target/gross) - q shares per name (skip trades < 0.3% of the
   book); never buy. Persist the ledger per fill (already in place). Tests: extend tests/test_tranche_engine.py with a
   de-risk-trim case and a never-up case. Then the usual: box bundle deploy, restart after the close, LIVE_SYSTEM.md record.

## Leverage line (batch 13, 24 starts) — the owner's risk choice

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

Read: the package at 1.25x has roughly the deployed book's CAGR with far less drawdown; at 1.49x (no leverage change) it
adds ~1.5pp CAGR and ~5pp MaxDD on the 26yr; at 1.65x it buys back the deployed book's drawdown for ~3pp more CAGR (26yr).
Sharpe is flat across the line (leverage is a dial, not an edge) — pick the MaxDD you can hold.
