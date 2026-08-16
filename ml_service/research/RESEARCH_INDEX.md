# RESEARCH INDEX — everything ever tested on this strategy, and its verdict

**Master list. Start here.** `research/` now holds **6 markdown files, not 19** — the 14 older
findings docs were merged verbatim into `ARCHIVE_PRIOR_PROGRAMS.md` and the originals deleted,
because they overlapped heavily and it was not obvious which was current.

**Master list. Start here.** One line per idea, newest program first. If an idea is not on this
list, it has not been tested. If it is, do not re-test it without reading the linked detail —
several entries record *why* a re-test would be wasted, and several record why an earlier
"dead" verdict is **not** safe.

| file | what it holds |
|---|---|
| `RESEARCH_INDEX.md` | **this file** — the master list of everything tried |
| `LOG.md` | one full entry per experiment: hypothesis, method, numbers, audit, verdict |
| `BASELINE.md` | the current audited champion + the bar a challenger must clear |
| `BUGS.md` | every leak, accounting error and data defect found (A1-A9, B1-B8, C, D1-D8, E1-E6, F1-F6) |
| `IDEAS.md` | ranked backlog of untested hypotheses |
| `ARCHIVE_PRIOR_PROGRAMS.md` | the 14 older findings docs, merged verbatim (dynamic leverage, threads A/B/C/R/S/S3/T/X, futures, frontier, walk-forward, live-mirror, VRP plan) — **all superseded, read its warning header first** |
| `../data/wrds/UNIVERSE_MANIFEST.md` | which universe pickles are clean, which are poisoned |
| `../docs/LIVE_SYSTEM.md` | what is actually deployed |

---

## ⚠️ READ FIRST — three facts that invalidate large parts of the older record

1. **Both universe pickles were rebuilt 2026-07-28 after 8 defects** (look-ahead membership,
   survivorship via suffixed tickers, spliced ticker histories, deleted delisting losses, …).
   26yr CAGR was overstated by **10.4pp**. **Every number predating 2026-07-28 is suspect.**
2. **The "0.6pp/start noise floor" understated reality ~4×.** True 8yr per-start σ is **7.07pp**.
   Any conclusion from a 3-4 start A/B is unreliable — this has produced a **wrong sign** and a
   **retracted win**. Standard is now **≥12 monthly starts, judged on sign-consistency**.
3. **The research baseline itself was wrong twice** (BUGS A5, A9) — flat financing overcharging
   3.05pp/yr, and a missing credit gate. **Honest 26yr baseline: +12.72% / 0.549 / −55.9%**,
   not the +11.26% / 0.509 / −64.8% quoted throughout the older docs.

---

# PROGRAM 3 — autonomous alpha research, 2026-08-14/15 (32 cycles, 1,850 configs)

## ✅ Candidates that passed audit (NONE DEPLOYED)

| # | change | 26yr effect | audit status |
|---|---|---|---|
| 1 | **Rebalance-phase tranching K=4** | +0.52pp CAGR, +0.031 Sharpe, worst DD −71.1%→−64.5% | full gate passed ($33k, cost×3, concentration 11-28%) |
| 2 | **Remove the inverse-vol overlay**, constant leverage | +0.21pp CAGR, **+0.035 Sharpe (12/12)**, **+4.11pp MaxDD (12/12)** | passed; advantage **grows** with cost |
| 3 | **BOTH (1+2)** — they stack additively | +0.07pp CAGR, **+0.074 Sharpe (10/12)**, +7.7pp MaxDD | 12 audit cells passed; **misses 8yr Sharpe bar at 8/12** |

**Honest limit on all three:** the *drawdown* gain is a **level effect** (matched dMaxDD ≈ 0).
Exposure-independent are the Sharpe residual (+0.03…+0.08) and a ~60% cut in outcome dispersion.

## ❌ Killed this program — with the reason

| idea | verdict | why |
|---|---|---|
| Earnings-window variance avoidance | KILLED | my first pass was confounded (t=+20.2 → **+0.73** within-date); real effect ≈ +0.016 Sharpe, below detection floor |
| Index-addition reversal | VOID | right sign, decays correctly, but fails multiple-testing (6 cells, Bonferroni p≈0.18) and is economically negligible at top-5 concentration |
| Partial-adjustment rebalancing | KILLED | −7 to −10pp CAGR. Dilutes the book with **stale** positions. Tranching ≠ smoothing |
| Ex-ante holdings-based vol estimator | KILLED | **0/12** on matched exposure (−0.092…−0.099). No timing skill exists to improve |
| Vol-normalised trailing stop | KILLED | dead at every k, 3/12; flat 30% and 50% controls also worse than the deployed 40% |
| Alternative momentum rankings (multi-horizon, 52w-high, vol-adjusted, blend) | KILLED | −0.11 to −0.32 Sharpe; **worse the more they replace** the deployed ordering |
| Book breadth (top_n 3→40) | FALSIFIED | Sharpe peaks at **5 on BOTH horizons**; monotone decline either side |
| **`roe` momentum-quality filter** | **KILLED** | passed shift/cost/pool robustness and 10/12 on 26yr — then **1/5 on the pre-registered profitability family, 0/5 on 8yr**, and −0.160 at 2/12 on the 8yr. One lucky draw out of nine arms |
| gp_assets / net_margin / operating_margin / gross_margin quality | KILLED | −0.021 to −0.073 (26yr), −0.16 to −0.26 (8yr) |
| Sleeve-level risk parity | FAILS two-horizon | 8yr −0.015…−0.053 (0-2/12); 26yr Sharpe-neutral with a drawdown dose-response. A CAGR-for-DD trade, not a mistake |
| Cross-sleeve overlap (boost / flatten) | KILLED | \|dSharpe\| ≤ 0.008 everywhere. Agreement carries no information; current doubling is harmless. **Question closed** |
| Momentum universe tiering (sp500 / sp400 / sp600 / combos) | KILLED | **every subset loses**; mid-only −14.56pp. Pool **breadth** beats size tier |
| Alternative credit gates (`ccc_bb`, `tedrate`, OR-combinations) | KILLED | all negative, 0-3/12, negative matched residuals. **The deployed HY-OAS p95 gate is not improvable from local data** |
| Mid-cycle signal exit | NOT PROMOTED | the ONLY mechanism whose drawdown gain **survives** exposure-matching (+5.6…+7.8pp) — but matched Sharpe is **negative**. A preference instrument |
| Momentum sleeve tilt (60/70/80% momentum) | NOT PROMOTED | clean monotone dose-response **through** the baseline, but 7-8/12 on Sharpe |
| Cap water-filling | MEASURED, NEGLIGIBLE | the 10% cap **does** bind (gross +2.3%) but fixing it is worth only +0.39pp. **Docstring caveat closed** |
| `park idle cash IEF` | **VOID — never executed** | returned exactly 0.000 on every metric; the hook only fires below 1.0× leverage, which never happens |

## 📊 Facts established about the strategy (not ideas — measurements)

- **100% of the selection edge is the RANKING; the filters are worth −0.06pp.** Confirmed twice.
- **Concentration is load-bearing.** top_n=5 optimal on both horizons; every universe subset loses.
- **1.49× is not optimal** — Sharpe and Sortino peak at the *lowest* leverage tested, 12/12.
- **The vol overlay has NEGATIVE timing skill** (~−1pp CAGR/yr at matched exposure, both horizons).
- **The credit gate is excellent** (+0.61pp / +0.021 / +8.54pp, 12/12) **and not improvable**.
- **The strategy BEATS passive EW SP1500 over 26yr** (+1.55pp CAGR, +2.6pp DD) — the earlier
  "at parity / behind on risk" conclusion was an artefact of the two baseline defects.
- Selection is worth **+20pp CAGR on the 8yr but only +4.5pp on the 26yr** — the recent window
  flatters stock-picking ~4×.

---

# PROGRAM 2 — dynamic leverage, 2026-08-13/14 → `ARCHIVE_PRIOR_PROGRAMS.md`

**All timing rules dead:** EWMA vol · asymmetric vol windows · equity-curve trend ·
distance-from-peak · recovery ramp · vol-of-vol · market vol · curve inversion · `umd_crash`.
**Levering UP hurts** monotonically. **Off-cadence gate application** passed 23/24 starts,
walk-forward OOS, a sensitivity plateau and all three sub-periods — then **died at 99.6% event
concentration** (its whole edge was ~5 days; helped 2008/2020, hurt 2001/2022).

# PROGRAM 1 — thread programs, 2026-06/07 → `ARCHIVE_PRIOR_PROGRAMS.md`

Cross-sectional rank axis · ML ranking (orth IC 0.008) · news sentiment (FNSPID + FinBERT, IC≈0)
· weather · attention/Wikipedia · short strategies (27+ rounds) · crypto · futures global-macro ·
aging · dispersion · book-crowding · opportunity-set · residual momentum · hysteresis ·
entry/exit-timing · cluster caps · cash parking · valuation scoring. **All dead or shelved.**
⚠️ Most were judged on 3-4 starts and/or poisoned universes — see "READ FIRST" above.

---

# BLOCKED, not dead

| idea | blocked on |
|---|---|
| Wider selection pool (Russell 3000) | needs a CLEAN R3000 build; `expanded_r3000_universe.pkl` is pre-audit |
| VIX / options-implied term structure | no local options data; FRED fetch fails on SSL |
| TED-spread funding gate | LIBOR discontinued — series **ends 2022-01-21**, cannot ship |
| Off-cadence gate application | `ibkr_engine` has no off-cadence path — and it failed concentration anyway |
| WRDS membership re-download with PERMNO/GVKEY | ~Sept 2026; closes the 9.3% join gap |
