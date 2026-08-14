# Backtest universe pickles — which one to use, and which are poisoned

**Last audited 2026-07-26.** Read this before pointing any backtest at a `*_universe*.pkl`.

| file | period | status | use it? |
|---|---|---|---|
| `complete_sp1500_universe.pkl` | 2016-06 → 2025-12 | ✅ **AUDITED CLEAN** | **YES — the canonical 8yr universe.** All 8yr numbers (+28.3%/0.95 etc.) come from here. |
| `sp1500_universe_2000.pkl` | 2000 → 2025 | ✅ **REBUILT CLEAN 2026-07-27** | **YES** — 3,642 tickers, coverage 82%(2001)→99%(2025). |
| `sp1500_universe_2000.POISONED_DO_NOT_USE.pkl` | 2000 → 2025 | ⛔ quarantined original | **NEVER** — kept only as evidence. |
| `expanded_sp1500_universe.pkl` | — | 🗑️ orphan (0 code references) | no — deletion candidate |
| `expanded_r3000_universe.pkl` | — | research-only (1 reference) | only `research/alpha_universe_inspect.py` |
| `r2000_universe.pkl` | — | research-only (4 references) | short-strategy research only |

---

## ⛔ Why `sp1500_universe_2000.pkl` (26yr) is poisoned

Two independent build bugs, both in `scripts/build_universe_2000.py`, both **fixed 2026-07-26**
(commit `8c5ac0c` + the survivorship fix). The pickle on disk still predates the fixes.

**Bug 1 — LOOK-AHEAD membership.** The builder treated every row of the WRDS membership files
as proof of membership, but each row carries an `Index Constituent` flag of 1 (in the index)
or 0 (not). Including the 0-rows gave, at 2001-01-02:
`SP500 866 / SP400 1136 / SP600 1508 = ~3510 names` vs the true `500/400/600 = 1500` (**2.3x**).
99.8% of the spurious names had 0-flag rows *before they ever joined*, so the backtest could
hold tomorrow's index members today. **The WRDS data was always correct** — filtering
`Index Constituent == 1` reproduces true index sizes exactly (verified 493/400/596 @2001-01-02,
500/400/599 @2015-06-01).

**Bug 2 — SURVIVORSHIP BIAS.** The price filter matched CRSP tickers against **raw** membership
symbols, but WRDS suffixes delisted/reused tickers (`AAI-199908`, `AAMRQ-201312`, `FRC-200709`)
— **59.1% of symbols (2,536/4,293) carry one**, and CRSP tickers have no suffix, so every one
of those names silently failed to match and was dropped. Suffixed names are precisely the
companies that were **delisted** (Enron, Lehman, AMR, Altaba, First Republic, SIVB). Result:
only 1,743 tradeable tickers, with coverage of index members climbing **36% (2001) → 96% (2025)**
— textbook survivorship bias, inflating every 26yr number. After the fix: **3,548 tickers.**

⇒ **Every 26yr figure in this repo predating 2026-07-26 is suspect**, including the canonical
`+20.7% / 0.77 / −63.8%` re-baseline. 8yr figures are unaffected (different, clean build).

---

## ✅ Why `complete_sp1500_universe.pkl` (8yr) passed audit

Checked exhaustively on 2026-07-26 for the same failure modes plus several more:

- **Membership sizes correct every year:** 503-505 / 398-401 / 600-602 ≈ 1,500-1,505.
- **No suffix pollution:** 0 of its membership symbols carry a date suffix (the 26yr bug is absent).
- **Survivorship — the real test.** Of names that were index members *inside the price window*:
  **2,132 of 2,209 have prices (96.5%)**. The 77 missing are overwhelmingly `…Q` bankruptcy
  tickers (BBBYQ, ASNAQ, AKRXQ, ACETQ…) plus two ticker-format cases (BRK.B, BF.B).
  Residual skew is real but small: miss-rate 9.4% among departed names vs 0.5% among current.
  *Mitigation:* both sleeves require price strength (momentum needs `dist_sma200 > 0`; value
  needs `> −15%`), so bankrupt-bound names are structurally excluded from selection anyway.
- **Delistings ARE in the data:** 1,010 of 3,125 symbols have terminated price series;
  **368 are true delisting-while-member events** — of which **351 are acquisitions** (last price
  ≈ takeout value) and only **4** are bankruptcy-like crashes.
- **Known failures present with correct end dates:** FRC → 2023-04-28, SBNY → 2023-03-10,
  XLNX, ATVI, TWTR, CERN, CTXS. (**SIVB is absent** — a genuine single-name gap.)
- **No look-ahead:** of names joining the index after the price start, only 5/158 have late
  price starts, and all 5 are ticker renames (EG, IQV, DINO, CPAY, DAY) — which makes them
  *untradeable* early, i.e. conservative, not inflating.
- **Delisting mark-to-market is CONSERVATIVE, not inflating.** The backtester values a held
  name with no price at `entry_px`. Measured against the standard alternative (carry forward
  last observed price): entry_px = **+26.55%** vs carry-forward = **+27.21%** — the shipped
  behavior *understates* return by ~0.7pp because acquired names are marked at cost instead of
  the takeout price. Stale marks occur on only **1.07%** of position-days.
  (A "every delisting = total loss" scenario gives +5.37%, but that is not a real scenario —
  95%+ of the delistings here are acquisitions.)

**Verdict: the 8yr universe is not materially inflated.** The one honest caveat is the 77
missing bankruptcy tickers; the strategy's own trend filters make that largely moot.


---

## 🔴 CORRECTED 26yr NUMBERS (2026-07-27) — the old ones were inflated by ~half

Re-ran the canonical config on the clean universe (2-start, 1.49x, integer shares, $50k,
6.3% financing, vol-scaling, `vol_scale_cap=1.0`):

| variant | POISONED universe | **CLEAN universe** |
|---|---|---|
| baseline | +20.8% / 0.78 / −63.5% | **+9.9% / 0.47 / −59.9%** |
| + credit gate p95 | +21.1% / 0.79 / −56.5% | **+10.3% / 0.49 / −54.8%** |

**The 26yr CAGR was overstated by 10.9pp and Sharpe by 0.31** — look-ahead membership plus
survivorship bias accounted for roughly half the reported through-cycle return.
**Do not quote the old 26yr figures anywhere.**

Two things survive the correction:
- **The credit gate still works** on clean data: +0.4pp CAGR, +0.02 Sharpe, **+5.1pp MaxDD**
  (−59.9% → −54.8%). The deployed feature is validated, not rescued.
- **The 8yr figures are unaffected** (+28.3% / 0.95) — different, audited-clean universe.

Honest read: through a full cycle incl. dot-com + GFC the strategy does **~10% CAGR at 1.49x,
Sharpe ~0.47**. The 2018-25 window was a momentum-friendly bull market; treat 26yr as the
regime-neutral expectation and 8yr as the favourable-regime one.

### Residual known gaps in the clean 26yr universe (documented, not fixed)
- ~176 membership names still unmatched to CRSP prices (4.3%): failures whose bankruptcy
  ticker has a different root (ENRNQ→ENE, LEHMQ→LEH, WAMUQ→WM, AAMRQ→AMR) and share-class /
  foreign formats (BRK.B, BF.B). Needs a CUSIP/PERMNO join rather than ticker matching.
- Ticker REUSE across eras can splice two companies into one series (pre-existing).


---

# ✅ FINAL — FULLY AUDITED UNIVERSES (2026-07-28)

Both canonical universes rebuilt by the SAME audited builder (`scripts/build_universe_2000.py`,
date range via `BUILD_UNIVERSE_START/END`) after **8 defects** were found and fixed.

## The 8 defects (each was found only after fixing the previous one)
1. **Look-ahead membership** — `Index Constituent` flag ignored → 2.3x too many names (3,510 vs 1,500).
2. **Survivorship via suffixed tickers** — price filter matched raw `AAI-199908` forms → every
   delisted name dropped (59.1% of symbols carry a suffix).
3. **Bankruptcy Q-ticker, price side** — membership `SIVBQ` vs CRSP `SIVB`.
4. **Bankruptcy Q-ticker, membership side** — prices added but `get_sp1500()` still returned the
   unreachable Q-ticker (91 names had prices the strategy could never see).
5. **Ticker-keyed prices** — renames lost pre-rename history; **940/3,642 (25.8%) series spliced
   two different companies** (AA = Alcoa→Arconic). Now PERMNO-keyed, 1:1, no splicing.
6. **Delisting losses deleted** — CRSP leaves `DlyRet` NaN on the final row and the loop treated
   it as a 0% return. SVB's $39.37→$0.40 vanished; a held bankruptcy realised −85% not −99.9%.
7. **Fundamentals ticker-joined** — coverage ramped 66%(2001)→88%(2025); the VALUE sleeve (35% of
   book) needs roe+gross_margin, so early value picks came from a survivor-skewed subset.
   Now joined via `LPERMNO` (100% populated in compustat_quarterly).
8. **Zero prices** (introduced by fix #6, caught in validation) — CRSP writes `DlyPrc = 0` for
   "no valid price"; anchoring back-adjustment on a 0 zeroed whole columns (666,293 cells).
   Now zeros are treated as missing and the chain anchors on the last VALID positive price.

## Measured improvement
| metric | before audit | after |
|---|---|---|
| 26yr coverage @2001 | 35% | **83.6%** |
| 26yr coverage @2010 | 53% | **94.4%** |
| 8yr coverage @2016 | 88.5% | **94.8%** |
| 26yr fundamentals roe @2001 | 66.2% | **95.8%** |
| delisting capture (SIVB/BBBY/AKRX/FRC) | −85% / dropped | **−100%** |
| spliced series | 940 | **0** |
| zero-price cells | 666,293 | **0** |

## 🔴 CORRECTED CANONICAL NUMBERS (live-mirror: 1.49x, integer shares, $50k, 6.3% financing,
## vol-scaling, vol_scale_cap=1.0; 3-start 8yr / 2-start 26yr)

| | pre-audit | **AUDITED** |
|---|---|---|
| **8yr 2018-25 baseline** | +28.6% / 0.97 / −33.2% | **+25.1% / 0.87 / −39.5%** |
| 8yr + credit gate | — | +24.5% / 0.86 / −39.1% |
| **26yr 2001-25 baseline** | +20.8% / 0.78 / −63.5% | **+10.4% / 0.48 / −65.8%** |
| **26yr + credit gate** | +21.1% / 0.79 / −56.5% | **+10.9% / 0.50 / −58.1%** |

**Both horizons were inflated.** 8yr CAGR overstated by 3.5pp and drawdown understated by 6.3pp;
26yr CAGR overstated by 10.4pp and Sharpe by 0.30. Do not quote any pre-2026-07-28 figure.

**The credit gate still earns its place on clean data (26yr): +0.5pp CAGR, +0.02 Sharpe, and
+7.7pp of drawdown protection (−65.8% → −58.1%).** On the 8yr it is ~free (−0.6pp CAGR, +0.4pp DD)
because that window contains no credit crisis — exactly as originally documented.

---

# 🔴 CANONICAL NUMBERS REVISED AGAIN (2026-08-11) — gross_margin bound

The `[-1,1]` gross-margin bound (commit `74d57cb`) changes both horizons. Measured with
`research/threadGM_gross_margin_ab.py` (live-mirror 1.49x, integer shares, $50k, 6.3%
financing, `clear_deployed()`, 3-start 8yr / 2-start 26yr):

| | pre-bound | **POST-BOUND (current)** |
|---|---|---|
| **8yr 2018-25** | +25.08% / 0.87 / −39.48% | **+23.58% / 0.84 / −38.23%** |
| **26yr 2001-25** | +10.41% / 0.48 / −65.80% | **+11.29% / 0.51 / −64.75%** |

**Harness validated:** the pre-bound arm reproduces the previously documented canonical
numbers *exactly* (+25.1/0.87/−39.5 and +10.4/0.48/−65.8), so the deltas are real.

**The 8yr −1.50pp is NOISE, not lost alpha** — established by decomposition + census:
- `threadGM2` splits it: value-bound −0.91pp (8yr) / −0.32pp (26yr); lowvol-bound −0.70pp
  (8yr) / **+1.19pp** (26yr). Additivity holds (−1.61 vs −1.50; +0.87 vs +0.88).
- `threadGM3` census: the **entire** 8yr value cost is **4 picks across 101 rebalances from
  2 distinct names (AMP, HBAN)**. On the 26yr sample (n=11) the effect **reverses** to
  −0.85pp. Every affected name is a **financial** (AMP/HBAN/ZION/JPM), where
  `(saleq-cogsq)/saleq` is undefined by construction.
- The lowvol half carries **all** the drawdown gain (+1.16pp 8yr, +1.05pp 26yr).
- Worst gm in either universe is ~12; **VIR live was 4,561.72 — 380x more extreme**, so the
  backtest never experienced the live failure mode at all.

⇒ Keep the bound (it removes definitionally-impossible values regardless of performance).

## ⚠️ ERROR BARS (2026-08-14, `research/threadCANON_confidence.py`) — READ BEFORE QUOTING

Those figures came from **3 and 2 starts**. Measured over **12 monthly starts** per horizon:

| | quoted | mean | σ (per start) | range | 95% CI on mean |
|---|---|---|---|---|---|
| 8yr CAGR | +23.58% | **+22.75%** | **7.07pp** | +15.03% … +34.55% | **±4.00pp** |
| 8yr Sharpe | 0.84 | 0.793 | 0.164 | 0.61 … 1.10 | ±0.093 |
| 26yr CAGR | +11.29% | **+11.26%** | 2.75pp | +8.78% … +16.71% | **±1.56pp** |
| 26yr Sharpe | 0.51 | 0.509 | 0.081 | 0.44 … 0.66 | ±0.046 |

The quoted numbers are **not biased** — they sit on the 12-start means. But the 8yr CAGR spans
**19.5pp on entry month alone**. Quote as **8yr +22.75% ±4.00pp** and **26yr +11.26% ±1.56pp**;
two decimals off a handful of starts implies precision that does not exist. Prefer the 26yr
(≈2.5× tighter) when the horizons disagree.

**Consequence for A/B work:** a 3-start comparison cannot distinguish ~2pp of edge from calendar
luck. Sign-consistency across ≥12 starts is the only test with power — that is what killed the
value-weight finding (6/12) and validated the leverage one (23/24).

---

## Remaining known gap (one item)
399 of 4,305 membership symbols (9.3%) never resolve to a CRSP ticker — mostly companies WRDS
identifies ONLY by a post-bankruptcy ticker that never existed in CRSP (`AAMRQ` vs CRSP `AMR`),
plus share-class dots (`BRK.B`) and ADRs. A name-based bridge was TESTED AND REJECTED: it matched
LEHMQ→Lehigh Valley RR and WAMUQ→Wampler Longacre — wrong 3 of 4 times, and wrong prices are worse
than missing ones. **Clean fix: re-download the S&P membership files from WRDS with PERMNO/GVKEY
included** (next quarterly upload, ~Sept). That would make the join exact.
