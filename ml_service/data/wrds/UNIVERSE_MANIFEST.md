# Backtest universe pickles — which one to use, and which are poisoned

**Last audited 2026-07-26.** Read this before pointing any backtest at a `*_universe*.pkl`.

| file | period | status | use it? |
|---|---|---|---|
| `complete_sp1500_universe.pkl` | 2016-06 → 2025-12 | ✅ **AUDITED CLEAN** | **YES — the canonical 8yr universe.** All 8yr numbers (+28.3%/0.95 etc.) come from here. |
| `sp1500_universe_2000.pkl` | 2000 → 2025 | ⛔ **POISONED — see below** | **NO** until replaced by the rebuild. |
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
