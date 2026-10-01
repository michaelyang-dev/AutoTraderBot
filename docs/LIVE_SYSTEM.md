# Live System — Source of Truth (verified 2026-07-10, full parity audit)

Read this before analyzing the live engines. Canonical machine-readable config:
`ml_service/live_config.py`. Canonical expectation numbers: bottom of this file.

## Architecture

```
signal server :5001 (signal_server.py + signal_builder.py, Polygon prices, WRDS Compustat/IBES fundamentals w/ FMP fallback)
        │  same signals to both
        ├── IBKR LIVE engine    (ml_service/ibkr_engine.py, real $, acct U25698604)
        └── ALPACA PAPER mirror (server/tradingEngine.js, ~$1.4M paper)
backtest = ml_service/main_production_backtest.py (WRDS data), SHARES the sleeve code
           (strategies/multi_strategy_engine.py) with the signal server.
```

## The strategy (v12) — every parameter, all three systems

| Parameter | IBKR live | Alpaca paper | Backtest equivalent |
|---|---|---|---|
| Signals | `:5001` /signals | same | same shared sleeve code |
| Sleeves | **mom .70 / val .21 / lowvol .09 (2026-09-08, was .50/.35/.15)**; bear .1111/.3333/.5556, **sector 0.00**, breadth-blended | same | same (`PROD_WEIGHTS_*`) |
| Risk-parity in sleeves | **NO** (signal_builder skips it) | same | model with `use_rp=False` |
| Rebalance | **TRANCHED 2026-09-08: 4 virtual sub-books of NAV/4, ONE rebuilt every 5 trading days** (`TRANCHES=4`, `TRANCHE_STRIDE=5`; state `ibkr_tranche_state.json`; per-book trailing peaks keyed `book:SYM`); legacy 20-day single book = `IBKR_TRANCHES=1` | still 20-day single book (paper mirror); **since 2026-10-01 every rebalance resizes held targets BOTH ways (0.3%-of-NAV band, `server/rebalanceResize.js`)** — before, held names were never topped up | clean-room `tranches=4, tranche_stride=5` (`research/VERIFY2_cleanroom.py`) |
| Position cap | 15% of NAV (≈10% of the 1.49x book) | same | model with `cap: 0.10` |
| Trailing stop | 40% from peak | same | same |
| Take-profit | none | none (deleted 2026-06-25) | none |
| Sizing | **closed-loop** (2026-07-04): `_calibrate_quantities` targets 1.49x × vol_scale of MEASURED gross; old `LEVERAGE=1.8` is only a safety ceiling | `MAX_CASH_DEPLOY_PCT=1.49` × volScale — **FIXED 2026-07-12 from 1.60** (it was deploying ~1.60x: live gross/equity 1.55x with top pos 13.8%, 15% cap not binding, so the "caps→1.49" premise was false; fractional shares mean the constant IS the deployed leverage) | overlay leverage on 1x returns |
| Vol-scaling | target 0.15 1x-equiv (0.2235 on levered NAV), lookback 40d, floor 0.30, **de-risk-only cap 1.0** | same policy | `vol_scaling` flags (+ `vol_scale_cap: 1.0` via research fork) |
| **Credit de-risk gate (LIVE 2026-07-18; depth 0.50 → 0.00 on 2026-09-08)** | **`DERISK = 0.0`: the sub-book being rebuilt goes FLAT** while HY-OAS ≥ p95 (one book per 5 sessions, so the de-risk is gradual by construction; the 0.5 depth was chosen for the single book where 0.0 would have been a one-day liquidation). Was: halve gross-leverage target while HY-OAS ≥ p95 of its expanding history (`credit_gate.py`; FRED BAMLH0A0HYM2; cron 8:35 refreshes `data/credit_signal_live.parquet`; engine reads file only, FAIL-SAFE 1.0; applies at rebalance + /deploy, exactly the validated cadence; Telegram on gate-ON + on stale feed) | not ported (paper mirror) | `livemirror_backtest` gate_cols=[hy_oas] p95×0.5 — 26yr MaxDD −63.5→−56.6 at +0.3pp CAGR; OOS +16.6pp DD (`research/THREAD_T_FINDINGS.md`) |
| Financing | IBKR margin ~6.3%/yr on the borrowed portion | paper (model the same) | overlay (`research/leverage_financing_test.py`) |
| Shorts / GLD / VIXM / SPY-parking / trend bucket | all OFF | all OFF (order-path inventory closed 2026-07-10) | all OFF (defaults 0) |
| Signal outage | skip rebalance, HOLD | HOLD (consensus fallback **neutered 2026-07-10**, alert-only) | n/a (always has signals) |

## Verified divergence ledger (2026-07-10 audit)

**Measured / by-design (no action needed):**
- **Data feed** — WRDS CRSP (backtest) vs Polygon (live signals). Measured 2026-07-10:
  daily-return corr **0.9998** median across 1,517 names; 6-1 momentum rank Spearman
  **0.996** with **top-25 picks 25/25 identical**; low-vol Spearman 0.995. Price-side
  data difference ≈ nil for ranking.
- **Execution timing** — live fills near the open, backtest at close (~7bps/side, inside
  the conservative 10bps/side cost model; real measured costs run BELOW the model).
- **Integer shares** at small NAV — compensated by closed-loop sizing.
- **Fundamentals**: SAME vendor both sides — live signal_builder loads WRDS Compustat +
  WRDS IBES as PRIMARY (FMP is only the fallback if those parquets fail). The remaining
  difference is STALENESS: live WRDS files refresh quarterly (manual download), so live
  fundamentals lag up to ~1 quarter vs the backtest's point-in-time data. Earnings-boost
  multipliers (_revenue_surprise/_beat_streak) are OFF on both sides; the eps_surprise_last
  feature is ON on both sides from the same IBES source.
- **No margin-call mechanics** in the levered backtest DD figures.

**Deliberate policy remainder:**
- **Alpaca vol-scaling was INERT until 2026-07-11** (sampled 60-second returns as daily —
  scale pinned at 1.0). Fixed: daily-close equity from Alpaca portfolio history, recomputed
  once per ET day, target 0.2235 matching IBKR. First live use: 2026-07-14 rebalance.
- **Alpaca-only circuit breakers** (daily/weekly/peak-DD halts). IBKR + backtest have
  none. Keep-or-remove is an open decision.
- JS-only latent paths (earnings-eve exit, legacy-bucket exits): dormant — they only touch
  positions NOT tagged `ml`, and the whole book is ml-tagged. The ml cooldown can delay a
  rebalance re-buy ~30min (self-heals same day).

## EDGAR freshness overlay — ENABLED LIVE 2026-07-13 (roe-only)
- `scripts/edgar_fundamentals_patch.py` (daily cron 18:40): per-feature proof gate vs
  Compustat -> `data/edgar_feature_overlay.json`. Coverage: roe **96%** (raised from ~93%;
  see roe-coverage note below) / gm 23% / d2e 10% / eps_surprise 100% (FMP events, 91.5%
  sign agreement vs IBES on 57,864 events).
- **roe coverage 95.4%→96.5% (2026-07-15):** appended niq/seqq specs for minority-interest
  firms (parent niq = `ProfitLoss`−NCI; parent seqq = incl-NCI equity−`MinorityInterest`)
  + preferred-heavy filers (`NetIncomeLossAvailableToCommonStockholdersDiluted`). Appended
  after the base specs → cannot alter any currently-passing symbol; gate still arbitrates.
  Calibration (5,948 events): +1.1pp roe coverage, accuracy flat at 98.8%. validated 1430→1442.
- **Reconciliation monitor** `scripts/edgar_overlay_reconcile.py` (cron 18:55, after the
  patch): (1) WALK-FORWARD — replays extraction over the Compustat history we already own,
  scores vs truth (2026-07-15: roe **98.57%** over 8,419 events); (2) GO-FORWARD — archives
  each overlay to `data/edgar_overlay_archive/` and, when a later WRDS upload reveals a
  patched quarter's truth, diffs served-vs-truth into `data/edgar_reconcile_ledger.json`.
  Telegram-alerts if realized roe accuracy < 97%. Report: `data/edgar_reconcile_report.json`.
  Third-vendor spot-check (`research/edgar_fmp_crosscheck.py`, 2026-07-15): the live roe
  patches agree with FMP within 10% on **98.4%** of 249 names (disagreers = ROE denominator
  instability for near-zero-equity firms, not extraction error).
- **LIVE build now `build_signals_v9(..., edgar_overlay=True)`** — the engines trade the
  overlay (fresh ROE) signals. First effect: 7/14 rebalance. A/B validated +0.5pp CAGR
  (roe-only, PIT-honest). The no-overlay build is now a monitoring BASELINE (served at
  `/signals?edgar=1`; diff logged "OVERLAY LIVE: N BUY vs baseline M"). At flip: 244 roe
  patched, 2 of 24 names changed (DOCU/SFM in, LIF/SABR out).
- **Patcher health**: `data_freshness_check._check_edgar_overlay` (in /data + cron
  9am/1pm/**19:30** weekdays) alerts on Telegram if the daily patcher fails silently
  (overlay stale vs expected run / roe-patched collapse / log traceback).
- eps-surprise boost REMOVED from live 2026-07-12 (divergence #61: live-only unvalidated
  feature; PIT A/B showed -0.5pp CAGR). Overlay live-set = roe only (gm/d2e unharvestable).
- Vintage guards: Compustat-datadate guard (3 ratios) + IBES-vintage guard (eps) mean a
  fresh WRDS upload automatically retires stale overlay entries — WRDS is now the
  quarterly CALIBRATION ANCHOR, not a live dependency.
- **TRUE A/B verdict (2026-07-12, `research/assistant_ab_test.py` v3, PIT-honest):**
  upload-world (live today) 18.75%/0.84 vs upload+assistant (live after flip)
  **19.25%/0.855 — +0.5pp CAGR, same-signed both starts, no DD cost**; ideal-fresh
  ceiling 23.2%/0.95. Summer-gap arms: the missing Sep upload alone costs ~0-0.2pp CAGR
  but deepens MaxDD (assistant trims ~0.8pp of it). Diagnostic: gm/d2e staleness costs
  ~2.4pp under the real upload-freeze model (OVERTURNS the old "gm/d2e harmless" 63td
  result) — unharvestable today (no 99%-certifiable source; XBRL + LLM both failed).
  PIT trap for future research: historical EDGAR reconstruction MUST use earliest-filed
  instances (`spec_value(..., pit=True)`) — latest-filed serves restated comparatives
  with ~16-month availability lag and fakes a negative A/B (v1 bug, fixed).
  **B++ outcome test (v4): real gm/d2e extractions at achievable accuracy (92-96%,
  334/50 symbols) = same CAGR as roe-only but MaxDD -37.6% vs -30.4% — the 99% bar is
  PROVEN by outcomes, not assumed. gm/d2e stay excluded.**

## Capital flows (deposits/withdrawals) — 2026-07-12
- `data/ibkr_capital_flows.json` ledger; **/deposit <amt> [date]** (owner-only) records
  manually; **EOD auto-detect** reconciles cash vs yesterday's close snapshot with every
  fill accounted (reqExecutions, restart-proof) and auto-records any unexplained
  residue >= max($400, 1.2% NAV) with a Telegram announcement (+undo hint).
- Consumers: vol-scaling returns are FLOW-ADJUSTED (a $10K deposit on $33K NAV would
  otherwise read +30% day and floor the scale for 40d), /pnl Today excludes flows and
  shows them on a 💵 row, All-time P&L measures vs initial+flows.
- Deploy policy: new cash deploys automatically at the next scheduled rebalance
  (closed-loop sizing reads live NAV). No mid-cycle AUTO-buys — but **/deploy**
  (owner-only, preview then `/deploy go`, market-hours + healthy-uplink rails,
  10-min plan expiry) tops up CURRENT holdings pro-rata to the SAME
  EFFECTIVE_LEVERAGE x vol_scale x NAV target the rebalance uses: same names, same
  relative weights (caps respected by construction), deploy-only (never sells),
  rebalance schedule untouched. Validated on the live book 2026-07-12: deficit
  $1,949 -> plan $1,935 (99%), projected exactly 1.490x, max position 14.6% < 15%.
- Snapshot seed guard: startup seeds the close snapshot ONLY on a trading day after
  16:00 ET — a Sunday restart once replaced Friday's close with deposit-inflated
  values, which would have broken detection and faked a +7% return.
- First real flow: $2,458 credited ~2026-07-12 (IBKR partial fill of a $20K request),
  measured via cash reconciliation, ledgered. Uplink honesty (same session): Error
  1100/1102 now push Telegram alerts; connectivity probe requires a positions
  round-trip (reqCurrentTime is answered locally by the Gateway and can lie); all
  panels carry a red banner while the uplink is down.

## Canonical expectation numbers (2026-07-10 re-baseline)

`research/final_live_config_test.py` — exact live config, levered 1.49x, financed 6.3%:

| Period | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| 2018–2025 (3-start avg) | **+28.3%** | **0.95** | **-33.9%** |
| 2001–2025 (2-start avg) | **+20.7%** | 0.77 | **-63.8%** |

The old **+25.4%** / **+21.0%** headlines were the legacy config (RP-on, cap .15, no
vol-scaling, 1x, no financing) — superseded for live expectations. The two config
differences (no-RP, NAV-cap) were audited AND backtest BETTER than the legacy config —
do not "fix" live to match old research; new research must import
`ml_service/live_config.py`.

Vol-scaling was validated best-of-5 policies at 1x AND at leverage; up-scaling variants
(e.g. t.20/cap1.5) LOSE at leverage — financing cost + Reg-T clamping + variance drag
(`research/volpolicy_levered_test.py`). Do not re-tune without new evidence.

## 🔴 dist_sma200 coverage collapse — INCIDENT 2026-08-18, FIXED + DEPLOYED 2026-08-19

**Symptom.** Telegram COVERAGE ALARM: `dist_sma200 30.6% < floor 85%`. The live book ran from
~16:11 to ~18:28 on 2026-08-18 with **1,044 of 1,504 SP1500 names invisible to the momentum and
lowvol sleeves**, while every upstream check reported success (`Massive: 1539/1539 symbols
loaded`, `Data quality gate PASSED`).

**Why nothing caught it.** Coverage is not a *fetch* property, it is a *depth* property, and
nothing measured depth. `dist_sma200` is the only feature needing 200 CONTIGUOUS bars —
`c.rolling(200).mean()` uses pandas' default `min_periods=200`, so a short symbol yields NaN,
`get_feature_map` drops NaN keys, `dist_sma200.get(sym, 0)` returns 0, fails `> 0`, and the name
vanishes from the sleeve **with no log and no error**. `vol_60d` (60 bars) and `ret_126d` (touches
only 2 rows) are untouched by the same truncation — hence the diagnostic signature
**99.4% / 99.4% / 30.6%**. The per-symbol disk cache in `massive_data_provider` is reused for 18h
with no check on how much history each file holds, so a truncated cache serves the whole window.

**Trading impact: NONE.** No orders in the window (engine logs show only IBKR connectivity);
`ibkr_rebal_state.json` showed 5/20 trading days since the 2026-08-11 rebalance.

**Fix (commit 10046bc, data layer, parity-neutral).** `fetch_bars_batch` now measures cache depth
on load; if the **median** falls below 80% of expected sessions it discards the cache and refetches
everything. Set-level by design — genuine short histories are normal and permanent (ADIG 10 bars,
HONA 35, MFP 30) and per-symbol refetching would burn API budget forever and never succeed. Also
logs the depth distribution every fetch, and reports when a refetch is STILL short (= the vendor is
the problem, not the cache).

**Explicitly NOT fixed by relaxing `min_periods`.** The backtest universe builder uses the
identical `rolling(200).mean()` (`scripts/build_universe_2000.py:262`). Loosening it live would
silently change which names the live book selects relative to every validated backtest number.
The formula is right; the input was short. **Fix the input, never the filter.**

**Verified live 2026-08-19 → 08-22:** new telemetry present
(`cache depth: median 379 bars (expected ~379)`), depth alarm never fired, coverage restored to
**98.1–99.1%**, no COVERAGE ALARM since. Unit test: `ml_service/tests/test_cache_depth_guard.py`
(6 assertions incl. the negative cases — healthy cache not churned, genuine short-history IPOs do
not trigger a mass refetch).

### 🔴 RECURRENCE 2026-09-01 — the depth hypothesis is REFUTED. Root cause STILL UNKNOWN.

`dist_sma200` fell to **19.7%** (1,208 of 1,504 names missing) — worse than the first incident.
**The depth telemetry added on 08-19 disproves the mechanism that fix was built on:** during this
incident cache depth was healthy (`median 378 bars (expected ~379)`, only 7-21 symbols below 200)
while coverage sat at 19.7%. The depth guard correctly never fired, because depth was never the
problem. **Mass cache truncation was a hypothesis, not a finding, and it is now dead.** The depth
guard is retained (it is cheap and its telemetry is what produced this refutation) but it is NOT
the fix for this failure.

**Post-hoc forensics have now failed TWICE** — the per-symbol cache turns over within ~18h and the
condition self-heals before anyone can inspect it. Both times the disk state was already healthy by
the time it was examined.

**The surviving constraint, which any future explanation must satisfy:** `vol_60d` (60 contiguous
bars) and `ret_126d` stayed at **99.4% through BOTH incidents**. So the final ~126 rows are intact
and only the 200-contiguous-bar window breaks. That rules out simple end-staleness — a symbol NaN
on recent rows would take vol_60d down too.

**Response (commit dc6206c), two parts:**
1. **Forensic capture at alarm time** — the guard now writes
   `data/coverage_forensics_<ts>.json` classifying every missing name as STALE / SHORT / GAP plus
   the union-index shape. Whichever bucket dominates names the cause.
2. **Engine coverage gate** — coverage is published in `/health` and `ibkr_engine` now REFUSES to
   rebalance while it is degraded. The engine checked staleness but never correctness; both
   incidents produced perfectly FRESH signals, so it would have rebuilt the book from ~20% of the
   universe and logged nothing unusual. Neither landed on a rebalance day — that was luck, not
   design. Skipping a rebalance costs at most 20 sessions of drift; trading a corrupt book does not
   unwind. Test: `ml_service/tests/test_coverage_gate.py`.

### ✅ ROOT CAUSE FOUND 2026-09-02 — vendor date hole + `rolling(200)`'s NaN sensitivity. FIXED.

**Evidence (direct, not inferred).** 16 cache files written at 18:28 on the bad day survived the
morning refresh (too young). **Every one is missing 2026-08-28 — a real trading day** — while all
40 sampled good files have it (0/40 disagree). FISV additionally lacks 2025-11-12: this is a
recurring vendor property, not a one-off.

**Mechanism.** Polygon transiently omits a real trading date from per-symbol history (backfill
lag). The 09:33 Sep-1 batch (1,204 files) lacked Aug 28; the 09:50 batch (318) had it — and
318/1539 ≈ the 19.7% that survived. Symbols that HAVE the date put it in the union index;
everyone else gets a NaN row from `reindex`, inside the 200-bar window. `rolling(200).mean()`
(default `min_periods`) → NaN → `get_feature_map` drops the key → `dist_sma200.get(sym, 0)`
fails `> 0` → silent exclusion. It heals when the 18h cache rolls and the backfill has landed.

**Why ONLY dist_sma200 broke — the constraint every earlier theory failed:** pandas
`pct_change()` PADS NaNs by default, so `ret_*` and `vol_*` walk straight through the hole and
report 99.4%. Only features computed with `rolling()` on the raw close see it. Depth telemetry
cannot see it either (378 vs 379 bars is within calendar jitter). Row-level sparsity checks
cannot see it (one date at ~80% presence is invisible next to the 96% baseline from stale files).

**Fix (commit below), parity-RESTORING not formula-changing.** In `_compute_features_from_raw`,
after `reindex`, forward-fill INTERIOR gaps only (between first and last real print), limit 3
sessions. CRSP gives the backtest a row for every trading day per security, so its `rolling(200)`
never meets an interior NaN — live must match that INPUT structure. Leading NaNs untouched; a
gap >3 sessions (real halt/delisting) stays NaN and the name stays excluded. `rolling(200).mean()`
and its `min_periods` are deliberately NOT relaxed. Plus: a vendor-hole detector at the data
layer (`fetch_bars_batch`: any date missing from >20% of covering symbols → ERROR) and a
per-build hole report + Telegram in the coverage guard.

**Test:** `ml_service/tests/test_vendor_hole.py` — reproduces the exact incident (a real date
missing from most symbols, present in others): dist_sma200 restored for holed names, 5-day halt
still excluded (limit respected), output bit-identical to raw `rolling(200)` for clean names,
provider scan flags the date. 10/10 locally and on the box.

**What the earlier fixes were:** the 08-19 depth guard addressed a refuted hypothesis (kept: cheap,
and its telemetry is what falsified it). The 09-02 engine coverage gate remains the backstop — if
a hole ever exceeds the fill limit, the engine still refuses to rebalance on a degraded book.

---

## WRDS refresh 2026-09-07 — fundamentals DEPLOYED; daily price panel added; what WRDS can and cannot give us now

**Deployed live (box + local, signal-server restarted):** `compustat_fundamentals_quarterly.parquet`
rebuilt as a HYBRID — fresh Compustat Fundamentals Quarterly (INDL/STD/C/USD, deduped on
gvkey+datadate) for 2015→2026-08-31, back-filled with every row of the previous file it lacked.
558,037 rows, unique on (tic, datadate), columns/dtypes identical to the old file. Verified: 100% of
the old file's SP1500 rows present; through the live loader's own view (latest row per tic by rdq)
**1,964 SP1500 names get a newer quarter, 760 unchanged, 0 older.** rdq through 2026-09-03. Old file
kept as `compustat_fundamentals_quarterly.bak-2026-07-12.parquet` on both machines.
**Verified live 2026-09-07 20:46 ET:** post-restart build logged `Fundamentals loaded from WRDS Compustat (13232 tickers)` (was 13,022); coverage dist_sma200 99.1% / roe 94.5%; no alarms.

**Known gap in the new pull:** it is INDL-format only, so ~25 US financials that file under FS (BK,
BRKL, PINC…) have no fresh rows — they are carried from the old file (stale to 2026-04-30). Fix =
re-pull Fundamentals Quarterly with Industry Format **FS** only (small) and merge.

**New research assets (not used live):**
- `compustat_security_monthly.parquet` — 7.0M rows, every security 1990-01→2026-09, tic/gvkey/iid
  + monthly prices. Resolves 64 of the 102 membership tickers the CRSP-ticker join could not (60 of
  them link to a PERMNO); total membership resolution now ~99%.
- `compustat_security_daily/secd_YYYY.parquet` — 2010-01-04→2026-09-04, USD-priced, all
  domiciles (first pass wrongly dropped non-US-incorporated S&P members like APTV/JCI/NXPI; re-run
  without the `fic` filter). Matches CRSP **exactly** on the 2025 overlap (7/7 names, corr 1.0000,
  raw closes identical). This is the 2026 price source for the universe rebuild, since CRSP cannot
  supply it (below).
- `compustat_fundamentals_quarterly_2026-09_RAW.parquet` — all 679 columns, 1980→2026-08.
- Raw zips archived under `data/wrds/raw_downloads_2026-09-07/` with a README.

**What WRDS can no longer give us (important for anyone planning a re-download):**
- **S&P historical index constituents have been REMOVED from Compustat** (S&P licensing). Only
  "S&P Current" (today's snapshot, gvkey-keyed) remains. Our four `*_membership_history.parquet`
  files (daily, 1990→2026-05-01) are therefore **irreplaceable — never delete or overwrite them.**
  Membership after 2026-05 must be maintained from periodic S&P-Current snapshots.
- **CRSP is annual-update only on this subscription** (quarterly Stock/CCM are greyed out): daily
  prices, delistings, names and the CCM link refresh once a year in February. CRSP data ends
  2025-12-31 until Feb 2027. 2026 prices come from Compustat Security Daily instead.
- Still wanted: **Index Constituents – S&P Current** for 500/400/600 (small), to extend membership
  forward and validate the identifier map.

**2026 price coverage of the SP1500 on the corrected Security Daily panel: 1,484 of 1,485 still-listed
names (99.9%).** The 31 that first looked missing were all explained: **25 acquisitions/delistings in
2026** (HOLX, SEE, CTRA, MASI, AL, APLS… `dlrsn=01`; TMHC last print 2026-07-23, reason pending) and
**5 ticker/name changes** — BK→`BNY`, SATS→`ECHO`, IAC→`PPLI` (People Inc), EXPI→`AGNT`, FDP→`DMC`.
Five ticker changes in four months among 1,510 names: **anything keyed on ticker drifts within
weeks.** CUSIP (8-char) matched all of them on the first try and exists on both the CRSP and
Compustat sides — it is the bridge to use in the universe rebuild (CUSIP→gvkey via Security Monthly,
CUSIP→PERMNO via CRSP security info; CCM link as the cross-check).
**Live is unaffected:** the live universe is the weekly-refreshed `sp1500_members.json`; the served
signal set already carries BNY/ECHO/PPLI/AGNT/DMC and has dropped the old tickers and delisted names.

---

## 🔴 dist_sma200 coverage alarm #3 — 2026-09-08 16:13 ET — ROOT CAUSE: clock-only partial-session guard. FIXED (pending deploy)

**Symptom.** `/health` coverage `dist_sma200 0.2799`, `coverage_ok False`; forensics
`data/coverage_forensics_20260908-161334.json`: union index last row 2026-09-08, pool 1,504,
**1,083 names missing**, sample buckets STALE 399 / SHORT 1 / GAP 0 / ABSENT 0 — i.e. the missing
names simply had no 2026-09-08 bar yet (last bar 09-04; Labor Day 09-07), `nan_in_trailing_200 = 1`.

**Root cause.** `_is_partial_session()` was clock-only: `idx[-1] == today and hour < 16`. At 16:13 the
session had closed, so the guard said "final", but the vendor had published the day's bar for only
421 of 1,504 names. The 1,083 NaNs on the last row made `rolling(200)` NaN and the coverage guard
fired (correctly) — the engine would have refused to trade on it (correctly). Different mechanism
from #1/#2 (vendor date hole inside the window, fixed 09-02 with the interior ffill): this is the
post-close publication lag, which the interior ffill deliberately does not touch (a trailing NaN
must never be forward-filled — that would be a stale price).

**Fix (`signal_builder.py`).** DATA RULE added to the same single-source guard: the last row is
also partial when its print count is below **90%** of the max count over the previous 5 rows
(`_last_row_coverage`, `PARTIAL_ROW_MIN_COVERAGE = 0.90`). Applied at BOTH call sites — the matrix
trim in `_build_universe` and the `today` selection in `build_signals_v9` (via
`_raw_last_row_coverage` on the raw bar dict, same measure) — so they cannot diverge. On an
incomplete row the builder computes on the last completed session, exactly as it does intraday.
Tests: `tests/test_partial_row_guard.py` (10 checks: the 421/1,504 state is partial, 100% is not,
89/91% threshold, clock rule unchanged, raw-dict measure equals the matrix measure).

**Why it was harmless to the book.** The engine only trades at the open, after the 09:15 refresh
rebuilds the universe on complete bars; and `fetch_signals` refuses degraded coverage (2026-09-02
gate). The alarm is the guard working. This fix removes the false alarm and the 15-minute-cadence
rebuilds on a half-published day.

## 🔴 Ticker-reuse SPLICE in live prices (BNY) — FOUND + FIXED + DEPLOYED 2026-09-08 23:10 ET

**Found by** the same-day signal-parity test (backtest sleeves on the CRSP/Compustat universe vs the live
server's BUYs for 2026-09-04): overlap 17/25, and the live momentum sleeve ranked **BNY #2** while the backtest
ranked the real BNY Mellon at +53% (not top-5). The live per-ticker history for "BNY" (`massive_cache/BNY_adj`)
starts 2025-03-07 at **$10.52** — the closed-end muni fund that held the ticker — and jumps to $137 on
2026-05-21, the day BK renamed to BNY: a **+1,263% one-day return** and a fake +1,608% 12-month momentum. The
vendor keys history by ticker; the backtest keys by PERMNO and is immune. Cache scan: 2 splices in 1,539
names (BNY x14, SOLS x487,400); the other large days (MRNA +177% 2026-08-19, APLS +135%, CORT +109%) are
genuine and below the threshold. It would have been the largest buy of the first tranche day.

**Fix (`signal_builder._splice_guard`, `SPLICE_MAX_RATIO = 4.0`):** any one-day close ratio above 4x is two
securities under one ticker; everything before that bar is discarded (the jump bar is the new security's first
close). The name is then too short for the 200/252-bar features until real history accrues — exactly what the
backtest sees for a fresh listing. GME's record +134.8% day (2.35x) is untouched. Tests
`tests/test_splice_guard.py` (7). Deployed 23:10; rebuild: `SPLICE GUARD: 2 symbol(s) truncated ... BNY@2026-05-21
x14, SOLS@2025-10-30 x487400`; BUY list 23 names, BNY gone, coverage 99.1%.

**Parity after the fix:** live vs backtest BUY overlap on 2026-09-04 **21/23** (was 17/25); remaining
differences PAYX/RDDT (live-only) and PTC (backtest-only) are vendor/fundamentals noise. Momentum top-5
identical (SNDK, MU, LITE, WDC, STX).

**Other parity evidence gathered the same evening** (`research/AUDIT_final_parity.py`,
`AUDIT_cost2_final149.py`): one-bar signal delay does not hurt (LIVE +1.1pp, FINAL +0.3pp CAGR → no
look-ahead); live account vs clean-room since the 2026-08-11 rebalance (no deposits in window): live +7.05%
vs backtest +5.46%, daily-return corr 0.72, holdings overlap 21/25; membership file matches public
index-change dates on 12/12 events tested; split/dividend continuity 7/7 vs an independent source; best-5-day
dependence ~25% of 8yr log-return for LIVE, ~22% for FINAL (structure advantage intact with those days
removed); **realistic costs for a $61k / 4-book account are ~2x the modelled 10 bp** (IBKR $1 minimums on
~$600 median orders → ~16 bp + slippage): at 2x costs 8yr FINAL_1.49 +28.4% / 0.933 / −39.0% vs LIVE +26.5% /
0.887 / −39.7% (8 starts); 26yr FINAL_1.49 +15.6% / 0.642 / −49.2% vs LIVE +13.6% / 0.577 / −56.0% (4 starts).
Quote the cost-2x line as the live expectation until NAV grows enough for the $1 minimum to stop binding.

## 🔴 Book-1 day 2026-09-16 — watchdog killed the rebuild mid-way; retry double-exited 4 PAYX + 1 MU (~$1.4k). FIXED, deploy 16:12 ET

**What happened.** 09:32:21 book 1 rebuild started; the rebuild is 4–6 min of blocking awaits (4 s price tick
per name + 3 s per order) with no heartbeat, so at 09:36:02 the HANG watchdog (360 s) force-restarted the
process after 15 exits + 16 buys had filled. The ledger was only written at the END of a rebuild, so the
restarted engine (09:36:03) reloaded the pre-rebuild ledger; `_reconcile_books` charged the already-executed
sells to the LARGEST holders (books 0/2/3) and attempt 2 (09:38:50–09:41:06) then "exited" book 1's phantom
slices again: **SELL 4 PAYX @117.82 and SELL 1 MU @924.24 that belonged to other books**. Also 1 NTNX trim,
+1 MXL, +1 ELF (7 ELF had filled during the restart and were correctly credited to book 1), +5 BX, +2 MRNA.
Net damage ≈ $1.4k of unintended sales (2.3% of NAV); no double buys; ledger == IBKR (39 names) afterwards.
Books 0/2/3 now hold slightly less PAYX/MU than designed until their own rebuild days.

**Fixes (`ibkr_engine.py`, tests 47/47):** ledger + peaks persisted after EVERY fill (a crash leaves an
accurate book; the retry completes only what is missing — tested: crash on the 3rd buy → ledger == broker,
retry sells nothing twice); watchdog heartbeat after every price fetch and every order in the rebuild and in
the tranched stop check. Deployed to the box; engine restart scheduled 16:12 ET (after the close).

## ✅ First tranche day 2026-09-09 — executed as designed (verified 09-10 01:45 ET)

09:31:50 `Tranche day count: 5/5` → transition split of the 25 holdings into 4 books (books 1–3 verified equal
to the exact whole-share 4-way split of the 09-08 positions, untouched) → credit gate off (HY-OAS 2.68,
pctile 2.9%) → vol-scale 0.86 (realized 26.1%) → book-0 target 1.27x → `SIZING book 0: NAV/4=$15,444, mult 1.58
→ projected gross $19,802 = 1.28x`. Orders 09:33:38–09:35:32, all market, all filled, no duplicates, no price
failures: 14 exits of book-0 slices (SEZL, PAYC, STX, APPF, DDOG, DUOL, LITE, PANW, BKNG, META, CRWD, YELP,
RNG — plus STX/LITE, which ARE in the signals but round to 0 shares at $898/$1,018 on a $15.4k book, exactly
the backtest's `int()` truncation), 3 trims (SNDK, MU, WDC), 17 buys (CORT, NTNX, DELL, MXL, OGN, GKOS,
ADSK, FTNT, PAYO, TGTX, ADBE, RDDT, BSY, CARG, NFLX, PTC, WDAY). Book 0 after: 21 names, gross 1.28x of
NAV/4, max weight 11.3% (SNDK), min 3.1%. Books ledger == IBKR positions exactly (34 names). Trailing peaks
all book-keyed (87). Account after: NAV $61,662, gross $57,098 = 0.93x (books 1–3 still legacy-sized), cash
+$4,562. Fills vs the 09:30 open: median −14 bp, mean +16 bp (TGTX bought at the day's high, +286 bp — market
orders in thin names at the open; the cost of the design, not a defect). No stops fired. Overnight 00:14–00:20
the Gateway's nightly restart produced Error 1100/326; the watchdog force-restarted the engine, which
reconnected at 00:20:03 and reloaded `tranche state: counter 0/5, next book 1, initialized=True`. Next: book 1
on 2026-09-16.

**Corrective buy-back 2026-09-16 15:45 ET (owner: "do manual rebalance").** The double exit had taken 4 PAYX and
1 MU out of book 0 (the ledger reconcile removes a deficit from the largest holder; book 0 held 5 PAYX / 1 MU).
Book 1 itself needed no correction: attempt 2 built it from the 09:38 signals at the 1.34x book target (mult
1.64, projected gross $20,141 = 1.34x of NAV/4 $15,006). Fix executed with the engine STOPPED (so the ledger edit
could not be overwritten by a stop-check save): separate API client (clientId 7) bought 4 PAYX @ 116.55 and 1 MU
@ 923.92 (marketable limits, outsideRth), then `data/ibkr_tranche_state.json` book 0 credited PAYX 1→5, MU 0→1,
engine started 15:45:51 and reloaded `counter 0/5, next book 2` — no reconcile deficit/surplus logged afterwards
(ledger == IBKR: 39 names, PAYX 13 = 5+0+4+4, MU 3 = 1+0+1+1). Side effect: cash $351 → −$1,040 (the $1.4k the
phantom exits had freed was already spent by book 1's build); gross ≈ 1.02x NAV, inside the 1.80 ceiling; the
next rebuilds (book 2 on 09-23, book 3 on 09-30) resize normally. Engine down 15:45:38–15:45:51 only.

## Alpaca paper mirror: held names were never topped up — FIXED 2026-10-01 (deployed after the close)
The owner asked why LITE sits in Alpaca but not IBKR. Same `:5001` signals, different book mechanics: LITE was a BUY at
every IBKR rebuild (09-09, 09-23, 09-30; rank 21/23 on 10-01) but rounds to 0 shares in a NAV/4 book under bear weights
(book-3 target ≈$567 = 0.58 share; IBKR sold its last legacy share 09-30 @ $982.97). The backtest truncates the same way
(`int()`), so this is expected behavior, not a bug; `research/EXP063_account_rounding.py` tests rounding at the account
level instead. Alpaca bought LITE before 07-15 (avg $809.78) and kept it. Returns to 09-30: from 07-20 IBKR +16.9% /
Alpaca +5.8% / SPY +2.8%; from 09-08 +0.5% / −0.2% / −0.4%.

**Defect found on the way:** Alpaca was **0.83x invested with $246k cash** (NAV $1.48M, target 1.49x; vol-scale 1.0, no
breaker). STEP 2 skips held symbols ("Already held — skip") and STEP 1g only trimmed (> 20% over target and > $5k), so at
every rebalance held names stayed under target and the cash the new names did not absorb sat idle from 09-10 on. The
backtest and IBKR resize every held target in both directions unless |Δ| < 0.3% of the book.
**Fix:** STEP 1g resizes held targets both ways with the backtest's 0.3% band — once per rebalance episode
(`rebalState.resized_for`), top-ups only outside the close buffer, never past 1.49x × volScale, always leaving room for
the new names' targets. Sizing is the pure `server/rebalanceResize.js` (`planRebalanceResizes`), unit-tested by
`node server/tests/rebalance_resize.test.js` (9 tests incl. 5,000 randomized portfolios).
**Dry run on the live paper account (read-only, 17:30 ET):** next rebalance under the OLD rule → 0.92x (trims $69k, no
top-ups); under the NEW rule → **1.49x** (1 trim: SNDK $119k → $49k; 11 top-ups = $838k: MXL $31k → $201k, MU $24k →
$190k, NTNX $21k → $170k, …), with 11 exits ($675k) and 11 new names ($879k of targets). Next Alpaca rebalance ≈ 10-08
(15/20 trading days on 10-01): the idle cash is deployed then, not off-schedule.
**Still different BY DESIGN:** one 20-day book (vs 4 tranche books); whole shares on a ~$1.5M account (rounding negligible
there, material in IBKR's ~$15k books); intraday stops (IBKR at the close since 09-30; EXP-062: intraday −4.88pp CAGR on
the 8yr). Alpaca-vs-IBKR therefore measures tranching + small-account rounding + stop timing, not two strategies.
**Deployed 2026-10-01 17:53 ET:** box `git pull --ff-only` bundle 060a6c9..567174c (no tracked changes, no pm2 watch);
`node --check` + the 9 tests pass on the box; `pm2 restart trading-engine` ONLY (IBKR engine and signal server untouched —
no Python changed). Clean start: Alpaca connected, cash $246,116.80, NAV $1,483,269, 23 positions rehydrated (all `ml`),
stderr empty; `js_rebal_state.json` unchanged (15/20, last 09-10) — first resize at the next Alpaca rebalance (≈10-08).

## After-close deploy 2026-09-30 16:12-16:14 ET — engine on eb2741c
Box pulled 5af1e4e..eb2741c; all 19 test files pass on the box. Engine restarted 16:14:05 (after the 16:05 close
mark, NAV $61,856): tranche state counter 0/5, next book 0; connected; Telegram bot up; vol-scale 0.98 on the 40
sessions ending at today's close (completed-sessions window, eea223f). **Now live in the engine:** trailing stops
evaluated once in the last 10 minutes before the close with the backtest rule (EXP-062, first evaluation 2026-10-01
15:50; rollback IBKR_STOP_AT_CLOSE=0) and stock-split handling (first calendar read at 2026-10-01's first check).
**17:50 signal-server restart on eb2741c — verified 17:56:** the settle rule refetched all 1,541 files written before
the 17:00 settle (2 min), quality gate latest = 2026-09-30, so the EVENING build is on TODAY's close (the original
stale-evening bug is gone); 10 short-history names now in the feature set (ADIG, BNY, FDXF, HONA, MBGL, MFP, Q, SOLS,
VGNT, VSNT); coverage dist_sma200 99.5% / roe 95.4% / gm 99.5% / vol_60d 99.9%; 23 BUY; member-only breadth 19.1%
(blend 0); healthy, not stale.
**18:23 refresh job:** first correctly labelled archive file `closes_2026-09-30` (1,541 final closes); data_gaps patched
16/25 short names with history through today, but logged FAILED — check_data_gaps had no return and the forked runner
(0155bab) reads None as 'returned False', so it had failed every night since 09-30 00:40 (housekeeping, never paged;
the status line read PARTIAL). Fixed 060a6c9 (explicit return True, test), on the box 18:50; every STEPS function now
ends in an explicit return.

## Book 3, 2026-09-30 — CLEAN (first rebuild on SSGA membership + final-bar cache)
09:19 pre-open refresh reused the final 09-29 bars (1,541 from cache, 0 vendor calls) · served signals identical to the
verified set (23 BUY, breadth 21.6%, blend 0) · 09:32:32 tranche 5/5 → book 3 · credit gate off (HY-OAS 3.02, pctile
11.8%) · vol-scale 0.98 → target 1.46x (the running engine still included the pre-session NAV seed; 1.41x once the
eea223f engine restarts) · SIZING mult 1.68 → projected 1.51x (whole-share lumpiness, MU ~$1,074) · 9 exits (SEZL 4,
PAYC 2, APPF 2, SNDK 1, LITE 1, ADSK 2, YELP 20, WDC 2, RNG 4) and 21 buys all filled 09:34:23-09:36:06 (MXL 21 @93.82,
MU 1 @1,074.18, NTNX 24, CORT 12, OGN 111, AMD 2, FTNT 5, CRWD 4, PANW 2, ATRC 19, ADBE 2, DOCS 35, RDDT 6, CARG 28,
BSY 12, PAYX 4, DUOL 1, BX 7, WDAY 4, AXTI 7, MRNA 3) · overlay: no trims. 09:52: ledger == IBKR, 0 open orders,
NAV $62,065, gross $86,882 = 1.40x, 33 positions, AvailableFunds $39,256. Next: book 0 on 2026-10-07.
False alarm the same night: a 03:12 OFFLINE test (temp builder copy with no data folder, Telegram vars exported) sent
"COVERAGE ALARM: roe 0.0%" to the owner; the live server never alarmed. Offline runs now strip the Telegram vars.

## Live-vs-backtest parity audit 2026-09-30 02:00-03:30 ET — 1 live-only failure mode, 2 parity gaps fixed (held for after the close), 4 measured differences
Compared the live path (signal_builder + ibkr_engine) with the validated backtest (build_universe_v2 universe,
main_production_backtest / VERIFY2 clean room) on the validated 8yr universe and on today's data.
**Identical:** feature formulas (same windows, pandas std, SMAs); sleeve code (shared); the probability transform
(proportional, top = 0.95, the engine renormalises); the 0.5% combined-weight floor; closed-loop integer sizing per
book with the 15% cap; the 0.3% min-trade band; the tranche schedule; SI / earnings / price-target / EPS inputs
(zero in both); fundamentals coverage gaps (secondary share classes, negative equity — same in both); the UMD crash
detector (live uses today's members over its history: corr 0.998, max |diff| 0.008, crash flags agree on 82/82 days,
2026-05-11..09-04).
**Found and fixed (commit a2967dc, deploy after the close):**
- **Stock splits (live-only failure).** The engine never adjusted the ledger or the stop peaks: a 2:1 split reads as
  −50% vs the pre-split peak → the 40% stop fires in EVERY book holding the name at the next open, and the extra
  shares were credited to one book. No held/buy name has a split on the calendar now (CRWD 4:1 on 2026-07-02 predates
  both accounts' holdings). Now: the day's splits (Massive/Polygon reference API) scale each book's shares and
  peaks before any reconciliation; calendar down + split-like quantity change → stop and rebuild held, owner alerted.
- **Minimum history 252 → 21 bars** (the backtest keeps a name once ret_20d exists). The validated lowvol sleeve held
  a <252-bar name on 34/437 sampled 2018-26 rebalance dates (7.8%, mean 14% of the sleeve; e.g. SNDK after its 2025
  spin-off); live could never. Today: identical BUY list.
- **Breadth over members only** — live also counted ~22 ETFs (0.4pp today).
**Measured, accepted or owner decisions:**
- The backtest's breadth set also holds ex- and future members (~350 extra names): live-like minus backtest breadth
  mean +0.11pp, p10/p90 −1.5/+1.7pp; the blend differs by >0.1 on 7 of 517 sampled dates, never by >0.2.
- **Stops — THE LARGEST GAP (EXP-062, fix e572e7e, deploy after the close).** Live raised peaks on intraday prices
  and sold on intraday dips; the backtest uses daily closes for both. Same engine + deployed package, 8yr 12 starts:
  live-style stops **−4.88pp CAGR / −0.104 Sharpe, worse on 12/12 starts**, MaxDD −0.57pp, ~60% more stop-outs
  (peaks-from-highs alone −2.89pp; intraday triggers alone −0.38pp). Fix: peaks and stops evaluated once in the last
  10 minutes before the scheduled close (13:00 on half days) with the backtest's rule; IBKR_STOP_AT_CLOSE=0 rolls back.
  26yr (12 starts): −1.56pp CAGR / −0.041 Sharpe / MaxDD −1.01pp, worse on 12/12 for all three. Existing peaks (set from intraday prices) are kept, so held names keep a
  slightly tighter reference until they make a new closing high or are rebought.
- **Execution:** live rebuilds at the open on the prior close's signals; the backtest trades at the same close. Bounded
  by the shift+1 test (−0.35pp CAGR); research/BUGS.md A1 corrected (it claimed live traded near the close).
- **BNY:** the vendor's per-ticker history splices a closed-end fund before 2026-05-21; live truncates it (excluded
  until the 21-bar deploy, then a short-history name). With its true history it would be HOLD today (+34% 12m).
- New positions below 0.3% of a book are bought live but skipped by the clean room — negligible at our sizes.

## Live data verification 2026-09-30 01:00-03:00 ET (before the book-3 rebuild) — 7 defects; 4 fixed before the open, 1 research-only, 2 held for after the close
**Trigger:** owner asked to verify that the data the strategy trades on is correct. Everything below was measured on
fresh 2026-09-29 data (offline rebuilds into temp caches, read-only IBKR/FMP/FRED queries).

**Independent checks that PASSED:**
- **Prices:** IBKR daily TRADES bars vs Massive, 80 names (23 BUY, 36 held, 40 random, SPY): same last session, no date
  missing on either side over 260 sessions, closes to the cent (worst single day in a year 0.155%), ret_20/126/252d,
  dist_sma50/200 within 0.02pp and vol_60d within 0.06pp.
- **Feature code:** builder feature maps vs an independent pandas derivation from the same bars: max |diff| 0.000000.
- **Fundamentals:** vs FMP statements for the same fiscal quarter (30 names): net income and equity identical; ROE
  identical except VSXY (defect 4); GM / D/E differ by vendor definition (Compustat COGS excludes D&A) — Compustat is the
  backtest's source. EDGAR overlay patches ROE for 21 late filers, none a BUY.
- **Membership:** our S&P 500 list == FMP's 503 constituents exactly; all 23 BUYs are current S&P 1500 members.
  (`sp500_history.get_sp500_on_date` is 13 names stale, but the live path overrides it with the S&P 1500 list.)
- **Credit gate:** stored HY-OAS == FRED through 09-25 (2.93, pctile 9.9%); FRED published 09-28 = 3.02 after the
  08:35 run — gate far from p95.
- **Breadth:** recomputed from raw closes 21.4% (identical). Equal-weight 50d/200d 21.4% / 43.1%; cap-weighted
  46.6% / 71.1% — the rally is narrow and the equal-weight measure sees it (1 month: median S&P 500 stock −5.9%,
  cap-weighted −0.7%).
- **Price basis:** the backtest's prices are total-return (CRSP ret chain / Compustat trfd); live bars are split-adjusted
  price-only. Rebuilt on dividend-adjusted bars (6,167 dividends, 1,028 names): identical BUY list, weights within 0.01,
  breadth 22.0% vs 21.4%.

**Defects:**
1. **Evening signals a session stale — FIXED LIVE (3c0e76e, restart 01:39).** The price cache was valid by age alone
   (18h), so the 17:50 / 18:33 builds reused files fetched that morning, before the day's bar existed: the 09-29 evening
   signals (and the evening book-3 dry run) were computed on 09-28's close. The fresh build differs: +LGND/RDDT/VICR,
   −ADSK/ILMN, most weights moved. Worse case: after the Sunday 21:20 restart the 18h clock expires Monday ~15:20, so
   Monday's refetch cached a ~15:05 intraday snapshot as Monday's bar, reused by the evening builds AND Tuesday's 09:18
   pre-open build (09-29 09:18:51 did exactly this) until the ~09:33 refetch — a Tuesday rebalance could have traded on
   it. Now: a file is reused only if <18h old AND written after the last settled close (weekday 17:00 ET); the
   partial-session clock rule runs to 17:00. Strictly more conservative (20,000 random cases). The evening restart
   refetches the completed session; the pre-open build reuses those final bars with no vendor call before the open.
2. **15 renamed tickers dropped at every rebalance — FIXED LIVE (166a4b6).** The refresh job's `data_gaps` step
   rewrote AGNT, CALY, CVSA, DCH, DMC, ECHO, EFOR, FISV, MPT, MRSH, OPLN, P, PPLI, VMRK, VSXY from yfinance every evening
   (~18:30) with `end=today`, which yfinance treats as EXCLUSIVE: the files ended a session early, looked fresh by mtime,
   and were reused by the 18:33 build and every pre-open build → NaN last bar → no dist_sma200 → absent from the
   momentum/lowvol sleeves (coverage 99.1% → 98.1%, far above the 85% alarm). Logged nightly since at least 09-23. Now: a
   cached symbol whose last bar is older than the session most symbols end on is refetched (content-based, whoever wrote
   the file); yfinance `end=tomorrow` in the runtime patch and in `data_gaps`; `data_gaps` never overwrites a file with
   older data. None of the 15 is a BUY on correct data today.
3. **Price archive mislabelled — writer FIXED (166a4b6); history QUARANTINED 2026-09-30 ~10:10 ET (owner approved):
   all 91 old files moved to `data/price_archive/_mislabeled_pre_2026-09-30/` with a README.** `archive_daily_prices` labelled
   each cache file's last row with the wall-clock date: `closes_D` held mostly D-1's closes (Mondays: a ~15:05
   snapshot). Research only; nothing reads it. Now archives only bars that had settled when the file was written, under
   their own session date.
4. **Fundamentals quarter mixing — FIXED (eea223f; on the box since ~02:10, loaded by the post-06:00 signal-server
   restart together with defect 6).** `groupby("tic").last()` takes each column's
   last NON-NULL value, so a blank field in the newest quarter came from an older quarter (dlcq for 208 of 1,496 pool
   names; seqq VSXY/MDT; cogsq CPB); the backtest keeps one record per quarter (NaN stays NaN, missing debt = 0).
   A/B on 09-29 data: identical BUY list, PAYX/NTNX weights move in the 4th decimal. Deploy after the close.
5. **Vol-scale depends on overnight restarts — fix ON DISK (eea223f), active at the next ENGINE restart (deliberately
   not restarted before the rebalance).** `record_nav()` seeds today's
   entry at engine start; the 00:21 ET restart after IBKR's nightly reset wrote 09-29's after-hours NAV as
   "2026-09-30". Included, it swaps the oldest real return for a ~0 pseudo-return: vol 22.82% / scale 0.979 / book
   target 1.46x instead of 23.65% / 0.945 / 1.41x. The completed-sessions window reproduces the logged 09-09 value
   (26.1% / 0.86) exactly; 09-16 and 09-23 are identical either way. Removing the seed before the open was blocked (a
   live state-file edit needs the owner's OK), so book 3 rebuilds at 1.46x (inside the validated range, ~$800 more gross
   than the convention). Fix: `compute_vol_scale` ignores today's entry until the 16:05 close mark exists.
6. **Stale S&P 600 membership changed picks — FIXED (baf7a8a; live from the 06:00 scrape + a signal-server restart).**
   `scrape_sp1500.py` read all three lists from Wikipedia; its S&P 500 / 400 pages matched SPY / MDY holdings exactly,
   but its S&P 600 page had not applied the September rebalance: vs SPSM's 09-28 holdings it lacked 15 new members
   (ARQT, ATRC, AXTI, BLDR, CPRI, DK, HOS, HRI, PRK, RUSHB, SAM, TAP, TENB, TMP, TTD) and kept 11 removed ones (AMSF,
   CCOI, FBRT, HLX, LEG, MATW, NABL, NXRT, SHEN, VRRM, plus CWEN.A where the index holds CWEN). Rebuilt with the ETF
   list, **ATRC replaces LGND (lowvol) and AXTI replaces VICR (momentum)** in the 09-29 BUY list; no held name leaves
   the index. Now: SSGA daily holdings (SPY / MDY / SPSM) first — standard-library xlsx parser (no openpyxl on the box),
   used only if the count is in range and >= 90% of names overlap Wikipedia or the last-good list — then Wikipedia,
   then last-good; the file records its source per index. (The Wikipedia S&P 600 page is the documented weak link.)
7. **Membership fetched without certificate verification — FIXED (5af1e4e).** The scraper's SSL probe sent Python's
   default User-Agent; Wikipedia answers 403 and ANY exception switched every fetch to an unverified SSL context.
   Verification now drops only on a genuine certificate error (verified SSL confirmed working from the box).
Also: `tests/test_mom_equal_weight.py` blanked (not popped) the EXP-059 flags — `load_dotenv` re-set them from the
box's `.env` during import, so its "code default" checks failed on the box only.

**Verified live after the 01:39 restart:** served signals == the all-fresh ground truth on all 1,502 symbols; as-of
2026-09-29; 23 BUY; breadth 21.4%, blend 0; dist_sma200 coverage 99.1%. The same code on a copy of the live cache
(real mtimes) reproduced it exactly (1,522 pre-settle files refetched, 15 stale-content files refetched).
**Book-3 dry run on the correct data** (engine's own functions, 01:08 ET prices): target 1.46x (vol-scale 0.98, credit
off), sizing mult 1.65. Sells SEZL 4, PAYC 2, APPF 2, SNDK 1, LITE 1, ADSK 2, YELP 20, WDC 2, RNG 4. Buys MXL 22, MU 1,
NTNX 24, CORT 12, OGN 109, AMD 2, FTNT 5, CRWD 4, PANW 2, LGND 3, ADBE 2, DOCS 35, RDDT 6, CARG 27, BSY 12, PAYX 4,
DUOL 1, BX 6, WDAY 4, MRNA 2, VICR 1. LITE / SNDK round to 0 shares (book-3 targets ≈$567 = 0.58 / 0.33 of a share).
After: book 3 21 names (MXL 15.0%, MU 13.8%, NTNX 12.2% of the quarter), account 33 positions, gross ≈1.39x NAV.
Lesson: an evening dry run is only as fresh as the signals' AS-OF session — check it (`_uni_cache_date` / the
"computing on last completed session" log line) before quoting trades.
**06:00-06:07 ET — membership fix LIVE:** the 06:00 cron scrape used SSGA holdings for all three indexes (503 / 400 /
603, identical to the 02:14 rehearsal); signal server restarted 06:03 (1,526 bars reused, 15 new members fetched),
build 06:04:31 as-of 2026-09-29, 23 BUY, breadth 21.6%, coverage ok — served signals IDENTICAL to the precomputed
expected set on all 1,506 members. **Final book-3 plan** (engine functions, 06:07 prices, the RUNNING engine's
vol-scale 0.979 → 1.46x): sells SEZL 4, PAYC 2, APPF 2, SNDK 1, LITE 1, ADSK 2, YELP 20, WDC 2, RNG 4; buys MXL 21,
MU 1, NTNX 24, CORT 11, OGN 108, AMD 2, FTNT 5, CRWD 4, PANW 2, ATRC 18, ADBE 2, DOCS 34, RDDT 6, CARG 27, BSY 11,
PAYX 4, DUOL 1, BX 6, WDAY 4, AXTI 7, MRNA 2; book 3 → 21 names at 1.47x; account 33 positions, 1.39x gross.

## "DATA REFRESH PARTIAL — Failed: enhanced_data" 2026-09-29 17:40 — false alarm; refresh job hardened (0155bab)
**What happened:** the 17:30 cron `scripts/refresh_data.py` step `enhanced_data` hit its 600 s SIGALRM budget. That step ran
`fetch_all_data.main()`: the FMP caches (price targets/DCF 3-day TTL, growth/EV/profiles 7-day TTL, econ calendar 1-day)
PLUS a full 1,502-name options sweep — the same sweep the separate `options` step repeats later. On the weekly day the
7-day FMP caches expire (~7.5 min of FMP calls) the duplicate ~4-min sweep pushed it past 600 s (killed at options
900/1502). **No data was lost:** every FMP file was saved 17:32-17:37 and the `options` step captured the 09-29 snapshot
at 18:32 (1,495 names). **No live impact:** live signals are built with `build_signals_v9(raw, enhanced_data=None, ...)`
(the FMP enhanced maps are deliberately empty live, matching the backtest's `DEPLOYED_EMPTY_MAPS`), and the engine's
go/no-go gate (`/health` is_stale) tracks price bars only.
**Defects found and fixed:** (1) SIGALRM timeouts raised inside steps can be swallowed by the fetchers' broad
`except Exception` (or fire mid-save); (2) steps catch their own errors and return False, and the wrapper only retried on
exceptions — retries never ran; (3) a step could "succeed" without writing anything; (4) the alert called research-only
inputs "critical"; (5) options snapshots were labelled with the wall-clock date (Sunday 21:00 runs stored Friday's
session as 09-27).
**Now:** each step runs in a forked child with a hard kill at its budget, retries on timeout/exception/False, and must
leave its declared output files within their max age. `STEPS` table (budget ~2x slowest run): enhanced_fmp 1500 s
(FMP only), VIX 180 s [LIVE], fundamentals 3600 s (live fallback only), options 900 s, snapshots, price archive,
journal, Fama-French, gap scan, cache flush. Alerts say "LIVE INPUT FAILED" or "research data only (live trading NOT
affected)" and name the reason. Options are labelled by the session in the contracts' `day.last_updated`.
`tests/test_refresh_runner.py` (17 checks) pins the runner semantics AND asserts every live `build_signals_v9` call
passes `enhanced_data=None` — if that ever changes, the test fails and enhanced_fmp must be re-classified as live.
Verified on the box 09-30 00:35-00:40 through the new runner: enhanced_fmp ok (1 s, caches fresh), VIX ok, options ok
(241 s, 1,495 names, labelled session 2026-09-29 from a 00:40 run and deduped, no 09-30 rows); tests 17/17 + tranche
47/47 on the box. History cleanup: 18 weekend-labelled dates (25,930 rows) each duplicated an existing Friday row and
were dropped; backup `data/enhanced_data/options_history.pre-weekend-cleanup-2026-09-30.parquet`; 109 clean session dates.

## Telegram /commands silent 2026-09-27/28 — FIXED (cca48e7, engine restarted 14:03 ET 09-28)
The poll loop was started with a bare `asyncio.create_task()`; asyncio keeps only weak references to tasks, so the GC
destroyed it mid-await ("Task was destroyed but it is pending!" at 09-27 09:43 and 09-28 00:30, both right after an IB
reconnect). Outbound alerts kept working; /commands got no replies (6 updates were pending at Telegram). Fix:
`_ensure_telegram_task()` holds `self._tg_task` and restarts the loop at the top of every main-loop cycle if it is done.
Verified: listener started, pending updates 0, confirmation message delivered.

## Book 2, 2026-09-23 — first rebuild under the EXP-059 package: CLEAN
09:32:50 signals (22 BUY; overnight refresh vs the 09-22 dry run: NFLX out, DUOL in; breadth 26.5%, blend 0) ·
credit gate off (HY-OAS pctile 2.4%) · vol-scale 0.96 -> target 1.43x · SIZING mult 1.66 -> projected 1.45x · 13 exits
(SEZL, PAYC, STX, APPF, DDOG, PAYX, SNDK, LITE, BKNG, META, CRWD, YELP, WDC, RNG — SNDK/LITE are still signals but round to 0
shares at NAV/4) + 20 buys, 37 fills, no rejects/partials, done 09:37:30 (no watchdog). Overlay: "no book above target —
no trims" (books 0/1/3 at 1.30/1.33/0.65x). After: book 2 = 20 names, 1.46x, max MXL 14.6%; ledger == broker (36 names);
every book-2 name has a book-keyed stop peak; tranche state counter 0, next book 3 (2026-09-30). Account gross 1.18x,
NAV $61,827, cash -$11,302 (margin, expected at 1.18x).

## EXP-059 package — SHIPPED 2026-09-22 21:50-21:57 ET (owner's go, box + GitHub at 9f6e562)

**What is live now:** `IBKR_OVERLAY_DOWN=1`, `MOM_EQUAL_WEIGHT=1`, `PROD_BULL_WEIGHTS=0.80,0.15,0.05` in the box `.env`
(no comment lines — `export $(cat .env | xargs)` in both start scripts cannot parse them; that bit me once tonight before
any restart). Gate before shipping: the package re-measured under the LIVE 15% combiner cap (arm `cap0.15_package`, 24
starts): 26yr +1.56pp / +0.064 Sharpe (24/24) / +5.5pp MaxDD, 15/25 years, CI [+0.024, +0.115]; 8yr +3.4pp / +0.105 (24/24),
fails only 2020-concentration. All 7 test suites + lint green locally and on the box.

**Trap found and fixed before restart:** `signal_server.py` imports the strategy module (which reads the two signal
flags at import) BEFORE its `load_dotenv()`, and `start_signal_server.sh` did not export `.env` — the flags would have
been silently ignored. Fix: the start script now exports `.env` exactly like the engine's; both services log the
resolved flags at startup (`EXP-059 flags (resolved at import): MOM_EQUAL_WEIGHT=True PROD_WEIGHTS_BULL={0.8,0.15,0.05}`;
`EXP-059 de-risk overlay (IBKR_OVERLAY_DOWN): ON`). Verified in both logs.

**Signal-side verification:** server restarted 21:50:45, health OK at 65s, cache rewritten 21:51:47 (the first fetch
after restart returned the pre-change 18:28 cache — "serving while refreshing" — so compare only after the rewrite).
BUY list 22 names, identical set before/after; momentum names' relative weights shifted (CORT 0.744→0.679, NTNX
0.702→0.641, DELL 0.563→0.514). **Breadth is 25.9% → blend 0.00: the combiner is on the BEAR weights (11/33/56), so
the 80/15/5 bull split is dormant until breadth > 35%.** Same formula as the backtest; confirmed real with an
independent 119-name sample fetched fresh from Massive (31.9% above the 50d SMA). Pre/post signal snapshots saved as
`data/signals_{pre,post}_exp059_2026-09-22.json`.

**Engine restart 21:56:48:** `Loaded tranche state: counter 4/5, next book 2`, overlay ON, connected, NAV $62,971,
books 21/20/23/18 names, vol-scale 0.96. Ledger snapshot `data/ibkr_tranche_state.pre-exp059-restart.json`.

**Dry run of 2026-09-23 (book 2) with the new signals and the engine's own pure functions:** target 1.43x (vol-scale
0.96), sizing mult 1.65, projected gross $22,533 on a $15,743 quarter, 20 names, max weight 14.9%, LITE/SNDK round to 0;
15 legacy exits (SEZL, PAYC, STX, APPF, DDOG, PAYX, SNDK, DUOL, LITE, BKNG, META, CRWD, YELP, WDC, RNG) and 19 buys
(MXL 23, CORT 15, NTNX 24, DELL 2, OGN 110, AMD 2, FTNT 6, PAYO 166, ABBV 4, ADSK 2, ADBE 2, DOCS 35, RDDT 5, CARG 26,
BSY 11, ELF 7, BX 6, NFLX 10, MRNA 3); MU held. Overlay: no trims (books 0/1/3 at 1.30/1.31/0.65x, all below target).
The morning's actual orders are compared against this list.

**Breadth / "dormant bull split" verified end to end (2026-09-22 22:00-22:40 ET), owner asked twice:**
1. Live number is right: the server's 25.9% (computed on the 09-21 session — the partial-session guard dropped the empty
   09-22 row at 21:50; Massive's daily bars arrived minutes later) matches an independent full-universe recomputation
   from fresh Massive bars: 25.8% as of 09-21, 26.5% as of 09-22 (1,532 names, same 50-day-SMA definition).
2. Same formula both sides: live `breadth = share of feature-map names with dist_sma50 > 0`, `blend = clip((breadth-0.35)/0.25, 0, 1)`;
   backtest `main_production_backtest.py:411-416` and the clean room use the identical expression and the same bear
   weights constant.
3. Pool difference is negligible: the backtest counts every name in the panel (median 1,854 incl. ex-members still
   trading), live counts current members (1,502); over 2018-2026 the two breadth series differ by 0.009 on average
   (max 0.045) and the blend differs by more than 0.1 on 1.6% of days.
4. It is a normal state: in the backtest's own data blend = 0 on 18% of days since 2018 and blend < 1 on ~half; the
   backtest's breadth fell from 0.53 (Aug 24) to 0.40 (Sep 1) and was 0.46 on Sep 4, its last day — the live 0.26 on
   Sep 21 continues that slide (a narrow, mega-cap-led September).
5. Consequence: while breadth < 35% every rebuild uses 11/33/56 momentum/value/lowvol, in the backtest as in live;
   the 80/15/5 split engages as breadth recovers (linearly from 35% to 60%). The package's validated numbers include
   these regimes. Nothing to fix.
6. History of the live blend (recomputed 2026-09-22 from fresh vendor bars, 1,532 names, session before each build):
   Jul 28 breadth 68% / blend 1.00 (70% momentum) · Aug 11 62% / 1.00 (70%) · Aug 25 54% / 0.75 (55%) · Sep 9 book 0:
   40% / 0.21 (24%) · Sep 16 book 1: 31% / 0.00 (11%) · Sep 23 book 2: 26% / 0.00 (11%). Weekly breadth 08-14 66% ->
   08-21 53% -> 08-28 48% -> 09-04 45% -> 09-11 34% -> 09-18 26%. The summer books were momentum-led; book 1 was the first
   fully defensive build. SNDK/MU/WDC entered on Aug 11 (70% momentum) and were re-selected Sep 9.
(Research-side accessor trap found on the way: the raw universe's `get_sp500()` returns a legacy S&P 500 TICKER list;
the clean room replaces it with `bt._get_sp1500` (PERMNO-keyed). Scripts must use the latter — E-060e in BUGS.md.)

**Rollback:** delete the three lines from `.env`, `pm2 restart signal-server --update-env && pm2 restart ibkr-engine --update-env`
(after the close). Previous state of this section follows for the record.

## EXP-059 package — BUILT BEHIND FLAGS (2026-09-17), record before shipping

Research result (`ml_service/research/FRONTIER_059.md`): three small changes, each gate-tested on 24 starts x 2
horizons, together +1.5pp CAGR / +0.06 Sharpe / +5pp MaxDD on the 26yr (17/25 years, CI > 0, intact at 2x cost).
The live code paths exist and are covered by tests, but every flag defaults OFF so the deployed behaviour is
byte-for-byte unchanged until the owner flips them (`.env` on the box + restart after the close):

| flag | where | ON value | what it does |
|---|---|---|---|
| `IBKR_OVERLAY_DOWN` | `ibkr_engine.TRANCHE_OVERLAY_DOWN`, hook at the end of `rebalance_tranche()` | `1` | after the rebuilding book is done, every other book more than 5% above today's vol-scale x credit-gate target is trimmed pro rata (`_overlay_trims`, whole shares, 0.3% min-trade band, sells bounded by the broker position, per-fill ledger persistence). Never levers up. |
| `MOM_EQUAL_WEIGHT` | `multi_strategy_engine.MOM_EQUAL_WEIGHT` -> `_weight_picks` (signal server) | `1` | strategy1 returns 1/N for its top-5 instead of score weights capped at 2/N |
| `PROD_BULL_WEIGHTS` | `multi_strategy_engine.PROD_WEIGHTS_BULL` (signal server) | `0.80,0.15,0.05` | bull sleeve split (must sum to 1; asserted at import) |

Tests: `tests/test_overlay_trims.py` (11 checks incl. 300-case parity with the clean-room formula),
`tests/test_mom_equal_weight.py` (7), `tests/test_tranche_engine.py` unchanged (47). The signal-server flags take
effect at the next signal build; the engine flag at the next tranche day. Roll back = unset + restart.

## Tranched rebalance — DEPLOYED 2026-09-08 17:43 ET (FINAL @1.49x), first tranche day 2026-09-09 open

**Deployed** on the owner's go: box fast-forwarded 8ac11df → d59d550 (126 files; the four scp'd
files were reset first, three scp'd test files removed), pushed to GitHub from the box, tree clean.
`signal-server` restarted 17:40 (weights 70/21/9, gate DERISK 0.0, partial-row guard all confirmed
loaded from the venv; first rebuild saw only 29% of names printed for 09-08 → computed on 09-04,
coverage dist_sma200 99.1%, coverage_ok True, 25 BUY). `ibkr-engine` restarted 17:43 (NAV $61,705,
vol-scale 0.86 on 26% realized vol, "Market closed — waiting"). Fresh tranche state → first tranche
day (transition + book 0) at the 09-09 open. Expected book-0 orders from the 09-04-close signals:
~18 sells (~$10.5k, mostly quarter-slices of names leaving the signals) and ~19 buys (~$12.9k)
against a book NAV of ~$15.4k at a 1.31× target; the 09-15 refresh recomputes on 09-08 closes so the
exact names may shift. Local `git push` over HTTPS needs `gh auth login` (token expired); pushes
went through the box's deploy key.

**Decision.** Owner chose FINAL @1.49x on 2026-09-08 after the v2 re-verification and the 17-point
leakage audit (`ml_service/research/FINAL_RECOMMENDATION.md` §0). Three changes, everything else
identical: (1) 4 sub-books rebalanced on a 5-session stagger instead of one 20-day book;
(2) credit gate depth 0.50 → 0.00; (3) bull sleeve mix 50/35/15 → 70/21/9. Leverage stays 1.49x
closed-loop, vol overlay, 40% stop, 15% cap, top-5 momentum all unchanged.

**Engine (`ibkr_engine.py`).** `TRANCHES=4`, `TRANCHE_STRIDE=5` (env-overridable; `IBKR_TRANCHES=1`
is the rollback to the untouched legacy path). `rebalance()` dispatches to `rebalance_tranche()`,
which keeps its own trading-day counter in `data/ibkr_tranche_state.json` and, every 5 sessions,
rebuilds ONE book: same signal gate (stale + coverage), same emergency short cover, same
vol-scale × credit-gate target, same `_calibrate_quantities` closed-loop integer sizing — on
NAV/4 with the 15% cap per book. Orders are the net delta of that book only; sells are bounded by
(actual position − other books' holdings) so no book can sell another's shares
(`_tranche_orders`, pure). `_reconcile_books` re-syncs the books to the broker before every
tranche day and every stop check (stops, manual trades, partial fills). Trailing stops are
per book (`trailing_peaks["<book>:<SYM>"]`) and sell only that book's slice
(`_check_trailing_stops_tranched`). Telegram `/status` and `/rebal` show the book cadence.

**Transition.** First tranche day after deploy: existing holdings are split 4 ways in whole
shares (remainder to the lowest books) and book 0 is rebuilt; books 1–3 follow at 5-session
intervals, so the whole account is on the new schedule after 20 sessions. A fresh state fires on
the first NEW trading day after the restart, at the open — never mid-session on the deploy day
(the counter starts one short of the stride and, if the engine starts during a session, today is
pre-marked as counted). Legacy single-book trailing peaks are carried into every book's slice at
the transition, so a holding already 30% off its high keeps its 40% stop reference. Deploying any
time on 2026-09-08 therefore gives the first tranche day at the 09-09 open, which is when the
legacy 20-day clock (day 20) would have fired anyway.

**Tests.** `tests/test_tranche_engine.py` (39 checks, offline: split conservation, reconcile
shortfall/surplus, bounded sells, tiny-delta skip, cap/closed-loop sizing on NAV/4, gate ×0.00
→ empty book, deploy-day guard, legacy-peak carry-over, cadence and same-day idempotence,
per-book stop sells only that slice, a 22-session fake-clock scenario with rotation 0-1-2-3-0, an
external sell absorbed by reconcile, a mid-cycle restart reloading the state file, and the
books-equal-broker invariant every session; legacy fallback). `tests/lint_engine_names.py`
(AST undefined-name pass, pyflakes-equivalent) clean on all four changed files. Existing engine
tests (coverage gate 5, cache depth 6, vendor hole 10) unchanged and passing.

**Deploy steps (on the owner's go).** The box (`/home/ubuntu/AutoTraderBot`, HEAD 8ac11df) carries
four locally modified files (`ibkr_engine.py`, `signal_builder.py`, `signal_server.py`,
`massive_data_provider.py`) that are byte-identical (md5) to the committed 09-02/09-07 fixes — they
were scp'd, never pulled. So: `git checkout -- ml_service/{ibkr_engine,signal_builder,signal_server,massive_data_provider}.py`
→ `git pull` → `pm2 restart signal-server` (picks up `PROD_WEIGHTS_BULL` and the partial-row guard)
→ `pm2 restart ibkr-engine` → confirm `/rebal` shows "transition pending" and `/health` coverage_ok
→ first tranche day at the next open. Engine runs under `ml_service/venv` (Python 3.12). **Rollback:** `IBKR_TRANCHES=1` in the pm2 env restores the
20-day path (the state file is ignored); set `credit_gate.DERISK` back to 0.5 if rolling back,
because 0.0 on a single book is a one-day full liquidation.

**Known limits.** `/deploy` (idle-cash top-up) still sizes against the whole book, not a
tranche — a manual command, use sparingly. Alpaca paper still runs the single 20-day book
from the same signals (so it is no longer a mirror of IBKR).

## Gotchas that previously caused misdiagnosis
- **Alpaca multi-strategy buckets are DISABLED**: `momentum: 0, mean_reversion: 0,
  mega_cap: 0` — all 30 slots are the v12 factor bucket. Log labels "ML 22/30, MOM 0/0"
  reflect this. The dormant bucket code is not live.
- **"v9.6" in comments = signal-generation version; "v12" = strategy version.** Consistent,
  not a mismatch.
- **Holdings lag the BUY list by up to 20 days** (rebalance cadence) — expected, not a bug.
- **Leverage config numbers differ on purpose** (IBKR adaptive w/ 1.8 ceiling; JS 1.6):
  both land ~1.49x effective. Do not "align" the numbers.
- The rebalance clock is persisted + NYSE-calendar-gated on BOTH engines (holiday
  phantom-count fixed 2026-07-04); the two day-counts must always match.
- `record_nav` close marks are write-once per day (evening restarts must not corrupt them).
- Wikipedia SP1500 scraper has per-index last-good fallback — check the `stale` field in
  `sp1500_members.json`.
- **PARTIAL-SESSION GUARD (2026-08-11, commits 2ef61fb + 5304eea) — live-only silent
  exclusion.** v12 is a daily-CLOSE strategy, but `signal_server` rebuilds every
  `REFRESH_MINUTES` (15) **during market hours**, so the in-progress bar entered the price
  matrix. A symbol whose bar hasn't printed is NaN, and pandas defaults `rolling()`'s
  `min_periods` to the window, so **one** missing bar makes `c.rolling(200).mean()` NaN →
  `get_feature_map` DROPS NaN keys entirely → `dist_sma200.get(sym, 0)` returns 0, fails
  `> 0`, and the name is silently cut from the momentum AND lowvol sleeves. No log, no alarm.
  Measured 2026-08-04 10:11 ET: **SNDK had 199/200 bars and vanished despite a 12-1 score of
  30.37 — 4x the #1 name**; 21 of 1539 symbols affected. `ret_252d` survives the same gap
  (`pct_change` touches 2 rows), which is why a name can post a huge score and be invisible.
  The backtest cannot express this (it only reads completed bars), so the guard restores
  parity rather than adding divergence, and makes the book deterministic through the day.
  Chose this over loosening `min_periods`, which would change feature VALUES and require
  mirroring in `build_universe_2000.py`. Only drops while the session is open (after 16:00 ET
  the bar is final; the 15-min loop is gated on `_is_market_hours()`).
  ⚠️ **`_is_partial_session()` is the single source of truth and must stay that way.** The
  first cut trimmed only the price matrix while `build_signals_v9` still took `today` from
  SPY's raw bars — when the guard fired, features ended on the completed session while the
  lookup asked for the in-progress one → **EMPTY BOOK (0 names) + parity alarm**. Caught by an
  efficacy test before any restart. There is now also a reconcile step: if `today` is absent
  from the universe's features, trust the universe and log a warning.
  Verified: identity on real data (book byte-identical, 25 names) AND efficacy on an injected
  partial session (old = SNDK silently absent; new = SNDK present, 25 names).
- **GROSS_MARGIN NOW BOUNDED TO [-1,1] (2026-08-10, commit 74d57cb) — was a LIVE-ONLY book
  collapse.** `gross_margin = (saleq-cogsq)/saleq`, so |gm| > 1 is impossible; Compustat rows
  with saleq ~ 0 produce garbage. VIR carried **gm = 4561.72**. `strategy_value` had only a
  LOWER bound (`g < 0.15`), so it passed and scored **1140.2 vs 0.4669** for the best real name
  (TTD). The sleeve's 2/N cap does not contain this (see the `strategy_value` docstring — the
  cap is renormalized away), so VIR took ~93% of the value sleeve and ten legitimate names
  (ADBE ADSK BSY DUOL LIF META PAYC PAYX PTC SEZL) fell under the 0.005 floor:
  **live served 15 BUY names where every backtest runs 25.** Second effect: `strategy5` feeds
  gm through `zscore()`, and one 4561.72 in n=1487 drags mean to ~3.42 / std to ~118, so every
  legitimate name collapsed to z ~ -0.025 — measured **z-spread 1.0000 -> 0.0012**, i.e. the
  low-vol quality factor was switched OFF, not merely tilted.
  LIVE-ONLY because the EDGAR overlay (roe-only) injects VIR roe = 0.326 over its true -0.6186;
  without the overlay VIR fails `roe >= 0.05` and never reaches the scorer. **Verified: overlay
  OFF = 25 names, ON = 15.** Fixed via `_sane_gross_margin` in shared code (value + lowvol), the
  same bound the overlay already applied to its own values (`signal_builder.py:496`).
  Post-fix live: **25 names, VIR absent, max name 16.5% of book.**
  ⚠️ The BACKTEST universes carry the same corrupt values (8yr: 80,398 obs outside [-1,1],
  **99.2% of dates**; 26yr: 186,842, **99.7% of dates**), so the low-vol z-score was degraded
  there too. The fix corrects both sides together (parity preserved) but **the canonical
  numbers were computed with the contamination and must be re-run.**
- **SECTOR SLEEVE (s3) ZEROED 2026-07-26 — parity by construction.** `strategy3_sector_rotation`
  returns sector **ETFs** (XLK/XLE/...), which the live path can never deliver (signal_builder
  emits SP1500 *members*; ETFs are not members), so live silently dropped them while the
  backtest bought them — active whenever bear/crash weights applied (s3 was .10 there).
  Measured (`research/THREAD_S3_FINDINGS.md`, full live-mirror, both periods): dropping s3 and
  renormalizing = **+0.1pp/+0.2pp CAGR, IDENTICAL Sharpe (0.96/0.79), -0.3pp/-0.2pp MaxDD** —
  inside noise. So s3 was zeroed in `PROD_WEIGHTS_BEAR/CRASH` (survivors renormalized) to make
  live and backtest the same book. **Verified live-neutral on deploy: same 24 names, max
  probability change 0.0002.** s3 code path retained, zero-weighted.
- **UMD CRASH MODE was ACTIVE on 2026-07-26** (20d price-UMD -0.079 vs -0.05 threshold) — the
  book runs `PROD_WEIGHTS_CRASH` (momentum cut, lowvol/value raised). If the top signal looks
  like a defensive/odd name rather than the strongest momentum, check the UMD state FIRST:
  served "probability" is the blended **portfolio weight** (max = 0.95), NOT a momentum score.
- **UNIVERSE PARITY (fixed 2026-07-25, c8a097a) — the biggest divergence found to date:**
  live momentum+lowvol sleeves selected from **SP500-only** (FastUniverse.get_sp500 →
  sp500_constituents.json, 503 names) from launch until 2026-07-25, while every validated
  number selects from SP1500. Measured (clean-PIT live-mirror A/B, 2018-25 3-start):
  **−15.7pp CAGR / −0.38 Sharpe**. Fixed by overriding get_sp500 → get_sp1500_on_date in
  signal_builder (same source the value sleeve always used). Also fixed same-commit:
  rev-surprise ×1.10 / beat-streak ×1.05 boosts zeroed (were live-only, unvalidated);
  roe seqq>0 guard (spurious value-eligibility for negative-equity names, e.g. SABR).
  First trading effect: Aug-11 rebalance — expect HEAVY turnover into mid-cap momentum.
- **Trailing-stop KeyError (fixed 2026-07-16, fb3f539):** every stop fire crashed the rest of
  that cycle (`check_trailing_stops` double-deleted `trailing_peaks[sym]` — `sell_position`
  already removes it). The sale always completed; an "⚠️ IBKR Engine error: '<SYM>'" telegram
  right after a stop was THIS, not a failed sale. First exposed by the SNDK −40.2% stop.
- **Engine runs clientId 2 since 2026-07-17** (`IB_CLIENT_ID=2` in `.env`). The gateway's
  internal clientId-1 session WEDGED during IBKR's nightly server reset (00:14 ET, "Error
  326: client id already in use" while the container stayed up) — engine flapped 00:14→09:01
  until switched. If 326 ever recurs on clientId 2: bump the id again or restart the
  `ibgateway` container (may require 2FA re-auth). Telegram/pnl/deploy "breaking" during such
  an outage is downstream of the dead IB connection, not a bot bug.
- **"/deploy: nothing to deploy" with fresh cash is usually CORRECT**, not a bug: deploy only
  buys the deficit to (1.49 × vol_scale × NAV). Elevated vol → scale ~0.5 → target below
  current gross → inert. New cash goes to work at the next rebalance / when vol normalizes.

## Services (PM2 on EC2 54.158.238.15)
- `ibkr-engine` — LIVE v12, IBKR (IB Gateway in docker, engine connects :4001)
- `trading-engine` — Alpaca PAPER v12 mirror
- `signal-server` — v12 signal brain on `:5001` (restarted 17:50 daily)
- `edgar-monitor` — stopped (intentional)

Key crons: `refresh_data.py` 17:30 (fundamentals timeout 3600s), `scrape_sp1500.py` 06:00,
`sentiment_collector.py` every 6h.

Telegram bot (ibkr-engine): `/pnl /daily /portfolio /positions /alpaca /status /signals
/rebal /data /connection` read-only for allowlist; `/reconnect` owner-only (restarts the
Gateway; approve IB-Key 2FA when asked). Never log into IBKR Client Portal/web — it kills
the Gateway session.

> `docs/BACKTEST_LIVE_PARITY.md` (April 2026) describes the older multi-strategy setup —
> historical only, superseded by this document.
