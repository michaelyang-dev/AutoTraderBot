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
| Sleeves | mom .50 / val .35 / lowvol .15 (bear .1111/.3333/.5556, **sector 0.00**, breadth-blended) | same | same (`PROD_WEIGHTS_*`) |
| Risk-parity in sleeves | **NO** (signal_builder skips it) | same | model with `use_rp=False` |
| Rebalance | 20 trading days, persisted `ibkr_rebal_state.json`, NYSE-calendar gated | same via `js_rebal_state.json` | `rebal_days: 20` |
| Position cap | 15% of NAV (≈10% of the 1.49x book) | same | model with `cap: 0.10` |
| Trailing stop | 40% from peak | same | same |
| Take-profit | none | none (deleted 2026-06-25) | none |
| Sizing | **closed-loop** (2026-07-04): `_calibrate_quantities` targets 1.49x × vol_scale of MEASURED gross; old `LEVERAGE=1.8` is only a safety ceiling | `MAX_CASH_DEPLOY_PCT=1.49` × volScale — **FIXED 2026-07-12 from 1.60** (it was deploying ~1.60x: live gross/equity 1.55x with top pos 13.8%, 15% cap not binding, so the "caps→1.49" premise was false; fractional shares mean the constant IS the deployed leverage) | overlay leverage on 1x returns |
| Vol-scaling | target 0.15 1x-equiv (0.2235 on levered NAV), lookback 40d, floor 0.30, **de-risk-only cap 1.0** | same policy | `vol_scaling` flags (+ `vol_scale_cap: 1.0` via research fork) |
| **Credit de-risk gate (LIVE 2026-07-18)** | halve gross-leverage target while HY-OAS ≥ p95 of its expanding history (`credit_gate.py`; FRED BAMLH0A0HYM2; cron 8:35 refreshes `data/credit_signal_live.parquet`; engine reads file only, FAIL-SAFE 1.0; applies at rebalance + /deploy, exactly the validated cadence; Telegram on gate-ON + on stale feed) | not ported (paper mirror) | `livemirror_backtest` gate_cols=[hy_oas] p95×0.5 — 26yr MaxDD −63.5→−56.6 at +0.3pp CAGR; OOS +16.6pp DD (`research/THREAD_T_FINDINGS.md`) |
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

---

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
