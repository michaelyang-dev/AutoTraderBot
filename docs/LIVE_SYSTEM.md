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
| Sleeves | mom .50 / val .35 / lowvol .15 (bear .10/.30/.50/.10, breadth-blended) | same | same (`PROD_WEIGHTS_*`) |
| Risk-parity in sleeves | **NO** (signal_builder skips it) | same | model with `use_rp=False` |
| Rebalance | 20 trading days, persisted `ibkr_rebal_state.json`, NYSE-calendar gated | same via `js_rebal_state.json` | `rebal_days: 20` |
| Position cap | 15% of NAV (≈10% of the 1.49x book) | same | model with `cap: 0.10` |
| Trailing stop | 40% from peak | same | same |
| Take-profit | none | none (deleted 2026-06-25) | none |
| Sizing | **closed-loop** (2026-07-04): `_calibrate_quantities` targets 1.49x × vol_scale of MEASURED gross; old `LEVERAGE=1.8` is only a safety ceiling | `MAX_CASH_DEPLOY_PCT=1.6` × volScale (fractional, same effective) | overlay leverage on 1x returns |
| Vol-scaling | target 0.15 1x-equiv (0.2235 on levered NAV), lookback 40d, floor 0.30, **de-risk-only cap 1.0** | same policy | `vol_scaling` flags (+ `vol_scale_cap: 1.0` via research fork) |
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
