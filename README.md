# AutoTrader v12

Autonomous long-only equity system that trades the S&P 1500 with a momentum / value / low-volatility-quality
sleeve mix, ~1.49x closed-loop leverage, and a four-book staggered ("tranched") rebalance. Live money runs on
Interactive Brokers; an Alpaca paper account mirrors the same signals. Data pipeline, signal generation, sizing,
execution, risk controls and monitoring are fully automated on one AWS EC2 host.

> **Status — 2026-10-01.** Live on IBKR. Last tranche rebuild: book 3 on 2026-09-30 (30/30 orders filled,
> ledger == broker). Next rebuild: book 0 on 2026-10-07. Market breadth is ~19% of S&P 1500 members above their
> 50-day average, so the sleeve mix is fully on its defensive (bear) weights. Every live data input was
> independently re-verified on 2026-09-30 ([Data status](#data-status)), and a live-vs-backtest parity audit the
> same day fixed the remaining gaps ([Parity](#live-vs-backtest-parity)).

**Sources of truth:** [`docs/LIVE_SYSTEM.md`](docs/LIVE_SYSTEM.md) (operating ledger: every deploy, incident and
verification, dated) · [`ml_service/live_config.py`](ml_service/live_config.py) (canonical machine-readable config)
· [`ml_service/research/LOG.md`](ml_service/research/LOG.md) (every experiment and its verdict).

---

## Contents

- [Validated performance](#validated-performance)
- [Strategy](#strategy)
- [Portfolio construction and risk](#portfolio-construction-and-risk)
- [Architecture](#architecture)
- [Data status](#data-status)
- [Data pipeline and guards](#data-pipeline-and-guards)
- [Live vs backtest parity](#live-vs-backtest-parity)
- [Operations](#operations)
- [Tests](#tests)
- [Project structure](#project-structure)
- [Research summary](#research-summary)
- [Known limitations](#known-limitations)
- [Change history](#change-history)

---

## Validated performance

Backtest of the **deployed configuration** (the EXP-059 package on the tranched book) in the independent
clean-room engine (`research/VERIFY2_cleanroom.py`, patched by `research/EXP059_frontier.py`), 24 monthly start
dates per horizon, ending 2026-08-31:

| Horizon | CAGR (mean; range over starts) | Sharpe | Max drawdown (mean; worst start) |
|---|---|---|---|
| 2018 → 2026 | **+35.8%** (+27.4% … +41.3%) | 1.13 | **−31.4%** (−33.7%) |
| 2001 → 2026 | **+19.8%** (+17.0% … +23.0%) | 0.79 | **−41.3%** (−51.6%) |

What is inside those numbers: point-in-time S&P 1500 membership; survivorship-free total-return prices;
point-in-time quarterly fundamentals by report date; 1.49x closed-loop leverage built from **whole shares at
NAV/4 per book, starting from $50K**; the vol-scaling overlay; the credit gate at depth 0.00; trailing stops on
closing prices; **10 bps per trade** (5 commission + 5 slippage); margin interest from a **time-varying** broker
financing curve (benchmark + spread: ~3.3% average over 2001-26, ~3.9% over 2018-26, ~5.8% today). No look-ahead
(shift-tests and forward-IC audits in `research/BUGS.md`).

How to read it:

- **The 2001-2026 row is the planning number.** 2018-2026 is a momentum-friendly decade; the long horizon includes
  the dot-com bust, the GFC and 2020 and is the through-cycle stress lens. Forward results are regime-dependent.
- **Drawdowns are large by design.** Leverage plus concentrated momentum names: the mean worst drawdown is ~41%
  on the long horizon and the worst start lost 52%. Margin calls are not modeled.
- **Returns are fat-tailed.** A large share of the total comes from a handful of days; a single start date can
  move 8-year CAGR by several points (the range column).
- **The tranche structure is a risk-shaping change, not proven alpha.** Its Sharpe edge has a confidence interval
  that spans zero (`research/LOG.md` cycle 56). The EXP-059 package on top of it: 26yr +1.56pp CAGR / +0.064
  Sharpe (24/24 starts) / +5.5pp MaxDD; 8yr +3.4pp / +0.105 (24/24).

---

## Strategy

Three sleeves pick stocks independently from the **S&P 1500**; their weights are blended by a market-breadth
regime and a momentum-crash detector. All sleeve code lives once in `strategies/multi_strategy_engine.py` and is
imported by both the live signal builder and the backtester.

| Sleeve | Picks | Score | Filters |
|---|---|---|---|
| **Momentum** | top 5, **equal weight** | skip-month momentum `ret_252d − ret_20d`; ×1.15 if 20d vol < 25% and score > 20%; ×1.05 if ROE > 15% | above 200-day SMA; when SPY is below its 200-day SMA, top-3 sectors ×2 and bottom-3 excluded |
| **Value** (quality + long-term reversal) | top 10, score-weighted | `−ret_252d×0.30 + gross_margin×0.25 + min(ROE,0.5)×0.25` | ROE > 5%, gross margin 15-100%, dist to 200-day SMA > −15%, debt/equity < 3 |
| **Low-vol quality** | top 10, score-weighted | mean z-score of low 60d vol, gross margin, low debt/equity, 6-month momentum | needs ≥ 2 of the 4 factors |

**Sleeve mix (regime blend).** Breadth = share of index members above their 50-day SMA. The mix moves linearly
from the bear weights at ≤ 35% breadth to the bull weights at ≥ 60%:

| | Momentum | Value | Low-vol |
|---|---|---|---|
| Bull (≥ 60% breadth) | 80% | 15% | 5% |
| Bear (≤ 35% breadth) | 11.1% | 33.3% | 55.6% |
| Momentum crash (price-based UMD 20-day sum < −0.05) | 16.7% | 50% | 33.3% |

The breadth rule and both endpoints were re-tested in EXP-061 (2026-09-22): always-bull is worse on 8/8 starts
over 26 years, and in 2025 — the last narrow, AI-led year — the blend was ~14 points better than always-bull.

**Combination.** Sleeve weights × blended sleeve allocation, summed per name, each name clipped at 15%, gross ≤ 1,
names below 0.5% dropped. The signal server publishes every member with a `probability` proportional to its
combined weight (top name = 0.95); the engine renormalises them.

**Deliberately not used** (tested; see [Research summary](#research-summary)): FMP "enhanced" data (price
targets, DCF, growth), short interest, EPS / revenue-surprise boosts, ML rankers, sentiment, VIX timing.

---

## Portfolio construction and risk

**Tranched book (live since 2026-09-09).** The account is four virtual sub-books of NAV/4. Every 5th trading day
one book is rebuilt to the current signals (so each book holds for 20 trading days); the other three are left
alone. Bookkeeping is per book (`data/ibkr_tranche_state.json`) and reconciled against the broker on every check.

| Control | Rule |
|---|---|
| **Leverage** | each rebuilt book is sized **closed-loop** to 1.49x × vol-scale × credit-gate of its NAV, in whole shares (four passes over integer quantities; the backtest sizes identically) |
| **Vol-scale** | `clamp(0.15×1.49 / realised 40-day NAV vol, 0.30, 1.0)` — de-risk only; window ends at the last **completed** close |
| **Credit gate** | HY-OAS ≥ 95th percentile of its expanding history → the rebuilding book goes **flat** (depth 0.00); FRED data, fail-safe 1.0 |
| **De-risk overlay** | on each rebuild day, any other book more than 5% above today's target is trimmed pro rata; never levers up |
| **Position cap** | 15% of the book's NAV per name |
| **Trailing stop** | 40% below the book's peak, **evaluated once in the last 10 minutes before the close** with the backtest's rule (peak from closes); sells only that book's slice |
| **Stock splits** | each trading day, before any check, that day's splits (Massive/Polygon reference) scale every book's shares by the ratio and its peak by the inverse |
| **Min trade** | existing holdings are not resized for changes below 0.3% of the book NAV |
| **Trade gates** | no trading on stale signals, on degraded feature coverage (floors: 85% for trend / 60d vol / 6-month return, 80% gross margin, 70% ROE), outside market hours, or on non-trading days |
| **Watchdog** | the engine force-restarts if its loop stalls > 6 minutes; ledger and peaks persist after every fill |
| **Kill switch** | `pm2 stop ibkr-engine` |

Rollback flags (`.env`, no comment lines): `IBKR_TRANCHES=1` (single book), `IBKR_STOP_AT_CLOSE=0` (old intraday
stops), `IBKR_SPLIT_CHECK=0`, `IBKR_OVERLAY_DOWN=0`, `MOM_EQUAL_WEIGHT=0`, unset `PROD_BULL_WEIGHTS` (70/21/9).

---

## Architecture

```
              Massive/Polygon bars ─┐   SSGA SPY/MDY/SPSM holdings ─┐   fundamentals + EDGAR ROE ─┐
                                    ▼                               ▼                             ▼
                  ┌──────────────────────────────────────────────────────────────────────────────────┐
                  │  signal_server.py (FastAPI :5001) → signal_builder.build_signals_v9              │
                  │  price cache (settle rule) · guards · features · 3 sleeves · breadth/UMD blend   │
                  │  rebuilt every 15 min in market hours; /signals, /health (stale + coverage flags) │
                  └───────────────────────────────┬──────────────────────────────────────────────────┘
                                                  │ GET /health, /signals
                       ┌──────────────────────────┴───────────────────────────┐
                       ▼                                                      ▼
        ┌───────────────────────────────┐                     ┌───────────────────────────────┐
        │ ibkr_engine.py  (LIVE money)  │                     │ server/tradingEngine.js       │
        │ 4 tranche books, closed-loop  │  FRED HY-OAS ──►    │ Alpaca PAPER mirror           │
        │ sizing, gates, stops at close │  credit_gate.py     │ single 20-day book, fractional│
        └───────────────────────────────┘                     └───────────────────────────────┘
                 Telegram alerts + read-only commands · PM2 on AWS EC2 · weekly AWS Backup snapshot
```

---

## Data status

Last full verification **2026-09-30** (`docs/LIVE_SYSTEM.md` → "Live data verification" and "parity audit").

| Input | Source | Refresh | Verified 2026-09-30 |
|---|---|---|---|
| **Daily prices** (1,541 symbols incl. ETFs) | Massive (Polygon) split-adjusted daily bars; yfinance only for names whose vendor history is short | full refetch after each session settles (17:50 restart); the pre-open build reuses those final bars; 15-min rebuilds in market hours | closes **equal IBKR's to the cent** on 80 names (all buys, all holdings, 40 random), no missing sessions; features equal an independent recomputation exactly |
| **Index membership** | SSGA daily holdings of SPY / MDY / SPSM (Wikipedia fallback, then last-good) | 06:00 weekdays | **equal the funds' holdings exactly** (503 / 400 / 603); the Wikipedia S&P 600 page had missed the September rebalance |
| **Fundamentals** (ROE, gross margin, debt/equity) | standardized quarterly fundamentals, newest whole row per company by report date; EDGAR ROE overlay for companies that filed after the last update | periodic dataset updates (last 2026-09-07); EDGAR patch 18:40 + reconciliation 18:55 | net income and equity **equal FMP's** for the same quarter; GM / D/E differ only by vendor definition (the backtest uses the same source as live) |
| **Credit spread** | FRED ICE BofA US High Yield OAS | 08:35 weekdays | **equals FRED**; gate off (11.8th percentile on 2026-09-30) |
| **Momentum-crash detector** | own price matrix (price-based UMD) | each build | matches the backtest's series (corr 0.998; crash/no-crash agree on 82/82 days) |
| **Breadth** | own feature map, index members only | each build | recomputed independently from raw closes |
| **Stock splits** | Massive (Polygon) reference splits | each trading day | — |
| **NAV history** (vol-scale) | IBKR, authoritative 16:05 close mark | daily | — |

Research-only inputs, **not read by live signals**: FMP enhanced data, options snapshots, VIX, Fama-French,
sentiment, the daily close archive (`data/price_archive`, correctly labelled from 2026-09-30; older files
quarantined in `_mislabeled_pre_2026-09-30/`).

---

## Data pipeline and guards

**Price cache.** A cached per-symbol file is reused only if it is < 18 h old **and** was written after the last
settled close (17:00 ET); a cached file whose last bar is older than the session most symbols end on is refetched
regardless of who wrote it. Evening builds therefore use the day's final bars, and the pre-open build needs no
vendor call.

**Signal-side guards** (each has a regression test): partial-session guard (today's bar is never used before 17:00
ET, nor when < 90% of names printed); vendor-hole fill (interior gaps ≤ 3 sessions); splice guard (a one-day price
ratio > 4x means two securities under one ticker; history truncated); minimum history 21 bars (the backtest's
rule); quality gate; feature-coverage guard (the engine refuses to trade below its per-feature floors); EXP-059
flags logged at start-up.

**Nightly refresh** (`scripts/refresh_data.py`): a table of steps, each run in a forked child with a hard time
budget, retries, and output-freshness post-conditions; alerts say whether a failure touches **live** inputs or
research data only.

| Time (ET) | Job |
|---|---|
| 06:00 weekdays | `scrape_sp1500.py` — S&P 500/400/600 membership (SSGA → Wikipedia → last-good), sector map |
| 08:35 weekdays | `credit_gate.py` — FRED HY-OAS percentile |
| 09:00 / 13:00 / 19:30 weekdays | `data_freshness_check.py` |
| 17:30 weekdays, 21:00 Sun | `scripts/refresh_data.py` — FMP research data, VIX, fundamentals fallback, options, snapshots, close archive, Fama-French, data-gap repair; restarts the signal server |
| 17:50 weekdays, 21:20 Sun | `pm2 restart signal-server` (refetches the settled session) |
| 18:40 / 18:55 weekdays | EDGAR fundamentals patch / overlay reconciliation |
| Sat 00:30 | AWS Backup snapshot of the instance |

---

## Live vs backtest parity

Audited 2026-09-30 on the validated universe and on live data.

**Identical:** feature formulas; sleeve code (shared module); the probability → weight mapping; the 0.5% weight
floor and 15% clip; closed-loop whole-share sizing per book; the 0.3% min-trade band; the tranche schedule;
inputs that are empty on both sides (short interest, earnings boosts, price targets); the crash detector.

**Fixed on 2026-09-30** (details and evidence in `docs/LIVE_SYSTEM.md`):

| Gap | Effect | Fix |
|---|---|---|
| Trailing stops were evaluated **intraday** with peaks from intraday highs; the backtest uses closes | EXP-062, same engine: −4.88pp CAGR (2018-25) and −1.56pp (2001-25), worse on 12/12 starts on both, ~60% more stop-outs, no drawdown benefit | stops once at the close, backtest rule |
| No **stock-split** handling — a 2:1 split read as −50% and would fire every book's stop | live-only failure mode | daily split calendar scales shares and peaks |
| Live dropped names with < 252 bars; the backtest keeps them from 21 bars | backtest's low-vol sleeve held such a name on 7.8% of rebalance dates | 21-bar minimum |
| Breadth also counted ~22 ETFs | ~0.4pp of breadth | members only |
| Evening price files predated the close; a renamed-ticker repair wrote histories one session short; membership lagged | stale evening signals; 15 tickers silently unranked; 2 buys wrong on 2026-09-30 | settle rule, content guard, SSGA membership |

**Measured and accepted:** live trades at the next open on the prior close's signals while the backtest trades at
that close (bounded by a one-day-lag test at −0.35pp CAGR); the backtest's breadth set also contains past and future
members (±1.5pp on most days); live prices are split-adjusted price-only while the backtest is total-return
(identical buy list on 2026-09-29).

---

## Operations

PM2 services on the EC2 host:

| Service | What it runs |
|---|---|
| `ibkr-engine` | `start_ibkr_engine.sh` → `ml_service/ibkr_engine.py` (live account via IB Gateway, port 4001) |
| `signal-server` | `start_signal_server.sh` → `ml_service/signal_server.py` (port 5001) |
| `trading-engine` | `server/index.js` → Alpaca paper engine |
| `pm2-logrotate` | log rotation |

```bash
pm2 list                                   # service status
curl -s localhost:5001/health              # last update, is_stale, coverage, coverage_ok
pm2 logs ibkr-engine --lines 50            # engine log (rebalances, fills, stops, splits)
pm2 restart signal-server                  # rebuild signals (start script exports .env)
pm2 stop ibkr-engine                       # kill switch
```

**Telegram:** fills, stops, splits, gate changes, data-refresh failures (live vs research-only), coverage alarms
and the daily summary; read-only commands for the allow-listed users (`/daily`, `/pnl`, status), actions owner-only.

**Deploy:** commit and test locally → push → on the host `git pull` (or `git pull --ff-only <bundle>`) → run
`ml_service/tests` on the host → restart only the affected service, outside market hours. Never put comment lines
in `.env` (both start scripts `export $(cat .env | xargs)`).

---

## Tests

`ml_service/tests/` — plain-Python regression suites, one per failure mode found in production; run each with
`python3 tests/<file>.py` from `ml_service/`.

| Suite | Pins |
|---|---|
| `test_tranche_engine` | books, reconciliation, bounded sells, sizing, cadence, crash/retry recovery |
| `test_stop_at_close` · `test_split_handling` | close-window stops (incl. half days) · split adjustment, fail-safe |
| `test_vol_window` · `test_overlay_trims` | completed-session vol window · de-risk overlay parity with the clean room |
| `test_cache_settle` · `test_cache_depth_guard` · `test_vendor_hole` · `test_splice_guard` · `test_partial_row_guard` | price-cache validity and the price-data guards |
| `test_universe_parity` · `test_fund_rowwise` · `test_mom_equal_weight` · `test_coverage_gate` | backtest-parity rules and the coverage gate |
| `test_sp1500_scrape` · `test_refresh_runner` · `test_freshness_edgar_rule` · `test_daily_fallback` | membership scraper, nightly runner, freshness rules, `/daily` |
| `lint_engine_names` | undefined-name check for `ibkr_engine.py` and `credit_gate.py` |

---

## Project structure

```
AutoTraderBot/
├── README.md · STRUCTURE.md · docs/LIVE_SYSTEM.md (operating ledger) · docs/BACKTEST_LIVE_PARITY.md
├── start_ibkr_engine.sh · start_signal_server.sh · ecosystem.config.js
├── server/                    Alpaca paper engine (Node.js): tradingEngine.js, index.js, journal.js
└── ml_service/
    ├── ibkr_engine.py         LIVE engine: tranches, sizing, gates, stops, splits, Telegram
    ├── signal_server.py       LIVE signal API (FastAPI)
    ├── signal_builder.py      LIVE signal harness: guards, features, fundamentals, blend
    ├── strategies/multi_strategy_engine.py   THE strategy (sleeves + regime weights), shared with the backtest
    ├── massive_data_provider.py   price bars + cache rules
    ├── scrape_sp1500.py · sp1500_membership.py   membership
    ├── credit_gate.py · data_freshness_check.py · live_config.py
    ├── scripts/               refresh_data.py, EDGAR patch/reconcile, universe builders, ops tools
    ├── main_production_backtest.py   backtester (shares the strategy module)
    ├── research/              VERIFY2_cleanroom.py, EXP0xx experiments, LOG.md, BUGS.md
    ├── tests/                 regression suites
    └── data/                  (not in git) caches, datasets, state files, archives
```

---

## Research summary

`ml_service/research/LOG.md` holds every experiment with its protocol (≥ 12 monthly starts, sign consistency,
both horizons) and verdict. Highlights:

- **Shipped:** tranched rebalance (risk shaping); credit gate (HY-OAS p95, depth 0.00); EXP-059 package
  (equal-weight momentum, 80/15/5 bull split, de-risk-only overlay); real-time price-based crash detector; EDGAR ROE
  freshness overlay; close-based stops (EXP-062 parity fix).
- **Kept after re-testing:** the breadth blend and its thresholds (EXP-061); 1.49x leverage; the 40% stop.
- **Rejected:** ML rankers (IC ≈ 0.01), fundamental factor blends inside momentum, short interest, FMP enhanced
  data, news/NLP sentiment, weather and attention data, short strategies, crypto momentum, VIX gates, timing rules
  for leverage, second return engines (EXP-060; a small gold line is the only survivor).

---

## Known limitations

- **Regime dependence.** The strategy is momentum-led; through-cycle drawdowns are deep (see the table above) and
  leverage amplifies them. Margin calls are not modeled.
- **Whole shares on a small account.** Each book is NAV/4, so very expensive names (e.g. ~$1,000+ shares) can
  round to zero in every book (LITE and SNDK in Sep 2026). Measured at real historical prices with the account held
  at today's ~$62K, this costs ≈1.6pp/yr on 2018-26 and ≈0.5pp/yr on 2001-26 versus fractional shares — more than
  the figures above include, because the backtest's account grows past this size. An account-level rounding switch
  that recovers most of it (floor or round the four books' sum; better on all 24 start dates of both horizons) is
  built and **off by default** (`IBKR_ACCOUNT_ROUNDING`, research cycle 61).
- **Credit-gate overlay (fix pending deploy, 2026-10-02).** In a gate event the backtest flattens all four books on
  the next tranche day; the live overlay skipped a zero target, so the other books would wait for their own
  rebuilds. Fixed in code and tested; goes live with the next engine deploy.
- **Execution at the open.** Live fills one overnight after the signal close (bounded at −0.35pp/yr).
- **Fundamentals freshness.** The fundamentals dataset is updated periodically; between updates only ROE is refreshed
  (EDGAR overlay).
- **Vendor quirks.** Per-ticker vendor history can splice two securities after a ticker change (e.g. BNY); such
  names are truncated and treated as short-history until clean history accrues.
- **Paper mirror ≠ live.** The Alpaca engine runs the same signals as a single 20-day book on a ~$1.5M paper
  account (whole-share rounding negligible) with intraday stops; since 2026-10-01 each rebalance resizes held names
  to target in both directions, like the backtest (before, it never topped them up and drifted to 0.83x invested).
  Comparing it with IBKR measures tranching, small-account rounding and stop timing — not two different strategies.
- **Statistical confidence.** Improvements are validated across many start dates, but several edges are small
  relative to start-date dispersion; plan with ranges, not point estimates.

---

## Change history

- **2026-09-30** — Data-layer verification and parity audit: settle-aware price cache and stale-content guard;
  renamed-ticker repair fixed; membership from SSGA fund holdings with verified SSL; whole-row fundamentals; stops at
  the close; stock-split handling; 21-bar minimum history; member-only breadth; completed-session vol window;
  nightly refresh runner fixes; close archive relabelled. Book 3 rebuilt cleanly.
- **2026-09-22** — EXP-059 package shipped (equal-weight momentum, 80/15/5 bull split, de-risk-only overlay).
- **2026-09-08** — Tranched rebalance deployed (4 books, 5-day stride), sleeves 70/21/9 base, credit gate depth 0.00.
- **2026-08 → 09-08** — Live-only failure fixes: gross-margin bound and partial-session guard (Aug), coverage gate
  and vendor-hole fill (Sep 2), splice guard (Sep 8).
- **2026-07** — Closed-loop sizing, vol-scaling live, EDGAR ROE overlay (07-13), credit gate (07-18), live sleeves
  aligned to the full S&P 1500 (07-25), real-time price-based crash detector.

Earlier history and every incident: `docs/LIVE_SYSTEM.md`.

---

## License

MIT. Not financial advice. Past performance — and backtests in particular — do not guarantee future results.
