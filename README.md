# AutoTrader v12

Autonomous multi-factor equity trading system running on AWS EC2. Trades the S&P 1500 universe through Interactive Brokers and Alpaca with 1.5x margin leverage. Fully automated data pipeline, signal generation, order execution, risk management, and monitoring.

**Honest backtest (deployed config, no look-ahead, start-day averaged, 10bps costs). Two horizons — read both:**

| Metric (at 1x) | Recent regime (2018-2025) | Through-cycle (2000-2025) |
|--------|---------------------------|---------------------------|
| **CAGR** | 22.0% +/- 2.5% | ~11-13.5% |
| **Sharpe** | 0.90 | ~0.63 |
| **Max Drawdown** | -28.3% | **-59%** (GFC); ~-47% with the live vol-scaling overlay |

> **Why two numbers, and which to believe.** The 22% is real but it's a *favorable-regime* number — 2018-2025 was a momentum-friendly, mega-cap-tech-led market with no sustained bear. Extend the same strategy back across **26 years** (including the 2000-02 dot-com bust and the 2008 GFC) and the honest through-cycle CAGR is **low-teens at 1x, with a ~-59% drawdown** in the GFC (the 15% low-vol sleeve barely dents a real deleveraging bear; the live vol-scaling overlay cuts it to ~-47% for ~2pp of CAGR). The pre-2010 market is **structurally different** (different liquidity, sector mix, and no AI/mega-cap momentum regime), so treat the 26-year figure as a *through-cycle stress lens*, not a prediction — and the 22% as the *good-times* case. The truth forward is regime-dependent and lands between them. **Plan with the conservative number; the −59% is why leverage stays modest** (a −59% 1x drawdown is ruin at 1.49x).

> **These numbers reflect the ACTUAL deployed config** (enhanced data OFF, short interest OFF), verified June 2026 by re-running the backtest with the exact data the live system uses. An earlier claim of 25.4% CAGR / 1.00 Sharpe was **inflated** — it came from a run with enhanced data + short interest *enabled* using static (look-ahead-biased) snapshots, neither of which is used in production. See [Backtest Results](#backtest-results).

> **Note on leverage:** The leverage numbers are estimates — the backtest runs at 1x only. Live runs at ~1.49x effective (IBKR `LEVERAGE=1.8`, Alpaca `MAX_CASH_DEPLOY_PCT=1.6` — different config numbers, same effective leverage because of account-size/rounding differences). At 1.49x, returns scale by ~1.49x minus margin interest (~5-6% on the borrowed portion), and drawdowns amplify by ~1.49x. Margin-call risk exists if the portfolio drops below maintenance margin (~25%).

---

## Table of Contents

- [Strategy](#strategy)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Signal Generation](#signal-generation)
- [Risk Management](#risk-management)
- [Rebalance Logic](#rebalance-logic)
- [Configuration](#configuration)
- [Backtest Results](#backtest-results)
- [Research Findings](#research-findings)
- [Project Structure](#project-structure)
- [Setup & Deployment](#setup--deployment)
- [Operations & Monitoring](#operations--monitoring)
- [Known Limitations](#known-limitations)

---

## Strategy

Three-sleeve momentum/value/quality approach. Each sleeve independently selects stocks, then weights are blended proportionally.

### Momentum Sleeve (50% weight)

Selects the top 5 stocks by skip-month momentum (12-month return minus last month's return). This is the classic Jegadeesh & Titman (1993) momentum factor, which avoids the short-term reversal effect.

**Scoring:**
```
base_score = ret_252d - ret_20d    (skip-month momentum)
```

**Filters:**
- Must be above 200-day SMA (trend filter — excludes stocks in downtrends)
- Consolidation boost: if vol_20d < 25% annualized AND score > 20%, multiply by 1.15x
- Quality boost: if ROE > 15%, multiply by 1.05x

**Position sizing:** Signal-proportional — higher-conviction stocks get larger allocations, capped at 15% per position.

### Value Sleeve (35% weight)

Selects top 10 stocks by a composite of profitability + valuation + momentum:

```
score = -ret_252d * 0.30 + gross_margin * 0.25 + min(ROE, 0.50) * 0.25
```

**Filters:**
- ROE > 5%, gross margin > 15%
- Not in severe downtrend (dist_sma200 > -15%)
- Debt/equity < 3.0

### Low-Vol Quality Sleeve (15% weight)

Selects top 10 stocks by z-score composite across the S&P 1500:

```
composite = mean(z_inv_vol, z_gross_margin, z_inv_leverage, z_momentum_6m, z_inv_ev_rev, z_fwd_earnings_yield)
```

Acts as a defensive anchor — no trend filter, works in both bull and bear markets.

### Weight Blending

In normal conditions, the three sleeves combine at their stated weights. During stress:

- **UMD crash** (Fama-French momentum factor 20-day sum < -0.05): Shifts to 15% momentum / 45% value / 30% lowvol / 10% sector
- **Low breadth** (< 35% of stocks above 50-day SMA): Blends toward bear weights (10% momentum / 20% value / 60% lowvol)

---

## Architecture

```
                        ┌──────────────────────────────────────┐
                        │          Signal Server               │
                        │       (FastAPI, port 5001)            │
                        │                                      │
                        │  Massive/Polygon ──→ Daily prices     │
                        │  WRDS Compustat  ──→ ROE, margins,    │
                        │                      debt/equity      │
                        │  WRDS IBES      ──→ EPS surprise      │
                        │  Fama-French    ──→ UMD crash regime   │
                        │  Wikipedia      ──→ SP1500 membership  │
                        │                                      │
                        │  ┌────────────────────────────────┐  │
                        │  │ _compute_features_from_raw()   │  │
                        │  │ strategy1_momentum_reversal()  │  │
                        │  │ strategy5_lowvol_quality()     │  │
                        │  │ _strategy_value()              │  │
                        │  └──────────────┬─────────────────┘  │
                        │                 │                     │
                        │     ~22 BUY signals + weights         │
                        └────────────┬─────────────────────────┘
                                     │ GET /signals (JSON)
                        ┌────────────┴────────────┐
                        ↓                         ↓
              ┌──────────────────┐    ┌──────────────────┐
              │  Alpaca Engine   │    │   IBKR Engine     │
              │  (Node.js)       │    │   (Python/asyncio)│
              │                  │    │                   │
              │  Paper trading   │    │  Paper or Live    │
              │  Port 3000       │    │  Port 4001 (live) │
              │                  │    │  Port 4002 (paper)│
              │  1.5x leverage   │    │  1.5x leverage    │
              │  40% trail stop  │    │  40% trail stop   │
              │  20-day rebal    │    │  20-day rebal     │
              │  15% position cap│    │  15% position cap │
              │  Trim on rebal   │    │  Trim on rebal    │
              └──────────────────┘    └──────────────────┘

              All services managed by PM2 on AWS EC2 (Ubuntu)
```

### Service Communication

1. **Signal server** generates fresh signals every 15 minutes during market hours
2. **Trading engines** poll `/signals` to get current BUY list and weights
3. **On rebalance day** (every 20 trading days): engines sell dropouts, trim oversized, buy new positions, top up undersized
4. **Between rebalance days**: only trailing stops fire (40% from peak), positions drift
5. **Telegram bot** sends trade alerts, error notifications, and daily portfolio summary

---

## Data Pipeline

All data updates are automated via cron jobs (Mon-Fri, Eastern Time):

| Time | Script | What it does | Frequency |
|------|--------|-------------|-----------|
| 6:00 AM | `scrape_sp1500.py` | Scrape SP500/SP400/SP600 membership from Wikipedia, update sector map from FMP | Daily |
| 5:30 PM | `refresh_data.py` | Refresh FMP fundamentals (income, ratios, earnings for ~1,500 stocks), VIX cache, Fama-French factors from Ken French's website, options snapshots from Polygon, price archive, fundamentals cache flush, journal cleanup | Daily |
| 5:50 PM | `pm2 restart signal-server` | Restart signal server to pick up fresh data from 5:30 refresh | Daily |
| Quarterly | Manual WRDS upload | Re-download Compustat + IBES parquets from WRDS | ~Every 3 months |
| Quarterly | Log cleanup | Delete logs older than 90 days | Automatic |

### Data Sources

| Source | What it provides | Update method |
|--------|-----------------|---------------|
| **Massive/Polygon** | Daily OHLCV prices for ~1,500 stocks | Live API, refreshed every 15 min |
| **WRDS Compustat** | ROE, gross margin, debt/equity, revenue growth (via `seqq`, `saleq`, `cogsq`, `niq`) | Quarterly manual upload, uses `rdq` (report date) for point-in-time |
| **WRDS IBES** | Earnings surprise, consensus estimates | Quarterly manual upload, uses `ANNDATS` for point-in-time |
| **Fama-French** | mktrf, smb, hml, rmw, cma, umd (momentum) factors | Daily auto-download from Ken French website |
| **FMP** | Fundamentals fallback, VIX, sector classification | Daily auto-refresh via API |
| **Wikipedia** | SP500, SP400, SP600 constituent lists | Daily auto-scrape |

### What is NOT used (tested, hurts returns)

| Data | Why disabled |
|------|-------------|
| FMP enhanced data (price targets, DCF, financial growth, analyst estimates) | Point-in-time backtest proved it hurts CAGR by -2pp (dilutes momentum signal) |
| Short interest (Ortex, Compustat, FINRA) | Hurts CAGR by -3pp |
| ML cross-sectional features | IC=0.014, too weak to beat hand-tuned factors |

---

## Signal Generation

The signal builder (`signal_builder.py`) produces signals through this pipeline:

1. **Fetch prices** for ~1,500 SP1500 stocks (550 calendar days of history for SMA200)
2. **Compute technical features**: returns at 7 horizons (5d-252d), volatility at 3 horizons (10d/20d/60d, annualized), SMA distances (50d/200d), RSI-14
3. **Load fundamentals** from WRDS Compustat: ROE (`niq*4/seqq`), gross margin (`(saleq-cogsq)/saleq`), debt/equity (`(dlttq+dlcq)/seqq`)
4. **Run three strategy functions** that each return `{symbol: weight}` dicts
5. **Check UMD crash regime**: if Fama-French momentum factor 20-day sum < -0.05, shift to defensive weights
6. **Breadth blend**: interpolate between bull and bear weight configs based on % of stocks above 50-day SMA
7. **Combine** all sleeve picks with blended weights
8. **Apply 15% position cap**, normalize to sum to 1.0
9. **Output** ~22 BUY signals with signal-proportional weights

### Signal Output Format

```json
{
  "signals": [
    {"symbol": "SNDK", "probability": 0.95, "rank": 1, "is_top_5": true, "signal": "BUY"},
    {"symbol": "MU",   "probability": 0.54, "rank": 2, "is_top_5": true, "signal": "BUY"},
    ...
    {"symbol": "AAPL", "probability": 0.00, "rank": 500, "is_top_5": false, "signal": "HOLD"}
  ]
}
```

The `probability` field represents the combined sleeve weight (normalized 0-0.95). The trading engines convert this to dollar allocations: `target_$ = equity * leverage * (prob_i / sum(all_buy_probs))`, capped at 15% per position.

---

## Risk Management

| Control | Value | How it works |
|---------|-------|-------------|
| **Position cap** | 15% max | No single stock can exceed 15% of portfolio. Enforced in signal builder and both trading engines. |
| **Trailing stop** | 40% from peak | Peak price tracked per position (persisted to disk on IBKR). If price drops 40% from peak, position is sold immediately. Checked every 10 minutes (IBKR) or every cycle (Alpaca). |
| **UMD crash detection** | umd_20d < -0.05 | When the Fama-French momentum factor crashes, sleeve weights shift from momentum-heavy to value/lowvol-heavy. |
| **Breadth regime** | < 35% above SMA50 | Blends toward defensive weights when market breadth is poor. |
| **Stale signal rejection** | is_stale flag | IBKR engine refuses to trade if signal server reports stale data. Sends Telegram alert. |
| **Order sanity check** | 0 < qty < 10,000 | IBKR engine rejects absurd order quantities. |
| **Auto-reconnect** | 30s retry | IBKR engine reconnects on Gateway disconnect, cancels all pending orders to prevent duplicates. |
| **Kill switch** | `pm2 stop ibkr-engine` | Instantly stops all trading. |

---

## Rebalance Logic

Every **20 trading days**, the engine performs a full portfolio reconstruction (matching the backtest exactly):

1. **Sell** all positions not in the current signal set (no minimum hold period on rebalance day)
2. **Trim** positions that are >20% above target weight (sell the excess, not the whole position)
3. **Buy** new positions that appeared in signals
4. **Top up** existing positions that are undersized
5. **Mark rebalance complete** — next rebalance in 20 trading days

**Between rebalance days:** No buying, no selling. Only trailing stops fire. Positions drift naturally (lets winners run).

**On restart:** Both engines allow immediate rebalance on first boot to ensure the portfolio converges to target. The IBKR engine only rebalances during market hours (9:30 AM - 4:00 PM ET).

---

## Configuration

All parameters are synchronized between backtest (`main_production_backtest.py`), signal builder (`signal_builder.py`), Alpaca engine (`tradingEngine.js`), and IBKR engine (`ibkr_engine.py`):

| Parameter | Value | Where set |
|-----------|-------|-----------|
| Momentum weight | 50% | `signal_builder.py` STRATEGY_CONFIG_BULL |
| Value weight | 35% | `signal_builder.py` STRATEGY_CONFIG_BULL |
| Low-vol weight | 15% | `signal_builder.py` STRATEGY_CONFIG_BULL |
| Momentum picks | Top 5 | `signal_builder.py` build_signals_v9(top_n=5) |
| Rebalance period | 20 trading days | `tradingEngine.js` REBAL_INTERVAL_CYCLES / `ibkr_engine.py` REBAL_DAYS |
| Trailing stop | 40% from peak | `tradingEngine.js` TRAILING_STOP_PCT / `ibkr_engine.py` TRAILING_STOP |
| Position cap | 15% per stock | `signal_builder.py` / `tradingEngine.js` MAX_POSITION_PCT / `ibkr_engine.py` POSITION_CAP |
| Leverage (effective) | ~1.49x both accounts | IBKR `LEVERAGE=1.8` (small $30K acct, heavy share rounding); Alpaca `MAX_CASH_DEPLOY_PCT=1.6` ($1.3M acct, light rounding). Different numbers, same ~1.49x effective. |
| Min trade (top-up skip) | 1% of portfolio, existing holdings only | `ibkr_engine.py` MIN_TRADE_PCT (new positions never skipped) |
| Bull sleeve weights | 50 / 35 / 15 (mom/val/lowvol) | `signal_builder.py` STRATEGY_CONFIG_BULL / backtest config |
| Bear sleeve weights | 10 / 30 / 50 / 10 (mom/val/lowvol/sector) | `signal_builder.py` STRATEGY_CONFIG_BEAR (backtest parameterized to match) |
| UMD-crash weights | 15 / 45 / 30 / 10 | identical in both `signal_builder.py` and `main_production_backtest.py` |
| Universe | SP1500 (~1,500) | `scrape_sp1500.py` → `sp1500_members.json` |
| Short interest | Disabled (hurts ~3pp) | `signal_builder.py` (`_si_change_rank = {}`) |
| Enhanced data | Disabled (PIT version hurts; static is look-ahead) | `signal_server.py` (passes None to build_signals_v9) |
| Market data | Real-time (IBKR streaming bundle, June 2026) | `ibkr_engine.py` `reqMarketDataType(1)` |
| Transaction costs | 10bps round-trip | `main_production_backtest.py` COST_BPS + SLIPPAGE_BPS |
| Fundamentals source | WRDS Compustat (seqq, not ceqq) + IBES | `signal_builder.py` _fill_fundamentals() |

### Environment Variables

```env
# Broker credentials
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# IBKR (port 4001=live, 4002=paper)
IB_PORT=4002
IB_HOST=127.0.0.1

# Data providers
FMP_API_KEY=...
MASSIVE_API_KEY=...

# Alerts
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Optional
SIGNAL_SERVER_PORT=5001
```

---

## Backtest Results

### Methodology

- **Universe:** S&P 1500 (point-in-time membership from WRDS)
- **Period:** 2018-2025 primary (curated universe); 2000-2025 through-cycle stress test on the long-history universe (see [26-Year Through-Cycle Results](#26-year-through-cycle-results-2000-2025-at-1x-deployed-config))
- **Look-ahead prevention:** All snapshot data (fin_growth, ev_data, estimates, price_targets, revenue_surprise, beat_streak, earnings_signals, short interest) is cleared before backtesting
- **Transaction costs:** 10bps round-trip (5bps commission + 5bps slippage)
- **Robustness:** 7-day start-day averaging to control for start-date sensitivity
- **Leverage:** 1x only (no leverage in backtest)
- **Fundamentals:** WRDS Compustat with `rdq` (report date) for point-in-time, using `seqq` for equity (matching live exactly)

### Results (2018-2025, at 1x) — DEPLOYED CONFIG

| Metric | Value |
|--------|-------|
| **CAGR** | 22.0% +/- 2.5% |
| **Sharpe Ratio** | 0.90 |
| **Max Drawdown** | -28.3% |

### Data-condition verification (June 2026)

The backtest was re-run with the v12 config under 4 data conditions to confirm what's deployed and trace an earlier inflated claim. All start-day averaged, same params:

| Condition | CAGR | Sharpe | MaxDD | |
|-----------|------|--------|-------|--|
| **A. DEPLOYED (enhanced OFF, SI OFF)** | **22.0%** | **0.90** | **-28.3%** | ← what live actually runs |
| B. enhanced ON, SI OFF | 23.6% | 0.98 | -33.9% | static enhanced = look-ahead bias |
| C. enhanced OFF, SI ON | 19.2% | 0.82 | -31.7% | confirms SI hurts (-2.8pp) |
| D. EVERYTHING ON | 23.9% | 1.00 | -32.7% | source of the old "25.4%/1.00" claim |

**Takeaway:** The honest deployed number is condition A (22.0% / 0.90). The previously reported 25.4% / 1.00 matches condition D — enhanced data + short interest both enabled with static (look-ahead) snapshots. That's inflated and not what trades. Disabling short interest is confirmed correct (it hurts even without look-ahead: 19.2% vs 22.0%). Enhanced data only "helps" via look-ahead; the point-in-time version hurts (hence disabled).

### Year-by-Year (at 1x)

> Note: the year-by-year figures below are from the earlier (pre-correction) run and are directionally indicative but slightly optimistic. The corrected full-period number is 22.0% CAGR (see verification table above).

| Year | Strategy | S&P 500 | vs SPY |
|------|----------|---------|--------|
| 2018 | +16.2% | -5.2% | +21.4pp |
| 2019 | +13.5% | +31.1% | -17.6pp |
| 2020 | +26.1% | +17.3% | +8.8pp |
| 2021 | +23.5% | +30.5% | -7.0pp |
| 2022 | -5.2% | -18.6% | +13.5pp |
| 2023 | +32.5% | +26.7% | +5.8pp |
| 2024 | +49.5% | +25.6% | +23.9pp |
| 2025 | +17.2% | +18.0% | -0.8pp |

### 26-Year Through-Cycle Results (2000-2025, at 1x, deployed config)

Run on the long-history universe (`sp1500_universe_2000.pkl`, 1,743 names back to 2000), 3-start-day averaged. This window contains the two sustained bears the 2018-2025 sample lacks — the dot-com bust and the GFC — and is the real stress test.

| Metric | Baseline | + vol-scaling overlay (live) |
|--------|----------|------------------------------|
| **CAGR** | ~13.5% (11.0% canonical start) | ~9-11% |
| **Sharpe** | 0.63 | 0.54 |
| **Max Drawdown** | **-59%** | **-47%** |

**Crash-window drawdowns (1x):**

| Window | Drawdown | Vol-scaling helped? |
|--------|----------|---------------------|
| Dot-com (2000-02) | -29% | No (slow grind, low realized vol never triggered de-risk) |
| **GFC (2008-09)** | **-59% → -47%** | Yes (fast high-vol crash) |
| COVID (2020) | -46% → -31% | Yes |
| 2022 bear | -29% | — |

**Two findings from the long horizon:**

1. **The recent 22% is regime, not edge.** True through-cycle CAGR is low-teens with brutal (-47% to -59%) crash drawdowns. Momentum gets crushed in a real deleveraging bear regardless of the low-vol sleeve. The vol-scaling overlay is validated against a real GFC (cuts the worst drawdown ~12pp) — but only for fast, high-vol crashes, not slow grinds.
2. **Sleeve-weight tweaks don't add edge.** Over 26 years, deployed 50/35/15, 60/40 mom/val, and pure-momentum are **statistically identical** (~13.5% CAGR, Sharpe 0.61-0.63, -59% MaxDD). A 60/40 tilt looks better *only* in-sample (2018-2025: 20.9% vs 18.6%) but is **worse out-of-sample** (2000-2017, which the config was never tuned on: 7.3% vs 7.7%, Sharpe 0.42 vs 0.44). Textbook recent-regime overfit — the deployed config is retained. See `research/phase6_6040_longhorizon.py`.

---

## Research Findings

Extensive testing conducted May 2026 across ML, alternative factors, and portfolio construction methods. All tests used the same honest backtesting methodology (no look-ahead, start-day averaging).

### ML Approaches

| Approach | Metric | Finding |
|----------|--------|---------|
| LightGBM cross-sectional ranker | OOS IC = +0.014 | Positive in 7/7 years but too weak to beat hand-tuned factors. Top features: net_margin, momentum acceleration, GP/assets |
| XGBoost LambdaRank | NDCG@8 | Zero impact on returns |
| Crash risk model | AUC = 0.673 | Identifies risky stocks (top 10% has 2-3x crash rate) but filtering them kills returns because high-momentum stocks ARE volatile |
| Regime detection (FF + FRED macro) | 58% accuracy | Marginal — not enough data (478 samples) for reliable prediction |

### Alternative Factors

| Factor | Impact on CAGR |
|--------|---------------|
| Accruals (Sloan 1996) | -10pp |
| Gross profitability (Novy-Marx) | -15pp |
| Book-to-market | -12pp |
| Cash flow yield | -13pp |
| Earnings quality | -11pp |
| All Compustat factors combined | -5pp |
| Short interest (Compustat) | -3pp |
| FMP enhanced data (point-in-time) | -2pp |

**Why?** Every fundamental factor **dilutes the momentum signal**. Momentum and value are negatively correlated (Fama-French). Blending them in the same scoring function weakens both. The correct approach is separate sleeves with separate allocations — which is what v12 does.

### Position Sizing

| Method | Avg CAGR |
|--------|----------|
| **Signal-proportional** | **24.9%** |
| Equal weight | 20.2% |
| Blended (60% signal / 40% inv-vol) | 20.2% |
| Inverse-volatility | 12.2% |
| Risk-adjusted momentum | 10.4% |
| Quality-gated | 11-15% |

### Concentration Analysis (n=5 vs broader)

Bootstrap confidence intervals show the Sharpe difference between n=5 and n=30 is **not statistically significant** (Sharpe diff CI: [-0.76, +1.19]). On 25-year data, n=5 (16.9% CAGR) and n=10 (16.6% CAGR) are essentially identical. The concentration choice is a **risk decision, not a return decision** — n=5 has higher tail risk (max single position drifts to 44%) but statistically indistinguishable returns.

| n | 25yr CAGR | 25yr Sharpe | Max Position Drift |
|---|-----------|-------------|-------------------|
| 5 | 16.9% | 0.79 | 44% |
| 10 | 16.6% | 0.80 | 37% |
| 15 | ~16% | ~0.80 | 29% |

### NLP / Alternative Data

| Signal | Gate 1 (Predictive?) | Gate 2 (Orthogonal?) | Gate 3 (Binds on picks?) |
|--------|---------------------|---------------------|------------------------|
| Lazy Prices filing similarity (Cohen et al.) | IC=+0.079, CI: [+0.010, +0.146] — YES | Correlation -0.017 — YES | **0% binding rate — FAILS** |

The filing similarity signal is real and orthogonal but **never fires on momentum picks** — companies with strong momentum don't gut their filing language. Signals that flag losers are redundant with momentum by construction. Useful signals must create dispersion **within winners**.

### Uncorrelated Strategy Combinations

| Strategy | CAGR | Sharpe | Correlation with Momentum |
|----------|------|--------|--------------------------|
| Our momentum | 27.6% | 1.15 | — |
| GLD trend-following | 6.6% | 0.87 | **-0.023** |
| SPY index trend | 6.5% | 0.58 | +0.48 |
| Short-horizon mean reversion | 6.2% | 0.35 | +0.47 |

GLD trend has near-zero correlation — the diversification thesis is validated. A 70/30 momentum/GLD combo cuts MaxDD from -18% to -12% but costs ~6pp CAGR. The proper version is a diversified managed-futures sleeve (gold + commodities + FX + rates) at 10-15% allocation — future research.

### Key Conclusion

The strategy is at its efficient frontier with available data. Every signal-level improvement either dilutes momentum (fundamentals, ML) or is redundant (NLP distress detection). The remaining gains live at the **portfolio level** — combining uncorrelated return streams (cross-asset trend) rather than improving the stock selection engine. The strategy resists improvement because momentum is already doing most of the work that other signals would do.

---

## Project Structure

### File roles at a glance

The codebase has four kinds of files. The thing that ties it together: the **strategy logic lives once** in `strategies/multi_strategy_engine.py` (the momentum/value/low-vol sleeves + regime weights), and is imported by *both* the live signal builder and the backtester — so they can't drift apart.

```
        strategies/multi_strategy_engine.py   ← THE strategy (single source of truth)
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
  signal_builder.py    main_production_backtest.py
   (LIVE signals)       (BACKTEST)
```

**1. LIVE — trades real money (runs 24/7 on AWS under PM2):**

| File | What it does |
|------|--------------|
| `strategies/multi_strategy_engine.py` | **The strategy brain.** Momentum/value/low-vol sleeve functions + `PROD_WEIGHTS_*` regime weights. Shared with the backtest. |
| `signal_builder.py` | Live harness — runs the brain on live data, applies regime blend + 15% cap, outputs ~22 ranked BUY signals. |
| `signal_server.py` | FastAPI server (port 5001). Refreshes signals every 15 min; serves `/signals` and `/health`. |
| `ibkr_engine.py` | IBKR **live** engine (Python/asyncio) — trades the $30K account. |
| `server/tradingEngine.js` | Alpaca **paper** engine (Node.js) — the $1.3M shadow account. |
| `massive_data_provider.py`, `wrds_data_provider.py`, `scrape_sp1500.py`, `sp1500_membership.py`, `sp500_history.py` | Data + universe (Polygon prices, WRDS Fama-French/FRED, S&P 1500 membership). |
| `scripts/refresh_data.py` | Daily 5:30 PM cron — refreshes prices, fundamentals, VIX, Fama-French. |

**2. BACKTEST — validates the strategy:**

| File | What it does |
|------|--------------|
| `main_production_backtest.py` | **THE canonical backtester.** Imports the same strategy brain as live. Produces the official 22.0% number. |
| `research/v12_ground_truth.py` | Reproduces the deployed number cleanly (enhanced/SI cleared). Run this to verify the headline figure. |
| `research/locate_v12_number.py`, `research/compare_v12_bear_configs.py` | Audit scripts (traced the old inflated 25.4%, compared bear weights). |

**3. RESEARCH — archive of (mostly rejected) experiments:** ~107 scripts in `research/` plus standalone files like `crypto_momentum_backtest.py`, `factor_timing_backtest.py`, `ml_enhance_test.py`. Historical record only — not run in normal operation. See [Research Findings](#research-findings).

**4. DORMANT — built but disabled:** `edgar_realtime.py`, `event_short_manager.py`, `event_detector.py`, `xbrl_detector.py` (the short sleeve — all short strategies lose money; kept for data collection only).

### Full tree

```
AutoTraderBot/
├── .env                              ← API keys (not in git)
├── .env.example                      ← Template for .env
├── ecosystem.config.js               ← PM2 service configuration
├── package.json                      ← Node.js dependencies
├── README.md
│
├── server/
│   ├── index.js                      ← Express API server (Alpaca proxy)
│   ├── tradingEngine.js              ← Alpaca trading engine (Node.js)
│   ├── journal.js                    ← Trade journal / order tracking
│   └── grafanaMetrics.js             ← Metrics endpoint
│
├── ml_service/
│   ├── signal_builder.py             ← LIVE: signal harness (v12 config, uses shared brain)
│   ├── signal_server.py              ← LIVE: FastAPI server (/signals, /health)
│   ├── ibkr_engine.py                ← LIVE: IBKR trading engine (async, ib_insync)
│   ├── main_production_backtest.py              ← BACKTEST: the one canonical backtester
│   ├── wrds_universe.py              ← Universe builder from WRDS data
│   ├── wrds_data_provider.py         ← WRDS data loading (FF, FRED)
│   ├── massive_data_provider.py      ← Polygon/Massive price data
│   ├── scrape_sp1500.py              ← Daily SP1500 membership scraper
│   ├── data_pipeline.py              ← FMP data fetching
│   ├── sp500_universe.py             ← Universe management
│   ├── edgar_realtime.py             ← EDGAR 8-K monitor (dormant)
│   ├── event_short_manager.py        ← Short sleeve manager (disabled)
│   │
│   ├── strategies/
│   │   ├── multi_strategy_engine.py  ← THE STRATEGY BRAIN (shared by live + backtest):
│   │   │                                sleeves + strategy_value() + PROD_WEIGHTS_*
│   │   ├── alpha_engine.py           ← ML alpha pipeline (research)
│   │   └── xgboost_ranker.py         ← XGBoost ranking model (research)
│   │
│   ├── scripts/
│   │   ├── refresh_data.py           ← Daily data refresh (cron)
│   │   ├── rebalance_now.py          ← Manual one-time rebalance
│   │   ├── build_universe_2000.py    ← Build 25-year backtest universe
│   │   ├── ibkr_resync.py            ← IBKR position resync tool
│   │   ├── manual_rebalance.py       ← Manual rebalance tool
│   │   └── buy_missing.py            ← Buy missing positions tool
│   │
│   ├── research/                     ← Research scripts:
│   │   ├── *ML*: alpha_engine, train_alpha, xgboost_ranker
│   │   ├── *Factors*: compustat_alpha_factors, sleeve_attribution
│   │   ├── *Validation*: concentration_validation, n_risk_and_costs
│   │   ├── *Improvements*: improvements_v12, adaptive_stop, tier1
│   │   ├── *NLP*: lazy_prices (EDGAR filing similarity)
│   │   └── *Portfolio*: uncorrelated_strategies
│   │
│   └── data/
│       ├── wrds/                     ← WRDS parquets (Compustat, IBES, FF, FRED)
│       ├── enhanced_data/            ← FMP snapshot history
│       ├── massive_cache/            ← Polygon price bar cache (~30MB)
│       ├── fundamentals_cache/       ← FMP fundamentals (daily flush, ~640MB)
│       └── price_archive/            ← Daily closing price snapshots
│
├── tests/                            ← Test suite (needs update)
├── deploy/                           ← Deployment scripts
└── logs/                             ← Application logs
```

---

## Setup & Deployment

### Prerequisites

- AWS EC2 instance (t3.medium or larger, Ubuntu 24.04)
- Python 3.11+
- Node.js 18+
- PM2 (`npm install -g pm2`)
- WRDS subscription (for Compustat/IBES data)
- API keys: Alpaca, FMP, Massive/Polygon, Telegram

### Installation

```bash
# Clone
git clone https://github.com/michaelyang-dev/AutoTraderBot.git
cd AutoTraderBot

# Python environment
cd ml_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Node environment
cd ../server
npm install

# Configuration
cd ..
cp .env.example .env
# Edit .env with your API keys

# Upload WRDS data
# Download from WRDS: Compustat quarterly, IBES actuals/surprise/price_targets,
# Fama-French 5-factors + momentum, FRED rates
# Place parquets in ml_service/data/wrds/

# Set up cron jobs
crontab -e
# Add the cron entries from the Data Pipeline section

# Start services
pm2 start ecosystem.config.js
pm2 save
```

### Switching IBKR from Paper to Live

1. Start IB Gateway with live credentials (port 4001)
2. Set `IB_PORT=4001` in `.env`
3. `pm2 restart ibkr-engine`
4. Monitor first rebalance via Telegram and `pm2 logs ibkr-engine`

---

## Operations & Monitoring

### Telegram Alerts

The system sends Telegram notifications for:
- Every buy and sell order (symbol, shares, price)
- Connection loss and reconnection
- Stale signal detection
- Engine crashes
- Daily portfolio summary at 4:05 PM ET (NAV, position count, LIVE/PAPER mode)

### Common Commands

```bash
# Check all services
pm2 list

# Signal server health
curl http://localhost:5001/health

# View trading logs
pm2 logs ibkr-engine --lines 50
pm2 logs signal-server --lines 20

# Emergency stop
pm2 stop ibkr-engine

# Resume trading
pm2 restart ibkr-engine

# Force manual rebalance (caution)
cd ml_service && python3 scripts/rebalance_now.py

# Check data freshness
ls -la ml_service/data/wrds/fama_french_5factors_momentum_daily.parquet
tail -3 logs/refresh_data.log
```

### Quarterly Maintenance

1. Download fresh Compustat + IBES data from WRDS
2. Upload parquets to `ml_service/data/wrds/` on AWS
3. Restart signal server: `pm2 restart signal-server`

---

## Known Limitations

### Backtest vs Live Divergences

| Item | Backtest | Live | Impact |
|------|----------|------|--------|
| **Bear exposure** | Scales equity to 40% (60% cash) | Rotates sleeve weights (stays fully invested) | Live gets ~3pp higher CAGR, ~5pp worse MaxDD |
| **Execution** | Trades at close price | Market orders during the day | ~0.1% slippage difference |
| **Trailing stops** | Checked daily at close | Checked every 10 minutes | Live catches crashes faster |
| **Leverage** | 1x only | ~1.49x margin | Not simulated in backtest |
| **Position rounding** | Exact (fractional) dollar amounts | Whole shares only (IBKR blocks fractional via API) | On the $30K IBKR acct this is large — a $1,900 share is 6% of equity. This is why IBKR runs `LEVERAGE=1.8` to deploy ~1.49x effective. |
| **Costs** | 10bps round-trip | ~2-5bps (Alpaca free, IBKR ~1-2bps + spread) | Live costs lower than backtest |

### Strategic Limitations

- **Momentum crashes:** Rare (~2-3 per decade) but violent. The UMD crash detector and SMA200 filter provide partial protection, but the first few days of a crash are unavoidable.
- **Concentration risk:** 50% in the momentum sleeve with only 5 picks. Max single position drifts to ~44% between rebalances. Bootstrap analysis shows n=5 vs n=10 Sharpe is statistically indistinguishable — concentration is a risk choice, not a return advantage.
- **Statistical significance:** Paired bootstrap shows the strategy's Sharpe advantage over SPY buy-and-hold is NOT statistically significant on 8 years of data (Sharpe diff CI: [-0.45, +0.75]). The point estimate favors the strategy (+0.19 Sharpe) but the confidence interval includes zero. 25-year data narrows the interval but still doesn't achieve significance.
- **Cost sensitivity:** Annual turnover ~1,100%. At realistic 20bps round-trip costs (vs 10bps in backtest), cost drag is ~2.2% annually. This is the highest-leverage variable — every additional 10bps costs ~1.1% of return.
- **Regime dependence (the big one):** The headline 22% CAGR is a *favorable-regime* number. Over the full 2000-2025 cycle the same strategy earns only **low-teens at 1x with -47% to -59% crash drawdowns** — the 2018-2025 era was momentum-friendly with no sustained bear. The pre-2010 market is structurally different (liquidity, sector mix, no mega-cap momentum regime), so the recent edge may not persist. Plan with the through-cycle number, not the recent one. The n=5 concentration advantage likewise exists mainly in the 2019-2024 era; on pre-2018 data broader portfolios perform similarly.
- **Data dependency:** Live trading requires functioning APIs (Polygon, FMP, WRDS). If all data sources fail simultaneously, the engine stops trading (stale signal rejection).
- **Single-country exposure:** SP1500 only (US equities). No international diversification.
- **Value sleeve mislabeling:** The "value" sleeve is actually quality + long-term reversal (ROE, gross margin, -ret_252d) with no genuine valuation metric (no B/M, FCF yield, or EV/EBIT). It's partially redundant with the low-vol quality sleeve.
- **Backtest/live code is now unified (June 2026):** Previously the value sleeve and regime weights were written *twice* (in `signal_builder.py` and `main_production_backtest.py`), which caused a bear-weight drift (backtest 10/20/60 vs live 10/30/50 — both tested ~equal, live kept). These have been **consolidated** into `strategies/multi_strategy_engine.py` as `strategy_value()` and `PROD_WEIGHTS_BULL/BEAR/CRASH` — a single source of truth imported by both. Change a weight once, it changes everywhere. Verified: backtest unchanged (22.0%) and live signals byte-identical after the refactor.
- **Reproducibility:** Running `python main_production_backtest.py` directly executes its hardcoded `configs` list (currently old v10 experiments), NOT the deployed v12. To reproduce the deployed numbers, use `research/v12_ground_truth.py`, which runs the exact v12 config with enhanced/SI cleared.

### Future Research Directions

1. **Diversified trend sleeve** (highest priority): Gold + commodities + FX + rates trend-following at 10-15% allocation. Validated as genuinely uncorrelated (-0.023 with momentum). Requires futures/ETF data.
2. **International momentum**: Japanese, European, EM equities — uncorrelated with US momentum, IBKR already supports international trading.
3. **Dispersion within winners**: Signals that differentiate among stocks all going up — earnings revision momentum, positioning/crowding, management tone. Harder target but the only way to improve stock selection.
4. **Intra-rebalance drift trigger**: Trim positions when they cross 25% weight instead of waiting 20 days. Reduces tail risk from position concentration drift.

---

## License

MIT. Not financial advice. Past performance does not guarantee future results. Use at your own risk.
