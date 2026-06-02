# AutoTrader v12

Autonomous multi-factor equity trading system. Runs 24/5 on AWS, trades the SP1500 universe via Alpaca and Interactive Brokers with 1.5x leverage.

**Backtest (2018-2025, honest, no look-ahead):** 25.4% CAGR | 1.00 Sharpe | -26.4% MaxDD at 1x

---

## Strategy

Three-sleeve momentum/value/quality approach with 20-day batch rebalancing:

| Sleeve | Weight | What it does |
|--------|--------|-------------|
| **Momentum** | 50% | Skip-month momentum (12-1), top 5 picks, SMA200 trend filter, quality boosts |
| **Value** | 35% | ROE + gross margin + low debt + momentum composite, top 10 picks |
| **Low-Vol Quality** | 15% | Inverse volatility + margins + 6-month momentum, top 10 picks |

**Risk management:**
- 15% max per position (prevents concentration)
- 40% trailing stop per position
- UMD crash regime detection (shifts to defensive weights)
- Breadth-based bull/bear weight blending
- 20-day batch rebalance (no mid-cycle trading)

**What's disabled (tested, hurts returns):**
- Short interest data (-3pp CAGR)
- Enhanced FMP data / analyst estimates (-2pp CAGR)
- ML cross-sectional ranking (IC=0.014, too weak)
- All Compustat alpha factors (dilute momentum signal)

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │          Signal Server           │
                    │      (FastAPI, port 5001)        │
                    │                                  │
                    │  Massive/Polygon ─→ Prices       │
                    │  WRDS Compustat  ─→ Fundamentals  │
                    │  Fama-French     ─→ UMD Regime    │
                    │  Wikipedia       ─→ SP1500 Members│
                    │                                  │
                    │  strategy1_momentum_reversal()    │
                    │  strategy5_lowvol_quality()       │
                    │  _strategy_value()                │
                    │         ↓                        │
                    │  22 BUY signals + weights         │
                    └──────────┬──────────────────────┘
                               │ HTTP /signals
                    ┌──────────┴──────────┐
                    ↓                     ↓
          ┌─────────────────┐   ┌─────────────────┐
          │  Alpaca Engine  │   │   IBKR Engine    │
          │  (Node.js)      │   │   (Python)       │
          │  Port 3000      │   │   Port 4001/4002 │
          │                 │   │                  │
          │  Paper/Live     │   │  Paper/Live      │
          │  1.5x leverage  │   │  1.5x leverage   │
          │  Trailing stops │   │  Trailing stops  │
          │  Batch rebal    │   │  Batch rebal     │
          └─────────────────┘   └─────────────────┘
```

All services managed by PM2 on AWS EC2.

---

## Data Pipeline

Fully automated via cron (Mon-Fri):

| Time (ET) | Job | What it does |
|-----------|-----|-------------|
| 6:00 AM | `scrape_sp1500.py` | SP500/SP400/SP600 membership from Wikipedia |
| 5:30 PM | `refresh_data.py` | FMP fundamentals, VIX, Fama-French, options, price archive, cache flush |
| 5:50 PM | `pm2 restart signal-server` | Reload signals with fresh data |

**Manual (quarterly):** WRDS Compustat + IBES data upload for backtesting.

---

## Files

```
ml_service/
├── signal_builder.py              ← Signal generation (v12 config)
├── signal_server.py               ← FastAPI server, serves /signals and /health
├── ibkr_engine.py                 ← IBKR trading engine (async, ib_insync)
├── wrds_universe.py               ← Universe builder from WRDS data
├── fast_backtest.py               ← Backtester (uses exact production strategy code)
├── massive_data_provider.py       ← Polygon/Massive price data provider
├── scrape_sp1500.py               ← Daily SP1500 membership scraper
├── strategies/
│   └── multi_strategy_engine.py   ← Strategy functions (momentum, value, lowvol)
├── scripts/
│   ├── refresh_data.py            ← Daily data refresh (cron)
│   ├── rebalance_now.py           ← Manual one-time rebalance
│   └── build_universe_2000.py     ← Build 25-year backtest universe
├── research/                      ← ML research scripts (LightGBM, regime, etc.)
└── data/
    ├── wrds/                      ← WRDS parquets (Compustat, IBES, FF, FRED)
    ├── enhanced_data/             ← FMP snapshots + history
    ├── massive_cache/             ← Polygon price bar cache
    └── fundamentals_cache/        ← FMP fundamentals (daily flush)

server/
└── tradingEngine.js               ← Alpaca trading engine (Node.js)
```

---

## Configuration (v12)

All parameters match between backtest and live:

| Parameter | Value |
|-----------|-------|
| Sleeve weights | 50% momentum / 35% value / 15% low-vol |
| Momentum picks | Top 5 |
| Rebalance | Every 20 trading days |
| Trailing stop | 40% from peak |
| Position cap | 15% max per stock |
| Leverage | 1.5x (Reg T margin) |
| Universe | SP1500 (~1,500 stocks) |
| Short interest | Disabled |
| Enhanced data | Disabled |

---

## Backtest Results

**2018-2025 (7-day start-day averaged, no look-ahead, 10bps costs):**

| Metric | At 1x | At 1.5x |
|--------|-------|---------|
| CAGR | 25.4% | ~38% |
| Sharpe | 1.00 | ~1.00 |
| Max Drawdown | -26.4% | ~-40% |

| Year | Strategy | SPY |
|------|----------|-----|
| 2018 | +16.2% | -5.2% |
| 2019 | +13.5% | +31.1% |
| 2020 | +26.1% | +17.3% |
| 2021 | +23.5% | +30.5% |
| 2022 | -5.2% | -18.6% |
| 2023 | +32.5% | +26.7% |
| 2024 | +49.5% | +25.6% |
| 2025 | +17.2% | +18.0% |

**25-year (2001-2025):** 16.9% CAGR, 0.79 Sharpe, -37.1% MaxDD

---

## Research Findings

Extensive testing of ML and alternative approaches (May 2026):

| Approach | Result |
|----------|--------|
| LightGBM cross-sectional ranker | IC=0.014 (too weak to beat factors) |
| Compustat alpha factors (accruals, B/M, CF yield) | All hurt momentum by -2 to -15pp |
| Crash risk model (AUC=0.673) | Filtering risky stocks kills returns |
| Regime detection (FF + FRED macro) | 58% accuracy, marginal |
| Factor timing / dynamic weights | Noise, momentum wins in all conditions |
| Risk-adjusted momentum | Worse than raw momentum |
| Inverse-vol position sizing | Worse than signal-proportional |
| Quality gates | Remove best momentum stocks |

**Conclusion:** Pure momentum with signal-proportional sizing is the efficient frontier with available data. Institutional-grade ML requires alternative data (NLP, options flow, order book).

---

## Quick Start

```bash
# Clone
git clone https://github.com/michaelyang-dev/AutoTraderBot.git
cd AutoTraderBot

# Configure
cp .env.example .env
# Edit .env with your API keys (Alpaca, FMP, Massive, Telegram)

# Install Python dependencies
cd ml_service && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install Node dependencies
cd ../server && npm install

# Start services
pm2 start ecosystem.config.js

# Check status
pm2 list
curl http://localhost:5001/health
```

---

## Monitoring

- **Telegram alerts:** Trades, errors, connection loss, daily summary
- **Kill switch:** `pm2 stop ibkr-engine` (instant stop)
- **Health check:** `curl http://localhost:5001/health`
- **Logs:** `pm2 logs signal-server --lines 20`

---

## License

MIT. Not financial advice. Past performance does not guarantee future results.
