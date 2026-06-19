# Project Structure

Two asset classes, kept cleanly separate.

## 🟢 Equities — LIVE ($30K IBKR + paper shadow)

The live equity system. **Do not move these files** — PM2, cron, and imports
reference their current paths; moving them breaks live trading.

```
ml_service/
  ibkr_engine.py            # live IBKR trading engine (PM2: ibkr-engine)
  signal_server.py          # serves signals over HTTP (PM2: signal-server)
  signal_builder.py         # v12 strategy → target weights
  main_production_backtest.py  # the production backtest (FastBacktester)
  strategies/               # equity strategy sleeves (momentum/value/low-vol)
  research/                 # equity research scripts (phase8–13, etc.)
  data/                     # equity data (WRDS, fundamentals, features, NAV…)
server/                     # Alpaca paper engine (Node) + Grafana metrics
```

## 🟠 Crypto — IN DEVELOPMENT (self-contained, US-accessible via IBKR)

All crypto work lives under `ml_service/crypto/`. Nothing here touches the live
equity system.

```
ml_service/crypto/
  data/         # spot OHLCV providers + CME-futures/basis fetcher
  strategies/   # basis carry (core), trend (research)
  research/     # crypto backtests
  backtest/     # crypto backtest engine
ml_service/data/crypto/     # crypto data files (spot, CME futures, basis history)
```

**Direction (US person):** market-neutral **basis carry** as the core (long
spot/ETF + short CME futures, harvest the futures premium), with a possible
trend sleeve. Executed in the existing IBKR account (spot via Paxos, CME BTC/ETH
futures, spot ETFs). See `ml_service/crypto/README.md`.

**Data:** spot OHLCV is already available (Polygon/Massive `X:BTCUSD`, Alpaca,
FMP). CME futures/basis history is fetched via the IBKR API.
