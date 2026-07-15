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

**Python envs:** production (PM2 + cron on AWS) runs from `ml_service/venv`; local
dev (`start.sh`) uses the repo-root `.venv` — two separate environments, so keep
both in sync when adding a dependency.

## 🟠 Crypto — RESEARCH COMPLETE → PAPER-SHADOW (self-contained, US-accessible)

All crypto work lives under `ml_service/crypto/`. Nothing here touches the live
equity system.

```
ml_service/crypto/
  data/         # Binance CDN fetchers (survivorship-complete spot + funding)
  backtest/     # research battery (momentum, structural_edges, trend, beta_product…)
  strategy/     # beta_signal.py (live daily target), paper_trade.py, risk_controls.py
  research/     # crypto backtests
ml_service/data/crypto/     # crypto data files (Binance close/funding parquets)
```

**Finding:** after a survivorship-clean sweep across every frequency and signal
class, **no systematic alpha survives out-of-sample (2023+)** — momentum, basis,
funding carry, trend and intraday edges all decayed post-2020-21. (The early
"+47% momentum" was a survivorship / meme-exclusion mirage on incomplete data.)

**Product = risk-managed BTC/ETH beta** (not alpha): 60/40, vol-target 30%, 200d
regime de-risk → Sharpe ~1.24 / MaxDD −30% vs BTC buy&hold 0.90 / −77%. US-deployable
as **spot on Coinbase/Kraken**. A carry+beta paper-trade book runs daily via cron
(`crypto/strategy/paper_trade.py`); `beta_signal.py` emits today's BTC/ETH/cash
target from free Coinbase public candles. See `ml_service/crypto/README.md` + `FINDINGS.md`.

**Data:** FREE + survivorship-complete from the Binance `data.binance.vision` CDN
(683 USDT perps incl. delisted + 57 funding series). No paid vendor, no CME/IBKR feed.
