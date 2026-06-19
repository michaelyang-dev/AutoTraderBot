# Crypto module

Self-contained crypto research/trading. Does **not** import from or affect the
live equity system.

## Strategy direction (US person, IBKR)

Optimizing for **risk-adjusted return with leverage available** → the principled
core is **market-neutral basis carry**, not directional momentum (you lever your
highest-Sharpe book, not your highest-CAGR one).

- **Core — basis / cash-and-carry:** long spot (or spot ETF) + short CME futures;
  harvest the futures premium as it converges to spot. Market-neutral. Leverage
  is the CAGR dial. **Net edge = basis − financing cost** (must stay positive).
- **Optional — trend sleeve:** time-series momentum that can exit/short in
  downtrends (only if the bake-off shows it improves combined Sharpe).

All US-legal in the existing IBKR account: spot (Paxos), CME BTC/ETH futures,
spot ETFs (IBIT/ETHA). No offshore venues, no perps.

## Data

| Need | Source | Status |
|---|---|---|
| Spot OHLCV | Polygon/Massive (`X:BTCUSD`), Alpaca, FMP | ✅ available |
| CME futures / basis history | IBKR API (`reqHistoricalData`) | ⏳ to build |

## Layout

- `data/`       — spot provider + CME basis fetcher
- `strategies/` — `basis_carry.py`, `trend.py`
- `backtest/`   — crypto backtest engine (survivorship-clean, fees + financing modeled)
- `research/`   — backtests / the basis-vs-trend bake-off

Crypto data files live in `ml_service/data/crypto/`.

## Status

Scaffolding only. Next: build the CME basis fetcher → backtest basis (with
financing costs) → paper → small live.
