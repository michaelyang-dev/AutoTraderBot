# Crypto module

Self-contained crypto research/trading. Does **not** import from or affect the live equity system.

## Conclusion (read `FINDINGS.md` for the full writeup)

After an exhaustive, survivorship-clean search across **every frequency (1h→daily) and signal
class**, **no systematic alpha survives out-of-sample (2023+).** Momentum, reversal, funding carry,
basis, funding-positioning, trend, and intraday edges all decayed after the 2020–21 mania. Liquid
crypto is efficient. (The early "+47% momentum" was a **survivorship + meme-exclusion mirage** on
incomplete CoinMarketCap data — disproven on the complete Binance universe.)

**Recommended product = risk-managed BTC/ETH beta** (not alpha): 60/40, vol-target 30%, 200d
regime de-risk. **Sharpe 1.24, MaxDD −30%** vs BTC buy&hold 0.90 / −77%; positive every year incl.
2022 (+5% vs BTC −64%). Robust across params. US-deployable as **spot on Coinbase/Kraken**.

## Data (FREE, survivorship-complete — no paid vendor needed)

| File | What | Source |
|---|---|---|
| `data/crypto/binance_close/qvol.parquet` | 683 USDT perps incl. delisted, daily | `data.binance.vision` CDN |
| `data/crypto/binance_funding.parquet` | 57 perp funding series (8h→daily) | same |
| `data/crypto/binance_close/qvol_1h.parquet` | 40 liquid majors, hourly | same |

Survivorship-complete for the **tradable** Binance-perp universe (correct test/live match). Audited
by `backtest/data_integrity.py` (no forward-fill, no back-fill, costless-exit immaterial).

## Layout

- `data/` — fetchers: `binance_klines_fetcher.py` (full survivorship universe via CDN bucket
  listing), `binance_funding_fetcher.py`, `binance_hourly_fetcher.py`
- `backtest/` — the research battery: `momentum_binance.py`, `data_integrity.py`,
  `structural_edges.py`, `trend_v2.py`, `intraday_edges.py`, **`beta_product.py`** (the product),
  `beta_robust.py`
- `strategy/` — **`beta_signal.py`** (live daily target allocation, Coinbase public candles)

## Run the product signal

```
python crypto/strategy/beta_signal.py        # today's BTC/ETH/cash target
```

## Status

Research **complete**. Product validated + live signal works. Next (if deploying): paper-shadow
the `beta_signal` allocation, then small spot live on Coinbase/Kraken — mirror the equity bot's
shadow-first discipline.
