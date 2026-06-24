# Live System — Source of Truth (verified 2026-06-24)

Read this before analyzing the live engines, so the same confusions don't recur.

## The two engines run the SAME strategy (v12)

Both the **IBKR live** engine and the **Alpaca paper** engine consume the **same v12
signals** from the shared **signal server (`:5001`)** and apply the same portfolio rules.
They are NOT different strategies. Verified side-by-side:

| Parameter | IBKR live (`ml_service/ibkr_engine.py`) | Alpaca paper (`server/tradingEngine.js`) |
|---|---|---|
| Signal source | signal server `:5001` (v12, top_n=5) | signal server `:5001` (v12, top_n=5) |
| Rebalance cadence | `REBAL_DAYS = 20` | `REBAL_INTERVAL_CYCLES = 20*390` (20 trading days) |
| Max positions | `MAX_POSITIONS = 30` (~22-25 held) | `MAX_OPEN_POSITIONS = 30` |
| Position cap | `POSITION_CAP = 0.15` | `MAX_POSITION_PCT = 0.15` |
| Trailing stop | `TRAILING_STOP = 0.40` | `TRAILING_STOP_PCT = 0.40` |
| Effective leverage | `LEVERAGE = 1.80` → **~1.49x** | `MAX_CASH_DEPLOY_PCT = 1.60` → **~1.49x** |
| Direction | long-only (`SHORT_ENABLED=False`) | long-only |
| Vol-scaling | on (target ~0.15 1x / 0.22 levered) | on (same) |

The leverage *config numbers differ on purpose* (1.8 vs 1.6) to land on the **same ~1.49x
effective**: IBKR's $30K account has integer-share rounding drag (1.8 → 1.49), Alpaca's
$1.3M account has negligible rounding (1.6 → 1.49). **Do not "align" them to the same
number** — that would change the effective leverage.

## Why Alpaca shows bigger gains than IBKR (it's NOT strategy)
1. **Size:** Alpaca paper ≈ **$1.33M** vs IBKR live **$34K** (~39x). Same % move = ~39x the
   dollars. This is the whole story if comparing dollar P&L.
2. **Fractional vs integer:** Alpaca holds exact weights, fully invested; IBKR rounds to
   whole shares → small cash drag at $34K.
3. **Costs:** IBKR pays real commissions/slippage; Alpaca paper is frictionless + optimistic marks.

The % gap = the cost of running live at small size. It shrinks as the account grows.

## Gotchas that previously caused misdiagnosis
- **Alpaca multi-strategy buckets are DISABLED**, not active: `momentum: 0, mean_reversion: 0,
  mega_cap: 0` — all 30 slots go to the `ml_medium` (= v12 factor) bucket. Log labels like
  "ML 22/30, MOM 0/0" reflect this. Don't read the dormant `MOM`/`MR`/`MEGACAP` code as live.
- **`REBALANCE_INTERVAL: 5` was a dead/unused constant** (removed 2026-06-24). The real
  cadence is `REBAL_INTERVAL_CYCLES = 20*390`. Don't reintroduce it.
- **"v9.6" in comments = the signal-generation version** (what `build_signals_v9` computes);
  **"v12" = the strategy/portfolio version.** Both are correct and consistent — not a mismatch.
- **Holdings lag the live BUY list by up to 20 days** (the rebalance cadence). If IBKR holds a
  few names not in today's signal BUY list (and is missing a few new ones), that's EXPECTED —
  it rebalances every 20 trading days, not daily. Both engines do this.

## Services (PM2 on EC2 54.158.238.15)
- `ibkr-engine` — LIVE v12, $30K, IBKR (IB Gateway in docker, socat 4003→4001, connects :4001)
- `trading-engine` — Alpaca PAPER v12, $1.3M
- `signal-server` — v12 signal brain on `:5001` (restarts 17:50 daily)
- `edgar-monitor` — stopped (intentional)
- `pm2-logrotate`

Key crons: `refresh_data.py` 17:30 (enhanced-data timeout 600s as of 2026-06-23),
`scrape_sp1500.py` 06:00, `sentiment_collector.py` every 6h, `data_freshness_check.py` 09:00/13:00.

## Strategy validity (v12 production backtest, 2018-2025, 1x)
CAGR **+21.0%**, Sharpe **1.10**, MaxDD **−24%**, alpha **+6.9%** vs SPY. Weakest year 2022
(+0.3%, the bear); best 2024 (+46.8%). Realistic live alpha after frictions ≈ 15-25%/yr.

> NOTE: `docs/BACKTEST_LIVE_PARITY.md` (April 2026) describes the **older multi-strategy**
> slot config (`momentum:3, mega_cap:2, flex:1`). That has since been superseded by the
> single-strategy v12 setup documented here (all slots → factor signals).
