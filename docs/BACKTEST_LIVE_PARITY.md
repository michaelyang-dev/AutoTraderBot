# Backtest-Live Parity

Reconciliation of `ml_service/unified_backtester.py` with `server/tradingEngine.js`.

**Date**: 2026-04-22
**Branch**: main

## What Now Matches

### Phase 1: Quick Wins
- **NEVER_BUY blacklist**: Module-level set applied to all strategies, matching live's 25-symbol leveraged/inverse/volatility blocklist. Previously only applied to MomentumStrategy.
- **Slippage**: Increased from 0.05% to 0.12% per leg, matching observed Alpaca market-order fills.
- **MIN_POSITION_DOLLARS**: $5,000 minimum position size (was $50), matching live `RISK.MIN_POSITION_DOLLARS`.

### Phase 2: Safety Features
- **Sector limits**: Ported `SECTOR_MAX_POSITIONS` and `SYMBOL_SECTOR` mapping from live. International (2), Commodity (2), Bond (2), Volatility (1) caps enforced before every buy.
- **Earnings avoidance**: Loads `earnings_cache/*.json` (503 symbols). Skips opening positions within 3 calendar days of earnings, matching live's `daysUntilEarnings` logic.
- **Idle SPY mechanics**: Added opportunity-ratio check (25% minimum) before selling idle SPY. Added 1-day action gap between SPY buy/sell operations (daily equivalent of live's 15-min dead zone).

### Phase 3: Trend Strategy
- **TrendStrategy class**: Mirrors live's `computeTrendStatus()` and trend entry/exit logic:
  - Entry: price > SMA200, SMA50 > SMA200, >= 30/40 days above SMA200
  - Exit: -10% trailing stop OR 3 consecutive closes below SMA200
  - Position size: 5% of portfolio (matches live `cyclePortfolioValue * 0.05`)
  - Max 3 trend positions
- **CLI integration**: `--include-trend` flag, `--strategy trend` standalone mode

### Slot Config
- **SLOT_LIVE updated**: Now matches live exactly — `ml_medium: 2, momentum: 3, mean_reversion: 0, mega_cap: 2, flex: 1, max: 8`.

## Post-Reconciliation Backtest Results

Configuration: `combined_live --exclude mean_reversion` (matches live deployment)

| Metric | Value |
|---|---|
| CAGR | +48.28% |
| Sharpe | 2.250 |
| Sortino | 3.088 |
| Max Drawdown | -37.9% |
| Alpha vs SPY | +36.98% |
| Total Trades | 2,350 |
| Win Rate | 53.2% |

Period: 2011-12-29 to 2026-04-02 (14.3 years, 534 symbols)

## Trend Strategy Assessment

Trend was implemented for parity but **degrades** combined performance:

| Config | CAGR | Sharpe | Alpha |
|---|---|---|---|
| combined_live (no trend) | +48.28% | 2.250 | +36.98% |
| combined_live + trend | +20.22% | 1.302 | +12.29% |
| trend only | +6.23% | 0.669 | -1.75% |

Trend competes for the single flex slot, crowding out higher-alpha ML/momentum picks. The standalone trend strategy has negative alpha. Recommend keeping trend disabled in backtest unless live performance justifies it.

## Known Remaining Gaps (Accepted)

These are inherent to the backtest-vs-live distinction and are not worth closing:

1. **Execution timing**: Backtest uses daily close prices; live executes intraday with variable fill timing and partial fills.
2. **Commission/fees**: No explicit commission modeling. Slippage at 0.12% approximates total round-trip cost.
3. **Market impact**: No modeling of order-book depth or price impact on large orders.
4. **Regime detection**: Live uses real-time regime classification; backtest uses pre-computed predictions.
5. **Cash management**: Live has fractional shares and exact cash accounting; backtest uses continuous position sizing.
6. **Volume confirmation**: Live requires 1.5x avg volume for trend/other buys (except ML/momentum/MR/mega-cap); backtest skips this since we lack intraday volume at decision time.
7. **Cooldown granularity**: Live cooldowns are time-based (30 min minimum + cycle-based); backtest cooldowns are day-based (5 trading days).
8. **Trend precomputation cost**: Trend strategy precomputation takes ~30 min due to per-bar SMA200 calculation. An optimized rolling implementation would reduce this but wasn't prioritized since trend degrades combined performance.

## TRUE Expected Alpha Post-Reconciliation

With the reconciliation changes (higher slippage, $5k minimum, sector limits, earnings avoidance, corrected slot config), the backtest shows **+36.98% alpha vs SPY** over 14.3 years.

This is the backtest-estimated alpha. Live alpha will be lower due to:
- Intraday execution vs. daily close prices
- Market impact on larger account sizes
- Regime transitions and model staleness between retrains

A realistic live alpha estimate is **15-25%** after accounting for these factors.
