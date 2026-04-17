# ⚡ AutoTrader Engine

Fully autonomous paper trading bot with Alpaca integration. Two modes: **Simulated** (fake prices, no API keys needed) and **Alpaca Paper** (real market data, real paper orders).

Includes a 5-strategy signal consensus engine, ATR-based position sizing, trailing stop-loss, an earnings calendar filter, a SPY market regime filter, and a market hours window guard.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone <your-repo> auto-trader
cd auto-trader
npm install
```

### 2. Get API Keys

**Alpaca (required for live mode)**
1. Sign up at **https://app.alpaca.markets** (free)
2. Select **Paper Trading** in the top-left
3. Go to **API Keys → Generate New Key**
4. Copy your **Key ID** and **Secret Key**

**Financial Modeling Prep (optional — earnings calendar)**
1. Sign up at **https://financialmodelingprep.com/developer/docs** (free tier: 250 req/day)
2. Copy your API key

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
PORT=3001

# Optional — earnings calendar filter
FMP_API_KEY=your_fmp_api_key_here
```

If `FMP_API_KEY` is omitted the bot runs normally — earnings filtering is silently skipped.

### 4. Run

```bash
npm start          # server + frontend together

npm run server     # Express API only  → http://localhost:3001
npm run client     # React app only    → http://localhost:3000
```

### 5. Switch Modes

Use the toggle at the top of the UI:
- **⚡ Simulated** — fake prices, instant feedback, no API keys needed
- **🔌 Alpaca Paper** — real market data, real paper orders

---

## 📁 Project Structure

```
auto-trader/
├── .env.example                  ← Copy to .env, fill in your keys
├── package.json
│
├── server/
│   └── index.js                  ← Express backend — proxies Alpaca + FMP APIs
│
└── src/
    ├── App.js                    ← Root — mode selector, layout, ConnectionBanner
    │
    ├── config/
    │   └── constants.js          ← All tunable parameters (RISK, MARKET_HOURS, UNIVERSE…)
    │
    ├── engine/
    │   ├── indicators.js         ← SMA, EMA, RSI, MACD, Bollinger Bands, ATR
    │   ├── signalEngine.js       ← 5-strategy consensus scorer
    │   ├── regimeEngine.js       ← SPY SMA-based market regime (BULLISH/CAUTIOUS/BEARISH)
    │   ├── priceEngine.js        ← Simulated price generation (sim mode only)
    │   ├── tradeExecutor.js      ← Simulated trade execution (sim mode only)
    │   ├── alpacaClient.js       ← Frontend API client → Express proxy
    │   ├── fmpClient.js          ← FMP earnings calendar client → Express proxy
    │   ├── livePriceEngine.js    ← Alpaca real-time price fetching + SPY history
    │   ├── liveTradeExecutor.js  ← Alpaca order execution with all filters applied
    │   ├── stockScreener.js      ← Scans full market for tradeable stocks
    │   └── index.js              ← Barrel export
    │
    ├── hooks/
    │   ├── useAutoTrader.js      ← Simulated mode state + tick engine
    │   └── useAlpacaTrader.js    ← Live mode state — regime, polling, trade cycles
    │
    ├── components/
    │   ├── Header.js             ← Logo, regime badge, pause/speed controls
    │   ├── StatsBar.js           ← Portfolio value, returns, cash, win-rate
    │   ├── EquityChart.js        ← Equity curve
    │   ├── AllocationChart.js    ← Capital allocation pie chart
    │   ├── PositionsPanel.js     ← Open positions + SL/TP progress bars
    │   ├── ActivityFeed.js       ← Real-time trade and system log
    │   ├── MarketScanner.js      ← Signal table for all tracked stocks
    │   ├── StrategyLegend.js     ← Per-strategy stats
    │   └── index.js              ← Barrel export
    │
    ├── utils/
    │   └── formatters.js         ← Currency/percent helpers
    │
    └── styles/
        └── theme.js              ← Colors, fonts, global CSS
```

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      React Frontend                       │
│                                                           │
│  useAutoTrader          useAlpacaTrader                  │
│  (simulation)           (live mode)                      │
│       │                      │                            │
│  tradeExecutor       liveTradeExecutor                   │
│  priceEngine         livePriceEngine                     │
│                       regimeEngine  ←── SPY SMA filter   │
│                       fmpClient     ←── earnings calendar │
│                       alpacaClient  ←── market data/orders│
│                                                           │
│  ────────────────────────────────────────────────────── │
│                   Shared Components                       │
│   Header · StatsBar · EquityChart · ActivityFeed · …     │
└──────────────────────────────┬───────────────────────────┘
                               │ HTTP (localhost:3001)
┌──────────────────────────────┴───────────────────────────┐
│                   Express Server (server/index.js)        │
│                                                           │
│   /api/account    /api/positions   /api/orders            │
│   /api/bars       /api/snapshots   /api/clock             │
│   /api/screener   /api/earnings    /api/health            │
│                        │                    │             │
│               Alpaca SDK            FMP fetch()           │
└───────────────────┬────────────────────┬─────────────────┘
                    │ HTTPS              │ HTTPS
          ┌─────────┴──────┐   ┌────────┴────────┐
          │ Alpaca Paper   │   │ Financial        │
          │ Trading API    │   │ Modeling Prep    │
          └────────────────┘   └─────────────────┘
```

**Why a separate server?**
- API keys never reach the browser
- All Alpaca and FMP calls go through one authenticated proxy
- Easy to add logging, webhooks, or a database later

---

## 🧠 How the Bot Decides

### 1. Stock Screener

At startup the bot runs a live screener against the full Alpaca asset list and selects the **top 75 most active stocks** that pass these filters:

| Filter       | Threshold             |
|--------------|-----------------------|
| Price        | $10 – $1,500          |
| Daily volume | ≥ 500,000 shares      |
| Trade count  | ≥ 1,000 trades/day    |

Falls back to the hardcoded 12-stock `UNIVERSE` in `constants.js` if the screener fails.

---

### 2. Signal Engine (5-Strategy Consensus)

Every stock in the universe is scored on each cycle by five independent strategies. Each strategy casts a full vote (+1 buy / −1 sell) on a strong signal, or a partial vote (+0.2–0.3) on a soft directional read.

| Strategy       | Buy trigger                        | Sell trigger                       |
|----------------|------------------------------------|------------------------------------|
| SMA Crossover  | 10-SMA crosses **above** 30-SMA    | 10-SMA crosses **below** 30-SMA    |
| RSI Mean-Rev   | RSI < 28 (oversold)                | RSI > 72 (overbought)              |
| MACD Trend     | MACD line crosses above signal     | MACD line crosses below signal     |
| Bollinger Bands| Price touches lower band           | Price touches upper band           |
| Momentum       | 12-bar return > +3.5%              | 12-bar return < −2.5%              |

**Consensus score** = buy votes − sell votes:

| Score   | Consensus    | Action in BULLISH regime         |
|---------|--------------|----------------------------------|
| ≥ 2     | STRONG BUY   | Buy (all regimes except BEARISH) |
| ≥ 1     | BUY          | Buy (BULLISH regime only)        |
| ≤ −1    | SELL         | Close position                   |
| ≤ −2    | STRONG SELL  | Close position                   |
| other   | HOLD         | No action                        |

---

### 3. SPY Market Regime Filter

Before placing any buy order, the bot classifies the current market environment using SPY's daily closes against its own moving averages. SPY is always fetched with 220 bars so SMA200 is always available.

| Regime       | Condition                              | Bot behavior                                      |
|--------------|----------------------------------------|---------------------------------------------------|
| 🟢 BULLISH   | SPY > SMA50 **and** SPY > SMA200       | Normal — full position sizes, BUY + STRONG BUY   |
| 🟡 CAUTIOUS  | SPY < SMA50, SPY > SMA200              | STRONG BUY signals only, position sizes **halved** |
| 🔴 BEARISH   | SPY < SMA50 **and** SPY < SMA200       | **No new buys** — only sells, stop-losses, take-profits |

**Recovery rule:** After a BEARISH period, the regime does not flip to CAUTIOUS or BULLISH until SPY has closed above its 50-SMA for **3 consecutive days**. This prevents false recoveries from brief bounces.

Every regime change is logged to the Activity Feed with SPY's current price, SMA50, and SMA200. The current regime is displayed as a colored badge in the dashboard header.

---

### 4. Earnings Calendar Filter

Before each trade cycle the bot fetches upcoming earnings dates from the Financial Modeling Prep API (results are cached for 4 hours to stay within the free tier's 250 req/day limit).

| Situation                              | Action                                              |
|----------------------------------------|-----------------------------------------------------|
| Stock has earnings within **3 days**   | Skip the buy entirely, log the reason               |
| Held position has earnings **tomorrow**| Sell the position before the announcement           |

If `FMP_API_KEY` is not set, or if the FMP API is unavailable, the filter is skipped gracefully and normal trading continues.

---

### 5. ATR-Based Position Sizing

Rather than allocating a flat percentage of the portfolio to every stock, position sizes are scaled inversely with each stock's **Average True Range** (14-period, close-to-close). Volatile stocks get smaller allocations; stable stocks get larger ones.

```
volatilityScale  = ATR_TARGET_PCT / stockAtrPct
dynPositionPct   = clamp(MAX_POSITION_PCT × volatilityScale, MIN_POSITION_PCT, MAX_POSITION_PCT)
```

The `regimeMult` (0.5 in CAUTIOUS, 1.0 otherwise) is applied on top before clamping.

| Stock | Typical daily ATR% | Approx. allocation (at $100k) |
|-------|--------------------|-------------------------------|
| TSLA  | ~3.5%              | ~4% ($4k)                     |
| NVDA  | ~2.5%              | ~6% ($6k)                     |
| AAPL  | ~1.2%              | ~12% ($12k)                   |
| JPM   | ~0.6%              | capped at 15% ($15k)          |

---

### 6. Trailing Stop-Loss

Positions use a **trailing stop** rather than a fixed stop from entry price. The peak price is recorded at entry and updated whenever the price makes a new high. The stop triggers when the price falls more than `TRAILING_STOP_PCT` below that peak.

```
trail drop = (currentPrice − peakPrice) / peakPrice
exit if   trail drop ≤ −TRAILING_STOP_PCT  (default −5%)
```

Switch back to a fixed stop-loss by setting `USE_TRAILING_STOP: false` in `constants.js`.

---

### 7. Market Hours Filter (live mode only)

Even when the market is officially open, the bot enforces two quiet windows to avoid elevated volatility:

| Window              | Duration   | Behavior                                          |
|---------------------|------------|---------------------------------------------------|
| Opening buffer      | First 15 min | Entire trade cycle skipped                      |
| Closing buffer      | Last 30 min  | Stop-loss/take-profit checks run; no new buys   |

---

### 8. Risk Parameters

All configurable in `src/config/constants.js`:

```js
RISK = {
  MAX_POSITION_PCT:    0.15,   // max 15% of portfolio per stock (before ATR scaling)
  MIN_POSITION_PCT:    0.03,   // floor: never allocate less than 3%
  STOP_LOSS_PCT:      -0.05,   // fixed stop (used when USE_TRAILING_STOP: false)
  TAKE_PROFIT_PCT:     0.10,   // +10% take-profit
  MAX_OPEN_POSITIONS:  6,      // max concurrent positions
  MAX_CASH_DEPLOY_PCT: 0.90,   // max 90% of available cash per buy
  TRAILING_STOP_PCT:   0.05,   // 5% trail from peak
  USE_TRAILING_STOP:   true,   // toggle trailing vs. fixed stop
  ATR_TARGET_PCT:      0.01,   // reference ATR% for sizing (1%)
}

MARKET_HOURS = {
  OPEN_BUFFER_MINS:  15,       // skip first 15 min after open
  CLOSE_BUFFER_MINS: 30,       // no new buys in last 30 min
}
```

---

## 🔌 Server API Endpoints

| Method | Endpoint                  | Description                            |
|--------|---------------------------|----------------------------------------|
| GET    | `/api/health`             | Connection check                       |
| GET    | `/api/account`            | Account balance & buying power         |
| GET    | `/api/positions`          | Open positions with unrealized P&L     |
| POST   | `/api/orders`             | Place a market or limit order          |
| GET    | `/api/orders`             | List recent orders                     |
| DELETE | `/api/orders`             | Cancel all open orders                 |
| DELETE | `/api/positions/:symbol`  | Close a specific position              |
| DELETE | `/api/positions`          | Close all positions                    |
| GET    | `/api/clock`              | Market open/close status and times     |
| GET    | `/api/quote/:symbol`      | Latest quote for one symbol            |
| GET    | `/api/snapshots?symbols=` | Batch real-time snapshots              |
| GET    | `/api/bars/:symbol`       | Historical daily bars                  |
| GET    | `/api/screener`           | Top 75 most active tradeable stocks    |
| GET    | `/api/earnings?symbols=`  | Upcoming earnings dates (FMP)          |

---

## ⚙️ Customization

| What to change | Where |
|----------------|-------|
| Stock universe | `UNIVERSE` array in `src/config/constants.js` |
| Risk parameters | `RISK` object in `src/config/constants.js` |
| Market hours buffers | `MARKET_HOURS` in `src/config/constants.js` |
| Trailing vs. fixed stop | `USE_TRAILING_STOP` in `src/config/constants.js` |
| Regime recovery days | `REGIME_RECOVERY_DAYS` in `src/engine/regimeEngine.js` |
| Trade/price poll frequency | `PRICE_POLL_MS` / `TRADE_CYCLE_MS` in `src/hooks/useAlpacaTrader.js` |
| Add a new strategy | Add a vote section in `src/engine/signalEngine.js` |
| Extend indicators | Add a function to `src/engine/indicators.js` |

---

## ⚠️ Important Notes

- **Paper trading only** — no real money is used at any point
- The Alpaca paper API simulates real market conditions but fills may differ from live trading
- Market data on Alpaca paper accounts uses IEX exchange data
- Past simulated performance does not predict future results
- PDT (Pattern Day Trader) rules apply: accounts below $25,000 are limited to 3 round-trips per 5 trading days
- The bot only places orders during regular market hours (9:30 AM – 4:00 PM ET), with the additional open/close buffers applied on top
- FMP's free tier allows 250 API calls/day; the earnings cache (4-hour TTL) ensures the bot never exceeds this under normal operation

---

## 📄 License

MIT — use it however you want. Not financial advice.
