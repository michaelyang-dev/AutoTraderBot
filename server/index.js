// ══════════════════════════════════════════
//  SERVER — Express backend for Alpaca API
//  Handles: account, positions, orders, market data
//  The React frontend calls this instead of Alpaca directly
//  (API keys stay on the server, never exposed to browser)
// ══════════════════════════════════════════

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const https = require("https");
const fs = require("fs");
const { send: sendNotification } = require("./notifications");

// ── Global uncaught error handlers — alert via email, then let PM2 restart ──
process.on("uncaughtException", (err) => {
  console.error("UNCAUGHT EXCEPTION:", err);
  sendNotification(`🚨 UNCAUGHT ERROR — ${err.message}\n${err.stack?.split("\n").slice(0, 4).join("\n") || ""}`, { deduplicate: true, immediate: true });
  // Give time for the email to send before PM2 restarts
  setTimeout(() => process.exit(1), 3000);
});

process.on("unhandledRejection", (reason) => {
  console.error("UNHANDLED REJECTION:", reason);
  const msg = reason instanceof Error ? reason.message : String(reason);
  const stack = reason instanceof Error ? reason.stack?.split("\n").slice(0, 4).join("\n") : "";
  sendNotification(`🚨 UNHANDLED REJECTION — ${msg}\n${stack}`, { deduplicate: true, immediate: true });
});
const path = require("path");
const Alpaca = require("@alpacahq/alpaca-trade-api");
const Database = require("better-sqlite3");

// ── Trade Journal SQLite database ──
const TRADE_DB_PATH = path.join(__dirname, "..", "ml_service", "data", "trades.db");
const db = new Database(TRADE_DB_PATH);
db.pragma("journal_mode = WAL");

db.exec(`
  CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    strategy TEXT NOT NULL,
    ml_confidence REAL,
    portfolio_value REAL,
    pnl REAL,
    notes TEXT
  );

  CREATE TABLE IF NOT EXISTS daily_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    portfolio_value REAL NOT NULL,
    cash REAL NOT NULL,
    positions_count INTEGER NOT NULL,
    daily_pnl REAL
  );

  CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
  CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
  CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_snapshots(date);
`);

const insertTrade = db.prepare(`
  INSERT INTO trades (timestamp, symbol, action, shares, price, strategy, ml_confidence, portfolio_value, pnl, notes)
  VALUES (@timestamp, @symbol, @action, @shares, @price, @strategy, @ml_confidence, @portfolio_value, @pnl, @notes)
`);

const insertSnapshot = db.prepare(`
  INSERT INTO daily_snapshots (date, portfolio_value, cash, positions_count, daily_pnl)
  VALUES (@date, @portfolio_value, @cash, @positions_count, @daily_pnl)
  ON CONFLICT(date) DO UPDATE SET
    portfolio_value = @portfolio_value,
    cash = @cash,
    positions_count = @positions_count,
    daily_pnl = @daily_pnl
`);

console.log(`📓 Trade journal DB: ${TRADE_DB_PATH}`);

// ── Backtest cache directories ──
const CACHE_ROOT = path.join(__dirname, "backtest_data");
const DAILY_DIR  = path.join(CACHE_ROOT, "daily");
const EARN_DIR   = path.join(CACHE_ROOT, "earnings");
[CACHE_ROOT, DAILY_DIR, EARN_DIR].forEach((d) => {
  if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
});

// ── Lightweight JSON fetch (no extra dep needed) ──
function fetchJSON(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      let raw = "";
      res.on("data", (chunk) => (raw += chunk));
      res.on("end", () => {
        try { resolve(JSON.parse(raw)); }
        catch (e) { reject(new Error("FMP response parse error")); }
      });
    }).on("error", reject);
  });
}

const app = express();
app.use(cors());
app.use(express.json());

// ── Validate env ──
const { ALPACA_API_KEY, ALPACA_SECRET_KEY, FMP_API_KEY } = process.env;
if (!FMP_API_KEY) {
  console.warn("⚠️  FMP_API_KEY not set — earnings calendar will be disabled.");
}
if (!ALPACA_API_KEY || !ALPACA_SECRET_KEY || ALPACA_API_KEY === "your_paper_api_key_here") {
  console.error("\n❌  Missing Alpaca API keys!");
  console.error("   1. Copy .env.example → .env");
  console.error("   2. Paste your paper trading keys from https://app.alpaca.markets\n");
  process.exit(1);
}

// ── Alpaca client ──
const alpaca = new Alpaca({
  keyId: ALPACA_API_KEY,
  secretKey: ALPACA_SECRET_KEY,
  paper: true,
});

// ══════════════════════════════════════════
//  ACCOUNT
// ══════════════════════════════════════════

app.get("/api/account", async (req, res) => {
  try {
    const account = await alpaca.getAccount();
    res.json({
      id: account.id,
      cash: parseFloat(account.cash),
      portfolio_value: parseFloat(account.portfolio_value),
      buying_power: parseFloat(account.buying_power),
      equity: parseFloat(account.equity),
      long_market_value: parseFloat(account.long_market_value),
      short_market_value: parseFloat(account.short_market_value),
      initial_margin: parseFloat(account.initial_margin),
      last_equity: parseFloat(account.last_equity),
      pdt_status: account.pattern_day_trader,
      trading_blocked: account.trading_blocked,
      status: account.status,
    });
  } catch (err) {
    console.error("Account error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════
//  POSITIONS
// ══════════════════════════════════════════

app.get("/api/positions", async (req, res) => {
  try {
    const positions = await alpaca.getPositions();
    res.json(
      positions.map((p) => ({
        symbol: p.symbol,
        qty: parseFloat(p.qty),
        avg_entry_price: parseFloat(p.avg_entry_price),
        current_price: parseFloat(p.current_price),
        market_value: parseFloat(p.market_value),
        unrealized_pl: parseFloat(p.unrealized_pl),
        unrealized_plpc: parseFloat(p.unrealized_plpc),
        side: p.side,
      }))
    );
  } catch (err) {
    console.error("Positions error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════
//  ORDERS — Place / List / Cancel
// ══════════════════════════════════════════

// Place an order
app.post("/api/orders", async (req, res) => {
  try {
    const { symbol, qty, side, type = "market", time_in_force = "day", limit_price } = req.body;

    if (!symbol || !qty || !side) {
      return res.status(400).json({ error: "symbol, qty, and side are required" });
    }

    const orderParams = {
      symbol: symbol.toUpperCase(),
      qty: parseInt(qty),
      side,             // "buy" or "sell"
      type,             // "market", "limit", "stop", "stop_limit"
      time_in_force,    // "day", "gtc", "ioc", "fok"
    };

    if (type === "limit" && limit_price) {
      orderParams.limit_price = parseFloat(limit_price);
    }

    console.log(`📤 ORDER: ${side.toUpperCase()} ${qty} ${symbol} (${type})`);
    const order = await alpaca.createOrder(orderParams);

    res.json({
      id: order.id,
      symbol: order.symbol,
      qty: order.qty,
      side: order.side,
      type: order.type,
      status: order.status,
      filled_qty: order.filled_qty,
      filled_avg_price: order.filled_avg_price,
      created_at: order.created_at,
    });
  } catch (err) {
    console.error("Order error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// List recent orders
app.get("/api/orders", async (req, res) => {
  try {
    const { status = "all", limit = 50 } = req.query;
    const orders = await alpaca.getOrders({ status, limit: parseInt(limit), direction: "desc" });
    res.json(
      orders.map((o) => ({
        id: o.id,
        symbol: o.symbol,
        qty: parseFloat(o.qty),
        filled_qty: parseFloat(o.filled_qty || 0),
        side: o.side,
        type: o.type,
        status: o.status,
        filled_avg_price: o.filled_avg_price ? parseFloat(o.filled_avg_price) : null,
        created_at: o.created_at,
        filled_at: o.filled_at,
      }))
    );
  } catch (err) {
    console.error("List orders error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Cancel all orders
app.delete("/api/orders", async (req, res) => {
  try {
    await alpaca.cancelAllOrders();
    res.json({ message: "All orders cancelled" });
  } catch (err) {
    console.error("Cancel orders error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Close a specific position
app.delete("/api/positions/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    await alpaca.closePosition(symbol.toUpperCase());
    res.json({ message: `Position ${symbol} closed` });
  } catch (err) {
    console.error("Close position error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Close ALL positions
app.delete("/api/positions", async (req, res) => {
  try {
    await alpaca.closeAllPositions();
    res.json({ message: "All positions closed" });
  } catch (err) {
    console.error("Close all positions error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════
//  MARKET DATA — Bars, Quotes, Snapshots
// ══════════════════════════════════════════

// Get latest quote for a symbol
app.get("/api/quote/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const snapshot = await alpaca.getSnapshot(symbol.toUpperCase());
    res.json({
      symbol: symbol.toUpperCase(),
      price: parseFloat(snapshot.LatestTrade?.Price || snapshot.DailyBar?.ClosePrice || 0),
      bid: parseFloat(snapshot.LatestQuote?.BidPrice || 0),
      ask: parseFloat(snapshot.LatestQuote?.AskPrice || 0),
      high: parseFloat(snapshot.DailyBar?.HighPrice || 0),
      low: parseFloat(snapshot.DailyBar?.LowPrice || 0),
      open: parseFloat(snapshot.DailyBar?.OpenPrice || 0),
      close: parseFloat(snapshot.DailyBar?.ClosePrice || 0),
      volume: parseInt(snapshot.DailyBar?.Volume || 0),
    });
  } catch (err) {
    console.error("Quote error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Get snapshots for multiple symbols
app.get("/api/snapshots", async (req, res) => {
  try {
    const symbols = (req.query.symbols || "").split(",").filter(Boolean);
    if (symbols.length === 0) {
      return res.status(400).json({ error: "Provide ?symbols=AAPL,MSFT,..." });
    }

    const snapshots = await alpaca.getSnapshots(symbols);
    // Alpaca SDK returns an array of snapshot objects (not a symbol→snapshot map).
    // Property names are PascalCase: LatestTrade.Price, DailyBar.ClosePrice, etc.
    const result = {};
    for (const snap of snapshots) {
      const sym = snap.symbol;
      if (!sym) continue;
      result[sym] = {
        price: parseFloat(snap.LatestTrade?.Price || snap.DailyBar?.ClosePrice || 0),
        bid: parseFloat(snap.LatestQuote?.BidPrice || 0),
        ask: parseFloat(snap.LatestQuote?.AskPrice || 0),
        high: parseFloat(snap.DailyBar?.HighPrice || 0),
        low: parseFloat(snap.DailyBar?.LowPrice || 0),
        open: parseFloat(snap.DailyBar?.OpenPrice || 0),
        close: parseFloat(snap.DailyBar?.ClosePrice || 0),
        prevClose: parseFloat(snap.PrevDailyBar?.ClosePrice || 0),
        volume: parseInt(snap.DailyBar?.Volume || 0),
      };
    }
    res.json(result);
  } catch (err) {
    console.error("Snapshots error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// Get historical bars for a symbol (for indicator calculations)
app.get("/api/bars/:symbol", async (req, res) => {
  try {
    const { symbol } = req.params;
    const { timeframe = "1Day", limit = 60 } = req.query;
    const numLimit = parseInt(limit);

    // Alpaca getBarsV2 requires a "start" date — without it only ~1 bar returns.
    // Fetch all bars from start to now, then return the last `numLimit` bars so
    // the result always includes the most recent data up to today.
    const calendarDaysNeeded = Math.ceil(numLimit * 1.6) + 10; // 1.6x for weekends/holidays + buffer
    const start = new Date();
    start.setDate(start.getDate() - calendarDaysNeeded);
    const startISO = start.toISOString().split("T")[0]; // "YYYY-MM-DD"

    const allBars = [];
    const barIterator = alpaca.getBarsV2(symbol.toUpperCase(), {
      timeframe,
      start: startISO,
      adjustment: "split",
    });

    for await (const bar of barIterator) {
      allBars.push({
        t: bar.Timestamp,
        o: parseFloat(bar.OpenPrice),
        h: parseFloat(bar.HighPrice),
        l: parseFloat(bar.LowPrice),
        c: parseFloat(bar.ClosePrice),
        v: parseInt(bar.Volume),
      });
    }

    // Return only the last N bars (most recent) to match the requested limit
    const bars = allBars.slice(-numLimit);
    console.log(`Bars ${symbol}: requested ${numLimit}, fetched ${allBars.length}, returning ${bars.length} (start: ${startISO}, last: ${bars.length > 0 ? bars[bars.length-1].t : 'none'})`);
    res.json(bars);
  } catch (err) {
    console.error("Bars error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════
//  CLOCK — Is the market open?
// ══════════════════════════════════════════

app.get("/api/clock", async (req, res) => {
  try {
    const clock = await alpaca.getClock();
    res.json({
      is_open: clock.is_open,
      timestamp: clock.timestamp,
      next_open: clock.next_open,
      next_close: clock.next_close,
    });
  } catch (err) {
    console.error("Clock error:", err.message);
    res.status(500).json({ error: err.message });
  }
});


// ══════════════════════════════════════════
//  STOCK SCREENER — Most active tradeable stocks
// ══════════════════════════════════════════

app.get("/api/screener", async (req, res) => {
  try {
    // Get the most active stocks by volume
    const assets = await alpaca.getAssets({ status: "active", asset_class: "us_equity" });

    // Filter to tradeable, non-OTC stocks
    const tradeable = assets.filter(
      (a) => a.tradable && a.exchange !== "OTC" && !a.symbol.includes(".")
    );

    // Get snapshots for top symbols (batch in groups to avoid rate limits)
    const symbols = tradeable.map((a) => a.symbol);
    const batchSize = 50;
    let allScreened = [];

    for (let i = 0; i < Math.min(symbols.length, 500); i += batchSize) {
      const batch = symbols.slice(i, i + batchSize);
      try {
        const snapshots = await alpaca.getSnapshots(batch);
        for (const snap of snapshots) {
          const sym = snap.symbol;
          if (!sym) continue;
          const price = parseFloat(snap.LatestTrade?.Price || 0);
          // Use today's DailyBar when market is open; fall back to PrevDailyBar
          // when market is closed (DailyBar Volume = 0 before open)
          const todayVol = parseInt(snap.DailyBar?.Volume || 0);
          const prevVol  = parseInt(snap.PrevDailyBar?.Volume || 0);
          const todayN   = parseInt(snap.DailyBar?.TradeCount || 0);
          const prevN    = parseInt(snap.PrevDailyBar?.TradeCount || 0);
          const volume     = todayVol > 0 ? todayVol : prevVol;
          const tradeCount = todayN   > 0 ? todayN   : prevN;
          const prevClose = parseFloat(snap.PrevDailyBar?.ClosePrice || 0);
          const change = prevClose > 0 ? (price - prevClose) / prevClose : 0;

          if (
            price >= 10 &&
            price <= 1500 &&
            volume >= 500000 &&
            tradeCount >= 1000
          ) {
            allScreened.push({
              sym,
              base: price,
              sector: "Screened",
              volume,
              price,
              change,
              tradeCount,
            });
          }
        }
      } catch (err) {
        // skip failed batches, continue with others
      }
    }

    // Sort by volume (most active first) and return top 75
    allScreened.sort((a, b) => b.volume - a.volume);
    const top = allScreened.slice(0, 75);

    console.log(`📋 Screener found ${allScreened.length} stocks, returning top ${top.length}`);
    res.json(top);
  } catch (err) {
    console.error("Screener error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════
//  EARNINGS CALENDAR — Financial Modeling Prep
// ══════════════════════════════════════════

app.get("/api/earnings", async (req, res) => {
  if (!FMP_API_KEY) {
    return res.json({});   // earnings disabled — bot continues without it
  }

  try {
    const symbols = (req.query.symbols || "").split(",").filter(Boolean).map(s => s.toUpperCase());

    // Fetch a 7-day window so we cover 3+ trading days regardless of weekends
    const fmt = (d) => d.toISOString().split("T")[0];
    const from = fmt(new Date());
    const to   = fmt(new Date(Date.now() + 7 * 24 * 60 * 60 * 1000));

    const url = `https://financialmodelingprep.com/stable/earnings-calendar?from=${from}&to=${to}&apikey=${FMP_API_KEY}`;
    const events = await fetchJSON(url);

    if (!Array.isArray(events)) {
      // FMP returned an error object (bad key, rate limit, or endpoint issue)
      const msg = events?.["Error Message"] || JSON.stringify(events);
      console.warn("FMP earnings warning:", msg);
      return res.json({});
    }

    // Build { SYMBOL: "YYYY-MM-DD" } — keep nearest date if a symbol appears twice
    const result = {};
    for (const event of events) {
      if (!event.symbol || !event.date) continue;
      if (symbols.length && !symbols.includes(event.symbol)) continue;
      if (!result[event.symbol] || event.date < result[event.symbol]) {
        result[event.symbol] = event.date;
      }
    }

    res.json(result);
  } catch (err) {
    console.error("Earnings error:", err.message);
    res.json({});   // non-fatal — return empty so the bot keeps trading
  }
});

// ══════════════════════════════════════════
//  FUNDAMENTALS — Individual stock quality filter
// ══════════════════════════════════════════

// ══════════════════════════════════════════
//  BACKTESTING — Historical data fetch + cache
// ══════════════════════════════════════════

// All symbols we backtest (all 37 UNIVERSE symbols + SPY for the regime filter)
const BT_SYMBOLS = [
  // Individual stocks (12)
  "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
  // Sector ETFs (11)
  "XLE","XLF","XLV","XLI","XLK","XLY","XLP","XLU","XLRE","XLB","XLC",
  // International ETFs (6)
  "EWZ","EWJ","FXI","INDA","EFA","EEM",
  // Commodities (4)
  "GLD","SLV","USO","DBC",
  // Bonds (3)
  "TLT","HYG","LQD",
  // Volatility (1)
  "VIXY",
  // Regime indicator (not in UNIVERSE, fetched for SPY SMA200 check)
  "SPY",
];

function dailyCacheFile(sym, warmupStart, end) {
  return path.join(DAILY_DIR, `${sym}_${warmupStart}_${end}.json`);
}
function earnCacheFile(start, end) {
  return path.join(EARN_DIR, `earnings_${start}_${end}.json`);
}

// Fetch all daily bars for one symbol via Alpaca getBarsV2
async function fetchDailyBars(sym, start, end) {
  const bars = [];
  const iter = alpaca.getBarsV2(sym, {
    start,
    end,
    timeframe: "1Day",
    adjustment: "split",
    feed: "iex",
  });
  for await (const b of iter) {
    bars.push({
      t: b.Timestamp,
      o: parseFloat(b.OpenPrice),
      h: parseFloat(b.HighPrice),
      l: parseFloat(b.LowPrice),
      c: parseFloat(b.ClosePrice),
      v: parseInt(b.Volume),
    });
  }
  return bars;
}

// Fetch earnings calendar from FMP for a date range
async function fetchEarnings(start, end) {
  if (!FMP_API_KEY) return {};
  try {
    const url = `https://financialmodelingprep.com/stable/earnings-calendar?from=${start}&to=${end}&apikey=${FMP_API_KEY}`;
    const events = await fetchJSON(url);
    if (!Array.isArray(events)) return {};
    const result = {};
    for (const e of events) {
      if (!e.symbol || !e.date) continue;
      if (!result[e.symbol]) result[e.symbol] = [];
      if (!result[e.symbol].includes(e.date)) result[e.symbol].push(e.date);
    }
    return result;
  } catch { return {}; }
}

// SSE endpoint: fetch all data with progress streaming
app.get("/api/backtest/fetch", async (req, res) => {
  const { start, end } = req.query;
  if (!start || !end) {
    return res.status(400).json({ error: "start and end required (YYYY-MM-DD)" });
  }

  // Extend start back 300 calendar days for indicator warm-up
  const warmupDate = new Date(start);
  warmupDate.setDate(warmupDate.getDate() - 300);
  const warmupStart = warmupDate.toISOString().split("T")[0];

  // SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const send = (type, data) => {
    res.write(`data: ${JSON.stringify({ type, ...data })}\n\n`);
  };

  send("start", { total: BT_SYMBOLS.length + 1, warmupStart });

  let completed = 0;
  const errors = [];

  // Fetch daily bars for each symbol
  for (const sym of BT_SYMBOLS) {
    const cacheFile = dailyCacheFile(sym, warmupStart, end);
    if (fs.existsSync(cacheFile)) {
      send("progress", { sym, status: "cached", completed: ++completed });
      continue;
    }
    try {
      send("progress", { sym, status: "fetching", completed });
      const bars = await fetchDailyBars(sym, warmupStart, end);
      fs.writeFileSync(cacheFile, JSON.stringify(bars));
      send("progress", { sym, status: "done", bars: bars.length, completed: ++completed });
    } catch (err) {
      errors.push(`${sym}: ${err.message}`);
      send("progress", { sym, status: "error", error: err.message, completed: ++completed });
    }
    // Small delay to avoid hammering the API
    await new Promise((r) => setTimeout(r, 200));
  }

  // Fetch earnings
  const earnFile = earnCacheFile(warmupStart, end);
  if (fs.existsSync(earnFile)) {
    send("progress", { sym: "EARNINGS", status: "cached", completed: ++completed });
  } else {
    try {
      send("progress", { sym: "EARNINGS", status: "fetching", completed });
      const earn = await fetchEarnings(warmupStart, end);
      fs.writeFileSync(earnFile, JSON.stringify(earn));
      send("progress", { sym: "EARNINGS", status: "done", completed: ++completed });
    } catch (err) {
      errors.push(`EARNINGS: ${err.message}`);
      send("progress", { sym: "EARNINGS", status: "error", error: err.message, completed: ++completed });
    }
  }

  send("done", { errors, warmupStart });
  res.end();
});

// Data retrieval endpoint: return cached data as JSON
app.get("/api/backtest/data", (req, res) => {
  const { start, end } = req.query;
  if (!start || !end) {
    return res.status(400).json({ error: "start and end required" });
  }

  const warmupDate = new Date(start);
  warmupDate.setDate(warmupDate.getDate() - 300);
  const warmupStart = warmupDate.toISOString().split("T")[0];

  const daily = {};
  for (const sym of BT_SYMBOLS) {
    const cacheFile = dailyCacheFile(sym, warmupStart, end);
    if (!fs.existsSync(cacheFile)) {
      return res.status(404).json({ error: `Missing cache for ${sym}. Run /api/backtest/fetch first.` });
    }
    daily[sym] = JSON.parse(fs.readFileSync(cacheFile, "utf8"));
  }

  const earnFile = earnCacheFile(warmupStart, end);
  const earnings = fs.existsSync(earnFile)
    ? JSON.parse(fs.readFileSync(earnFile, "utf8"))
    : {};

  res.json({ daily, earnings, warmupStart, start, end });
});

// Cache status check
app.get("/api/backtest/cache-status", (req, res) => {
  const { start, end } = req.query;
  if (!start || !end) return res.status(400).json({ error: "start and end required" });

  const warmupDate = new Date(start);
  warmupDate.setDate(warmupDate.getDate() - 300);
  const warmupStart = warmupDate.toISOString().split("T")[0];

  const status = {};
  for (const sym of BT_SYMBOLS) {
    const f = dailyCacheFile(sym, warmupStart, end);
    status[sym] = fs.existsSync(f) ? "cached" : "missing";
  }
  const earnFile = earnCacheFile(warmupStart, end);
  status.EARNINGS = fs.existsSync(earnFile) ? "cached" : "missing";
  const allCached = Object.values(status).every((s) => s === "cached");
  res.json({ status, allCached, warmupStart });
});

// ══════════════════════════════════════════
//  TRADE JOURNAL
// ══════════════════════════════════════════

app.post("/api/trade-journal", (req, res) => {
  try {
    const { timestamp, symbol, action, shares, price, strategy, ml_confidence, portfolio_value, pnl, notes } = req.body;
    insertTrade.run({
      timestamp: timestamp || new Date().toISOString(),
      symbol, action,
      shares: parseFloat(shares),
      price: parseFloat(price),
      strategy,
      ml_confidence: ml_confidence != null ? parseFloat(ml_confidence) : null,
      portfolio_value: portfolio_value != null ? parseFloat(portfolio_value) : null,
      pnl: pnl != null ? parseFloat(pnl) : null,
      notes: notes || null,
    });
    res.json({ ok: true });
  } catch (err) {
    console.error("Trade journal insert error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/trade-journal", (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 100, 500);
    const trades = db.prepare("SELECT * FROM trades ORDER BY id DESC LIMIT ?").all(limit);
    res.json(trades);
  } catch (err) {
    console.error("Trade journal query error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/trade-journal/snapshot", (req, res) => {
  try {
    const { date, portfolio_value, cash, positions_count, daily_pnl } = req.body;
    insertSnapshot.run({
      date,
      portfolio_value: parseFloat(portfolio_value),
      cash: parseFloat(cash),
      positions_count: parseInt(positions_count),
      daily_pnl: daily_pnl != null ? parseFloat(daily_pnl) : null,
    });
    res.json({ ok: true });
  } catch (err) {
    console.error("Snapshot insert error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/trade-journal/snapshots", (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 30, 365);
    const snapshots = db.prepare("SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT ?").all(limit);
    res.json(snapshots);
  } catch (err) {
    console.error("Snapshot query error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ══════════════════════════════════════════
//  HEALTH CHECK
// ══════════════════════════════════════════

app.get("/api/health", async (req, res) => {
  try {
    const account = await alpaca.getAccount();
    res.json({
      status: "ok",
      connected: true,
      account_status: account.status,
      paper: true,
    });
  } catch (err) {
    res.json({ status: "error", connected: false, error: err.message });
  }
});

// ══════════════════════════════════════════
//  TRADING ENGINE — Server-side autonomous trading
// ══════════════════════════════════════════

const createTradingEngine = require("./tradingEngine");

// Build the FMP earnings fetcher for the trading engine (returns { SYMBOL: "nearest-date" })
function fetchEarningsFromFMP(symbols) {
  if (!FMP_API_KEY) return Promise.resolve({});
  const fmt = (d) => d.toISOString().split("T")[0];
  const from = fmt(new Date());
  const to   = fmt(new Date(Date.now() + 7 * 24 * 60 * 60 * 1000));
  const url  = `https://financialmodelingprep.com/stable/earnings-calendar?from=${from}&to=${to}&apikey=${FMP_API_KEY}`;
  return fetchJSON(url).then((events) => {
    if (!Array.isArray(events)) return {};
    const result = {};
    const symSet = new Set(symbols.map(s => s.toUpperCase()));
    for (const e of events) {
      if (!e.symbol || !e.date) continue;
      if (symSet.size && !symSet.has(e.symbol)) continue;
      if (!result[e.symbol] || e.date < result[e.symbol]) {
        result[e.symbol] = e.date;
      }
    }
    return result;
  }).catch(() => ({}));
}

const engine = createTradingEngine({
  alpaca,
  insertTrade,
  insertSnapshot,
  fetchEarningsFromFMP,
});

// Get full trading state (polled by React frontend)
app.get("/api/trading-state", (req, res) => {
  try {
    const state = engine.getState();
    res.json(state);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get activity feed / logs
app.get("/api/activity-feed", (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 200, 500);
    const feed = engine.getActivityFeed(limit);
    res.json(feed);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Start trading engine
app.post("/api/trading/start", async (req, res) => {
  try {
    if (engine.isRunning()) {
      return res.json({ status: "already_running" });
    }
    await engine.start();
    res.json({ status: "started" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Stop trading engine
app.post("/api/trading/stop", (req, res) => {
  try {
    engine.stop();
    res.json({ status: "stopped" });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Engine status (lightweight check)
app.get("/api/trading/status", (req, res) => {
  res.json({ running: engine.isRunning() });
});

// ══════════════════════════════════════════
//  START
// ══════════════════════════════════════════

const PORT = process.env.PORT || 3001;
app.listen(PORT, async () => {
  console.log(`\n⚡ AutoTrader API server running on http://localhost:${PORT}`);
  console.log(`   Mode: PAPER TRADING`);
  console.log(`   Endpoints:`);
  console.log(`     GET  /api/health      — connection check`);
  console.log(`     GET  /api/account     — account info`);
  console.log(`     GET  /api/positions   — open positions`);
  console.log(`     POST /api/orders      — place order`);
  console.log(`     GET  /api/orders      — list orders`);
  console.log(`     GET  /api/clock       — market hours`);
  console.log(`     GET  /api/snapshots   — live prices`);
  console.log(`     GET  /api/bars/:sym   — historical bars`);
  console.log(`     GET  /api/earnings    — upcoming earnings (FMP)`);
  console.log(`     GET  /api/backtest/fetch   — download + cache historical data (SSE)`);
  console.log(`     GET  /api/backtest/data    — retrieve cached backtest data`);
  console.log(`     GET  /api/backtest/cache-status — check cache completeness`);
  console.log(`     GET  /api/trade-journal  — recent trades`);
  console.log(`     POST /api/trade-journal  — record a trade`);
  console.log(`     GET  /api/trade-journal/snapshots — daily snapshots`);
  console.log(`     GET  /api/trading-state  — full engine state (poll)`);
  console.log(`     GET  /api/activity-feed  — engine activity log`);
  console.log(`     POST /api/trading/start  — start trading engine`);
  console.log(`     POST /api/trading/stop   — stop trading engine`);
  console.log(`     GET  /api/trading/status — engine running status\n`);

  // Auto-start trading engine on server boot
  console.log("🚀 Auto-starting trading engine...");
  try {
    await engine.start();
    console.log("✅ Trading engine started successfully.");
  } catch (err) {
    console.error("❌ Trading engine failed to start:", err.message);
    console.error("   The server is still running — you can start the engine manually via POST /api/trading/start");
  }
});
