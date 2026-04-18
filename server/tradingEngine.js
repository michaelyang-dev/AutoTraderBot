// ══════════════════════════════════════════════════════════════════════
//  TRADING ENGINE — Server-side consolidated trading logic
//  Runs all signal analysis, regime detection, trend following,
//  and trade execution directly on the Express server.
//  No React dependency — uses Alpaca SDK + Node http for ML signals.
// ══════════════════════════════════════════════════════════════════════

const http = require("http");
const notify = require("./notifications");

// ══════════════════════════════════════════
//  CONSTANTS (inlined from frontend config)
// ══════════════════════════════════════════

const INITIAL_CASH = 100000;

const RISK = {
  MAX_POSITION_PCT: 0.15,
  STOP_LOSS_PCT: -0.08,
  TAKE_PROFIT_PCT: 0.15,
  MAX_OPEN_POSITIONS: 6,
  MAX_CASH_DEPLOY_PCT: 0.90,
  REBALANCE_INTERVAL: 5,
  TRAILING_STOP_PCT: 0.08,
  USE_TRAILING_STOP: true,
  ATR_TARGET_PCT: 0.01,
  MIN_POSITION_PCT: 0.03,
  LOSS_COOLDOWN_CYCLES: 3,
  SECTOR_MAX_POSITIONS: { International: 2, Commodity: 2, Bond: 2, Volatility: 1 },
  VOLUME_CONFIRM_RATIO: 1.5,
};

const MARKET_HOURS = { OPEN_BUFFER_MINS: 15, CLOSE_BUFFER_MINS: 30 };

const NEVER_BUY = new Set([
  "VIXY","UVXY","VXX","SVXY",
  "TQQQ","SQQQ","QQQ3",
  "SPXU","SPXS","SDS","UPRO",
  "QID","SDOW",
  "LABU","LABD",
  "JNUG","JDST","NUGT","DUST",
  "FNGU","FNGD",
  "SOXL","SOXS",
  "YANG","YINN",
]);

const UNIVERSE = [
  { sym: "AAPL", base: 189, sector: "Tech" }, { sym: "GOOGL", base: 141, sector: "Tech" },
  { sym: "MSFT", base: 378, sector: "Tech" }, { sym: "AMZN", base: 178, sector: "Consumer" },
  { sym: "TSLA", base: 248, sector: "Auto" }, { sym: "NVDA", base: 880, sector: "Semis" },
  { sym: "META", base: 505, sector: "Tech" }, { sym: "NFLX", base: 628, sector: "Media" },
  { sym: "AMD", base: 164, sector: "Semis" }, { sym: "JPM", base: 196, sector: "Finance" },
  { sym: "V", base: 278, sector: "Finance" }, { sym: "UNH", base: 527, sector: "Health" },
  { sym: "XLE", base: 93, sector: "Energy" }, { sym: "XLF", base: 48, sector: "Finance" },
  { sym: "XLV", base: 145, sector: "Health" }, { sym: "XLI", base: 130, sector: "Industrial" },
  { sym: "XLK", base: 218, sector: "Tech" }, { sym: "XLY", base: 195, sector: "Consumer" },
  { sym: "XLP", base: 79, sector: "Staples" }, { sym: "XLU", base: 72, sector: "Utilities" },
  { sym: "XLRE", base: 39, sector: "REIT" }, { sym: "XLB", base: 85, sector: "Materials" },
  { sym: "XLC", base: 92, sector: "Media" },
  { sym: "EWZ", base: 29, sector: "International" }, { sym: "EWJ", base: 71, sector: "International" },
  { sym: "FXI", base: 29, sector: "International" }, { sym: "INDA", base: 50, sector: "International" },
  { sym: "EFA", base: 79, sector: "International" }, { sym: "EEM", base: 43, sector: "International" },
  { sym: "GLD", base: 285, sector: "Commodity" }, { sym: "SLV", base: 31, sector: "Commodity" },
  { sym: "USO", base: 70, sector: "Commodity" }, { sym: "DBC", base: 22, sector: "Commodity" },
  { sym: "TLT", base: 85, sector: "Bond" }, { sym: "HYG", base: 77, sector: "Bond" },
  { sym: "LQD", base: 104, sector: "Bond" },
  { sym: "VIXY", base: 14, sector: "Volatility" },
];

const CONSENSUS_THRESHOLDS = { STRONG_BUY: 2, BUY: 1, SELL: -1, STRONG_SELL: -2 };
const REGIME_RECOVERY_DAYS = 3;
const SPY_IDLE_RESERVE_PCT = 0.30;
const SPY_IDLE_THRESHOLD_PCT = 0.20;
const SPY_IDLE_INVEST_PCT = 0.85;
const CIRCUIT_BREAKER_PCT = 0.02;
const PRICE_POLL_MS = 15000;
const TRADE_CYCLE_MS = 60000;

const SECTOR_MAP = {
  AAPL: "Tech", MSFT: "Tech", GOOGL: "Tech", GOOG: "Tech", META: "Tech",
  NVDA: "Semis", AMD: "Semis", INTC: "Semis", QCOM: "Semis", AVGO: "Semis", MU: "Semis", TSM: "Semis",
  CRM: "Tech", ORCL: "Tech", SAP: "Tech", ADBE: "Tech", NOW: "Tech", SNOW: "Tech",
  PLTR: "Tech", UBER: "Tech", LYFT: "Tech", SHOP: "Tech", TWLO: "Tech", ZM: "Tech",
  NET: "Tech", DDOG: "Tech", MDB: "Tech", CRWD: "Tech", ZS: "Tech", OKTA: "Tech",
  PANW: "Tech", FTNT: "Tech", CYBR: "Tech",
  AMZN: "Consumer", TSLA: "Auto", GM: "Auto", F: "Auto",
  WMT: "Consumer", TGT: "Consumer", COST: "Consumer", HD: "Consumer", LOW: "Consumer",
  NKE: "Consumer", SBUX: "Consumer", MCD: "Consumer", YUM: "Consumer", CMG: "Consumer",
  BABA: "Consumer", JD: "Consumer", PDD: "Consumer",
  JPM: "Finance", BAC: "Finance", WFC: "Finance", GS: "Finance", MS: "Finance",
  C: "Finance", BLK: "Finance", AXP: "Finance", V: "Finance", MA: "Finance",
  PYPL: "Finance", SQ: "Finance", COIN: "Finance", SCHW: "Finance", USB: "Finance",
  UNH: "Health", JNJ: "Health", PFE: "Health", ABBV: "Health", MRK: "Health",
  LLY: "Health", BMY: "Health", AMGN: "Health", GILD: "Health", BIIB: "Health",
  MRNA: "Health", BNTX: "Health", CVS: "Health", CI: "Health", HUM: "Health",
  MDT: "Health", ABT: "Health", TMO: "Health", DHR: "Health", ISRG: "Health",
  NFLX: "Media", DIS: "Media", PARA: "Media", WBD: "Media", CMCSA: "Media",
  T: "Media", VZ: "Media", TMUS: "Media",
  SPOT: "Media", SNAP: "Media", PINS: "Media", RDDT: "Media",
  XOM: "Energy", CVX: "Energy", COP: "Energy", SLB: "Energy", EOG: "Energy",
  OXY: "Energy", PSX: "Energy", VLO: "Energy", MPC: "Energy",
  BA: "Industrial", CAT: "Industrial", GE: "Industrial", HON: "Industrial",
  LMT: "Industrial", RTX: "Industrial", NOC: "Industrial", DE: "Industrial",
  UPS: "Industrial", FDX: "Industrial", CSX: "Industrial",
  AMT: "REIT", PLD: "REIT", EQIX: "REIT", SPG: "REIT",
  NEE: "Utilities", SO: "Utilities", DUK: "Utilities",
};

function getSector(sym) {
  const u = UNIVERSE.find(x => x.sym === sym);
  if (u) return u.sector;
  return SECTOR_MAP[sym] || "Other";
}

// ══════════════════════════════════════════
//  HELPER FUNCTIONS
// ══════════════════════════════════════════

function fmtVol(v) {
  return v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : (v / 1e3).toFixed(0) + "k";
}

function daysUntilEarnings(dateStr) {
  const nowET = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
  nowET.setHours(0, 0, 0, 0);
  const target = new Date(dateStr + "T00:00:00");
  return Math.round((target - nowET) / 86400000);
}

function getETTime(isoTimestamp) {
  return new Date(
    new Date(isoTimestamp).toLocaleString("en-US", { timeZone: "America/New_York" })
  );
}

function marketWindowMins(clock) {
  const et = getETTime(clock.timestamp);
  const totalMins = et.getHours() * 60 + et.getMinutes();
  const minutesSinceOpen = totalMins - (9 * 60 + 30);
  const minutesUntilClose = 16 * 60 - totalMins;
  return { minutesSinceOpen, minutesUntilClose };
}

// ══════════════════════════════════════════
//  TECHNICAL INDICATORS
// ══════════════════════════════════════════

function sma(arr, period) {
  if (arr.length < period) return null;
  return arr.slice(-period).reduce((a, b) => a + b, 0) / period;
}

function ema(arr, period) {
  if (arr.length < period) return null;
  const k = 2 / (period + 1);
  let e = arr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < arr.length; i++) {
    e = arr[i] * k + e * (1 - k);
  }
  return e;
}

function rsi(arr, period = 14) {
  if (arr.length < period + 1) return 50;
  let gains = 0, losses = 0;
  for (let i = arr.length - period; i < arr.length; i++) {
    const d = arr[i] - arr[i - 1];
    if (d > 0) gains += d; else losses -= d;
  }
  const rs = gains / (losses || 0.001);
  return 100 - 100 / (1 + rs);
}

function macd(arr) {
  const e12 = ema(arr, 12);
  const e26 = ema(arr, 26);
  if (e12 === null || e26 === null) return { m: 0, s: 0 };
  const m = e12 - e26;
  return { m, s: m * 0.82 };
}

function atr(arr, period = 14) {
  if (arr.length < period + 1) return null;
  let sum = 0;
  for (let i = arr.length - period; i < arr.length; i++) sum += Math.abs(arr[i] - arr[i - 1]);
  return sum / period;
}

function avgVolume(arr, period = 20) {
  if (!arr || arr.length < period + 1) return null;
  const w = arr.slice(-(period + 1), -1);
  return w.reduce((a, b) => a + b, 0) / w.length;
}

function bollinger(arr, period = 20) {
  if (arr.length < period) return null;
  const sl = arr.slice(-period);
  const mean = sl.reduce((a, b) => a + b, 0) / period;
  const std = Math.sqrt(sl.reduce((a, b) => a + (b - mean) ** 2, 0) / period);
  return { upper: mean + 2 * std, middle: mean, lower: mean - 2 * std };
}

function weeklyTrend(prices) {
  if (!prices || prices.length < 150) return null;
  const weekly = [];
  for (let i = prices.length - 1; i >= 0 && weekly.length < 30; i -= 5) {
    weekly.unshift(prices[i]);
  }
  if (weekly.length < 30) return null;
  const smaFast = sma(weekly, 5);
  const smaSlow = sma(weekly, 15);
  if (smaFast === null || smaSlow === null) return null;
  return { trend: smaFast > smaSlow ? "BULLISH" : "BEARISH", smaFast, smaSlow };
}

// ══════════════════════════════════════════
//  SIGNAL ENGINE — Multi-strategy consensus
// ══════════════════════════════════════════

function getSignals(prices) {
  if (prices.length < 35) {
    return { consensus: "WAIT", signals: {}, score: 0, indicators: {} };
  }

  const signals = {};
  let buyVotes = 0;
  let sellVotes = 0;

  // SMA Crossover
  const s10 = sma(prices, 10);
  const s30 = sma(prices, 30);
  const ps10 = sma(prices.slice(0, -1), 10);
  const ps30 = sma(prices.slice(0, -1), 30);

  if (s10 > s30 && ps10 <= ps30) {
    signals.sma = "BUY"; buyVotes++;
  } else if (s10 < s30 && ps10 >= ps30) {
    signals.sma = "SELL"; sellVotes++;
  } else if (s10 > s30) {
    signals.sma = "BULLISH"; buyVotes += 0.3;
  } else {
    signals.sma = "BEARISH"; sellVotes += 0.3;
  }

  // RSI
  const rsiVal = rsi(prices);
  if (rsiVal < 28) { signals.rsi = "BUY"; buyVotes++; }
  else if (rsiVal > 72) { signals.rsi = "SELL"; sellVotes++; }
  else if (rsiVal < 40) { signals.rsi = "BULLISH"; buyVotes += 0.3; }
  else if (rsiVal > 60) { signals.rsi = "BEARISH"; sellVotes += 0.3; }
  else { signals.rsi = "NEUTRAL"; }

  // MACD
  const mc = macd(prices);
  const pmc = macd(prices.slice(0, -1));
  if (mc.m > mc.s && pmc.m <= pmc.s) {
    signals.macd = "BUY"; buyVotes++;
  } else if (mc.m < mc.s && pmc.m >= pmc.s) {
    signals.macd = "SELL"; sellVotes++;
  } else if (mc.m > mc.s) {
    signals.macd = "BULLISH"; buyVotes += 0.3;
  } else {
    signals.macd = "BEARISH"; sellVotes += 0.3;
  }

  // Bollinger Bands
  const bb = bollinger(prices);
  const currentPrice = prices[prices.length - 1];
  if (bb) {
    if (currentPrice <= bb.lower) { signals.boll = "BUY"; buyVotes++; }
    else if (currentPrice >= bb.upper) { signals.boll = "SELL"; sellVotes++; }
    else if (currentPrice < bb.middle) { signals.boll = "BULLISH"; buyVotes += 0.2; }
    else { signals.boll = "BEARISH"; sellVotes += 0.2; }
  }

  // Momentum
  const momVal = prices.length > 12
    ? (currentPrice - prices[prices.length - 13]) / prices[prices.length - 13]
    : 0;

  if (momVal > 0.035) { signals.mom = "BUY"; buyVotes++; }
  else if (momVal < -0.025) { signals.mom = "SELL"; sellVotes++; }
  else if (momVal > 0) { signals.mom = "BULLISH"; buyVotes += 0.2; }
  else { signals.mom = "BEARISH"; sellVotes += 0.2; }

  // Consensus
  const score = buyVotes - sellVotes;
  let consensus = "HOLD";
  if (score >= CONSENSUS_THRESHOLDS.STRONG_BUY) consensus = "STRONG BUY";
  else if (score >= CONSENSUS_THRESHOLDS.BUY) consensus = "BUY";
  else if (score <= CONSENSUS_THRESHOLDS.STRONG_SELL) consensus = "STRONG SELL";
  else if (score <= CONSENSUS_THRESHOLDS.SELL) consensus = "SELL";

  return {
    consensus,
    signals,
    score,
    indicators: { rsi: rsiVal, sma10: s10, sma30: s30, macd: mc, momentum: momVal, bollinger: bb },
  };
}

// ══════════════════════════════════════════
//  REGIME ENGINE — SPY market trend filter
// ══════════════════════════════════════════

function computeRegime(spyPrices) {
  if (!spyPrices || spyPrices.length < 200) {
    return { regime: "BULLISH", sma50: null, sma200: null, consecutiveDaysAbove50: 0 };
  }

  const price = spyPrices[spyPrices.length - 1];
  const sma50Val = sma(spyPrices, 50);
  const sma200Val = sma(spyPrices, 200);

  let regime;
  if (price > sma50Val && price > sma200Val) regime = "BULLISH";
  else if (price > sma200Val) regime = "CAUTIOUS";
  else regime = "BEARISH";

  let consecutiveDaysAbove50 = 0;
  for (let i = spyPrices.length - 1; i >= 0 && consecutiveDaysAbove50 < 10; i--) {
    if (spyPrices[i] <= sma50Val) break;
    consecutiveDaysAbove50++;
  }

  return { regime, sma50: sma50Val, sma200: sma200Val, consecutiveDaysAbove50 };
}

// ══════════════════════════════════════════
//  TREND ENGINE — Long-term trend detection
// ══════════════════════════════════════════

function computeTrendStatus(prices) {
  if (!prices || prices.length < 200) return null;

  const price = prices[prices.length - 1];
  const sma50 = sma(prices, 50);
  const sma200v = sma(prices, 200);
  if (sma50 === null || sma200v === null) return null;

  const priceAbove200 = price > sma200v;

  const lookback = Math.min(40, prices.length - 199);
  let daysAbove200 = 0;
  for (let i = 0; i < lookback; i++) {
    const endIdx = prices.length - i;
    const barPrice = prices[endIdx - 1];
    let barSma200 = 0;
    for (let j = endIdx - 200; j < endIdx; j++) barSma200 += prices[j];
    barSma200 /= 200;
    if (barPrice > barSma200) daysAbove200++;
  }

  const isStrongUptrend = priceAbove200 && sma50 > sma200v && daysAbove200 >= 30;

  return { isStrongUptrend, price, sma50, sma200: sma200v, daysAbove200, priceAbove200 };
}

// ══════════════════════════════════════════
//  ML SIGNAL FETCH (Node http)
// ══════════════════════════════════════════

function fetchMLSignals() {
  return new Promise((resolve) => {
    const req = http.get("http://localhost:5001/signals", { timeout: 5000 }, (res) => {
      let data = "";
      res.on("data", (chunk) => data += chunk);
      res.on("end", () => {
        try { resolve(JSON.parse(data)); } catch { resolve(null); }
      });
    });
    req.on("error", () => resolve(null));
    req.on("timeout", () => { req.destroy(); resolve(null); });
  });
}

// ══════════════════════════════════════════
//  FACTORY — createTradingEngine
// ══════════════════════════════════════════

module.exports = function createTradingEngine({ alpaca, insertTrade, insertSnapshot, fetchEarningsFromFMP }) {

  // ── Alpaca SDK wrappers ──

  async function getAccount() {
    const acct = await alpaca.getAccount();
    return {
      cash: parseFloat(acct.cash),
      portfolio_value: parseFloat(acct.portfolio_value),
    };
  }

  async function getPositions() {
    const raw = await alpaca.getPositions();
    return raw.map(p => ({
      symbol: p.symbol,
      qty: parseFloat(p.qty),
      avg_entry_price: parseFloat(p.avg_entry_price),
      current_price: parseFloat(p.current_price),
      unrealized_pl: parseFloat(p.unrealized_pl),
      unrealized_plpc: parseFloat(p.unrealized_plpc),
      market_value: parseFloat(p.market_value),
    }));
  }

  async function getClock() {
    const clock = await alpaca.getClock();
    return {
      is_open: clock.is_open,
      timestamp: clock.timestamp,
      next_open: clock.next_open,
      next_close: clock.next_close,
    };
  }

  async function placeOrder({ symbol, qty, side, type = "market", time_in_force = "day" }) {
    try {
      return await alpaca.createOrder({ symbol, qty, side, type, time_in_force });
    } catch (err) {
      notify.send(`🚨 ORDER REJECTED — ${symbol} ${side} ${qty} shares | Reason: ${err.message}`, { deduplicate: true, immediate: true });
      throw err;
    }
  }

  async function closePosition(symbol) {
    try {
      return await alpaca.closePosition(symbol);
    } catch (err) {
      notify.send(`🚨 ORDER REJECTED — ${symbol} close | Reason: ${err.message}`, { deduplicate: true, immediate: true });
      throw err;
    }
  }

  async function getOrders(status = "all", limit = 50) {
    return await alpaca.getOrders({ status, limit });
  }

  async function fetchBars(symbol, limit = 150) {
    const calDays = Math.ceil(limit * 1.6) + 10;
    const start = new Date();
    start.setDate(start.getDate() - calDays);
    const startISO = start.toISOString().split("T")[0];

    const bars = [];
    const iter = alpaca.getBarsV2(symbol, { timeframe: "1Day", start: startISO, adjustment: "split" });
    for await (const bar of iter) {
      bars.push({ c: parseFloat(bar.ClosePrice), v: parseInt(bar.Volume) });
    }
    return bars.slice(-limit);
  }

  async function fetchSnapshots(symbols) {
    const snaps = await alpaca.getSnapshots(symbols);
    const result = {};
    for (const snap of snaps) {
      if (!snap.symbol) continue;
      result[snap.symbol] = {
        price: parseFloat(snap.LatestTrade?.Price || snap.DailyBar?.ClosePrice || 0),
        volume: parseInt(snap.DailyBar?.Volume || 0),
      };
    }
    return result;
  }

  // ── Trade journal ──

  function recordTrade(trade) {
    try {
      insertTrade.run({
        timestamp: trade.timestamp || new Date().toISOString(),
        symbol: trade.symbol,
        action: trade.action,
        shares: parseFloat(trade.shares),
        price: parseFloat(trade.price),
        strategy: trade.strategy,
        ml_confidence: trade.ml_confidence != null ? parseFloat(trade.ml_confidence) : null,
        portfolio_value: trade.portfolio_value != null ? parseFloat(trade.portfolio_value) : null,
        pnl: trade.pnl != null ? parseFloat(trade.pnl) : null,
        notes: trade.notes || null,
      });
    } catch (err) {
      console.error("Trade journal write failed:", err.message);
    }
  }

  function recordDailySnapshot(snap) {
    try {
      insertSnapshot.run({
        date: snap.date,
        portfolio_value: parseFloat(snap.portfolio_value),
        cash: parseFloat(snap.cash),
        positions_count: parseInt(snap.positions_count),
        daily_pnl: snap.daily_pnl != null ? parseFloat(snap.daily_pnl) : null,
      });
    } catch (err) {
      console.error("Daily snapshot write failed:", err.message);
    }
  }

  // ── Earnings cache (4h TTL) ──

  let _earningsCache = { data: {}, fetchedAt: 0 };
  const EARNINGS_TTL_MS = 4 * 60 * 60 * 1000;

  async function fetchEarnings(symbols) {
    if (Date.now() - _earningsCache.fetchedAt < EARNINGS_TTL_MS) {
      return _earningsCache.data;
    }
    try {
      const data = await fetchEarningsFromFMP(symbols);
      _earningsCache = { data, fetchedAt: Date.now() };
      return data;
    } catch {
      return _earningsCache.data;
    }
  }

  // ══════════════════════════════════════════
  //  STATE
  // ══════════════════════════════════════════

  let running = false;
  let priceHist = {};
  let volHist = {};
  let regime = "BULLISH";
  let prevRegime = "BULLISH";
  let mlSignals = null;
  let mlStatus = "down";
  let cash = 0;
  let portfolioValue = 0;
  let initialPortfolioValue = null;
  let positions = {};
  let positionsRaw = [];
  let marketOpen = false;
  let connected = false;
  let error = null;
  let idleSpyShares = 0;
  let trailingPeaks = {};
  let cooldowns = {};
  let trendPositions = {};
  let trendBreakCounts = {};
  let circuitBreaker = { date: null, morningValue: null, tripped: false };
  let cycleNumber = 0;
  let tick = 0;
  const activityLog = [];
  const portfolioHist = [];
  let tradeCount = { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 };
  let pricePollInterval = null;
  let tradeCycleInterval = null;
  let prevMlStatus = "down";
  let prevMarketOpen = false;
  let dailyStats = {
    date: null, buys: 0, sells: 0, wins: 0, losses: 0,
    summarySent: false, weekStartValue: null,
  };

  // ── Activity log ──

  function addLog(msg, type = "info") {
    const entry = { msg, type, tick, time: new Date().toLocaleTimeString() };
    activityLog.push(entry);
    if (activityLog.length > 500) activityLog.splice(0, activityLog.length - 500);
    console.log(`[${type}] ${msg}`);
  }

  // ══════════════════════════════════════════
  //  INIT — Load account, positions, price history
  // ══════════════════════════════════════════

  async function init() {
    try {
      addLog("Initializing trading engine...", "system");

      // 1. Check health / connectivity
      const acct = await getAccount();
      cash = acct.cash;
      portfolioValue = acct.portfolio_value;
      initialPortfolioValue = portfolioValue;
      connected = true;
      addLog(`Connected to Alpaca. Cash: $${cash.toFixed(2)}, Portfolio: $${portfolioValue.toFixed(2)}`, "system");

      // 2. Load positions
      positionsRaw = await getPositions();
      positions = {};
      for (const p of positionsRaw) {
        positions[p.symbol] = {
          shares: p.qty,
          avgPrice: p.avg_entry_price,
          currentPrice: p.current_price,
          unrealizedPl: p.unrealized_pl,
          unrealizedPlPct: p.unrealized_plpc,
          marketValue: p.market_value,
        };
      }
      addLog(`Loaded ${positionsRaw.length} position(s): ${positionsRaw.map(p => p.symbol).join(", ") || "none"}`, "system");

      // 3. Restore idle SPY tracking
      const spyPos = positionsRaw.find(p => p.symbol === "SPY");
      if (spyPos) {
        idleSpyShares = spyPos.qty;
        addLog(`Restored idle SPY tracking: ${idleSpyShares} shares`, "system");
      }

      // 4. Check clock
      const clock = await getClock();
      marketOpen = clock.is_open;
      addLog(`Market is ${marketOpen ? "OPEN" : "CLOSED"}. Next ${marketOpen ? "close" : "open"}: ${new Date(marketOpen ? clock.next_close : clock.next_open).toLocaleString()}`, "system");

      // 5. Load historical bars for all UNIVERSE symbols + SPY
      addLog("Fetching historical bars for all universe symbols...", "system");
      const symbols = UNIVERSE.map(s => s.sym);

      const results = await Promise.allSettled(
        symbols.map(async (sym) => {
          const bars = await fetchBars(sym, 150);
          return { sym, closes: bars.map(b => b.c), volumes: bars.map(b => b.v) };
        })
      );

      let loadedCount = 0;
      results.forEach((result) => {
        if (result.status === "fulfilled") {
          const { sym, closes, volumes } = result.value;
          priceHist[sym] = closes;
          volHist[sym] = volumes;
          loadedCount++;
          if (closes.length < 35) {
            addLog(`Warning: ${sym} only has ${closes.length} bars (need 35 for indicators)`, "system");
          }
        } else {
          addLog(`Failed to fetch bars: ${result.reason}`, "error");
        }
      });

      addLog(`Price history loaded for ${loadedCount}/${symbols.length} symbols`, "system");

      // Always fetch SPY with 220 bars for regime filter
      if (!symbols.includes("SPY")) {
        try {
          const spyBars = await fetchBars("SPY", 220);
          priceHist.SPY = spyBars.map(b => b.c);
          addLog(`SPY bars loaded: ${priceHist.SPY.length} bars for regime filter`, "system");
        } catch (err) {
          addLog(`Failed to fetch SPY bars for regime filter: ${err.message}`, "error");
        }
      }

      // 6. Initial regime calculation
      const regimeResult = computeRegime(priceHist.SPY);
      regime = regimeResult.regime;
      prevRegime = regime;
      addLog(`Initial regime: ${regime} (SPY SMA50: ${regimeResult.sma50?.toFixed(2) || "N/A"}, SMA200: ${regimeResult.sma200?.toFixed(2) || "N/A"})`, "system");

      error = null;
      addLog("Trading engine initialized successfully", "system");
    } catch (err) {
      error = err.message;
      connected = false;
      addLog(`Initialization failed: ${err.message}`, "error");
      notify.send(`🚨 ALPACA CONNECTION FAILED — cannot execute trades. Error: ${err.message}`, { deduplicate: true, immediate: true });
      throw err;
    }
  }

  // ══════════════════════════════════════════
  //  PRICE POLL (every 15s)
  // ══════════════════════════════════════════

  async function pollPrices() {
    try {
      tick++;
      const symbols = UNIVERSE.map(s => s.sym);
      const snapshotSymbols = symbols.includes("SPY") ? symbols : [...symbols, "SPY"];
      const snapshots = await fetchSnapshots(snapshotSymbols);

      // Update price history
      const nextHist = {};
      const nextVol = {};
      for (const sym of symbols) {
        const prev = priceHist[sym] || [];
        const prevVols = volHist[sym] || [];
        const snap = snapshots[sym];
        if (snap && snap.price > 0) {
          nextHist[sym] = [...prev.slice(-100), snap.price];
          nextVol[sym] = [...prevVols.slice(-100), snap.volume || 0];
        } else {
          nextHist[sym] = prev;
          nextVol[sym] = prevVols;
        }
      }

      // Keep SPY history (250 entries for SMA200)
      const spyPrev = priceHist.SPY || [];
      const spySnap = snapshots.SPY;
      nextHist.SPY = spySnap && spySnap.price > 0
        ? [...spyPrev.slice(-250), spySnap.price]
        : spyPrev;

      priceHist = nextHist;
      volHist = nextVol;

      // Refresh account, positions, clock
      const [acct, rawPos, clock] = await Promise.all([
        getAccount(),
        getPositions(),
        getClock(),
      ]);

      cash = acct.cash;
      portfolioValue = acct.portfolio_value;
      if (initialPortfolioValue === null) initialPortfolioValue = portfolioValue;
      positionsRaw = rawPos;
      marketOpen = clock.is_open;

      positions = {};
      for (const p of positionsRaw) {
        positions[p.symbol] = {
          shares: p.qty,
          avgPrice: p.avg_entry_price,
          currentPrice: p.current_price,
          unrealizedPl: p.unrealized_pl,
          unrealizedPlPct: p.unrealized_plpc,
          marketValue: p.market_value,
        };
      }

      // Portfolio history (keep last 200)
      portfolioHist.push({ tick, value: portfolioValue, time: Date.now() });
      if (portfolioHist.length > 200) portfolioHist.splice(0, portfolioHist.length - 200);

      // Fetch ML signals — server returns { signals: [...], is_stale: bool }
      const mlData = await fetchMLSignals();
      if (mlData && Array.isArray(mlData.signals) && !mlData.is_stale) {
        mlSignals = mlData.signals;
        mlStatus = "ok";
      } else if (mlData && Array.isArray(mlData.signals) && mlData.is_stale) {
        mlSignals = null;   // stale → don't use for trading decisions
        mlStatus = "stale";
      } else {
        mlSignals = null;
        mlStatus = "down";
      }

      // ML status transition notifications
      if (prevMlStatus === "ok" && mlStatus !== "ok") {
        notify.send("🚨 ML SERVER DOWN — falling back to consensus engine. Check pm2 logs.", { deduplicate: true, immediate: true });
      } else if (prevMlStatus !== "ok" && mlStatus === "ok") {
        notify.send("✅ ML SERVER RECOVERED — ML signals active again.", { immediate: true });
      }
      prevMlStatus = mlStatus;

      // Compute regime with recovery logic
      const regimeResult = computeRegime(priceHist.SPY);
      prevRegime = regime;

      if (prevRegime === "BEARISH" && regimeResult.regime !== "BEARISH") {
        if (regimeResult.consecutiveDaysAbove50 >= REGIME_RECOVERY_DAYS) {
          regime = regimeResult.regime;
          addLog(`Regime recovery: ${prevRegime} -> ${regime} (${regimeResult.consecutiveDaysAbove50} days above 50-SMA)`, "system");
        } else {
          regime = "BEARISH";
        }
      } else {
        regime = regimeResult.regime;
      }

      // Market open transition notification
      if (marketOpen && !prevMarketOpen) {
        const buyCount = mlSignals ? mlSignals.filter(s => s.signal === "BUY").length : 0;
        notify.send(`🔔 MARKET OPEN — Bot is trading. Regime: ${regime}. ML signals: ${buyCount} BUY.`);
      }
      prevMarketOpen = marketOpen;

      connected = true;
      error = null;
    } catch (err) {
      addLog(`Price poll error: ${err.message}`, "error");
      notify.send(`🚨 ALPACA CONNECTION FAILED — cannot execute trades. Error: ${err.message}`, { deduplicate: true, immediate: true });
      error = err.message;
    }
  }

  // ══════════════════════════════════════════
  //  TRADE CYCLE (every 60s) — Full executor
  // ══════════════════════════════════════════

  async function runTradeCycle() {
    try {
      cycleNumber++;

      const isOnCooldown = (sym) => {
        const lossAt = cooldowns[sym];
        return lossAt !== undefined && (cycleNumber - lossAt) < RISK.LOSS_COOLDOWN_CYCLES;
      };

      // Fetch live state from Alpaca
      const [account, currentPositions, clock] = await Promise.all([
        getAccount(),
        getPositions(),
        getClock(),
      ]);

      if (!clock.is_open) {
        addLog(`Market is closed. Next open: ${new Date(clock.next_open).toLocaleString()}`, "system");
        return;
      }

      // Market hours window check
      const { minutesSinceOpen, minutesUntilClose } = marketWindowMins(clock);

      if (minutesSinceOpen < MARKET_HOURS.OPEN_BUFFER_MINS) {
        addLog(`Opening buffer: ${(MARKET_HOURS.OPEN_BUFFER_MINS - minutesSinceOpen).toFixed(0)} min until trading begins (avoiding open volatility).`, "system");
        return;
      }

      let skipNewBuys = minutesUntilClose < MARKET_HOURS.CLOSE_BUFFER_MINS;
      if (skipNewBuys) {
        addLog(`Close buffer: ${minutesUntilClose.toFixed(0)} min until close -- stop-loss checks only, no new buys.`, "system");
      }

      let cycleCash = account.cash;
      const cyclePortfolioValue = account.portfolio_value;

      // Daily loss circuit breaker
      const todayDate = new Date().toISOString().split("T")[0];
      if (circuitBreaker.date !== todayDate) {
        // Notify circuit breaker reset if it was tripped yesterday
        if (circuitBreaker.tripped) {
          notify.send("✅ CIRCUIT BREAKER RESET — New trading day, buys enabled.", { immediate: true });
        }
        circuitBreaker.date = todayDate;
        circuitBreaker.morningValue = cyclePortfolioValue;
        circuitBreaker.tripped = false;

        // Reset daily stats for the new day
        const dayOfWeek = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" })).getDay();
        dailyStats = {
          date: todayDate, buys: 0, sells: 0, wins: 0, losses: 0,
          summarySent: false,
          weekStartValue: (dayOfWeek === 1 || dailyStats.weekStartValue === null)
            ? cyclePortfolioValue : dailyStats.weekStartValue,
        };
      }
      if (!circuitBreaker.morningValue) {
        circuitBreaker.morningValue = cyclePortfolioValue;
      }
      const dayDrop = (cyclePortfolioValue - circuitBreaker.morningValue) / circuitBreaker.morningValue;
      if (dayDrop <= -CIRCUIT_BREAKER_PCT) {
        circuitBreaker.tripped = true;
      }
      if (circuitBreaker.tripped) {
        skipNewBuys = true;
        const dropPct = (((cyclePortfolioValue - circuitBreaker.morningValue) / circuitBreaker.morningValue) * 100).toFixed(2);
        addLog(`Circuit breaker activated -- portfolio down ${dropPct}% today ($${circuitBreaker.morningValue.toFixed(0)} -> $${cyclePortfolioValue.toFixed(0)}), no new buys until tomorrow.`, "error");
        notify.send(`🚨 CIRCUIT BREAKER — Portfolio down ${dropPct}% today ($${circuitBreaker.morningValue.toFixed(0)} -> $${cyclePortfolioValue.toFixed(0)}). No new buys until tomorrow.`, { deduplicate: true, immediate: true });
      }

      // Fetch upcoming earnings (cached)
      const allSymbols = [...new Set([...UNIVERSE.map(u => u.sym), ...currentPositions.map(p => p.symbol)])];
      const earningsMap = await fetchEarnings(allSymbols);
      const earningsSymbols = Object.keys(earningsMap);
      if (earningsSymbols.length > 0) {
        addLog(`Earnings next 7 days: ${earningsSymbols.map(s => `${s} (${earningsMap[s]})`).join(", ")}`, "system");
      }

      // ── STEP 1: Trailing/Fixed Stop-loss & Take-profit on existing positions ──
      const closedSymbols = new Set();
      for (const pos of currentPositions) {
        const { symbol, qty, current_price: curr, unrealized_pl, unrealized_plpc } = pos;
        if (symbol === "SPY" && idleSpyShares > 0) continue;
        if (trendPositions[symbol]) continue;

        // Update trailing peak
        if (RISK.USE_TRAILING_STOP) {
          if (!trailingPeaks[symbol] || curr > trailingPeaks[symbol]) {
            trailingPeaks[symbol] = curr;
          }
        }

        let stopTriggered = false;
        let stopMsg = "";

        if (RISK.USE_TRAILING_STOP) {
          const peak = trailingPeaks[symbol] || curr;
          const dropFromPeak = (curr - peak) / peak;
          if (dropFromPeak <= -RISK.TRAILING_STOP_PCT) {
            stopTriggered = true;
            stopMsg = `TRAIL-STOP ${symbol}: ${qty} shares @ $${curr.toFixed(2)} | Peak $${peak.toFixed(2)}, drop ${(dropFromPeak * 100).toFixed(1)}%`;
          }
        } else {
          if (unrealized_plpc <= RISK.STOP_LOSS_PCT) {
            stopTriggered = true;
            stopMsg = `STOP-LOSS ${symbol}: ${qty} shares @ $${curr.toFixed(2)} | P&L: $${unrealized_pl.toFixed(2)}`;
          }
        }

        if (stopTriggered) {
          try {
            await closePosition(symbol);
            delete trailingPeaks[symbol];
            closedSymbols.add(symbol);
            if (unrealized_plpc < 0) cooldowns[symbol] = cycleNumber;
            addLog(stopMsg, "sell");
            tradeCount.sells++;
            if (unrealized_pl >= 0) tradeCount.wins++; else tradeCount.losses++;
            tradeCount.totalPnL += unrealized_pl;
            recordTrade({ symbol, action: "sell", shares: qty, price: curr, strategy: "stop-loss", portfolio_value: cyclePortfolioValue, pnl: unrealized_pl });
            dailyStats.sells++;
            if (unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            notify.send(`🛑 STOP-LOSS ${symbol} | ${qty} shares @ $${curr.toFixed(2)} | Loss: $${unrealized_pl.toFixed(2)} (${(unrealized_plpc * 100).toFixed(1)}%)`);
          } catch (err) {
            addLog(`Failed to close ${symbol}: ${err.message}`, "error");
          }
        } else if (unrealized_plpc >= RISK.TAKE_PROFIT_PCT) {
          try {
            await closePosition(symbol);
            delete trailingPeaks[symbol];
            closedSymbols.add(symbol);
            addLog(`TAKE-PROFIT ${symbol}: ${qty} shares @ $${curr.toFixed(2)} | P&L: +$${unrealized_pl.toFixed(2)}`, "profit");
            tradeCount.sells++;
            tradeCount.wins++;
            tradeCount.totalPnL += unrealized_pl;
            recordTrade({ symbol, action: "sell", shares: qty, price: curr, strategy: "take-profit", portfolio_value: cyclePortfolioValue, pnl: unrealized_pl });
            dailyStats.sells++;
            dailyStats.wins++;
            notify.send(`🎯 TAKE-PROFIT ${symbol} | ${qty} shares @ $${curr.toFixed(2)} | Gain: +$${unrealized_pl.toFixed(2)} (+${(unrealized_plpc * 100).toFixed(1)}%)`);
          } catch (err) {
            addLog(`Failed to close ${symbol}: ${err.message}`, "error");
          }
        }
      }

      // ── STEP 1b: Earnings-eve exits ──
      for (const pos of currentPositions) {
        if (closedSymbols.has(pos.symbol)) continue;
        if (pos.symbol === "SPY" && idleSpyShares > 0) continue;
        const earningsDate = earningsMap[pos.symbol];
        if (!earningsDate) continue;
        const days = daysUntilEarnings(earningsDate);
        if (days === 1) {
          try {
            await closePosition(pos.symbol);
            delete trailingPeaks[pos.symbol];
            closedSymbols.add(pos.symbol);
            addLog(`EARNINGS SELL ${pos.symbol}: earnings tomorrow (${earningsDate}) -- exiting to avoid overnight announcement risk`, "sell");
            tradeCount.sells++;
            if (pos.unrealized_pl >= 0) tradeCount.wins++; else tradeCount.losses++;
            tradeCount.totalPnL += pos.unrealized_pl;
            recordTrade({ symbol: pos.symbol, action: "sell", shares: pos.qty, price: pos.current_price, strategy: "earnings-sell", portfolio_value: cyclePortfolioValue, pnl: pos.unrealized_pl });
            dailyStats.sells++;
            if (pos.unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            notify.send(`📉 SELL ${pos.symbol} | ${pos.qty} shares @ $${pos.current_price.toFixed(2)} | Earnings tomorrow — P&L: $${pos.unrealized_pl.toFixed(2)} (${(pos.unrealized_plpc * 100).toFixed(1)}%)`);
          } catch (err) {
            addLog(`Earnings sell failed ${pos.symbol}: ${err.message}`, "error");
          }
        }
      }

      // ── STEP 2: Scan for signals ──
      if (regime !== "BULLISH") {
        const spyPrice = priceHist.SPY?.[priceHist.SPY.length - 1];
        addLog(`Regime: ${regime} | SPY $${spyPrice?.toFixed(2) || "N/A"} -- ${regime === "BEARISH" ? "buys suspended" : "STRONG BUY / ML >=65% only, 75% position size"}`, "system");
      }

      // Build ML signal map
      const mlMap = {};
      if (mlSignals && mlSignals.length > 0) {
        for (const s of mlSignals) mlMap[s.symbol] = s;
      }
      const mlActive = mlSignals !== null && Object.keys(mlMap).length > 0;

      if (mlActive) {
        const buyCount = mlSignals.filter(s => s.signal === "BUY").length;
        addLog(`ML active -- ${buyCount} BUY signal${buyCount !== 1 ? "s" : ""} above 55% threshold`, "system");
      } else {
        addLog("ML server offline -- using consensus engine (fallback mode)", "system");
      }

      const heldSymbols = new Set(currentPositions.map(p => p.symbol));
      const activePositionCount = currentPositions.filter(p => p.symbol !== "SPY").length;
      const opportunities = [];

      for (const { sym } of UNIVERSE) {
        const prices = priceHist[sym];
        const isMLBuy = mlActive && mlMap[sym]?.signal === "BUY";

        if (!prices || prices.length < 35) {
          if (isMLBuy) {
            addLog(`EVAL ${sym}: conf ${(mlMap[sym].probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | BLOCKED: insufficient price data (${prices ? prices.length : 0}/35 bars loaded)`, "system");
          }
          continue;
        }

        // Blacklist check
        if (NEVER_BUY.has(sym)) {
          const analysis = getSignals(prices);
          if (heldSymbols.has(sym) && !trendPositions[sym] && (analysis.consensus === "STRONG SELL" || analysis.consensus === "SELL")) {
            try {
              await closePosition(sym);
              const posData = currentPositions.find(p => p.symbol === sym);
              if (posData && posData.unrealized_plpc < 0) cooldowns[sym] = cycleNumber;
              addLog(`SELL ${sym}: ${analysis.consensus} -- closing blacklisted position`, "sell");
              if (posData) {
                tradeCount.sells++;
                if (posData.unrealized_pl >= 0) tradeCount.wins++; else tradeCount.losses++;
                tradeCount.totalPnL += posData.unrealized_pl;
                recordTrade({ symbol: sym, action: "sell", shares: posData.qty, price: posData.current_price, strategy: "ml", portfolio_value: cyclePortfolioValue, pnl: posData.unrealized_pl });
                dailyStats.sells++;
                if (posData.unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
                notify.send(`📉 SELL ${sym} | ${posData.qty} shares @ $${posData.current_price.toFixed(2)} | Blacklisted — P&L: $${posData.unrealized_pl.toFixed(2)} (${(posData.unrealized_plpc * 100).toFixed(1)}%)`);
              }
            } catch (err) {
              addLog(`Sell failed ${sym}: ${err.message}`, "error");
            }
          }
          if (isMLBuy) {
            addLog(`Blacklisted symbol ${sym} -- never buy (leveraged/inverse/volatility)`, "system");
          }
          continue;
        }

        const analysis = getSignals(prices);

        // Sell on SELL consensus
        if (heldSymbols.has(sym) && !trendPositions[sym] && (analysis.consensus === "STRONG SELL" || analysis.consensus === "SELL")) {
          try {
            await closePosition(sym);
            const posData = currentPositions.find(p => p.symbol === sym);
            if (posData && posData.unrealized_plpc < 0) cooldowns[sym] = cycleNumber;
            addLog(`SELL ${sym}: ${analysis.consensus} -- closing position`, "sell");
            if (posData) {
              tradeCount.sells++;
              if (posData.unrealized_pl >= 0) tradeCount.wins++; else tradeCount.losses++;
              tradeCount.totalPnL += posData.unrealized_pl;
              recordTrade({ symbol: sym, action: "sell", shares: posData.qty, price: posData.current_price, strategy: "ml", portfolio_value: cyclePortfolioValue, pnl: posData.unrealized_pl });
              dailyStats.sells++;
              if (posData.unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
              notify.send(`📉 SELL ${sym} | ${posData.qty} shares @ $${posData.current_price.toFixed(2)} | ${analysis.consensus} — P&L: $${posData.unrealized_pl.toFixed(2)} (${(posData.unrealized_plpc * 100).toFixed(1)}%)`);
            }
          } catch (err) {
            addLog(`Sell failed ${sym}: ${err.message}`, "error");
          }
        }

        // Pre-filter: already held or regime blocks ALL buys
        if (heldSymbols.has(sym)) {
          if (isMLBuy) {
            addLog(`EVAL ${sym}: conf ${(mlMap[sym].probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | BLOCKED: already holding position`, "system");
          }
          continue;
        }
        if (regime === "BEARISH") {
          if (isMLBuy) {
            addLog(`EVAL ${sym}: conf ${(mlMap[sym].probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | BLOCKED: BEARISH regime, all buys suspended`, "system");
          }
          continue;
        }

        if (isOnCooldown(sym)) {
          const remaining = RISK.LOSS_COOLDOWN_CYCLES - (cycleNumber - cooldowns[sym]);
          if (isMLBuy) {
            addLog(`EVAL ${sym}: conf ${(mlMap[sym].probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | BLOCKED: loss cooldown, ${remaining} cycle${remaining !== 1 ? "s" : ""} remaining`, "system");
          } else {
            addLog(`Skipping ${sym} -- cooldown active, ${remaining} cycle${remaining !== 1 ? "s" : ""} remaining`, "system");
          }
          continue;
        }

        // Earnings proximity check (pre-computed for ML eval log)
        const earningsDate = earningsMap[sym];
        const earningsDays = earningsDate ? daysUntilEarnings(earningsDate) : null;
        const earningsBlocked = earningsDays !== null && earningsDays >= 0 && earningsDays <= 3;

        if (mlActive) {
          // ML primary decision
          const mlSig = mlMap[sym];
          if (mlSig && mlSig.signal === "BUY") {
            if (regime === "CAUTIOUS" && mlSig.probability < 0.65) {
              addLog(`EVAL ${sym}: conf ${(mlSig.probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | BLOCKED: CAUTIOUS regime requires >=65% confidence`, "system");
            } else {
              addLog(`EVAL ${sym}: conf ${(mlSig.probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | earnings blocked: ${earningsBlocked}${earningsBlocked ? ` (${earningsDays}d -> ${earningsDate})` : ""} | cooldown: false | PASSED -> added to candidates`, "system");
              opportunities.push({
                sym,
                score: mlSig.probability,
                price: prices[prices.length - 1],
                consensus: `ML BUY (${(mlSig.probability * 100).toFixed(0)}%)`,
                rsiVal: analysis.indicators.rsi,
                mlConf: mlSig.probability,
              });
            }
          } else {
            if (mlSig && mlSig.probability >= 0.45) {
              addLog(`ML SKIP ${sym} -- confidence ${mlSig.probability.toFixed(2)} below threshold`, "system");
            }
          }
        } else {
          // Consensus fallback
          const signalQualifies = regime === "CAUTIOUS"
            ? analysis.consensus === "STRONG BUY"
            : analysis.consensus === "STRONG BUY" || analysis.consensus === "BUY";
          if (signalQualifies) {
            opportunities.push({
              sym,
              score: analysis.score,
              price: prices[prices.length - 1],
              consensus: analysis.consensus,
              rsiVal: analysis.indicators.rsi,
              mlConf: null,
            });
          }
        }
      }

      // ── STEP 3: Rank & execute buys ──
      opportunities.sort((a, b) => b.score - a.score);
      const slotsAvail = RISK.MAX_OPEN_POSITIONS - activePositionCount;

      // Pre-execution diagnostic summary
      if (mlActive && opportunities.length > 0) {
        addLog(`BUY FILTER CHECK -- ${opportunities.length} candidate${opportunities.length !== 1 ? "s" : ""} passed pre-filters | positions: ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | slots open: ${slotsAvail} | cash: $${cycleCash.toLocaleString("en-US", { maximumFractionDigits: 0 })} | portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`, "system");
      }

      if (skipNewBuys) {
        if (opportunities.length > 0) {
          addLog(`Close buffer active -- skipping ${opportunities.length} pending ML buy${opportunities.length !== 1 ? "s" : ""}: ${opportunities.map(o => o.sym).join(", ")}`, "system");
        }
        // Update state with fresh data
        positionsRaw = await getPositions();
        const updatedAcct = await getAccount();
        cash = updatedAcct.cash;
        portfolioValue = updatedAcct.portfolio_value;
        return;
      }

      if (slotsAvail <= 0) {
        if (opportunities.length > 0) {
          addLog(`Position limit full (${activePositionCount}/${RISK.MAX_OPEN_POSITIONS}) -- skipping all ${opportunities.length} ML buy${opportunities.length !== 1 ? "s" : ""}: ${opportunities.map(o => o.sym).join(", ")}`, "system");
        }
      } else if (opportunities.length > slotsAvail) {
        const cut = opportunities.slice(slotsAvail).map(o => o.sym);
        addLog(`${slotsAvail} slot${slotsAvail !== 1 ? "s" : ""} open -- processing top ${slotsAvail} of ${opportunities.length} candidates; cutting: ${cut.join(", ")}`, "system");
      }

      // ── STEP 3a: Sell idle SPY to free cash before ML buys ──
      const spyPos = currentPositions.find(p => p.symbol === "SPY");
      if (idleSpyShares > 0 && spyPos && opportunities.length > 0) {
        const spyShareCount = spyPos.qty;
        const spyPrice = spyPos.current_price;
        const idleValue = spyPos.market_value;
        try {
          await closePosition("SPY");
          idleSpyShares = 0;
          addLog(`[idle-spy] Selling ${spyShareCount} SPY shares ($${idleValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}) to fund ${opportunities.length} ML pick${opportunities.length !== 1 ? "s" : ""}`, "system");
          recordTrade({ symbol: "SPY", action: "sell", shares: spyShareCount, price: spyPrice, strategy: "idle-spy", portfolio_value: cyclePortfolioValue });
          notify.send(`🅿️ SPY IDLE SELL | ${spyShareCount} shares @ $${spyPrice.toFixed(2)} | Freeing cash for ${opportunities.length} ML pick${opportunities.length !== 1 ? "s" : ""}`);
          // Refresh cash after selling idle SPY
          const freshAcct = await getAccount();
          cycleCash = freshAcct.cash;
        } catch (err) {
          addLog(`[idle-spy] Failed to sell SPY: ${err.message}`, "error");
        }
      }

      // Track sectors bought this cycle
      const boughtThisCycle = {};

      for (const opp of opportunities.slice(0, Math.max(0, slotsAvail))) {
        // Sector position limit
        const sector = getSector(opp.sym);
        const sectorLimit = RISK.SECTOR_MAX_POSITIONS[sector];
        if (sectorLimit !== undefined) {
          const existingSectorCount = currentPositions.filter(p => getSector(p.symbol) === sector).length;
          const cycleCount = boughtThisCycle[sector] || 0;
          if (existingSectorCount + cycleCount >= sectorLimit) {
            addLog(`Skipping ${opp.sym} -- ${sector} limit reached (max ${sectorLimit})`, "system");
            continue;
          }
        }

        // Volume confirmation check -- skip for ML-driven trades
        if (!mlActive) {
          const vols = volHist[opp.sym];
          const avg = avgVolume(vols, 20);
          if (avg !== null) {
            const currVol = vols[vols.length - 1];
            const ratio = currVol / avg;
            if (ratio < RISK.VOLUME_CONFIRM_RATIO) {
              addLog(`Skipping ${opp.sym} -- volume ${fmtVol(currVol)} below 1.5x average of ${fmtVol(avg)} (${ratio.toFixed(2)}x)`, "system");
              continue;
            }
          }
        }

        // Skip if earnings within 3 calendar days
        const earningsDate = earningsMap[opp.sym];
        if (earningsDate) {
          const days = daysUntilEarnings(earningsDate);
          if (days >= 0 && days <= 3) {
            addLog(`SKIP ${opp.sym}: earnings in ${days} day(s) on ${earningsDate}`, "system");
            continue;
          }
        }

        // ATR-based position sizing
        const stockAtr = atr(priceHist[opp.sym], 14);
        const atrPct = stockAtr ? stockAtr / opp.price : RISK.ATR_TARGET_PCT;
        const volatilityScale = RISK.ATR_TARGET_PCT / atrPct;
        const regimeMult = regime === "CAUTIOUS" ? 0.75 : 1.0;
        const mlMult = opp.mlConf != null
          ? Math.min(1.0, Math.max(0.60, opp.mlConf * 1.6 - 0.28))
          : 1.0;
        const dynPositionPct = Math.max(
          RISK.MIN_POSITION_PCT,
          Math.min(RISK.MAX_POSITION_PCT, RISK.MAX_POSITION_PCT * volatilityScale * regimeMult * mlMult)
        );

        const maxAlloc = cyclePortfolioValue * dynPositionPct;
        const allocCash = Math.min(maxAlloc, cycleCash * RISK.MAX_CASH_DEPLOY_PCT);
        if (allocCash < opp.price) {
          addLog(`SKIP ${opp.sym} -- insufficient cash: need $${opp.price.toFixed(2)}/share, alloc would be $${allocCash.toFixed(2)} (${(dynPositionPct * 100).toFixed(1)}% of $${cyclePortfolioValue.toFixed(0)} portfolio, max 90% of $${cycleCash.toFixed(0)} cash)`, "system");
          continue;
        }

        const shares = Math.floor(allocCash / opp.price);
        if (shares <= 0) {
          addLog(`SKIP ${opp.sym} -- 0 shares computable at $${opp.price.toFixed(2)}/share with $${allocCash.toFixed(2)} allocated`, "system");
          continue;
        }

        try {
          const order = await placeOrder({ symbol: opp.sym, qty: shares, side: "buy", type: "market" });
          boughtThisCycle[sector] = (boughtThisCycle[sector] || 0) + 1;
          const mlNote = opp.mlConf != null ? `, ML ${(opp.mlConf * 100).toFixed(0)}% conf` : "";
          addLog(`BUY ${opp.sym}: ${shares} shares | ${opp.consensus} (score: ${opp.score.toFixed(2)}) | alloc ${(dynPositionPct * 100).toFixed(1)}% (ATR ${(atrPct * 100).toFixed(1)}%${regime === "CAUTIOUS" ? ", cautious 75%" : ""}${mlNote}) | Order: ${order.status}`, "buy");
          tradeCount.buys++;
          dailyStats.buys++;
          recordTrade({ symbol: opp.sym, action: "buy", shares, price: opp.price, strategy: opp.mlConf != null ? "ml" : "consensus", ml_confidence: opp.mlConf, portfolio_value: cyclePortfolioValue });
          notify.send(opp.mlConf != null
            ? `🤖 ML BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | Confidence: ${(opp.mlConf * 100).toFixed(0)}% | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
            : `📊 BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | ${opp.consensus} | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`);
        } catch (err) {
          addLog(`Buy failed ${opp.sym}: ${err.message}`, "error");
        }
      }

      // ── STEP 4: Trend trailing stop (10%) ──
      for (const pos of currentPositions) {
        const sym = pos.symbol;
        const tPos = trendPositions[sym];
        if (!tPos || closedSymbols.has(sym)) continue;

        const curr = pos.current_price;
        if (curr > (tPos.peakPrice || tPos.entryPrice)) {
          tPos.peakPrice = curr;
        }
        const peak = tPos.peakPrice || tPos.entryPrice;
        const dropFromPeak = (curr - peak) / peak;

        if (dropFromPeak <= -0.10) {
          try {
            await closePosition(sym);
            closedSymbols.add(sym);
            delete trendPositions[sym];
            delete trendBreakCounts[sym];
            const trendPnl = pos.unrealized_pl;
            addLog(`TREND-STOP ${sym}: @ $${curr.toFixed(2)} | Peak $${peak.toFixed(2)}, drop ${(dropFromPeak * 100).toFixed(1)}%`, "sell");
            tradeCount.sells++;
            if (trendPnl >= 0) tradeCount.wins++; else tradeCount.losses++;
            tradeCount.totalPnL += trendPnl;
            recordTrade({ symbol: sym, action: "sell", shares: pos.qty, price: curr, strategy: "trend-stop", portfolio_value: cyclePortfolioValue, pnl: trendPnl });
            dailyStats.sells++;
            if (trendPnl >= 0) dailyStats.wins++; else dailyStats.losses++;
            notify.send(`📈 TREND SELL ${sym} | ${pos.qty} shares @ $${curr.toFixed(2)} | Trail-stop hit, peak $${peak.toFixed(2)} | P&L: $${trendPnl.toFixed(2)}`);
          } catch (err) {
            addLog(`Trend-stop close failed ${sym}: ${err.message}`, "error");
          }
        }
      }

      // ── STEP 5: Trend SMA-break exits (3 consecutive closes below 200-SMA) ──
      for (const sym of Object.keys(trendPositions)) {
        if (closedSymbols.has(sym)) continue;
        const prices = priceHist[sym];
        if (!prices || prices.length < 200) continue;
        const ts = computeTrendStatus(prices);
        if (!ts) continue;

        if (!ts.priceAbove200) {
          trendBreakCounts[sym] = (trendBreakCounts[sym] || 0) + 1;
          if (trendBreakCounts[sym] >= 3) {
            try {
              await closePosition(sym);
              closedSymbols.add(sym);
              delete trendPositions[sym];
              delete trendBreakCounts[sym];
              const breakPos = currentPositions.find(p => p.symbol === sym);
              addLog(`TREND-BREAK ${sym}: 3 consecutive closes below 200-SMA -- exiting`, "sell");
              if (breakPos) {
                tradeCount.sells++;
                if (breakPos.unrealized_pl >= 0) tradeCount.wins++; else tradeCount.losses++;
                tradeCount.totalPnL += breakPos.unrealized_pl;
                recordTrade({ symbol: sym, action: "sell", shares: breakPos.qty, price: breakPos.current_price, strategy: "trend-break", portfolio_value: cyclePortfolioValue, pnl: breakPos.unrealized_pl });
                dailyStats.sells++;
                if (breakPos.unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
                notify.send(`📈 TREND SELL ${sym} | ${breakPos.qty} shares @ $${breakPos.current_price.toFixed(2)} | 200-SMA break | P&L: $${breakPos.unrealized_pl.toFixed(2)}`);
              }
            } catch (err) {
              addLog(`Trend-break close failed ${sym}: ${err.message}`, "error");
            }
          } else {
            addLog(`TREND ${sym}: day ${trendBreakCounts[sym]}/3 below 200-SMA`, "system");
          }
        } else {
          trendBreakCounts[sym] = 0;
        }
      }

      // ── STEP 6: Trend new entries ──
      if (!skipNewBuys && regime !== "BEARISH") {
        const trendCount = Object.keys(trendPositions).length;
        if (trendCount < 8) {
          const trendPosValue = Object.values(trendPositions).reduce((sum, tp) => {
            const alpacaPos = currentPositions.find(p => p.symbol === tp.sym);
            return sum + (alpacaPos ? alpacaPos.market_value : 0);
          }, 0);
          const trendPortPct = cyclePortfolioValue > 0 ? trendPosValue / cyclePortfolioValue : 0;

          if (trendPortPct < 0.30) {
            for (const { sym } of UNIVERSE) {
              if (NEVER_BUY.has(sym)) continue;
              if (trendPositions[sym] || heldSymbols.has(sym)) continue;
              const prices = priceHist[sym];
              if (!prices || prices.length < 200) continue;
              const ts = computeTrendStatus(prices);
              if (!ts?.isStrongUptrend) continue;

              const currPrice = prices[prices.length - 1];
              const allocCashTrend = cyclePortfolioValue * 0.05;
              if (allocCashTrend < currPrice || allocCashTrend > cycleCash) continue;
              const shares = Math.floor(allocCashTrend / currPrice);
              if (shares <= 0) continue;

              try {
                const order = await placeOrder({ symbol: sym, qty: shares, side: "buy", type: "market" });
                trendPositions[sym] = { entryPrice: currPrice, peakPrice: currPrice };
                trendBreakCounts[sym] = 0;
                addLog(`TREND-BUY ${sym}: ${shares} sh | ${ts.daysAbove200}/40 days above 200-SMA | Order: ${order.status}`, "buy");
                tradeCount.buys++;
                dailyStats.buys++;
                recordTrade({ symbol: sym, action: "buy", shares, price: currPrice, strategy: "trend", portfolio_value: cyclePortfolioValue });
                notify.send(`📈 TREND BUY ${sym} | ${shares} shares @ $${currPrice.toFixed(2)} | ${ts.daysAbove200}/40 days above 200-SMA`);
                if (Object.keys(trendPositions).length >= 8) break;
              } catch (err) {
                addLog(`Trend buy failed ${sym}: ${err.message}`, "error");
              }
            }
          }
        }
      }

      // ── STEP 7: Park idle cash in SPY ──
      if (!skipNewBuys) {
        try {
          const freshAcct = await getAccount();
          const freshCash = freshAcct.cash;
          const freshPortVal = freshAcct.portfolio_value;
          const reservedCash = freshPortVal * SPY_IDLE_RESERVE_PCT;
          const idleCash = freshCash - reservedCash;

          if (idleCash > freshPortVal * SPY_IDLE_THRESHOLD_PCT) {
            const parkAmount = idleCash * SPY_IDLE_INVEST_PCT;
            const spyPrice = priceHist.SPY?.[priceHist.SPY.length - 1];
            if (spyPrice && spyPrice > 0 && parkAmount >= spyPrice) {
              const spySharesToBuy = Math.floor(parkAmount / spyPrice);
              if (spySharesToBuy > 0) {
                await placeOrder({ symbol: "SPY", qty: spySharesToBuy, side: "buy", type: "market" });
                idleSpyShares += spySharesToBuy;
                addLog(`[idle-spy] Parking $${parkAmount.toLocaleString("en-US", { maximumFractionDigits: 0 })} -> ${spySharesToBuy} SPY @ $${spyPrice.toFixed(2)} | Total idle SPY: ${idleSpyShares} shares`, "system");
                recordTrade({ symbol: "SPY", action: "buy", shares: spySharesToBuy, price: spyPrice, strategy: "idle-spy", portfolio_value: cyclePortfolioValue });
                notify.send(`🅿️ SPY IDLE BUY | ${spySharesToBuy} shares @ $${spyPrice.toFixed(2)} | Idle cash parked`);
              }
            }
          }
        } catch (err) {
          addLog(`[idle-spy] SPY park failed: ${err.message}`, "error");
        }
      }

      // Refresh positions after all trades
      positionsRaw = await getPositions();
      const updatedAcct = await getAccount();
      cash = updatedAcct.cash;
      portfolioValue = updatedAcct.portfolio_value;

      positions = {};
      for (const p of positionsRaw) {
        positions[p.symbol] = {
          shares: p.qty,
          avgPrice: p.avg_entry_price,
          currentPrice: p.current_price,
          unrealizedPl: p.unrealized_pl,
          unrealizedPlPct: p.unrealized_plpc,
          marketValue: p.market_value,
        };
      }

      // Daily snapshot + Telegram summary near market close (last 5 minutes)
      if (minutesUntilClose <= 5) {
        const today = new Date().toISOString().split("T")[0];
        const activePos = positionsRaw.filter(p => p.symbol !== "SPY").length;
        const dailyPnl = portfolioValue - (circuitBreaker.morningValue || portfolioValue);
        recordDailySnapshot({
          date: today,
          portfolio_value: portfolioValue,
          cash: cash,
          positions_count: activePos,
          daily_pnl: dailyPnl,
        });

        // Send daily/weekly summary via Telegram (once per day)
        if (!dailyStats.summarySent) {
          dailyStats.summarySent = true;
          const dailyPnlPct = circuitBreaker.morningValue ? (dailyPnl / circuitBreaker.morningValue * 100) : 0;
          const spyIdlePos = positionsRaw.find(p => p.symbol === "SPY");
          const spyIdleValue = spyIdlePos ? spyIdlePos.market_value : 0;
          const etDay = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" })).getDay();
          const isFriday = etDay === 5;

          let summary = isFriday ? "📊 WEEKLY SUMMARY (Friday Close)" : "📊 DAILY SUMMARY";
          summary += "\n━━━━━━━━━━━━━━━";
          summary += `\nPortfolio: $${portfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
          if (isFriday && dailyStats.weekStartValue) {
            const weeklyPnl = portfolioValue - dailyStats.weekStartValue;
            const weeklyPnlPct = (weeklyPnl / dailyStats.weekStartValue * 100);
            summary += `\nWeekly P&L: ${weeklyPnl >= 0 ? "+" : ""}$${weeklyPnl.toFixed(0)} (${weeklyPnlPct >= 0 ? "+" : ""}${weeklyPnlPct.toFixed(2)}%)`;
          }
          summary += `\nDaily P&L: ${dailyPnl >= 0 ? "+" : ""}$${dailyPnl.toFixed(0)} (${dailyPnlPct >= 0 ? "+" : ""}${dailyPnlPct.toFixed(2)}%)`;
          summary += `\nPositions: ${activePos}/${RISK.MAX_OPEN_POSITIONS}`;
          summary += `\nTrades today: ${dailyStats.buys} buys, ${dailyStats.sells} sells`;
          summary += `\nWin/Loss: ${dailyStats.wins}/${dailyStats.losses}`;
          summary += `\nML Status: ${mlStatus}`;
          summary += `\nRegime: ${regime}`;
          summary += `\nSPY Idle: ${idleSpyShares} shares ($${spyIdleValue.toLocaleString("en-US", { maximumFractionDigits: 0 })})`;

          notify.send(summary, { immediate: true });
        }
      }

      addLog(`Cycle #${cycleNumber} complete | Cash: $${cash.toFixed(0)} | Portfolio: $${portfolioValue.toFixed(0)} | Positions: ${positionsRaw.length}`, "system");

    } catch (err) {
      addLog(`Trading cycle error: ${err.message}`, "error");
    }
  }

  // ══════════════════════════════════════════
  //  START / STOP
  // ══════════════════════════════════════════

  async function start() {
    if (running) {
      addLog("Engine already running", "system");
      return;
    }

    try {
      await init();
      running = true;

      // Start price polling (every 15s)
      pricePollInterval = setInterval(async () => {
        try {
          await pollPrices();
        } catch (err) {
          addLog(`Price poll interval error: ${err.message}`, "error");
        }
      }, PRICE_POLL_MS);

      // Start trade cycle (every 60s)
      tradeCycleInterval = setInterval(async () => {
        try {
          if (marketOpen) {
            await runTradeCycle();
          }
        } catch (err) {
          addLog(`Trade cycle interval error: ${err.message}`, "error");
        }
      }, TRADE_CYCLE_MS);

      // Run first poll immediately
      await pollPrices();

      if (marketOpen) {
        addLog("Trading engine started -- market is OPEN, executing first trade cycle...", "system");
        // Run first trade cycle after a short delay to let data settle
        setTimeout(async () => {
          try {
            await runTradeCycle();
          } catch (err) {
            addLog(`Initial trade cycle error: ${err.message}`, "error");
          }
        }, 5000);
      } else {
        addLog("Trading engine started -- market is CLOSED, will trade when market opens", "system");
      }
    } catch (err) {
      running = false;
      addLog(`Engine start failed: ${err.message}`, "error");
      throw err;
    }
  }

  function stop() {
    if (!running) {
      addLog("Engine not running", "system");
      return;
    }

    if (pricePollInterval) {
      clearInterval(pricePollInterval);
      pricePollInterval = null;
    }
    if (tradeCycleInterval) {
      clearInterval(tradeCycleInterval);
      tradeCycleInterval = null;
    }

    running = false;
    addLog("Trading engine stopped", "system");
  }

  // ══════════════════════════════════════════
  //  STATE ACCESSORS
  // ══════════════════════════════════════════

  function getState() {
    return {
      running,
      tick,
      cash,
      portfolioValue,
      initialPortfolioValue,
      positions,
      priceHist,
      portfolioHist,
      tradeCount,
      volHist,
      connected,
      marketOpen,
      regime,
      error,
      mlSignals,
      mlStatus,
      idleSpyShares,
      circuitBreaker: { ...circuitBreaker },
    };
  }

  function getActivityFeed(limit = 200) {
    return activityLog.slice(-limit);
  }

  // ── Public API ──
  return {
    start,
    stop,
    getState,
    getActivityFeed,
    isRunning: () => running,
  };
};
