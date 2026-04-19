// ══════════════════════════════════════════════════════════════════════
//  TRADING ENGINE — Server-side consolidated trading logic
//  Runs all signal analysis, regime detection, trend following,
//  and trade execution directly on the Express server.
//  No React dependency — uses Alpaca SDK + Node http for ML signals.
// ══════════════════════════════════════════════════════════════════════

const http = require("http");
const fs = require("fs");
const path = require("path");
const notify = require("./notifications");

// ══════════════════════════════════════════
//  CONSTANTS (inlined from frontend config)
// ══════════════════════════════════════════

const INITIAL_CASH = 100000;

const RISK = {
  MAX_POSITION_PCT: 0.15,
  STOP_LOSS_PCT: -0.08,
  TAKE_PROFIT_PCT: 0.15,
  MAX_OPEN_POSITIONS: 10,             // 3-strategy: ML(2)+Mom(4)+MR(2)+flex(2)
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

// ── Multi-strategy slot allocation ──
const SLOT_CONFIG = {
  ml_medium: 2,          // ML primary slots
  momentum: 4,           // Momentum primary slots
  mean_reversion: 2,     // Mean Reversion primary slots
  flex: 2,               // Shared flex pool
  max: 10,               // Hard cap (= RISK.MAX_OPEN_POSITIONS)
};

// ── Momentum strategy parameters ──
const MOM = {
  LOOKBACK: 63,          // 63 trading days for ranking
  TOP_N: 5,              // top 5 signals per cycle
  SMA_PERIOD: 200,       // trend filter
  VOL_PERIOD: 20,        // avg volume window
  VOL_MIN: 500000,       // minimum 20-day avg volume
  ATR_PERIOD: 14,
  STOP_LOSS: -0.08,      // -8% from entry
  TAKE_PROFIT: 0.20,     // +20% from entry
  TRAIL_STOP: -0.08,     // -8% from peak
  RANK_BREAK: 20,        // exit if rank > 20
  MAX_HOLD_DAYS: 60,     // trading days (approx 60 cycles at 1/day)
  BASE_PCT: 0.12,        // 12% position size
  HIGH_ATR_PCT: 0.08,    // 8% for volatile stocks
  HIGH_ATR_THRESH: 0.03, // ATR/price > 3% = volatile
  COOLDOWN_CYCLES: 5,    // 5-cycle cooldown after momentum sell
};

// ── Mean Reversion strategy parameters ──
const MR = {
  DROP_PERIOD: 30,          // 30 trading days to measure drop
  DROP_THRESHOLD: -0.15,    // minimum drop to trigger signal (-15%)
  DROP_MAX: -0.30,          // drop level for maximum confidence (-30%)
  SMA_PERIOD: 200,          // trend filter (above 200-SMA)
  SMA_SHORT: 20,            // take-profit target (20-day SMA recovery)
  VOL_PERIOD: 20,           // avg volume window
  VOL_MIN: 500000,          // minimum 20-day avg volume
  ATR_PERIOD: 14,
  STOP_LOSS: -0.10,         // -10% from entry
  MAX_HOLD_DAYS: 10,        // trading days (approx 10 cycles at 1/day)
  BASE_PCT: 0.12,           // 12% position size
  HIGH_ATR_PCT: 0.08,       // 8% for volatile stocks
  HIGH_ATR_THRESH: 0.04,    // ATR/price > 4% = volatile (higher than momentum's 3%)
  COOLDOWN_CYCLES: 5,       // 5-cycle cooldown after MR sell
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

// ── Dynamic universe loading (S&P 500 + ETFs, ~510 symbols) ──
// Loaded from ml_service/sp500_universe.py's cached JSON, with fallback
const ETF_SYMBOLS = [
  "SPY",
  "XLK","XLF","XLV","XLE","XLI","XLP","XLY","XLB","XLU","XLRE","XLC",
  "EWZ","EWJ","FXI","INDA","EFA","EEM","VGK","VWO","IEFA",
  "GLD","SLV","USO","DBC","CPER",
  "TLT","IEF","SHY","HYG","LQD",
  "VIXY","UUP",
];

function loadUniverseSymbols() {
  // Try loading S&P 500 list from the Python-generated cache
  const cacheFile = path.join(__dirname, "..", "ml_service", "data", "sp500_constituents.json");
  let sp500 = [];
  try {
    const raw = fs.readFileSync(cacheFile, "utf8");
    sp500 = JSON.parse(raw);
    if (!Array.isArray(sp500) || sp500.length < 400) sp500 = [];
  } catch (e) { /* fall through to fallback */ }

  if (sp500.length === 0) {
    // Minimal fallback — original 32 stocks
    sp500 = [
      "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
      "CRM","ORCL","ADBE","CSCO","QCOM","COST","WMT","HD","LOW",
      "LLY","JNJ","ABBV","BAC","GS","MS","CVX","XOM","CAT","DE","BA",
    ];
  }

  return [...new Set([...sp500, ...ETF_SYMBOLS])].sort();
}

// UNIVERSE_SYMBOLS: flat array of ticker strings (~510)
let UNIVERSE_SYMBOLS = loadUniverseSymbols();

// Batch size for Alpaca API calls (snapshots support up to 200, bars individually)
const SNAPSHOT_BATCH_SIZE = 100;
const BAR_FETCH_CONCURRENCY = 5;   // parallel bar fetches (reduced from 20 for rate limits)
const BAR_FETCH_RETRIES = 3;       // retry failed bar fetches with exponential backoff
const SNAPSHOT_BATCH_DELAY_MS = 500; // delay between snapshot batches

// ── Alpaca symbol format mapping ──
// S&P 500 lists use hyphens (BF-B) but Alpaca uses dots (BF.B)
const ALPACA_SYMBOL_MAP = {
  "BF-B": "BF.B",
  "BRK-B": "BRK.B",
  "BRK-A": "BRK.A",
};
const REVERSE_SYMBOL_MAP = Object.fromEntries(
  Object.entries(ALPACA_SYMBOL_MAP).map(([k, v]) => [v, k])
);
function toAlpacaSymbol(sym) { return ALPACA_SYMBOL_MAP[sym] || sym; }
function fromAlpacaSymbol(sym) { return REVERSE_SYMBOL_MAP[sym] || sym; }

// ── Global Alpaca rate limiter (max 3 req/sec = 180 req/min, under 200 limit) ──
const RATE_LIMIT_MIN_INTERVAL_MS = 334; // ~3 req/sec
let _lastRequestTime = 0;
async function rateLimitWait() {
  const now = Date.now();
  const elapsed = now - _lastRequestTime;
  if (elapsed < RATE_LIMIT_MIN_INTERVAL_MS) {
    await new Promise(r => setTimeout(r, RATE_LIMIT_MIN_INTERVAL_MS - elapsed));
  }
  _lastRequestTime = Date.now();
}

const CONSENSUS_THRESHOLDS = { STRONG_BUY: 2, BUY: 1, SELL: -1, STRONG_SELL: -2 };
const REGIME_RECOVERY_DAYS = 3;
const SPY_IDLE_RESERVE_PCT = 0.30;
const SPY_IDLE_THRESHOLD_PCT = 0.20;
const SPY_IDLE_INVEST_PCT = 0.85;
const CIRCUIT_BREAKER_PCT = 0.02;
const PRICE_POLL_MS = 15000;
const TRADE_CYCLE_MS = 60000;

// ── Volatility Targeting ──
const VOL_TARGET = 0.15;    // 15% annualized target
const MAX_LEVERAGE = 1.5;
const MIN_LEVERAGE = 0.3;
const VOL_LOOKBACK = 20;    // trading days for realized vol

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
  // ETFs
  XLE: "Energy", XLF: "Finance", XLV: "Health", XLI: "Industrial",
  XLK: "Tech", XLY: "Consumer", XLP: "Staples", XLU: "Utilities",
  XLRE: "REIT", XLB: "Materials", XLC: "Media",
  EWZ: "International", EWJ: "International", FXI: "International",
  INDA: "International", EFA: "International", EEM: "International",
  VGK: "International", VWO: "International", IEFA: "International",
  GLD: "Commodity", SLV: "Commodity", USO: "Commodity", DBC: "Commodity", CPER: "Commodity",
  TLT: "Bond", IEF: "Bond", SHY: "Bond", HYG: "Bond", LQD: "Bond",
  VIXY: "Volatility", UUP: "Commodity",
  // Additional S&P 500 coverage
  CSCO: "Tech", NFLX: "Media", COST: "Staples", WMT: "Staples",
  INTU: "Tech", SNPS: "Tech", CDNS: "Tech", KLAC: "Semis", LRCX: "Semis", AMAT: "Semis",
  ACN: "Tech", IBM: "Tech", TXN: "Semis", ADI: "Semis", MCHP: "Semis", ON: "Semis",
  ADP: "Tech", FISV: "Tech", FIS: "Tech", GPN: "Tech", ADSK: "Tech",
  PG: "Staples", KO: "Staples", PEP: "Staples", PM: "Staples", MO: "Staples",
  CL: "Staples", KHC: "Staples", GIS: "Staples", SJM: "Staples", K: "Staples",
  MDLZ: "Staples", HSY: "Staples", HRL: "Staples", CPB: "Staples", CAG: "Staples",
  LIN: "Materials", APD: "Materials", SHW: "Materials", ECL: "Materials", PPG: "Materials",
  DD: "Materials", NEM: "Materials", FCX: "Materials", NUE: "Materials",
  D: "Utilities", AEP: "Utilities", EXC: "Utilities", SRE: "Utilities", ES: "Utilities",
  WEC: "Utilities", ED: "Utilities", DTE: "Utilities", AEE: "Utilities", CMS: "Utilities",
  BRK: "Finance", "BRK.B": "Finance", MMC: "Finance", AON: "Finance", TRV: "Finance",
  CB: "Finance", PNC: "Finance", TFC: "Finance", MTB: "Finance", FITB: "Finance",
  CFG: "Finance", KEY: "Finance", RF: "Finance", ZION: "Finance",
  SPGI: "Finance", ICE: "Finance", CME: "Finance", MSCI: "Finance", MCO: "Finance",
  UNP: "Industrial", CSX: "Industrial", NSC: "Industrial", WAB: "Industrial",
  ITW: "Industrial", EMR: "Industrial", ROK: "Industrial", ETN: "Industrial",
  PH: "Industrial", IR: "Industrial", DOV: "Industrial", GWW: "Industrial",
  AAL: "Industrial", DAL: "Industrial", UAL: "Industrial", LUV: "Industrial",
  VRTX: "Health", REGN: "Health", IDXX: "Health", IQV: "Health", ZTS: "Health",
  SYK: "Health", BDX: "Health", BSX: "Health", EW: "Health", BAX: "Health",
  HCA: "Health", CI: "Health", CNC: "Health", MOH: "Health",
  AVGO: "Semis",
};

function getSector(sym) {
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
//  MOMENTUM SIGNAL ENGINE
// ══════════════════════════════════════════

/**
 * Compute momentum rankings and generate buy signals from priceHist/volHist.
 * Mirrors MomentumStrategy in unified_backtester.py exactly.
 *
 * @param {Object} priceHist  - { symbol → [close1, close2, ...] }
 * @param {Object} volHist    - { symbol → [vol1, vol2, ...] }
 * @param {Set}    heldSymbols - symbols currently held
 * @param {Object} earningsMap - { symbol → earningsDateStr }
 * @returns {{ signals: Array, rankings: Object }} momentum opportunities and rankings
 */
function computeMomentumSignals(priceHist, volHist, heldSymbols, earningsMap) {
  const symbols = UNIVERSE_SYMBOLS.filter(s => !NEVER_BUY.has(s) && s !== "SPY");
  const rankings = {};
  const returns63 = {};

  // 1. Compute 63-day returns for all symbols with enough data
  for (const sym of symbols) {
    const prices = priceHist[sym];
    if (!prices || prices.length < MOM.LOOKBACK + 1) continue;
    const current = prices[prices.length - 1];
    const past = prices[prices.length - 1 - MOM.LOOKBACK];
    if (past > 0) {
      returns63[sym] = (current - past) / past;
    }
  }

  // 2. Rank by return (1 = best momentum)
  const sorted = Object.entries(returns63).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([sym], idx) => {
    rankings[sym] = idx + 1;
  });

  // 3. Generate signals for top N that pass filters
  const signals = [];
  let signalCount = 0;

  for (const [sym, ret] of sorted) {
    if (signalCount >= MOM.TOP_N) break;
    const rank = rankings[sym];

    // Already held
    if (heldSymbols.has(sym)) continue;

    const prices = priceHist[sym];
    const currentPrice = prices[prices.length - 1];

    // SMA200 filter: price must be above 200-day SMA
    const sma200 = sma(prices, MOM.SMA_PERIOD);
    if (sma200 === null || currentPrice <= sma200) continue;

    // Volume filter: 20-day avg volume > 500K
    const vols = volHist[sym];
    const avg = avgVolume(vols, MOM.VOL_PERIOD);
    if (avg !== null && avg < MOM.VOL_MIN) continue;

    // Earnings proximity check
    const earningsDate = earningsMap ? earningsMap[sym] : null;
    if (earningsDate) {
      const days = daysUntilEarnings(earningsDate);
      if (days >= 0 && days <= 3) continue;
    }

    // Confidence = (100 - rank) / 100
    const confidence = (100 - rank) / 100;

    // ATR-based position sizing
    const stockAtr = atr(prices, MOM.ATR_PERIOD);
    const atrPct = stockAtr ? stockAtr / currentPrice : 0.02;
    const positionPct = atrPct > MOM.HIGH_ATR_THRESH ? MOM.HIGH_ATR_PCT : MOM.BASE_PCT;

    signals.push({
      sym,
      score: confidence,
      price: currentPrice,
      consensus: `MOM RANK #${rank} (${(ret * 100).toFixed(1)}%)`,
      rsiVal: null,
      mlConf: null,
      strategy: "momentum",
      positionPct,
      rank,
    });
    signalCount++;
  }

  return { signals, rankings };
}

// ══════════════════════════════════════════
//  MEAN REVERSION SIGNAL ENGINE
// ══════════════════════════════════════════

/**
 * Compute mean reversion buy signals: stocks that dropped >15% in 30 days
 * but are still above 200-SMA and not at 30-day low (early bounce, not falling knife).
 *
 * @param {Object} priceHist   - { symbol → [close1, close2, ...] }
 * @param {Object} volHist     - { symbol → [vol1, vol2, ...] }
 * @param {Set}    heldSymbols - symbols currently held
 * @param {Object} earningsMap - { symbol → earningsDateStr }
 * @returns {Array} mean reversion buy signals
 */
function computeMeanReversionSignals(priceHist, volHist, heldSymbols, earningsMap) {
  const symbols = UNIVERSE_SYMBOLS.filter(s => !NEVER_BUY.has(s) && s !== "SPY");
  const signals = [];

  for (const sym of symbols) {
    const prices = priceHist[sym];
    if (!prices || prices.length < Math.max(MR.SMA_PERIOD, MR.DROP_PERIOD) + 1) continue;

    // Already held
    if (heldSymbols.has(sym)) continue;

    const currentPrice = prices[prices.length - 1];
    if (currentPrice <= 0) continue;

    // 30-day return
    const pastPrice = prices[prices.length - 1 - MR.DROP_PERIOD];
    if (!pastPrice || pastPrice <= 0) continue;
    const drop30d = (currentPrice - pastPrice) / pastPrice;

    // Must have dropped > 15% (drop30d is negative)
    if (drop30d > MR.DROP_THRESHOLD) continue;

    // Filter: above 200-day SMA (not in structural downtrend)
    const sma200 = sma(prices, MR.SMA_PERIOD);
    if (sma200 === null || currentPrice < sma200) continue;

    // Filter: NOT at 30-day low (want early bounce, not falling knife)
    const recentPrices = prices.slice(-MR.DROP_PERIOD);
    const low30d = Math.min(...recentPrices);
    if (currentPrice <= low30d * 1.001) continue;  // within 0.1% of low

    // Filter: volume
    const vols = volHist[sym];
    const avgVol20 = avgVolume(vols, MR.VOL_PERIOD);
    if (avgVol20 !== null && avgVol20 < MR.VOL_MIN) continue;

    // Earnings proximity check (7 days for MR)
    const earningsDate = earningsMap ? earningsMap[sym] : null;
    if (earningsDate) {
      const days = daysUntilEarnings(earningsDate);
      if (days >= 0 && days <= 7) continue;
    }

    // Confidence: linear scale from 0.50 at -15% to 0.95 at -30%
    const dropRange = MR.DROP_MAX - MR.DROP_THRESHOLD;  // -0.15
    const dropPct = Math.min(1.0, Math.max(0.0, (drop30d - MR.DROP_THRESHOLD) / dropRange));
    const confidence = 0.50 + dropPct * 0.45;

    // ATR-based position sizing
    const stockAtr = atr(prices, MR.ATR_PERIOD);
    const atrPct = stockAtr ? stockAtr / currentPrice : 0.02;
    const positionPct = atrPct > MR.HIGH_ATR_THRESH ? MR.HIGH_ATR_PCT : MR.BASE_PCT;

    signals.push({
      sym,
      score: confidence,
      price: currentPrice,
      consensus: `MR DROP ${(drop30d * 100).toFixed(1)}% (30d)`,
      rsiVal: null,
      mlConf: null,
      strategy: "mean_reversion",
      positionPct,
      drop30d,
    });
  }

  // Sort by confidence descending (biggest drops first)
  signals.sort((a, b) => b.score - a.score);
  return signals;
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
    await rateLimitWait();
    const acct = await alpaca.getAccount();
    return {
      cash: parseFloat(acct.cash),
      portfolio_value: parseFloat(acct.portfolio_value),
    };
  }

  async function getPositions() {
    await rateLimitWait();
    const raw = await alpaca.getPositions();
    return raw.map(p => ({
      symbol: fromAlpacaSymbol(p.symbol),
      qty: parseFloat(p.qty),
      avg_entry_price: parseFloat(p.avg_entry_price),
      current_price: parseFloat(p.current_price),
      unrealized_pl: parseFloat(p.unrealized_pl),
      unrealized_plpc: parseFloat(p.unrealized_plpc),
      market_value: parseFloat(p.market_value),
    }));
  }

  async function getClock() {
    await rateLimitWait();
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
      const alpacaSym = toAlpacaSymbol(symbol);
      await rateLimitWait();
      return await alpaca.createOrder({ symbol: alpacaSym, qty, side, type, time_in_force });
    } catch (err) {
      notify.send(`🚨 ORDER REJECTED — ${symbol} ${side} ${qty} shares | Reason: ${err.message}`, { deduplicate: true, immediate: true });
      throw err;
    }
  }

  async function closePosition(symbol) {
    try {
      const alpacaSym = toAlpacaSymbol(symbol);
      await rateLimitWait();
      return await alpaca.closePosition(alpacaSym);
    } catch (err) {
      notify.send(`🚨 ORDER REJECTED — ${symbol} close | Reason: ${err.message}`, { deduplicate: true, immediate: true });
      throw err;
    }
  }

  async function getOrders(status = "all", limit = 50) {
    await rateLimitWait();
    return await alpaca.getOrders({ status, limit });
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  async function fetchBars(symbol, limit = 250) {
    const alpacaSym = toAlpacaSymbol(symbol);
    const calDays = Math.ceil(limit * 1.6) + 10;
    const start = new Date();
    start.setDate(start.getDate() - calDays);
    const startISO = start.toISOString().split("T")[0];

    const bars = [];
    await rateLimitWait();
    const iter = alpaca.getBarsV2(alpacaSym, { timeframe: "1Day", start: startISO, adjustment: "split" });
    for await (const bar of iter) {
      bars.push({ c: parseFloat(bar.ClosePrice), v: parseInt(bar.Volume) });
    }
    return bars.slice(-limit);
  }

  async function fetchBarsWithRetry(symbol, limit = 250) {
    for (let attempt = 1; attempt <= BAR_FETCH_RETRIES; attempt++) {
      try {
        return await fetchBars(symbol, limit);
      } catch (err) {
        if (attempt < BAR_FETCH_RETRIES) {
          const backoff = Math.pow(2, attempt - 1) * 1000; // 1s, 2s, 4s
          await sleep(backoff);
        } else {
          throw err;
        }
      }
    }
  }

  async function fetchSnapshots(symbols) {
    // Batch into chunks of SNAPSHOT_BATCH_SIZE with delays between batches
    const result = {};
    const alpacaSymbols = symbols.map(toAlpacaSymbol);
    for (let i = 0; i < alpacaSymbols.length; i += SNAPSHOT_BATCH_SIZE) {
      const batch = alpacaSymbols.slice(i, i + SNAPSHOT_BATCH_SIZE);
      const batchNum = Math.floor(i / SNAPSHOT_BATCH_SIZE) + 1;
      try {
        await rateLimitWait();
        const snaps = await alpaca.getSnapshots(batch);
        for (const snap of snaps) {
          if (!snap.symbol) continue;
          const origSym = fromAlpacaSymbol(snap.symbol);
          result[origSym] = {
            price: parseFloat(snap.LatestTrade?.Price || snap.DailyBar?.ClosePrice || 0),
            volume: parseInt(snap.DailyBar?.Volume || 0),
          };
        }
      } catch (err) {
        addLog(`Snapshot batch ${batchNum} failed: ${err.message} — retrying after 2s`, "error");
        await sleep(2000);
        try {
          await rateLimitWait();
          const snaps = await alpaca.getSnapshots(batch);
          for (const snap of snaps) {
            if (!snap.symbol) continue;
            const origSym = fromAlpacaSymbol(snap.symbol);
            result[origSym] = {
              price: parseFloat(snap.LatestTrade?.Price || snap.DailyBar?.ClosePrice || 0),
              volume: parseInt(snap.DailyBar?.Volume || 0),
            };
          }
          addLog(`Snapshot batch ${batchNum} retry succeeded`, "system");
        } catch (retryErr) {
          addLog(`Snapshot batch ${batchNum} retry failed, skipping ${batch.length} symbols`, "error");
        }
      }
      // Delay between batches to stay under rate limit
      if (i + SNAPSHOT_BATCH_SIZE < alpacaSymbols.length) {
        await sleep(SNAPSHOT_BATCH_DELAY_MS);
      }
    }
    return result;
  }

  async function fetchBarsParallel(symbols, limit = 250) {
    // Fetch bars for many symbols with controlled concurrency + retries
    const results = {};
    let loaded = 0;
    let failed = 0;
    const failedSymbols = [];

    for (let i = 0; i < symbols.length; i += BAR_FETCH_CONCURRENCY) {
      const batch = symbols.slice(i, i + BAR_FETCH_CONCURRENCY);
      const settled = await Promise.allSettled(
        batch.map(async (sym) => {
          const bars = await fetchBarsWithRetry(sym, limit);
          return { sym, closes: bars.map(b => b.c), volumes: bars.map(b => b.v) };
        })
      );
      for (let j = 0; j < settled.length; j++) {
        const r = settled[j];
        if (r.status === "fulfilled") {
          results[r.value.sym] = r.value;
          loaded++;
        } else {
          failed++;
          failedSymbols.push(batch[j]);
        }
      }
      // Small delay between concurrency batches to avoid rate limits
      if (i + BAR_FETCH_CONCURRENCY < symbols.length) {
        await sleep(200);
      }
    }

    addLog(`Bars loaded: ${loaded}/${symbols.length} symbols (${failed} failed)`, "system");
    if (failedSymbols.length > 0) {
      addLog(`Failed symbols: ${failedSymbols.slice(0, 20).join(", ")}${failedSymbols.length > 20 ? ` (+${failedSymbols.length - 20} more)` : ""}`, "error");
    }
    return results;
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
  let cooldowns = {};              // strategy-specific: { "sym:strategy" → cycleNumber }
  let trendPositions = {};
  let trendBreakCounts = {};
  let circuitBreaker = { date: null, morningValue: null, tripped: false };
  let cycleNumber = 0;
  let tick = 0;

  // ── Volatility targeting state ──
  let dailyReturns = [];     // store daily portfolio returns for vol calculation
  let currentVolScale = 1.0; // current position size multiplier
  let previousDayValue = null; // previous day's portfolio value for daily return calc
  const activityLog = [];
  const portfolioHist = [];
  let tradeCount = { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 };

  // ── Strategy attribution ──
  // Tracks which strategy owns each position: { symbol → "ml" | "momentum" | "mean_reversion" }
  let positionStrategy = {};

  // ── Momentum strategy state ──
  let momRankings = {};            // { symbol → rank (1-based) } — refreshed each cycle
  let momEntryPrices = {};         // { symbol → entry price }
  let momPeakPrices = {};          // { symbol → peak since entry }
  let momEntryDates = {};          // { symbol → cycleNumber at entry }
  let momTradeCount = { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 };
  let mlTradeCount = { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 };

  // ── Mean Reversion strategy state ──
  let mrEntryPrices = {};          // { symbol → entry price }
  let mrEntryDates = {};           // { symbol → cycleNumber at entry }
  let mrTradeCount = { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 };
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
      addLog(`Fetching historical bars for ${UNIVERSE_SYMBOLS.length} universe symbols (concurrency=${BAR_FETCH_CONCURRENCY})...`, "system");
      const symbols = UNIVERSE_SYMBOLS;

      const barResults = await fetchBarsParallel(symbols, 250);
      let lowBarCount = 0;
      for (const sym of symbols) {
        const r = barResults[sym];
        if (r) {
          priceHist[sym] = r.closes;
          volHist[sym] = r.volumes;
          if (r.closes.length < 35) lowBarCount++;
        }
      }
      if (lowBarCount > 0) {
        addLog(`Warning: ${lowBarCount} symbols have < 35 bars`, "system");
      }

      // Always fetch SPY separately with retries — regime depends on it
      addLog("Fetching SPY bars (dedicated, 3 retries)...", "system");
      let spyLoaded = false;
      for (let attempt = 1; attempt <= 3; attempt++) {
        try {
          const spyBars = await fetchBars("SPY", 220);
          priceHist.SPY = spyBars.map(b => b.c);
          addLog(`SPY bars loaded: ${priceHist.SPY.length} bars for regime filter`, "system");
          spyLoaded = true;
          break;
        } catch (err) {
          addLog(`SPY bars attempt ${attempt}/3 failed: ${err.message}`, "error");
          if (attempt < 3) await sleep(2000);
        }
      }
      if (!spyLoaded) {
        addLog("CRITICAL: SPY bars failed all 3 attempts — defaulting to BULLISH regime", "error");
      }

      // 6. Initial regime calculation
      const regimeResult = computeRegime(priceHist.SPY);
      regime = regimeResult.regime;
      if (!spyLoaded) regime = "BULLISH"; // safe fallback if SPY data missing
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
      const symbols = UNIVERSE_SYMBOLS;
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
          nextHist[sym] = [...prev.slice(-260), snap.price];
          nextVol[sym] = [...prevVols.slice(-260), snap.volume || 0];
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

      // Strategy-specific cooldown: key = "sym:strategy"
      const isOnCooldown = (sym, strategy = "ml") => {
        const key = `${sym}:${strategy}`;
        const lossAt = cooldowns[key];
        const cooldownLen = strategy === "momentum" ? MOM.COOLDOWN_CYCLES
          : strategy === "mean_reversion" ? MR.COOLDOWN_CYCLES
          : RISK.LOSS_COOLDOWN_CYCLES;
        return lossAt !== undefined && (cycleNumber - lossAt) < cooldownLen;
      };
      const setCooldown = (sym, strategy = "ml") => {
        cooldowns[`${sym}:${strategy}`] = cycleNumber;
      };

      // Count positions by strategy for slot allocation
      const countByStrategy = () => {
        let ml = 0, mom = 0, mr = 0;
        for (const pos of currentPositions) {
          if (pos.symbol === "SPY" && idleSpyShares > 0) continue;
          if (trendPositions[pos.symbol]) continue;
          const strat = positionStrategy[pos.symbol] || "ml";
          if (strat === "momentum") mom++;
          else if (strat === "mean_reversion") mr++;
          else ml++;
        }
        return { ml, mom, mr, total: ml + mom + mr };
      };

      // Check if a strategy has slot capacity
      const hasSlotCapacity = (strategy, counts) => {
        if (counts.total >= SLOT_CONFIG.max) return false;
        const stratMap = { momentum: "momentum", mean_reversion: "mean_reversion", ml: "ml_medium" };
        const countMap = { momentum: counts.mom, mean_reversion: counts.mr, ml: counts.ml };
        const primaryKey = stratMap[strategy] || "ml_medium";
        const primaryUsed = countMap[strategy] || counts.ml;
        const primaryLimit = SLOT_CONFIG[primaryKey] || 0;
        // Can use primary slot
        if (primaryUsed < primaryLimit) return true;
        // Or flex slot if available
        const flexUsed = counts.total
          - Math.min(counts.ml, SLOT_CONFIG.ml_medium)
          - Math.min(counts.mom, SLOT_CONFIG.momentum)
          - Math.min(counts.mr, SLOT_CONFIG.mean_reversion);
        return flexUsed < SLOT_CONFIG.flex;
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

      // ── Volatility targeting: track daily returns and update scale ──
      if (previousDayValue !== null && previousDayValue > 0) {
        const dailyRet = (cyclePortfolioValue - previousDayValue) / previousDayValue;
        dailyReturns.push(dailyRet);
        if (dailyReturns.length > 30) dailyReturns.splice(0, dailyReturns.length - 30);
      }
      previousDayValue = cyclePortfolioValue;

      if (dailyReturns.length >= VOL_LOOKBACK) {
        const last20 = dailyReturns.slice(-VOL_LOOKBACK);
        const mean = last20.reduce((a, b) => a + b, 0) / last20.length;
        const variance = last20.reduce((s, r) => s + (r - mean) ** 2, 0) / last20.length;
        const dailyVol = Math.sqrt(variance);
        const annualVol = dailyVol * Math.sqrt(252);
        if (annualVol > 0) {
          let scale = VOL_TARGET / annualVol;
          scale = Math.max(MIN_LEVERAGE, Math.min(scale, MAX_LEVERAGE));
          currentVolScale = scale;
        } else {
          currentVolScale = 1.0;
        }
        addLog(`[vol-targeting] Vol: ${(annualVol * 100).toFixed(2)}%, Scale: ${currentVolScale.toFixed(3)}`, "system");
      } else {
        currentVolScale = 1.0;
      }

      // Fetch upcoming earnings (cached)
      const allSymbols = [...new Set([...UNIVERSE_SYMBOLS, ...currentPositions.map(p => p.symbol)])];
      const earningsMap = await fetchEarnings(allSymbols);
      const earningsSymbols = Object.keys(earningsMap);
      if (earningsSymbols.length > 0) {
        const earningsPreview = earningsSymbols.slice(0, 15).map(s => `${s} (${earningsMap[s]})`).join(", ");
        const suffix = earningsSymbols.length > 15 ? ` ... +${earningsSymbols.length - 15} more` : "";
        addLog(`Earnings next 7 days (${earningsSymbols.length}): ${earningsPreview}${suffix}`, "system");
      }

      // ── STEP 1: Trailing/Fixed Stop-loss & Take-profit on existing positions ──
      //    (Momentum positions have their own exit rules in STEP 1c)
      const closedSymbols = new Set();
      for (const pos of currentPositions) {
        const { symbol, qty, current_price: curr, unrealized_pl, unrealized_plpc } = pos;
        if (symbol === "SPY" && idleSpyShares > 0) continue;
        if (trendPositions[symbol]) continue;
        if (positionStrategy[symbol] === "momentum") continue;  // handled in STEP 1c
        if (positionStrategy[symbol] === "mean_reversion") continue;  // handled in STEP 1d

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
            const strat = positionStrategy[symbol] || "ml";
            await closePosition(symbol);
            delete trailingPeaks[symbol];
            closedSymbols.add(symbol);
            if (unrealized_plpc < 0) setCooldown(symbol, strat);
            // Clean up strategy-specific state
            if (strat === "momentum") {
              delete momEntryPrices[symbol];
              delete momPeakPrices[symbol];
              delete momEntryDates[symbol];
            } else if (strat === "mean_reversion") {
              delete mrEntryPrices[symbol];
              delete mrEntryDates[symbol];
            }
            delete positionStrategy[symbol];
            addLog(stopMsg, "sell");
            const stratTracker = strat === "momentum" ? momTradeCount : strat === "mean_reversion" ? mrTradeCount : mlTradeCount;
            tradeCount.sells++; stratTracker.sells++;
            if (unrealized_pl >= 0) { tradeCount.wins++; stratTracker.wins++; }
            else { tradeCount.losses++; stratTracker.losses++; }
            tradeCount.totalPnL += unrealized_pl;
            stratTracker.totalPnL += unrealized_pl;
            recordTrade({ symbol, action: "sell", shares: qty, price: curr, strategy: `${strat}-stop-loss`, portfolio_value: cyclePortfolioValue, pnl: unrealized_pl });
            dailyStats.sells++;
            if (unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            notify.send(`🛑 STOP-LOSS ${symbol} | ${qty} shares @ $${curr.toFixed(2)} | Loss: $${unrealized_pl.toFixed(2)} (${(unrealized_plpc * 100).toFixed(1)}%)`);
          } catch (err) {
            addLog(`Failed to close ${symbol}: ${err.message}`, "error");
          }
        } else if (unrealized_plpc >= RISK.TAKE_PROFIT_PCT) {
          try {
            const strat = positionStrategy[symbol] || "ml";
            await closePosition(symbol);
            delete trailingPeaks[symbol];
            closedSymbols.add(symbol);
            if (strat === "momentum") {
              delete momEntryPrices[symbol];
              delete momPeakPrices[symbol];
              delete momEntryDates[symbol];
            } else if (strat === "mean_reversion") {
              delete mrEntryPrices[symbol];
              delete mrEntryDates[symbol];
            }
            delete positionStrategy[symbol];
            addLog(`TAKE-PROFIT ${symbol}: ${qty} shares @ $${curr.toFixed(2)} | P&L: +$${unrealized_pl.toFixed(2)}`, "profit");
            const stratTracker = strat === "momentum" ? momTradeCount : strat === "mean_reversion" ? mrTradeCount : mlTradeCount;
            tradeCount.sells++; stratTracker.sells++;
            tradeCount.wins++; stratTracker.wins++;
            tradeCount.totalPnL += unrealized_pl;
            stratTracker.totalPnL += unrealized_pl;
            recordTrade({ symbol, action: "sell", shares: qty, price: curr, strategy: `${strat}-take-profit`, portfolio_value: cyclePortfolioValue, pnl: unrealized_pl });
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
            const strat = positionStrategy[pos.symbol] || "ml";
            await closePosition(pos.symbol);
            delete trailingPeaks[pos.symbol];
            closedSymbols.add(pos.symbol);
            if (strat === "momentum") {
              delete momEntryPrices[pos.symbol];
              delete momPeakPrices[pos.symbol];
              delete momEntryDates[pos.symbol];
            } else if (strat === "mean_reversion") {
              delete mrEntryPrices[pos.symbol];
              delete mrEntryDates[pos.symbol];
            }
            delete positionStrategy[pos.symbol];
            addLog(`EARNINGS SELL ${pos.symbol}: earnings tomorrow (${earningsDate}) -- exiting to avoid overnight announcement risk`, "sell");
            const stratTracker = strat === "momentum" ? momTradeCount : strat === "mean_reversion" ? mrTradeCount : mlTradeCount;
            tradeCount.sells++; stratTracker.sells++;
            if (pos.unrealized_pl >= 0) { tradeCount.wins++; stratTracker.wins++; }
            else { tradeCount.losses++; stratTracker.losses++; }
            tradeCount.totalPnL += pos.unrealized_pl;
            stratTracker.totalPnL += pos.unrealized_pl;
            recordTrade({ symbol: pos.symbol, action: "sell", shares: pos.qty, price: pos.current_price, strategy: `${strat}-earnings-sell`, portfolio_value: cyclePortfolioValue, pnl: pos.unrealized_pl });
            dailyStats.sells++;
            if (pos.unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            notify.send(`📉 SELL ${pos.symbol} | ${pos.qty} shares @ $${pos.current_price.toFixed(2)} | Earnings tomorrow — P&L: $${pos.unrealized_pl.toFixed(2)} (${(pos.unrealized_plpc * 100).toFixed(1)}%)`);
          } catch (err) {
            addLog(`Earnings sell failed ${pos.symbol}: ${err.message}`, "error");
          }
        }
      }

      // ── STEP 1c: Momentum-specific exits ──
      for (const pos of currentPositions) {
        const sym = pos.symbol;
        if (closedSymbols.has(sym)) continue;
        if (sym === "SPY" && idleSpyShares > 0) continue;
        if (trendPositions[sym]) continue;
        if (positionStrategy[sym] !== "momentum") continue;

        const { qty, current_price: curr, unrealized_pl } = pos;
        const entryPrice = momEntryPrices[sym] || pos.avg_entry_price;
        const peak = momPeakPrices[sym] || curr;
        const entryCycle = momEntryDates[sym] || cycleNumber;

        // Update peak price tracking
        if (curr > peak) momPeakPrices[sym] = curr;
        const currentPeak = momPeakPrices[sym] || curr;

        let exitTriggered = false;
        let exitReason = "";
        let exitMsg = "";

        // 1. Stop-loss: -8% from entry
        const entryReturn = (curr - entryPrice) / entryPrice;
        if (entryReturn <= MOM.STOP_LOSS) {
          exitTriggered = true;
          exitReason = "mom-stop-loss";
          exitMsg = `MOM STOP-LOSS ${sym}: ${(entryReturn * 100).toFixed(1)}% from entry $${entryPrice.toFixed(2)}`;
        }

        // 2. Take-profit: +20% from entry
        if (!exitTriggered && entryReturn >= MOM.TAKE_PROFIT) {
          exitTriggered = true;
          exitReason = "mom-take-profit";
          exitMsg = `MOM TAKE-PROFIT ${sym}: +${(entryReturn * 100).toFixed(1)}% from entry $${entryPrice.toFixed(2)}`;
        }

        // 3. Trailing stop: -8% from peak
        if (!exitTriggered) {
          const dropFromPeak = (curr - currentPeak) / currentPeak;
          if (dropFromPeak <= MOM.TRAIL_STOP) {
            exitTriggered = true;
            exitReason = "mom-trail-stop";
            exitMsg = `MOM TRAIL-STOP ${sym}: peak $${currentPeak.toFixed(2)}, drop ${(dropFromPeak * 100).toFixed(1)}%`;
          }
        }

        // 4. Rank break: falls out of top 20
        if (!exitTriggered && momRankings[sym] && momRankings[sym] > MOM.RANK_BREAK) {
          exitTriggered = true;
          exitReason = "mom-rank-break";
          exitMsg = `MOM RANK-BREAK ${sym}: rank dropped to #${momRankings[sym]} (threshold: top ${MOM.RANK_BREAK})`;
        }

        // 5. Max hold: 60 cycles (~60 trading days at 1 cycle/minute during market hours)
        //    In live trading, 1 cycle = 1 minute, so 60 trading days ≈ 60*390 = 23400 cycles.
        //    But we approximate: each day ~390 cycles, so max hold = 60 * 390 cycles.
        //    Simpler: track calendar days from entry instead.
        if (!exitTriggered) {
          const cyclesHeld = cycleNumber - entryCycle;
          // ~390 trade cycles per trading day (6.5h * 60min/h)
          const approxDaysHeld = cyclesHeld / 390;
          if (approxDaysHeld >= MOM.MAX_HOLD_DAYS) {
            exitTriggered = true;
            exitReason = "mom-max-hold";
            exitMsg = `MOM MAX-HOLD ${sym}: held ~${approxDaysHeld.toFixed(0)} trading days (max: ${MOM.MAX_HOLD_DAYS})`;
          }
        }

        if (exitTriggered) {
          try {
            await closePosition(sym);
            closedSymbols.add(sym);
            delete momEntryPrices[sym];
            delete momPeakPrices[sym];
            delete momEntryDates[sym];
            delete positionStrategy[sym];
            if (unrealized_pl < 0) setCooldown(sym, "momentum");
            addLog(exitMsg, unrealized_pl >= 0 ? "profit" : "sell");
            tradeCount.sells++;
            momTradeCount.sells++;
            if (unrealized_pl >= 0) { tradeCount.wins++; momTradeCount.wins++; }
            else { tradeCount.losses++; momTradeCount.losses++; }
            tradeCount.totalPnL += unrealized_pl;
            momTradeCount.totalPnL += unrealized_pl;
            recordTrade({ symbol: sym, action: "sell", shares: qty, price: curr, strategy: exitReason, portfolio_value: cyclePortfolioValue, pnl: unrealized_pl });
            dailyStats.sells++;
            if (unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            const pnlSign = unrealized_pl >= 0 ? "+" : "";
            notify.send(`📊 MOM ${exitReason.replace("mom-", "").toUpperCase()} ${sym} | ${qty} shares @ $${curr.toFixed(2)} | P&L: ${pnlSign}$${unrealized_pl.toFixed(2)}`);
          } catch (err) {
            addLog(`Momentum exit failed ${sym}: ${err.message}`, "error");
          }
        }
      }

      // ── STEP 1d: Mean Reversion-specific exits ──
      for (const pos of currentPositions) {
        const sym = pos.symbol;
        if (closedSymbols.has(sym)) continue;
        if (sym === "SPY" && idleSpyShares > 0) continue;
        if (trendPositions[sym]) continue;
        if (positionStrategy[sym] !== "mean_reversion") continue;

        const { qty, current_price: curr, unrealized_pl } = pos;
        const entryPrice = mrEntryPrices[sym] || pos.avg_entry_price;
        const entryCycle = mrEntryDates[sym] || cycleNumber;

        let exitTriggered = false;
        let exitReason = "";
        let exitMsg = "";

        // 1. Stop-loss: -10% from entry
        const entryReturn = (curr - entryPrice) / entryPrice;
        if (entryReturn <= MR.STOP_LOSS) {
          exitTriggered = true;
          exitReason = "mr-stop-loss";
          exitMsg = `MR STOP-LOSS ${sym}: ${(entryReturn * 100).toFixed(1)}% from entry $${entryPrice.toFixed(2)}`;
        }

        // 2. Take-profit: price recovers to 20-day SMA
        if (!exitTriggered) {
          const prices = priceHist[sym];
          if (prices && prices.length >= MR.SMA_SHORT) {
            const sma20 = sma(prices, MR.SMA_SHORT);
            if (sma20 !== null && curr >= sma20) {
              exitTriggered = true;
              exitReason = "mr-take-profit-sma20";
              exitMsg = `MR TAKE-PROFIT ${sym}: price $${curr.toFixed(2)} recovered to 20-SMA $${sma20.toFixed(2)}`;
            }
          }
        }

        // 3. Max hold: 10 trading days (~10 cycles at 1/day equivalent)
        if (!exitTriggered) {
          const cyclesHeld = cycleNumber - entryCycle;
          const approxDaysHeld = cyclesHeld / 390;
          if (approxDaysHeld >= MR.MAX_HOLD_DAYS) {
            exitTriggered = true;
            exitReason = "mr-max-hold";
            exitMsg = `MR MAX-HOLD ${sym}: held ~${approxDaysHeld.toFixed(0)} trading days (max: ${MR.MAX_HOLD_DAYS})`;
          }
        }

        if (exitTriggered) {
          try {
            await closePosition(sym);
            closedSymbols.add(sym);
            delete mrEntryPrices[sym];
            delete mrEntryDates[sym];
            delete positionStrategy[sym];
            if (unrealized_pl < 0) setCooldown(sym, "mean_reversion");
            addLog(exitMsg, unrealized_pl >= 0 ? "profit" : "sell");
            tradeCount.sells++;
            mrTradeCount.sells++;
            if (unrealized_pl >= 0) { tradeCount.wins++; mrTradeCount.wins++; }
            else { tradeCount.losses++; mrTradeCount.losses++; }
            tradeCount.totalPnL += unrealized_pl;
            mrTradeCount.totalPnL += unrealized_pl;
            recordTrade({ symbol: sym, action: "sell", shares: qty, price: curr, strategy: exitReason, portfolio_value: cyclePortfolioValue, pnl: unrealized_pl });
            dailyStats.sells++;
            if (unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            const pnlSign = unrealized_pl >= 0 ? "+" : "";
            notify.send(`🔄 MR ${exitReason.replace("mr-", "").toUpperCase()} ${sym} | ${qty} shares @ $${curr.toFixed(2)} | P&L: ${pnlSign}$${unrealized_pl.toFixed(2)}`);
          } catch (err) {
            addLog(`Mean reversion exit failed ${sym}: ${err.message}`, "error");
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
        addLog(`ML V4 active -- ${buyCount} BUY signal${buyCount !== 1 ? "s" : ""} (top-${buyCount} cross-sectional ranking) | vol scale: ${currentVolScale.toFixed(3)}`, "system");
      } else {
        addLog("ML server offline -- using consensus engine (fallback mode)", "system");
      }

      const heldSymbols = new Set(currentPositions.map(p => p.symbol));
      const activePositionCount = currentPositions.filter(p => p.symbol !== "SPY").length;
      const opportunities = [];

      for (const sym of UNIVERSE_SYMBOLS) {
        const prices = priceHist[sym];
        const isMLBuy = mlActive && mlMap[sym]?.is_top_5 === true;

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
              const blStrat = positionStrategy[sym] || "ml";
              if (posData && posData.unrealized_plpc < 0) setCooldown(sym, blStrat);
              if (blStrat === "momentum") { delete momEntryPrices[sym]; delete momPeakPrices[sym]; delete momEntryDates[sym]; }
              else if (blStrat === "mean_reversion") { delete mrEntryPrices[sym]; delete mrEntryDates[sym]; }
              delete positionStrategy[sym];
              addLog(`SELL ${sym}: ${analysis.consensus} -- closing blacklisted position`, "sell");
              if (posData) {
                const blTracker = blStrat === "momentum" ? momTradeCount : blStrat === "mean_reversion" ? mrTradeCount : mlTradeCount;
                tradeCount.sells++; blTracker.sells++;
                if (posData.unrealized_pl >= 0) { tradeCount.wins++; blTracker.wins++; }
                else { tradeCount.losses++; blTracker.losses++; }
                tradeCount.totalPnL += posData.unrealized_pl;
                blTracker.totalPnL += posData.unrealized_pl;
                recordTrade({ symbol: sym, action: "sell", shares: posData.qty, price: posData.current_price, strategy: blStrat, portfolio_value: cyclePortfolioValue, pnl: posData.unrealized_pl });
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

        // Sell on SELL consensus (only for ML positions — momentum has its own exit rules)
        if (heldSymbols.has(sym) && !trendPositions[sym] && positionStrategy[sym] !== "momentum" && positionStrategy[sym] !== "mean_reversion" && (analysis.consensus === "STRONG SELL" || analysis.consensus === "SELL")) {
          try {
            await closePosition(sym);
            const posData = currentPositions.find(p => p.symbol === sym);
            if (posData && posData.unrealized_plpc < 0) setCooldown(sym, "ml");
            delete positionStrategy[sym];
            addLog(`SELL ${sym}: ${analysis.consensus} -- closing position`, "sell");
            if (posData) {
              tradeCount.sells++; mlTradeCount.sells++;
              if (posData.unrealized_pl >= 0) { tradeCount.wins++; mlTradeCount.wins++; }
              else { tradeCount.losses++; mlTradeCount.losses++; }
              tradeCount.totalPnL += posData.unrealized_pl;
              mlTradeCount.totalPnL += posData.unrealized_pl;
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

        if (isOnCooldown(sym, "ml")) {
          const cdKey = `${sym}:ml`;
          const remaining = RISK.LOSS_COOLDOWN_CYCLES - (cycleNumber - cooldowns[cdKey]);
          if (isMLBuy) {
            addLog(`EVAL ${sym}: conf ${(mlMap[sym].probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | BLOCKED: ML cooldown, ${remaining} cycle${remaining !== 1 ? "s" : ""} remaining`, "system");
          }
          continue;
        }

        // Earnings proximity check (pre-computed for ML eval log)
        const earningsDate = earningsMap[sym];
        const earningsDays = earningsDate ? daysUntilEarnings(earningsDate) : null;
        const earningsBlocked = earningsDays !== null && earningsDays >= 0 && earningsDays <= 3;

        if (mlActive) {
          // ML V4: top-N cross-sectional ranking (BUY if is_top_5)
          const mlSig = mlMap[sym];
          if (mlSig && mlSig.is_top_5) {
            if (regime === "CAUTIOUS" && mlSig.rank > 2) {
              addLog(`EVAL ${sym}: rank #${mlSig.rank} conf ${(mlSig.probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | BLOCKED: CAUTIOUS regime, only top 2 picks allowed`, "system");
            } else {
              addLog(`EVAL ${sym}: rank #${mlSig.rank} conf ${(mlSig.probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | earnings blocked: ${earningsBlocked}${earningsBlocked ? ` (${earningsDays}d -> ${earningsDate})` : ""} | cooldown: false | PASSED -> added to candidates`, "system");
              opportunities.push({
                sym,
                score: mlSig.probability,
                price: prices[prices.length - 1],
                consensus: `ML V4 #${mlSig.rank} (${(mlSig.probability * 100).toFixed(0)}%)`,
                rsiVal: analysis.indicators.rsi,
                mlConf: mlSig.probability,
              });
            }
          } else {
            if (mlSig && mlSig.rank <= 10) {
              addLog(`ML SKIP ${sym} -- rank #${mlSig.rank}, not in top 5`, "system");
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

      // ── STEP 2b: Compute momentum signals ──
      const momResult = computeMomentumSignals(priceHist, volHist, heldSymbols, earningsMap);
      momRankings = momResult.rankings;  // store globally for exit checks

      // Filter momentum signals: check cooldowns and regime
      const momOpportunities = [];
      if (regime !== "BEARISH") {
        for (const sig of momResult.signals) {
          if (isOnCooldown(sig.sym, "momentum")) continue;
          // First-to-fire: skip if ML already claimed this symbol
          if (opportunities.some(o => o.sym === sig.sym)) continue;
          momOpportunities.push(sig);
        }
      }

      if (momOpportunities.length > 0) {
        addLog(`MOM signals: ${momOpportunities.length} candidate${momOpportunities.length !== 1 ? "s" : ""} (${momOpportunities.map(o => `${o.sym} #${o.rank}`).join(", ")})`, "system");
      }

      // ── STEP 2c: Compute mean reversion signals ──
      const mrRawSignals = computeMeanReversionSignals(priceHist, volHist, heldSymbols, earningsMap);

      // Filter MR signals: check cooldowns and regime, first-to-fire overlap
      const mrOpportunities = [];
      if (regime !== "BEARISH") {
        for (const sig of mrRawSignals) {
          if (isOnCooldown(sig.sym, "mean_reversion")) continue;
          // First-to-fire: skip if ML or momentum already claimed this symbol
          if (opportunities.some(o => o.sym === sig.sym)) continue;
          if (momOpportunities.some(o => o.sym === sig.sym)) continue;
          mrOpportunities.push(sig);
        }
      }

      if (mrOpportunities.length > 0) {
        addLog(`MR signals: ${mrOpportunities.length} candidate${mrOpportunities.length !== 1 ? "s" : ""} (${mrOpportunities.map(o => `${o.sym} ${(o.drop30d * 100).toFixed(1)}%`).join(", ")})`, "system");
      }

      // ── STEP 3: Rank & execute buys (strategy-aware slot allocation) ──
      // Tag ML opportunities with strategy
      for (const opp of opportunities) {
        if (!opp.strategy) opp.strategy = "ml";
      }

      // Merge: ML first (highest priority), then momentum, then mean reversion
      const allOpportunities = [...opportunities, ...momOpportunities, ...mrOpportunities];
      // Sort within each strategy group by score, ML > momentum > MR
      const stratPriority = { ml: 0, momentum: 1, mean_reversion: 2 };
      allOpportunities.sort((a, b) => {
        const pa = stratPriority[a.strategy] ?? 9;
        const pb = stratPriority[b.strategy] ?? 9;
        if (pa !== pb) return pa - pb;
        return b.score - a.score;
      });

      const counts = countByStrategy();
      const slotsAvail = SLOT_CONFIG.max - counts.total;

      // Pre-execution diagnostic summary
      if (allOpportunities.length > 0) {
        addLog(`BUY FILTER CHECK -- ${allOpportunities.length} candidate${allOpportunities.length !== 1 ? "s" : ""} (${opportunities.length} ML + ${momOpportunities.length} MOM + ${mrOpportunities.length} MR) | positions: ${counts.total}/${SLOT_CONFIG.max} (ML ${counts.ml}/${SLOT_CONFIG.ml_medium}, MOM ${counts.mom}/${SLOT_CONFIG.momentum}, MR ${counts.mr}/${SLOT_CONFIG.mean_reversion}) | slots open: ${slotsAvail} | cash: $${cycleCash.toLocaleString("en-US", { maximumFractionDigits: 0 })}`, "system");
      }

      if (skipNewBuys) {
        if (allOpportunities.length > 0) {
          addLog(`Close buffer active -- skipping ${allOpportunities.length} pending buys: ${allOpportunities.map(o => o.sym).join(", ")}`, "system");
        }
        // Update state with fresh data
        positionsRaw = await getPositions();
        const updatedAcct = await getAccount();
        cash = updatedAcct.cash;
        portfolioValue = updatedAcct.portfolio_value;
        return;
      }

      if (slotsAvail <= 0) {
        if (allOpportunities.length > 0) {
          addLog(`Position limit full (${counts.total}/${SLOT_CONFIG.max}) -- skipping all ${allOpportunities.length} buys: ${allOpportunities.map(o => o.sym).join(", ")}`, "system");
        }
      }

      // ── STEP 3a: Sell idle SPY to free cash before buys ──
      const spyPos = currentPositions.find(p => p.symbol === "SPY");
      if (idleSpyShares > 0 && spyPos && allOpportunities.length > 0) {
        const spyShareCount = spyPos.qty;
        const spyPrice = spyPos.current_price;
        const idleValue = spyPos.market_value;
        try {
          await closePosition("SPY");
          idleSpyShares = 0;
          addLog(`[idle-spy] Selling ${spyShareCount} SPY shares ($${idleValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}) to fund ${allOpportunities.length} pick${allOpportunities.length !== 1 ? "s" : ""}`, "system");
          recordTrade({ symbol: "SPY", action: "sell", shares: spyShareCount, price: spyPrice, strategy: "idle-spy", portfolio_value: cyclePortfolioValue });
          notify.send(`🅿️ SPY IDLE SELL | ${spyShareCount} shares @ $${spyPrice.toFixed(2)} | Freeing cash for ${allOpportunities.length} pick${allOpportunities.length !== 1 ? "s" : ""}`);
          const freshAcct = await getAccount();
          cycleCash = freshAcct.cash;
        } catch (err) {
          addLog(`[idle-spy] Failed to sell SPY: ${err.message}`, "error");
        }
      }

      // Track sectors bought this cycle and live slot counts
      const boughtThisCycle = {};
      let liveCounts = { ...counts };  // mutable copy for tracking during execution

      for (const opp of allOpportunities) {
        // Check strategy-specific slot capacity
        if (!hasSlotCapacity(opp.strategy, liveCounts)) {
          addLog(`SKIP ${opp.sym} -- no ${opp.strategy} slot available (ML ${liveCounts.ml}/${SLOT_CONFIG.ml_medium}, MOM ${liveCounts.mom}/${SLOT_CONFIG.momentum}, MR ${liveCounts.mr}/${SLOT_CONFIG.mean_reversion}, total ${liveCounts.total}/${SLOT_CONFIG.max})`, "system");
          continue;
        }

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

        // Volume confirmation check -- skip for ML-driven, momentum, and MR trades
        if (opp.strategy !== "ml" && opp.strategy !== "momentum" && opp.strategy !== "mean_reversion") {
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

        // Skip if earnings within 3 calendar days (already filtered for momentum and MR in their signal functions)
        if (opp.strategy !== "momentum" && opp.strategy !== "mean_reversion") {
          const earningsDate = earningsMap[opp.sym];
          if (earningsDate) {
            const days = daysUntilEarnings(earningsDate);
            if (days >= 0 && days <= 3) {
              addLog(`SKIP ${opp.sym}: earnings in ${days} day(s) on ${earningsDate}`, "system");
              continue;
            }
          }
        }

        // Position sizing — strategy-specific, with vol targeting
        let dynPositionPct;
        if (opp.strategy === "momentum" || opp.strategy === "mean_reversion") {
          // Momentum/MR: fixed 12% or 8% based on ATR (already computed in signal)
          dynPositionPct = opp.positionPct;
        } else {
          // ML: ATR-based dynamic sizing with regime and confidence multipliers
          const stockAtr = atr(priceHist[opp.sym], 14);
          const atrPct = stockAtr ? stockAtr / opp.price : RISK.ATR_TARGET_PCT;
          const volatilityScale = RISK.ATR_TARGET_PCT / atrPct;
          const regimeMult = regime === "CAUTIOUS" ? 0.75 : 1.0;
          const mlMult = opp.mlConf != null
            ? Math.min(1.0, Math.max(0.60, opp.mlConf * 1.6 - 0.28))
            : 1.0;
          dynPositionPct = Math.max(
            RISK.MIN_POSITION_PCT,
            Math.min(RISK.MAX_POSITION_PCT, RISK.MAX_POSITION_PCT * volatilityScale * regimeMult * mlMult)
          );
        }

        // Apply volatility targeting scale to ALL strategies
        const maxAlloc = cyclePortfolioValue * dynPositionPct * currentVolScale;
        const allocCash = Math.min(maxAlloc, cycleCash * RISK.MAX_CASH_DEPLOY_PCT);
        if (allocCash < opp.price) {
          addLog(`SKIP ${opp.sym} -- insufficient cash: need $${opp.price.toFixed(2)}/share, alloc $${allocCash.toFixed(2)} (${(dynPositionPct * 100).toFixed(1)}% of portfolio)`, "system");
          continue;
        }

        const shares = Math.floor(allocCash / opp.price);
        if (shares <= 0) {
          addLog(`SKIP ${opp.sym} -- 0 shares at $${opp.price.toFixed(2)}/share with $${allocCash.toFixed(2)} allocated`, "system");
          continue;
        }

        try {
          const order = await placeOrder({ symbol: opp.sym, qty: shares, side: "buy", type: "market" });
          boughtThisCycle[sector] = (boughtThisCycle[sector] || 0) + 1;

          // Track strategy attribution
          positionStrategy[opp.sym] = opp.strategy;
          const stratTracker = opp.strategy === "momentum" ? momTradeCount : opp.strategy === "mean_reversion" ? mrTradeCount : mlTradeCount;
          tradeCount.buys++; stratTracker.buys++;
          dailyStats.buys++;

          // Update live slot counts
          if (opp.strategy === "momentum") {
            liveCounts.mom++;
            // Track momentum entry state
            momEntryPrices[opp.sym] = opp.price;
            momPeakPrices[opp.sym] = opp.price;
            momEntryDates[opp.sym] = cycleNumber;
          } else if (opp.strategy === "mean_reversion") {
            liveCounts.mr++;
            // Track MR entry state
            mrEntryPrices[opp.sym] = opp.price;
            mrEntryDates[opp.sym] = cycleNumber;
          } else {
            liveCounts.ml++;
          }
          liveCounts.total++;

          if (opp.strategy === "momentum") {
            addLog(`MOM BUY ${opp.sym}: ${shares} shares | ${opp.consensus} | alloc ${(dynPositionPct * 100).toFixed(1)}% | Order: ${order.status}`, "buy");
            recordTrade({ symbol: opp.sym, action: "buy", shares, price: opp.price, strategy: "momentum", portfolio_value: cyclePortfolioValue });
            notify.send(`📊 MOM BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | ${opp.consensus} | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`);
          } else if (opp.strategy === "mean_reversion") {
            addLog(`MR BUY ${opp.sym}: ${shares} shares | ${opp.consensus} | conf ${(opp.score * 100).toFixed(0)}% | alloc ${(dynPositionPct * 100).toFixed(1)}% | Order: ${order.status}`, "buy");
            recordTrade({ symbol: opp.sym, action: "buy", shares, price: opp.price, strategy: "mean_reversion", portfolio_value: cyclePortfolioValue });
            notify.send(`🔄 MR BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | ${opp.consensus} | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`);
          } else {
            const mlNote = opp.mlConf != null ? `, ML ${(opp.mlConf * 100).toFixed(0)}% conf` : "";
            addLog(`ML BUY ${opp.sym}: ${shares} shares | ${opp.consensus} (score: ${opp.score.toFixed(2)}) | alloc ${(dynPositionPct * 100).toFixed(1)}%${mlNote} | Order: ${order.status}`, "buy");
            recordTrade({ symbol: opp.sym, action: "buy", shares, price: opp.price, strategy: opp.mlConf != null ? "ml" : "consensus", ml_confidence: opp.mlConf, portfolio_value: cyclePortfolioValue });
            notify.send(opp.mlConf != null
              ? `🤖 ML BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | Confidence: ${(opp.mlConf * 100).toFixed(0)}% | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
              : `📊 BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | ${opp.consensus} | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`);
          }
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
            for (const sym of UNIVERSE_SYMBOLS) {
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
          // Strategy breakdown
          let mlPos = 0, momPos = 0, mrPos = 0;
          for (const p of positionsRaw) {
            if (p.symbol === "SPY" && idleSpyShares > 0) continue;
            if (trendPositions[p.symbol]) continue;
            const strat = positionStrategy[p.symbol];
            if (strat === "momentum") momPos++;
            else if (strat === "mean_reversion") mrPos++;
            else mlPos++;
          }
          const flexUsed = Math.max(0, mlPos + momPos + mrPos - SLOT_CONFIG.ml_medium - SLOT_CONFIG.momentum - SLOT_CONFIG.mean_reversion);
          summary += `\nSlots: ML ${mlPos}/${SLOT_CONFIG.ml_medium} | MOM ${momPos}/${SLOT_CONFIG.momentum} | MR ${mrPos}/${SLOT_CONFIG.mean_reversion} | Flex ${flexUsed}/${SLOT_CONFIG.flex}`;
          summary += `\nML P&L: $${mlTradeCount.totalPnL.toFixed(0)} (${mlTradeCount.wins}W/${mlTradeCount.losses}L)`;
          summary += `\nMOM P&L: $${momTradeCount.totalPnL.toFixed(0)} (${momTradeCount.wins}W/${momTradeCount.losses}L)`;
          summary += `\nMR P&L: $${mrTradeCount.totalPnL.toFixed(0)} (${mrTradeCount.wins}W/${mrTradeCount.losses}L)`;
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
    // Count positions by strategy
    let mlPositions = 0, momPositions = 0, mrPositions = 0;
    for (const sym of Object.keys(positions)) {
      if (sym === "SPY" && idleSpyShares > 0) continue;
      if (trendPositions[sym]) continue;
      const strat = positionStrategy[sym] || "ml";
      if (strat === "momentum") momPositions++;
      else if (strat === "mean_reversion") mrPositions++;
      else mlPositions++;
    }

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
      volTargeting: { scale: currentVolScale, dailyReturnsCount: dailyReturns.length },
      circuitBreaker: { ...circuitBreaker },
      // Strategy breakdown
      positionStrategy: { ...positionStrategy },
      slotConfig: SLOT_CONFIG,
      strategyStats: {
        ml: { positions: mlPositions, slots: SLOT_CONFIG.ml_medium, trades: { ...mlTradeCount } },
        momentum: { positions: momPositions, slots: SLOT_CONFIG.momentum, trades: { ...momTradeCount } },
        mean_reversion: { positions: mrPositions, slots: SLOT_CONFIG.mean_reversion, trades: { ...mrTradeCount } },
        flex: { used: Math.max(0, mlPositions + momPositions + mrPositions - SLOT_CONFIG.ml_medium - SLOT_CONFIG.momentum - SLOT_CONFIG.mean_reversion), total: SLOT_CONFIG.flex },
      },
      momRankings: { ...momRankings },
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
