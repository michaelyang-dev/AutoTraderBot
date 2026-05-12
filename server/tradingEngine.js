// ══════════════════════════════════════════════════════════════════════
//  TRADING ENGINE — Server-side consolidated trading logic
//  Runs all signal analysis, regime detection, trend following,
//  and trade execution directly on the Express server.
//  No React dependency — uses Alpaca SDK + Node http for v9.6 factor signals.
// ══════════════════════════════════════════════════════════════════════

const http = require("http");
const fs = require("fs");
const path = require("path");
const notify = require("./notifications");
const journal = require("../db/journal");
const { toAlpacaSymbol, fromAlpacaSymbol } = require("./symbolMap");

// ══════════════════════════════════════════
//  CONSTANTS (inlined from frontend config)
// ══════════════════════════════════════════

const INITIAL_CASH = 100000;

const RISK = {
  MAX_POSITION_PCT: 0.156,              // v10.2: 8 positions * 15.6% = 125% equity (1.25x leverage)
  STOP_LOSS_PCT: -0.35,               // v10.2: -35% trailing stop from peak
  TAKE_PROFIT_PCT: 1.00,              // effectively disabled — exits via rebalance
  MAX_OPEN_POSITIONS: 8,              // top-8 picks
  MAX_CASH_DEPLOY_PCT: 1.25,          // v10.2: 1.25x leverage (use margin)
  REBALANCE_INTERVAL: 5,
  TRAILING_STOP_PCT: 0.35,            // v10.2: -35% trailing stop (backtested: +27.4% CAGR, 1.41 Sharpe with leverage)
  USE_TRAILING_STOP: true,            // v10.2: trailing stop enabled
  ATR_TARGET_PCT: 0.01,
  MIN_POSITION_PCT: 0.03,
  LOSS_COOLDOWN_CYCLES: 3,
  MIN_COOLDOWN_MS: 30 * 60 * 1000,    // 30 min minimum cooldown regardless of cycle count
  SECTOR_MAX_POSITIONS: { International: 2, Commodity: 2, Bond: 2, Volatility: 1 },
  VOLUME_CONFIRM_RATIO: 1.5,
  MIN_POSITION_DOLLARS: 5000,   // skip positions smaller than $5k
};

// ── Strategy kill switches ──
const DISABLE_TREND_STRATEGY = true;  // Backtest proved trend hurts (-8.41pp alpha, -0.364 Sharpe)
const ENABLE_MOMENTUM_REGIME_FILTER = true;  // Skip momentum buys when SPY < 50-SMA (OOS: +11.8pp CAGR, +0.69 Sharpe, -7.6pp DD)
const ENABLE_SPY_PARKING = false;  // Walk-forward validated OFF: +0.42 Sharpe, +3.4% CAGR (commit f8bd464)

// ── Multi-strategy slot allocation ──
// v9.6: all slots go to factor strategy (multi-strategy engine handles diversification)
const SLOT_CONFIG = {
  ml_medium: 8,          // v9.6 top-8 picks (kept as "ml_medium" for API compatibility)
  momentum: 0,           // Disabled — v9.6 has its own momentum strategy (S1)
  mean_reversion: 0,     // Disabled
  mega_cap: 0,           // Disabled
  flex: 0,
  max: 8,                // Hard cap (= RISK.MAX_OPEN_POSITIONS)
};

// ── Momentum strategy parameters ──
const MOM = {
  LOOKBACK: 252,           // 12-month lookback for 12-1 momentum
  SKIP: 21,                // skip most recent 21 days (reversal avoidance)
  TOP_N: 5,                // top 5 signals per cycle
  SMA_PERIOD: 200,         // trend filter
  VOL_PERIOD: 20,          // avg volume window
  VOL_MIN: 500000,         // minimum 20-day avg volume
  ATR_PERIOD: 14,
  STOP_LOSS: -0.08,
  TAKE_PROFIT: 0.20,
  TRAIL_STOP: -0.08,
  RANK_BREAK: 20,
  MAX_HOLD_DAYS: 60,
  BASE_PCT: 0.12,
  HIGH_ATR_PCT: 0.08,
  HIGH_ATR_THRESH: 0.03,
  COOLDOWN_CYCLES: 5,
  HIGH_52WK_THRESHOLD: 0.92,  // within 8% of 52-week high
  CONSISTENCY_THRESHOLD: 0.55, // 55%+ positive days
  CONSISTENCY_BOOST: 0.10,     // +10% rank boost
};

// ── Mean Reversion strategy parameters ──
const MR = {
  SMA_PERIOD: 200,          // trend filter (above 200-SMA)
  RSI_PERIOD: 14,           // RSI lookback
  RSI_ENTRY: 30,            // RSI < 30 entry
  RSI_EXIT: 50,             // RSI > 50 exit (mean reversion complete)
  BB_PERIOD: 20,            // Bollinger Band period
  BB_STD: 2.0,              // Bollinger Band std devs
  VOL_SPIKE: 1.5,           // volume >= 1.5x 20-day avg
  VOL_PERIOD: 20,           // avg volume window
  VOL_MIN: 500000,          // minimum 20-day avg volume
  ATR_PERIOD: 14,
  ATR_STOP_MULT: 2.0,       // stop = entry - 2xATR
  ATR_TP_MULT: 3.0,         // TP = entry + 3xATR
  MAX_HOLD_DAYS: 15,        // max hold days
  BASE_PCT: 0.12,
  HIGH_ATR_PCT: 0.08,
  HIGH_ATR_THRESH: 0.04,
  COOLDOWN_CYCLES: 5,
};

// ── Mega-Cap Overlay strategy parameters ──
const MEGACAP = {
  TOP_N_UNIVERSE: 15,
  PICKS: 3,
  LOOKBACK: 252,          // 12-month lookback (was 60)
  SKIP: 21,               // skip most recent month
  SMA_PERIOD: 200,
  SMA_EXIT_PERIOD: 50,    // 50-day SMA trend break exit
  STOP_LOSS: -0.05,
  TAKE_PROFIT: 0.20,
  MAX_HOLD_DAYS: 90,
  BASE_PCT: 0.12,
  COOLDOWN_CYCLES: 2,
  HIGH_52WK_THRESHOLD: 0.95,  // within 5% of 52-week high
  UNIVERSE: [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "V", "MA", "COST",
  ],
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
  "VIXY","VIXM","UUP",
];

function loadUniverseSymbols() {
  // v10: Load full SP1500 from weekly Wikipedia scrape
  const sp1500File = path.join(__dirname, "..", "ml_service", "data", "sp1500_members.json");
  const sp500File = path.join(__dirname, "..", "ml_service", "data", "sp500_constituents.json");
  let members = [];

  // Try SP1500 first (preferred)
  try {
    const raw = JSON.parse(fs.readFileSync(sp1500File, "utf8"));
    members = [...(raw.sp500 || []), ...(raw.sp400 || []), ...(raw.sp600 || [])];
    if (members.length < 1000) members = [];  // sanity check
  } catch (e) { /* fall through */ }

  // Fallback to SP500-only
  if (members.length === 0) {
    try {
      members = JSON.parse(fs.readFileSync(sp500File, "utf8"));
      if (!Array.isArray(members) || members.length < 400) members = [];
    } catch (e) { /* fall through */ }
  }

  // Last resort fallback
  if (members.length === 0) {
    members = [
      "AAPL","GOOGL","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD","JPM","V","UNH",
      "CRM","ORCL","ADBE","CSCO","QCOM","COST","WMT","HD","LOW",
      "LLY","JNJ","ABBV","BAC","GS","MS","CVX","XOM","CAT","DE","BA",
    ];
  }

  return [...new Set([...members, ...ETF_SYMBOLS])].sort();
}

// UNIVERSE_SYMBOLS: flat array of ticker strings (~1540 SP1500 + ETFs)
let UNIVERSE_SYMBOLS = loadUniverseSymbols();

// Batch size for Alpaca API calls (snapshots support up to 200, bars individually)
const SNAPSHOT_BATCH_SIZE = 100;
const BAR_FETCH_CONCURRENCY = 5;   // parallel bar fetches (reduced from 20 for rate limits)
const BAR_FETCH_RETRIES = 3;       // retry failed bar fetches with exponential backoff
const SNAPSHOT_BATCH_DELAY_MS = 500; // delay between snapshot batches

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
// ── Multi-layer circuit breaker ──
const CB_DAILY_LIMIT  = 0.04;  // Layer 1: 4% max daily loss
const CB_WEEKLY_LIMIT = 0.08;  // Layer 2: 8% max weekly loss (5 trading days)
const CB_PEAK_DD_LIMIT = 0.20; // Layer 3: 20% drawdown from all-time peak
const CB_STATE_FILE = path.join(__dirname, "..", "ml_service", "data", "circuit_breaker_state.json");

const PRICE_POLL_MS = 15000;
const TRADE_CYCLE_MS = 60000;

// ── Volatility Targeting ──
const VOL_TARGET = 0.18;    // v10: 18% annualized target (backtested: 1.09 Sharpe)
const MAX_LEVERAGE = 1.5;
const MIN_LEVERAGE = 0.3;
const VOL_LOOKBACK = 40;    // v10: 40-day lookback (matches backtest)

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
  const candidates = [];

  for (const sym of symbols) {
    const prices = priceHist[sym];
    if (!prices || prices.length < MOM.LOOKBACK + 1) continue;
    if (heldSymbols.has(sym)) continue;

    const currentPrice = prices[prices.length - 1];
    if (currentPrice <= 0) continue;

    // 12-1 momentum: skip recent month
    const priceSkip = prices[prices.length - 1 - MOM.SKIP];
    const price12m = prices[prices.length - 1 - MOM.LOOKBACK];
    if (price12m <= 0) continue;
    const mom12_1 = (priceSkip / price12m) - 1.0;

    // SMA200 filter
    const sma200 = sma(prices, MOM.SMA_PERIOD);
    if (sma200 === null || currentPrice <= sma200) continue;

    // 52-week high filter
    if (prices.length >= 252) {
      const high52 = Math.max(...prices.slice(-252));
      if (currentPrice / high52 < MOM.HIGH_52WK_THRESHOLD) continue;
    } else continue;

    // Volume filter
    const vols = volHist[sym];
    const avg = avgVolume(vols, MOM.VOL_PERIOD);
    if (avg !== null && avg < MOM.VOL_MIN) continue;

    // Earnings proximity
    const earningsDate = earningsMap ? earningsMap[sym] : null;
    if (earningsDate) {
      const days = daysUntilEarnings(earningsDate);
      if (days >= 0 && days <= 3) continue;
    }

    // Consistency score: % positive days in last 252 days
    const recentPrices = prices.slice(-252);
    let positiveDays = 0;
    for (let i = 1; i < recentPrices.length; i++) {
      if (recentPrices[i] > recentPrices[i-1]) positiveDays++;
    }
    const pctPositive = positiveDays / (recentPrices.length - 1);
    const consistencyBoost = pctPositive > MOM.CONSISTENCY_THRESHOLD ? MOM.CONSISTENCY_BOOST : 0.0;
    const score = mom12_1 + consistencyBoost;

    // ATR-based position sizing
    const stockAtr = atr(prices, MOM.ATR_PERIOD);
    const atrPct = stockAtr ? stockAtr / currentPrice : 0.02;
    const positionPct = atrPct > MOM.HIGH_ATR_THRESH ? MOM.HIGH_ATR_PCT : MOM.BASE_PCT;

    candidates.push({
      sym, score, price: currentPrice, mom12_1, positionPct,
      consensus: `MOM 12-1 ${(mom12_1 * 100).toFixed(1)}% (cons ${(pctPositive*100).toFixed(0)}%)`,
      rsiVal: null, mlConf: null, strategy: "momentum",
    });
  }

  // Sort by score descending, take top N
  candidates.sort((a, b) => b.score - a.score);

  // Build rankings for exit logic and add rank to each candidate
  const rankings = {};
  candidates.forEach((c, idx) => { rankings[c.sym] = idx + 1; c.rank = idx + 1; });

  return { signals: candidates.slice(0, MOM.TOP_N), rankings };
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
    if (!prices || prices.length < Math.max(MR.SMA_PERIOD, MR.BB_PERIOD, MR.RSI_PERIOD + 1) + 1) continue;
    if (heldSymbols.has(sym)) continue;

    const currentPrice = prices[prices.length - 1];
    if (currentPrice <= 0) continue;

    // 1. Above 200-day SMA (trend filter)
    const sma200 = sma(prices, MR.SMA_PERIOD);
    if (sma200 === null || currentPrice < sma200) continue;

    // 2. RSI < 30
    const rsiVal = rsi(prices, MR.RSI_PERIOD);
    if (rsiVal >= MR.RSI_ENTRY) continue;

    // 3. Below lower Bollinger Band
    const bb = bollinger(prices, MR.BB_PERIOD);
    if (!bb || currentPrice >= bb.lower) continue;

    // 4. Volume spike >= 1.5x 20-day avg
    const vols = volHist[sym];
    if (!vols || vols.length < MR.VOL_PERIOD + 1) continue;
    const avgVol20 = avgVolume(vols, MR.VOL_PERIOD);
    if (avgVol20 === null || avgVol20 < MR.VOL_MIN) continue;
    const currVol = vols[vols.length - 1];
    if (currVol < avgVol20 * MR.VOL_SPIKE) continue;

    // Earnings proximity (7 days for MR)
    const earningsDate = earningsMap ? earningsMap[sym] : null;
    if (earningsDate) {
      const days = daysUntilEarnings(earningsDate);
      if (days >= 0 && days <= 7) continue;
    }

    // ATR for dynamic stops
    const stockAtr = atr(prices, MR.ATR_PERIOD);
    const atrVal = stockAtr || currentPrice * 0.02;
    const atrStop = currentPrice - MR.ATR_STOP_MULT * atrVal;
    const atrTp = currentPrice + MR.ATR_TP_MULT * atrVal;

    const confidence = (MR.RSI_ENTRY - rsiVal) / MR.RSI_ENTRY;
    const atrPct = stockAtr ? stockAtr / currentPrice : 0.02;
    const positionPct = atrPct > MR.HIGH_ATR_THRESH ? MR.HIGH_ATR_PCT : MR.BASE_PCT;

    signals.push({
      sym, score: confidence, price: currentPrice,
      consensus: `MR DUAL RSI=${rsiVal.toFixed(0)} BB<lower VOL=${(currVol/avgVol20).toFixed(1)}x`,
      rsiVal, mlConf: null, strategy: "mean_reversion",
      positionPct, atrStop, atrTp,
    });
  }

  signals.sort((a, b) => b.score - a.score);
  return signals;
}

// ══════════════════════════════════════════
//  MEGA-CAP OVERLAY SIGNAL ENGINE
// ══════════════════════════════════════════

/**
 * Compute mega-cap overlay signals: top 3 of the 15 largest S&P 500 stocks
 * by 60-day momentum. Always active regardless of regime.
 *
 * @param {Object} priceHist   - { symbol → [close1, close2, ...] }
 * @param {Set}    heldSymbols - symbols currently held
 * @returns {Array} mega-cap overlay buy signals
 */
function computeMegaCapSignals(priceHist, heldSymbols) {
  const signals = [];
  const candidates = [];

  for (const sym of MEGACAP.UNIVERSE) {
    const prices = priceHist[sym];
    if (!prices || prices.length < MEGACAP.LOOKBACK + 1) continue;
    if (heldSymbols.has(sym)) continue;

    const currentPrice = prices[prices.length - 1];

    // SMA200 filter
    const sma200 = sma(prices, MEGACAP.SMA_PERIOD);
    if (sma200 === null || currentPrice <= sma200) continue;

    // 12-1 momentum
    const priceSkip = prices[prices.length - 1 - MEGACAP.SKIP];
    const price12m = prices[prices.length - 1 - MEGACAP.LOOKBACK];
    if (price12m <= 0) continue;
    const mom12_1 = (priceSkip / price12m) - 1.0;

    // 52-week high filter (within 5%)
    if (prices.length >= 252) {
      const high52 = Math.max(...prices.slice(-252));
      if (currentPrice / high52 < MEGACAP.HIGH_52WK_THRESHOLD) continue;
    }

    candidates.push({ sym, mom12_1, price: currentPrice });
  }

  // Sort by 12-1 momentum, take top PICKS
  candidates.sort((a, b) => b.mom12_1 - a.mom12_1);
  let signalCount = 0;

  for (const { sym, mom12_1, price } of candidates) {
    if (signalCount >= MEGACAP.PICKS) break;
    const rank = candidates.findIndex(c => c.sym === sym) + 1;
    const confidence = (MEGACAP.UNIVERSE.length - rank + 1) / MEGACAP.UNIVERSE.length;

    signals.push({
      sym, score: confidence, price,
      consensus: `MCAP 12-1 #${rank} (${(mom12_1 * 100).toFixed(1)}%)`,
      rsiVal: null, mlConf: null, strategy: "mega_cap",
      positionPct: MEGACAP.BASE_PCT, rank,
    });
    signalCount++;
  }

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

function fetchShortSignals() {
  return new Promise((resolve) => {
    const req = http.get("http://localhost:5001/short-signals", { timeout: 5000 }, (res) => {
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

module.exports = function createTradingEngine({ alpaca, fetchEarningsFromFMP }) {

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

      // Pre-order validation: check for duplicate pending orders for same symbol+side
      // This is a last-resort guard in case heldSymbols check was bypassed
      if (side === "buy") {
        try {
          await rateLimitWait();
          const openOrders = await alpaca.getOrders({ status: "open", symbols: alpacaSym, limit: 10 });
          const dupBuy = openOrders.find(o => o.side === "buy" && o.symbol === alpacaSym);
          if (dupBuy) {
            addLog(`[pre-order] BLOCKED duplicate buy for ${symbol} — pending order ${dupBuy.id} already exists`, "error");
            throw new Error(`Duplicate buy blocked: pending order ${dupBuy.id} for ${symbol}`);
          }
        } catch (err) {
          if (err.message.includes("Duplicate buy blocked")) throw err;
          // If order check fails, proceed cautiously — Alpaca will reject if truly invalid
        }
      }

      // Use client_order_id for idempotency
      const clientOrderId = `${symbol}_${side}_${Date.now()}`;
      await rateLimitWait();
      return await alpaca.createOrder({
        symbol: alpacaSym, qty, side, type, time_in_force,
        client_order_id: clientOrderId,
      });
    } catch (err) {
      notify.send(`🚨 ORDER REJECTED — ${symbol} ${side} ${qty} shares | Reason: ${err.message}`, { deduplicate: true, immediate: true });
      throw err;
    }
  }

  async function closePosition(symbol) {
    try {
      const alpacaSym = toAlpacaSymbol(symbol);
      await rateLimitWait();
      const result = await alpaca.closePosition(alpacaSym);

      // Journal: record sell order submitted
      try {
        const pos = positionsRaw.find(p => p.symbol === symbol);
        journal.recordOrderSubmitted({
          alpaca_order_id: result?.id || null,
          symbol, side: "sell",
          qty: pos ? pos.qty : 0,
          strategy: positionStrategy[symbol] || "legacy",
          regime,
          intended_price: pos ? pos.current_price : null,
        });
      } catch (_) { /* never crash trading loop */ }

      return result;
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
  let shortSignals = null;
  let shortStatus = "down";
  let cash = 0;
  let portfolioValue = 0;
  let initialPortfolioValue = null;
  let positions = {};
  let positionsRaw = [];
  let marketOpen = false;
  let connected = false;
  let error = null;
  let idleSpyShares = 0;
  let lastIdleSpyActionAt = 0;  // Date.now() timestamp of last idle-SPY action
  let trailingPeaks = {};
  let cooldowns = {};              // strategy-specific: { "sym:strategy" → { cycle, expireTime } }
  let trendPositions = {};
  let trendBreakCounts = {};

  // ── Alert rate limiting (in-memory) ──
  const _alertState = {
    lastAlertAt: 0,
    windowMs: 5 * 60 * 1000,       // 5-minute suppression window
    pendingCount: 0,
    wasDisconnected: false,         // tracks whether we need a recovery message
  };

  function _shouldAlertConnection() {
    const now = Date.now();
    if (now - _alertState.lastAlertAt > _alertState.windowMs) {
      _alertState.lastAlertAt = now;
      _alertState.pendingCount = 1;
      _alertState.wasDisconnected = true;
      return { send: true, count: 1 };
    }
    _alertState.pendingCount++;
    return { send: false, count: _alertState.pendingCount };
  }

  function _checkConnectionRecovery() {
    if (_alertState.wasDisconnected) {
      _alertState.wasDisconnected = false;
      _alertState.pendingCount = 0;
      return true;
    }
    return false;
  }

  // ── Multi-layer circuit breaker state ──
  let circuitBreaker = {
    // Layer 1: Daily
    dailyDate: null,           // YYYY-MM-DD of current tracking day
    marketOpenValue: null,     // portfolio value at market open
    dailyHalted: false,
    // Layer 2: Weekly (rolling 5 trading days)
    weeklyValues: [],          // [{date, openValue}] — last 5 trading days
    weeklyHalted: false,
    weeklyResetDate: null,     // Monday date for auto-reset
    // Layer 3: Peak drawdown (auto-recovery with escalating cooldowns)
    peakValue: 0,
    peakDate: null,
    peakHalted: false,
    peakHaltedAt: null,        // ISO timestamp when halt triggered
    peakHaltCount: 0,          // consecutive halts (escalates cooldown: 1h, 4h, manual)
    // Notification dedup
    _dailyNotified: false,
    _weeklyNotified: false,
    _peakNotified: false,
  };
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

  // ── ML strategy state ──
  let mlEntryDates = {};             // { symbol → cycleNumber at entry }
  let mlTradeCount = { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 };

  // ── Mean Reversion strategy state ──
  let mrEntryPrices = {};          // { symbol → entry price }
  let mrEntryDates = {};           // { symbol → cycleNumber at entry }
  let mrAtrStops = {};             // { symbol → ATR stop price }
  let mrAtrTps = {};               // { symbol → ATR take-profit price }
  let mrTradeCount = { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 };

  // ── Mega-Cap Overlay strategy state ──
  let mcEntryPrices = {};          // { symbol → entry price }
  let mcPeakPrices = {};           // { symbol → peak since entry }
  let mcEntryDates = {};           // { symbol → cycleNumber at entry }
  let mcTradeCount = { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 };
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

  // ── Circuit breaker persistence ──

  function loadCircuitBreakerState() {
    try {
      if (fs.existsSync(CB_STATE_FILE)) {
        const raw = fs.readFileSync(CB_STATE_FILE, "utf8");
        const saved = JSON.parse(raw);
        circuitBreaker.peakValue = saved.peakValue || 0;
        circuitBreaker.peakDate = saved.peakDate || null;
        circuitBreaker.peakHalted = saved.peakHalted || false;
        circuitBreaker.peakHaltedAt = saved.peakHaltedAt || null;
        circuitBreaker.peakHaltCount = saved.peakHaltCount || 0;
        circuitBreaker.weeklyValues = saved.weeklyValues || [];
        addLog(`[circuit-breaker] State loaded: peak=$${circuitBreaker.peakValue.toFixed(0)} (${circuitBreaker.peakDate || "never"})${circuitBreaker.peakHalted ? " | PEAK HALT ACTIVE" : ""}`, "system");
      }
    } catch (err) {
      addLog(`[circuit-breaker] Failed to load state: ${err.message}`, "error");
    }
  }

  function saveCircuitBreakerState() {
    try {
      const state = {
        peakValue: circuitBreaker.peakValue,
        peakDate: circuitBreaker.peakDate,
        peakHalted: circuitBreaker.peakHalted,
        peakHaltedAt: circuitBreaker.peakHaltedAt || null,
        peakHaltCount: circuitBreaker.peakHaltCount || 0,
        weeklyValues: circuitBreaker.weeklyValues,
        savedAt: new Date().toISOString(),
      };
      // Atomic write: write to temp file, then rename
      const tmpFile = CB_STATE_FILE + ".tmp";
      fs.writeFileSync(tmpFile, JSON.stringify(state, null, 2));
      fs.renameSync(tmpFile, CB_STATE_FILE);
    } catch (err) {
      addLog(`[circuit-breaker] Failed to save state: ${err.message}`, "error");
    }
  }

  /**
   * Check all 3 circuit breaker layers.
   * @param {number} currentValue — current portfolio value
   * @returns {{safe: boolean, reason: string, layer: string, details: object}}
   */
  function checkCircuitBreakers(currentValue) {
    const todayDate = new Date().toISOString().split("T")[0];
    const etNow = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
    const dayOfWeek = etNow.getDay(); // 0=Sun, 1=Mon

    // ── New day detection & daily reset ──
    if (circuitBreaker.dailyDate !== todayDate) {
      if (circuitBreaker.dailyHalted) {
        addLog(`[circuit-breaker] Daily halt auto-reset at market open (new day: ${todayDate})`, "system");
        notify.send("✅ CIRCUIT BREAKER — Daily halt reset. New trading day.", { immediate: true });
      }
      circuitBreaker.dailyDate = todayDate;
      circuitBreaker.marketOpenValue = currentValue;
      circuitBreaker.dailyHalted = false;
      circuitBreaker._dailyNotified = false;

      // Track opening value for weekly rolling window
      circuitBreaker.weeklyValues.push({ date: todayDate, openValue: currentValue });
      if (circuitBreaker.weeklyValues.length > 5) {
        circuitBreaker.weeklyValues = circuitBreaker.weeklyValues.slice(-5);
      }
    }

    // ── Monday auto-reset for weekly halt ──
    if (dayOfWeek === 1 && circuitBreaker.weeklyResetDate !== todayDate) {
      if (circuitBreaker.weeklyHalted) {
        addLog(`[circuit-breaker] Weekly halt auto-reset at Monday open (${todayDate})`, "system");
        notify.send("✅ CIRCUIT BREAKER — Weekly halt reset. New trading week.", { immediate: true });
      }
      circuitBreaker.weeklyHalted = false;
      circuitBreaker._weeklyNotified = false;
      circuitBreaker.weeklyResetDate = todayDate;
    }

    if (!circuitBreaker.marketOpenValue) {
      circuitBreaker.marketOpenValue = currentValue;
    }

    // ── Update peak value ──
    if (currentValue > circuitBreaker.peakValue) {
      circuitBreaker.peakValue = currentValue;
      circuitBreaker.peakDate = todayDate;
    }

    // ── Layer 3: Peak drawdown (check first — most severe) ──
    // Auto-recovery with escalating cooldowns:
    //   1st halt → 1 hour cooldown, then auto-reset
    //   2nd halt → 4 hours cooldown, then auto-reset
    //   3rd+ halt → manual reset required (persistent problem)
    if (circuitBreaker.peakHalted) {
      const haltCount = circuitBreaker.peakHaltCount || 1;
      const haltedAt = circuitBreaker.peakHaltedAt ? new Date(circuitBreaker.peakHaltedAt).getTime() : 0;
      const elapsedMs = Date.now() - haltedAt;
      const cooldownMs = haltCount === 1 ? 60 * 60 * 1000    // 1 hour
                       : haltCount === 2 ? 4 * 60 * 60 * 1000 // 4 hours
                       : Infinity;                              // manual only
      const cooldownLabel = haltCount === 1 ? "1h" : haltCount === 2 ? "4h" : "manual";

      if (haltedAt > 0 && elapsedMs >= cooldownMs && cooldownMs < Infinity) {
        // Auto-recovery: cooldown elapsed
        circuitBreaker.peakHalted = false;
        circuitBreaker._peakNotified = false;
        circuitBreaker.peakValue = currentValue;
        circuitBreaker.peakDate = todayDate;
        saveCircuitBreakerState();
        addLog(`[circuit-breaker] Peak halt AUTO-RESET after ${cooldownLabel} cooldown (halt #${haltCount}). New peak $${currentValue.toFixed(0)}.`, "system");
        notify.send(`✅ CIRCUIT BREAKER — Peak drawdown halt auto-reset after ${cooldownLabel} cooldown (halt #${haltCount}). New peak: $${currentValue.toFixed(0)}. Next halt will use ${haltCount >= 2 ? "manual reset" : "4h cooldown"}.`, { immediate: true });
      } else {
        const remaining = cooldownMs < Infinity ? Math.ceil((cooldownMs - elapsedMs) / 60000) : 0;
        const resetMsg = cooldownMs < Infinity ? `Auto-reset in ${remaining}min (${cooldownLabel} cooldown, halt #${haltCount}).` : `Manual reset required (halt #${haltCount}, 3+ consecutive).`;
        return {
          safe: false,
          reason: `Peak drawdown halt active (peak $${circuitBreaker.peakValue.toFixed(0)} on ${circuitBreaker.peakDate}, current $${currentValue.toFixed(0)}, DD ${((1 - currentValue / circuitBreaker.peakValue) * 100).toFixed(1)}%). ${resetMsg}`,
          layer: "peak",
          details: getCircuitBreakerDetails(currentValue),
        };
      }
    }

    const peakDD = circuitBreaker.peakValue > 0 ? 1 - currentValue / circuitBreaker.peakValue : 0;
    if (peakDD >= CB_PEAK_DD_LIMIT) {
      circuitBreaker.peakHalted = true;
      circuitBreaker.peakHaltedAt = new Date().toISOString();
      circuitBreaker.peakHaltCount = (circuitBreaker.peakHaltCount || 0) + 1;
      const haltNum = circuitBreaker.peakHaltCount;
      const ddPct = (peakDD * 100).toFixed(1);
      const recoveryMsg = haltNum === 1 ? "Auto-reset in 1h." : haltNum === 2 ? "Auto-reset in 4h." : "Manual reset required (3+ consecutive halts).";
      addLog(`[circuit-breaker] *** LAYER 3: PEAK DRAWDOWN ${ddPct}% *** Peak $${circuitBreaker.peakValue.toFixed(0)} → $${currentValue.toFixed(0)}. Halt #${haltNum}. ${recoveryMsg}`, "error");
      notify.send(`🚨🚨 CIRCUIT BREAKER LAYER 3: PEAK DRAWDOWN ${ddPct}% — Peak $${circuitBreaker.peakValue.toFixed(0)} → Current $${currentValue.toFixed(0)}. Halt #${haltNum}. ${recoveryMsg}`, { deduplicate: false, immediate: true });
      try { journal.logEvent({ event_type: "circuit_breaker", severity: "critical", message: `Peak DD ${ddPct}%`, metadata: { layer: "peak", dd_pct: peakDD, peak: circuitBreaker.peakValue, current: currentValue }, portfolio_value: currentValue }); } catch (_) {}
      saveCircuitBreakerState();
      return {
        safe: false,
        reason: `Peak drawdown ${ddPct}% exceeds ${CB_PEAK_DD_LIMIT * 100}% limit`,
        layer: "peak",
        details: getCircuitBreakerDetails(currentValue),
      };
    }

    // ── Layer 2: Weekly loss ──
    if (circuitBreaker.weeklyHalted) {
      return {
        safe: false,
        reason: "Weekly loss halt active. Auto-resets Monday at market open.",
        layer: "weekly",
        details: getCircuitBreakerDetails(currentValue),
      };
    }

    if (circuitBreaker.weeklyValues.length > 0) {
      const weekStart = circuitBreaker.weeklyValues[0].openValue;
      const weeklyLoss = weekStart > 0 ? (weekStart - currentValue) / weekStart : 0;
      if (weeklyLoss >= CB_WEEKLY_LIMIT) {
        circuitBreaker.weeklyHalted = true;
        const lossPct = (weeklyLoss * 100).toFixed(1);
        addLog(`[circuit-breaker] ** LAYER 2: WEEKLY LOSS ${lossPct}% ** Week start $${weekStart.toFixed(0)} → $${currentValue.toFixed(0)}. Halted until Monday.`, "error");
        notify.send(`🚨 CIRCUIT BREAKER LAYER 2: WEEKLY LOSS ${lossPct}% — $${weekStart.toFixed(0)} → $${currentValue.toFixed(0)}. New trades halted until Monday.`, { deduplicate: true, immediate: true });
        try { journal.logEvent({ event_type: "circuit_breaker", severity: "error", message: `Weekly loss ${lossPct}%`, metadata: { layer: "weekly", loss_pct: weeklyLoss, weekStart, current: currentValue }, portfolio_value: currentValue }); } catch (_) {}
        saveCircuitBreakerState();
        return {
          safe: false,
          reason: `Weekly loss ${lossPct}% exceeds ${CB_WEEKLY_LIMIT * 100}% limit`,
          layer: "weekly",
          details: getCircuitBreakerDetails(currentValue),
        };
      }
    }

    // ── Layer 1: Daily loss ──
    if (circuitBreaker.dailyHalted) {
      return {
        safe: false,
        reason: "Daily loss halt active. Auto-resets tomorrow at market open.",
        layer: "daily",
        details: getCircuitBreakerDetails(currentValue),
      };
    }

    const dailyLoss = circuitBreaker.marketOpenValue > 0
      ? (circuitBreaker.marketOpenValue - currentValue) / circuitBreaker.marketOpenValue : 0;
    if (dailyLoss >= CB_DAILY_LIMIT) {
      circuitBreaker.dailyHalted = true;
      const lossPct = (dailyLoss * 100).toFixed(1);
      addLog(`[circuit-breaker] * LAYER 1: DAILY LOSS ${lossPct}% * Open $${circuitBreaker.marketOpenValue.toFixed(0)} → $${currentValue.toFixed(0)}. Halted until tomorrow.`, "error");
      if (!circuitBreaker._dailyNotified) {
        notify.send(`🚨 CIRCUIT BREAKER LAYER 1: DAILY LOSS ${lossPct}% — $${circuitBreaker.marketOpenValue.toFixed(0)} → $${currentValue.toFixed(0)}. New trades halted until tomorrow.`, { deduplicate: true, immediate: true });
        try { journal.logEvent({ event_type: "circuit_breaker", severity: "warning", message: `Daily loss ${lossPct}%`, metadata: { layer: "daily", loss_pct: dailyLoss, openValue: circuitBreaker.marketOpenValue, current: currentValue }, portfolio_value: currentValue }); } catch (_) {}
        circuitBreaker._dailyNotified = true;
      }
      saveCircuitBreakerState();
      return {
        safe: false,
        reason: `Daily loss ${lossPct}% exceeds ${CB_DAILY_LIMIT * 100}% limit`,
        layer: "daily",
        details: getCircuitBreakerDetails(currentValue),
      };
    }

    // All clear
    return { safe: true, reason: "All layers OK", layer: "none", details: getCircuitBreakerDetails(currentValue) };
  }

  function getCircuitBreakerDetails(currentValue) {
    const dailyLoss = circuitBreaker.marketOpenValue > 0
      ? (circuitBreaker.marketOpenValue - currentValue) / circuitBreaker.marketOpenValue : 0;
    const weekStart = circuitBreaker.weeklyValues.length > 0
      ? circuitBreaker.weeklyValues[0].openValue : currentValue;
    const weeklyLoss = weekStart > 0 ? (weekStart - currentValue) / weekStart : 0;
    const peakDD = circuitBreaker.peakValue > 0 ? 1 - currentValue / circuitBreaker.peakValue : 0;

    return {
      daily: {
        limit: CB_DAILY_LIMIT,
        current: dailyLoss,
        halted: circuitBreaker.dailyHalted,
        openValue: circuitBreaker.marketOpenValue,
        currentValue,
        pct: `${(dailyLoss * 100).toFixed(2)}%`,
      },
      weekly: {
        limit: CB_WEEKLY_LIMIT,
        current: weeklyLoss,
        halted: circuitBreaker.weeklyHalted,
        weekStartValue: weekStart,
        currentValue,
        tradingDays: circuitBreaker.weeklyValues.length,
        pct: `${(weeklyLoss * 100).toFixed(2)}%`,
      },
      peak: {
        limit: CB_PEAK_DD_LIMIT,
        current: peakDD,
        halted: circuitBreaker.peakHalted,
        peakValue: circuitBreaker.peakValue,
        peakDate: circuitBreaker.peakDate,
        currentValue,
        pct: `${(peakDD * 100).toFixed(2)}%`,
      },
    };
  }

  // ══════════════════════════════════════════
  //  INIT — Load account, positions, price history
  // ══════════════════════════════════════════

  async function init() {
    try {
      addLog("Initializing trading engine...", "system");

      // 0. Initialize trade journal DB
      try {
        const journalDbPath = path.join(__dirname, "..", "data", "journal.db");
        journal.initDb(journalDbPath);
        addLog("Trade journal DB initialized", "system");
        journal.logEvent({ event_type: "service_restart", severity: "info", message: "Trading engine starting" });
      } catch (jErr) {
        addLog(`WARNING: Journal DB init failed: ${jErr.message}`, "error");
      }

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

      // 2b. Rehydrate strategy tags from journal
      for (const p of positionsRaw) {
        if (p.symbol === "SPY") continue;
        try {
          const entry = journal.findLatestEntryTrade(p.symbol);
          if (entry && entry.strategy) {
            positionStrategy[p.symbol] = entry.strategy;
            // Restore trendPositions map so exit logic (trailing stop, SMA-break) works
            if (entry.strategy === "trend") {
              trendPositions[p.symbol] = {
                entryPrice: entry.fill_price || p.avg_entry_price,
                peakPrice: Math.max(entry.fill_price || 0, p.current_price),
              };
              trendBreakCounts[p.symbol] = 0;
            }
            // Restore ML entry date so minimum hold period works across restarts
            if (entry.strategy === "ml" && entry.submitted_at) {
              const entryDate = new Date(entry.submitted_at);
              const now = new Date();
              const calendarDays = (now - entryDate) / (1000 * 60 * 60 * 24);
              // Approximate trading days ≈ calendar days × 5/7 (weekdays only)
              const tradingDays = calendarDays * (5 / 7);
              const estimatedCyclesHeld = Math.round(tradingDays * 390);
              mlEntryDates[p.symbol] = cycleNumber - estimatedCyclesHeld;
              addLog(`[rehydrate] ${p.symbol} ML entry date restored: ~${tradingDays.toFixed(1)} trading days ago`, "system");
            }
            addLog(`[rehydrate] ${p.symbol} tagged as ${entry.strategy} from journal (bought ${entry.submitted_at})`, "system");
          } else {
            // Tag GLD/VIXM as hedge even without journal entry
            const hedgeSymbols = new Set(["GLD", "VIXM"]);
            if (hedgeSymbols.has(p.symbol)) {
              positionStrategy[p.symbol] = "hedge";
              addLog(`[rehydrate] ${p.symbol} tagged as hedge (known hedge symbol)`, "system");
            } else {
              // Tag as "ml" (not "legacy") so it counts toward ML slots correctly
              positionStrategy[p.symbol] = "ml";
              addLog(`[rehydrate] ${p.symbol} has no journal entry — tagged ml`, "system");
            }
          }
        } catch (err) {
          positionStrategy[p.symbol] = "legacy";
          addLog(`[rehydrate] ${p.symbol} journal lookup failed: ${err.message} — tagged legacy`, "error");
        }
      }

      // 3. Restore idle SPY tracking
      if (ENABLE_SPY_PARKING) {
        const spyPos = positionsRaw.find(p => p.symbol === "SPY");
        if (spyPos) {
          idleSpyShares = spyPos.qty;
          addLog(`Restored idle SPY tracking: ${idleSpyShares} shares`, "system");
        }
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
      // Load persisted circuit breaker state (peak value, weekly values)
      loadCircuitBreakerState();
      // Initialize peak with current portfolio value if not set
      if (circuitBreaker.peakValue === 0) {
        const acctInit = await getAccount();
        circuitBreaker.peakValue = acctInit.portfolio_value;
        circuitBreaker.peakDate = new Date().toISOString().split("T")[0];
        saveCircuitBreakerState();
        addLog(`[circuit-breaker] Peak initialized: $${circuitBreaker.peakValue.toFixed(0)}`, "system");
      }

      addLog("Trading engine initialized successfully", "system");
    } catch (err) {
      error = err.message;
      connected = false;
      addLog(`Initialization failed: ${err.message}`, "error");
      const alert = _shouldAlertConnection();
      if (alert.send) {
        notify.send(`⚠️ Alpaca connection issue on startup — retrying. Error: ${err.message}`, { deduplicate: true, immediate: true });
      }
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
      // BUG FIX: Only append snapshot price once per calendar day to avoid
      // polluting daily bar history with repeated intraday prices (every 15s).
      // Multiple intraday appends distort SMA50/SMA200 and cause false regime flips.
      const spyPrev = priceHist.SPY || [];
      const spySnap = snapshots.SPY;
      if (spySnap && spySnap.price > 0) {
        const todayStr = new Date().toISOString().split("T")[0];
        if (!priceHist._spyLastAppendDate || priceHist._spyLastAppendDate !== todayStr) {
          // First poll of the day: append new price as today's bar
          nextHist.SPY = [...spyPrev.slice(-250), spySnap.price];
          nextHist._spyLastAppendDate = todayStr;
        } else {
          // Subsequent polls: update today's price in-place (last element)
          nextHist.SPY = spyPrev.length > 0
            ? [...spyPrev.slice(0, -1), spySnap.price]
            : [spySnap.price];
          nextHist._spyLastAppendDate = todayStr;
        }
      } else {
        nextHist.SPY = spyPrev;
        nextHist._spyLastAppendDate = priceHist._spyLastAppendDate;
      }

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
        // Tag untracked positions as "legacy" (pre-existing or from previous session)
        if (!positionStrategy[p.symbol] && p.symbol !== "SPY") {
          positionStrategy[p.symbol] = "legacy";
          addLog(`[reconcile] Position ${p.symbol} (${p.qty} shares) has no strategy tag — marked as 'legacy'`, "system");
        }
      }

      // Journal: sync positions and prices every poll cycle
      try {
        const priceMap = {};
        for (const p of positionsRaw) {
          if (p.symbol === "SPY") continue;
          priceMap[p.symbol] = p.current_price;
          journal.upsertPosition({
            symbol: p.symbol,
            qty: p.qty,
            avg_cost: p.avg_entry_price,
            current_price: p.current_price,
            unrealized_pnl: p.unrealized_pl,
            unrealized_pnl_pct: p.unrealized_plpc,
            strategy: positionStrategy[p.symbol] || null,
          });
        }
        if (Object.keys(priceMap).length > 0) journal.updatePositionPrices(priceMap);
        // Remove journal positions that Alpaca no longer holds (sold/closed externally)
        journal.removeStalePositions(Object.keys(priceMap));
      } catch (_) { /* never crash trading loop */ }

      // Portfolio history (keep last 200)
      portfolioHist.push({ tick, value: portfolioValue, time: Date.now() });
      if (portfolioHist.length > 200) portfolioHist.splice(0, portfolioHist.length - 200);

      // Fetch ML signals — server returns { signals: [...], is_stale: bool }
      const mlData = await fetchMLSignals();
      if (mlData && Array.isArray(mlData.signals) && !mlData.is_stale) {
        mlSignals = mlData.signals;
        mlStatus = "ok";
        // Journal: record each ML signal
        // Signal recording disabled — was writing 1506 rows every 10s,
        // bloating journal.db to 2GB+ daily. Signals are already in signal_cache_v9.json.
      } else if (mlData && Array.isArray(mlData.signals) && mlData.is_stale) {
        mlSignals = null;   // stale → don't use for trading decisions
        mlStatus = "stale";
      } else {
        mlSignals = null;
        mlStatus = "down";
      }

      // ML status transition notifications
      if (prevMlStatus === "ok" && mlStatus !== "ok") {
        notify.send("🚨 SIGNAL SERVER DOWN — falling back to consensus engine. Check pm2 logs.", { deduplicate: true, immediate: true });
      } else if (prevMlStatus !== "ok" && mlStatus === "ok") {
        notify.send("✅ SIGNAL SERVER RECOVERED — v10 signals active again.", { immediate: true });
      }
      prevMlStatus = mlStatus;

      // Fetch Short Sleeve signals
      const shortData = await fetchShortSignals();
      if (shortData && Array.isArray(shortData.signals) && !shortData.is_stale) {
        shortSignals = shortData.signals;
        shortStatus = "ok";
        const shortCount = shortSignals.filter(s => s.signal === "SHORT").length;
        const coverCount = shortSignals.filter(s => s.signal === "COVER").length;
        if (shortCount > 0 || coverCount > 0) {
          addLog(`Short sleeve active -- ${shortCount} SHORT, ${coverCount} COVER signals`);
        }
      } else {
        shortSignals = null;
        shortStatus = shortData ? "stale" : "down";
      }

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

      // Journal: log regime changes
      if (regime !== prevRegime) {
        try {
          journal.logEvent({
            event_type: "regime_change", severity: "info",
            message: `${prevRegime} -> ${regime}`,
            metadata: { from: prevRegime, to: regime, sma50: regimeResult.sma50, sma200: regimeResult.sma200 },
            portfolio_value: portfolioValue,
          });
        } catch (_) {}
      }

      // Regime diagnostic: log SPY price vs SMAs every cycle for debugging
      if (regimeResult.sma50 !== null && regimeResult.sma200 !== null) {
        const spyNow = priceHist.SPY?.[priceHist.SPY.length - 1];
        addLog(`[regime] SPY $${spyNow?.toFixed(2)} | SMA50 $${regimeResult.sma50.toFixed(2)} | SMA200 $${regimeResult.sma200.toFixed(2)} | bars: ${priceHist.SPY?.length || 0} | result: ${regimeResult.regime} | applied: ${regime}`, "system");
      }

      // Market open transition notification
      if (marketOpen && !prevMarketOpen) {
        const buyCount = mlSignals ? mlSignals.filter(s => s.signal === "BUY").length : 0;
        notify.send(`🔔 MARKET OPEN — Bot is trading. Regime: ${regime}. v10 signals: ${buyCount} BUY.`);
      }
      prevMarketOpen = marketOpen;

      connected = true;
      error = null;
      // Send recovery notification if we were previously disconnected
      if (_checkConnectionRecovery()) {
        addLog("[connection] Alpaca connection restored", "system");
        notify.send(`✅ Alpaca connection restored — trading resumed normally.`, { immediate: true });
      }
    } catch (err) {
      addLog(`Price poll error: ${err.message}`, "error");
      const alert = _shouldAlertConnection();
      if (alert.send) {
        notify.send(`⚠️ Alpaca connection issue — this cycle skipped, retrying. Error: ${err.message}`, { immediate: true });
      } else {
        addLog(`[connection] Suppressed alert (${alert.count} errors in 5-min window)`, "system");
      }
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
      // Uses BOTH cycle count AND wall-clock minimum (30 min) to prevent churn
      const isOnCooldown = (sym, strategy = "ml") => {
        const key = `${sym}:${strategy}`;
        const cd = cooldowns[key];
        if (!cd) return false;
        const cooldownLen = strategy === "momentum" ? MOM.COOLDOWN_CYCLES
          : strategy === "mean_reversion" ? MR.COOLDOWN_CYCLES
          : strategy === "mega_cap" ? MEGACAP.COOLDOWN_CYCLES
          : RISK.LOSS_COOLDOWN_CYCLES;
        const cycleActive = (cycleNumber - cd.cycle) < cooldownLen;
        const timeActive = Date.now() < cd.expireTime;
        return cycleActive || timeActive;
      };
      const setCooldown = (sym, strategy = "ml") => {
        const expireTime = Date.now() + RISK.MIN_COOLDOWN_MS;
        cooldowns[`${sym}:${strategy}`] = { cycle: cycleNumber, expireTime };
        addLog(`[cooldown] SET ${sym}:${strategy} until ${new Date(expireTime).toLocaleTimeString()}`, "system");
      };

      // Count positions by strategy for slot allocation
      // Accounts for in-flight orders: pending SELLs reduce count, pending BUYs increase count
      // "legacy" positions (pre-existing, untagged) are tracked but excluded from slot limits
      const countByStrategy = (positionsOverride) => {
        const posToCount = positionsOverride || currentPositions;
        // Build sets of symbols with pending sells/buys
        const pendingSells = new Set();
        const pendingBuySyms = new Map(); // symbol -> strategy
        for (const o of pendingOrders) {
          const sym = fromAlpacaSymbol(o.symbol);
          if (o.side === "sell") {
            pendingSells.add(sym);
          } else if (o.side === "buy") {
            // Only count pending buys for symbols we don't already hold
            const alreadyHeld = posToCount.some(p => p.symbol === sym);
            if (!alreadyHeld) {
              pendingBuySyms.set(sym, positionStrategy[sym] || "ml");
            }
          }
        }

        let ml = 0, mom = 0, mr = 0, mc = 0, trend = 0, legacy = 0;
        for (const pos of posToCount) {
          if (pos.symbol === "SPY" && idleSpyShares > 0) continue;
          // Skip hedge positions (GLD, VIXM) — managed separately, don't count toward slot limit
          if (positionStrategy[pos.symbol] === "hedge") continue;
          // Skip positions with pending sell — they're on the way out
          if (pendingSells.has(pos.symbol)) continue;
          const strat = positionStrategy[pos.symbol] || "legacy";
          if (strat === "legacy") legacy++;
          else if (strat === "momentum") mom++;
          else if (strat === "mean_reversion") mr++;
          else if (strat === "mega_cap") mc++;
          else if (strat === "trend" || trendPositions[pos.symbol]) trend++;
          else ml++;
        }

        // Add pending buys (positions on the way in)
        for (const [, strat] of pendingBuySyms) {
          if (strat === "momentum") mom++;
          else if (strat === "mean_reversion") mr++;
          else if (strat === "mega_cap") mc++;
          else if (strat === "trend") trend++;
          else ml++;
        }

        return { ml, mom, mr, mc, trend, legacy, total: ml + mom + mr + mc + trend + legacy };
      };

      // Check if a strategy has slot capacity
      const hasSlotCapacity = (strategy, counts) => {
        if (counts.total >= SLOT_CONFIG.max) return false;
        // Trend has no dedicated slot — competes for flex only
        if (strategy === "trend") {
          const flexUsed = counts.total
            - Math.min(counts.ml, SLOT_CONFIG.ml_medium)
            - Math.min(counts.mom, SLOT_CONFIG.momentum)
            - Math.min(counts.mr, SLOT_CONFIG.mean_reversion)
            - Math.min(counts.mc, SLOT_CONFIG.mega_cap);
          return flexUsed < SLOT_CONFIG.flex;
        }
        const stratMap = { momentum: "momentum", mean_reversion: "mean_reversion", ml: "ml_medium", mega_cap: "mega_cap" };
        const countMap = { momentum: counts.mom, mean_reversion: counts.mr, ml: counts.ml, mega_cap: counts.mc };
        const primaryKey = stratMap[strategy] || "ml_medium";
        const primaryUsed = countMap[strategy] || counts.ml;
        const primaryLimit = SLOT_CONFIG[primaryKey] || 0;
        // Can use primary slot
        if (primaryUsed < primaryLimit) return true;
        // Or flex slot if available
        const flexUsed = counts.total
          - Math.min(counts.ml, SLOT_CONFIG.ml_medium)
          - Math.min(counts.mom, SLOT_CONFIG.momentum)
          - Math.min(counts.mr, SLOT_CONFIG.mean_reversion)
          - Math.min(counts.mc, SLOT_CONFIG.mega_cap);
        return flexUsed < SLOT_CONFIG.flex;
      };

      // Fetch live state from Alpaca
      const [account, currentPositions, clock, pendingOrders] = await Promise.all([
        getAccount(),
        getPositions(),
        getClock(),
        getOrders("open", 50).catch(() => []),
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
      let cyclePortfolioValue = account.portfolio_value;

      // Multi-layer circuit breaker (daily 4%, weekly 8%, peak 20%)
      const cbResult = checkCircuitBreakers(cyclePortfolioValue);
      if (!cbResult.safe) {
        skipNewBuys = true;
        addLog(`[circuit-breaker] ${cbResult.layer.toUpperCase()} HALT: ${cbResult.reason}`, "error");
      }

      // Reset daily stats on new day
      const todayDate = new Date().toISOString().split("T")[0];
      if (dailyStats.date !== todayDate) {
        const dayOfWeek = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" })).getDay();
        dailyStats = {
          date: todayDate, buys: 0, sells: 0, wins: 0, losses: 0,
          summarySent: false,
          weekStartValue: (dayOfWeek === 1 || dailyStats.weekStartValue === null)
            ? cyclePortfolioValue : dailyStats.weekStartValue,
        };
      }

      // Persist circuit breaker state each cycle (peak tracking)
      saveCircuitBreakerState();

      // ── Volatility targeting: track daily returns and update scale ──
      if (previousDayValue !== null && previousDayValue > 0) {
        const dailyRet = (cyclePortfolioValue - previousDayValue) / previousDayValue;
        dailyReturns.push(dailyRet);
        if (dailyReturns.length > 60) dailyReturns.splice(0, dailyReturns.length - 60);
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
        if (positionStrategy[symbol] === "hedge") continue;  // v10: GLD/VIXM managed in STEP 1g

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
            const strat = positionStrategy[symbol] || "legacy";
            await closePosition(symbol);
            delete trailingPeaks[symbol];
            closedSymbols.add(symbol);
            setCooldown(symbol, strat);
            // Clean up strategy-specific state
            if (strat === "momentum") {
              delete momEntryPrices[symbol];
              delete momPeakPrices[symbol];
              delete momEntryDates[symbol];
            } else if (strat === "mean_reversion") {
              delete mrEntryPrices[symbol];
              delete mrEntryDates[symbol];
            } else if (strat === "mega_cap") {
              delete mcEntryPrices[symbol];
              delete mcPeakPrices[symbol];
              delete mcEntryDates[symbol];
            } else if (strat === "ml") {
              delete mlEntryDates[symbol];
            }
            delete positionStrategy[symbol];
            addLog(stopMsg, "sell");
            const stratTracker = strat === "momentum" ? momTradeCount : strat === "mean_reversion" ? mrTradeCount : strat === "mega_cap" ? mcTradeCount : mlTradeCount;
            tradeCount.sells++; stratTracker.sells++;
            if (unrealized_pl >= 0) { tradeCount.wins++; stratTracker.wins++; }
            else { tradeCount.losses++; stratTracker.losses++; }
            tradeCount.totalPnL += unrealized_pl;
            stratTracker.totalPnL += unrealized_pl;
            const exitReason = RISK.USE_TRAILING_STOP ? "trail-stop" : "stop-loss";
            try { journal.closePosition({ symbol, fillPrice: curr, exitReason }); } catch (_) {}
            dailyStats.sells++;
            if (unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            if (RISK.USE_TRAILING_STOP) {
              const peak = trailingPeaks[symbol] || curr;
              const drop = ((curr - peak) / peak * 100).toFixed(1);
              notify.send(`🛑 TRAIL-STOP ${symbol} | ${qty} shares @ $${curr.toFixed(2)} | Peak $${peak.toFixed(2)}, drop ${drop}% | P&L: $${unrealized_pl.toFixed(2)}`);
            } else {
              notify.send(`🛑 STOP-LOSS ${symbol} | ${qty} shares @ $${curr.toFixed(2)} | Loss: $${unrealized_pl.toFixed(2)} (${(unrealized_plpc * 100).toFixed(1)}%)`);
            }
          } catch (err) {
            addLog(`Failed to close ${symbol}: ${err.message}`, "error");
          }
        } else if (unrealized_plpc >= RISK.TAKE_PROFIT_PCT) {
          try {
            const strat = positionStrategy[symbol] || "legacy";
            await closePosition(symbol);
            delete trailingPeaks[symbol];
            closedSymbols.add(symbol);
            setCooldown(symbol, strat);
            if (strat === "momentum") {
              delete momEntryPrices[symbol];
              delete momPeakPrices[symbol];
              delete momEntryDates[symbol];
            } else if (strat === "mean_reversion") {
              delete mrEntryPrices[symbol];
              delete mrEntryDates[symbol];
            } else if (strat === "mega_cap") {
              delete mcEntryPrices[symbol];
              delete mcPeakPrices[symbol];
              delete mcEntryDates[symbol];
            } else if (strat === "ml") {
              delete mlEntryDates[symbol];
            }
            delete positionStrategy[symbol];
            addLog(`TAKE-PROFIT ${symbol}: ${qty} shares @ $${curr.toFixed(2)} | P&L: +$${unrealized_pl.toFixed(2)}`, "profit");
            const stratTracker = strat === "momentum" ? momTradeCount : strat === "mean_reversion" ? mrTradeCount : strat === "mega_cap" ? mcTradeCount : mlTradeCount;
            tradeCount.sells++; stratTracker.sells++;
            tradeCount.wins++; stratTracker.wins++;
            tradeCount.totalPnL += unrealized_pl;
            stratTracker.totalPnL += unrealized_pl;
            try { journal.closePosition({ symbol, fillPrice: curr, exitReason: "take-profit" }); } catch (_) {}
            dailyStats.sells++;
            dailyStats.wins++;
            notify.send(`🎯 TAKE-PROFIT ${symbol} | ${qty} shares @ $${curr.toFixed(2)} | Gain: +$${unrealized_pl.toFixed(2)} (+${(unrealized_plpc * 100).toFixed(1)}%)`);
          } catch (err) {
            addLog(`Failed to close ${symbol}: ${err.message}`, "error");
          }
        }
      }

      // ── STEP 1b: Earnings-eve exits (skip for ML/v9.6 positions — strategy holds through earnings) ──
      for (const pos of currentPositions) {
        if (closedSymbols.has(pos.symbol)) continue;
        if (pos.symbol === "SPY" && idleSpyShares > 0) continue;
        if (positionStrategy[pos.symbol] === "ml") continue;  // v9.6 holds through earnings
        const earningsDate = earningsMap[pos.symbol];
        if (!earningsDate) continue;
        const days = daysUntilEarnings(earningsDate);
        if (days === 1) {
          try {
            const strat = positionStrategy[pos.symbol] || "legacy";
            await closePosition(pos.symbol);
            delete trailingPeaks[pos.symbol];
            closedSymbols.add(pos.symbol);
            setCooldown(pos.symbol, strat);
            if (strat === "momentum") {
              delete momEntryPrices[pos.symbol];
              delete momPeakPrices[pos.symbol];
              delete momEntryDates[pos.symbol];
            } else if (strat === "mean_reversion") {
              delete mrEntryPrices[pos.symbol];
              delete mrEntryDates[pos.symbol];
            } else if (strat === "ml") {
              delete mlEntryDates[pos.symbol];
            }
            delete positionStrategy[pos.symbol];
            addLog(`EARNINGS SELL ${pos.symbol}: earnings tomorrow (${earningsDate}) -- exiting to avoid overnight announcement risk`, "sell");
            const stratTracker = strat === "momentum" ? momTradeCount : strat === "mean_reversion" ? mrTradeCount : strat === "mega_cap" ? mcTradeCount : mlTradeCount;
            tradeCount.sells++; stratTracker.sells++;
            if (pos.unrealized_pl >= 0) { tradeCount.wins++; stratTracker.wins++; }
            else { tradeCount.losses++; stratTracker.losses++; }
            tradeCount.totalPnL += pos.unrealized_pl;
            stratTracker.totalPnL += pos.unrealized_pl;
            try { journal.closePosition({ symbol: pos.symbol, fillPrice: pos.current_price, exitReason: "earnings-sell" }); } catch (_) {}
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
            setCooldown(sym, "momentum");
            addLog(exitMsg, unrealized_pl >= 0 ? "profit" : "sell");
            tradeCount.sells++;
            momTradeCount.sells++;
            if (unrealized_pl >= 0) { tradeCount.wins++; momTradeCount.wins++; }
            else { tradeCount.losses++; momTradeCount.losses++; }
            tradeCount.totalPnL += unrealized_pl;
            momTradeCount.totalPnL += unrealized_pl;
            try { journal.closePosition({ symbol: sym, fillPrice: curr, exitReason }); } catch (_) {}
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

        // 1. ATR-based stop
        if (curr <= (mrAtrStops[sym] || 0)) {
          exitTriggered = true;
          exitReason = "mr-atr-stop";
          exitMsg = `MR ATR-STOP ${sym}: price $${curr.toFixed(2)} <= stop $${(mrAtrStops[sym] || 0).toFixed(2)}`;
        }

        // 2. ATR-based take-profit
        if (!exitTriggered && curr >= (mrAtrTps[sym] || Infinity)) {
          exitTriggered = true;
          exitReason = "mr-atr-tp";
          exitMsg = `MR ATR-TP ${sym}: price $${curr.toFixed(2)} >= target $${(mrAtrTps[sym] || 0).toFixed(2)}`;
        }

        // 3. RSI > 50 exit (mean reversion complete)
        if (!exitTriggered) {
          const rsiVal = rsi(priceHist[sym] || [], MR.RSI_PERIOD);
          if (rsiVal > MR.RSI_EXIT) {
            exitTriggered = true;
            exitReason = "mr-rsi-exit";
            exitMsg = `MR RSI-EXIT ${sym}: RSI ${rsiVal.toFixed(0)} > ${MR.RSI_EXIT} (mean reversion complete)`;
          }
        }

        // 4. Max hold: 15 trading days
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
            delete mrAtrStops[sym];
            delete mrAtrTps[sym];
            delete positionStrategy[sym];
            setCooldown(sym, "mean_reversion");
            addLog(exitMsg, unrealized_pl >= 0 ? "profit" : "sell");
            tradeCount.sells++;
            mrTradeCount.sells++;
            if (unrealized_pl >= 0) { tradeCount.wins++; mrTradeCount.wins++; }
            else { tradeCount.losses++; mrTradeCount.losses++; }
            tradeCount.totalPnL += unrealized_pl;
            mrTradeCount.totalPnL += unrealized_pl;
            try { journal.closePosition({ symbol: sym, fillPrice: curr, exitReason }); } catch (_) {}
            dailyStats.sells++;
            if (unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            const pnlSign = unrealized_pl >= 0 ? "+" : "";
            notify.send(`🔄 MR ${exitReason.replace("mr-", "").toUpperCase()} ${sym} | ${qty} shares @ $${curr.toFixed(2)} | P&L: ${pnlSign}$${unrealized_pl.toFixed(2)}`);
          } catch (err) {
            addLog(`Mean reversion exit failed ${sym}: ${err.message}`, "error");
          }
        }
      }

      // ── STEP 1e: Mega-Cap Overlay exits ──
      for (const pos of currentPositions) {
        const sym = pos.symbol;
        if (closedSymbols.has(sym)) continue;
        if (sym === "SPY" && idleSpyShares > 0) continue;
        if (trendPositions[sym]) continue;
        if (positionStrategy[sym] !== "mega_cap") continue;

        const { qty, current_price: curr, unrealized_pl } = pos;
        const entryPrice = mcEntryPrices[sym] || pos.avg_entry_price;
        const peak = mcPeakPrices[sym] || curr;
        const entryCycle = mcEntryDates[sym] || cycleNumber;

        // Update peak price tracking
        if (curr > peak) mcPeakPrices[sym] = curr;

        let exitTriggered = false;
        let exitReason = "";
        let exitMsg = "";

        // 1. Stop-loss: -5% from entry
        const entryReturn = (curr - entryPrice) / entryPrice;
        if (entryReturn <= MEGACAP.STOP_LOSS) {
          exitTriggered = true;
          exitReason = "mcap-stop-loss";
          exitMsg = `MCAP STOP-LOSS ${sym}: ${(entryReturn * 100).toFixed(1)}% from entry $${entryPrice.toFixed(2)}`;
        }

        // 2. Take-profit: +20% from entry
        if (!exitTriggered && entryReturn >= MEGACAP.TAKE_PROFIT) {
          exitTriggered = true;
          exitReason = "mcap-take-profit";
          exitMsg = `MCAP TAKE-PROFIT ${sym}: +${(entryReturn * 100).toFixed(1)}% from entry $${entryPrice.toFixed(2)}`;
        }

        // 3. Max hold: 90 trading days
        if (!exitTriggered) {
          const cyclesHeld = cycleNumber - entryCycle;
          const approxDaysHeld = cyclesHeld / 390;
          if (approxDaysHeld >= MEGACAP.MAX_HOLD_DAYS) {
            exitTriggered = true;
            exitReason = "mcap-max-hold";
            exitMsg = `MCAP MAX-HOLD ${sym}: held ~${approxDaysHeld.toFixed(0)} trading days (max: ${MEGACAP.MAX_HOLD_DAYS})`;
          }
        }

        // 4. SMA50 trend break
        if (!exitTriggered) {
          const prices = priceHist[sym];
          if (prices && prices.length >= MEGACAP.SMA_EXIT_PERIOD) {
            const sma50 = sma(prices, MEGACAP.SMA_EXIT_PERIOD);
            if (sma50 !== null && curr < sma50) {
              exitTriggered = true;
              exitReason = "mcap-sma50-break";
              exitMsg = `MCAP SMA50-BREAK ${sym}: price $${curr.toFixed(2)} < SMA50 $${sma50.toFixed(2)}`;
            }
          }
        }

        if (exitTriggered) {
          try {
            await closePosition(sym);
            closedSymbols.add(sym);
            delete mcEntryPrices[sym];
            delete mcPeakPrices[sym];
            delete mcEntryDates[sym];
            delete positionStrategy[sym];
            setCooldown(sym, "mega_cap");
            addLog(exitMsg, unrealized_pl >= 0 ? "profit" : "sell");
            tradeCount.sells++;
            mcTradeCount.sells++;
            if (unrealized_pl >= 0) { tradeCount.wins++; mcTradeCount.wins++; }
            else { tradeCount.losses++; mcTradeCount.losses++; }
            tradeCount.totalPnL += unrealized_pl;
            mcTradeCount.totalPnL += unrealized_pl;
            try { journal.closePosition({ symbol: sym, fillPrice: curr, exitReason }); } catch (_) {}
            dailyStats.sells++;
            if (unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
            const pnlSign = unrealized_pl >= 0 ? "+" : "";
            notify.send(`🏛️ MCAP ${exitReason.replace("mcap-", "").toUpperCase()} ${sym} | ${qty} shares @ $${curr.toFixed(2)} | P&L: ${pnlSign}$${unrealized_pl.toFixed(2)}`);
          } catch (err) {
            addLog(`Mega-cap exit failed ${sym}: ${err.message}`, "error");
          }
        }
      }

      // ── STEP 1f: Rebalance exits — sell positions no longer in v9.6 top-N ──
      // v10.2: 20-day minimum hold before rebalance sell (backtested: +27.4% CAGR with 1.25x leverage)
      // Trailing stops still fire immediately regardless of hold period.
      const REBAL_MIN_HOLD_CYCLES = 20 * 390;  // 20 trading days * 390 cycles/day
      if (mlSignals && mlSignals.length > 0) {
        const mlBuySet = new Set(mlSignals.filter(s => s.signal === "BUY").map(s => s.symbol));
        for (const pos of currentPositions) {
          const sym = pos.symbol;
          if (closedSymbols.has(sym)) continue;
          if (sym === "SPY" && idleSpyShares > 0) continue;
          if (trendPositions[sym]) continue;
          // Only rebalance-sell ML positions (not momentum/trend/etc)
          if (positionStrategy[sym] && positionStrategy[sym] !== "ml") continue;

          if (!mlBuySet.has(sym)) {
            // Enforce 10-day minimum hold before rebalance exit
            const entryCycle = mlEntryDates[sym];
            if (entryCycle != null) {
              const cyclesHeld = cycleNumber - entryCycle;
              if (cyclesHeld < REBAL_MIN_HOLD_CYCLES) {
                const daysHeld = (cyclesHeld / 390).toFixed(1);
                addLog(`HOLD ${sym}: dropped from top-8 but min-hold active (${daysHeld}d / 20d) -- skipping rebalance sell`, "system");
                continue;
              }
            }
            try {
              await closePosition(sym);
              closedSymbols.add(sym);
              setCooldown(sym, "ml");
              delete mlEntryDates[sym];
              delete positionStrategy[sym];
              delete trailingPeaks[sym];
              const { qty, current_price: curr, unrealized_pl, unrealized_plpc } = pos;
              addLog(`REBALANCE SELL ${sym}: no longer in v10 top-${RISK.MAX_OPEN_POSITIONS} -- closing | P&L: $${unrealized_pl.toFixed(2)}`, "sell");
              tradeCount.sells++; mlTradeCount.sells++;
              if (unrealized_pl >= 0) { tradeCount.wins++; mlTradeCount.wins++; }
              else { tradeCount.losses++; mlTradeCount.losses++; }
              tradeCount.totalPnL += unrealized_pl;
              mlTradeCount.totalPnL += unrealized_pl;
              try { journal.closePosition({ symbol: sym, fillPrice: curr, exitReason: "rebalance" }); } catch (_) {}
              dailyStats.sells++;
              if (unrealized_pl >= 0) dailyStats.wins++; else dailyStats.losses++;
              notify.send(`🔄 REBALANCE ${sym} | ${qty} shares @ $${curr.toFixed(2)} | No longer in top-8 — P&L: $${unrealized_pl.toFixed(2)} (${(unrealized_plpc * 100).toFixed(1)}%)`);
            } catch (err) {
              addLog(`Rebalance sell failed ${sym}: ${err.message}`, "error");
            }
          }
        }
      }

      // Refresh positions and cash after stop-loss + rebalance sells so slot counts are accurate
      let activePositions = currentPositions;
      if (closedSymbols.size > 0) {
        try {
          activePositions = await getPositions();
          const freshAcct = await getAccount();
          cycleCash = freshAcct.cash;
          cyclePortfolioValue = freshAcct.portfolio_value;
          addLog(`[post-exit] Refreshed: ${activePositions.length} positions, $${cycleCash.toLocaleString("en-US", {maximumFractionDigits: 0})} cash`, "system");
        } catch (_) {
          // Fallback: filter closed symbols from snapshot
          activePositions = currentPositions.filter(p => !closedSymbols.has(p.symbol));
        }
      }

      // ── STEP 1g: Maintain GLD + VIXM hedge allocations (2% each) ──
      // v10: permanent 2% VIXM (tail hedge), 2% GLD (trend-following)
      // GLD: buy when above 252-day SMA, sell when below
      // VIXM: always hold 2%
      const HEDGE_PCT = {};  // GLD and VIXM removed — GLD adds <0.6% CAGR, causes 50 orders/day churning
      // Sell any removed hedges (e.g. VIXM) that are still held
      for (const removedHedge of ["VIXM"]) {
        const oldPos = activePositions.find(p => p.symbol === removedHedge);
        if (oldPos && oldPos.qty > 0) {
          addLog(`[hedge] SELLING removed hedge ${removedHedge}: ${oldPos.qty} shares`, "sell");
          await placeOrder({ symbol: removedHedge, qty: oldPos.qty, side: "sell", type: "market" });
          delete positionStrategy[removedHedge];
        }
      }
      for (const hedgeSym of Object.keys(HEDGE_PCT)) {
        const targetPct = HEDGE_PCT[hedgeSym];
        const hedgePos = activePositions.find(p => p.symbol === hedgeSym);
        const hedgeValue = hedgePos ? hedgePos.market_value : 0;
        const targetValue = cyclePortfolioValue * targetPct;
        const currentPct = hedgeValue / cyclePortfolioValue;

        // GLD trend filter: hold if price > 252-day SMA
        // Hysteresis band: buy when >2% above SMA, sell when >2% below SMA
        // This prevents whipsaw when price oscillates around the SMA
        let shouldBuy = false;   // only buy when clearly above
        let shouldSell = false;  // only sell when clearly below
        if (hedgeSym === "GLD" && priceHist.GLD) {
          const gldPrices = priceHist.GLD;
          if (gldPrices.length >= 252) {
            const sma252 = gldPrices.slice(-252).reduce((a, b) => a + b, 0) / 252;
            const currGld = gldPrices[gldPrices.length - 1];
            const pctFromSma = (currGld - sma252) / sma252;
            shouldBuy = pctFromSma > 0.02;   // >2% above SMA to buy
            shouldSell = pctFromSma < -0.02; // >2% below SMA to sell
            shouldHold = !shouldSell;         // hold unless clearly below
          }
        }

        if (shouldSell && hedgePos) {
          // GLD >2% below SMA — sell
          try {
            await closePosition(hedgeSym);
            closedSymbols.add(hedgeSym);
            addLog(`[hedge] SELL ${hedgeSym}: >2% below 252d SMA — trend off`, "sell");
              try { journal.closePosition({ symbol: hedgeSym, fillPrice: hedgePos.current_price, exitReason: "hedge-trend-off" }); } catch (_) {}
              notify.send(`🛡️ HEDGE SELL ${hedgeSym} | >2% below 252d SMA — trend off`);
          } catch (err) {
            addLog(`[hedge] Failed to sell ${hedgeSym}: ${err.message}`, "error");
          }
        } else if (shouldBuy && !hedgePos) {
          // GLD >2% above SMA and no position — buy
          const hedgePrice = priceHist[hedgeSym]?.[priceHist[hedgeSym].length - 1];
          if (hedgePrice && hedgePrice > 0 && cycleCash > targetValue * 0.5) {
            const shares = Math.floor(targetValue / hedgePrice);
            if (shares > 0) {
              try {
                await placeOrder({ symbol: hedgeSym, qty: shares, side: "buy", type: "market" });
                cycleCash -= shares * hedgePrice;
                positionStrategy[hedgeSym] = "hedge";
                addLog(`[hedge] BUY ${shares} ${hedgeSym} @ $${hedgePrice.toFixed(2)} (${(targetPct * 100).toFixed(0)}% target)`, "buy");
                notify.send(`🛡️ HEDGE BUY ${shares} ${hedgeSym} @ $${hedgePrice.toFixed(2)} | ${(targetPct * 100).toFixed(0)}% target`);
              } catch (err) {
                addLog(`[hedge] Failed to buy ${hedgeSym}: ${err.message}`, "error");
              }
            }
          }
        }
        // No action in the dead zone (-2% to +2% from SMA) — prevents whipsaw
      }

      // Tag hedge positions so they're not sold by rebalance logic
      for (const hSym of ["GLD", "VIXM"]) {
        if (positionStrategy[hSym] !== "hedge" && activePositions.find(p => p.symbol === hSym)) {
          positionStrategy[hSym] = "hedge";
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
        addLog(`v10 active -- ${buyCount} BUY signal${buyCount !== 1 ? "s" : ""} (top-${buyCount} cross-sectional ranking) | vol scale: ${currentVolScale.toFixed(3)}`, "system");
      } else {
        addLog("Signal server offline -- using consensus engine (fallback mode)", "system");
      }

      // Include both held positions AND pending buy orders to prevent duplicate buys.
      // Uses activePositions (refreshed after rebalance sells) for accurate count.
      const heldSymbols = new Set(activePositions.map(p => p.symbol));
      for (const o of pendingOrders) {
        if (o.side === "buy") heldSymbols.add(fromAlpacaSymbol(o.symbol));
      }
      const activePositionCount = activePositions.filter(p => p.symbol !== "SPY").length;
      const opportunities = [];

      for (const sym of UNIVERSE_SYMBOLS) {
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
              const blStrat = positionStrategy[sym] || "legacy";
              setCooldown(sym, blStrat);
              if (blStrat === "momentum") { delete momEntryPrices[sym]; delete momPeakPrices[sym]; delete momEntryDates[sym]; }
              else if (blStrat === "mean_reversion") { delete mrEntryPrices[sym]; delete mrEntryDates[sym]; }
              else if (blStrat === "mega_cap") { delete mcEntryPrices[sym]; delete mcPeakPrices[sym]; delete mcEntryDates[sym]; }
              else if (blStrat === "ml") { delete mlEntryDates[sym]; }
              delete positionStrategy[sym];
              addLog(`SELL ${sym}: ${analysis.consensus} -- closing blacklisted position`, "sell");
              if (posData) {
                const blTracker = blStrat === "momentum" ? momTradeCount : blStrat === "mean_reversion" ? mrTradeCount : blStrat === "mega_cap" ? mcTradeCount : mlTradeCount;
                tradeCount.sells++; blTracker.sells++;
                if (posData.unrealized_pl >= 0) { tradeCount.wins++; blTracker.wins++; }
                else { tradeCount.losses++; blTracker.losses++; }
                tradeCount.totalPnL += posData.unrealized_pl;
                blTracker.totalPnL += posData.unrealized_pl;
                try { journal.closePosition({ symbol: sym, fillPrice: posData.current_price, exitReason: "blacklist" }); } catch (_) {}
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

        // Consensus sell disabled when ML/v9.6 is active — v9.6 exits via rebalance (Step 1f).
        // Old technical signals (SMA, RSI, MACD) would fight the factor-based strategy.
        const ML_MIN_HOLD_CYCLES = 5 * 390;
        const excludedStrategies = new Set(["momentum", "mean_reversion", "legacy", "mega_cap", "trend", "ml"]);
        if (heldSymbols.has(sym) && !trendPositions[sym] && !excludedStrategies.has(positionStrategy[sym]) && (analysis.consensus === "STRONG SELL" || analysis.consensus === "SELL")) {
          // Check minimum hold period before allowing consensus-based exit
          const mlEntryCycle = mlEntryDates[sym];
          if (mlEntryCycle != null) {
            const mlCyclesHeld = cycleNumber - mlEntryCycle;
            if (mlCyclesHeld < ML_MIN_HOLD_CYCLES) {
              const approxDaysHeld = (mlCyclesHeld / 390).toFixed(1);
              addLog(`HOLD ${sym}: ${analysis.consensus} signal but ML min-hold active (${approxDaysHeld}d / 5.0d) -- skipping sell`, "system");
              continue;
            }
          }
          try {
            await closePosition(sym);
            const posData = currentPositions.find(p => p.symbol === sym);
            // Always set cooldown on consensus-sell to prevent buy-sell-buy churn
            setCooldown(sym, "ml");
            delete mlEntryDates[sym];
            delete positionStrategy[sym];
            addLog(`SELL ${sym}: ${analysis.consensus} -- closing position (held ${mlEntryCycle != null ? ((cycleNumber - mlEntryCycle) / 390).toFixed(1) : "?"}d)`, "sell");
            if (posData) {
              tradeCount.sells++; mlTradeCount.sells++;
              if (posData.unrealized_pl >= 0) { tradeCount.wins++; mlTradeCount.wins++; }
              else { tradeCount.losses++; mlTradeCount.losses++; }
              tradeCount.totalPnL += posData.unrealized_pl;
              mlTradeCount.totalPnL += posData.unrealized_pl;
              try { journal.closePosition({ symbol: sym, fillPrice: posData.current_price, exitReason: "consensus-sell" }); } catch (_) {}
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
          const cd = cooldowns[cdKey];
          const remainCycles = Math.max(0, RISK.LOSS_COOLDOWN_CYCLES - (cycleNumber - cd.cycle));
          const remainMs = Math.max(0, cd.expireTime - Date.now());
          const remainMin = Math.ceil(remainMs / 60000);
          if (isMLBuy) {
            addLog(`EVAL ${sym}: conf ${(mlMap[sym].probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | BLOCKED: ML cooldown, ${remainCycles} cycle${remainCycles !== 1 ? "s" : ""} / ${remainMin}min remaining`, "system");
          }
          continue;
        }

        // Earnings proximity check (pre-computed for ML eval log)
        const earningsDate = earningsMap[sym];
        const earningsDays = earningsDate ? daysUntilEarnings(earningsDate) : null;
        const earningsBlocked = earningsDays !== null && earningsDays >= 0 && earningsDays <= 3;

        if (mlActive) {
          // v9.6: top-N cross-sectional ranking (BUY signal)
          // v9.6 handles regime internally via breadth blending — no CAUTIOUS filter needed
          const mlSig = mlMap[sym];
          if (mlSig && mlSig.signal === "BUY") {
            addLog(`EVAL ${sym}: rank #${mlSig.rank} conf ${(mlSig.probability * 100).toFixed(0)}% | cash $${cycleCash.toFixed(0)} | regime ${regime} | slots ${activePositionCount}/${RISK.MAX_OPEN_POSITIONS} | earnings blocked: ${earningsBlocked}${earningsBlocked ? ` (${earningsDays}d -> ${earningsDate})` : ""} | cooldown: false | PASSED -> added to candidates`, "system");
            opportunities.push({
              sym,
              score: mlSig.probability,
              price: prices[prices.length - 1],
              consensus: `v10 #${mlSig.rank} (${(mlSig.probability * 100).toFixed(0)}%)`,
              rsiVal: analysis.indicators.rsi,
              mlConf: mlSig.probability,
            });
          } else {
            if (mlSig && mlSig.rank <= 10) {
              addLog(`ML SKIP ${sym} -- rank #${mlSig.rank}, not in top 8`, "system");
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
      const spySma50 = computeRegime(priceHist.SPY).sma50;
      const spyNowPrice = priceHist.SPY?.[priceHist.SPY.length - 1];
      const momRegimeBlocked = ENABLE_MOMENTUM_REGIME_FILTER
        && spySma50 != null && spyNowPrice != null
        && spyNowPrice < spySma50;

      if (momRegimeBlocked) {
        addLog(`[momentum] SKIPPED — SPY $${spyNowPrice.toFixed(2)} below 50-SMA $${spySma50.toFixed(2)} (regime filter)`, "system");
      } else if (regime !== "BEARISH") {
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
        addLog(`MR signals: ${mrOpportunities.length} candidate${mrOpportunities.length !== 1 ? "s" : ""} (${mrOpportunities.map(o => `${o.sym} RSI=${o.rsiVal?.toFixed(0) ?? "?"}`).join(", ")})`, "system");
      }

      // ── STEP 2d: Compute mega-cap overlay signals (ALWAYS active) ──
      const mcRawSignals = computeMegaCapSignals(priceHist, heldSymbols);

      // Filter mega-cap signals: check cooldowns, first-to-fire overlap
      const mcOpportunities = [];
      for (const sig of mcRawSignals) {
        if (isOnCooldown(sig.sym, "mega_cap")) continue;
        // First-to-fire: skip if other strategies already claimed this symbol
        if (opportunities.some(o => o.sym === sig.sym)) continue;
        if (momOpportunities.some(o => o.sym === sig.sym)) continue;
        if (mrOpportunities.some(o => o.sym === sig.sym)) continue;
        mcOpportunities.push(sig);
      }

      if (mcOpportunities.length > 0) {
        addLog(`MCAP signals: ${mcOpportunities.length} candidate${mcOpportunities.length !== 1 ? "s" : ""} (${mcOpportunities.map(o => `${o.sym} #${o.rank}`).join(", ")})`, "system");
      }

      // ── STEP 3: Rank & execute buys (strategy-aware slot allocation) ──
      // Tag ML opportunities with strategy
      for (const opp of opportunities) {
        if (!opp.strategy) opp.strategy = "ml";
      }

      // Merge: ML first (highest priority), then momentum, then mean reversion, then mega-cap
      const allOpportunities = [...opportunities, ...momOpportunities, ...mrOpportunities, ...mcOpportunities];
      // Sort within each strategy group by score, ML > momentum > MR > mega-cap
      const stratPriority = { ml: 0, momentum: 1, mean_reversion: 2, mega_cap: 3 };
      allOpportunities.sort((a, b) => {
        const pa = stratPriority[a.strategy] ?? 9;
        const pb = stratPriority[b.strategy] ?? 9;
        if (pa !== pb) return pa - pb;
        return b.score - a.score;
      });

      const counts = countByStrategy(activePositions);
      const slotsAvail = SLOT_CONFIG.max - counts.total;

      // Pre-execution diagnostic summary
      if (allOpportunities.length > 0) {
        addLog(`BUY FILTER CHECK -- ${allOpportunities.length} candidate${allOpportunities.length !== 1 ? "s" : ""} (${opportunities.length} ML + ${momOpportunities.length} MOM + ${mrOpportunities.length} MR + ${mcOpportunities.length} MCAP) | positions: ${counts.total}/${SLOT_CONFIG.max} (ML ${counts.ml}/${SLOT_CONFIG.ml_medium}, MOM ${counts.mom}/${SLOT_CONFIG.momentum}, MR ${counts.mr}/${SLOT_CONFIG.mean_reversion}, MCAP ${counts.mc}/${SLOT_CONFIG.mega_cap}, TREND ${counts.trend}, LEGACY ${counts.legacy}) | slots open: ${slotsAvail} | cash: $${cycleCash.toLocaleString("en-US", { maximumFractionDigits: 0 })}`, "system");
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
      if (ENABLE_SPY_PARKING) {
        const spyPos = currentPositions.find(p => p.symbol === "SPY");
        if (idleSpyShares > 0 && spyPos && allOpportunities.length > 0) {
          const spyShareCount = spyPos.qty;
          const spyPrice = spyPos.current_price || spyPos.avg_entry_price;
          const idleValue = spyPos.market_value;

          // Estimate total opportunity size from position sizing
          const estOppSize = allOpportunities.reduce((sum, opp) => {
            const pct = opp.positionPct || RISK.MAX_POSITION_PCT;
            return sum + cyclePortfolioValue * pct * currentVolScale;
          }, 0);

          // Only sell SPY if opportunities are meaningful relative to parked SPY
          const minOppRatio = 0.25;
          const meaningfulOpps = estOppSize >= idleValue * minOppRatio;

          // Dead zone: don't rebalance SPY more than once per 15 min wall-clock
          const IDLE_SPY_MIN_GAP_MS = 15 * 60 * 1000;
          const msSinceLastAction = Date.now() - lastIdleSpyActionAt;
          const outsideDeadZone = msSinceLastAction >= IDLE_SPY_MIN_GAP_MS;

          if (meaningfulOpps && outsideDeadZone) {
            try {
              await closePosition("SPY");
              idleSpyShares = 0;
              lastIdleSpyActionAt = Date.now();
              addLog(`[idle-spy] Selling ${spyShareCount} SPY ($${idleValue.toFixed(0)}) to fund ${allOpportunities.length} picks (est $${estOppSize.toFixed(0)})`, "system");
              try { journal.closePosition({ symbol: "SPY", fillPrice: spyPrice, exitReason: "idle-spy-sell" }); } catch (_) {}
              notify.send(`🅿️ SPY IDLE SELL | ${spyShareCount} shares @ $${spyPrice.toFixed(2)} | Freeing cash for ${allOpportunities.length} pick${allOpportunities.length !== 1 ? "s" : ""}`);
              const freshAcct = await getAccount();
              cycleCash = freshAcct.cash;
            } catch (err) {
              addLog(`[idle-spy] Failed to sell SPY: ${err.message}`, "error");
            }
          } else if (!meaningfulOpps) {
            addLog(`[idle-spy] Skip SPY sell -- opps $${estOppSize.toFixed(0)} < ${(minOppRatio * 100).toFixed(0)}% of idle $${idleValue.toFixed(0)}`, "system");
          } else {
            const minsRemaining = Math.ceil((IDLE_SPY_MIN_GAP_MS - msSinceLastAction) / 60000);
            addLog(`[idle-spy] Skip SPY sell -- last action ${Math.floor(msSinceLastAction / 60000)}min ago, need ${minsRemaining}min more`, "system");
          }
        }
      }

      // Track sectors bought this cycle and live slot counts
      const boughtThisCycle = {};
      let liveCounts = { ...counts };  // mutable copy for tracking during execution

      for (const opp of allOpportunities) {
        // Check strategy-specific slot capacity
        if (!hasSlotCapacity(opp.strategy, liveCounts)) {
          addLog(`SKIP ${opp.sym} -- no ${opp.strategy} slot available (ML ${liveCounts.ml}/${SLOT_CONFIG.ml_medium}, MOM ${liveCounts.mom}/${SLOT_CONFIG.momentum}, MR ${liveCounts.mr}/${SLOT_CONFIG.mean_reversion}, MCAP ${liveCounts.mc}/${SLOT_CONFIG.mega_cap}, TREND ${liveCounts.trend || 0}, total ${liveCounts.total}/${SLOT_CONFIG.max})`, "system");
          continue;
        }

        // Sector position limit
        const sector = getSector(opp.sym);
        const sectorLimit = RISK.SECTOR_MAX_POSITIONS[sector];
        if (sectorLimit !== undefined) {
          const existingSectorCount = activePositions.filter(p => getSector(p.symbol) === sector).length;
          const cycleCount = boughtThisCycle[sector] || 0;
          if (existingSectorCount + cycleCount >= sectorLimit) {
            addLog(`Skipping ${opp.sym} -- ${sector} limit reached (max ${sectorLimit})`, "system");
            continue;
          }
        }

        // Volume confirmation check -- skip for ML-driven, momentum, MR, and mega-cap trades
        if (opp.strategy !== "ml" && opp.strategy !== "momentum" && opp.strategy !== "mean_reversion" && opp.strategy !== "mega_cap") {
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
        // Disabled for ML/v9.6: the strategy already factors in earnings via eps_surprise boost
        // and the backtest does not avoid earnings — adding this filter causes cash drag
        if (opp.strategy !== "ml" && opp.strategy !== "momentum" && opp.strategy !== "mean_reversion") {
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
        if (opp.strategy === "momentum" || opp.strategy === "mean_reversion" || opp.strategy === "mega_cap") {
          // Momentum/MR/Mega-cap: fixed % (already computed in signal)
          dynPositionPct = opp.positionPct;
        } else {
          // v9.6 ML: equal-weight sizing (12% per position, no ATR/regime scaling)
          // The backtest uses fixed equal weights — ATR scaling and regime multipliers
          // would cause the live system to diverge from backtested performance.
          dynPositionPct = RISK.MAX_POSITION_PCT;
        }

        // Apply volatility targeting scale to ALL strategies
        // v10.2: use buying power (portfolio × leverage) not just cash
        const buyingPower = cyclePortfolioValue * RISK.MAX_CASH_DEPLOY_PCT;  // 1.25x leverage
        const cashReserve = cyclePortfolioValue * 0.02;
        const availableBuyingPower = Math.max(0, buyingPower - (cyclePortfolioValue - cycleCash) - cashReserve);
        if (availableBuyingPower < opp.price) {
          addLog(`SKIP ${opp.sym} -- buying power: $${availableBuyingPower.toFixed(0)} available (${RISK.MAX_CASH_DEPLOY_PCT}x leverage), need $${opp.price.toFixed(2)}/share`, "system");
          continue;
        }

        // v10.2: vol-targeting enabled for ML positions
        const volMult = currentVolScale;
        const maxAlloc = cyclePortfolioValue * dynPositionPct * volMult;
        const allocCash = Math.min(maxAlloc, availableBuyingPower);
        if (allocCash < opp.price) {
          addLog(`SKIP ${opp.sym} -- insufficient buying power: need $${opp.price.toFixed(2)}/share, alloc $${allocCash.toFixed(2)} (${(dynPositionPct * 100).toFixed(1)}% of portfolio)`, "system");
          continue;
        }

        const shares = Math.floor(allocCash / opp.price);
        if (shares <= 0) {
          addLog(`SKIP ${opp.sym} -- 0 shares at $${opp.price.toFixed(2)}/share with $${allocCash.toFixed(2)} allocated`, "system");
          continue;
        }

        const cost = shares * opp.price;

        // Minimum position size gate — skip tiny positions
        if (cost < RISK.MIN_POSITION_DOLLARS) {
          addLog(`[size] SKIP ${opp.sym} -- calculated size $${cost.toFixed(0)} < minimum $${RISK.MIN_POSITION_DOLLARS}`, "system");
          continue;
        }
        if (cost > availableBuyingPower) {
          addLog(`SKIP ${opp.sym} -- order cost $${cost.toFixed(0)} exceeds buying power $${availableBuyingPower.toFixed(0)}`, "system");
          continue;
        }

        try {
          const order = await placeOrder({ symbol: opp.sym, qty: shares, side: "buy", type: "market" });
          // Decrement cycleCash so subsequent buys in this cycle see reduced cash
          cycleCash -= cost;
          boughtThisCycle[sector] = (boughtThisCycle[sector] || 0) + 1;

          // Track strategy attribution
          positionStrategy[opp.sym] = opp.strategy;
          const stratTracker = opp.strategy === "momentum" ? momTradeCount
            : opp.strategy === "mean_reversion" ? mrTradeCount
            : opp.strategy === "mega_cap" ? mcTradeCount
            : mlTradeCount;
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
            mrAtrStops[opp.sym] = opp.atrStop;
            mrAtrTps[opp.sym] = opp.atrTp;
          } else if (opp.strategy === "mega_cap") {
            liveCounts.mc++;
            // Track mega-cap entry state
            mcEntryPrices[opp.sym] = opp.price;
            mcPeakPrices[opp.sym] = opp.price;
            mcEntryDates[opp.sym] = cycleNumber;
          } else {
            liveCounts.ml++;
            mlEntryDates[opp.sym] = cycleNumber;
          }
          liveCounts.total++;

          // Journal: record order submitted with attribution
          try {
            journal.recordOrderSubmitted({
              alpaca_order_id: order.id || null,
              client_order_id: order.client_order_id || null,
              symbol: opp.sym, side: "buy", qty: shares,
              strategy: opp.strategy,
              signal_prob: opp.mlConf || null,
              ml_rank: opp.mlRank || null,
              regime,
              intended_price: opp.price,
            });
          } catch (_) { /* never crash trading loop */ }

          if (opp.strategy === "mega_cap") {
            addLog(`MCAP BUY ${opp.sym}: ${shares} shares | ${opp.consensus} | alloc ${(dynPositionPct * 100).toFixed(1)}% | Order: ${order.status}`, "buy");
            notify.send(`🏛️ MCAP BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | ${opp.consensus} | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`);
          } else if (opp.strategy === "momentum") {
            addLog(`MOM BUY ${opp.sym}: ${shares} shares | ${opp.consensus} | alloc ${(dynPositionPct * 100).toFixed(1)}% | Order: ${order.status}`, "buy");
            notify.send(`📊 MOM BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | ${opp.consensus} | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`);
          } else if (opp.strategy === "mean_reversion") {
            addLog(`MR BUY ${opp.sym}: ${shares} shares | ${opp.consensus} | conf ${(opp.score * 100).toFixed(0)}% | alloc ${(dynPositionPct * 100).toFixed(1)}% | Order: ${order.status}`, "buy");
            notify.send(`🔄 MR BUY ${opp.sym} | ${shares} shares @ $${opp.price.toFixed(2)} | ${opp.consensus} | Portfolio: $${cyclePortfolioValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`);
          } else {
            const mlNote = opp.mlConf != null ? `, ML ${(opp.mlConf * 100).toFixed(0)}% conf` : "";
            addLog(`ML BUY ${opp.sym}: ${shares} shares | ${opp.consensus} (score: ${opp.score.toFixed(2)}) | alloc ${(dynPositionPct * 100).toFixed(1)}%${mlNote} | Order: ${order.status}`, "buy");
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
            delete positionStrategy[sym];
            const trendPnl = pos.unrealized_pl;
            addLog(`TREND-STOP ${sym}: @ $${curr.toFixed(2)} | Peak $${peak.toFixed(2)}, drop ${(dropFromPeak * 100).toFixed(1)}%`, "sell");
            tradeCount.sells++;
            if (trendPnl >= 0) tradeCount.wins++; else tradeCount.losses++;
            tradeCount.totalPnL += trendPnl;
            try { journal.closePosition({ symbol: sym, fillPrice: curr, exitReason: "trend-stop" }); } catch (_) {}
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
              delete positionStrategy[sym];
              const breakPos = currentPositions.find(p => p.symbol === sym);
              addLog(`TREND-BREAK ${sym}: 3 consecutive closes below 200-SMA -- exiting`, "sell");
              if (breakPos) {
                tradeCount.sells++;
                if (breakPos.unrealized_pl >= 0) tradeCount.wins++; else tradeCount.losses++;
                tradeCount.totalPnL += breakPos.unrealized_pl;
                try { journal.closePosition({ symbol: sym, fillPrice: breakPos.current_price, exitReason: "trend-break" }); } catch (_) {}
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
      if (DISABLE_TREND_STRATEGY) {
        addLog(`[trend] SKIPPED — DISABLE_TREND_STRATEGY flag is true`, "system");
      } else if (!skipNewBuys && regime !== "BEARISH") {
        const trendCount = Object.keys(trendPositions).length;
        let trendCounts = countByStrategy();
        if (trendCount < 3 && trendCounts.total < SLOT_CONFIG.max) {
          const trendPosValue = Object.values(trendPositions).reduce((sum, tp) => {
            const alpacaPos = currentPositions.find(p => p.symbol === tp.sym);
            return sum + (alpacaPos ? alpacaPos.market_value : 0);
          }, 0);
          const trendPortPct = cyclePortfolioValue > 0 ? trendPosValue / cyclePortfolioValue : 0;

          if (trendPortPct < 0.30) {
            let trendBuysThisCycle = 0;
            for (const sym of UNIVERSE_SYMBOLS) {
              if (NEVER_BUY.has(sym)) continue;
              if (trendPositions[sym] || heldSymbols.has(sym)) continue;
              const prices = priceHist[sym];
              if (!prices || prices.length < 200) continue;
              const ts = computeTrendStatus(prices);
              if (!ts?.isStrongUptrend) continue;

              // Slot limit checks
              if (trendCounts.total >= SLOT_CONFIG.max) break;
              if (!hasSlotCapacity("trend", trendCounts)) {
                addLog(`TREND SKIP ${sym} -- no flex slot available (total ${trendCounts.total}/${SLOT_CONFIG.max})`, "system");
                break;
              }
              if (Object.keys(trendPositions).length + trendBuysThisCycle >= 3) break;

              const currPrice = prices[prices.length - 1];
              const allocCashTrend = cyclePortfolioValue * 0.05;
              if (allocCashTrend < currPrice || allocCashTrend > cycleCash) continue;
              const shares = Math.floor(allocCashTrend / currPrice);
              if (shares <= 0) continue;
              const trendCost = shares * currPrice;
              if (trendCost < RISK.MIN_POSITION_DOLLARS) {
                addLog(`[size] SKIP ${sym} (trend) -- calculated size $${trendCost.toFixed(0)} < minimum $${RISK.MIN_POSITION_DOLLARS}`, "system");
                continue;
              }

              try {
                const order = await placeOrder({ symbol: sym, qty: shares, side: "buy", type: "market" });
                trendPositions[sym] = { entryPrice: currPrice, peakPrice: currPrice };
                trendBreakCounts[sym] = 0;
                positionStrategy[sym] = "trend";
                trendBuysThisCycle++;
                trendCounts.trend++;
                trendCounts.total++;
                cycleCash -= shares * currPrice;
                addLog(`TREND-BUY ${sym}: ${shares} sh | ${ts.daysAbove200}/40 days above 200-SMA | Order: ${order.status}`, "buy");
                tradeCount.buys++;
                dailyStats.buys++;
                try { journal.recordOrderSubmitted({ alpaca_order_id: order.id || null, symbol: sym, side: "buy", qty: shares, strategy: "trend", regime, intended_price: currPrice }); } catch (_) {}
                notify.send(`📈 TREND BUY ${sym} | ${shares} shares @ $${currPrice.toFixed(2)} | ${ts.daysAbove200}/40 days above 200-SMA`);
              } catch (err) {
                addLog(`Trend buy failed ${sym}: ${err.message}`, "error");
              }
            }
          }
        }
      }

      // ── STEP 7: Park idle cash in SPY ──
      if (ENABLE_SPY_PARKING && !skipNewBuys) {
        try {
          const freshAcct = await getAccount();
          const freshCash = freshAcct.cash;
          const freshPortVal = freshAcct.portfolio_value;
          const reservedCash = freshPortVal * SPY_IDLE_RESERVE_PCT;
          const idleCash = freshCash - reservedCash;

          if (idleCash > freshPortVal * SPY_IDLE_THRESHOLD_PCT) {
            // Dead zone: don't rebalance SPY more than once per 15 min
            const IDLE_SPY_MIN_GAP_MS = 15 * 60 * 1000;
            const msSinceLastAction = Date.now() - lastIdleSpyActionAt;
            if (msSinceLastAction < IDLE_SPY_MIN_GAP_MS) {
              const minsRemaining = Math.ceil((IDLE_SPY_MIN_GAP_MS - msSinceLastAction) / 60000);
              addLog(`[idle-spy] Skip SPY park -- last action ${Math.floor(msSinceLastAction / 60000)}min ago, need ${minsRemaining}min more`, "system");
            } else {
              const parkAmount = idleCash * SPY_IDLE_INVEST_PCT;
              const spyPrice = priceHist.SPY?.[priceHist.SPY.length - 1];
              if (spyPrice && spyPrice > 0 && parkAmount >= spyPrice) {
                const spySharesToBuy = Math.floor(parkAmount / spyPrice);
                if (spySharesToBuy > 0) {
                  const spyOrder = await placeOrder({ symbol: "SPY", qty: spySharesToBuy, side: "buy", type: "market" });
                  idleSpyShares += spySharesToBuy;
                  lastIdleSpyActionAt = Date.now();
                  addLog(`[idle-spy] Parking $${parkAmount.toLocaleString("en-US", { maximumFractionDigits: 0 })} -> ${spySharesToBuy} SPY @ $${spyPrice.toFixed(2)} | Total idle SPY: ${idleSpyShares} shares`, "system");
                  try { journal.recordOrderSubmitted({ alpaca_order_id: spyOrder.id || null, symbol: "SPY", side: "buy", qty: spySharesToBuy, strategy: "idle-spy", regime, intended_price: spyPrice }); } catch (_) {}
                  notify.send(`🅿️ SPY IDLE BUY | ${spySharesToBuy} shares @ $${spyPrice.toFixed(2)} | Idle cash parked`);
                }
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

      // ── STEP 4: Short Sleeve Execution ──
      if (shortSignals && shortSignals.length > 0 && !circuitBreaker.halted) {
        for (const sig of shortSignals) {
          try {
            const sym = sig.symbol;
            if (!sym) continue;

            // COVER signals — buy to close short positions
            if (sig.signal === "COVER") {
              const pos = activePositions.find(p => p.symbol === sym && parseFloat(p.qty) < 0);
              if (pos) {
                const qty = Math.abs(parseFloat(pos.qty));
                addLog(`SHORT COVER: ${sym} | ${sig.exit_reason} | days=${sig.days_held} | pnl=${(sig.current_pnl * 100).toFixed(1)}%`);
                const order = await alpaca.createOrder({
                  symbol: sym, qty, side: "buy", type: "market", time_in_force: "day",
                });
                journal.recordOrderSubmitted({
                  alpaca_order_id: order.id, symbol: sym, side: "buy", qty,
                  strategy: "event_short", signal_prob: sig.probability,
                  regime, intended_price: parseFloat(pos.current_price),
                });
                positionStrategy[sym] = undefined;
              }
            }

            // SHORT signals — sell short to open new positions
            if (sig.signal === "SHORT") {
              // Check not already held (long or short)
              if (heldSymbols.has(sym)) continue;
              // Check circuit breaker
              if (circuitBreaker.halted) continue;

              const price = parseFloat(priceHist[sym]?.[priceHist[sym]?.length - 1] || 0);
              if (price <= 0) continue;

              // Position size: 10% of portfolio (1/MAX_POSITIONS)
              const shortAlloc = portfolioValue * (sig.position_pct || 0.10);
              const shares = Math.floor(shortAlloc / price);
              if (shares < 1 || shortAlloc < 2000) continue;

              addLog(`SHORT ENTRY: ${shares} ${sym} @ $${price.toFixed(2)} | event=${sig.event_type} | filed=${sig.filing_date}`);
              const order = await alpaca.createOrder({
                symbol: sym, qty: shares, side: "sell", type: "market", time_in_force: "day",
              });
              journal.recordOrderSubmitted({
                alpaca_order_id: order.id, symbol: sym, side: "sell", qty: shares,
                strategy: "event_short", signal_prob: sig.probability,
                regime, intended_price: price,
              });
              positionStrategy[sym] = "event_short";
              heldSymbols.add(sym);
            }
          } catch (err) {
            addLog(`Short sleeve error (${sig.symbol}): ${err.message}`, "error");
          }
        }
      }

      // Daily snapshot + Telegram summary near market close (last 5 minutes)
      if (minutesUntilClose <= 5) {
        const today = new Date().toISOString().split("T")[0];
        const activePos = positionsRaw.filter(p => p.symbol !== "SPY").length;
        const dailyPnl = portfolioValue - (circuitBreaker.marketOpenValue || portfolioValue);
        // Journal: daily snapshot
        try {
          const dailyPnlPct = circuitBreaker.marketOpenValue ? (dailyPnl / circuitBreaker.marketOpenValue) : 0;
          const spyClose = priceHist.SPY?.[priceHist.SPY.length - 1] || null;
          journal.writeDailySnapshot({
            date: today,
            portfolio_value: portfolioValue,
            cash,
            equity: portfolioValue,
            positions_count: activePos,
            day_pnl: dailyPnl,
            day_pnl_pct: dailyPnlPct,
            regime,
            spy_close: spyClose,
            strategy_pnl: {
              ml: mlTradeCount.totalPnL, momentum: momTradeCount.totalPnL,
              mean_reversion: mrTradeCount.totalPnL, mega_cap: mcTradeCount.totalPnL,
            },
            slot_usage: {
              ml: mlTradeCount.buys, momentum: momTradeCount.buys,
              mean_reversion: mrTradeCount.buys, mega_cap: mcTradeCount.buys,
            },
            circuit_breaker_state: circuitBreaker.peakHalted ? "peak_halt" : circuitBreaker.weeklyHalted ? "weekly_halt" : circuitBreaker.dailyHalted ? "daily_halt" : "ok",
            peak_value: circuitBreaker.peakValue,
            drawdown_pct: circuitBreaker.peakValue > 0 ? (1 - portfolioValue / circuitBreaker.peakValue) : 0,
          });
        } catch (_) {}

        // Send daily/weekly summary via Telegram (once per day)
        if (!dailyStats.summarySent) {
          dailyStats.summarySent = true;
          const dailyPnlPct = circuitBreaker.marketOpenValue ? (dailyPnl / circuitBreaker.marketOpenValue * 100) : 0;
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
          summary += `\nSlots: v10 ${mlPos}/${SLOT_CONFIG.ml_medium} | MOM ${momPos}/${SLOT_CONFIG.momentum} | MR ${mrPos}/${SLOT_CONFIG.mean_reversion} | Flex ${flexUsed}/${SLOT_CONFIG.flex}`;
          summary += `\nv10 P&L: $${mlTradeCount.totalPnL.toFixed(0)} (${mlTradeCount.wins}W/${mlTradeCount.losses}L)`;
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
    let mlPositions = 0, momPositions = 0, mrPositions = 0, trendPositionCount = 0, legacyPositions = 0;
    for (const sym of Object.keys(positions)) {
      if (sym === "SPY" && idleSpyShares > 0) continue;
      const strat = positionStrategy[sym] || "legacy";
      if (strat === "legacy") legacyPositions++;
      else if (strat === "momentum") momPositions++;
      else if (strat === "mean_reversion") mrPositions++;
      else if (strat === "trend" || trendPositions[sym]) trendPositionCount++;
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
      circuitBreaker: {
        state: circuitBreaker.peakHalted ? "peak_halt" : circuitBreaker.weeklyHalted ? "weekly_halt" : circuitBreaker.dailyHalted ? "daily_halt" : "ok",
        dailyHalted: circuitBreaker.dailyHalted,
        weeklyHalted: circuitBreaker.weeklyHalted,
        peakHalted: circuitBreaker.peakHalted,
        marketOpenValue: circuitBreaker.marketOpenValue,
        peakValue: circuitBreaker.peakValue,
        peakDate: circuitBreaker.peakDate,
        details: getCircuitBreakerDetails(portfolioValue),
      },
      // Strategy breakdown
      positionStrategy: { ...positionStrategy },
      slotConfig: SLOT_CONFIG,
      strategyStats: {
        ml: { positions: mlPositions, slots: SLOT_CONFIG.ml_medium, trades: { ...mlTradeCount } },
        momentum: { positions: momPositions, slots: SLOT_CONFIG.momentum, trades: { ...momTradeCount } },
        mean_reversion: { positions: mrPositions, slots: SLOT_CONFIG.mean_reversion, trades: { ...mrTradeCount } },
        trend: { positions: trendPositionCount, slots: 0 },
        legacy: { positions: legacyPositions, slots: 0 },
        flex: { used: Math.max(0, mlPositions + momPositions + mrPositions + trendPositionCount + legacyPositions - SLOT_CONFIG.ml_medium - SLOT_CONFIG.momentum - SLOT_CONFIG.mean_reversion), total: SLOT_CONFIG.flex },
      },
      momRankings: { ...momRankings },
    };
  }

  function getActivityFeed(limit = 200) {
    return activityLog.slice(-limit);
  }

  function getCircuitBreakerStatus() {
    const currentValue = portfolioValue || 0;
    const cbResult = checkCircuitBreakers(currentValue);
    return {
      state: circuitBreaker.peakHalted ? "peak_halt" : circuitBreaker.weeklyHalted ? "weekly_halt" : circuitBreaker.dailyHalted ? "daily_halt" : "ok",
      safe: cbResult.safe,
      reason: cbResult.reason,
      layer: cbResult.layer,
      portfolioValue: currentValue,
      layers: getCircuitBreakerDetails(currentValue),
    };
  }

  function resetPeakCircuitBreaker() {
    if (!circuitBreaker.peakHalted) {
      return { success: false, message: "Peak circuit breaker is not currently halted." };
    }
    const oldPeak = circuitBreaker.peakValue;
    circuitBreaker.peakHalted = false;
    circuitBreaker._peakNotified = false;
    circuitBreaker.peakHaltedAt = null;
    circuitBreaker.peakHaltCount = 0; // manual reset clears escalation
    // Reset peak to current value so it doesn't immediately re-trigger
    circuitBreaker.peakValue = portfolioValue || circuitBreaker.peakValue;
    circuitBreaker.peakDate = new Date().toISOString().split("T")[0];
    saveCircuitBreakerState();
    addLog(`[circuit-breaker] Peak halt MANUALLY RESET. Old peak $${oldPeak.toFixed(0)}, new peak $${circuitBreaker.peakValue.toFixed(0)}.`, "system");
    notify.send(`✅ CIRCUIT BREAKER — Peak drawdown halt manually reset. New peak: $${circuitBreaker.peakValue.toFixed(0)}`, { immediate: true });
    return { success: true, message: `Peak halt reset. New peak: $${circuitBreaker.peakValue.toFixed(0)}` };
  }

  // ── Public API ──
  return {
    start,
    stop,
    getState,
    getActivityFeed,
    isRunning: () => running,
    getCircuitBreakerStatus,
    resetPeakCircuitBreaker,
  };
};
