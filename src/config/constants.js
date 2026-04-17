// ══════════════════════════════════════════
//  CONFIG — All tunable parameters in one place
// ══════════════════════════════════════════

export const INITIAL_CASH = 100000;

// Risk Management
export const RISK = {
  MAX_POSITION_PCT: 0.15,       // max 15% of portfolio per stock
  STOP_LOSS_PCT: -0.08,         // -8% fixed stop loss (used when USE_TRAILING_STOP is false)
  TAKE_PROFIT_PCT: 0.15,        // +15% take profit
  MAX_OPEN_POSITIONS: 6,        // max concurrent positions
  MAX_CASH_DEPLOY_PCT: 0.90,    // deploy max 90% of available cash per trade
  REBALANCE_INTERVAL: 5,        // check signals every N ticks
  TRAILING_STOP_PCT: 0.08,      // trail 8% below peak price
  USE_TRAILING_STOP: true,      // true = trailing stop, false = fixed stop-loss
  ATR_TARGET_PCT: 0.01,         // reference volatility (1%) — stocks above this get scaled down
  MIN_POSITION_PCT: 0.03,       // floor: never allocate less than 3% of portfolio
  LOSS_COOLDOWN_CYCLES: 3,      // trade cycles a stock is blocked after a losing exit
  // Per-sector position caps — only sectors listed here are capped; unlisted sectors are uncapped.
  // Individual stocks (Tech, Finance, etc.) have no limit so the bot fills all 6 slots freely.
  SECTOR_MAX_POSITIONS: {
    International: 2,  // EWZ, EWJ, FXI, INDA, EFA, EEM
    Commodity:     2,  // GLD, SLV, USO, DBC
    Bond:          2,  // TLT, HYG, LQD
    Volatility:    1,  // VIXY — high-risk, cap at 1
  },
  VOLUME_CONFIRM_RATIO: 1.5,   // current volume must be ≥ 1.5x the 20-day average to buy
};

// Market hours trading windows (live mode only)
export const MARKET_HOURS = {
  OPEN_BUFFER_MINS: 15,         // skip trading in first 15 min after open (opening volatility)
  CLOSE_BUFFER_MINS: 30,        // block new buys in last 30 min before close (overnight risk)
};

// Blacklist — leveraged, inverse, and volatility-decay products unsuitable for swing trading
export const NEVER_BUY = new Set([
  "VIXY","UVXY","VXX","SVXY",         // volatility
  "TQQQ","SQQQ","QQQ3",               // leveraged Nasdaq
  "SPXU","SPXS","SDS","UPRO",         // leveraged S&P
  "QID","SDOW",                        // leveraged Dow / Nasdaq inverse
  "LABU","LABD",                       // leveraged biotech
  "JNUG","JDST","NUGT","DUST",        // leveraged gold miners
  "FNGU","FNGD",                       // leveraged FAANG
  "SOXL","SOXS",                       // leveraged semis
  "YANG","YINN",                       // leveraged China
]);

// Stock Universe — 37 symbols: 12 individual stocks + 25 ETFs
export const UNIVERSE = [
  // ── Individual Stocks (12) ──
  { sym: "AAPL",  base: 189, sector: "Tech"     },
  { sym: "GOOGL", base: 141, sector: "Tech"     },
  { sym: "MSFT",  base: 378, sector: "Tech"     },
  { sym: "AMZN",  base: 178, sector: "Consumer" },
  { sym: "TSLA",  base: 248, sector: "Auto"     },
  { sym: "NVDA",  base: 880, sector: "Semis"    },
  { sym: "META",  base: 505, sector: "Tech"     },
  { sym: "NFLX",  base: 628, sector: "Media"    },
  { sym: "AMD",   base: 164, sector: "Semis"    },
  { sym: "JPM",   base: 196, sector: "Finance"  },
  { sym: "V",     base: 278, sector: "Finance"  },
  { sym: "UNH",   base: 527, sector: "Health"   },

  // ── Sector ETFs (SPDR, 11) — uncapped, one per sector ──
  { sym: "XLE",  base: 93,  sector: "Energy"     },
  { sym: "XLF",  base: 48,  sector: "Finance"    },
  { sym: "XLV",  base: 145, sector: "Health"     },
  { sym: "XLI",  base: 130, sector: "Industrial" },
  { sym: "XLK",  base: 218, sector: "Tech"       },
  { sym: "XLY",  base: 195, sector: "Consumer"   },
  { sym: "XLP",  base: 79,  sector: "Staples"    },
  { sym: "XLU",  base: 72,  sector: "Utilities"  },
  { sym: "XLRE", base: 39,  sector: "REIT"       },
  { sym: "XLB",  base: 85,  sector: "Materials"  },
  { sym: "XLC",  base: 92,  sector: "Media"      },

  // ── International ETFs (6) — capped at 2 positions ──
  { sym: "EWZ",  base: 29,  sector: "International" },  // Brazil
  { sym: "EWJ",  base: 71,  sector: "International" },  // Japan
  { sym: "FXI",  base: 29,  sector: "International" },  // China
  { sym: "INDA", base: 50,  sector: "International" },  // India
  { sym: "EFA",  base: 79,  sector: "International" },  // Developed ex-US
  { sym: "EEM",  base: 43,  sector: "International" },  // Emerging Markets

  // ── Commodities (4) — capped at 2 positions ──
  { sym: "GLD",  base: 285, sector: "Commodity" },  // Gold
  { sym: "SLV",  base: 31,  sector: "Commodity" },  // Silver
  { sym: "USO",  base: 70,  sector: "Commodity" },  // Oil
  { sym: "DBC",  base: 22,  sector: "Commodity" },  // Broad Commodities

  // ── Bonds (3) — capped at 2 positions ──
  { sym: "TLT",  base: 85,  sector: "Bond" },  // 20+ Year Treasury
  { sym: "HYG",  base: 77,  sector: "Bond" },  // High Yield Corporate
  { sym: "LQD",  base: 104, sector: "Bond" },  // Investment Grade Corporate

  // ── Volatility (1) — HIGH RISK, capped at 1 position ──
  { sym: "VIXY", base: 14,  sector: "Volatility" },  // Short-term VIX futures
];

// Strategy definitions
export const STRATEGIES = {
  sma:  { name: "SMA Cross",    icon: "📊", color: "#10b981", desc: "Buy when 10-SMA crosses above 30-SMA" },
  rsi:  { name: "RSI Mean-Rev", icon: "📉", color: "#f59e0b", desc: "Buy when RSI < 28, sell when RSI > 72" },
  macd: { name: "MACD Trend",   icon: "📈", color: "#8b5cf6", desc: "Buy/sell on MACD-signal crossovers" },
  boll: { name: "Bollinger",    icon: "🎯", color: "#ec4899", desc: "Buy at lower band, sell at upper band" },
  mom:  { name: "Momentum",     icon: "🚀", color: "#06b6d4", desc: "Buy on strong uptrend, sell when fading" },
};

// Signal thresholds for consensus
export const CONSENSUS_THRESHOLDS = {
  STRONG_BUY: 2,
  BUY: 1,
  SELL: -1,
  STRONG_SELL: -2,
};

// Speed presets
export const SPEED_OPTIONS = [
  { label: "Slow",   value: 1500 },
  { label: "Normal", value: 800 },
  { label: "Fast",   value: 400 },
  { label: "Turbo",  value: 150 },
];

// Sector map for screener-discovered stocks (supplements UNIVERSE sectors)
export const SECTOR_MAP = {
  // Tech
  AAPL: "Tech", MSFT: "Tech", GOOGL: "Tech", GOOG: "Tech", META: "Tech",
  NVDA: "Semis", AMD: "Semis", INTC: "Semis", QCOM: "Semis", AVGO: "Semis", MU: "Semis", TSM: "Semis",
  CRM: "Tech", ORCL: "Tech", SAP: "Tech", ADBE: "Tech", NOW: "Tech", SNOW: "Tech",
  PLTR: "Tech", UBER: "Tech", LYFT: "Tech", SHOP: "Tech", TWLO: "Tech", ZM: "Tech",
  NET: "Tech", DDOG: "Tech", MDB: "Tech", CRWD: "Tech", ZS: "Tech", OKTA: "Tech",
  PANW: "Tech", FTNT: "Tech", CYBR: "Tech",
  // Consumer
  AMZN: "Consumer", TSLA: "Auto", GM: "Auto", F: "Auto",
  WMT: "Consumer", TGT: "Consumer", COST: "Consumer", HD: "Consumer", LOW: "Consumer",
  NKE: "Consumer", SBUX: "Consumer", MCD: "Consumer", YUM: "Consumer", CMG: "Consumer",
  BABA: "Consumer", JD: "Consumer", PDD: "Consumer",
  // Finance
  JPM: "Finance", BAC: "Finance", WFC: "Finance", GS: "Finance", MS: "Finance",
  C: "Finance", BLK: "Finance", AXP: "Finance", V: "Finance", MA: "Finance",
  PYPL: "Finance", SQ: "Finance", COIN: "Finance", SCHW: "Finance", USB: "Finance",
  // Health
  UNH: "Health", JNJ: "Health", PFE: "Health", ABBV: "Health", MRK: "Health",
  LLY: "Health", BMY: "Health", AMGN: "Health", GILD: "Health", BIIB: "Health",
  MRNA: "Health", BNTX: "Health", CVS: "Health", CI: "Health", HUM: "Health",
  MDT: "Health", ABT: "Health", TMO: "Health", DHR: "Health", ISRG: "Health",
  // Media / Comms
  NFLX: "Media", DIS: "Media", PARA: "Media", WBD: "Media", CMCSA: "Media",
  T: "Media", VZ: "Media", TMUS: "Media", GOOGL: "Tech",
  SPOT: "Media", SNAP: "Media", PINS: "Media", RDDT: "Media",
  // Energy
  XOM: "Energy", CVX: "Energy", COP: "Energy", SLB: "Energy", EOG: "Energy",
  OXY: "Energy", PSX: "Energy", VLO: "Energy", MPC: "Energy",
  // Industrials
  BA: "Industrial", CAT: "Industrial", GE: "Industrial", HON: "Industrial",
  LMT: "Industrial", RTX: "Industrial", NOC: "Industrial", DE: "Industrial",
  UPS: "Industrial", FDX: "Industrial", CSX: "Industrial",
  // Real Estate / Utilities
  AMT: "REIT", PLD: "REIT", EQIX: "REIT", SPG: "REIT",
  NEE: "Utilities", SO: "Utilities", DUK: "Utilities",
};

/**
 * Returns the sector for a given symbol.
 * Checks UNIVERSE first (authoritative), then SECTOR_MAP, defaults to "Other".
 */
export function getSector(sym) {
  const universeEntry = UNIVERSE.find((u) => u.sym === sym);
  if (universeEntry) return universeEntry.sector;
  return SECTOR_MAP[sym] ?? "Other";
}

// Chart colors for pie/allocations
export const PIE_COLORS = [
  "#10b981", "#06b6d4", "#8b5cf6",
  "#f59e0b", "#ec4899", "#ef4444", "#334155",
];
