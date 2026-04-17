// ══════════════════════════════════════════
//  STOCK SCREENER — Filters the market down
//  to the best tradeable candidates each day
// ══════════════════════════════════════════

import * as alpaca from "./alpacaClient";
import { getSector } from "../config/constants";

// Screener filters — tweak these as needed
const MIN_PRICE = 10;        // skip penny stocks
const MAX_PRICE = 1500;      // skip ultra-expensive stocks
const MIN_VOLUME = 500000;   // minimum daily volume
const MIN_TRADE_COUNT = 1000; // minimum number of trades

/**
 * Fetch the most active stocks from Alpaca and filter them.
 * Returns an array of { sym, base, sector } objects
 * that can replace the UNIVERSE array.
 */
export async function screenStocks() {
  try {
    const response = await fetch("/api/screener");
    if (!response.ok) throw new Error("Screener API failed");
    const stocks = await response.json();
    // Enrich each stock with its sector (UNIVERSE → SECTOR_MAP → "Other")
    return stocks.map((s) => ({ ...s, sector: getSector(s.sym) }));
  } catch (err) {
    console.error("Screener error:", err.message);
    return null; // return null so caller knows to use fallback
  }
}