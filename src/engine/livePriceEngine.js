// ══════════════════════════════════════════
//  LIVE PRICE ENGINE — Fetches real market data from Alpaca
//  Replaces priceEngine.js for live trading
// ══════════════════════════════════════════

import * as alpaca from "./alpacaClient";
import { UNIVERSE } from "../config/constants";

/**
 * Initialize price history by fetching historical bars from Alpaca.
 * Gets 60 days of daily bars so indicators can calculate immediately.
 *
 * @returns {Promise<{[sym]: number[]}>}
 */
export async function initLivePriceHistory(universe = null) {
  const hist = {};
  const volHist = {};
  const stockList = universe || UNIVERSE;
  const symbols = stockList.map((s) => s.sym);

  // Fetch bars for all universe symbols in parallel
  // 150 bars needed: 60 for daily indicators + 150 for weekly SMA sampling (every 5th bar → 30 weekly bars)
  const results = await Promise.allSettled(
    symbols.map(async (sym) => {
      const bars = await alpaca.getBars(sym, "1Day", 150);
      return { sym, closes: bars.map((b) => b.c), volumes: bars.map((b) => b.v) };
    })
  );

  results.forEach((result) => {
    if (result.status === "fulfilled") {
      const { sym, closes, volumes } = result.value;
      hist[sym] = closes;
      volHist[sym] = volumes;
      if (closes.length < 35) {
        console.warn(`⚠️ ${sym}: only ${closes.length} bars loaded (need 35 for indicators)`);
      }
    } else {
      console.error("Failed to fetch bars:", result.reason);
    }
  });

  // Summary log for debugging
  const barCounts = Object.entries(hist).map(([sym, arr]) => `${sym}:${arr.length}`);
  console.log(`📊 Price history loaded — ${barCounts.length} symbols: ${barCounts.join(", ")}`);

  // Always fetch SPY with 220 bars — SMA200 needs 200+ closes for the regime filter.
  // Do this separately so universe stocks aren't penalised with extra fetches.
  if (!symbols.includes("SPY")) {
    try {
      const spyBars = await alpaca.getBars("SPY", "1Day", 220);
      hist.SPY = spyBars.map((b) => b.c);
    } catch (err) {
      console.error("Failed to fetch SPY bars for regime filter:", err.message);
    }
  }

  return { hist, volHist };
}

/**
 * Update price history with latest snapshots from Alpaca.
 * Appends the latest price to each symbol's array.
 *
 * @param {object} prevHist - Previous price history
 * @returns {Promise<{[sym]: number[]}>}
 */
export async function updateLivePrices(prevHist, prevVolHist = {}, universe = null) {
  try {
    const stockList = universe || UNIVERSE;
    const symbols = stockList.map((s) => s.sym);

    // Include SPY in the snapshot batch so regime stays up-to-date each poll
    const snapshotSymbols = symbols.includes("SPY") ? symbols : [...symbols, "SPY"];
    const snapshots = await alpaca.getSnapshots(snapshotSymbols);

    const next = {};
    const nextVol = {};
    symbols.forEach((sym) => {
      const prev = prevHist[sym] || [];
      const prevVols = prevVolHist[sym] || [];
      const snap = snapshots[sym];
      if (snap && snap.price > 0) {
        next[sym] = [...prev.slice(-100), snap.price];          // keep last 100 prices
        nextVol[sym] = [...prevVols.slice(-100), snap.volume ?? 0]; // today's intraday volume
      } else {
        next[sym] = prev;
        nextVol[sym] = prevVols;
      }
    });

    // Keep SPY history (250 entries so SMA200 always has enough data)
    const spyPrev = prevHist.SPY || [];
    const spySnap = snapshots.SPY;
    next.SPY = spySnap && spySnap.price > 0
      ? [...spyPrev.slice(-250), spySnap.price]
      : spyPrev;

    return { hist: next, volHist: nextVol };
  } catch (err) {
    console.error("Price update error:", err.message);
    return { hist: prevHist, volHist: prevVolHist }; // return unchanged on error
  }
}
