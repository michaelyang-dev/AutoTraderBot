// ══════════════════════════════════════════
//  TREND ENGINE — Long-term trend detection
//  Identifies strong uptrends using 50/200-SMA analysis
// ══════════════════════════════════════════

import { sma } from "./indicators";

/**
 * Compute trend status for a price series.
 *
 * Strong uptrend = ALL of:
 *   1. Current price > 200-day SMA
 *   2. 50-day SMA > 200-day SMA (golden-cross zone)
 *   3. At least 30 of the last 40 bars closed above their own 200-day SMA
 *
 * Requires at least 200 bars.
 *
 * @param {number[]} prices - Chronological close prices
 * @returns {{ isStrongUptrend, price, sma50, sma200, daysAbove200, priceAbove200 } | null}
 */
export function computeTrendStatus(prices) {
  if (!prices || prices.length < 200) return null;

  const price   = prices[prices.length - 1];
  const sma50   = sma(prices, 50);
  const sma200v = sma(prices, 200);
  if (sma50 === null || sma200v === null) return null;

  const priceAbove200 = price > sma200v;

  // Rolling 200-SMA presence check over the last 40 bars.
  // For each of the last 40 bars we need at least 200 prior bars, so the
  // earliest bar we can check is prices[199].  If the series is shorter than
  // 239 bars we check however many bars are available.
  const lookback = Math.min(40, prices.length - 199);
  let daysAbove200 = 0;
  for (let i = 0; i < lookback; i++) {
    const endIdx  = prices.length - i; // exclusive slice end → last bar is prices[endIdx-1]
    const barPrice = prices[endIdx - 1];
    // Compute the 200-SMA at this bar without creating a slice array
    let barSma200 = 0;
    for (let j = endIdx - 200; j < endIdx; j++) barSma200 += prices[j];
    barSma200 /= 200;
    if (barPrice > barSma200) daysAbove200++;
  }

  const isStrongUptrend = priceAbove200 && sma50 > sma200v && daysAbove200 >= 30;

  return {
    isStrongUptrend,
    price,
    sma50,
    sma200: sma200v,
    daysAbove200,
    priceAbove200,
  };
}
