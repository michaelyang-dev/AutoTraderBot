// ══════════════════════════════════════════
//  REGIME ENGINE — SPY market trend filter
//  Classifies market conditions so the bot
//  can size and gate positions appropriately.
// ══════════════════════════════════════════

import { sma } from "./indicators";

// Number of consecutive daily closes above the 50-SMA required
// before the bot re-enables buys after a BEARISH period.
export const REGIME_RECOVERY_DAYS = 3;

/**
 * Compute the current market regime from SPY daily closes.
 * Requires 200+ bars; defaults to BULLISH if data is insufficient
 * so startup never silently blocks all trades.
 *
 * Regimes:
 *   BULLISH  — SPY above SMA50 AND SMA200 → trade normally
 *   CAUTIOUS — SPY below SMA50 but above SMA200 → halve sizes, STRONG BUY only
 *   BEARISH  — SPY below SMA50 AND SMA200   → no new buys at all
 *
 * Recovery rule (applied by the caller, not here):
 *   After BEARISH, buys are only re-enabled once `consecutiveDaysAbove50 >= REGIME_RECOVERY_DAYS`.
 *
 * @param {number[]} spyPrices - Daily close prices for SPY (need 200+)
 * @returns {{ regime: "BULLISH"|"CAUTIOUS"|"BEARISH", sma50: number|null, sma200: number|null, consecutiveDaysAbove50: number }}
 */
export function computeRegime(spyPrices) {
  if (!spyPrices || spyPrices.length < 200) {
    return { regime: "BULLISH", sma50: null, sma200: null, consecutiveDaysAbove50: 0 };
  }

  const price    = spyPrices[spyPrices.length - 1];
  const sma50Val  = sma(spyPrices, 50);
  const sma200Val = sma(spyPrices, 200);

  let regime;
  if (price > sma50Val && price > sma200Val) {
    regime = "BULLISH";
  } else if (price > sma200Val) {
    regime = "CAUTIOUS";  // below 50-SMA but long-term trend still intact
  } else {
    regime = "BEARISH";
  }

  // Count consecutive recent bars where close > 50-SMA (capped at 10).
  // Uses the already-computed sma50Val as threshold — the 50-day SMA moves
  // by fractions of a percent per day, so this is accurate enough for a
  // "3 consecutive days" recovery guard. Avoids O(n²) slice+recompute.
  let consecutiveDaysAbove50 = 0;
  for (let i = spyPrices.length - 1; i >= 0 && consecutiveDaysAbove50 < 10; i--) {
    if (spyPrices[i] <= sma50Val) break;
    consecutiveDaysAbove50++;
  }

  return { regime, sma50: sma50Val, sma200: sma200Val, consecutiveDaysAbove50 };
}
