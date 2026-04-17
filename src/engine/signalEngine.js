// ══════════════════════════════════════════
//  SIGNAL ENGINE — Multi-strategy consensus
// ══════════════════════════════════════════

import { sma, rsi, macd, bollinger } from "./indicators";
import { CONSENSUS_THRESHOLDS } from "../config/constants";

/**
 * Analyze a single stock's price history across all 5 strategies.
 * Returns a consensus rating, individual signals, a numeric score,
 * and raw indicator values.
 *
 * @param {number[]} prices - Price history array
 * @returns {{ consensus: string, signals: object, score: number, indicators: object }}
 */
export function getSignals(prices) {
  if (prices.length < 35) {
    return { consensus: "WAIT", signals: {}, score: 0, indicators: {} };
  }

  const signals = {};
  let buyVotes = 0;
  let sellVotes = 0;

  // ── SMA Crossover ──
  const s10 = sma(prices, 10);
  const s30 = sma(prices, 30);
  const ps10 = sma(prices.slice(0, -1), 10);
  const ps30 = sma(prices.slice(0, -1), 30);

  if (s10 > s30 && ps10 <= ps30) {
    signals.sma = "BUY";
    buyVotes++;
  } else if (s10 < s30 && ps10 >= ps30) {
    signals.sma = "SELL";
    sellVotes++;
  } else if (s10 > s30) {
    signals.sma = "BULLISH";
    buyVotes += 0.3;
  } else {
    signals.sma = "BEARISH";
    sellVotes += 0.3;
  }

  // ── RSI ──
  const rsiVal = rsi(prices);
  if (rsiVal < 28) {
    signals.rsi = "BUY";
    buyVotes++;
  } else if (rsiVal > 72) {
    signals.rsi = "SELL";
    sellVotes++;
  } else if (rsiVal < 40) {
    signals.rsi = "BULLISH";
    buyVotes += 0.3;
  } else if (rsiVal > 60) {
    signals.rsi = "BEARISH";
    sellVotes += 0.3;
  } else {
    signals.rsi = "NEUTRAL";
  }

  // ── MACD ──
  const mc = macd(prices);
  const pmc = macd(prices.slice(0, -1));
  if (mc.m > mc.s && pmc.m <= pmc.s) {
    signals.macd = "BUY";
    buyVotes++;
  } else if (mc.m < mc.s && pmc.m >= pmc.s) {
    signals.macd = "SELL";
    sellVotes++;
  } else if (mc.m > mc.s) {
    signals.macd = "BULLISH";
    buyVotes += 0.3;
  } else {
    signals.macd = "BEARISH";
    sellVotes += 0.3;
  }

  // ── Bollinger Bands ──
  const bb = bollinger(prices);
  const currentPrice = prices[prices.length - 1];
  if (bb) {
    if (currentPrice <= bb.lower) {
      signals.boll = "BUY";
      buyVotes++;
    } else if (currentPrice >= bb.upper) {
      signals.boll = "SELL";
      sellVotes++;
    } else if (currentPrice < bb.middle) {
      signals.boll = "BULLISH";
      buyVotes += 0.2;
    } else {
      signals.boll = "BEARISH";
      sellVotes += 0.2;
    }
  }

  // ── Momentum ──
  const momVal =
    prices.length > 12
      ? (currentPrice - prices[prices.length - 13]) / prices[prices.length - 13]
      : 0;

  if (momVal > 0.035) {
    signals.mom = "BUY";
    buyVotes++;
  } else if (momVal < -0.025) {
    signals.mom = "SELL";
    sellVotes++;
  } else if (momVal > 0) {
    signals.mom = "BULLISH";
    buyVotes += 0.2;
  } else {
    signals.mom = "BEARISH";
    sellVotes += 0.2;
  }

  // ── Consensus ──
  const score = buyVotes - sellVotes;
  let consensus = "HOLD";
  if (score >= CONSENSUS_THRESHOLDS.STRONG_BUY)  consensus = "STRONG BUY";
  else if (score >= CONSENSUS_THRESHOLDS.BUY)     consensus = "BUY";
  else if (score <= CONSENSUS_THRESHOLDS.STRONG_SELL) consensus = "STRONG SELL";
  else if (score <= CONSENSUS_THRESHOLDS.SELL)    consensus = "SELL";

  return {
    consensus,
    signals,
    score,
    indicators: {
      rsi: rsiVal,
      sma10: s10,
      sma30: s30,
      macd: mc,
      momentum: momVal,
      bollinger: bb,
    },
  };
}
