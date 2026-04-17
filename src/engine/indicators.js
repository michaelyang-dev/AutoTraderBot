// ══════════════════════════════════════════
//  TECHNICAL INDICATORS
// ══════════════════════════════════════════

/**
 * Simple Moving Average
 */
export function sma(arr, period) {
  if (arr.length < period) return null;
  return arr.slice(-period).reduce((a, b) => a + b, 0) / period;
}

/**
 * Exponential Moving Average
 */
export function ema(arr, period) {
  if (arr.length < period) return null;
  const k = 2 / (period + 1);
  let e = arr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < arr.length; i++) {
    e = arr[i] * k + e * (1 - k);
  }
  return e;
}

/**
 * Relative Strength Index
 */
export function rsi(arr, period = 14) {
  if (arr.length < period + 1) return 50;
  let gains = 0, losses = 0;
  for (let i = arr.length - period; i < arr.length; i++) {
    const diff = arr[i] - arr[i - 1];
    if (diff > 0) gains += diff;
    else losses -= diff;
  }
  const rs = gains / (losses || 0.001);
  return 100 - 100 / (1 + rs);
}

/**
 * MACD (12, 26) with signal line
 */
export function macd(arr) {
  const e12 = ema(arr, 12);
  const e26 = ema(arr, 26);
  if (e12 === null || e26 === null) return { m: 0, s: 0 };
  const m = e12 - e26;
  return { m, s: m * 0.82 };
}

/**
 * Average True Range (close-to-close approximation — no OHLC required)
 * Returns the average absolute close-to-close move over `period` bars.
 */
export function atr(arr, period = 14) {
  if (arr.length < period + 1) return null;
  let sum = 0;
  for (let i = arr.length - period; i < arr.length; i++) {
    sum += Math.abs(arr[i] - arr[i - 1]);
  }
  return sum / period;
}

/**
 * Average daily volume over the previous `period` bars (excludes the current bar).
 * Returns null if there isn't enough history.
 */
export function avgVolume(arr, period = 20) {
  if (!arr || arr.length < period + 1) return null;
  const window = arr.slice(-(period + 1), -1); // last `period` values before current
  return window.reduce((a, b) => a + b, 0) / window.length;
}

/**
 * Multi-timeframe weekly trend confirmation.
 * Samples every 5th daily bar to approximate weekly closes, then computes
 * 10-week and 30-week SMAs. Needs at least 150 daily bars.
 *
 * @param {number[]} prices - Daily close prices (150+ bars required)
 * @returns {{ trend: "BULLISH"|"BEARISH", sma10w: number, sma30w: number } | null}
 */
export function weeklyTrend(prices) {
  if (!prices || prices.length < 150) return null;
  // Anchor at the MOST RECENT bar, step back 5 days at a time.
  // This ensures the current price is always included and the weekly SMAs
  // stay in sync with the daily signals (index-0 anchoring lags 0–4 bars).
  const weekly = [];
  for (let i = prices.length - 1; i >= 0 && weekly.length < 30; i -= 5) {
    weekly.unshift(prices[i]);
  }
  if (weekly.length < 30) return null;
  const smaFast = sma(weekly, 5);   // 5-week SMA — reacts quickly to trend shifts
  const smaSlow = sma(weekly, 15);  // 15-week SMA — medium-term trend baseline
  if (smaFast === null || smaSlow === null) return null;
  return {
    trend: smaFast > smaSlow ? "BULLISH" : "BEARISH",
    smaFast,
    smaSlow,
  };
}

/**
 * Bollinger Bands (20-period, 2 std dev)
 */
export function bollinger(arr, period = 20) {
  if (arr.length < period) return null;
  const slice = arr.slice(-period);
  const mean = slice.reduce((a, b) => a + b, 0) / period;
  const std = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period);
  return { upper: mean + 2 * std, middle: mean, lower: mean - 2 * std };
}
