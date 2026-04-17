// ══════════════════════════════════════════
//  PRICE ENGINE — Simulated market prices
// ══════════════════════════════════════════

/**
 * Generate a simulated stock price with trend, volatility, and random jumps.
 * Each symbol gets a unique price path via a hash seed.
 */
export function simPrice(base, tick, sym) {
  const hash = sym.charCodeAt(0) + sym.charCodeAt(1) * 7;
  const trend = Math.sin(tick * 0.015 + hash) * base * 0.09;
  const vol   = Math.sin(tick * 0.04 + hash * 2) * base * 0.05;
  const noise = (Math.random() - 0.48) * base * 0.025;
  const jump  = Math.random() < 0.02 ? (Math.random() - 0.5) * base * 0.06 : 0;
  return Math.max(base * 0.6, base + trend + vol + noise + jump);
}

/**
 * Generate a simulated daily volume for a stock.
 * Base volume scales inversely with price (~$100 stock ≈ 2M shares/day).
 * Includes a sine-wave trend, random noise, and occasional volume spikes.
 */
export function simVolume(base, tick, sym) {
  const hash = sym.charCodeAt(0) * 11 + (sym.charCodeAt(sym.length - 1) || 0) * 7;
  const baseVol = Math.round(200_000_000 / base); // $100 → 2M, $500 → 400K
  const drift = Math.sin(tick * 0.025 + hash * 0.1) * baseVol * 0.4;
  const noise = (Math.random() - 0.45) * baseVol * 0.6; // slight positive skew
  const spike = Math.random() < 0.15 ? baseVol * (1.0 + Math.random()) : 0; // 15% chance of spike
  return Math.max(10_000, Math.round(baseVol + drift + noise + spike));
}

/**
 * Initialize volume history with 25 pre-seeded ticks so the 20-day avg
 * is available from the very first trade cycle.
 */
export function initVolumeHistory(universe) {
  const hist = {};
  universe.forEach((s) => {
    const vols = [];
    for (let i = -25; i <= 0; i++) {
      vols.push(simVolume(s.base, i, s.sym));
    }
    hist[s.sym] = vols;
  });
  return hist;
}

/**
 * Advance all volumes by one tick.
 */
export function advanceVolumes(prevHist, universe, tick) {
  const next = {};
  universe.forEach((s) => {
    next[s.sym] = [...(prevHist[s.sym] || []).slice(-199), simVolume(s.base, tick, s.sym)];
  });
  return next;
}

/**
 * Initialize price history with 150 pre-seeded bars (ticks -149 to 0)
 * so weekly-trend SMAs are available from the very first trade cycle.
 */
export function initPriceHistory(universe) {
  const hist = {};
  universe.forEach((s) => {
    const prices = [];
    for (let i = -149; i <= 0; i++) {
      prices.push(simPrice(s.base, i, s.sym));
    }
    hist[s.sym] = prices;
  });
  return hist;
}

/**
 * Advance all prices by one tick.
 * Returns a new price history object (immutable update).
 */
export function advancePrices(prevHist, universe, tick) {
  const next = {};
  universe.forEach(s => {
    next[s.sym] = [...(prevHist[s.sym] || []).slice(-199), simPrice(s.base, tick, s.sym)];
  });
  return next;
}
