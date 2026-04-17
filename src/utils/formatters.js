// ══════════════════════════════════════════
//  UTILITY FUNCTIONS
// ══════════════════════════════════════════

/**
 * Format number as currency string: $1,234.56
 */
export function fmt(n) {
  return "$" + (n ?? 0).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/**
 * Format number as percentage: +12.34%
 */
export function fmtPct(n) {
  return (n >= 0 ? "+" : "") + (n * 100).toFixed(2) + "%";
}

/**
 * Calculate total portfolio value (cash + positions).
 */
export function calcPortfolioValue(cash, positions, priceHist, trendPositions = {}) {
  const posVal = Object.entries(positions).reduce((sum, [sym, pos]) => {
    const prices = priceHist[sym];
    return sum + (prices ? prices[prices.length - 1] * pos.shares : 0);
  }, 0);
  const trendVal = Object.entries(trendPositions).reduce((sum, [sym, pos]) => {
    const prices = priceHist[sym];
    return sum + (prices ? prices[prices.length - 1] * pos.shares : 0);
  }, 0);
  return cash + posVal + trendVal;
}

/**
 * Get enriched position entries with current price, P&L, P&L %.
 */
export function getPositionEntries(positions, priceHist) {
  return Object.entries(positions)
    .map(([sym, p]) => {
      const prices = priceHist[sym];
      const curr = prices ? prices[prices.length - 1] : p.avgPrice;
      const pnl = (curr - p.avgPrice) * p.shares;
      const pnlPct = (curr - p.avgPrice) / p.avgPrice;
      return { sym, ...p, curr, pnl, pnlPct };
    })
    .sort((a, b) => b.pnl - a.pnl);
}

/**
 * Build allocation data for pie chart.
 */
export function getAllocationData(positions, priceHist, cash) {
  const data = Object.entries(positions).map(([sym, p]) => {
    const prices = priceHist[sym];
    const curr = prices ? prices[prices.length - 1] : p.avgPrice;
    return { name: sym, value: curr * p.shares };
  });
  if (cash > 0) data.push({ name: "Cash", value: cash });
  return data;
}
