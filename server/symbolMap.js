// ═════���════════════════════════════════════════════════════════════════
//  Symbol Mapping — Alpaca ↔ S&P 500 format
//
//  S&P 500 lists use hyphens (BF-B) but Alpaca API requires dots (BF.B).
//  Single source of truth for all symbol mapping across the codebase.
// ══════════════��═══════════════════════════════���═══════════════════════

const ALPACA_SYMBOL_MAP = {
  "BF-B": "BF.B",
  "BRK-B": "BRK.B",
  "BRK-A": "BRK.A",
};

const REVERSE_SYMBOL_MAP = Object.fromEntries(
  Object.entries(ALPACA_SYMBOL_MAP).map(([k, v]) => [v, k])
);

function toAlpacaSymbol(sym) { return ALPACA_SYMBOL_MAP[sym] || sym; }
function fromAlpacaSymbol(sym) { return REVERSE_SYMBOL_MAP[sym] || sym; }

module.exports = { ALPACA_SYMBOL_MAP, REVERSE_SYMBOL_MAP, toAlpacaSymbol, fromAlpacaSymbol };
