// ══════════════════════════════════════════════════════════════════════
//  REBALANCE RESIZING for the Alpaca PAPER mirror (2026-10-01). Pure — no I/O, no engine state —
//  so it is unit-tested on its own: node server/tests/rebalance_resize.test.js
//
//  On a rebalance day the backtest (clean room: min_trade 0.003 of the book) and IBKR
//  (TRANCHE_MIN_TRADE_PCT = 0.003 of the book) resize EVERY held target name to its weight, in BOTH
//  directions, unless the change is under 0.3% of the book. This mirror used to trim only (> 20% over
//  target and > $5k) and never top up — STEP 2 skips held symbols — so cash freed by exits/trims that the
//  new names did not absorb sat idle: on 2026-10-01 it was 0.83x invested with $246k cash against a 1.49x
//  target, which made the IBKR-vs-Alpaca comparison meaningless.
// ══════════════════════════════════════════════════════════════════════

const REBAL_BAND_PCT = 0.003;   // no-trade band, fraction of the portfolio (= the backtest's min_trade)
const EPS = 1e-6;                // dollars: keeps float noise (sum of ten 0.1s = 0.9999999999999999) from costing a share
const wholeShares = (dollars, price) => Math.max(0, Math.floor((dollars + EPS) / price));

/**
 * Plan the rebalance-day resizes of HELD target names.
 *
 * positions   Alpaca positions ({symbol, qty, market_value, current_price, side})
 * buySignals  today's BUY signals ({symbol, probability}); weight = probability / sum(probability)
 * closed      symbols sold earlier this cycle (not resized, not counted as deployed)
 * skip        symbols held but not managed here (pending buy orders, non-ML tags, SPY): counted as deployed
 *
 * Sells bring overweight names down to target. Buys bring underweight names up to target, never taking the
 * account past leverage x volScale, and always leaving room for the NEW names' targets (bought in STEP 3).
 * Whole shares, floored, like every other order this engine places.
 */
function planRebalanceResizes({ positions, buySignals, portfolioValue, volScale, leverage, maxPositionPct,
                                closed = new Set(), skip = new Set(), band = REBAL_BAND_PCT }) {
  const sells = [];
  const wanted = [];
  const tProb = (buySignals || []).reduce((s, x) => s + (Number(x.probability) || 0), 0);
  if (!(tProb > 0) || !(portfolioValue > 0) || !(volScale > 0) || !(leverage > 0)) {
    return { sells, buys: [], deployable: 0, deployed: 0, reserve: 0, room: 0, short: [] };
  }
  const targetPctOf = (sig) => Math.min((Number(sig.probability) / tProb) * leverage * volScale, maxPositionPct);
  const bySym = new Map(buySignals.map(s => [s.symbol, s]));
  const deployable = portfolioValue * leverage * volScale;
  const held = new Set();
  let deployed = 0;
  for (const pos of positions || []) {
    const sym = pos.symbol;
    if (closed.has(sym)) continue;
    held.add(sym);
    const mv = Math.abs(parseFloat(pos.market_value || 0)) || 0;
    deployed += mv;
    if (skip.has(sym)) continue;
    if (parseFloat(pos.qty) < 0 || pos.side === "short") continue;
    const sig = bySym.get(sym);
    if (!sig) continue;                                   // not a target any more: STEP 1f sells it
    const price = parseFloat(pos.current_price);
    if (!(price > 0)) continue;
    const targetPct = targetPctOf(sig);
    const targetVal = portfolioValue * targetPct;
    const gap = targetVal - mv;
    if (Math.abs(gap) < portfolioValue * band) continue;  // inside the no-trade band
    const qty = wholeShares(Math.abs(gap), price);
    if (qty <= 0) continue;
    const order = { symbol: sym, qty, price, targetPct, targetVal, currentVal: mv };
    if (gap > 0) wanted.push(order); else sells.push(order);
  }
  // Room for top-ups: what the sells free, minus what the NEW names (BUY signals not held) will take in STEP 3.
  let reserve = 0;
  for (const sig of buySignals) {
    if (!held.has(sig.symbol) && !skip.has(sig.symbol)) reserve += portfolioValue * targetPctOf(sig);
  }
  const freed = sells.reduce((s, o) => s + o.qty * o.price, 0);
  let room = deployable - deployed + freed - reserve;
  const roomStart = room;
  wanted.sort((a, b) => (b.targetVal - b.currentVal) - (a.targetVal - a.currentVal));   // largest shortfall first
  const buys = [];
  const short = [];
  for (const o of wanted) {
    const q = Math.min(o.qty, wholeShares(Math.max(0, room), o.price));
    if (q > 0) {
      buys.push(q === o.qty ? o : { ...o, qty: q });
      room -= q * o.price;
    }
    if (q < o.qty) short.push({ symbol: o.symbol, wanted: o.qty, got: q });
  }
  return { sells, buys, deployable, deployed, reserve, room: roomStart, short };
}

module.exports = { REBAL_BAND_PCT, planRebalanceResizes };
