// Unit tests for server/rebalanceResize.js (the Alpaca paper mirror's rebalance-day resizing).
// Run: node server/tests/rebalance_resize.test.js   (no framework; exits non-zero on the first failure)
const assert = require("assert");
const { planRebalanceResizes, REBAL_BAND_PCT } = require("../rebalanceResize");

const base = { portfolioValue: 1_000_000, volScale: 1.0, leverage: 1.49, maxPositionPct: 0.15 };
const sigs = (n, p = 0.1) => Array.from({ length: n }, (_, i) => ({ symbol: `S${i}`, probability: p }));
const pos = (symbol, mv, price = 100, extra = {}) => ({ symbol, qty: mv / price, market_value: mv, current_price: price, side: "long", ...extra });
let n = 0;
function test(name, fn) { fn(); n++; console.log(`  ok  ${name}`); }

test("band is the backtest's min_trade (0.3%)", () => assert.strictEqual(REBAL_BAND_PCT, 0.003));

test("under-target held name is topped up to target (whole shares, floored)", () => {
  // 10 equal names: target 0.1 x 1.49 = 14.9% = $149,000 each. S0 holds $100,000 -> needs $49,000 = 490 sh @ $100.
  // The 9 unheld names reserve 9 x $149,000, so the room is exactly $49,000.
  const r = planRebalanceResizes({ ...base, positions: [pos("S0", 100_000)], buySignals: sigs(10) });
  assert.deepStrictEqual(r.sells, []);
  assert.strictEqual(r.buys.length, 1);
  assert.strictEqual(r.buys[0].symbol, "S0");
  assert.strictEqual(r.buys[0].qty, 490);
  assert.ok(Math.abs(r.reserve - 9 * 149_000) < 1e-6);
});

test("over-target name is trimmed; names inside the band are left alone", () => {
  const r = planRebalanceResizes({ ...base, buySignals: sigs(10),
    positions: [pos("S0", 160_000), pos("S1", 150_000), pos("S2", 147_000)] });   // gaps -11k, -1k, +2k (band $3k)
  assert.deepStrictEqual(r.sells.map(o => [o.symbol, o.qty]), [["S0", 110]]);
  assert.deepStrictEqual(r.buys, []);
});

test("a held name that is no longer a BUY is not touched here (STEP 1f sells it)", () => {
  const r = planRebalanceResizes({ ...base, buySignals: sigs(10), positions: [pos("XYZ", 10_000)] });
  assert.deepStrictEqual(r.sells, []);
  assert.deepStrictEqual(r.buys, []);
  assert.strictEqual(r.deployed, 10_000);      // still counted as deployed until it is sold
});

test("closed / skipped / short positions are never resized; skipped ones still count as deployed", () => {
  const r = planRebalanceResizes({ ...base, buySignals: sigs(10),
    positions: [pos("S0", 10_000), pos("S1", 10_000), pos("S2", -10_000, 100, { qty: -100, side: "short" })],
    closed: new Set(["S0"]), skip: new Set(["S1"]) });
  assert.deepStrictEqual(r.sells, []);
  assert.deepStrictEqual(r.buys, []);
  assert.strictEqual(r.deployed, 20_000);      // S1 + |S2|; S0 was sold this cycle
});

test("targets are capped at 15% of the portfolio", () => {
  const r = planRebalanceResizes({ ...base, positions: [pos("A", 100_000)],
    buySignals: [{ symbol: "A", probability: 0.9 }, { symbol: "B", probability: 0.1 }] });
  // A: 0.9 x 1.49 = 134% -> capped 15% = $150,000; B (unheld) reserves 14.9% = $149,000; room 1.49M - 100k - 149k.
  assert.strictEqual(r.buys.length, 1);
  assert.strictEqual(r.buys[0].qty, 500);
  assert.ok(Math.abs(r.buys[0].targetPct - 0.15) < 1e-12);
});

test("top-ups never crowd out the new names, and take the largest shortfall first", () => {
  // vol-scale 0.5 -> deployable $745,000; 10 names at 7.45% = $74,500 each. Nine held at $50,000; S9 is new.
  // A $100,000 SPY park (skip) eats room: 745k - (450k + 100k) - 74.5k reserve = $120,500 for 9 x $24,500 wanted.
  const positions = [...Array.from({ length: 9 }, (_, i) => pos(`S${i}`, i === 0 ? 40_000 : 50_000)), pos("SPY", 100_000)];
  const r = planRebalanceResizes({ ...base, volScale: 0.5, positions, buySignals: sigs(10), skip: new Set(["SPY"]) });
  const spent = r.buys.reduce((s, o) => s + o.qty * o.price, 0);
  assert.ok(spent <= r.room + 1e-5, `spent ${spent} > room ${r.room}`);
  assert.ok(Math.abs(r.room - (745_000 - 440_000 - 100_000 - 74_500)) < 1e-6, String(r.room));
  assert.strictEqual(r.buys[0].symbol, "S0");             // its shortfall ($34,500) is the largest
  assert.strictEqual(r.buys[0].qty, 345);
  assert.ok(r.short.length > 0);                           // the rest were cut to fit
});

test("degenerate inputs plan nothing", () => {
  for (const extra of [{ buySignals: [] }, { buySignals: [{ symbol: "A", probability: 0 }] },
                       { buySignals: sigs(3), portfolioValue: 0 }, { buySignals: sigs(3), volScale: 0 }]) {
    const r = planRebalanceResizes({ ...base, positions: [pos("S0", 1_000)], ...extra });
    assert.deepStrictEqual([r.sells, r.buys], [[], []]);
  }
});

test("randomized invariants (5,000 portfolios)", () => {
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  for (let k = 0; k < 5000; k++) {
    const nv = 50_000 + rnd() * 2_000_000;
    const vs = 0.3 + rnd() * 0.7;
    const nSig = 1 + Math.floor(rnd() * 28);
    const buySignals = Array.from({ length: nSig }, (_, i) => ({ symbol: `S${i}`, probability: 0.01 + rnd() }));
    const positions = [];
    for (let i = 0; i < nSig + 5; i++) {
      if (rnd() < 0.6) positions.push(pos(`S${i}`, rnd() * nv * 0.2, 1 + rnd() * 2000));
    }
    const closed = new Set(positions.filter(() => rnd() < 0.1).map(p => p.symbol));
    const r = planRebalanceResizes({ positions, buySignals, portfolioValue: nv, volScale: vs, leverage: 1.49,
                                     maxPositionPct: 0.15, closed });
    const tProb = buySignals.reduce((s, x) => s + x.probability, 0);
    const tgt = s => nv * Math.min(buySignals.find(x => x.symbol === s).probability / tProb * 1.49 * vs, 0.15);
    const bysym = new Map(positions.map(p => [p.symbol, p]));
    for (const o of [...r.sells, ...r.buys]) {
      assert.ok(Number.isInteger(o.qty) && o.qty > 0);
      assert.ok(!closed.has(o.symbol));
      assert.ok(buySignals.some(x => x.symbol === o.symbol));
      assert.ok(Math.abs(tgt(o.symbol) - bysym.get(o.symbol).market_value) >= nv * REBAL_BAND_PCT);
    }
    for (const o of r.sells) assert.ok(bysym.get(o.symbol).market_value - o.qty * o.price >= tgt(o.symbol) - o.price - 1e-6);
    for (const o of r.buys) assert.ok(bysym.get(o.symbol).market_value + o.qty * o.price <= tgt(o.symbol) + 1e-5);
    const spent = r.buys.reduce((s, o) => s + o.qty * o.price, 0);
    assert.ok(spent <= Math.max(0, r.room) + 1e-5);
    // after the resizes and the new names' targets, the account never exceeds leverage x vol-scale
    const after = r.deployed - r.sells.reduce((s, o) => s + o.qty * o.price, 0) + spent + r.reserve;
    assert.ok(after <= Math.max(r.deployable, r.deployed + r.reserve) + 1e-6);
  }
});

console.log(`rebalance_resize: ${n} tests passed`);
