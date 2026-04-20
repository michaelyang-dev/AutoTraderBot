#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Trade Journal — Test Suite
//  All tests run against an in-memory SQLite database.
//
//  Usage: node scripts/test_journal.js
// ══════════════════════════════════════════════════════════════════════

const journal = require("../db/journal");

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (condition) {
    passed++;
    console.log(`  ✓ ${msg}`);
  } else {
    failed++;
    console.error(`  ✗ ${msg}`);
  }
}

function assertClose(a, b, tolerance, msg) {
  assert(Math.abs(a - b) < tolerance, `${msg} (got ${a}, expected ~${b})`);
}

// ══════════════════════════════════════════
//  Init
// ══════════════════════════════════════════

console.log("\n═══ Trade Journal Test Suite ═══\n");

journal.initDb(":memory:");

// ══════════════════════════════════════════
//  Test 1: Submit → Fill happy path
// ══════════════════════════════════════════

console.log("Test 1: Submit → Fill happy path");
{
  const tradeId = journal.recordOrderSubmitted({
    alpaca_order_id: "order-001",
    client_order_id: "client-001",
    symbol: "AAPL",
    side: "buy",
    qty: 10,
    strategy: "ml",
    signal_prob: 0.72,
    ml_rank: 2,
    regime: "BULLISH",
    intended_price: 150.00,
    stop_loss: 138.00,
    take_profit: 172.50,
  });

  assert(tradeId != null, "recordOrderSubmitted returns tradeId");

  const trade = journal.getTradeById(tradeId);
  assert(trade.status === "pending", "Initial status is pending");
  assert(trade.symbol === "AAPL", "Symbol recorded correctly");
  assert(trade.strategy === "ml", "Strategy recorded correctly");
  assert(trade.signal_prob === 0.72, "Signal probability recorded");
  assert(trade.intended_price === 150.00, "Intended price recorded");

  const result = journal.recordOrderFilled({
    alpaca_order_id: "order-001",
    filled_qty: 10,
    fill_price: 150.25,
    fill_time: "2024-01-15T10:30:00Z",
    commission: 0,
  });

  assert(result != null, "recordOrderFilled returns result");
  assert(result.slippageBps != null, "Slippage computed");

  const filled = journal.getTradeById(tradeId);
  assert(filled.status === "filled", "Status updated to filled");
  assert(filled.fill_price === 150.25, "Fill price recorded");
  assert(filled.filled_qty === 10, "Filled qty recorded");
}

// ══════════════════════════════════════════
//  Test 2: Submit → Reject
// ══════════════════════════════════════════

console.log("\nTest 2: Submit → Reject");
{
  journal.recordOrderSubmitted({
    alpaca_order_id: "order-reject-001",
    symbol: "GME",
    side: "buy",
    qty: 100,
    strategy: "consensus",
    intended_price: 25.00,
  });

  journal.recordOrderRejected({
    alpaca_order_id: "order-reject-001",
    reason: "insufficient buying power",
  });

  const trade = journal.getTradeByAlpacaId("order-reject-001");
  assert(trade.status === "rejected", "Status is rejected");
  assert(trade.notes === "insufficient buying power", "Rejection reason recorded");
}

// ══════════════════════════════════════════
//  Test 3: Submit → Partial → Fill
// ══════════════════════════════════════════

console.log("\nTest 3: Submit → Partial → Fill");
{
  journal.recordOrderSubmitted({
    alpaca_order_id: "order-partial-001",
    symbol: "NVDA",
    side: "buy",
    qty: 50,
    strategy: "momentum",
    intended_price: 800.00,
  });

  // Partial fill
  const db = journal.getDb();
  const stmt = db.prepare(`
    UPDATE trades SET status = 'partial', filled_qty = 30, fill_price = 800.10, updated_at = datetime('now')
    WHERE alpaca_order_id = 'order-partial-001'
  `);
  stmt.run();

  const partial = journal.getTradeByAlpacaId("order-partial-001");
  assert(partial.status === "partial", "Status is partial after partial fill");
  assert(partial.filled_qty === 30, "Partial filled_qty recorded");

  // Complete fill
  journal.recordOrderFilled({
    alpaca_order_id: "order-partial-001",
    filled_qty: 50,
    fill_price: 800.15,
    fill_time: "2024-01-15T10:35:00Z",
  });

  const complete = journal.getTradeByAlpacaId("order-partial-001");
  assert(complete.status === "filled", "Status is filled after complete fill");
  assert(complete.filled_qty === 50, "Full filled_qty recorded");
}

// ══════════════════════════════════════════
//  Test 4: Buy → Sell with realized P&L
// ══════════════════════════════════════════

console.log("\nTest 4: Buy → Sell with realized P&L");
{
  // Buy
  const buyId = journal.recordOrderSubmitted({
    alpaca_order_id: "order-pnl-buy",
    symbol: "MSFT",
    side: "buy",
    qty: 20,
    strategy: "ml",
    intended_price: 400.00,
    regime: "BULLISH",
  });

  journal.recordOrderFilled({
    alpaca_order_id: "order-pnl-buy",
    filled_qty: 20,
    fill_price: 400.00,
    fill_time: "2024-01-10T10:00:00Z",
  });

  // Create position
  journal.upsertPosition({
    symbol: "MSFT",
    qty: 20,
    avg_cost: 400.00,
    strategy: "ml",
    entry_regime: "BULLISH",
    entry_trade_id: buyId,
    opened_at: "2024-01-10T10:00:00Z",
  });

  // Sell
  const sellId = journal.recordOrderSubmitted({
    alpaca_order_id: "order-pnl-sell",
    symbol: "MSFT",
    side: "sell",
    qty: 20,
    strategy: "ml",
    intended_price: 420.00,
  });

  journal.recordOrderFilled({
    alpaca_order_id: "order-pnl-sell",
    filled_qty: 20,
    fill_price: 420.00,
    fill_time: "2024-01-15T14:00:00Z",
  });

  // Close position
  const result = journal.closePosition({
    symbol: "MSFT",
    sellTradeId: sellId,
    fillPrice: 420.00,
    exitReason: "take-profit",
  });

  assert(result != null, "closePosition returns result");
  assertClose(result.realizedPnl, 400.00, 0.01, "Realized P&L: (420-400)*20 = $400");
  assertClose(result.realizedPnlPct, 0.05, 0.001, "Realized P&L %: 5%");
  assert(result.holdDays >= 4, "Hold days >= 4 (Jan 10 → Jan 15)");

  // Verify position removed
  const positions = journal.getOpenPositions();
  const msft = positions.find(p => p.symbol === "MSFT");
  assert(!msft, "MSFT removed from positions table after close");

  // Verify sell trade has P&L
  const sellTrade = journal.getTradeById(sellId);
  assertClose(sellTrade.realized_pnl, 400.00, 0.01, "Sell trade has realized_pnl");
  assert(sellTrade.exit_reason === "take-profit", "Exit reason recorded on sell trade");
}

// ══════════════════════════════════════════
//  Test 5: Slippage calculation
// ══════════════════════════════════════════

console.log("\nTest 5: Slippage calculation");
{
  journal.recordOrderSubmitted({
    alpaca_order_id: "order-slip-001",
    symbol: "TSLA",
    side: "buy",
    qty: 5,
    strategy: "momentum",
    intended_price: 200.00,
  });

  const result = journal.recordOrderFilled({
    alpaca_order_id: "order-slip-001",
    filled_qty: 5,
    fill_price: 200.50,
    fill_time: "2024-01-15T10:30:00Z",
  });

  // Slippage = (200.50 - 200.00) / 200.00 * 10000 = 25 bps
  assert(result.slippageBps === 25, `Slippage is 25 bps (got ${result.slippageBps})`);

  const trade = journal.getTradeByAlpacaId("order-slip-001");
  assert(trade.slippage_bps === 25, "Slippage persisted in DB");
}

// ══════════════════════════════════════════
//  Test 6: Idempotent reconcile (fill twice)
// ══════════════════════════════════════════

console.log("\nTest 6: Idempotent reconcile (fill same order twice)");
{
  journal.recordOrderSubmitted({
    alpaca_order_id: "order-idem-001",
    symbol: "AMD",
    side: "buy",
    qty: 15,
    strategy: "ml",
    intended_price: 160.00,
  });

  journal.recordOrderFilled({
    alpaca_order_id: "order-idem-001",
    filled_qty: 15,
    fill_price: 160.20,
    fill_time: "2024-01-15T10:30:00Z",
  });

  // Second fill call — should not error
  journal.recordOrderFilled({
    alpaca_order_id: "order-idem-001",
    filled_qty: 15,
    fill_price: 160.20,
    fill_time: "2024-01-15T10:30:00Z",
  });

  const trade = journal.getTradeByAlpacaId("order-idem-001");
  assert(trade.status === "filled", "Status still filled after second call");
  assert(trade.filled_qty === 15, "Filled qty unchanged after second call");
}

// ══════════════════════════════════════════
//  Test 7: Position upsert (cost basis averaging)
// ══════════════════════════════════════════

console.log("\nTest 7: Position upsert (cost basis averaging)");
{
  // First buy: 10 shares @ $100
  journal.upsertPosition({
    symbol: "GOOG",
    qty: 10,
    avg_cost: 100.00,
    cost_basis: 1000.00,
    strategy: "ml",
    opened_at: "2024-01-10T10:00:00Z",
  });

  let pos = journal.getOpenPositions().find(p => p.symbol === "GOOG");
  assert(pos.qty === 10, "First buy: 10 shares");
  assertClose(pos.cost_basis, 1000.00, 0.01, "First buy: cost basis $1000");

  // Add to position: now 25 shares @ new avg
  journal.upsertPosition({
    symbol: "GOOG",
    qty: 25,
    avg_cost: 104.00,
    cost_basis: 2600.00,
    current_price: 110.00,
  });

  pos = journal.getOpenPositions().find(p => p.symbol === "GOOG");
  assert(pos.qty === 25, "After add: 25 shares");
  assertClose(pos.cost_basis, 2600.00, 0.01, "After add: cost basis $2600");
  assert(pos.strategy === "ml", "Strategy preserved from first upsert (COALESCE)");
}

// ══════════════════════════════════════════
//  Test 8: Trailing stop highest price tracking
// ══════════════════════════════════════════

console.log("\nTest 8: Trailing stop highest price tracking");
{
  journal.upsertPosition({
    symbol: "META",
    qty: 5,
    avg_cost: 300.00,
    highest_price_since_entry: 310.00,
    strategy: "ml",
    opened_at: "2024-01-10T10:00:00Z",
  });

  // Update with lower price — highest should stay at 310
  journal.updatePositionPrices({ META: 305.00 });
  let pos = journal.getOpenPositions().find(p => p.symbol === "META");
  assert(pos.highest_price_since_entry >= 310.00, "Highest price not decreased by lower update (via MAX)");

  // Update with higher price — highest should increase
  journal.updatePositionPrices({ META: 320.00 });
  pos = journal.getOpenPositions().find(p => p.symbol === "META");
  assert(pos.highest_price_since_entry >= 320.00, "Highest price increased to 320");
  assertClose(pos.current_price, 320.00, 0.01, "Current price updated to 320");
}

// ══════════════════════════════════════════
//  Test 9: Daily snapshot
// ══════════════════════════════════════════

console.log("\nTest 9: Daily snapshot write and query");
{
  journal.writeDailySnapshot({
    date: "2024-01-15",
    portfolio_value: 105000,
    cash: 20000,
    equity: 105000,
    positions_count: 8,
    day_pnl: 500,
    day_pnl_pct: 0.48,
    regime: "BULLISH",
    spy_close: 480.00,
    spy_day_pct: 0.35,
    strategy_pnl: { ml: 300, momentum: 150, mean_reversion: 50 },
    slot_usage: { ml: 3, momentum: 2, mean_reversion: 1, mega_cap: 0 },
    peak_value: 105000,
    drawdown_pct: 0,
  });

  journal.writeDailySnapshot({
    date: "2024-01-16",
    portfolio_value: 104500,
    cash: 19000,
    equity: 104500,
    positions_count: 9,
    day_pnl: -500,
    day_pnl_pct: -0.48,
    regime: "BULLISH",
    peak_value: 105000,
    drawdown_pct: -0.48,
  });

  const snapshots = journal.getDailySnapshots("2024-01-15", "2024-01-16");
  assert(snapshots.length === 2, "Two snapshots returned");
  assert(snapshots[0].date === "2024-01-15", "First snapshot is Jan 15");

  const stratPnl = JSON.parse(snapshots[0].strategy_pnl_json);
  assert(stratPnl.ml === 300, "Strategy P&L JSON parsed correctly");

  const prev = journal.getPreviousSnapshot("2024-01-16");
  assert(prev && prev.date === "2024-01-15", "getPreviousSnapshot returns Jan 15 for before Jan 16");

  // Upsert same date — should update, not duplicate
  journal.writeDailySnapshot({
    date: "2024-01-15",
    portfolio_value: 105100,
    cash: 20100,
    equity: 105100,
    positions_count: 8,
    day_pnl: 600,
    day_pnl_pct: 0.57,
    regime: "BULLISH",
  });

  const updated = journal.getDailySnapshots("2024-01-15", "2024-01-15");
  assert(updated.length === 1, "Upsert: still one snapshot for Jan 15");
  assertClose(updated[0].portfolio_value, 105100, 0.01, "Upsert: portfolio value updated to 105100");
}

// ══════════════════════════════════════════
//  Test 10: Signals
// ══════════════════════════════════════════

console.log("\nTest 10: Signal recording and acted_on marking");
{
  const sigId = journal.recordSignal({
    symbol: "AAPL",
    signal_type: "BUY",
    probability: 0.85,
    rank: 1,
    is_top_5: true,
    price_at_signal: 150.00,
    regime: "BULLISH",
    model_version: "v5c",
  });

  assert(sigId != null, "recordSignal returns signalId");

  journal.markSignalActedOn(sigId);

  const db = journal.getDb();
  const sig = db.prepare("SELECT * FROM signals WHERE id = ?").get(sigId);
  assert(sig.acted_on === 1, "Signal marked as acted_on");
  assert(sig.probability === 0.85, "Signal probability stored");
  assert(sig.is_top_5 === 1, "Signal is_top_5 stored as 1");
}

// ══════════════════════════════════════════
//  Test 11: Events
// ══════════════════════════════════════════

console.log("\nTest 11: Event logging");
{
  journal.logEvent({
    event_type: "circuit_breaker",
    severity: "critical",
    message: "Daily loss limit hit: -3.2%",
    metadata: { layer: "daily", loss_pct: -3.2, portfolio: 96800 },
    portfolio_value: 96800,
  });

  journal.logEvent({
    event_type: "regime_change",
    severity: "info",
    message: "BULLISH → CAUTIOUS",
    metadata: { from: "BULLISH", to: "CAUTIOUS", spy: 472.5 },
    portfolio_value: 100000,
  });

  journal.logEvent({
    event_type: "service_restart",
    severity: "info",
    message: "Trading engine restarted",
  });

  const events = journal.getRecentEvents(10);
  assert(events.length === 3, "3 events recorded");
  assert(events[0].event_type === "service_restart", "Most recent event first (by id DESC)");

  const cbEvents = journal.getEventsByType("circuit_breaker", 10);
  assert(cbEvents.length === 1, "1 circuit_breaker event");
  assert(cbEvents[0].severity === "critical", "Severity is critical");

  const meta = JSON.parse(cbEvents[0].metadata_json);
  assert(meta.layer === "daily", "Metadata JSON parsed correctly");
}

// ══════════════════════════════════════════
//  Test 12: Strategy stats aggregation
// ══════════════════════════════════════════

console.log("\nTest 12: Strategy stats aggregation");
{
  const stats = journal.getStrategyStats("2000-01-01");
  assert(stats.length > 0, "Strategy stats returned");

  const mlStat = stats.find(s => s.strategy === "ml");
  assert(mlStat != null, "ML strategy stats exist");
  assert(mlStat.total_trades > 0, "ML has trades");
}

// ══════════════════════════════════════════
//  Test 13: Cancel flow
// ══════════════════════════════════════════

console.log("\nTest 13: Order cancel");
{
  journal.recordOrderSubmitted({
    alpaca_order_id: "order-cancel-001",
    symbol: "NFLX",
    side: "buy",
    qty: 8,
    strategy: "mega_cap",
    intended_price: 600.00,
  });

  journal.recordOrderCanceled({ alpaca_order_id: "order-cancel-001" });

  const trade = journal.getTradeByAlpacaId("order-cancel-001");
  assert(trade.status === "canceled", "Status is canceled");
}

// ══════════════════════════════════════════
//  Test 14: Multi-lot FIFO P&L (closePosition with averaged cost basis)
// ══════════════════════════════════════════

console.log("\nTest 14: Multi-lot FIFO P&L via closePosition");
{
  // Buy 10 @ $100, then add 10 @ $120 → avg cost $110, cost basis $2200
  journal.upsertPosition({
    symbol: "FIFO1",
    qty: 10,
    avg_cost: 100,
    current_price: 105,
    strategy: "ml",
    opened_at: "2025-01-01T10:00:00Z",
  });
  journal.upsertPosition({
    symbol: "FIFO1",
    qty: 20,
    avg_cost: 110,
    current_price: 115,
    strategy: null,  // should preserve "ml" via COALESCE
  });

  const pos = journal.getDb().prepare("SELECT * FROM positions WHERE symbol = 'FIFO1'").get();
  assertClose(pos.qty, 20, 0.01, "Multi-lot: 20 shares");
  assertClose(pos.avg_cost, 110, 0.01, "Multi-lot: avg cost $110");
  assertClose(pos.cost_basis, 2200, 1, "Multi-lot: cost basis $2200");
  assert(pos.strategy === "ml", "Multi-lot: strategy preserved from first upsert");

  // Sell trade for closePosition
  const sellId = journal.recordOrderSubmitted({
    alpaca_order_id: "order-fifo-sell-001",
    symbol: "FIFO1", side: "sell", qty: 20,
    strategy: "ml", intended_price: 130,
  });
  journal.recordOrderFilled({
    alpaca_order_id: "order-fifo-sell-001",
    filled_qty: 20, fill_price: 130,
  });

  const result = journal.closePosition({
    symbol: "FIFO1",
    sellTradeId: sellId,
    fillPrice: 130,
    exitReason: "take_profit",
  });

  // P&L = (130 * 20) - 2200 = 2600 - 2200 = $400
  assert(result != null, "closePosition returns result");
  assertClose(result.realizedPnl, 400, 1, "Multi-lot P&L: (130-110)*20 = $400");
  assertClose(result.realizedPnlPct, 0.1818, 0.01, "Multi-lot P&L%: 400/2200 ≈ 18.18%");

  const closedPos = journal.getDb().prepare("SELECT * FROM positions WHERE symbol = 'FIFO1'").get();
  assert(!closedPos, "FIFO1 removed from positions after close");

  const sellTrade = journal.getTradeById(sellId);
  assert(sellTrade.exit_reason === "take_profit", "Exit reason recorded on sell trade");
  assertClose(sellTrade.realized_pnl, 400, 1, "Sell trade has realized_pnl $400");
}

// ══════════════════════════════════════════
//  Test 15: removeStalePositions cleans up and stamps P&L
// ══════════════════════════════════════════

console.log("\nTest 15: removeStalePositions");
{
  // Setup: position for STALE1 in journal
  journal.upsertPosition({
    symbol: "STALE1",
    qty: 15,
    avg_cost: 200,
    current_price: 210,
    strategy: "momentum",
    opened_at: "2025-01-05T10:00:00Z",
  });

  // A filled sell trade exists for STALE1
  const sellId = journal.recordOrderSubmitted({
    alpaca_order_id: "order-stale-sell-001",
    symbol: "STALE1", side: "sell", qty: 15,
    strategy: "momentum", intended_price: 220,
  });
  journal.recordOrderFilled({
    alpaca_order_id: "order-stale-sell-001",
    filled_qty: 15, fill_price: 225,
  });

  // Also add a position for HELD1 that is still held
  journal.upsertPosition({
    symbol: "HELD1",
    qty: 10,
    avg_cost: 100,
    current_price: 110,
    strategy: "ml",
  });

  // Alpaca only holds HELD1 — STALE1 (and any other leftover test positions) are gone
  const removed = journal.removeStalePositions(["HELD1"]);
  assert(removed >= 1, `Removed stale positions (${removed} removed, at least STALE1)`);

  // STALE1 should be gone from positions
  const stalePos = journal.getDb().prepare("SELECT * FROM positions WHERE symbol = 'STALE1'").get();
  assert(!stalePos, "STALE1 removed from positions table");

  // HELD1 should still be there
  const heldPos = journal.getDb().prepare("SELECT * FROM positions WHERE symbol = 'HELD1'").get();
  assert(heldPos != null, "HELD1 still in positions table");

  // Sell trade should have P&L stamped: (225 * 15) - (200 * 15) = 3375 - 3000 = $375
  const sellTrade = journal.getTradeById(sellId);
  assertClose(sellTrade.realized_pnl, 375, 1, "Stale close P&L: (225-200)*15 = $375");
  assert(sellTrade.exit_reason === "position_closed", "Exit reason stamped as position_closed");
  assert(sellTrade.hold_days != null, "Hold days computed");
}

// ══════════════════════════════════════════
//  Results
// ══════════════════════════════════════════

console.log(`\n═══ Results: ${passed} passed, ${failed} failed ═══\n`);
process.exit(failed > 0 ? 1 : 0);
