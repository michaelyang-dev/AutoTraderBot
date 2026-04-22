// ══════════════════════════════════════════════════════════════════════
//  Trade Journal — SQLite persistence layer
//
//  All functions are synchronous (better-sqlite3) and wrapped in
//  try/catch — they never throw into the trading loop.
//
//  Usage:
//    const journal = require("../db/journal");
//    journal.initDb();                        // or journal.initDb(":memory:")
//    journal.recordOrderSubmitted({...});
// ══════════════════════════════════════════════════════════════════════

const Database = require("better-sqlite3");
const fs = require("fs");
const path = require("path");

const SCHEMA_PATH = path.join(__dirname, "schema.sql");
const MIGRATIONS_DIR = path.join(__dirname, "migrations");
const DEFAULT_DB_PATH = path.join(__dirname, "..", "data", "journal.db");

let db = null;

// ── Prepared statements (populated by initDb) ───────────────────────
let stmts = {};

// ══════════════════════════════════════════
//  MIGRATIONS
// ══════════════════════════════════════════

function runMigrations() {
  try {
    if (!fs.existsSync(MIGRATIONS_DIR)) return;

    // Get current schema version
    const row = db.prepare("SELECT MAX(version) as v FROM schema_version").get();
    const currentVersion = row?.v || 1;

    // Find migration files: NNN_name.sql, sorted by version number
    const files = fs.readdirSync(MIGRATIONS_DIR)
      .filter(f => f.endsWith(".sql"))
      .map(f => ({ file: f, version: parseInt(f.split("_")[0], 10) }))
      .filter(m => !isNaN(m.version) && m.version > currentVersion)
      .sort((a, b) => a.version - b.version);

    for (const m of files) {
      const sql = fs.readFileSync(path.join(MIGRATIONS_DIR, m.file), "utf8");
      db.exec(sql);
      console.log(`journal: applied migration ${m.file} (v${m.version})`);
    }
  } catch (err) {
    console.error("journal: migration failed:", err.message);
    throw err; // migrations are critical — don't swallow
  }
}

// ══════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════

function initDb(dbPath) {
  dbPath = dbPath || DEFAULT_DB_PATH;

  // Ensure parent directory exists for file-based DBs
  if (dbPath !== ":memory:") {
    const dir = path.dirname(dbPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  }

  db = new Database(dbPath);

  // Apply schema
  const schema = fs.readFileSync(SCHEMA_PATH, "utf8");
  db.exec(schema);

  // Run pending migrations
  runMigrations();

  // Prepare all statements once
  stmts = {
    // ── Trades ──
    insertTrade: db.prepare(`
      INSERT INTO trades (
        alpaca_order_id, client_order_id, symbol, side, qty,
        strategy, signal_id, signal_prob, ml_rank, regime,
        intended_price, stop_loss, take_profit, status, submitted_at
      ) VALUES (
        @alpaca_order_id, @client_order_id, @symbol, @side, @qty,
        @strategy, @signal_id, @signal_prob, @ml_rank, @regime,
        @intended_price, @stop_loss, @take_profit, @status, @submitted_at
      )
    `),

    updateFilled: db.prepare(`
      UPDATE trades SET
        status = 'filled', filled_qty = @filled_qty, fill_price = @fill_price,
        fill_time = @fill_time, slippage_bps = @slippage_bps, commission = @commission,
        updated_at = datetime('now')
      WHERE alpaca_order_id = @alpaca_order_id
    `),

    updatePartial: db.prepare(`
      UPDATE trades SET
        status = 'partial', filled_qty = @filled_qty, fill_price = @fill_price,
        updated_at = datetime('now')
      WHERE alpaca_order_id = @alpaca_order_id
    `),

    updateRejected: db.prepare(`
      UPDATE trades SET status = 'rejected', notes = @reason, updated_at = datetime('now')
      WHERE alpaca_order_id = @alpaca_order_id
    `),

    updateCanceled: db.prepare(`
      UPDATE trades SET status = 'canceled', updated_at = datetime('now')
      WHERE alpaca_order_id = @alpaca_order_id
    `),

    updateExit: db.prepare(`
      UPDATE trades SET
        realized_pnl = @realized_pnl, realized_pnl_pct = @realized_pnl_pct,
        hold_days = @hold_days, exit_reason = @exit_reason,
        entry_trade_id = @entry_trade_id, updated_at = datetime('now')
      WHERE id = @id
    `),

    getTradeByAlpacaId: db.prepare(`SELECT * FROM trades WHERE alpaca_order_id = @alpaca_order_id`),
    getTradeById: db.prepare(`SELECT * FROM trades WHERE id = @id`),
    getTradesBySymbol: db.prepare(`SELECT * FROM trades WHERE symbol = @symbol ORDER BY submitted_at DESC LIMIT @limit`),
    getTradesByStrategy: db.prepare(`SELECT * FROM trades WHERE strategy = @strategy AND submitted_at >= @since ORDER BY submitted_at DESC`),
    getTradesByDateRange: db.prepare(`SELECT * FROM trades WHERE submitted_at >= @from AND submitted_at <= @to ORDER BY submitted_at`),

    // Find entry trade for FIFO matching — only buys with remaining shares
    findEntryTrade: db.prepare(`
      SELECT *, (COALESCE(filled_qty, qty) - COALESCE(consumed_qty, 0)) AS remaining
      FROM trades
      WHERE symbol = @symbol AND side = 'buy' AND status = 'filled'
        AND (COALESCE(filled_qty, qty) - COALESCE(consumed_qty, 0)) > 0.001
      ORDER BY fill_time ASC
      LIMIT 1
    `),

    // Strategy-aware FIFO: oldest buy with remaining shares for this symbol+strategy
    findEntryTradeByStrategy: db.prepare(`
      SELECT *, (COALESCE(filled_qty, qty) - COALESCE(consumed_qty, 0)) AS remaining
      FROM trades
      WHERE symbol = @symbol AND side = 'buy' AND status = 'filled'
        AND strategy = @strategy
        AND (COALESCE(filled_qty, qty) - COALESCE(consumed_qty, 0)) > 0.001
      ORDER BY fill_time ASC
      LIMIT 1
    `),

    // Consume shares from a buy trade (increment consumed_qty)
    consumeEntryShares: db.prepare(`
      UPDATE trades
      SET consumed_qty = COALESCE(consumed_qty, 0) + @consumed,
          updated_at = datetime('now')
      WHERE id = @id
    `),

    // Reset all consumed_qty (for full P&L recompute)
    resetConsumedQty: db.prepare(`
      UPDATE trades SET consumed_qty = 0 WHERE side = 'buy' AND status = 'filled'
    `),

    // Reset all sell P&L fields (for full recompute)
    resetSellPnl: db.prepare(`
      UPDATE trades
      SET realized_pnl = NULL, realized_pnl_pct = NULL,
          entry_trade_id = NULL, hold_days = NULL
      WHERE side = 'sell' AND status = 'filled'
    `),

    // Find ALL filled sells (for full recompute), chronological order
    findAllFilledSells: db.prepare(`
      SELECT * FROM trades
      WHERE side = 'sell' AND status = 'filled'
      ORDER BY fill_time ASC, id ASC
    `),

    // ── Positions ──
    upsertPosition: db.prepare(`
      INSERT INTO positions (
        symbol, qty, avg_cost, cost_basis, current_price, current_value,
        unrealized_pnl, unrealized_pnl_pct, strategy, entry_regime,
        entry_trade_id, opened_at, days_held, stop_loss, take_profit,
        trailing_stop, highest_price_since_entry, updated_at
      ) VALUES (
        @symbol, @qty, @avg_cost, @cost_basis, @current_price, @current_value,
        @unrealized_pnl, @unrealized_pnl_pct, @strategy, @entry_regime,
        @entry_trade_id, @opened_at, @days_held, @stop_loss, @take_profit,
        @trailing_stop, @highest_price_since_entry, datetime('now')
      ) ON CONFLICT(symbol) DO UPDATE SET
        qty = @qty, avg_cost = @avg_cost, cost_basis = @cost_basis,
        current_price = @current_price, current_value = @current_value,
        unrealized_pnl = @unrealized_pnl, unrealized_pnl_pct = @unrealized_pnl_pct,
        strategy = COALESCE(@strategy, positions.strategy),
        days_held = @days_held,
        stop_loss = COALESCE(@stop_loss, positions.stop_loss),
        take_profit = COALESCE(@take_profit, positions.take_profit),
        trailing_stop = COALESCE(@trailing_stop, positions.trailing_stop),
        highest_price_since_entry = MAX(COALESCE(@highest_price_since_entry, 0), COALESCE(positions.highest_price_since_entry, 0)),
        updated_at = datetime('now')
    `),

    deletePosition: db.prepare(`DELETE FROM positions WHERE symbol = @symbol`),
    getPosition: db.prepare(`SELECT * FROM positions WHERE symbol = @symbol`),
    getOpenPositions: db.prepare(`SELECT * FROM positions ORDER BY symbol`),

    updatePositionPrice: db.prepare(`
      UPDATE positions SET
        current_price = @current_price,
        current_value = qty * @current_price,
        unrealized_pnl = (qty * @current_price) - cost_basis,
        unrealized_pnl_pct = CASE WHEN cost_basis > 0 THEN ((qty * @current_price) - cost_basis) / cost_basis ELSE 0 END,
        highest_price_since_entry = MAX(COALESCE(highest_price_since_entry, 0), @current_price),
        days_held = CAST(julianday('now') - julianday(opened_at) AS INTEGER),
        updated_at = datetime('now')
      WHERE symbol = @symbol
    `),

    // ── Daily Snapshots ──
    insertSnapshot: db.prepare(`
      INSERT INTO daily_snapshots (
        date, portfolio_value, cash, equity, positions_count,
        day_pnl, day_pnl_pct, regime, spy_close, spy_day_pct,
        strategy_pnl_json, slot_usage_json, circuit_breaker_state,
        peak_value, drawdown_pct
      ) VALUES (
        @date, @portfolio_value, @cash, @equity, @positions_count,
        @day_pnl, @day_pnl_pct, @regime, @spy_close, @spy_day_pct,
        @strategy_pnl_json, @slot_usage_json, @circuit_breaker_state,
        @peak_value, @drawdown_pct
      ) ON CONFLICT(date) DO UPDATE SET
        portfolio_value = @portfolio_value, cash = @cash, equity = @equity,
        positions_count = @positions_count, day_pnl = @day_pnl, day_pnl_pct = @day_pnl_pct,
        regime = @regime, spy_close = @spy_close, spy_day_pct = @spy_day_pct,
        strategy_pnl_json = @strategy_pnl_json, slot_usage_json = @slot_usage_json,
        circuit_breaker_state = @circuit_breaker_state,
        peak_value = @peak_value, drawdown_pct = @drawdown_pct
    `),

    getPreviousSnapshot: db.prepare(`
      SELECT * FROM daily_snapshots WHERE date < @before_date ORDER BY date DESC LIMIT 1
    `),
    getDailySnapshots: db.prepare(`
      SELECT * FROM daily_snapshots WHERE date >= @from AND date <= @to ORDER BY date
    `),

    // ── Signals ──
    insertSignal: db.prepare(`
      INSERT INTO signals (
        symbol, signal_time, signal_type, probability, rank,
        is_top_5, price_at_signal, regime, model_version, acted_on
      ) VALUES (
        @symbol, @signal_time, @signal_type, @probability, @rank,
        @is_top_5, @price_at_signal, @regime, @model_version, @acted_on
      )
    `),
    markSignalActedOn: db.prepare(`UPDATE signals SET acted_on = 1 WHERE id = @id`),
    getSignalsBySymbol: db.prepare(`
      SELECT * FROM signals WHERE symbol = @symbol AND signal_time >= @since ORDER BY signal_time DESC
    `),

    // ── Events ──
    insertEvent: db.prepare(`
      INSERT INTO events (event_type, severity, message, metadata_json, portfolio_value_at_event)
      VALUES (@event_type, @severity, @message, @metadata_json, @portfolio_value_at_event)
    `),
    getRecentEvents: db.prepare(`SELECT * FROM events ORDER BY id DESC LIMIT @limit`),
    getEventsByType: db.prepare(`
      SELECT * FROM events WHERE event_type = @event_type ORDER BY id DESC LIMIT @limit
    `),

    // ── Strategy stats ──
    getStrategyStats: db.prepare(`
      SELECT
        strategy,
        COUNT(*) as total_trades,
        SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as buys,
        SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as sells,
        SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
        SUM(COALESCE(realized_pnl, 0)) as total_pnl,
        AVG(CASE WHEN realized_pnl IS NOT NULL THEN realized_pnl END) as avg_pnl,
        AVG(CASE WHEN hold_days IS NOT NULL THEN hold_days END) as avg_hold_days,
        AVG(CASE WHEN slippage_bps IS NOT NULL THEN slippage_bps END) as avg_slippage_bps
      FROM trades
      WHERE submitted_at >= @since
      GROUP BY strategy
    `),
  };

  return db;
}

// ══════════════════════════════════════════
//  WRITE OPERATIONS
// ══════════════════════════════════════════

function recordOrderSubmitted({
  alpaca_order_id, client_order_id, symbol, side, qty,
  strategy, signal_id = null, signal_prob = null, ml_rank = null,
  regime = null, intended_price = null, stop_loss = null, take_profit = null,
}) {
  try {
    const info = stmts.insertTrade.run({
      alpaca_order_id: alpaca_order_id || null,
      client_order_id: client_order_id || null,
      symbol, side,
      qty: parseFloat(qty),
      strategy,
      signal_id: signal_id || null,
      signal_prob: signal_prob != null ? parseFloat(signal_prob) : null,
      ml_rank: ml_rank != null ? parseInt(ml_rank) : null,
      regime: regime || null,
      intended_price: intended_price != null ? parseFloat(intended_price) : null,
      stop_loss: stop_loss != null ? parseFloat(stop_loss) : null,
      take_profit: take_profit != null ? parseFloat(take_profit) : null,
      status: "pending",
      submitted_at: new Date().toISOString(),
    });
    return info.lastInsertRowid;
  } catch (err) {
    console.error("journal: recordOrderSubmitted failed:", err.message);
    return null;
  }
}

function recordOrderFilled({ alpaca_order_id, filled_qty, fill_price, fill_time, commission = 0 }) {
  try {
    const existing = stmts.getTradeByAlpacaId.get({ alpaca_order_id });
    if (!existing) return null;

    const slippage_bps = existing.intended_price && existing.intended_price > 0
      ? Math.round((fill_price - existing.intended_price) / existing.intended_price * 10000)
      : null;

    stmts.updateFilled.run({
      alpaca_order_id,
      filled_qty: parseFloat(filled_qty),
      fill_price: parseFloat(fill_price),
      fill_time: fill_time || new Date().toISOString(),
      slippage_bps,
      commission: parseFloat(commission) || 0,
    });

    return { tradeId: existing.id, slippageBps: slippage_bps };
  } catch (err) {
    console.error("journal: recordOrderFilled failed:", err.message);
    return null;
  }
}

function recordOrderRejected({ alpaca_order_id, reason }) {
  try {
    stmts.updateRejected.run({ alpaca_order_id, reason: reason || "unknown" });
  } catch (err) {
    console.error("journal: recordOrderRejected failed:", err.message);
  }
}

function recordOrderCanceled({ alpaca_order_id }) {
  try {
    stmts.updateCanceled.run({ alpaca_order_id });
  } catch (err) {
    console.error("journal: recordOrderCanceled failed:", err.message);
  }
}

function upsertPosition({
  symbol, qty, avg_cost, cost_basis = null,
  current_price = null, current_value = null,
  unrealized_pnl = null, unrealized_pnl_pct = null,
  strategy = null, entry_regime = null, entry_trade_id = null,
  opened_at = null, days_held = null,
  stop_loss = null, take_profit = null, trailing_stop = null,
  highest_price_since_entry = null,
}) {
  try {
    const cb = cost_basis != null ? parseFloat(cost_basis) : parseFloat(qty) * parseFloat(avg_cost);
    const cv = current_value != null ? parseFloat(current_value)
      : (current_price != null ? parseFloat(qty) * parseFloat(current_price) : null);

    stmts.upsertPosition.run({
      symbol,
      qty: parseFloat(qty),
      avg_cost: parseFloat(avg_cost),
      cost_basis: cb,
      current_price: current_price != null ? parseFloat(current_price) : null,
      current_value: cv,
      unrealized_pnl: unrealized_pnl != null ? parseFloat(unrealized_pnl) : null,
      unrealized_pnl_pct: unrealized_pnl_pct != null ? parseFloat(unrealized_pnl_pct) : null,
      strategy: strategy || null,
      entry_regime: entry_regime || null,
      entry_trade_id: entry_trade_id != null ? parseInt(entry_trade_id) : null,
      opened_at: opened_at || new Date().toISOString(),
      days_held: days_held != null ? parseInt(days_held) : 0,
      stop_loss: stop_loss != null ? parseFloat(stop_loss) : null,
      take_profit: take_profit != null ? parseFloat(take_profit) : null,
      trailing_stop: trailing_stop != null ? parseFloat(trailing_stop) : null,
      highest_price_since_entry: highest_price_since_entry != null ? parseFloat(highest_price_since_entry) : null,
    });
  } catch (err) {
    console.error("journal: upsertPosition failed:", err.message);
  }
}

function updatePositionPrices(pricesMap) {
  try {
    const update = db.transaction(() => {
      for (const [symbol, price] of Object.entries(pricesMap)) {
        stmts.updatePositionPrice.run({ symbol, current_price: parseFloat(price) });
      }
    });
    update();
  } catch (err) {
    console.error("journal: updatePositionPrices failed:", err.message);
  }
}

function closePosition({ symbol, sellTradeId = null, fillPrice, exitReason }) {
  try {
    const pos = stmts.getPosition.get({ symbol });
    if (!pos) return null;

    const realizedPnl = (parseFloat(fillPrice) * pos.qty) - pos.cost_basis;
    const realizedPnlPct = pos.cost_basis > 0 ? realizedPnl / pos.cost_basis : 0;
    const holdDays = pos.opened_at
      ? Math.round((Date.now() - new Date(pos.opened_at).getTime()) / (1000 * 60 * 60 * 24))
      : null;

    // Update the sell trade with P&L info
    if (sellTradeId) {
      stmts.updateExit.run({
        id: sellTradeId,
        realized_pnl: Math.round(realizedPnl * 100) / 100,
        realized_pnl_pct: Math.round(realizedPnlPct * 10000) / 10000,
        hold_days: holdDays,
        exit_reason: exitReason || null,
        entry_trade_id: pos.entry_trade_id || null,
      });
    }

    // Remove from positions table
    stmts.deletePosition.run({ symbol });

    return {
      realizedPnl: Math.round(realizedPnl * 100) / 100,
      realizedPnlPct: Math.round(realizedPnlPct * 10000) / 10000,
      holdDays,
    };
  } catch (err) {
    console.error("journal: closePosition failed:", err.message);
    return null;
  }
}

function removeStalePositions(heldSymbols) {
  try {
    const allPositions = stmts.getOpenPositions.all();
    const heldSet = new Set(heldSymbols);
    let removed = 0;

    for (const pos of allPositions) {
      if (heldSet.has(pos.symbol)) continue;

      // Find most recent filled sell trade for this symbol
      const sellTrade = db.prepare(
        `SELECT * FROM trades WHERE symbol = @symbol AND side = 'sell' AND status = 'filled'
         ORDER BY fill_time DESC LIMIT 1`
      ).get({ symbol: pos.symbol });

      if (sellTrade && sellTrade.fill_price) {
        // Compute P&L from position's cost basis
        const fillPrice = parseFloat(sellTrade.fill_price);
        const realizedPnl = (fillPrice * pos.qty) - pos.cost_basis;
        const realizedPnlPct = pos.cost_basis > 0 ? realizedPnl / pos.cost_basis : 0;
        const holdDays = pos.opened_at
          ? Math.round((Date.now() - new Date(pos.opened_at).getTime()) / (1000 * 60 * 60 * 24))
          : null;

        // Stamp P&L onto the sell trade if not already set
        if (sellTrade.realized_pnl == null) {
          stmts.updateExit.run({
            id: sellTrade.id,
            realized_pnl: Math.round(realizedPnl * 100) / 100,
            realized_pnl_pct: Math.round(realizedPnlPct * 10000) / 10000,
            hold_days: holdDays,
            exit_reason: sellTrade.exit_reason || "position_closed",
            entry_trade_id: pos.entry_trade_id || null,
          });
        }
      }

      stmts.deletePosition.run({ symbol: pos.symbol });
      removed++;
    }

    return removed;
  } catch (err) {
    console.error("journal: removeStalePositions failed:", err.message);
    return 0;
  }
}

function writeDailySnapshot({
  date, portfolio_value, cash, equity = null, positions_count,
  day_pnl = null, day_pnl_pct = null, regime = null,
  spy_close = null, spy_day_pct = null,
  strategy_pnl = null, slot_usage = null, circuit_breaker_state = null,
  peak_value = null, drawdown_pct = null,
}) {
  try {
    stmts.insertSnapshot.run({
      date,
      portfolio_value: parseFloat(portfolio_value),
      cash: parseFloat(cash),
      equity: equity != null ? parseFloat(equity) : null,
      positions_count: parseInt(positions_count),
      day_pnl: day_pnl != null ? parseFloat(day_pnl) : null,
      day_pnl_pct: day_pnl_pct != null ? parseFloat(day_pnl_pct) : null,
      regime: regime || null,
      spy_close: spy_close != null ? parseFloat(spy_close) : null,
      spy_day_pct: spy_day_pct != null ? parseFloat(spy_day_pct) : null,
      strategy_pnl_json: strategy_pnl ? JSON.stringify(strategy_pnl) : null,
      slot_usage_json: slot_usage ? JSON.stringify(slot_usage) : null,
      circuit_breaker_state: circuit_breaker_state || null,
      peak_value: peak_value != null ? parseFloat(peak_value) : null,
      drawdown_pct: drawdown_pct != null ? parseFloat(drawdown_pct) : null,
    });
  } catch (err) {
    console.error("journal: writeDailySnapshot failed:", err.message);
  }
}

function recordSignal({
  symbol, signal_time = null, signal_type = null,
  probability = null, rank = null, is_top_5 = false,
  price_at_signal = null, regime = null, model_version = null,
}) {
  try {
    const info = stmts.insertSignal.run({
      symbol,
      signal_time: signal_time || new Date().toISOString(),
      signal_type: signal_type || null,
      probability: probability != null ? parseFloat(probability) : null,
      rank: rank != null ? parseInt(rank) : null,
      is_top_5: is_top_5 ? 1 : 0,
      price_at_signal: price_at_signal != null ? parseFloat(price_at_signal) : null,
      regime: regime || null,
      model_version: model_version || null,
      acted_on: 0,
    });
    return info.lastInsertRowid;
  } catch (err) {
    console.error("journal: recordSignal failed:", err.message);
    return null;
  }
}

function markSignalActedOn(signalId) {
  try {
    stmts.markSignalActedOn.run({ id: signalId });
  } catch (err) {
    console.error("journal: markSignalActedOn failed:", err.message);
  }
}

function logEvent({
  event_type, severity = "info", message,
  metadata = null, portfolio_value = null,
}) {
  try {
    stmts.insertEvent.run({
      event_type,
      severity,
      message,
      metadata_json: metadata ? JSON.stringify(metadata) : null,
      portfolio_value_at_event: portfolio_value != null ? parseFloat(portfolio_value) : null,
    });
  } catch (err) {
    console.error("journal: logEvent failed:", err.message);
  }
}

// ══════════════════════════════════════════
//  READ OPERATIONS
// ══════════════════════════════════════════

function getOpenPositions() {
  try { return stmts.getOpenPositions.all(); }
  catch (err) { console.error("journal: getOpenPositions failed:", err.message); return []; }
}

function getTradeById(id) {
  try { return stmts.getTradeById.get({ id }); }
  catch (err) { console.error("journal: getTradeById failed:", err.message); return null; }
}

function getTradeByAlpacaId(alpacaOrderId) {
  try { return stmts.getTradeByAlpacaId.get({ alpaca_order_id: alpacaOrderId }); }
  catch (err) { console.error("journal: getTradeByAlpacaId failed:", err.message); return null; }
}

function getTradesBySymbol(symbol, limit = 50) {
  try { return stmts.getTradesBySymbol.all({ symbol, limit }); }
  catch (err) { console.error("journal: getTradesBySymbol failed:", err.message); return []; }
}

function getTradesByStrategy(strategy, since = "2000-01-01") {
  try { return stmts.getTradesByStrategy.all({ strategy, since }); }
  catch (err) { console.error("journal: getTradesByStrategy failed:", err.message); return []; }
}

function getTradesByDateRange(from, to) {
  try { return stmts.getTradesByDateRange.all({ from, to }); }
  catch (err) { console.error("journal: getTradesByDateRange failed:", err.message); return []; }
}

function getRecentEvents(limit = 50) {
  try { return stmts.getRecentEvents.all({ limit }); }
  catch (err) { console.error("journal: getRecentEvents failed:", err.message); return []; }
}

function getEventsByType(eventType, limit = 50) {
  try { return stmts.getEventsByType.all({ event_type: eventType, limit }); }
  catch (err) { console.error("journal: getEventsByType failed:", err.message); return []; }
}

function getStrategyStats(since = "2000-01-01") {
  try { return stmts.getStrategyStats.all({ since }); }
  catch (err) { console.error("journal: getStrategyStats failed:", err.message); return []; }
}

function getDailySnapshots(from, to) {
  try { return stmts.getDailySnapshots.all({ from, to }); }
  catch (err) { console.error("journal: getDailySnapshots failed:", err.message); return []; }
}

function getPreviousSnapshot(beforeDate) {
  try { return stmts.getPreviousSnapshot.get({ before_date: beforeDate }); }
  catch (err) { console.error("journal: getPreviousSnapshot failed:", err.message); return null; }
}

function findEntryTrade(symbol) {
  try { return stmts.findEntryTrade.get({ symbol }); }
  catch (err) { console.error("journal: findEntryTrade failed:", err.message); return null; }
}

function findEntryTradeByStrategy(symbol, strategy) {
  try { return stmts.findEntryTradeByStrategy.get({ symbol, strategy }); }
  catch (err) { console.error("journal: findEntryTradeByStrategy failed:", err.message); return null; }
}

function findAllFilledSells() {
  try { return stmts.findAllFilledSells.all(); }
  catch (err) { console.error("journal: findAllFilledSells failed:", err.message); return []; }
}

function consumeEntryShares(entryId, consumed) {
  try { stmts.consumeEntryShares.run({ id: entryId, consumed }); }
  catch (err) { console.error("journal: consumeEntryShares failed:", err.message); }
}

function resetForRecompute() {
  try {
    stmts.resetConsumedQty.run();
    stmts.resetSellPnl.run();
  } catch (err) {
    console.error("journal: resetForRecompute failed:", err.message);
  }
}

function updateExitPnl({ id, realized_pnl, realized_pnl_pct, hold_days, exit_reason, entry_trade_id }) {
  try {
    stmts.updateExit.run({
      id,
      realized_pnl: Math.round(realized_pnl * 100) / 100,
      realized_pnl_pct: Math.round(realized_pnl_pct * 10000) / 10000,
      hold_days: hold_days != null ? hold_days : null,
      exit_reason: exit_reason || null,
      entry_trade_id: entry_trade_id || null,
    });
  } catch (err) {
    console.error("journal: updateExitPnl failed:", err.message);
  }
}

function getDb() { return db; }

// ══════════════════════════════════════════
//  EXPORTS
// ══════════════════════════════════════════

module.exports = {
  initDb,
  getDb,

  // Write
  recordOrderSubmitted,
  recordOrderFilled,
  recordOrderRejected,
  recordOrderCanceled,
  upsertPosition,
  updatePositionPrices,
  closePosition,
  removeStalePositions,
  writeDailySnapshot,
  recordSignal,
  markSignalActedOn,
  logEvent,

  // Read
  getOpenPositions,
  getTradeById,
  getTradeByAlpacaId,
  getTradesBySymbol,
  getTradesByStrategy,
  getTradesByDateRange,
  getRecentEvents,
  getEventsByType,
  getStrategyStats,
  getDailySnapshots,
  getPreviousSnapshot,
  findEntryTrade,
  findEntryTradeByStrategy,
  findAllFilledSells,
  consumeEntryShares,
  resetForRecompute,
  updateExitPnl,
};
