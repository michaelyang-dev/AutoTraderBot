#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Seed Legacy Positions — Import current Alpaca positions into journal
//
//  One-time script. Fetches current positions from Alpaca and inserts
//  them into the journal as legacy positions with corresponding buy
//  trade rows.
//
//  Usage:
//    node scripts/seed_legacy_positions.js --dry-run   # preview only
//    node scripts/seed_legacy_positions.js              # execute
// ══════════════════════════════════════════════════════════════════════

require("dotenv").config();

const Alpaca = require("@alpacahq/alpaca-trade-api");
const path = require("path");
const journal = require("../db/journal");

const DRY_RUN = process.argv.includes("--dry-run");
const TIMEOUT_MS = 15000;

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

const { ALPACA_API_KEY, ALPACA_SECRET_KEY } = process.env;
if (!ALPACA_API_KEY || !ALPACA_SECRET_KEY) {
  log("ERROR: Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env");
  process.exit(1);
}

const alpaca = new Alpaca({
  keyId: ALPACA_API_KEY,
  secretKey: ALPACA_SECRET_KEY,
  paper: true,
});

const { fromAlpacaSymbol } = require("../server/symbolMap");

async function seed() {
  log(`\n══ Seed Legacy Positions ${DRY_RUN ? "(DRY RUN)" : ""} ══\n`);

  // Init journal DB
  const dbPath = path.join(__dirname, "..", "data", "journal.db");
  journal.initDb(dbPath);

  // Fetch current positions with 15s timeout
  log("Fetching positions from Alpaca...");
  let positions;
  try {
    positions = await withTimeout(alpaca.getPositions(), TIMEOUT_MS, "getPositions");
  } catch (err) {
    log(`FATAL: ${err.message}`);
    process.exit(1);
  }

  log(`Fetched ${positions.length} positions`);

  if (positions.length === 0) {
    log("No positions found. Nothing to seed.");
    process.exit(0);
  }

  const now = new Date().toISOString();
  const db = journal.getDb();
  let seeded = 0, skipped = 0;
  const seededSymbols = [];

  for (const pos of positions) {
    const symbol = fromAlpacaSymbol(pos.symbol);
    const qty = parseFloat(pos.qty);
    const avgCost = parseFloat(pos.avg_entry_price);
    const currentPrice = pos.current_price ? parseFloat(pos.current_price) : null;
    const marketValue = pos.market_value ? parseFloat(pos.market_value) : null;
    const unrealizedPnl = pos.unrealized_pl ? parseFloat(pos.unrealized_pl) : null;
    const unrealizedPnlPct = pos.unrealized_plpc ? parseFloat(pos.unrealized_plpc) : null;
    const costBasis = qty * avgCost;
    const legacyOrderId = `LEGACY_${symbol}`;

    // Skip if already seeded (idempotent)
    const existing = journal.getTradeByAlpacaId(legacyOrderId);
    if (existing) {
      log(`  SKIP ${symbol} — already seeded (trade id ${existing.id})`);
      skipped++;
      continue;
    }

    if (DRY_RUN) {
      log(`  DRY RUN: Would seed ${symbol} — ${qty} shares @ $${avgCost.toFixed(2)} | current $${currentPrice?.toFixed(2) || "?"} | P&L $${unrealizedPnl?.toFixed(2) || "?"}`);
      seeded++;
      seededSymbols.push(symbol);
      continue;
    }

    // Single transaction per position: trade insert + position insert + linkage
    const seedOne = db.transaction(() => {
      // Insert buy trade row
      const tradeId = journal.recordOrderSubmitted({
        alpaca_order_id: legacyOrderId,
        symbol,
        side: "buy",
        qty,
        strategy: "legacy",
        regime: "UNKNOWN",
        intended_price: avgCost,
      });

      journal.recordOrderFilled({
        alpaca_order_id: legacyOrderId,
        filled_qty: qty,
        fill_price: avgCost,
        fill_time: now,
      });

      // Stamp notes and zero slippage
      db.prepare("UPDATE trades SET notes = ?, slippage_bps = 0 WHERE id = ?")
        .run("Seeded at journal deployment. No historical fill data. Actual entry predates journal.", tradeId);

      // Insert position row linked to the trade
      journal.upsertPosition({
        symbol,
        qty,
        avg_cost: avgCost,
        cost_basis: costBasis,
        current_price: currentPrice,
        current_value: marketValue,
        unrealized_pnl: unrealizedPnl,
        unrealized_pnl_pct: unrealizedPnlPct,
        strategy: "legacy",
        entry_regime: "UNKNOWN",
        entry_trade_id: tradeId,
        opened_at: now,
        days_held: 0,
        highest_price_since_entry: currentPrice || avgCost,
      });

      return tradeId;
    });

    const tradeId = seedOne();

    log(`  SEED ${symbol} — ${qty} shares @ $${avgCost.toFixed(2)} | current $${currentPrice?.toFixed(2) || "?"} | P&L $${unrealizedPnl?.toFixed(2) || "?"} | trade #${tradeId}`);
    seeded++;
    seededSymbols.push(symbol);
  }

  // Log seeding event
  if (seeded > 0 && !DRY_RUN) {
    journal.logEvent({
      event_type: "journal_seeded",
      severity: "info",
      message: `Seeded ${seeded} legacy positions`,
      metadata: { symbols: seededSymbols },
    });
  }

  log(`\n══ Seed complete ══`);
  log(`  Seeded: ${seeded}`);
  log(`  Skipped (already exists): ${skipped}`);

  if (DRY_RUN) {
    log("\n  DRY RUN — nothing was written to the database.");
  }
}

seed().catch((err) => {
  log(`FATAL: ${err.message}\n${err.stack}`);
  process.exit(1);
});
