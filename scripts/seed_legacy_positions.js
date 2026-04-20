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

const REVERSE_SYMBOL_MAP = { "BF.B": "BF-B", "BRK.B": "BRK-B", "BRK.A": "BRK-A" };
function fromAlpacaSymbol(sym) { return REVERSE_SYMBOL_MAP[sym] || sym; }

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
  let seeded = 0, skipped = 0;

  for (const pos of positions) {
    const symbol = fromAlpacaSymbol(pos.symbol);
    const qty = parseFloat(pos.qty);
    const avgCost = parseFloat(pos.avg_entry_price);
    const currentPrice = pos.current_price ? parseFloat(pos.current_price) : null;
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
      log(`  DRY RUN: Would seed ${symbol} — ${qty} shares @ $${avgCost.toFixed(2)} (cost basis $${costBasis.toFixed(2)})`);
      seeded++;
      continue;
    }

    // Insert buy trade row
    const tradeId = journal.recordOrderSubmitted({
      alpaca_order_id: legacyOrderId,
      symbol,
      side: "buy",
      qty,
      strategy: "legacy",
      intended_price: avgCost,
    });

    journal.recordOrderFilled({
      alpaca_order_id: legacyOrderId,
      filled_qty: qty,
      fill_price: avgCost,
      fill_time: now,
    });

    // Update notes on the trade
    const db = journal.getDb();
    db.prepare("UPDATE trades SET notes = ?, slippage_bps = 0 WHERE id = ?")
      .run("Seeded at journal deployment, no historical fill data", tradeId);

    // Insert position row linked to the trade
    journal.upsertPosition({
      symbol,
      qty,
      avg_cost: avgCost,
      cost_basis: costBasis,
      current_price: currentPrice,
      strategy: "legacy",
      entry_regime: "UNKNOWN",
      entry_trade_id: tradeId,
      opened_at: now,
    });

    log(`  SEED ${symbol} — ${qty} shares @ $${avgCost.toFixed(2)} | cost basis $${costBasis.toFixed(2)} | trade #${tradeId}`);
    seeded++;
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
