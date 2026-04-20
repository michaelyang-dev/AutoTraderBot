#!/usr/bin/env node
// ═══════════════════════���══════════════════════════════════════════════
//  Cleanup Unknown Trades — Remove pre-deployment unknown trades
//
//  One-time script. Deletes all trades with strategy='unknown' from
//  the journal and logs an event.
//
//  Usage:
//    node scripts/cleanup_unknown_trades.js --dry-run   # preview only
//    node scripts/cleanup_unknown_trades.js              # execute
// ════════���══════════════════════���══════════════════════════════════════

const path = require("path");
const journal = require("../db/journal");

const DRY_RUN = process.argv.includes("--dry-run");

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

const dbPath = path.join(__dirname, "..", "data", "journal.db");
journal.initDb(dbPath);
const db = journal.getDb();

// Count before deleting
const count = db.prepare("SELECT COUNT(*) as n FROM trades WHERE strategy = 'unknown'").get().n;
log(`Found ${count} trades with strategy='unknown'`);

if (count === 0) {
  log("Nothing to clean up.");
  process.exit(0);
}

if (DRY_RUN) {
  // Show a sample
  const sample = db.prepare("SELECT id, symbol, side, status, submitted_at FROM trades WHERE strategy = 'unknown' ORDER BY submitted_at DESC LIMIT 10").all();
  log(`Sample (most recent 10):`);
  for (const t of sample) {
    log(`  #${t.id} ${t.side} ${t.symbol} [${t.status}] ${t.submitted_at}`);
  }
  log(`\nDRY RUN — would delete ${count} rows. No changes made.`);
  process.exit(0);
}

const result = db.prepare("DELETE FROM trades WHERE strategy = 'unknown'").run();
log(`Deleted ${result.changes} unknown trades`);

journal.logEvent({
  event_type: "manual_override",
  severity: "info",
  message: `Cleaned up ${result.changes} pre-deployment unknown trades`,
  metadata: { deleted_count: result.changes },
});

log("Done. Event logged.");
