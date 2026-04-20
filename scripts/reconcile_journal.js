#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Alpaca → Journal Reconciler
//
//  Pulls recent orders and fills from Alpaca and reconciles against
//  the journal DB. Designed to run every 5 min via cron or PM2.
//
//  - Updates status/fill_price/fill_time for known orders
//  - Inserts unknown orders with strategy='unknown'
//  - Computes slippage_bps for filled orders
//  - Idempotent: running twice changes nothing
//
//  Usage:
//    node scripts/reconcile_journal.js
//    node scripts/reconcile_journal.js --dry-run
// ══════════════════════════════════════════════════════════════════════

require("dotenv").config();

const Alpaca = require("@alpacahq/alpaca-trade-api");
const path = require("path");
const fs = require("fs");
const journal = require("../db/journal");

// ── Config ──
const LOG_DIR = path.join(__dirname, "..", "logs");
if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });

const LOG_FILE = path.join(LOG_DIR, "reconcile.log");
const DRY_RUN = process.argv.includes("--dry-run");

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  try { fs.appendFileSync(LOG_FILE, line + "\n"); } catch (_) {}
}

// ── Alpaca client ──
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

// Symbol mapping (same as tradingEngine.js)
const REVERSE_SYMBOL_MAP = { "BF.B": "BF-B", "BRK.B": "BRK-B", "BRK.A": "BRK-A" };
function fromAlpacaSymbol(sym) { return REVERSE_SYMBOL_MAP[sym] || sym; }

async function fetchWithRetry(fn, label, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const status = err.response?.status || err.statusCode;
      if (status === 429 && attempt < maxRetries) {
        log(`Rate limited on ${label}. Waiting 60s (attempt ${attempt}/${maxRetries})...`);
        await new Promise(r => setTimeout(r, 60000));
        continue;
      }
      throw err;
    }
  }
}

async function reconcile() {
  log(`\n── Reconcile start ${DRY_RUN ? "(DRY RUN)" : ""} ──`);

  // Init journal DB
  const dbPath = path.join(__dirname, "..", "data", "journal.db");
  journal.initDb(dbPath);

  // Fetch orders from last 24 hours
  const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  let orders;
  try {
    orders = await fetchWithRetry(() => alpaca.getOrders({
      status: "all",
      after: since,
      limit: 500,
      direction: "desc",
    }), "getOrders");
    log(`Fetched ${orders.length} orders from Alpaca (last 24h)`);
  } catch (err) {
    log(`ERROR fetching orders: ${err.message}`);
    process.exit(1);
  }

  let updated = 0, inserted = 0, skipped = 0;

  for (const order of orders) {
    const alpacaId = order.id;
    const symbol = fromAlpacaSymbol(order.symbol);
    const side = order.side;
    const qty = parseFloat(order.qty);
    const fillPrice = order.filled_avg_price ? parseFloat(order.filled_avg_price) : null;
    const filledQty = parseFloat(order.filled_qty || 0);
    const fillTime = order.filled_at || null;
    const status = order.status; // new, partially_filled, filled, done_for_day, canceled, expired, replaced, pending_cancel, pending_replace

    // Map Alpaca status to journal status
    let journalStatus;
    if (status === "filled") journalStatus = "filled";
    else if (status === "partially_filled") journalStatus = "partial";
    else if (["canceled", "expired", "replaced", "pending_cancel"].includes(status)) journalStatus = "canceled";
    else if (status === "new" || status === "accepted" || status === "pending_new") journalStatus = "pending";
    else journalStatus = "pending";

    // Check if we already have this order
    const existing = journal.getTradeByAlpacaId(alpacaId);

    if (existing) {
      // Already in journal — check if status needs updating
      if (existing.status === journalStatus && existing.filled_qty === filledQty) {
        skipped++;
        continue;
      }

      if (DRY_RUN) {
        log(`  DRY RUN: Would update ${symbol} ${side} (${existing.status} → ${journalStatus})`);
        updated++;
        continue;
      }

      if (journalStatus === "filled" && fillPrice) {
        journal.recordOrderFilled({
          alpaca_order_id: alpacaId,
          filled_qty: filledQty,
          fill_price: fillPrice,
          fill_time: fillTime,
        });
        log(`  Updated ${symbol} ${side}: ${existing.status} → filled @ $${fillPrice.toFixed(2)}`);
      } else if (journalStatus === "canceled") {
        journal.recordOrderCanceled({ alpaca_order_id: alpacaId });
        log(`  Updated ${symbol} ${side}: ${existing.status} → canceled`);
      }
      updated++;

    } else {
      // Not in journal — engine didn't write it (bug or manual trade)
      log(`  WARNING: Unknown order ${alpacaId} — ${symbol} ${side} ${qty} (${status})`);

      if (DRY_RUN) {
        log(`  DRY RUN: Would insert ${symbol} ${side} with strategy='unknown'`);
        inserted++;
        continue;
      }

      journal.recordOrderSubmitted({
        alpaca_order_id: alpacaId,
        client_order_id: order.client_order_id || null,
        symbol,
        side,
        qty,
        strategy: "unknown",
        intended_price: fillPrice || null,
      });

      if (journalStatus === "filled" && fillPrice) {
        journal.recordOrderFilled({
          alpaca_order_id: alpacaId,
          filled_qty: filledQty,
          fill_price: fillPrice,
          fill_time: fillTime,
        });
      } else if (journalStatus === "canceled") {
        journal.recordOrderCanceled({ alpaca_order_id: alpacaId });
      }

      inserted++;
    }
  }

  log(`── Reconcile done: ${updated} updated, ${inserted} inserted, ${skipped} unchanged ──`);
}

reconcile().catch((err) => {
  log(`FATAL: ${err.message}`);
  process.exit(1);
});
