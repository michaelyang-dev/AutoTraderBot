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

const { fromAlpacaSymbol } = require("../server/symbolMap");

const TIMEOUT_MS = 15000;

function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function fetchWithRetry(fn, label, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await withTimeout(fn(), TIMEOUT_MS, label);
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

async function runReconcile({ dryRun = false } = {}) {
  log(`\n── Reconcile start ${dryRun ? "(DRY RUN)" : ""} ──`);

  // Init journal DB
  const dbPath = path.join(__dirname, "..", "data", "journal.db");
  journal.initDb(dbPath);

  // Determine the earliest date to fetch — only pull orders after our first real trade
  const db = journal.getDb();
  const earliest = db.prepare("SELECT MIN(submitted_at) as earliest FROM trades WHERE strategy != 'unknown'").get();
  const fallback24h = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  const since = earliest?.earliest || fallback24h;
  log(`[reconcile] Fetching Alpaca orders since ${since}${earliest?.earliest ? "" : " (fallback 24h — no real trades yet)"}`);

  let orders;
  try {
    orders = await fetchWithRetry(() => alpaca.getOrders({
      status: "all",
      after: since,
      limit: 500,
      direction: "desc",
    }), "getOrders");
    log(`Fetched ${orders.length} orders from Alpaca`);
  } catch (err) {
    log(`ERROR fetching orders: ${err.message}`);
    throw err;
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

      if (dryRun) {
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

      if (dryRun) {
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

  const result = { updated, inserted, skipped };
  log(`── Reconcile done: ${updated} updated, ${inserted} inserted, ${skipped} unchanged ──`);

  // Log reconcile_run event
  if (!dryRun) {
    try {
      journal.logEvent({
        event_type: "reconcile_run",
        severity: "info",
        message: `Reconcile: ${updated} updated, ${inserted} inserted, ${skipped} unchanged`,
        metadata: result,
      });
    } catch (_) {}
  }

  return result;
}

// ── CLI entry point ──────────────────────────────────────────────────

if (require.main === module) {
  runReconcile({ dryRun: DRY_RUN }).catch((err) => {
    log(`FATAL: ${err.message}`);
    process.exit(1);
  });
}

module.exports = { runReconcile };
