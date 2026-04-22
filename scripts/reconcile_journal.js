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

  // ── P&L Reconciliation Pass (FIFO with share consumption) ───────────
  // Full recompute: reset consumed_qty + sell P&L, then process ALL sells
  // chronologically so consumed_qty accumulates correctly across trades.
  let pnlMatched = 0, pnlSkipped = 0;

  if (!dryRun) {
    // Reset for clean recompute — order matters
    journal.resetForRecompute();
    log(`[pnl] Reset consumed_qty and sell P&L for full FIFO recompute`);

    const allSells = journal.findAllFilledSells();
    if (allSells.length > 0) {
      log(`[pnl] Processing ${allSells.length} filled sell trades...`);
    }

    for (const sell of allSells) {
      const sellQty = parseFloat(sell.filled_qty || sell.qty);
      const sellPrice = parseFloat(sell.fill_price);
      if (!sellPrice || sellQty <= 0) {
        log(`[pnl] SKIP sell #${sell.id} ${sell.symbol} — no fill_price or zero qty`);
        pnlSkipped++;
        continue;
      }

      let remainingQty = sellQty;
      let weightedEntryValue = 0;
      let totalEntryCommission = 0;
      let primaryEntryId = null;
      let primaryEntryConsumed = 0;
      let earliestEntryTime = null;

      // Consume shares from matching buys (FIFO, oldest first)
      while (remainingQty > 0.001) {
        // Try strategy-aware match first, then any-strategy fallback
        let entry = null;
        if (sell.strategy && sell.strategy !== "unknown") {
          entry = journal.findEntryTradeByStrategy(sell.symbol, sell.strategy);
        }
        if (!entry) {
          entry = journal.findEntryTrade(sell.symbol);
        }
        if (!entry || !entry.fill_price) break;

        const available = entry.remaining; // computed column from query
        const consumed = Math.min(available, remainingQty);

        weightedEntryValue += consumed * parseFloat(entry.fill_price);
        totalEntryCommission += (parseFloat(entry.commission || 0) * consumed / parseFloat(entry.filled_qty || entry.qty));
        remainingQty -= consumed;

        // Track primary entry (the one contributing most shares)
        if (!primaryEntryId || consumed > primaryEntryConsumed) {
          primaryEntryId = entry.id;
          primaryEntryConsumed = consumed;
        }
        if (!earliestEntryTime || (entry.fill_time && entry.fill_time < earliestEntryTime)) {
          earliestEntryTime = entry.fill_time;
        }

        // Mark shares as consumed on the buy trade
        journal.consumeEntryShares(entry.id, consumed);
      }

      const matchedQty = sellQty - remainingQty;
      if (matchedQty < 0.001) {
        log(`[pnl] SKIP ${sell.symbol} sell #${sell.id} — no matching buy found (legacy position?)`);
        pnlSkipped++;
        continue;
      }

      // Compute P&L from weighted average entry price
      const avgEntryPrice = weightedEntryValue / matchedQty;
      const sellCommission = parseFloat(sell.commission || 0);
      const realizedPnl = (sellPrice * matchedQty) - (avgEntryPrice * matchedQty) - sellCommission - totalEntryCommission;
      const costBasis = avgEntryPrice * matchedQty;
      const realizedPnlPct = costBasis > 0 ? realizedPnl / costBasis : 0;

      // Hold days from earliest entry to sell
      let holdDays = null;
      if (sell.fill_time && earliestEntryTime) {
        const sellTime = new Date(sell.fill_time).getTime();
        const entryTime = new Date(earliestEntryTime).getTime();
        holdDays = Math.max(0, Math.round((sellTime - entryTime) / (1000 * 60 * 60 * 24)));
      }

      journal.updateExitPnl({
        id: sell.id,
        realized_pnl: realizedPnl,
        realized_pnl_pct: realizedPnlPct,
        hold_days: holdDays,
        exit_reason: sell.exit_reason || null,
        entry_trade_id: primaryEntryId,
      });

      if (remainingQty > 0.001) {
        log(`[pnl] ${sell.symbol} sell #${sell.id} ← buy #${primaryEntryId}: ${realizedPnl >= 0 ? "+" : ""}$${realizedPnl.toFixed(2)} (${matchedQty}/${sellQty} shares matched, ${remainingQty.toFixed(0)} unmatched)`);
      } else {
        log(`[pnl] ${sell.symbol} sell #${sell.id} ← buy #${primaryEntryId}: ${realizedPnl >= 0 ? "+" : ""}$${realizedPnl.toFixed(2)} (${(realizedPnlPct * 100).toFixed(2)}%) held ${holdDays ?? "?"}d`);
      }
      pnlMatched++;
    }

    if (allSells.length > 0) {
      log(`[pnl] Done: ${pnlMatched} matched, ${pnlSkipped} skipped`);
    }

    // Verification: log total realized P&L
    try {
      const totalRow = db.prepare(`
        SELECT COALESCE(SUM(realized_pnl), 0) AS total
        FROM trades WHERE side = 'sell' AND status = 'filled' AND realized_pnl IS NOT NULL
      `).get();
      log(`[pnl] Verification: total realized = $${totalRow.total.toFixed(2)}`);

      // Check for over-consumption
      const overConsumed = db.prepare(`
        SELECT COUNT(*) AS cnt FROM trades
        WHERE side = 'buy' AND status = 'filled'
          AND COALESCE(consumed_qty, 0) > COALESCE(filled_qty, qty) + 0.01
      `).get();
      if (overConsumed.cnt > 0) {
        log(`[pnl] WARNING: ${overConsumed.cnt} buy trades have consumed_qty > filled_qty (over-consumption bug)`);
      }
    } catch (_) {}
  }

  const result = { updated, inserted, skipped, pnlMatched, pnlSkipped };
  log(`── Reconcile done: ${updated} updated, ${inserted} inserted, ${skipped} unchanged, ${pnlMatched} P&L matched ──`);

  // Log reconcile_run event
  if (!dryRun) {
    try {
      journal.logEvent({
        event_type: "reconcile_run",
        severity: "info",
        message: `Reconcile: ${updated} updated, ${inserted} inserted, ${skipped} unchanged, ${pnlMatched} P&L matched`,
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
