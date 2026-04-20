#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Journal Backfill — Import historical Alpaca trades into journal DB
//
//  One-time import. Fetches all account activities (FILLs) since
//  account inception and inserts them into the journal.
//
//  - Current open positions get strategy='legacy'
//  - Historical closed trades are inserted as buy+sell pairs
//  - Computes realized P&L for sell trades (FIFO matching)
//
//  Usage:
//    node scripts/backfill_journal.js --dry-run   # preview only
//    node scripts/backfill_journal.js              # execute
// ═══════════════════════════════════════════════════════════════════���══

require("dotenv").config();

const Alpaca = require("@alpacahq/alpaca-trade-api");
const path = require("path");
const journal = require("../db/journal");

const DRY_RUN = process.argv.includes("--dry-run");

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
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

async function backfill() {
  log(`\n══ Journal Backfill ${DRY_RUN ? "(DRY RUN)" : ""} ══\n`);

  // Init journal DB
  const dbPath = path.join(__dirname, "..", "data", "journal.db");
  journal.initDb(dbPath);

  // ── 1. Fetch all fill activities ──
  log("Fetching all account activities (FILL type)...");
  let activities = [];
  let pageToken = null;

  do {
    const params = {
      activity_types: "FILL",
      direction: "asc",
      page_size: 100,
    };
    if (pageToken) params.page_token = pageToken;

    const page = await fetchWithRetry(() => alpaca.getAccountActivities(params), "getAccountActivities");
    if (!Array.isArray(page) || page.length === 0) break;

    activities.push(...page);
    // Alpaca pagination: last item's id is the page token for next page
    pageToken = page.length === 100 ? page[page.length - 1].id : null;
    await new Promise(r => setTimeout(r, 400));
  } while (pageToken);

  log(`Fetched ${activities.length} fill activities`);

  if (activities.length === 0) {
    log("No activities found. Nothing to backfill.");
    process.exit(0);
  }

  // ── 2. Fetch current positions for legacy tagging ──
  log("Fetching current positions...");
  const currentPositions = await fetchWithRetry(() => alpaca.getPositions(), "getPositions");
  const openSymbols = new Set(currentPositions.map(p => fromAlpacaSymbol(p.symbol)));
  log(`Open positions: ${[...openSymbols].join(", ") || "none"}`);

  // ── 3. Process activities ──
  let inserted = 0, skippedDup = 0;

  // Track buy inventory for FIFO P&L matching: { symbol: [{ qty, price, tradeId }] }
  const inventory = {};

  for (const act of activities) {
    const symbol = fromAlpacaSymbol(act.symbol);
    const side = act.side;
    const qty = parseFloat(act.qty);
    const price = parseFloat(act.price);
    const fillTime = act.transaction_time || act.timestamp;
    const orderId = act.order_id;

    // Skip if already in journal (idempotent)
    const existing = journal.getTradeByAlpacaId(orderId);
    if (existing) {
      skippedDup++;
      continue;
    }

    if (DRY_RUN) {
      log(`  DRY RUN: ${side.toUpperCase()} ${qty} ${symbol} @ $${price.toFixed(2)} (${fillTime})`);
      inserted++;
      continue;
    }

    // Determine strategy
    let strategy = "unknown";
    if (openSymbols.has(symbol) && side === "buy") {
      strategy = "legacy";
    }

    // Insert the trade
    const tradeId = journal.recordOrderSubmitted({
      alpaca_order_id: orderId,
      symbol,
      side,
      qty,
      strategy,
      intended_price: price,
    });

    journal.recordOrderFilled({
      alpaca_order_id: orderId,
      filled_qty: qty,
      fill_price: price,
      fill_time: fillTime,
    });

    // FIFO inventory tracking
    if (side === "buy") {
      if (!inventory[symbol]) inventory[symbol] = [];
      inventory[symbol].push({ qty, price, tradeId });
    } else if (side === "sell") {
      // Match against inventory (FIFO)
      let remainingQty = qty;
      let totalCost = 0;

      if (inventory[symbol] && inventory[symbol].length > 0) {
        while (remainingQty > 0 && inventory[symbol].length > 0) {
          const entry = inventory[symbol][0];
          const matchQty = Math.min(remainingQty, entry.qty);
          totalCost += matchQty * entry.price;
          entry.qty -= matchQty;
          remainingQty -= matchQty;
          if (entry.qty <= 0) inventory[symbol].shift();
        }

        const avgCost = totalCost / qty;
        const pnl = (price - avgCost) * qty;
        const pnlPct = avgCost > 0 ? (price - avgCost) / avgCost : 0;

        // Update the sell trade with P&L
        if (tradeId) {
          const db = journal.getDb();
          db.prepare(`
            UPDATE trades SET
              realized_pnl = ?, realized_pnl_pct = ?, exit_reason = 'backfill',
              updated_at = datetime('now')
            WHERE id = ?
          `).run(
            Math.round(pnl * 100) / 100,
            Math.round(pnlPct * 10000) / 10000,
            tradeId,
          );
        }
      }
    }

    inserted++;
  }

  // ── 4. Upsert current positions ──
  if (!DRY_RUN) {
    log("\nUpserting current positions...");
    for (const pos of currentPositions) {
      const symbol = fromAlpacaSymbol(pos.symbol);
      journal.upsertPosition({
        symbol,
        qty: parseFloat(pos.qty),
        avg_cost: parseFloat(pos.avg_entry_price),
        current_price: parseFloat(pos.current_price),
        unrealized_pnl: parseFloat(pos.unrealized_pl),
        unrealized_pnl_pct: parseFloat(pos.unrealized_plpc),
        strategy: "legacy",
        opened_at: new Date().toISOString(),
      });
      log(`  Position: ${symbol} — ${pos.qty} shares @ $${parseFloat(pos.avg_entry_price).toFixed(2)}`);
    }
  }

  // ── Summary ──
  log(`\n══ Backfill complete ══`);
  log(`  Activities processed: ${activities.length}`);
  log(`  Inserted: ${inserted}`);
  log(`  Skipped (duplicate): ${skippedDup}`);
  log(`  Current positions upserted: ${currentPositions.length}`);

  if (DRY_RUN) {
    log("\n  DRY RUN — nothing was written to the database.");
  }
}

backfill().catch((err) => {
  log(`FATAL: ${err.message}\n${err.stack}`);
  process.exit(1);
});
