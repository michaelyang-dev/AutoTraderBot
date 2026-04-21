#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Daily Report — Runs at 4:15 PM ET after market close
//
//  1. Invokes reconciler to pull latest fills
//  2. Fetches portfolio metrics from Alpaca + journal
//  3. Builds strategy breakdown, top movers
//  4. Sends full daily report via Telegram
//
//  Usage:
//    node scripts/daily_report.js
// ══════════════════════════════════════════════════════════════════════

require("dotenv").config();

const {
  getPortfolioState,
  formatMoney,
  formatPct,
  sendTelegram,
  getJournalDb,
  logScriptRun,
  nowET,
  ddOpen,
} = require("./lib/automationHelpers");
const { runReconcile } = require("./reconcile_journal");

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

async function dailyReport() {
  log("Daily report starting...");

  // ── 1. Run reconciler first ───────────────────────────────────────
  try {
    const reconcileResult = await runReconcile();
    log(`Reconcile: ${reconcileResult.updated} updated, ${reconcileResult.inserted} inserted`);
  } catch (err) {
    log(`Reconcile failed (continuing): ${err.message}`);
  }

  // ── 2. Fetch portfolio state ──────────────────────────────────────
  let portfolio;
  try {
    portfolio = await getPortfolioState();
  } catch (err) {
    log(`FATAL: Cannot fetch portfolio: ${err.message}`);
    logScriptRun("daily_report", "error", { error: err.message });
    await sendTelegram(`<b>❌ Daily Report FAILED</b>\n\nCannot reach Alpaca: ${err.message}`);
    process.exit(1);
  }

  // ── 3. Journal metrics ────────────────────────────────────────────
  const db = getJournalDb();
  const today = new Date().toISOString().split("T")[0];

  // Today's trades
  const todayTrades = db.prepare(`
    SELECT * FROM trades
    WHERE DATE(submitted_at) = ? AND status = 'filled'
    ORDER BY submitted_at
  `).all(today);

  const buys = todayTrades.filter(t => t.side === "buy");
  const sells = todayTrades.filter(t => t.side === "sell");
  const realizedPnl = sells.reduce((sum, t) => sum + (t.realized_pnl || 0), 0);

  // Strategy breakdown from today's trades
  const stratBreakdown = {};
  for (const t of todayTrades) {
    const s = t.strategy || "unknown";
    if (!stratBreakdown[s]) stratBreakdown[s] = { buys: 0, sells: 0, pnl: 0 };
    if (t.side === "buy") stratBreakdown[s].buys++;
    else {
      stratBreakdown[s].sells++;
      stratBreakdown[s].pnl += t.realized_pnl || 0;
    }
  }

  // Top movers (open positions sorted by unrealized P&L %)
  const topMovers = [...portfolio.positions]
    .sort((a, b) => Math.abs(b.unrealizedPnlPct) - Math.abs(a.unrealizedPnlPct))
    .slice(0, 5);

  // Drawdown from peak
  const dd = ddOpen(portfolio.equity);

  // Yesterday's snapshot for comparison
  const yesterdaySnap = db.prepare(`
    SELECT * FROM daily_snapshots WHERE date < ? ORDER BY date DESC LIMIT 1
  `).get(today);

  // ── 4. Build Telegram message ─────────────────────────────────────
  const dayEmoji = portfolio.dayPnl >= 0 ? "📈" : "📉";

  let msg = `<b>${dayEmoji} Daily Report — ${nowET()}</b>\n\n`;

  // Portfolio summary
  msg += `<b>Portfolio</b>\n`;
  msg += `Equity: <b>${formatMoney(portfolio.equity)}</b>\n`;
  msg += `Day P&amp;L: <b>${formatMoney(portfolio.dayPnl)}</b> (${formatPct(portfolio.dayPnlPct)})\n`;
  msg += `Cash: ${formatMoney(portfolio.cash)}\n`;
  msg += `Positions: ${portfolio.positionsCount}\n`;
  if (dd != null) msg += `Drawdown: ${formatPct(dd)}\n`;
  msg += `\n`;

  // Trading activity
  msg += `<b>Trades Today</b>\n`;
  msg += `Buys: ${buys.length} | Sells: ${sells.length}\n`;
  if (sells.length > 0) {
    msg += `Realized P&amp;L: <b>${formatMoney(realizedPnl)}</b>\n`;
  }
  msg += `\n`;

  // Strategy breakdown
  if (Object.keys(stratBreakdown).length > 0) {
    msg += `<b>Strategy Breakdown</b>\n`;
    for (const [strat, data] of Object.entries(stratBreakdown)) {
      const pnlStr = data.sells > 0 ? ` → ${formatMoney(data.pnl)}` : "";
      msg += `  ${strat}: ${data.buys}B / ${data.sells}S${pnlStr}\n`;
    }
    msg += `\n`;
  }

  // Top movers
  if (topMovers.length > 0) {
    msg += `<b>Top Movers</b>\n`;
    for (const p of topMovers) {
      const arrow = p.unrealizedPnl >= 0 ? "▲" : "▼";
      msg += `  ${p.symbol}: ${arrow} ${formatPct(p.unrealizedPnlPct)} (${formatMoney(p.unrealizedPnl)})\n`;
    }
    msg += `\n`;
  }

  // Comparison to yesterday
  if (yesterdaySnap) {
    const change = portfolio.equity - yesterdaySnap.portfolio_value;
    const changePct = yesterdaySnap.portfolio_value > 0
      ? change / yesterdaySnap.portfolio_value
      : 0;
    msg += `<b>vs Yesterday</b>\n`;
    msg += `Value: ${formatMoney(yesterdaySnap.portfolio_value)} → ${formatMoney(portfolio.equity)} (${formatPct(changePct)})\n`;
  }

  log(msg.replace(/<[^>]+>/g, ""));
  await sendTelegram(msg);

  // ── 5. Log event ──────────────────────────────────────────────────
  logScriptRun("daily_report", "ok", {
    equity: portfolio.equity,
    dayPnl: portfolio.dayPnl,
    buys: buys.length,
    sells: sells.length,
    realizedPnl,
    positionsCount: portfolio.positionsCount,
  });

  log("Daily report complete.");
}

dailyReport().catch((err) => {
  log(`FATAL: ${err.message}`);
  try { logScriptRun("daily_report", "error", { error: err.message }); } catch (_) {}
  process.exit(1);
});
