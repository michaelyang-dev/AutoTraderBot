#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Daily Report — Runs at 4:15 PM ET after market close
//
//  1. Invokes reconciler to pull latest fills
//  2. Fetches portfolio metrics from Alpaca + journal
//  3. Builds strategy breakdown, top movers, SPY comparison
//  4. Sends full daily report via Telegram
//
//  Usage:
//    node scripts/daily_report.js
// ══════════════════════════════════════════════════════════════════════

require("dotenv").config();

const {
  getPortfolioState,
  getSpyDayReturn,
  formatMoney,
  formatPct,
  formatTable,
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
    await sendTelegram(`<b>Daily Report FAILED</b>\n\nCannot reach Alpaca: ${err.message}`);
    process.exit(1);
  }

  // ── 3. Journal metrics ────────────────────────────────────────────
  const db = getJournalDb();
  const today = new Date().toISOString().split("T")[0];

  // Day P&L — priority: today's snapshot > yesterday's snapshot > Alpaca last_equity
  let dayPnl = null, dayPnlPct = null, dayPnlSource = null;

  const todaySnap = db.prepare(`SELECT * FROM daily_snapshots WHERE date = ?`).get(today);
  const yesterdaySnap = db.prepare(`SELECT * FROM daily_snapshots WHERE date < ? ORDER BY date DESC LIMIT 1`).get(today);

  if (todaySnap && todaySnap.portfolio_value > 0) {
    // (a) Today's snapshot exists — compare current equity to it
    dayPnl = portfolio.equity - todaySnap.portfolio_value;
    dayPnlPct = dayPnl / todaySnap.portfolio_value;
    dayPnlSource = "vs today's open snapshot";
  } else if (yesterdaySnap && yesterdaySnap.portfolio_value > 0) {
    // (b) Yesterday's snapshot
    dayPnl = portfolio.equity - yesterdaySnap.portfolio_value;
    dayPnlPct = dayPnl / yesterdaySnap.portfolio_value;
    dayPnlSource = "vs yesterday's close";
  } else if (portfolio.lastEquity > 0) {
    // (c) Alpaca's last_equity fallback
    dayPnl = portfolio.equity - portfolio.lastEquity;
    dayPnlPct = dayPnl / portfolio.lastEquity;
    dayPnlSource = "vs Alpaca last_equity -- no snapshot baseline yet";
  }
  // (d) If none available, dayPnl stays null

  // Today's trades
  const todayTrades = db.prepare(`
    SELECT * FROM trades
    WHERE DATE(submitted_at) = ? AND status = 'filled'
    ORDER BY submitted_at
  `).all(today);

  const buys = todayTrades.filter(t => t.side === "buy");
  const sells = todayTrades.filter(t => t.side === "sell");
  const realizedPnl = sells.reduce((sum, t) => sum + (t.realized_pnl || 0), 0);

  // Strategy breakdown — realized from sells, unrealized from positions
  const stratRealized = {};
  for (const t of todayTrades) {
    const s = t.strategy || "unknown";
    if (!stratRealized[s]) stratRealized[s] = { buys: 0, sells: 0, pnl: 0 };
    if (t.side === "buy") stratRealized[s].buys++;
    else {
      stratRealized[s].sells++;
      stratRealized[s].pnl += t.realized_pnl || 0;
    }
  }

  // Unrealized by strategy from journal positions
  const posRows = db.prepare(`SELECT strategy, SUM(unrealized_pnl) as unr FROM positions GROUP BY strategy`).all();
  const stratUnrealized = {};
  for (const r of posRows) {
    stratUnrealized[r.strategy || "unknown"] = r.unr || 0;
  }

  // Merge all strategy names
  const allStrats = new Set([...Object.keys(stratRealized), ...Object.keys(stratUnrealized)]);

  // Top movers (open positions sorted by unrealized P&L %)
  const topMovers = [...portfolio.positions]
    .sort((a, b) => Math.abs(b.unrealizedPnlPct) - Math.abs(a.unrealizedPnlPct))
    .slice(0, 5);

  // Drawdown from peak
  const dd = ddOpen(portfolio.equity);

  // SPY benchmark
  const spy = await getSpyDayReturn();

  // Average slippage today (excluding legacy)
  const slippageRow = db.prepare(`
    SELECT AVG(slippage_bps) as avg_slip, COUNT(*) as cnt
    FROM trades
    WHERE DATE(fill_time) = ? AND strategy != 'legacy' AND slippage_bps IS NOT NULL
  `).get(today);

  // Events summary (warnings/errors today)
  const eventSummary = db.prepare(`
    SELECT event_type, severity, COUNT(*) as cnt
    FROM events
    WHERE DATE(created_at) = ? AND severity IN ('warning', 'error', 'critical')
    GROUP BY event_type, severity
    ORDER BY cnt DESC
  `).all(today);

  // ── 4. Build Telegram message ─────────────────────────────────────
  const dayEmoji = (dayPnl != null && dayPnl >= 0) ? "📈" : "📉";

  let msg = `<b>${dayEmoji} Daily Report -- ${nowET()}</b>\n\n`;

  // Portfolio summary
  msg += `<b>Portfolio</b>\n`;
  msg += `Equity: <b>${formatMoney(portfolio.equity)}</b>\n`;
  if (dayPnl != null) {
    msg += `Day P&L: <b>${formatMoney(dayPnl)}</b> (${formatPct(dayPnlPct)})\n`;
    if (dayPnlSource) msg += `<i>${dayPnlSource}</i>\n`;
  } else {
    msg += `Day P&L: -- (no baseline yet, will populate tomorrow)\n`;
  }
  msg += `Cash: ${formatMoney(portfolio.cash)}\n`;
  msg += `Positions: ${portfolio.positionsCount}\n`;
  if (dd != null) msg += `Drawdown: ${formatPct(dd)}\n`;
  msg += `\n`;

  // SPY benchmark
  if (spy.ok && dayPnlPct != null) {
    const alpha = dayPnlPct - spy.dayPct;
    msg += `<b>Benchmark</b>\n`;
    msg += `SPY: ${formatPct(spy.dayPct)} | Portfolio: ${formatPct(dayPnlPct)} | Alpha: ${formatPct(alpha)}\n`;
    msg += `\n`;
  }

  // Trading activity
  msg += `<b>Trades Today</b>\n`;
  msg += `Buys: ${buys.length} | Sells: ${sells.length}\n`;
  if (sells.length > 0) {
    msg += `Realized P&L: <b>${formatMoney(realizedPnl)}</b>\n`;
  }
  if (slippageRow && slippageRow.cnt > 0) {
    msg += `Avg slippage: ${slippageRow.avg_slip.toFixed(1)} bps (${slippageRow.cnt} fills)\n`;
  }
  msg += `\n`;

  // Strategy breakdown table
  if (allStrats.size > 0) {
    const stratHeaders = ["Strategy", "Trades", "Realized", "Unrealized"];
    const stratRows = [];
    for (const s of [...allStrats].sort()) {
      const r = stratRealized[s] || { buys: 0, sells: 0, pnl: 0 };
      const u = stratUnrealized[s];
      const trades = r.buys + r.sells > 0 ? `${r.buys}B/${r.sells}S` : "--";
      const realized = r.sells > 0 ? formatMoney(r.pnl) : "--";
      const unrealized = u != null ? formatMoney(u) : "--";
      stratRows.push([s, trades, realized, unrealized]);
    }
    msg += `<b>Strategy Breakdown</b>\n`;
    msg += `<pre>${formatTable(stratHeaders, stratRows)}</pre>\n`;
    msg += `\n`;
  }

  // Top movers
  if (topMovers.length > 0) {
    msg += `<b>Top Movers</b>\n`;
    for (const p of topMovers) {
      const sign = p.unrealizedPnl >= 0 ? "+" : "";
      msg += `  ${p.symbol}: ${sign}${formatPct(p.unrealizedPnlPct)} (${formatMoney(p.unrealizedPnl)})\n`;
    }
    msg += `\n`;
  }

  // Events summary
  if (eventSummary.length > 0) {
    msg += `<b>Events</b>\n`;
    for (const e of eventSummary) {
      const icon = e.severity === "critical" ? "🚨" : e.severity === "error" ? "❌" : "⚠️";
      msg += `  ${icon} ${e.event_type} (${e.severity}): ${e.cnt}x\n`;
    }
    msg += `\n`;
  } else {
    msg += `<b>Events</b>\nNo warnings or errors.\n\n`;
  }

  // Comparison to yesterday
  if (yesterdaySnap) {
    const change = portfolio.equity - yesterdaySnap.portfolio_value;
    const changePct = yesterdaySnap.portfolio_value > 0
      ? change / yesterdaySnap.portfolio_value
      : 0;
    msg += `<b>vs Yesterday</b>\n`;
    msg += `Value: ${formatMoney(yesterdaySnap.portfolio_value)} -> ${formatMoney(portfolio.equity)} (${formatPct(changePct)})\n`;
  }

  log(msg.replace(/<[^>]+>/g, ""));
  await sendTelegram(msg);

  // ── 5. Log event ──────────────────────────────────────────────────
  logScriptRun("daily_report", "ok", {
    equity: portfolio.equity,
    dayPnl,
    dayPnlSource,
    buys: buys.length,
    sells: sells.length,
    realizedPnl,
    positionsCount: portfolio.positionsCount,
    spyDayPct: spy.ok ? spy.dayPct : null,
  });

  log("Daily report complete.");
}

dailyReport().catch((err) => {
  log(`FATAL: ${err.message}`);
  try { logScriptRun("daily_report", "error", { error: err.message }); } catch (_) {}
  process.exit(1);
});
