#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Weekly Report — Runs Sunday evening at 6:00 PM ET
//
//  7-day aggregation: total P&L, strategy breakdown, trade counts,
//  best/worst day, drawdown, SPY comparison.
//
//  Usage:
//    node scripts/weekly_report.js
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

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

async function weeklyReport() {
  log("Weekly report starting...");

  const db = getJournalDb();
  const now = new Date();
  const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];
  const today = now.toISOString().split("T")[0];

  // ── 1. Portfolio current state ────────────────────────────────────
  let portfolio;
  try {
    portfolio = await getPortfolioState();
  } catch (err) {
    log(`FATAL: Cannot fetch portfolio: ${err.message}`);
    logScriptRun("weekly_report", "error", { error: err.message });
    await sendTelegram(`<b>❌ Weekly Report FAILED</b>\n\nCannot reach Alpaca: ${err.message}`);
    process.exit(1);
  }

  // ── 2. Daily snapshots for the week ───────────────────────────────
  const snapshots = db.prepare(`
    SELECT * FROM daily_snapshots
    WHERE date >= ? AND date <= ?
    ORDER BY date ASC
  `).all(sevenDaysAgo, today);

  let weekStartValue = null;
  let weekEndValue = portfolio.equity;
  let bestDay = null, worstDay = null;
  let spyWeekStart = null, spyWeekEnd = null;

  if (snapshots.length > 0) {
    weekStartValue = snapshots[0].portfolio_value;
    for (const snap of snapshots) {
      const dayPnlPct = snap.day_pnl_pct || 0;
      if (!bestDay || dayPnlPct > bestDay.pct) {
        bestDay = { date: snap.date, pct: dayPnlPct, pnl: snap.day_pnl || 0 };
      }
      if (!worstDay || dayPnlPct < worstDay.pct) {
        worstDay = { date: snap.date, pct: dayPnlPct, pnl: snap.day_pnl || 0 };
      }
    }
    spyWeekStart = snapshots[0].spy_close;
    spyWeekEnd = snapshots[snapshots.length - 1].spy_close;
  }

  const weekPnl = weekStartValue ? weekEndValue - weekStartValue : null;
  const weekPnlPct = weekStartValue ? weekPnl / weekStartValue : null;
  const spyWeekPct = spyWeekStart && spyWeekEnd ? (spyWeekEnd - spyWeekStart) / spyWeekStart : null;

  // ── 3. Trade stats for the week ───────────────────────────────────
  const weekTrades = db.prepare(`
    SELECT * FROM trades
    WHERE DATE(submitted_at) >= ? AND status = 'filled'
    ORDER BY submitted_at
  `).all(sevenDaysAgo);

  const weekBuys = weekTrades.filter(t => t.side === "buy");
  const weekSells = weekTrades.filter(t => t.side === "sell");
  const weekRealizedPnl = weekSells.reduce((sum, t) => sum + (t.realized_pnl || 0), 0);
  const winners = weekSells.filter(t => (t.realized_pnl || 0) > 0);
  const losers = weekSells.filter(t => (t.realized_pnl || 0) < 0);
  const winRate = weekSells.length > 0 ? winners.length / weekSells.length : null;

  // Strategy breakdown
  const stratBreakdown = {};
  for (const t of weekTrades) {
    const s = t.strategy || "unknown";
    if (!stratBreakdown[s]) stratBreakdown[s] = { buys: 0, sells: 0, pnl: 0 };
    if (t.side === "buy") stratBreakdown[s].buys++;
    else {
      stratBreakdown[s].sells++;
      stratBreakdown[s].pnl += t.realized_pnl || 0;
    }
  }

  // ── 4. Drawdown ───────────────────────────────────────────────────
  const dd = ddOpen(portfolio.equity);

  // ── 5. Build Telegram message ─────────────────────────────────────
  const weekEmoji = weekPnl != null && weekPnl >= 0 ? "📊" : "📉";

  let msg = `<b>${weekEmoji} Weekly Report — ${nowET()}</b>\n`;
  msg += `<i>Week of ${sevenDaysAgo} to ${today}</i>\n\n`;

  // Portfolio summary
  msg += `<b>Portfolio</b>\n`;
  msg += `Equity: <b>${formatMoney(portfolio.equity)}</b>\n`;
  if (weekPnl != null) {
    msg += `Week P&amp;L: <b>${formatMoney(weekPnl)}</b> (${formatPct(weekPnlPct)})\n`;
  }
  if (dd != null) msg += `Drawdown from peak: ${formatPct(dd)}\n`;
  msg += `Positions: ${portfolio.positionsCount} | Cash: ${formatMoney(portfolio.cash)}\n`;
  msg += `\n`;

  // SPY comparison
  if (spyWeekPct != null) {
    const alpha = weekPnlPct != null ? weekPnlPct - spyWeekPct : null;
    msg += `<b>vs SPY</b>\n`;
    msg += `SPY: ${formatPct(spyWeekPct)} | Portfolio: ${formatPct(weekPnlPct)}`;
    if (alpha != null) msg += ` | Alpha: ${formatPct(alpha)}`;
    msg += `\n\n`;
  }

  // Trade stats
  msg += `<b>Trades</b>\n`;
  msg += `Buys: ${weekBuys.length} | Sells: ${weekSells.length}\n`;
  if (weekSells.length > 0) {
    msg += `Realized P&amp;L: <b>${formatMoney(weekRealizedPnl)}</b>\n`;
    msg += `Win rate: ${winRate != null ? (winRate * 100).toFixed(0) + "%" : "—"} (${winners.length}W / ${losers.length}L)\n`;
  }
  msg += `\n`;

  // Strategy breakdown
  if (Object.keys(stratBreakdown).length > 0) {
    msg += `<b>Strategy Breakdown</b>\n`;
    for (const [strat, data] of Object.entries(stratBreakdown).sort((a, b) => b[1].pnl - a[1].pnl)) {
      const pnlStr = data.sells > 0 ? ` → ${formatMoney(data.pnl)}` : "";
      msg += `  ${strat}: ${data.buys}B / ${data.sells}S${pnlStr}\n`;
    }
    msg += `\n`;
  }

  // Best/worst day
  if (bestDay && worstDay && snapshots.length > 1) {
    msg += `<b>Best/Worst Days</b>\n`;
    msg += `  Best:  ${bestDay.date} → ${formatPct(bestDay.pct)} (${formatMoney(bestDay.pnl)})\n`;
    msg += `  Worst: ${worstDay.date} → ${formatPct(worstDay.pct)} (${formatMoney(worstDay.pnl)})\n`;
  }

  log(msg.replace(/<[^>]+>/g, ""));
  await sendTelegram(msg);

  // ── 6. Log event ──────────────────────────────────────────────────
  logScriptRun("weekly_report", "ok", {
    equity: portfolio.equity,
    weekPnl,
    weekPnlPct,
    buys: weekBuys.length,
    sells: weekSells.length,
    realizedPnl: weekRealizedPnl,
    winRate,
    snapshots: snapshots.length,
  });

  log("Weekly report complete.");
}

weeklyReport().catch((err) => {
  log(`FATAL: ${err.message}`);
  try { logScriptRun("weekly_report", "error", { error: err.message }); } catch (_) {}
  process.exit(1);
});
