#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Weekly Report — Runs Sunday evening at 6:00 PM ET
//
//  7-day aggregation: total P&L, SPY comparison, strategy breakdown,
//  trade stats, win rate, avg hold, slippage, best/worst day, events.
//
//  Usage:
//    node scripts/weekly_report.js
// ══════════════════════════════════════════════════════════════════════

require("dotenv").config();

const {
  getAlpacaClient,
  getPortfolioState,
  fetchWithRetry,
  formatMoney,
  formatPct,
  formatTable,
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
    await sendTelegram(`<b>Weekly Report FAILED</b>\n\nCannot reach Alpaca: ${err.message}`);
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
  const hasHistory = snapshots.length >= 2;

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

  // ── 3. SPY 7-day return from Alpaca snapshot data ─────────────────
  let spyWeekPct = null;
  // Prefer snapshot-derived SPY data from daily_snapshots
  if (spyWeekStart && spyWeekEnd && spyWeekStart > 0) {
    spyWeekPct = (spyWeekEnd - spyWeekStart) / spyWeekStart;
  } else {
    // Fallback: try Alpaca getBarsV2
    try {
      const alpaca = getAlpacaClient();
      const bars = [];
      const iter = alpaca.getBarsV2("SPY", {
        timeframe: "1Day",
        start: sevenDaysAgo,
        adjustment: "split",
      });
      for await (const bar of iter) {
        bars.push(bar);
      }
      if (bars.length >= 2) {
        const first = parseFloat(bars[0].c);
        const last = parseFloat(bars[bars.length - 1].c);
        if (first > 0) spyWeekPct = (last - first) / first;
      }
    } catch (err) {
      log(`SPY bars fetch failed (non-fatal): ${err.message}`);
    }
  }

  const alpha = (weekPnlPct != null && spyWeekPct != null) ? weekPnlPct - spyWeekPct : null;

  // ── 4. Trade stats for the week ───────────────────────────────────
  const weekTrades = db.prepare(`
    SELECT * FROM trades
    WHERE DATE(submitted_at) >= ? AND status = 'filled'
    ORDER BY submitted_at
  `).all(sevenDaysAgo);

  const weekBuys = weekTrades.filter(t => t.side === "buy");
  const weekSells = weekTrades.filter(t => t.side === "sell");
  const weekRealizedPnl = weekSells.reduce((sum, t) => sum + (t.realized_pnl || 0), 0);

  // Closed trades (sells excluding legacy)
  const closedTrades = weekSells.filter(t => t.strategy !== "legacy");
  const winners = closedTrades.filter(t => (t.realized_pnl || 0) > 0);
  const losers = closedTrades.filter(t => (t.realized_pnl || 0) < 0);
  const winRate = closedTrades.length > 0 ? winners.length / closedTrades.length : null;

  // Avg hold days
  const holdDays = closedTrades.filter(t => t.hold_days != null).map(t => t.hold_days);
  const avgHold = holdDays.length > 0 ? holdDays.reduce((a, b) => a + b, 0) / holdDays.length : null;

  // Avg slippage (excluding legacy)
  const slippageRow = db.prepare(`
    SELECT AVG(slippage_bps) as avg_slip, COUNT(*) as cnt
    FROM trades
    WHERE DATE(fill_time) >= ? AND strategy != 'legacy' AND slippage_bps IS NOT NULL
  `).get(sevenDaysAgo);

  // Per-strategy breakdown
  const stratBreakdown = {};
  for (const t of weekTrades) {
    const s = t.strategy || "unknown";
    if (!stratBreakdown[s]) stratBreakdown[s] = { buys: 0, sells: 0, pnl: 0, wins: 0, losses: 0 };
    if (t.side === "buy") {
      stratBreakdown[s].buys++;
    } else {
      stratBreakdown[s].sells++;
      stratBreakdown[s].pnl += t.realized_pnl || 0;
      if ((t.realized_pnl || 0) > 0) stratBreakdown[s].wins++;
      else if ((t.realized_pnl || 0) < 0) stratBreakdown[s].losses++;
    }
  }

  // Unrealized by strategy from positions
  const posRows = db.prepare(`SELECT strategy, SUM(unrealized_pnl) as unr FROM positions GROUP BY strategy`).all();
  const stratUnrealized = {};
  for (const r of posRows) {
    stratUnrealized[r.strategy || "unknown"] = r.unr || 0;
  }
  const totalUnrealized = portfolio.positions.reduce((sum, p) => sum + p.unrealizedPnl, 0);

  // ── 5. Drawdown ───────────────────────────────────────────────────
  const dd = ddOpen(portfolio.equity);

  // ── 6. Events summary ─────────────────────────────────────────────
  const cbEvents = db.prepare(`
    SELECT COUNT(*) as cnt FROM events
    WHERE event_type = 'circuit_breaker' AND DATE(created_at) >= ?
  `).get(sevenDaysAgo).cnt;

  const regimeEvents = db.prepare(`
    SELECT COUNT(*) as cnt FROM events
    WHERE event_type = 'regime_change' AND DATE(created_at) >= ?
  `).get(sevenDaysAgo).cnt;

  const errorEvents = db.prepare(`
    SELECT COUNT(*) as cnt FROM events
    WHERE severity IN ('error', 'critical') AND DATE(created_at) >= ?
  `).get(sevenDaysAgo).cnt;

  // ── 7. Build Telegram message ─────────────────────────────────────
  const weekEmoji = weekPnl != null && weekPnl >= 0 ? "📊" : "📉";

  let msg = `<b>${weekEmoji} Weekly Report -- ${nowET()}</b>\n`;
  msg += `<i>Week of ${sevenDaysAgo} to ${today}</i>\n\n`;

  // Portfolio summary
  msg += `<b>Portfolio</b>\n`;
  msg += `Equity: <b>${formatMoney(portfolio.equity)}</b>\n`;
  if (weekPnl != null && hasHistory) {
    msg += `Week P&L: <b>${formatMoney(weekPnl)}</b> (${formatPct(weekPnlPct)})\n`;
  } else {
    msg += `Week P&L: -- (insufficient history)\n`;
  }
  if (dd != null) msg += `Drawdown from peak: ${formatPct(dd)}\n`;
  msg += `Positions: ${portfolio.positionsCount} | Cash: ${formatMoney(portfolio.cash)}\n`;
  msg += `Unrealized P&L: ${formatMoney(totalUnrealized)}\n`;
  msg += `\n`;

  // SPY comparison
  if (spyWeekPct != null && weekPnlPct != null && hasHistory) {
    msg += `<b>vs SPY</b>\n`;
    msg += `SPY: ${formatPct(spyWeekPct)} | Portfolio: ${formatPct(weekPnlPct)}`;
    if (alpha != null) msg += ` | Alpha: ${formatPct(alpha)}`;
    msg += `\n\n`;
  }

  // Trade stats
  msg += `<b>Trades</b>\n`;
  msg += `Total: ${weekTrades.length} (${weekBuys.length}B / ${weekSells.length}S)\n`;
  if (weekSells.length > 0) {
    msg += `Realized P&L: <b>${formatMoney(weekRealizedPnl)}</b>\n`;
  }
  if (closedTrades.length > 0) {
    msg += `Win rate: ${winRate != null ? (winRate * 100).toFixed(0) + "%" : "--"} (${winners.length}W / ${losers.length}L)\n`;
  }
  if (avgHold != null) {
    msg += `Avg hold: ${avgHold.toFixed(1)} days\n`;
  }
  if (slippageRow && slippageRow.cnt > 0) {
    msg += `Avg slippage: ${slippageRow.avg_slip.toFixed(1)} bps (${slippageRow.cnt} fills)\n`;
  }
  msg += `\n`;

  // Strategy breakdown table
  const allStrats = new Set([...Object.keys(stratBreakdown), ...Object.keys(stratUnrealized)]);
  if (allStrats.size > 0) {
    const stratHeaders = ["Strategy", "Trades", "Win%", "Realized", "Unrealized"];
    const stratRows = [];
    for (const s of [...allStrats].sort()) {
      const r = stratBreakdown[s] || { buys: 0, sells: 0, pnl: 0, wins: 0, losses: 0 };
      const u = stratUnrealized[s];
      const trades = r.buys + r.sells > 0 ? `${r.buys}B/${r.sells}S` : "--";
      const totalClosed = r.wins + r.losses;
      const winPct = totalClosed > 0 ? `${((r.wins / totalClosed) * 100).toFixed(0)}%` : "--";
      const realized = r.sells > 0 ? formatMoney(r.pnl) : "--";
      const unrealized = u != null ? formatMoney(u) : "--";
      stratRows.push([s, trades, winPct, realized, unrealized]);
    }
    msg += `<b>Strategy Breakdown</b>\n`;
    msg += `<pre>${formatTable(stratHeaders, stratRows)}</pre>\n`;
    msg += `\n`;
  }

  // Best/worst day
  if (bestDay && worstDay && hasHistory) {
    msg += `<b>Best/Worst Days</b>\n`;
    msg += `  Best:  ${bestDay.date} ${formatPct(bestDay.pct)} (${formatMoney(bestDay.pnl)})\n`;
    msg += `  Worst: ${worstDay.date} ${formatPct(worstDay.pct)} (${formatMoney(worstDay.pnl)})\n`;
    msg += `\n`;
  }

  // Events summary
  msg += `<b>Events</b>\n`;
  const eventLines = [];
  if (cbEvents > 0) eventLines.push(`Circuit breaker: ${cbEvents}x`);
  if (regimeEvents > 0) eventLines.push(`Regime change: ${regimeEvents}x`);
  if (errorEvents > 0) eventLines.push(`Errors: ${errorEvents}x`);
  if (eventLines.length > 0) {
    msg += eventLines.map(l => `  ${l}`).join("\n") + "\n";
  } else {
    msg += `No circuit breaker, regime, or error events.\n`;
  }

  log(msg.replace(/<[^>]+>/g, ""));
  await sendTelegram(msg);

  // ── 8. Log event ──────────────────────────────────────────────────
  logScriptRun("weekly_report", "ok", {
    equity: portfolio.equity,
    weekPnl,
    weekPnlPct,
    spyWeekPct,
    alpha,
    buys: weekBuys.length,
    sells: weekSells.length,
    realizedPnl: weekRealizedPnl,
    winRate,
    avgHold,
    snapshots: snapshots.length,
  });

  log("Weekly report complete.");
}

weeklyReport().catch((err) => {
  log(`FATAL: ${err.message}`);
  try { logScriptRun("weekly_report", "error", { error: err.message }); } catch (_) {}
  process.exit(1);
});
