#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Hourly Heartbeat — Silent unless something is broken
//
//  Runs every hour during market hours (9:30 AM – 4:00 PM ET).
//  Only sends a Telegram alert if a problem is detected:
//    - PM2 process down
//    - Journal DB not writable
//    - Drawdown exceeds threshold
//    - Error event spike in the last hour
//    - Regime change since last check
//
//  Usage:
//    node scripts/hourly_heartbeat.js
// ══════════════════════════════════════════════════════════════════════

require("dotenv").config();

const {
  getPortfolioState,
  checkServicesHealth,
  formatMoney,
  formatPct,
  sendTelegram,
  getJournalDb,
  logScriptRun,
  nowET,
  ddOpen,
} = require("./lib/automationHelpers");

const DRAWDOWN_ALERT_THRESHOLD = -0.02; // alert at -2% from peak
const ERROR_SPIKE_THRESHOLD = 5;         // alert if >5 errors in last hour

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

async function heartbeat() {
  log("Hourly heartbeat starting...");
  const alerts = [];

  // ── 1. PM2 processes (continuous vs cron aware) ────────────────────
  try {
    const svcChecks = checkServicesHealth();
    for (const c of svcChecks) {
      if (c.status === "FAIL") {
        alerts.push(`❌ <b>${c.name}:</b> ${c.detail}`);
      } else if (c.status === "WARN") {
        alerts.push(`⚠️ <b>${c.name}:</b> ${c.detail}`);
      }
    }
  } catch (err) {
    alerts.push(`⚠️ <b>PM2 check failed:</b> ${err.message}`);
  }

  // ── 2. Journal writable ───────────────────────────────────────────
  try {
    const db = getJournalDb();
    // Quick write test
    db.prepare("INSERT INTO events (event_type, severity, message, created_at) VALUES ('hourly_heartbeat', 'info', 'heartbeat probe', datetime('now'))").run();
  } catch (err) {
    alerts.push(`❌ <b>Journal DB not writable:</b> ${err.message}`);
  }

  // ── 3. Drawdown check ─────────────────────────────────────────────
  let portfolio = null;
  try {
    portfolio = await getPortfolioState();
    const dd = ddOpen(portfolio.equity);
    if (dd != null && dd < DRAWDOWN_ALERT_THRESHOLD) {
      alerts.push(`🔴 <b>Drawdown alert:</b> ${formatPct(dd)} from peak (equity: ${formatMoney(portfolio.equity)})`);
    }
  } catch (err) {
    alerts.push(`⚠️ <b>Alpaca unreachable:</b> ${err.message}`);
  }

  // ── 4. Error event spike ──────────────────────────────────────────
  try {
    const db = getJournalDb();
    const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
    const errorCount = db.prepare(`
      SELECT COUNT(*) as c FROM events
      WHERE severity IN ('error', 'critical')
      AND created_at > ?
    `).get(oneHourAgo).c;

    if (errorCount > ERROR_SPIKE_THRESHOLD) {
      alerts.push(`🚨 <b>Error spike:</b> ${errorCount} errors in the last hour`);
    }
  } catch (err) {
    log(`Error count check failed: ${err.message}`);
  }

  // ── 5. Regime change ──────────────────────────────────────────────
  try {
    const db = getJournalDb();
    const recentRegimes = db.prepare(`
      SELECT regime FROM daily_snapshots ORDER BY date DESC LIMIT 2
    `).all();

    if (recentRegimes.length === 2 && recentRegimes[0].regime !== recentRegimes[1].regime) {
      alerts.push(`🔄 <b>Regime change:</b> ${recentRegimes[1].regime} → ${recentRegimes[0].regime}`);
    }
  } catch (err) {
    log(`Regime check failed: ${err.message}`);
  }

  // ── Send alerts only if there are problems ────────────────────────
  if (alerts.length > 0) {
    let msg = `<b>⚠️ Heartbeat Alert — ${nowET()}</b>\n\n`;
    msg += alerts.join("\n");
    if (portfolio) {
      msg += `\n\nEquity: ${formatMoney(portfolio.equity)} | Positions: ${portfolio.positionsCount}`;
    }
    log(`Sending ${alerts.length} alert(s)`);
    await sendTelegram(msg);
  } else {
    log("All checks passed — no alerts.");
  }

  // ── Log event ─────────────────────────────────────────────────────
  logScriptRun("hourly_heartbeat", alerts.length > 0 ? "warning" : "ok", {
    alertCount: alerts.length,
    equity: portfolio?.equity || null,
  });

  log("Hourly heartbeat complete.");
}

heartbeat().catch((err) => {
  log(`FATAL: ${err.message}`);
  try { logScriptRun("hourly_heartbeat", "error", { error: err.message }); } catch (_) {}
  process.exit(1);
});
