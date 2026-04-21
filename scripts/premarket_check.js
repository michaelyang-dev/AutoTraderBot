#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Pre-Market Check — Runs at 9:00 AM ET before market open
//
//  9 checks: services, journal, Alpaca, account, circuit breakers,
//  regime, ML signals, disk, memory.
//
//  Sends a Telegram summary with pass/fail for each check.
//
//  Usage:
//    node scripts/premarket_check.js
// ══════════════════════════════════════════════════════════════════════

require("dotenv").config();

const {
  getPortfolioState,
  getPmStatus,
  getMlSignalsHealth,
  formatMoney,
  formatPct,
  sendTelegram,
  getJournalDb,
  logScriptRun,
  nowET,
  ddOpen,
} = require("./lib/automationHelpers");
const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

async function premkt() {
  log("Pre-market check starting...");
  const checks = [];
  let hasError = false;

  // ── 1. PM2 services ───────────────────────────────────────────────
  try {
    const procs = getPmStatus();
    if (!procs) {
      checks.push({ name: "Services", status: "WARN", detail: "PM2 not reachable" });
      hasError = true;
    } else {
      const down = procs.filter(p => p.status !== "online");
      if (down.length > 0) {
        checks.push({ name: "Services", status: "FAIL", detail: `Down: ${down.map(p => p.name).join(", ")}` });
        hasError = true;
      } else {
        const highRestarts = procs.filter(p => p.restarts > 5);
        const detail = procs.map(p => `${p.name}: ${p.status} (${p.restarts} restarts)`).join(", ");
        checks.push({
          name: "Services",
          status: highRestarts.length > 0 ? "WARN" : "OK",
          detail,
        });
        if (highRestarts.length > 0) hasError = true;
      }
    }
  } catch (err) {
    checks.push({ name: "Services", status: "FAIL", detail: err.message });
    hasError = true;
  }

  // ── 2. Journal DB ─────────────────────────────────────────────────
  try {
    const db = getJournalDb();
    const tradeCount = db.prepare("SELECT COUNT(*) as c FROM trades").get().c;
    const posCount = db.prepare("SELECT COUNT(*) as c FROM positions").get().c;
    checks.push({ name: "Journal DB", status: "OK", detail: `${tradeCount} trades, ${posCount} positions` });
  } catch (err) {
    checks.push({ name: "Journal DB", status: "FAIL", detail: err.message });
    hasError = true;
  }

  // ── 3. Alpaca connectivity ────────────────────────────────────────
  let portfolio = null;
  try {
    portfolio = await getPortfolioState();
    checks.push({ name: "Alpaca API", status: "OK", detail: `${portfolio.positionsCount} positions` });
  } catch (err) {
    checks.push({ name: "Alpaca API", status: "FAIL", detail: err.message });
    hasError = true;
  }

  // ── 4. Account status ─────────────────────────────────────────────
  if (portfolio) {
    const cashPct = portfolio.cash / portfolio.equity;
    const status = cashPct > 0.5 ? "WARN" : "OK";
    if (cashPct > 0.5) hasError = true;
    checks.push({
      name: "Account",
      status,
      detail: `Equity: ${formatMoney(portfolio.equity)}, Cash: ${formatMoney(portfolio.cash)} (${(cashPct * 100).toFixed(0)}%), Positions: ${portfolio.positionsCount}`,
    });
  }

  // ── 5. Circuit breakers ───────────────────────────────────────────
  try {
    const cbFile = path.join(__dirname, "..", "ml_service", "data", "circuit_breaker_state.json");
    if (fs.existsSync(cbFile)) {
      const cb = JSON.parse(fs.readFileSync(cbFile, "utf8"));
      const halted = cb.peakHalted || cb.weeklyHalted || cb.dailyHalted;
      checks.push({
        name: "Circuit Breakers",
        status: halted ? "WARN" : "OK",
        detail: halted
          ? `HALTED — peak:${cb.peakHalted}, weekly:${cb.weeklyHalted}, daily:${cb.dailyHalted}`
          : `Clear — peak: ${formatMoney(cb.peakValue || 0)}`,
      });
      if (halted) hasError = true;
    } else {
      checks.push({ name: "Circuit Breakers", status: "WARN", detail: "State file not found" });
    }
  } catch (err) {
    checks.push({ name: "Circuit Breakers", status: "WARN", detail: err.message });
  }

  // ── 6. Regime ─────────────────────────────────────────────────────
  try {
    const db = getJournalDb();
    const latest = db.prepare("SELECT regime FROM daily_snapshots ORDER BY date DESC LIMIT 1").get();
    checks.push({ name: "Regime", status: "OK", detail: latest?.regime || "unknown" });
  } catch (err) {
    checks.push({ name: "Regime", status: "WARN", detail: err.message });
  }

  // ── 7. ML signals ────────────────────────────────────────────────
  try {
    const ml = await getMlSignalsHealth();
    if (ml.ok) {
      checks.push({ name: "ML Signals", status: "OK", detail: `${ml.count} signals available` });
    } else {
      checks.push({ name: "ML Signals", status: "WARN", detail: ml.error });
      hasError = true;
    }
  } catch (err) {
    checks.push({ name: "ML Signals", status: "WARN", detail: err.message });
    hasError = true;
  }

  // ── 8. Disk space ─────────────────────────────────────────────────
  try {
    const dfOutput = execSync("df -h . | tail -1", { timeout: 5000 }).toString().trim();
    const parts = dfOutput.split(/\s+/);
    const usePct = parseInt(parts[4], 10);
    checks.push({
      name: "Disk",
      status: usePct > 90 ? "WARN" : "OK",
      detail: `${parts[4]} used (${parts[3]} avail)`,
    });
    if (usePct > 90) hasError = true;
  } catch (err) {
    checks.push({ name: "Disk", status: "WARN", detail: err.message });
  }

  // ── 9. Memory ─────────────────────────────────────────────────────
  try {
    const memInfo = process.memoryUsage();
    const rss = (memInfo.rss / 1024 / 1024).toFixed(0);
    checks.push({ name: "Memory", status: "OK", detail: `RSS: ${rss} MB (this process)` });
  } catch (err) {
    checks.push({ name: "Memory", status: "WARN", detail: err.message });
  }

  // ── Build Telegram message ────────────────────────────────────────
  const icon = hasError ? "⚠️" : "✅";
  const dd = portfolio ? ddOpen(portfolio.equity) : null;
  const ddStr = dd != null ? ` | DD: ${formatPct(dd)}` : "";

  let msg = `<b>${icon} Pre-Market Check — ${nowET()}</b>\n`;
  if (portfolio) {
    msg += `Equity: <b>${formatMoney(portfolio.equity)}</b>${ddStr}\n`;
  }
  msg += `\n`;

  for (const c of checks) {
    const emoji = c.status === "OK" ? "✅" : c.status === "WARN" ? "⚠️" : "❌";
    msg += `${emoji} <b>${c.name}</b>: ${c.detail}\n`;
  }

  log(msg.replace(/<[^>]+>/g, ""));

  await sendTelegram(msg);

  // ── Log event ─────────────────────────────────────────────────────
  const metadata = {
    checks: checks.map(c => ({ name: c.name, status: c.status })),
    equity: portfolio?.equity || null,
    hasError,
  };
  logScriptRun("premarket_check", hasError ? "warning" : "ok", metadata);

  log("Pre-market check complete.");
}

premkt().catch((err) => {
  log(`FATAL: ${err.message}`);
  try { logScriptRun("premarket_check", "error", { error: err.message }); } catch (_) {}
  process.exit(1);
});
