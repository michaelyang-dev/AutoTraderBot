// ══════════════════════════════════════════════════════════════════════
//  Automation Helpers — Shared utilities for all automation scripts
//
//  Used by: premarket_check, daily_report, hourly_heartbeat, weekly_report
// ══════════════════════════════════════════════════════════════════════

require("dotenv").config();

const Alpaca = require("@alpacahq/alpaca-trade-api");
const { execSync } = require("child_process");
const path = require("path");
const journal = require("../../db/journal");

const TIMEOUT_MS = 15000;

// ── Alpaca client ────────────────────────────────────────────────────

function getAlpacaClient() {
  const { ALPACA_API_KEY, ALPACA_SECRET_KEY } = process.env;
  if (!ALPACA_API_KEY || !ALPACA_SECRET_KEY) {
    throw new Error("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY");
  }
  return new Alpaca({
    keyId: ALPACA_API_KEY,
    secretKey: ALPACA_SECRET_KEY,
    paper: true,
  });
}

// ── Timeout + retry ──────────────────────────────────────────────────

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
        await new Promise(r => setTimeout(r, 60000));
        continue;
      }
      throw err;
    }
  }
}

// ── Portfolio state (Alpaca account + positions) ─────────────────────

async function getPortfolioState() {
  const alpaca = getAlpacaClient();
  const [account, positions] = await Promise.all([
    fetchWithRetry(() => alpaca.getAccount(), "getAccount"),
    fetchWithRetry(() => alpaca.getPositions(), "getPositions"),
  ]);

  const equity = parseFloat(account.equity);
  const cash = parseFloat(account.cash);
  const buyingPower = parseFloat(account.buying_power);
  const lastEquity = parseFloat(account.last_equity);
  const dayPnl = equity - lastEquity;
  const dayPnlPct = lastEquity > 0 ? dayPnl / lastEquity : 0;

  return {
    equity,
    cash,
    buyingPower,
    lastEquity,
    dayPnl,
    dayPnlPct,
    positionsCount: positions.length,
    positions: positions.map(p => ({
      symbol: p.symbol,
      qty: parseFloat(p.qty),
      avgCost: parseFloat(p.avg_entry_price),
      currentPrice: parseFloat(p.current_price),
      marketValue: parseFloat(p.market_value),
      unrealizedPnl: parseFloat(p.unrealized_pl),
      unrealizedPnlPct: parseFloat(p.unrealized_plpc),
      side: p.side,
    })),
    account,
  };
}

// ── PM2 process status ───────────────────────────────────────────────

function getPmStatus() {
  try {
    const raw = execSync("pm2 jlist 2>/dev/null", { timeout: 5000 }).toString();
    const procs = JSON.parse(raw);
    return procs.map(p => ({
      name: p.name,
      status: p.pm2_env?.status || "unknown",
      uptime: p.pm2_env?.pm_uptime || null,
      restarts: p.pm2_env?.restart_time || 0,
      cpu: p.monit?.cpu || 0,
      memory: p.monit?.memory || 0,
    }));
  } catch {
    return null;
  }
}

// ── ML signal server health ──────────────────────────────────────────

async function getMlSignalsHealth() {
  try {
    const res = await withTimeout(
      fetch("http://localhost:5001/signals", { signal: AbortSignal.timeout(5000) }),
      8000,
      "ML signals"
    );
    if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };
    const data = await res.json();
    const signals = Array.isArray(data) ? data : data.signals || [];
    return {
      ok: true,
      count: signals.length,
      topSignal: signals[0] || null,
    };
  } catch (err) {
    return { ok: false, error: err.message };
  }
}

// ── Formatting helpers ───────────────────────────────────────────────

function formatMoney(n) {
  if (n == null || isNaN(n)) return "$—";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(2)}`;
}

function formatPct(n, { withEmoji = false } = {}) {
  if (n == null || isNaN(n)) return "—%";
  const pct = (n * 100).toFixed(2);
  if (!withEmoji) return `${pct}%`;
  const emoji = n > 0 ? "🟢" : n < 0 ? "🔴" : "⚪";
  return `${emoji} ${pct}%`;
}

function formatTable(headers, rows) {
  if (!rows || rows.length === 0) return "(no data)";

  const widths = headers.map((h, i) =>
    Math.max(h.length, ...rows.map(r => String(r[i] ?? "").length))
  );

  const sep = widths.map(w => "─".repeat(w)).join("─┼─");
  const headerLine = headers.map((h, i) => h.padEnd(widths[i])).join(" │ ");
  const bodyLines = rows.map(r =>
    r.map((cell, i) => String(cell ?? "").padEnd(widths[i])).join(" │ ")
  );

  return [headerLine, sep, ...bodyLines].join("\n");
}

// ── Telegram (standalone — not using server/notifications.js batching) ──

async function sendTelegram(htmlMessage, { silent = false } = {}) {
  const token = process.env.TELEGRAM_BOT_TOKEN;
  const chatId = process.env.TELEGRAM_CHAT_ID;
  if (!token || !chatId) {
    console.log("[telegram] Disabled — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set");
    return false;
  }

  const url = `https://api.telegram.org/bot${token}/sendMessage`;
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chat_id: chatId,
          text: htmlMessage,
          parse_mode: "HTML",
          disable_notification: silent,
        }),
      });
      if (res.ok) return true;

      const body = await res.text();
      console.error(`[telegram] API error (attempt ${attempt}/2): ${res.status} — ${body}`);

      // Retry without parse_mode if HTML parsing fails
      if (res.status === 400 && body.includes("can't parse")) {
        const retry = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chat_id: chatId,
            text: htmlMessage,
            disable_notification: silent,
          }),
        });
        if (retry.ok) return true;
      }
    } catch (err) {
      console.error(`[telegram] Send failed (attempt ${attempt}/2): ${err.message}`);
    }
    if (attempt === 1) await new Promise(r => setTimeout(r, 5000));
  }
  return false;
}

// ── Journal DB access ────────────────────────────────────────────────

function getJournalDb() {
  const dbPath = path.join(__dirname, "..", "..", "data", "journal.db");
  journal.initDb(dbPath);
  return journal.getDb();
}

// ── Script event logging ─────────────────────────────────────────────

function logScriptRun(scriptName, status, metadata = {}) {
  try {
    const db = getJournalDb();
    db.prepare(`
      INSERT INTO events (event_type, severity, message, metadata_json, created_at)
      VALUES (?, ?, ?, ?, datetime('now'))
    `).run(
      scriptName,
      status === "error" ? "error" : "info",
      `${scriptName} completed: ${status}`,
      JSON.stringify(metadata),
    );
  } catch (err) {
    console.error(`[logScriptRun] Failed: ${err.message}`);
  }
}

// ── Time helpers ─────────────────────────────────────────────────────

function nowET() {
  return new Date().toLocaleString("en-US", { timeZone: "America/New_York" });
}

function ddOpen(portfolioValue) {
  try {
    const db = getJournalDb();
    const row = db.prepare(`
      SELECT MAX(portfolio_value) as peak
      FROM daily_snapshots
    `).get();
    if (!row?.peak || row.peak <= 0) return null;
    const dd = (portfolioValue - row.peak) / row.peak;
    return dd;
  } catch {
    return null;
  }
}

module.exports = {
  getAlpacaClient,
  getPortfolioState,
  getPmStatus,
  getMlSignalsHealth,
  formatMoney,
  formatPct,
  formatTable,
  sendTelegram,
  getJournalDb,
  logScriptRun,
  nowET,
  ddOpen,
  fetchWithRetry,
  withTimeout,
  TIMEOUT_MS,
};
