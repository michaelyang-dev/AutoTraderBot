// ══════════════════════════════════════════════════════════════════════
//  TELEGRAM NOTIFICATIONS — Non-blocking alerts for the trading bot
//
//  Features:
//    - Message batching: groups messages within a 2-second window
//    - Rate limiting: max 30 messages/minute to avoid Telegram throttling
//    - Deduplication: same error message suppressed for 5 minutes
//    - Silent degradation: if keys aren't set or API fails, bot keeps running
//
//  Usage:
//    const { sendTelegram } = require("./notifications");
//    sendTelegram("🤖 ML BUY AAPL ...");
//    sendTelegram("🚨 ERROR ...", { deduplicate: true, immediate: true });
// ══════════════════════════════════════════════════════════════════════

const TOKEN   = process.env.TELEGRAM_BOT_TOKEN;
const CHAT_ID = process.env.TELEGRAM_CHAT_ID;

let bot = null;

if (TOKEN && CHAT_ID) {
  try {
    const TelegramBot = require("node-telegram-bot-api");
    bot = new TelegramBot(TOKEN, { polling: false });
    console.log("📱 Telegram notifications enabled.");
  } catch (err) {
    console.warn("⚠️  node-telegram-bot-api not installed — Telegram notifications disabled.");
  }
} else {
  console.log("📱 Telegram notifications disabled (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env).");
}

// ── Message queue & batching ──────────────────────────────────────
let queue = [];
let flushTimer = null;
const BATCH_WINDOW_MS = 2000;
const MAX_MSG_LEN     = 4000; // Telegram limit is 4096; leave margin

// ── Rate limiting (30 API calls per minute) ──────────────────────
const sendTimestamps = [];
const MAX_PER_MINUTE  = 30;

// ── Deduplication (suppress identical messages for 5 min) ────────
const recentHashes   = new Map();
const DEDUP_WINDOW_MS = 5 * 60 * 1000;

function simpleHash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h) + str.charCodeAt(i);
    h |= 0;
  }
  return h;
}

function getTimestamp() {
  return new Date().toLocaleTimeString("en-US", {
    timeZone: "America/New_York",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}

function cleanDedup() {
  const now = Date.now();
  for (const [hash, ts] of recentHashes) {
    if (now - ts > DEDUP_WINDOW_MS) recentHashes.delete(hash);
  }
}

async function flush() {
  flushTimer = null;
  if (queue.length === 0 || !bot) return;

  const messages = queue.splice(0);

  // ── Rate limit check ──
  const now = Date.now();
  while (sendTimestamps.length > 0 && now - sendTimestamps[0] > 60000) {
    sendTimestamps.shift();
  }
  if (sendTimestamps.length >= MAX_PER_MINUTE) {
    queue.unshift(...messages);               // re-queue
    flushTimer = setTimeout(flush, 5000);     // retry in 5s
    console.warn("Telegram rate limit hit — deferring messages.");
    return;
  }

  // ── Combine & split at 4000 chars ──
  let combined = messages.join("\n\n");
  const chunks = [];
  while (combined.length > 0) {
    if (combined.length <= MAX_MSG_LEN) {
      chunks.push(combined);
      break;
    }
    let splitAt = combined.lastIndexOf("\n\n", MAX_MSG_LEN);
    if (splitAt <= 0) splitAt = MAX_MSG_LEN;
    chunks.push(combined.slice(0, splitAt));
    combined = combined.slice(splitAt).replace(/^\n+/, "");
  }

  for (const chunk of chunks) {
    try {
      await bot.sendMessage(CHAT_ID, chunk);
      sendTimestamps.push(Date.now());
    } catch (err) {
      console.error("Telegram send failed:", err.message);
      // Never crash — just log and drop
    }
  }
}

/**
 * Queue a Telegram notification.
 *
 * @param {string} message  Plain text with emoji — no Markdown/HTML needed
 * @param {object} [opts]
 * @param {boolean} [opts.deduplicate=false]  Skip if identical message sent within 5 min
 * @param {boolean} [opts.immediate=false]    Flush queue immediately (for errors)
 */
function sendTelegram(message, { deduplicate = false, immediate = false } = {}) {
  if (!bot) return;

  // ── Dedup check ──
  if (deduplicate) {
    cleanDedup();
    const hash = simpleHash(message);
    if (recentHashes.has(hash)) return;   // already sent recently
    recentHashes.set(hash, Date.now());
  }

  // ── Stamp & enqueue ──
  const stamped = `[${getTimestamp()} ET] ${message}`;
  queue.push(stamped);

  if (immediate) {
    if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
    flushTimer = setTimeout(flush, 0);
  } else if (!flushTimer) {
    flushTimer = setTimeout(flush, BATCH_WINDOW_MS);
  }
}

module.exports = { sendTelegram };
