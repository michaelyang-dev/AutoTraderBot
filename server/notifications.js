// ══════════════════════════════════════════════════════════════════════
//  TELEGRAM NOTIFICATIONS — Non-blocking alerts via Telegram Bot API
//
//  Features:
//    - Message batching: groups messages within a 2-second window
//    - Rate limiting: max 30 messages/minute
//    - Deduplication: same message suppressed for 5 minutes
//    - Retry once after 5s on failure
//    - Silent degradation: if keys aren't set or API fails, bot keeps running
//
//  Usage:
//    const notify = require("./notifications");
//    notify.send("🤖 ML BUY AAPL ...");
//    notify.send("🚨 ERROR ...", { deduplicate: true, immediate: true });
// ══════════════════════════════════════════════════════════════════════

const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID   = process.env.TELEGRAM_CHAT_ID;
const TELEGRAM_ENABLED   = !!(TELEGRAM_BOT_TOKEN && TELEGRAM_CHAT_ID);

if (TELEGRAM_ENABLED) {
  console.log("📬 Telegram notifications enabled.");
} else {
  console.log("📬 Telegram notifications disabled (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set).");
}

// ── Message queue & batching ──────────────────────────────────────
let queue = [];
let flushTimer = null;
const BATCH_WINDOW_MS = 2000;

// ── Rate limiting (30 messages per minute) ───────────────────────
const sendTimestamps = [];
const MAX_PER_MINUTE  = 30;

// ── Deduplication (suppress identical messages for 5 min) ────────
const recentHashes    = new Map();
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

/**
 * Derive a header line from the first message in a batch (for Telegram grouping).
 */
function deriveSubject(messages) {
  if (messages.length === 0) return "Notification";

  const first = messages[0];
  const noTs = first.replace(/^\[.*?ET\]\s*/, "");
  const noEmoji = noTs.replace(/^[^\w]*/, "");
  const cut = noEmoji.split(/[|\n]/)[0].trim();
  const subject = cut.length > 60 ? cut.slice(0, 57) + "..." : cut;

  if (messages.length === 1) return subject;
  return `${subject} (+${messages.length - 1} more)`;
}

// ── Telegram API ─────────────────────────────────────────────────

const TELEGRAM_API_URL = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;

/**
 * Send a message via Telegram Bot API. Retries once after 5s on failure.
 * Never throws — logs errors and moves on.
 */
async function sendTelegram(text) {
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const res = await fetch(TELEGRAM_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chat_id: TELEGRAM_CHAT_ID,
          text,
          parse_mode: "Markdown",
        }),
      });

      if (res.ok) return;

      const body = await res.text();
      console.error(`Telegram API error (attempt ${attempt}/2): ${res.status} — ${body}`);

      // If Markdown parsing fails, retry without parse_mode
      if (res.status === 400 && body.includes("can't parse")) {
        const retry = await fetch(TELEGRAM_API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            chat_id: TELEGRAM_CHAT_ID,
            text,
          }),
        });
        if (retry.ok) return;
      }
    } catch (err) {
      console.error(`Telegram send failed (attempt ${attempt}/2): ${err.message}`);
    }

    if (attempt === 1) {
      await new Promise(r => setTimeout(r, 5000));
    }
  }
}

// ── Flush batched messages ───────────────────────────────────────

async function flush() {
  flushTimer = null;
  if (queue.length === 0 || !TELEGRAM_ENABLED) return;

  const messages = queue.splice(0);

  // ── Rate limit check ──
  const now = Date.now();
  while (sendTimestamps.length > 0 && now - sendTimestamps[0] > 60000) {
    sendTimestamps.shift();
  }
  if (sendTimestamps.length >= MAX_PER_MINUTE) {
    queue.unshift(...messages);
    flushTimer = setTimeout(flush, 5000);
    console.warn("Telegram rate limit hit — deferring messages.");
    return;
  }

  const header = deriveSubject(messages);
  const body = `*AutoTrader: ${header}*\n\n${messages.join("\n\n")}`;

  await sendTelegram(body);
  sendTimestamps.push(Date.now());
}

/**
 * Queue a Telegram notification.
 *
 * @param {string} message  Plain text with emoji
 * @param {object} [opts]
 * @param {boolean} [opts.deduplicate=false]  Skip if identical message sent within 5 min
 * @param {boolean} [opts.immediate=false]    Flush queue immediately (for errors)
 */
function send(message, { deduplicate = false, immediate = false } = {}) {
  if (!TELEGRAM_ENABLED) return;

  // ── Dedup check ──
  if (deduplicate) {
    cleanDedup();
    const hash = simpleHash(message);
    if (recentHashes.has(hash)) return;
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

module.exports = { send };
