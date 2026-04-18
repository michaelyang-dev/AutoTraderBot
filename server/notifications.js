// ══════════════════════════════════════════════════════════════════════
//  EMAIL NOTIFICATIONS — Non-blocking alerts via Gmail SMTP
//
//  Features:
//    - Message batching: groups messages within a 2-second window
//    - Rate limiting: max 30 emails/minute
//    - Deduplication: same error message suppressed for 5 minutes
//    - Auto subject: derived from message content (e.g. "AutoTrader: ML BUY AAPL")
//    - Silent degradation: if keys aren't set or SMTP fails, bot keeps running
//
//  Usage:
//    const notify = require("./notifications");
//    notify.send("🤖 ML BUY AAPL ...");
//    notify.send("🚨 ERROR ...", { deduplicate: true, immediate: true });
// ══════════════════════════════════════════════════════════════════════

const EMAIL_USER         = process.env.EMAIL_USER;
const EMAIL_APP_PASSWORD = process.env.EMAIL_APP_PASSWORD;

let transporter = null;

if (EMAIL_USER && EMAIL_APP_PASSWORD) {
  try {
    const nodemailer = require("nodemailer");
    transporter = nodemailer.createTransport({
      service: "gmail",
      auth: { user: EMAIL_USER, pass: EMAIL_APP_PASSWORD },
    });
    // Verify SMTP connection asynchronously — don't block startup
    transporter.verify()
      .then(() => console.log(`📧 Email notifications enabled (${EMAIL_USER}).`))
      .catch((err) => {
        console.warn(`⚠️  Email SMTP verify failed: ${err.message}. Will retry on first send.`);
      });
  } catch (err) {
    console.warn("⚠️  nodemailer not installed — email notifications disabled.");
  }
} else {
  console.log("📧 Email notifications disabled (EMAIL_USER / EMAIL_APP_PASSWORD not set in .env).");
}

// ── Message queue & batching ──────────────────────────────────────
let queue = [];
let flushTimer = null;
const BATCH_WINDOW_MS = 2000;

// ── Rate limiting (30 emails per minute) ─────────────────────────
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
 * Derive a descriptive email subject from the first message in a batch.
 * Strips emoji and timestamp, takes text before the first "|" or newline.
 *   "🤖 ML BUY AAPL | 50 shares..."   →  "AutoTrader: ML BUY AAPL"
 *   "🛑 STOP-LOSS NVDA | 30 shares..." →  "AutoTrader: STOP-LOSS NVDA"
 *   "📊 DAILY SUMMARY\n━━━..."         →  "AutoTrader: DAILY SUMMARY"
 *   "🚨 CIRCUIT BREAKER — Portfolio…"  →  "AutoTrader: CIRCUIT BREAKER — Portfolio…"
 */
function deriveSubject(messages) {
  if (messages.length === 0) return "AutoTrader: Notification";

  const first = messages[0];
  // Remove [HH:MM AM/PM ET] timestamp prefix
  const noTs = first.replace(/^\[.*?ET\]\s*/, "");
  // Remove leading non-word characters (emoji, spaces, symbols before text)
  const noEmoji = noTs.replace(/^[^\w]*/, "");
  // Take text before first pipe or newline
  const cut = noEmoji.split(/[|\n]/)[0].trim();
  const subject = cut.length > 60 ? cut.slice(0, 57) + "..." : cut;

  if (messages.length === 1) return `AutoTrader: ${subject}`;
  return `AutoTrader: ${subject} (+${messages.length - 1} more)`;
}

async function flush() {
  flushTimer = null;
  if (queue.length === 0 || !transporter) return;

  const messages = queue.splice(0);

  // ── Rate limit check ──
  const now = Date.now();
  while (sendTimestamps.length > 0 && now - sendTimestamps[0] > 60000) {
    sendTimestamps.shift();
  }
  if (sendTimestamps.length >= MAX_PER_MINUTE) {
    queue.unshift(...messages);              // re-queue
    flushTimer = setTimeout(flush, 5000);    // retry in 5s
    console.warn("Email rate limit hit — deferring messages.");
    return;
  }

  const subject = deriveSubject(messages);
  const body    = messages.join("\n\n") + "\n\n—\nAutoTrader Bot";

  try {
    await transporter.sendMail({
      from: EMAIL_USER,
      to: EMAIL_USER,
      subject,
      text: body,
    });
    sendTimestamps.push(Date.now());
  } catch (err) {
    console.error("Email send failed:", err.message);
    // Never crash — just log and drop
  }
}

/**
 * Queue an email notification.
 *
 * @param {string} message  Plain text with emoji
 * @param {object} [opts]
 * @param {boolean} [opts.deduplicate=false]  Skip if identical message sent within 5 min
 * @param {boolean} [opts.immediate=false]    Flush queue immediately (for errors)
 */
function send(message, { deduplicate = false, immediate = false } = {}) {
  if (!transporter) return;

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

module.exports = { send };
