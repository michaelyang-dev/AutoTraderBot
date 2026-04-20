#!/usr/bin/env node
/**
 * Test Telegram notifications from AutoTrader.
 *
 * Usage:
 *   node scripts/test_telegram.js
 *   node scripts/test_telegram.js "Custom test message"
 */

require("dotenv").config();

const { TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID } = process.env;

if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) {
  console.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env");
  process.exit(1);
}

console.log(`Bot token: ${TELEGRAM_BOT_TOKEN.slice(0, 8)}...`);
console.log(`Chat ID:   ${TELEGRAM_CHAT_ID}`);

const notify = require("../server/notifications");

const message = process.argv[2] || "🤖 *AutoTrader Test*\n\nTelegram notifications are working.\n\nTimestamp: " + new Date().toISOString();

notify.send(message, { immediate: true });

// Give the async flush time to complete
setTimeout(() => {
  console.log("Test message sent. Check Telegram.");
  process.exit(0);
}, 10000);
