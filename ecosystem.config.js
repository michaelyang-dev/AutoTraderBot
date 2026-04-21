module.exports = {
  apps: [
    // ── Core services (always running) ──────────────────────────────
    {
      name: "trading-bot",
      script: "server/index.js",
      watch: false,
      autorestart: true,
      max_restarts: 10,
      env: {
        NODE_ENV: "production",
      },
    },
    {
      name: "ml-server",
      script: "ml_service/signal_server.py",
      interpreter: "python3",
      watch: false,
      autorestart: true,
      max_restarts: 5,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },

    // ── Reconciler (every 5 min) ────────────────────────────────────
    {
      name: "journal-reconciler",
      script: "scripts/reconcile_journal.js",
      cron_restart: "*/5 * * * *",
      autorestart: false,
      watch: false,
    },

    // ── Pre-market check (9:00 AM ET, Mon–Fri) ─────────────────────
    {
      name: "premarket-check",
      script: "scripts/premarket_check.js",
      cron_restart: "0 9 * * 1-5",
      autorestart: false,
      watch: false,
    },

    // ── Daily report (4:15 PM ET, Mon–Fri) ──────────────────────────
    {
      name: "daily-report",
      script: "scripts/daily_report.js",
      cron_restart: "15 16 * * 1-5",
      autorestart: false,
      watch: false,
    },

    // ── Hourly heartbeat (market hours, Mon–Fri) ────────────────────
    {
      name: "hourly-heartbeat",
      script: "scripts/hourly_heartbeat.js",
      cron_restart: "0 10-16 * * 1-5",
      autorestart: false,
      watch: false,
    },

    // ── Weekly report (Sunday 6:00 PM ET) ───────────────────────────
    {
      name: "weekly-report",
      script: "scripts/weekly_report.js",
      cron_restart: "0 18 * * 0",
      autorestart: false,
      watch: false,
    },
  ],
};
