module.exports = {
  apps: [
    // ── Core services (always running) ──────────────────────────────
    {
      name: "trading-bot",
      script: "server/index.js",
      cwd: "/home/ubuntu/AutoTraderBot",
      watch: false,
      autorestart: true,
      restart_delay: 10000,
      max_restarts: 10,
      env: {
        NODE_ENV: "production",
        PORT: 3001,
      },
    },
    {
      name: "ml-server",
      script: "/home/ubuntu/AutoTraderBot/ml_service/venv/bin/python3",
      args: "/home/ubuntu/AutoTraderBot/ml_service/signal_server.py",
      cwd: "/home/ubuntu/AutoTraderBot",
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
      cwd: "/home/ubuntu/AutoTraderBot",
      cron_restart: "*/5 * * * *",
      autorestart: false,
      watch: false,
    },

    // ── Pre-market check (9:25 AM ET, Mon–Fri) ─────────────────────
    {
      name: "premarket-check",
      script: "scripts/premarket_check.js",
      cwd: "/home/ubuntu/AutoTraderBot",
      cron_restart: "25 9 * * 1-5",
      autorestart: false,
      watch: false,
    },

    // ── Daily report (4:15 PM ET, Mon–Fri) ──────────────────────────
    {
      name: "daily-report",
      script: "scripts/daily_report.js",
      cwd: "/home/ubuntu/AutoTraderBot",
      cron_restart: "15 16 * * 1-5",
      autorestart: false,
      watch: false,
    },

    // ── Hourly heartbeat (10 AM–3 PM ET, Mon–Fri) ───────────────────
    {
      name: "hourly-heartbeat",
      script: "scripts/hourly_heartbeat.js",
      cwd: "/home/ubuntu/AutoTraderBot",
      cron_restart: "0 10-15 * * 1-5",
      autorestart: false,
      watch: false,
    },

    // ── Weekly report (Sunday 6:00 PM ET) ───────────────────────────
    {
      name: "weekly-report",
      script: "scripts/weekly_report.js",
      cwd: "/home/ubuntu/AutoTraderBot",
      cron_restart: "0 18 * * 0",
      autorestart: false,
      watch: false,
    },
  ],
};
