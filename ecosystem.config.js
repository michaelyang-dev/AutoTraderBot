module.exports = {
  apps: [
    // ── Live equity engine — REAL MONEY (IBKR) ─────────────────────
    // start_ibkr_engine.sh loads .env, then execs ml_service/ibkr_engine.py.
    // Added 2026-07-14: this engine was previously started ONLY by a manual
    // `pm2 start start_ibkr_engine.sh` and was absent from this file — so a
    // clean `pm2 start ecosystem.config.js` would NOT have brought up live trading.
    {
      name: "ibkr-engine",
      script: "start_ibkr_engine.sh",
      cwd: "/home/ubuntu/AutoTraderBot",
      watch: false,
      autorestart: true,
      restart_delay: 10000,
      max_restarts: 10,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "signal-server",
      script: "start_signal_server.sh",
      cwd: "/home/ubuntu/AutoTraderBot",
      watch: false,
      autorestart: true,
      max_restarts: 5,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
    {
      name: "trading-engine",
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
    // ── EDGAR realtime monitor — currently STOPPED on the server (short-data
    // collection paused); a from-git bring-up will start it. `pm2 stop
    // edgar-monitor` afterward to keep it paused.
    {
      name: "edgar-monitor",
      script: "start_edgar_monitor.sh",
      cwd: "/home/ubuntu/AutoTraderBot",
      watch: false,
      autorestart: true,
      max_restarts: 5,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },

    // Retired 2026-07-14: the journal-reconciler / premarket-check / daily-report /
    // hourly-heartbeat / weekly-report PM2 cron_restart apps were removed. They were
    // committed (bb7243f) but NEVER ran on the server — pm2 cron_restart is flaky
    // ("cron_restart on a stopped process doesn't fire", see below) and they were
    // never registered. Real-money alerts now come from ibkr-engine's own Telegram;
    // the Alpaca paper side has Grafana on :3001. The scripts remain under scripts/
    // — if ever wanted, wire them via the SYSTEM crontab (reliable), not pm2.

    // NOTE: data refreshes (weekday 5:30 PM + Sunday 9 PM ET) run via the SYSTEM
    // crontab — reliable. The old PM2 cron_restart apps for refresh were flaky
    // (cron_restart on a stopped process doesn't fire) and were removed 2026-06-19.
    // See `crontab -l` on the server.
  ],
};
