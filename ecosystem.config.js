module.exports = {
  apps: [
    {
      name: "autotrader",
      script: "server/index.js",
      watch: false,
      autorestart: true,
      max_restarts: 10,
      env: {
        NODE_ENV: "production",
      },
    },
    {
      name: "journal-reconciler",
      script: "scripts/reconcile_journal.js",
      cron_restart: "*/5 * * * *",
      autorestart: false,
      watch: false,
    },
  ],
};
