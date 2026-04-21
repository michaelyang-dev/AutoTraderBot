-- Migration 003: Add automation event types
-- Adds: script_run, premarket_check, daily_report, hourly_heartbeat, weekly_report

CREATE TABLE IF NOT EXISTS events_new (
  id                      INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type              TEXT NOT NULL CHECK (event_type IN (
    'circuit_breaker', 'regime_change', 'retrain', 'error',
    'manual_override', 'service_restart', 'journal_seeded',
    'trade_submitted', 'trade_filled', 'position_opened',
    'position_closed', 'daily_snapshot', 'reconcile_run',
    'startup', 'shutdown',
    'script_run', 'premarket_check', 'daily_report',
    'hourly_heartbeat', 'weekly_report'
  )),
  severity                TEXT NOT NULL DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'error', 'critical')),
  message                 TEXT NOT NULL,
  metadata_json           TEXT,
  portfolio_value_at_event REAL,
  created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO events_new SELECT * FROM events;

DROP TABLE events;

ALTER TABLE events_new RENAME TO events;

CREATE INDEX IF NOT EXISTS idx_events_type_created ON events(event_type, created_at);

INSERT INTO schema_version (version, applied_at) VALUES (3, datetime('now'));
