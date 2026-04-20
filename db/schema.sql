-- ══════════════════════════════════════════════════════════════════════
--  Trade Journal Schema v1
--  Dual-source: engine writes intent, Alpaca reconciler writes reality
-- ══════════════════════════════════════════════════════════════════════

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ── Schema version tracking ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version) VALUES (1);

-- ── Trades ───────────────────────────────────────────────────────────
-- One row per order. Intent fields captured on submit; reality fields
-- updated by reconciler when fills come in.

CREATE TABLE IF NOT EXISTS trades (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,

  -- Alpaca linkage
  alpaca_order_id   TEXT UNIQUE,
  client_order_id   TEXT,

  -- Core
  symbol            TEXT NOT NULL,
  side              TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
  qty               REAL NOT NULL,

  -- Attribution (captured at submit time — Alpaca can't give us these later)
  strategy          TEXT NOT NULL,
  signal_id         INTEGER REFERENCES signals(id),
  signal_prob       REAL,
  ml_rank           INTEGER,
  regime            TEXT,

  -- Intent (what we asked for)
  intended_price    REAL,
  stop_loss         REAL,
  take_profit       REAL,

  -- Reality (what actually happened — reconciler fills these in)
  status            TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'partial', 'rejected', 'canceled')),
  filled_qty        REAL,
  fill_price        REAL,
  fill_time         TEXT,
  slippage_bps      REAL,
  commission        REAL,

  -- Exit linkage (for sell trades)
  realized_pnl      REAL,
  realized_pnl_pct  REAL,
  hold_days         INTEGER,
  exit_reason       TEXT,
  entry_trade_id    INTEGER REFERENCES trades(id),

  -- Audit
  submitted_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
  notes             TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_submitted_at ON trades(submitted_at);
CREATE INDEX IF NOT EXISTS idx_trades_alpaca_order_id ON trades(alpaca_order_id);

-- ── Positions ────────────────────────────────────────────────────────
-- Current open positions. Updated on buy fill, removed on full sell.

CREATE TABLE IF NOT EXISTS positions (
  symbol                    TEXT PRIMARY KEY,
  qty                       REAL NOT NULL,
  avg_cost                  REAL NOT NULL,
  cost_basis                REAL NOT NULL,
  current_price             REAL,
  current_value             REAL,
  unrealized_pnl            REAL,
  unrealized_pnl_pct        REAL,
  strategy                  TEXT,
  entry_regime              TEXT,
  entry_trade_id            INTEGER REFERENCES trades(id),
  opened_at                 TEXT,
  days_held                 INTEGER,
  stop_loss                 REAL,
  take_profit               REAL,
  trailing_stop             REAL,
  highest_price_since_entry REAL,
  updated_at                TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Daily Snapshots ──────────────────────────────────────────────────
-- One row per trading day, written near market close.

CREATE TABLE IF NOT EXISTS daily_snapshots (
  date                  TEXT PRIMARY KEY,
  portfolio_value       REAL NOT NULL,
  cash                  REAL NOT NULL,
  equity                REAL,
  positions_count       INTEGER NOT NULL,
  day_pnl               REAL,
  day_pnl_pct           REAL,
  regime                TEXT,
  spy_close             REAL,
  spy_day_pct           REAL,
  strategy_pnl_json     TEXT,
  slot_usage_json       TEXT,
  circuit_breaker_state TEXT,
  peak_value            REAL,
  drawdown_pct          REAL,
  created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Signals ──────────────────────────────────────────────────────────
-- Every ML signal received from signal_server.

CREATE TABLE IF NOT EXISTS signals (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol          TEXT NOT NULL,
  signal_time     TEXT NOT NULL DEFAULT (datetime('now')),
  signal_type     TEXT,
  probability     REAL,
  rank            INTEGER,
  is_top_5        INTEGER,
  price_at_signal REAL,
  regime          TEXT,
  model_version   TEXT,
  acted_on        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals(symbol, signal_time);
CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(signal_time);

-- ── Events ───────────────────────────────────────────────────────────
-- Audit trail for system events.

CREATE TABLE IF NOT EXISTS events (
  id                      INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type              TEXT NOT NULL CHECK (event_type IN (
    'circuit_breaker', 'regime_change', 'retrain', 'error',
    'manual_override', 'service_restart'
  )),
  severity                TEXT NOT NULL DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'error', 'critical')),
  message                 TEXT NOT NULL,
  metadata_json           TEXT,
  portfolio_value_at_event REAL,
  created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_type_created ON events(event_type, created_at);
