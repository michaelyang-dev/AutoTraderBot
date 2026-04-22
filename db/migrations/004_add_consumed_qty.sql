-- Migration 004: Add consumed_qty to trades for FIFO share tracking
-- Tracks how many shares of a BUY trade have been matched to subsequent SELLs

ALTER TABLE trades ADD COLUMN consumed_qty REAL DEFAULT 0;

INSERT INTO schema_version (version, applied_at) VALUES (4, datetime('now'));
