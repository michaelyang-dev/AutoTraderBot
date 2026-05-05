#!/usr/bin/env python3
"""
Event Short Manager — Short Sleeve Signal Generator for v10
=============================================================
Generates short signals from SEC forced-selling events.
Runs alongside signal_server_v9.py as an independent sleeve.

Architecture:
  signal_server.py
    ├── /signals (long signals from signal_server_v9.py)
    └── /short-signals (short signals from this module)

  tradingEngine.js
    ├── Fetches /signals → executes longs
    └── Fetches /short-signals → executes shorts

Output format matches the existing signal JSON structure but with
signal="SHORT" instead of "BUY", plus short-specific fields.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from event_detector import EventDetector, CombinedEventDetector
from xbrl_detector import scan_universe, load_ticker_to_cik

logger = logging.getLogger(__name__)

# Strategy parameters (LOCKED — validated through 16 rounds of research)
MAX_POSITIONS = 10
HOLD_DAYS = 30
STOP_LOSS_PCT = 0.25
MIN_PRICE = 5.0
MIN_MKTCAP = 100  # $100M in Compustat units (millions)
SMA_PERIOD = 50
DEDUP_COOLDOWN_DAYS = 90
SLEEVE_KILL_DD = -0.35
SLEEVE_KILL_RESUME = -0.15


class EventShortManager:
    """
    Manages the event-driven short sleeve.
    Maintains state of active short positions and pending events.
    Called by signal_server.py to generate short signals.
    """

    def __init__(self, data_dir: str = "data", cache_dir: str = "data/short_sleeve"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.active_positions = {}  # symbol → {entry_date, entry_price, event_type, days_held}
        self.recent_entries = {}  # symbol → last entry date (for dedup)
        self.pending_events = []  # events detected but not yet entered
        self.trade_log = []
        self.sleeve_cumulative = 1.0
        self.sleeve_peak = 1.0
        self.halted = False

        # Detection
        self.detector = EventDetector()
        self.ticker_to_cik = {}
        self.last_scan_time = None
        self.scan_interval_minutes = 30

        # Price data (populated by signal_server)
        self.prices = {}  # symbol → current price
        self.sma50 = {}  # symbol → 50-day SMA

        # Load state from cache
        self._load_state()

    def _load_state(self):
        """Load persisted state from disk."""
        state_file = self.cache_dir / "short_sleeve_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.active_positions = state.get("active_positions", {})
                self.recent_entries = {k: pd.Timestamp(v) for k, v in
                                       state.get("recent_entries", {}).items()}
                self.sleeve_cumulative = state.get("sleeve_cumulative", 1.0)
                self.sleeve_peak = state.get("sleeve_peak", 1.0)
                self.halted = state.get("halted", False)
                logger.info(f"Loaded short sleeve state: {len(self.active_positions)} positions")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Persist state to disk."""
        state = {
            "active_positions": self.active_positions,
            "recent_entries": {k: str(v) for k, v in self.recent_entries.items()},
            "sleeve_cumulative": self.sleeve_cumulative,
            "sleeve_peak": self.sleeve_peak,
            "halted": self.halted,
            "last_update": datetime.now().isoformat(),
        }
        state_file = self.cache_dir / "short_sleeve_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def update_prices(self, price_map: dict, sma50_map: dict = None):
        """
        Update current prices and SMAs. Called by signal_server each cycle.

        Args:
            price_map: {symbol: current_price}
            sma50_map: {symbol: 50-day SMA value} (optional, computed if not provided)
        """
        self.prices = price_map
        if sma50_map:
            self.sma50 = sma50_map

    def scan_for_events(self):
        """
        Scan EDGAR for new forced-selling events.
        Rate-limited to once per scan_interval_minutes.
        """
        now = datetime.now()
        if self.last_scan_time and (now - self.last_scan_time).seconds < self.scan_interval_minutes * 60:
            return []

        # Only scan during market hours (9:00-18:00 ET weekdays)
        if now.weekday() >= 5 or now.hour < 9 or now.hour > 18:
            return []

        logger.info("Scanning EDGAR for new events...")
        self.last_scan_time = now

        try:
            events = self.detector.scan(lookback_days=1)
            new_events = []
            for event in events:
                symbol = event.get("ticker", "")
                if not symbol:
                    continue
                # Dedup cooldown
                if symbol in self.recent_entries:
                    last = self.recent_entries[symbol]
                    if isinstance(last, str):
                        last = pd.Timestamp(last)
                    if (now - last).days < DEDUP_COOLDOWN_DAYS:
                        continue
                # Already in position
                if symbol in self.active_positions:
                    continue
                new_events.append(event)

            if new_events:
                logger.info(f"Found {len(new_events)} new qualifying events")
                self.pending_events.extend(new_events)

            return new_events
        except Exception as e:
            logger.error(f"Event scan failed: {e}")
            return []

    def check_kill_switch(self):
        """Check sleeve-level circuit breaker."""
        if self.sleeve_peak > 0:
            dd = (self.sleeve_cumulative - self.sleeve_peak) / self.sleeve_peak
        else:
            dd = 0

        if dd < SLEEVE_KILL_DD and not self.halted:
            logger.warning(f"SHORT SLEEVE KILL SWITCH: DD={dd:.1%}")
            self.halted = True
        elif self.halted and dd > SLEEVE_KILL_RESUME:
            logger.info(f"Short sleeve kill switch released: DD={dd:.1%}")
            self.halted = False

        return self.halted

    def generate_signals(self, long_positions: set = None):
        """
        Generate short signals for the trading engine.

        Args:
            long_positions: set of symbols currently held long by v10
                           (conflict check — don't short what we're long)

        Returns:
            dict matching the signal_server.py output format:
            {
                "signals": [...],
                "last_update": "ISO timestamp",
                "is_stale": false,
                "strategy": "event_short_v1",
                "sleeve_status": {...}
            }
        """
        long_positions = long_positions or set()
        now = datetime.now()

        # Check kill switch
        self.check_kill_switch()

        # Scan for new events (rate-limited internally)
        self.scan_for_events()

        signals = []

        # 1. Generate EXIT signals for existing positions
        for symbol, pos in list(self.active_positions.items()):
            price = self.prices.get(symbol)
            if price is None:
                continue

            entry_price = pos.get("entry_price", price)
            days_held = pos.get("days_held", 0)
            should_exit = False
            exit_reason = ""

            # Hold period complete
            if days_held >= HOLD_DAYS:
                should_exit = True
                exit_reason = "hold_complete"

            # Stop-loss (stock rose 25% from entry)
            if price / entry_price - 1 > STOP_LOSS_PCT:
                should_exit = True
                exit_reason = "stop_loss"

            if should_exit:
                signals.append({
                    "symbol": symbol,
                    "signal": "COVER",  # buy to cover the short
                    "probability": 1.0,
                    "confidence": 1.0,
                    "ml_mode": "EVENT_SHORT",
                    "rank": 0,
                    "is_top_5": False,
                    "strategy": "event_short",
                    "exit_reason": exit_reason,
                    "entry_price": entry_price,
                    "days_held": days_held,
                    "current_pnl": -(price / entry_price - 1),
                })

                # Log the trade
                self.trade_log.append({
                    "symbol": symbol,
                    "event_type": pos.get("event_type", ""),
                    "entry_date": pos.get("entry_date", ""),
                    "exit_date": now.isoformat(),
                    "entry_price": entry_price,
                    "exit_price": price,
                    "days_held": days_held,
                    "pnl": -(price / entry_price - 1),
                    "exit_reason": exit_reason,
                })

                del self.active_positions[symbol]
            else:
                # Increment days held
                pos["days_held"] = days_held + 1

        # 2. Generate ENTRY signals from pending events
        if not self.halted and len(self.active_positions) < MAX_POSITIONS:
            slots_available = MAX_POSITIONS - len(self.active_positions)

            for event in self.pending_events[:]:
                if slots_available <= 0:
                    break

                symbol = event.get("ticker", "")
                if not symbol:
                    continue

                # Conflict check: don't short what v10 is long
                if symbol in long_positions:
                    logger.info(f"Skipping {symbol} — held long by v10")
                    continue

                # Already in position
                if symbol in self.active_positions:
                    continue

                # Price check
                price = self.prices.get(symbol)
                if price is None or price < MIN_PRICE:
                    continue

                # SMA filter
                sma = self.sma50.get(symbol)
                if sma is not None and price >= sma:
                    continue  # above SMA — skip

                # Generate SHORT signal
                signals.append({
                    "symbol": symbol,
                    "signal": "SHORT",
                    "probability": 0.95,
                    "confidence": 0.95,
                    "ml_mode": "EVENT_SHORT",
                    "rank": len(self.active_positions) + 1,
                    "is_top_5": True,
                    "strategy": "event_short",
                    "event_type": event.get("event_type", ""),
                    "filing_date": event.get("filing_date", ""),
                    "position_pct": 1.0 / MAX_POSITIONS,
                })

                # Track the entry
                self.active_positions[symbol] = {
                    "entry_date": now.isoformat(),
                    "entry_price": price,
                    "event_type": event.get("event_type", ""),
                    "days_held": 0,
                }
                self.recent_entries[symbol] = now

                self.pending_events.remove(event)
                slots_available -= 1

        # Save state
        self._save_state()

        # Build response
        return {
            "signals": signals,
            "last_update": now.isoformat(),
            "is_stale": False,
            "strategy": "event_short_v1",
            "sleeve_status": {
                "active_positions": len(self.active_positions),
                "max_positions": MAX_POSITIONS,
                "halted": self.halted,
                "pending_events": len(self.pending_events),
                "sleeve_dd": (self.sleeve_cumulative - self.sleeve_peak) / self.sleeve_peak
                    if self.sleeve_peak > 0 else 0,
                "total_trades": len(self.trade_log),
            },
        }

    def get_status(self):
        """Get sleeve status summary."""
        return {
            "active_positions": {
                sym: {
                    "entry_price": pos["entry_price"],
                    "event_type": pos["event_type"],
                    "days_held": pos["days_held"],
                    "current_price": self.prices.get(sym),
                    "pnl": -(self.prices.get(sym, pos["entry_price"]) / pos["entry_price"] - 1)
                        if self.prices.get(sym) else None,
                }
                for sym, pos in self.active_positions.items()
            },
            "max_positions": MAX_POSITIONS,
            "halted": self.halted,
            "pending_events": len(self.pending_events),
            "total_trades": len(self.trade_log),
            "recent_trades": self.trade_log[-5:] if self.trade_log else [],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = EventShortManager()
    print("Event Short Manager initialized")
    print(f"  Max positions: {MAX_POSITIONS}")
    print(f"  Hold period: {HOLD_DAYS} days")
    print(f"  Stop-loss: {STOP_LOSS_PCT:.0%}")
    print(f"  Active positions: {len(manager.active_positions)}")
    print(f"  Pending events: {len(manager.pending_events)}")
