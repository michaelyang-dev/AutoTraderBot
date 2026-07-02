#!/usr/bin/env python3
"""
IBKR Trading Engine
=====================
Executes the momentum long strategy via Interactive Brokers TWS API.
Consumes signals from signal_server.py and manages positions on IBKR.

Architecture:
    signal_server.py (port 5001) → /signals endpoint
        ↓
    ibkr_engine.py (this file) → reads signals, executes trades
        ↓
    IB Gateway (port 4002) → TWS API socket → IBKR servers

Features:
    - Fetches top-8 BUY signals from signal server
    - Rebalances portfolio to match signals
    - 15% position cap
    - 40% trailing stop per position (v12)
    - Regime detection (SPY vs SMA200)
    - Runs continuously during market hours
    - Telegram alerts for trades

Usage:
    cd ml_service && python3 ibkr_engine.py
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from ib_insync import *

# Setup
ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
os.chdir(ML_DIR)

from dotenv import load_dotenv
load_dotenv(ML_DIR.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ibkr_engine")

# Quiet ib_insync's per-tick portfolio/account spam: its Wrapper logs an INFO line
# for every position on every price change (updatePortfolio fires constantly), which
# buries connection/NAV/rebalance/error lines. Raising only this logger to WARNING
# keeps real ib_insync errors (they log at WARNING/ERROR) and our own errorEvent
# handler + main-loop connection checks, which are on separate loggers.
logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "4002"))       # 4001=live, 4002=paper
IBKR_INITIAL_CAPITAL = float(os.getenv("IBKR_INITIAL_CAPITAL", "30000"))  # funded amount, for since-inception P&L
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))
IB_GATEWAY_CONTAINER = os.getenv("IB_GATEWAY_CONTAINER", "ibgateway")  # docker container /reconnect restarts when the Gateway itself is down
SIGNAL_PORT = os.getenv("SIGNAL_SERVER_PORT", "5001")
SIGNAL_URL = f"http://localhost:{SIGNAL_PORT}/signals"
SIGNAL_HEALTH_URL = f"http://localhost:{SIGNAL_PORT}/health"

# v12 strategy parameters (must match backtest + signal_builder + tradingEngine.js)
MAX_POSITIONS = 30      # hold all combined sleeve picks (~22-25)
POSITION_CAP = 0.15     # v12: 15% max per position
TRAILING_STOP = 0.40    # v12: 40% trailing stop
LEVERAGE = 1.80         # 1.8x target to offset integer-share rounding drag (~1.44x effective)
REBALANCE_INTERVAL = 600  # check every 10 minutes
MIN_TRADE_PCT = 0.01    # don't trade if delta < 1% of portfolio (existing holdings only)

# Vol-scaling overlay (Phase 1): scale effective leverage DOWN when realized
# portfolio vol exceeds the target. Can only de-risk (never levers above base).
# Backtest (2018-2025): cuts 1.49x MaxDD ~-43%->-32% for ~2pp CAGR; helped in
# all 4 major selloffs; lowers turnover. Ramps in once 20+ NAV days accumulate.
VOL_SCALING = True
VOL_TARGET_1X = 0.15    # backtest semantic: target for the UNLEVERED (1x) portfolio vol
EFFECTIVE_LEVERAGE = 1.49  # account runs ~1.49x, so its NAV vol ~= 1.49 x the 1x vol.
# We measure the LEVERAGED NAV vol, so compare it against the leveraged target:
VOL_TARGET = VOL_TARGET_1X * EFFECTIVE_LEVERAGE  # ~0.22 account-NAV-vol target
VOL_LOOKBACK = 40       # trading days of NAV history for the vol estimate
VOL_SCALE_FLOOR = 0.30  # never cut effective leverage below 30% of base

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

# Short sleeve parameters (event-driven forced selling)
# DISABLED: EDGAR 8-K backtest showed -6.6% CAGR, -0.31 Sharpe (2015-2025)
# Only auditor_change produced positive returns; restatement/impairment/delisting lose money.
# Keep EDGAR monitor running for data collection; re-enable when validated.
SHORT_ENABLED = False
SHORT_MAX_POSITIONS = 10
SHORT_HOLD_DAYS = 30
SHORT_STOP_LOSS = 0.25   # exit if stock RISES 25% from entry
SHORT_ALLOCATION = 0.25  # 25% of NAV allocated to short sleeve
SHORT_MIN_PRICE = 5.0
SHORT_DEDUP_DAYS = 90
SHORT_EVENTS_FILE = ML_DIR / "data" / "short_sleeve" / "realtime_events.json"
# Primary event types that trigger shorts (validated in backtest)
SHORT_TRIGGER_TYPES = {"auditor_change", "financial_restatement", "material_impairment", "delisting_notice"}


def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        # Tag every message with the engine so IBKR vs Alpaca is unambiguous
        tag = "🟢 [IBKR LIVE] " if IB_PORT == 4001 else "🟡 [IBKR PAPER] "
        requests.post(url, json={"chat_id": TELEGRAM_CHAT, "text": tag + str(msg)}, timeout=10)
    except Exception:
        pass  # never crash trading loop for telegram


TRAILING_PEAKS_FILE = Path(__file__).resolve().parent / "data" / "ibkr_trailing_peaks.json"
NAV_HISTORY_FILE = Path(__file__).resolve().parent / "data" / "ibkr_nav_history.json"
CLOSE_SNAPSHOT_FILE = Path(__file__).resolve().parent / "data" / "ibkr_close_snapshot.json"
DISCONNECT_ESCALATE_SECS = 900  # only Telegram-alert if Gateway stays down >15 min (needs 2FA); brief blips are silent
HANG_TIMEOUT_SECS = 360  # watchdog: if the main loop makes no progress this long (e.g. Error-1100 hang), force-restart
REBAL_STATE_FILE = Path(__file__).resolve().parent / "data" / "ibkr_rebal_state.json"
REBAL_DAYS = 20  # rebalance cadence in trading days (must match backtest rebal_days)

class IBKREngine:
    def __init__(self):
        self.ib = IB()
        self.ib.errorEvent += self._on_ib_error   # catch Error 1100 connectivity loss
        self._last_progress = time.time()         # watchdog heartbeat
        self.positions = {}       # symbol -> {qty, avg_cost, market_value, peak_price}
        self.trailing_peaks = self._load_trailing_peaks()
        self.last_signals = []
        self.last_rebalance = None
        self.account_id = None
        self.running = False
        # Short sleeve state
        self.short_positions = {}  # symbol -> {qty, entry_price, entry_date, event_type, days_held}
        self.short_recent = {}     # symbol -> last entry datetime (dedup)
        self.short_sleeve_value = 0  # track sleeve P&L for kill switch
        self.short_sleeve_peak = 0
        self.short_halted = False
        # Drawdown-based position scaling
        self.portfolio_peak = 0.0
        # Rebalance clock — persisted to disk so the 20-day cadence survives restarts.
        # Defaults below = fresh-deploy behavior (allow an immediate first rebalance to converge);
        # _load_rebal_state() overrides them from disk if a prior state exists.
        self._trading_days_since_rebal = REBAL_DAYS
        self._last_rebal_date = None
        self._last_counted_day = None
        self._load_rebal_state()

    def _load_rebal_state(self):
        """Resume the rebalance clock from disk so a restart doesn't reset the
        20-day cadence (which would cause an extra rebalance at the next open)."""
        try:
            if REBAL_STATE_FILE.exists():
                import json
                from datetime import date
                d = json.load(open(REBAL_STATE_FILE))
                self._trading_days_since_rebal = d.get("trading_days_since_rebal", REBAL_DAYS)
                lrd = d.get("last_rebal_date");   self._last_rebal_date = date.fromisoformat(lrd) if lrd else None
                lcd = d.get("last_counted_day");  self._last_counted_day = date.fromisoformat(lcd) if lcd else None
                lr = d.get("last_rebalance");     self.last_rebalance = datetime.fromisoformat(lr) if lr else None
                log.info(f"Loaded rebal state: {self._trading_days_since_rebal}/{REBAL_DAYS} days since rebal, last_rebal={self._last_rebal_date}")
        except Exception as e:
            log.warning(f"Could not load rebal state: {e}")

    def _save_rebal_state(self):
        """Persist the rebalance clock so restarts resume the 20-day cadence."""
        try:
            import json
            d = {"trading_days_since_rebal": self._trading_days_since_rebal,
                 "last_rebal_date": self._last_rebal_date.isoformat() if self._last_rebal_date else None,
                 "last_counted_day": self._last_counted_day.isoformat() if self._last_counted_day else None,
                 "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None}
            with open(REBAL_STATE_FILE, "w") as f:
                json.dump(d, f)
        except Exception as e:
            log.warning(f"Could not save rebal state: {e}")

    def _load_trailing_peaks(self):
        """Load trailing peaks from disk (survives restarts)."""
        try:
            if TRAILING_PEAKS_FILE.exists():
                import json
                with open(TRAILING_PEAKS_FILE) as f:
                    data = json.load(f)
                log.info(f"Loaded {len(data)} trailing peaks from disk")
                return data
        except Exception as e:
            log.warning(f"Could not load trailing peaks: {e}")
        return {}

    def _save_trailing_peaks(self):
        """Persist trailing peaks to disk."""
        try:
            import json
            # Only save numeric peak prices, not datetime entries
            peaks = {k: v for k, v in self.trailing_peaks.items()
                     if isinstance(v, (int, float))}
            with open(TRAILING_PEAKS_FILE, "w") as f:
                json.dump(peaks, f)
        except Exception:
            pass

    # ── Vol-scaling overlay ──────────────────────────────────────────────
    def _load_nav_history(self):
        """Load [[date, nav], ...] history from disk."""
        try:
            if NAV_HISTORY_FILE.exists():
                import json
                with open(NAV_HISTORY_FILE) as f:
                    return json.load(f)
        except Exception as e:
            log.warning(f"Could not load NAV history: {e}")
        return []

    def _is_trading_day(self, d=None):
        """True if d (ISO 'YYYY-MM-DD' str, or None=today ET) is an NYSE trading day.
        Uses pandas_market_calendars when available (handles holidays like Juneteenth);
        falls back to a plain weekday check if the calendar can't be loaded."""
        from datetime import date as _date
        if d is None:
            from zoneinfo import ZoneInfo
            d = datetime.now(ZoneInfo("US/Eastern")).date()
        elif isinstance(d, str):
            d = _date.fromisoformat(d[:10])
        if d.weekday() >= 5:
            return False
        try:
            import pandas_market_calendars as mcal
            iso = d.isoformat()
            return len(mcal.get_calendar("NYSE").valid_days(iso, iso)) > 0
        except Exception:
            return True  # calendar unavailable -> weekday already passed above

    def record_nav(self, nav, at_close=False):
        """Append today's NAV once per trading day; keep last 70 days. Skips
        weekends/holidays so flat non-trading days don't dampen the realized-vol
        estimate used for vol-scaling (mirrors the backtest's trading-day series).

        at_close=True marks the authoritative 4pm value (the 16:05 EOD path) and may
        overwrite today's entry. All other callers (e.g. engine startup) only SEED a
        missing entry and never overwrite — an evening restart used to replace the 4pm
        close with an after-hours NAV, corrupting the next day's 'Today' P&L baseline."""
        try:
            import json
            if not self._is_trading_day():
                return
            today = datetime.now().date().isoformat()
            hist = self._load_nav_history()
            if hist and hist[-1][0] == today:
                if not at_close:
                    return               # today already recorded — never clobber the close mark
                hist[-1] = [today, nav]  # authoritative 4pm close value
            else:
                hist.append([today, nav])
            hist = hist[-70:]
            with open(NAV_HISTORY_FILE, "w") as f:
                json.dump(hist, f)
        except Exception:
            pass

    def _load_close_snapshot(self):
        """Load the most recent market-close portfolio snapshot (or None)."""
        try:
            if CLOSE_SNAPSHOT_FILE.exists():
                import json
                with open(CLOSE_SNAPSHOT_FILE) as f:
                    return json.load(f)
        except Exception as e:
            log.warning(f"Could not load close snapshot: {e}")
        return None

    async def _record_close_snapshot(self):
        """Snapshot the portfolio at the 4pm close so /portfolio and /pnl can show the
        official close value next to the live (after-hours) value. NAV is taken from the
        recorded EOD NAV history when available, so it matches the daily summary exactly."""
        try:
            import json
            s = await self._ibkr_snapshot()
            today = datetime.now().date().isoformat()           # server runs in ET
            hist = self._load_nav_history()
            close_nav = hist[-1][1] if (hist and hist[-1][0] == today and hist[-1][1]) else s["nav"]
            snap = {
                "date": today,
                "captured": datetime.now().strftime("%H:%M ET"),
                "nav": close_nav, "cash": s["cash"], "gross": s["gross"], "upl": s["upl"],
                "lev": (s["gross"] / close_nav if close_nav else 0),
                "positions": sorted(
                    [{"symbol": it.contract.symbol, "qty": it.position,
                      "value": it.marketValue, "upl": it.unrealizedPNL} for it in s["items"]],
                    key=lambda p: -p["value"]),
            }
            with open(CLOSE_SNAPSHOT_FILE, "w") as f:
                json.dump(snap, f)
            log.info(f"Close snapshot recorded: NAV ${close_nav:,.0f}, {len(snap['positions'])} positions")
        except Exception as e:
            log.warning(f"Close snapshot failed: {e}")

    # ── Disconnect alerting (suppress brief blips, escalate real outages) ──
    def _on_disconnect_detected(self):
        """Track outage start. Stay SILENT for brief blips; send ONE loud,
        actionable alert only once the Gateway has been down past the escalate
        threshold (the signature of an IBKR forced-2FA login that needs you)."""
        now = datetime.now()
        if self._disconnect_since is None:
            self._disconnect_since = now
        down_secs = (now - self._disconnect_since).total_seconds()
        if down_secs >= DISCONNECT_ESCALATE_SECS and not self._escalated:
            self._escalated = True
            send_telegram(
                f"🔴 IBKR Gateway still DOWN after {int(down_secs/60)} min — likely needs your 2FA.\n"
                f"Open the IBKR mobile app (IB Key) and approve the login, or the bot can't trade."
            )

    def _on_reconnect(self):
        """Reset outage state. Only confirm recovery if we'd escalated (i.e., it
        was a real outage you were alerted about) — brief blips stay silent."""
        if self._escalated:
            send_telegram("🟢 IBKR reconnected — Gateway back online, trading resumed.")
        self._disconnect_since = None
        self._escalated = False

    def _on_ib_error(self, reqId, errorCode, errorString, contract=None):
        """Log IBKR system messages. Error 1100 = connectivity to IBKR lost while the
        SOCKET stays open (isConnected() keeps returning True), which is exactly the
        silent-hang case the watchdog exists to catch. 1102 = restored."""
        if errorCode == 1100:
            log.error("IBKR Error 1100: connectivity to IBKR lost (socket still open) — watchdog armed")
        elif errorCode in (1101, 1102):
            log.info(f"IBKR Error {errorCode}: connectivity restored")

    def _start_watchdog(self):
        """Thread (not asyncio — survives a blocked loop): if the main loop makes no
        progress for HANG_TIMEOUT_SECS, alert and force-exit so PM2 restarts us with a
        clean reconnect. Catches Error-1100 hangs and any other stall that isConnected()
        can't see. Reconnect loops keep the heartbeat fresh, so this won't false-fire."""
        self._last_progress = time.time()

        def _watch():
            while True:
                time.sleep(30)
                stale = time.time() - getattr(self, "_last_progress", time.time())
                if stale > HANG_TIMEOUT_SECS:
                    log.error(f"WATCHDOG: no loop progress for {stale:.0f}s — force-restarting")
                    send_telegram(f"🔴 IBKR engine STUCK — no progress for {stale/60:.0f} min "
                                  f"(likely connectivity loss, e.g. you logged in elsewhere). "
                                  f"Auto-restarting; reply /status in ~1 min to confirm it recovered.")
                    os._exit(1)   # PM2 restarts -> fresh connect

        threading.Thread(target=_watch, daemon=True).start()
        log.info(f"Watchdog started (force-restart if no progress > {HANG_TIMEOUT_SECS}s)")

    def compute_vol_scale(self):
        """Scale effective leverage down when realized portfolio vol > VOL_TARGET.
        Returns a multiplier in [VOL_SCALE_FLOOR, 1.0]. 1.0 = no scaling (also
        the default until 20+ NAV days accumulate). Mirrors the backtest:
        vol_scale = clamp(VOL_TARGET / realized_vol, floor, 1.0)."""
        if not VOL_SCALING:
            return 1.0, None
        navs = [h[1] for h in self._load_nav_history()
                if self._is_trading_day(h[0])][-(VOL_LOOKBACK + 1):]
        if len(navs) < 20:
            return 1.0, None  # insufficient history — ramp-up, no scaling yet
        rets = [navs[i] / navs[i - 1] - 1 for i in range(1, len(navs)) if navs[i - 1] > 0]
        if len(rets) < 19:
            return 1.0, None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / len(rets)
        realized_vol = (var ** 0.5) * (252 ** 0.5)
        if realized_vol < 0.01:
            return 1.0, realized_vol
        vol_scale = min(1.0, max(VOL_SCALE_FLOOR, VOL_TARGET / realized_vol))
        return vol_scale, realized_vol

    def get_drawdown_scale(self, current_value):
        """Reduce position sizes as drawdown deepens from peak."""
        if self.portfolio_peak <= 0:
            return 1.0
        dd = 1 - current_value / self.portfolio_peak
        if dd <= 0.10: return 1.00
        if dd <= 0.15: return 0.85
        if dd <= 0.20: return 0.70
        if dd <= 0.25: return 0.50
        return 0.35

    async def connect(self):
        """Connect to IB Gateway."""
        log.info(f"Connecting to IB Gateway at {IB_HOST}:{IB_PORT}...")
        self.ib = IB()
        await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=False, timeout=20)
        # Request real-time market data (type 1) — live account has streaming subscription.
        # get_market_price() falls back to close if a symbol returns no real-time tick.
        self.ib.reqMarketDataType(1)
        accounts = self.ib.managedAccounts()
        self.account_id = accounts[0] if accounts else None
        log.info(f"Connected. Account: {self.account_id}")
        send_telegram(f"🟢 IBKR Engine connected. Account: {self.account_id}")

    async def get_account_summary(self):
        """Get account NAV and buying power."""
        summary = {}
        items = await self.ib.accountSummaryAsync()
        for item in items:
            if item.tag in ["NetLiquidation", "TotalCashValue", "BuyingPower"]:
                summary[item.tag] = float(item.value)
        return summary

    async def get_portfolio_value(self):
        """Get current portfolio NAV."""
        summary = await self.get_account_summary()
        return summary.get("NetLiquidation", 0)

    async def update_positions(self):
        """Refresh positions from IBKR."""
        self.positions = {}
        positions = await self.ib.reqPositionsAsync()
        for pos in positions:
            if pos.contract.secType == "STK" and pos.position != 0:
                sym = pos.contract.symbol
                self.positions[sym] = {
                    "qty": int(pos.position),
                    "avg_cost": pos.avgCost,
                    "contract": pos.contract,
                }
        log.info(f"Positions: {list(self.positions.keys())} ({len(self.positions)} total)")

    def fetch_signals(self):
        """Fetch BUY signals from signal server (with health check)."""
        try:
            # Quick health check first — refuse stale signals
            health = requests.get(SIGNAL_HEALTH_URL, timeout=5)
            if health.status_code == 200:
                h = health.json()
                if h.get("is_stale"):
                    log.warning(f"Signal server data is STALE (last update: {h.get('last_update')}) — NOT trading")
                    send_telegram(f"⚠️ IBKR skipping rebalance — signals are STALE")
                    return []
            resp = requests.get(SIGNAL_URL, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                signals = data.get("signals", data) if isinstance(data, dict) else data
                buys = [s for s in signals if s.get("signal") == "BUY"]
                buys.sort(key=lambda s: s.get("probability", 0), reverse=True)
                self.last_signals = buys[:MAX_POSITIONS]
                log.info(f"Signals: {[s['symbol'] for s in self.last_signals]}")
                return self.last_signals
        except Exception as e:
            log.error(f"Signal fetch failed: {e}")
        return []

    def is_market_open(self):
        """Check if US market is open."""
        try:
            resp = requests.get(SIGNAL_HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("market_open", False)
        except Exception:
            log.debug("Signal server unreachable for market_open check, using time fallback")
        # Fallback: check time
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("US/Eastern"))
        if now.weekday() >= 5:
            return False
        if now.hour < 9 or now.hour >= 16:
            return False
        if now.hour == 9 and now.minute < 30:
            return False
        return True

    async def get_market_price(self, contract):
        """Get current market price for a contract."""
        await self.ib.qualifyContractsAsync(contract)
        ticker = self.ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(4)  # longer wait for delayed data
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)
        if price and price > 0 and not util.isNan(price):
            return price
        # Fallback chain: last → close → delayed last → delayed close
        for attr in ["last", "close", "delayedLast", "delayedClose"]:
            val = getattr(ticker, attr, None)
            if val and val > 0 and not util.isNan(val):
                return val
        return None

    async def check_trailing_stops(self):
        """Check and execute trailing stops."""
        portfolio_value = await self.get_portfolio_value()
        if portfolio_value <= 0:
            return

        await self.update_positions()

        # Prune orphan peaks for stocks no longer held — prevents a stale high peak
        # from a prior stint causing an immediate/loose stop if the name is re-bought.
        # Guard on non-empty positions so a transient empty fetch can't nuke live peaks.
        if self.positions:
            for orphan in [s for s in list(self.trailing_peaks) if s not in self.positions]:
                del self.trailing_peaks[orphan]

        for sym, pos in list(self.positions.items()):
            contract = pos.get("contract")
            if not contract:
                contract = Stock(sym, "SMART", "USD")

            price = await self.get_market_price(contract)
            if not price:
                continue

            # Update trailing peak
            if sym not in self.trailing_peaks:
                self.trailing_peaks[sym] = price
            if price > self.trailing_peaks[sym]:
                self.trailing_peaks[sym] = price

            # Check stop
            peak = self.trailing_peaks[sym]
            dd = (price - peak) / peak
            if dd < -TRAILING_STOP:
                log.warning(f"TRAILING STOP: {sym} dropped {dd:.1%} from peak ${peak:.2f}")
                await self.sell_position(sym, pos["qty"], f"trailing_stop ({dd:.1%})")
                del self.trailing_peaks[sym]

        # Persist peaks to disk after every check
        self._save_trailing_peaks()

    async def sell_position(self, symbol, qty, reason="rebalance"):
        """Sell a position. Never sell more than currently held (prevents accidental shorts)."""
        # Refresh positions to get actual qty
        await self.update_positions()
        actual_pos = self.positions.get(symbol, {})
        actual_qty = actual_pos.get("qty", 0)

        if actual_qty <= 0:
            log.warning(f"SKIP SELL {symbol}: no long position (qty={actual_qty})")
            return None

        sell_qty = min(abs(qty), actual_qty)  # never sell more than held

        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        order = MarketOrder("SELL", sell_qty)
        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(3)

        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice

        if price and price > 0:
            log.info(f"SELL {sell_qty} {symbol}: {status} filled={filled} @ ${price:.2f} ({reason})")
            send_telegram(f"📉 SELL {sell_qty} {symbol} @ ${price:.2f} | {reason}")
        else:
            log.info(f"SELL {sell_qty} {symbol}: {status} filled={filled} — market order, fill pending ({reason})")
            send_telegram(f"📉 SELL {sell_qty} {symbol} — market order submitted, fill pending | {reason}")

        # Clean up tracking
        if symbol in self.trailing_peaks:
            del self.trailing_peaks[symbol]

        return trade

    async def buy_position(self, symbol, qty, reason="signal"):
        """Buy a position."""
        if qty <= 0:
            log.warning(f"SKIP BUY {symbol}: invalid qty={qty}")
            return None
        if qty > 10000:
            log.error(f"SANITY CHECK: {symbol} qty={qty} exceeds 10000 — aborting buy")
            send_telegram(f"🔴 SANITY CHECK: tried to buy {qty} shares of {symbol}!")
            return None
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        order = MarketOrder("BUY", qty)
        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(3)

        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice

        if price and price > 0:
            log.info(f"BUY {qty} {symbol}: {status} filled={filled} @ ${price:.2f} ({reason})")
            send_telegram(f"📈 BUY {qty} {symbol} @ ${price:.2f} | {reason}")
        else:
            # Market order not filled within the wait (e.g., queued pre-open) — avgFillPrice is 0.
            log.info(f"BUY {qty} {symbol}: {status} filled={filled} — market order, fill pending ({reason})")
            send_telegram(f"📈 BUY {qty} {symbol} — market order submitted, fill pending | {reason}")

        return trade

    async def rebalance(self):
        """Rebalance portfolio to match signal server picks.

        v12: Matches backtest exactly:
        - Full reconstruction every REBAL_DAYS trading days
        - On rebal day: sell ALL not-in-target, buy ALL in-target, resize ALL
        - Between rebal days: only trailing stops fire, NO buying/selling
        """
        # Rebal clock (_trading_days_since_rebal / _last_rebal_date / _last_counted_day) is
        # initialized AND loaded-from-disk in __init__, so it survives restarts.
        # Only count trading days when market is open.
        if not self.is_market_open():
            return

        # Count trading days (only increment once per calendar day)
        today = datetime.now().date()
        if self._last_counted_day != today:
            self._last_counted_day = today
            self._trading_days_since_rebal += 1
            log.info(f"Trading day count: {self._trading_days_since_rebal}/{REBAL_DAYS}")
            self._save_rebal_state()  # persist daily so the cadence survives restarts

        if self._trading_days_since_rebal < REBAL_DAYS:
            return  # only trailing stops fire between rebal days

        signals = self.fetch_signals()
        if not signals:
            log.warning("No signals available — skipping rebalance")
            return

        log.info(f"═══ REBALANCE DAY ({self._trading_days_since_rebal}d since last) ═══")
        self._trading_days_since_rebal = 0
        self._last_rebal_date = today
        self._save_rebal_state()

        # SAFETY: Close any accidental short positions first
        await self.update_positions()
        for sym, pos in list(self.positions.items()):
            if pos["qty"] < 0:
                log.warning(f"EMERGENCY COVER: {sym} has short position ({pos['qty']} shares)")
                contract = Stock(sym, "SMART", "USD")
                await self.ib.qualifyContractsAsync(contract)
                cover_order = MarketOrder("BUY", abs(pos["qty"]))
                cover_order.tif = "DAY"
                trade = self.ib.placeOrder(contract, cover_order)
                await asyncio.sleep(5)
                log.info(f"Cover order status: {trade.orderStatus.status} filled={trade.orderStatus.filled}")
                send_telegram(f"⚠️ COVERING accidental short: {sym} ({abs(pos['qty'])} shares)")

        portfolio_value = await self.get_portfolio_value()
        if portfolio_value <= 0:
            log.error("Portfolio value is 0 — skipping rebalance")
            return

        await self.update_positions()

        target_symbols = set(s["symbol"] for s in signals)
        held_symbols = set(self.positions.keys())

        # 1. SELL ALL positions not in signals (no min-hold on rebal day — matches backtest)
        for sym in held_symbols - target_symbols:
            pos = self.positions[sym]
            log.info(f"REBAL SELL {sym} — no longer in target set")
            await self.sell_position(sym, pos["qty"], "rebalance_exit")

        # Wait for sells to settle
        if held_symbols - target_symbols:
            await asyncio.sleep(2)
            await self.update_positions()
            portfolio_value = await self.get_portfolio_value()

        # Vol-scaling overlay: scale effective leverage by realized portfolio vol.
        # Logs the realized-leverage path for the live-vs-backtest watch-item.
        vol_scale, realized_vol = self.compute_vol_scale()
        eff_leverage = LEVERAGE * vol_scale
        if realized_vol is not None:
            log.info(f"VOL-SCALE: realized_vol={realized_vol:.1%} target={VOL_TARGET:.0%} "
                     f"-> scale={vol_scale:.2f} -> effective leverage {eff_leverage:.2f}x (base {LEVERAGE})")
            send_telegram(f"📊 Vol-scale: vol {realized_vol:.0%} → lev {eff_leverage:.2f}x ({vol_scale:.2f}× base)")
        else:
            log.info(f"VOL-SCALE: ramp-up (insufficient NAV history) — no scaling, leverage {LEVERAGE}x")

        # 2. Compute target weights — signal-proportional (matches backtest)
        # Higher conviction picks get more capital
        total_prob = sum(s.get("probability", 0) for s in signals)
        signal_targets = {}
        for sig in signals:
            prob = sig.get("probability", 0)
            if total_prob > 0 and prob > 0:
                w = (prob / total_prob) * eff_leverage
                w = min(w, POSITION_CAP)  # v12: cap at 15% (after vol-scale, matches backtest)
            else:
                w = eff_leverage / MAX_POSITIONS  # fallback equal weight
            signal_targets[sig["symbol"]] = portfolio_value * w

        # 3. BUY new positions / adjust existing
        for sig in signals:
            sym = sig["symbol"]
            contract = Stock(sym, "SMART", "USD")

            price = await self.get_market_price(contract)
            if not price or price <= 0:
                log.warning(f"Cannot get price for {sym} — skipping")
                continue

            target_value_per_position = signal_targets.get(sym, portfolio_value * LEVERAGE / MAX_POSITIONS)
            current_qty = self.positions.get(sym, {}).get("qty", 0)
            current_value = current_qty * price
            target_qty = int(target_value_per_position / price)
            delta_qty = target_qty - current_qty

            # Skip if delta is too small — ONLY for existing holdings (avoid churn).
            delta_value = abs(delta_qty * price)
            if current_qty > 0 and delta_value < portfolio_value * MIN_TRADE_PCT:
                continue

            if delta_qty > 0:
                # BUY (new position or top-up)
                await self.buy_position(sym, delta_qty, "rebalance")
                # Only set trailing peak if new position or price is higher than existing peak
                existing_peak = self.trailing_peaks.get(sym, 0)
                if price > existing_peak:
                    self.trailing_peaks[sym] = price
            elif delta_qty < 0:
                # TRIM (sell excess)
                await self.sell_position(sym, abs(delta_qty), "trim_overweight")

        self.last_rebalance = datetime.now()
        self._save_rebal_state()
        await self.update_positions()
        log.info(f"Rebalance complete. Positions: {list(self.positions.keys())}")

    # ═══════════════════════════════════════════════════════════════
    # SHORT SLEEVE
    # ═══════════════════════════════════════════════════════════════
    def load_short_events(self):
        """Load pending short events from EDGAR monitor."""
        if not SHORT_EVENTS_FILE.exists():
            return []
        try:
            with open(SHORT_EVENTS_FILE) as f:
                events = json.load(f)
            # Filter to primary events only
            primary = [e for e in events
                       if e.get("is_primary", False)
                       and any(t in SHORT_TRIGGER_TYPES for t in e.get("event_types", []))]
            return primary
        except Exception as e:
            log.error(f"Failed to load short events: {e}")
            return []

    async def check_short_exits(self):
        """Check short positions for exit conditions."""
        for sym, pos in list(self.short_positions.items()):
            if pos["qty"] == 0:
                continue

            contract = Stock(sym, "SMART", "USD")
            price = await self.get_market_price(contract)
            if not price:
                continue

            pos["days_held"] += 1
            should_exit = False
            reason = ""

            # Hold period complete
            if pos["days_held"] >= SHORT_HOLD_DAYS:
                should_exit = True
                reason = "hold_complete"

            # Stop loss: stock ROSE 25% from entry
            if price / pos["entry_price"] - 1 > SHORT_STOP_LOSS:
                should_exit = True
                reason = f"stop_loss (stock up {price/pos['entry_price']-1:.0%})"

            if should_exit:
                # Cover: buy to close
                await self.cover_short(sym, pos["qty"], reason)

    async def short_sell(self, symbol, qty, reason="event_short"):
        """Short sell a stock."""
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        order = MarketOrder("SELL", qty)
        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(3)

        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice

        log.info(f"SHORT {qty} {symbol}: {status} filled={filled} @ ${price:.2f} ({reason})")
        send_telegram(f"🔴 SHORT {qty} {symbol} @ ${price:.2f} | {reason}")

        if status in ["Filled", "Submitted", "PreSubmitted"]:
            self.short_positions[symbol] = {
                "qty": int(qty),
                "entry_price": price if price > 0 else 0,
                "entry_date": datetime.now().isoformat(),
                "event_type": reason,
                "days_held": 0,
            }
        return trade

    async def cover_short(self, symbol, qty, reason="exit"):
        """Buy to cover a short position."""
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        order = MarketOrder("BUY", abs(qty))
        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(3)

        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice

        entry = self.short_positions.get(symbol, {}).get("entry_price", 0)
        pnl = (entry - price) / entry * 100 if entry > 0 else 0

        log.info(f"COVER {qty} {symbol}: {status} @ ${price:.2f} P&L={pnl:+.1f}% ({reason})")
        send_telegram(f"🟢 COVER {qty} {symbol} @ ${price:.2f} | P&L={pnl:+.1f}% | {reason}")

        if symbol in self.short_positions:
            del self.short_positions[symbol]
        if symbol in self.short_recent:
            self.short_recent[symbol] = datetime.now()
        return trade

    def check_short_kill_switch(self):
        """Check sleeve-level circuit breaker (-35% DD → halt)."""
        if not self.short_positions:
            return False

        # Compute sleeve cumulative value (sum of all short P&L)
        sleeve_pnl = sum(
            pos.get("entry_price", 0) * pos.get("qty", 0)  # this is approximate
            for pos in self.short_positions.values()
        )
        if sleeve_pnl > self.short_sleeve_peak:
            self.short_sleeve_peak = sleeve_pnl

        if self.short_sleeve_peak > 0:
            dd = (sleeve_pnl - self.short_sleeve_peak) / self.short_sleeve_peak
            if dd < -0.35 and not self.short_halted:
                log.warning(f"SHORT KILL SWITCH: sleeve DD = {dd:.1%}")
                send_telegram(f"🛑 SHORT KILL SWITCH: DD = {dd:.1%} — halting new entries")
                self.short_halted = True
            elif self.short_halted and dd > -0.15:
                log.info(f"Short kill switch released: DD = {dd:.1%}")
                self.short_halted = False

        return self.short_halted

    async def process_short_events(self):
        """Process new short events from EDGAR monitor."""
        if not SHORT_ENABLED:
            return

        # Kill switch check
        if self.check_short_kill_switch():
            return

        if len(self.short_positions) >= SHORT_MAX_POSITIONS:
            return

        events = self.load_short_events()
        if not events:
            return

        portfolio_value = await self.get_portfolio_value()
        short_budget = portfolio_value * SHORT_ALLOCATION
        per_position = short_budget / SHORT_MAX_POSITIONS

        # Get long positions to avoid conflict (don't short what we're long)
        await self.update_positions()
        long_syms = set(self.positions.keys())

        for event in events:
            if len(self.short_positions) >= SHORT_MAX_POSITIONS:
                break

            ticker = event.get("ticker", "")
            if not ticker:
                continue

            # Already in short position
            if ticker in self.short_positions:
                continue

            # Conflict: held long
            if ticker in long_syms:
                log.info(f"Short skip {ticker} — held long")
                continue

            # Dedup cooldown
            if ticker in self.short_recent:
                last = self.short_recent[ticker]
                if isinstance(last, str):
                    last = datetime.fromisoformat(last)
                if (datetime.now() - last).days < SHORT_DEDUP_DAYS:
                    continue

            # Get price and check filters
            contract = Stock(ticker, "SMART", "USD")
            try:
                await self.ib.qualifyContractsAsync(contract)
            except Exception:
                log.debug(f"Failed to qualify contract for {ticker}")
                continue

            price = await self.get_market_price(contract)
            if not price or price < SHORT_MIN_PRICE:
                continue

            # SMA50 filter: must be below 50-day SMA (critical for avoiding V-recoveries)
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr="70 D",
                    barSizeSetting="1 day", whatToShow="ADJUSTED_LAST",
                    useRTH=True, formatDate=1)
                if bars and len(bars) >= 50:
                    closes = [b.close for b in bars[-50:]]
                    sma50 = sum(closes) / len(closes)
                    if price >= sma50:
                        log.info(f"Short skip {ticker} — above SMA50 ({price:.0f} >= {sma50:.0f})")
                        continue
                else:
                    log.warning(f"Short skip {ticker} — insufficient price history")
                    continue
            except Exception as e:
                log.warning(f"Short skip {ticker} — SMA50 check failed: {e}")
                continue

            # Size the position
            shares = int(per_position / price)
            if shares <= 0:
                continue

            event_types = ", ".join(event.get("event_types", []))
            log.info(f"Short signal: {ticker} — {event_types}")
            await self.short_sell(ticker, shares, f"event: {event_types}")
            self.short_recent[ticker] = datetime.now()

    # ═══════════════════════════════════════════════════════════════
    # TELEGRAM COMMAND BOT  (read-only — cannot place trades)
    # ═══════════════════════════════════════════════════════════════
    def _tg_send_raw(self, text, chat_id=None):
        """Send a reply to a Telegram chat (defaults to the primary chat). Tries HTML
        parse mode first (bold headers + <pre> aligned tables); if Telegram rejects the
        markup (400 parse error), resends as plain text so a reply is never dropped."""
        try:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                              json={"chat_id": chat_id or TELEGRAM_CHAT, "text": text,
                                    "parse_mode": "HTML"}, timeout=10)
            if r.status_code != 200:  # bad markup — fall back to plain so the user still gets it
                import re
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                              json={"chat_id": chat_id or TELEGRAM_CHAT,
                                    "text": re.sub(r"</?(b|i|code|pre)>", "", text)}, timeout=10)
        except Exception:
            pass

    def _tg_allowed_chats(self):
        """Chat IDs allowed to use the command bot. Live-editable via
        data/telegram_allowed.json (a JSON list of IDs) — no restart needed.
        Defaults to just the primary chat if the file is absent."""
        import json
        f = Path(__file__).resolve().parent / "data" / "telegram_allowed.json"
        try:
            if f.exists():
                return set(str(c) for c in json.load(open(f)))
        except Exception:
            pass
        return {str(TELEGRAM_CHAT)} if TELEGRAM_CHAT else set()

    def _alpaca_get(self, path):
        """GET the Alpaca paper API; returns parsed JSON or None."""
        ak = (os.getenv("ALPACA_API_KEY") or "").strip()
        sk = (os.getenv("ALPACA_SECRET_KEY") or "").strip()
        if not ak or not sk:
            return None
        try:
            r = requests.get("https://paper-api.alpaca.markets" + path,
                             headers={"APCA-API-KEY-ID": ak, "APCA-API-SECRET-KEY": sk}, timeout=10)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def _alpaca_daily_bars(self, symbols):
        """Per-symbol daily price data from the Alpaca DATA API (works for ANY symbol,
        not just Alpaca holdings — used to price the IBKR book's daily moves too).
        Returns {sym: {px, close, prev, bar_date}}: px = latest trade (live-ish, IEX),
        close = today's session close (dailyBar), prev = previous session close.
        NOTE: chosen over the positions API's intraday fields because those RESET after
        Alpaca's evening EOD processing (change_today=0 at night); daily bars stay correct."""
        ak = (os.getenv("ALPACA_API_KEY") or "").strip()
        sk = (os.getenv("ALPACA_SECRET_KEY") or "").strip()
        out = {}
        if not ak or not sk or not symbols:
            return out
        syms = sorted(set(symbols))
        for i in range(0, len(syms), 50):
            try:
                r = requests.get("https://data.alpaca.markets/v2/stocks/snapshots",
                                 params={"symbols": ",".join(syms[i:i + 50]), "feed": "iex"},
                                 headers={"APCA-API-KEY-ID": ak, "APCA-API-SECRET-KEY": sk},
                                 timeout=10)
                if r.status_code != 200:
                    continue
                for sym, s in r.json().items():
                    daily = s.get("dailyBar") or {}
                    prev = s.get("prevDailyBar") or {}
                    lt = s.get("latestTrade") or {}
                    out[sym] = {"px": lt.get("p") or daily.get("c"),
                                "close": daily.get("c"), "prev": prev.get("c"),
                                "bar_date": (daily.get("t") or "")[:10]}
            except Exception:
                continue
        return out

    async def _ibkr_snapshot(self):
        """Live IBKR NAV + per-position P&L from the engine's own connection."""
        summary = await self.get_account_summary()
        nav = summary.get("NetLiquidation", 0.0)
        cash = summary.get("TotalCashValue", 0.0)
        items = [it for it in self.ib.portfolio()
                 if it.position != 0 and it.contract.secType == "STK"]
        gross = sum(abs(it.marketValue) for it in items)
        upl = sum(it.unrealizedPNL for it in items)
        return {"nav": nav, "cash": cash, "gross": gross, "upl": upl,
                "lev": (gross / nav if nav else 0), "items": items}

    # Commands open to every allowlisted user — pure reads, no state change.
    # DEFAULT-DENY: anything NOT in this set is owner-only, so any action command
    # (now /reconnect, and any added later) is automatically restricted to the owner.
    READONLY_COMMANDS = frozenset({
        "start", "help", "portfolio", "positions", "alpaca", "pnl", "daily",
        "status", "signals", "rebal", "data", "connection"})

    @staticmethod
    def _tg_header(title, mkt_open=None):
        """Uniform one-line command header: bold title · time · market state."""
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("US/Eastern"))
        state = "" if mkt_open is None else (" · 🟢 open" if mkt_open else " · 🌙 closed")
        return f"<b>{title}</b> · <i>{now.strftime('%b %d, %I:%M %p').replace(' 0', ' ')}{state}</i>\n"

    @staticmethod
    def _tg_table(rows):
        """Monospace-align rows WITHOUT <pre>: one <code> span per line. Telegram puts
        a copy-chip overlay on <pre> blocks (it covers the top-right numbers); per-line
        <code> keeps the alignment and has no overlay."""
        return "\n".join(f"<code>{r}</code>" for r in rows)

    @staticmethod
    def _chip(x):
        """Green/red chip for gains/losses (Telegram has no colored text — squares are
        the color channel; 🟢/🔵 circles stay reserved for the IBKR/Alpaca book markers)."""
        return "🟩" if x >= 0 else "🟥"

    async def _handle_command(self, text, chat_id=None):
        cmd = text.split()[0].lower().lstrip("/").split("@")[0]
        # Owner = the configured TELEGRAM_CHAT_ID (8746062845). Only the owner may run
        # commands that act on the live engine; everyone else allowlisted gets reads only.
        is_owner = bool(TELEGRAM_CHAT) and str(chat_id) == str(TELEGRAM_CHAT)
        if cmd not in self.READONLY_COMMANDS and not is_owner:
            log.info(f"Telegram: non-owner chat {chat_id} blocked from action '/{cmd}'")
            return ("🔒 Read-only access.\nThat command performs an action on the live "
                    "engine and is restricted to the account owner.\nYou can use every "
                    "read-only command — send /help to see them.")
        if cmd in ("start", "help"):
            base = (self._tg_header("🤖 AutoTrader Bot") +
                    "\n<b>💵 Money</b>\n"
                    "/pnl — today, after-hours &amp; total gains\n"
                    "/daily — <i>today's move, position by position</i>\n"
                    "/portfolio — NAV, cash, leverage (both books)\n"
                    "\n<b>📋 Books</b>\n"
                    "/positions — IBKR holdings + P&amp;L since entry\n"
                    "/alpaca — Alpaca paper holdings + P&amp;L\n"
                    "\n<b>⚙️ System</b>\n"
                    "/status — engine + signal health\n"
                    "/connection — live data-feed round-trip test\n"
                    "/signals — current top picks\n"
                    "/rebal — rebalance countdown\n"
                    "/data — data freshness check")
            if is_owner:
                base += "\n\n<b>🔑 Owner</b>\n/reconnect — force a clean engine/Gateway reconnect"
            return base
        handlers = {"portfolio": self._cmd_portfolio, "positions": self._cmd_positions,
                    "alpaca": self._cmd_alpaca, "pnl": self._cmd_pnl, "daily": self._cmd_daily,
                    "status": self._cmd_status,
                    "signals": self._cmd_signals, "rebal": self._cmd_rebal, "data": self._cmd_data,
                    "connection": self._cmd_connection, "reconnect": self._cmd_reconnect}
        h = handlers.get(cmd)
        if not h:
            return f"Unknown command: /{cmd}. Try /help"
        out = h()
        return await out if asyncio.iscoroutine(out) else out

    async def _cmd_portfolio(self):
        import html as _h
        mkt_open = self.is_market_open()
        out = [self._tg_header("📊 PORTFOLIO", mkt_open)]
        try:
            s = await self._ibkr_snapshot()
            out.append("<b>🟢 IBKR LIVE</b>\n" + self._tg_table([
                f"{'NAV':<10}│ ${s['nav']:>10,.0f}",
                f"{'Cash':<10}│ ${s['cash']:>10,.0f}",
                f"{'Positions':<10}│ {len(s['items']):>11}",
                f"{'Leverage':<10}│ {s['lev']:>10.2f}x"]))
        except Exception as e:
            out.append(f"<b>🟢 IBKR LIVE</b> — ⚠️ {_h.escape(str(e))}")
        snap = self._load_close_snapshot()
        if snap:
            out.append(f"<i>at close {snap['date']}: ${snap['nav']:,.0f} · "
                       f"{len(snap.get('positions', []))} pos · {snap.get('lev', 0):.2f}x</i>")
        acct = self._alpaca_get("/v2/account"); poss = self._alpaca_get("/v2/positions")
        if acct:
            eq = float(acct.get("equity", 0)); cash = float(acct.get("cash", 0))
            gross = sum(abs(float(p["market_value"])) for p in poss) if poss else 0
            out.append("\n<b>🔵 ALPACA PAPER</b>\n" + self._tg_table([
                f"{'NAV':<10}│ ${eq:>10,.0f}",
                f"{'Cash':<10}│ ${cash:>10,.0f}",
                f"{'Positions':<10}│ {len(poss) if poss else 0:>11}",
                f"{'Leverage':<10}│ {(gross/eq if eq else 0):>10.2f}x"]))
        else:
            out.append("\n<b>🔵 ALPACA PAPER</b> — unavailable")
        return "\n".join(out)

    async def _cmd_positions(self):
        s = await self._ibkr_snapshot()
        items = sorted(s["items"], key=lambda it: -it.marketValue)
        rule = "   " + "─" * 6 + "┼" + "─" * 7 + "┼" + "─" * 8 + "┼" + "─" * 4
        rows = [f"   {'SYM':<6}│{'VALUE':>6} │{'P&L':>7} │{'%':>4}", rule]
        for it in items:
            cost = it.marketValue - it.unrealizedPNL
            pct = (it.unrealizedPNL / cost * 100) if cost else 0
            rows.append(f"{self._chip(it.unrealizedPNL)} {it.contract.symbol:<6}│{it.marketValue:>6,.0f} │{it.unrealizedPNL:>+7,.0f} │{pct:>+4.0f}")
        rows.append(rule)
        rows.append(f"{self._chip(s['upl'])} {'ALL':<6}│{s['gross']:>6,.0f} │{s['upl']:>+7,.0f} │")
        return (self._tg_header("📈 IBKR POSITIONS", self.is_market_open())
                + f"{len(items)} positions · P&amp;L since entry\n\n"
                + self._tg_table(rows))

    def _cmd_alpaca(self):
        poss = self._alpaca_get("/v2/positions")
        if poss is None:
            return "<b>🔵 ALPACA</b> — unavailable"
        poss = sorted(poss, key=lambda p: -float(p["market_value"]))
        upl = sum(float(p["unrealized_pl"]) for p in poss)
        gross = sum(abs(float(p["market_value"])) for p in poss)
        rule = "   " + "─" * 6 + "┼" + "─" * 10 + "┼" + "─" * 9 + "┼" + "─" * 4
        rows = [f"   {'SYM':<6}│{'VALUE':>9} │{'P&L':>8} │{'%':>4}", rule]
        for p in poss[:30]:
            rows.append(f"{self._chip(float(p['unrealized_pl']))} {p['symbol']:<6}│{float(p['market_value']):>9,.0f}"
                        f" │{float(p['unrealized_pl']):>+8,.0f} │{float(p['unrealized_plpc'])*100:>+4.0f}")
        rows.append(rule)
        rows.append(f"{self._chip(upl)} {'ALL':<6}│{gross:>9,.0f} │{upl:>+8,.0f} │")
        return (self._tg_header("🔵 ALPACA POSITIONS", self.is_market_open())
                + f"{len(poss)} positions · P&amp;L since entry\n\n"
                + self._tg_table(rows))

    async def _cmd_pnl(self):
        import html as _h
        mkt_open = self.is_market_open()
        out = [self._tg_header("💰 P&L", mkt_open)]
        today = datetime.now().date().isoformat()
        # ── IBKR: "Today" counts REGULAR HOURS only. Live NAV while the market is open;
        # frozen 4pm close after the bell (after-hours drift shown on its own line).
        try:
            s = await self._ibkr_snapshot()
            live_nav = s["nav"]
            snap = self._load_close_snapshot()
            hist = self._load_nav_history()
            prev = [p for p in hist if p[0] != today]
            have_close = bool(snap and snap.get("date") == today and snap.get("nav"))
            session_nav = live_nav if (mkt_open or not have_close) else snap["nav"]
            tag = "live" if mkt_open else ("4pm close" if have_close else "pre-mkt")
            rows = []
            if prev and prev[-1][1]:
                d = session_nav - prev[-1][1]
                lbl = "Today" if (mkt_open or have_close) else "Overnight"
                rows.append(f"{self._chip(d)} {lbl:<10}│{d:>+9,.0f} │{d/prev[-1][1]*100:>+6.2f}%")
            else:
                rows.append("Today      n/a (building history)")
            if not mkt_open and have_close:
                ah = live_nav - snap["nav"]
                rows.append(f"{self._chip(ah)} {'After-hrs':<10}│{ah:>+9,.0f} │{ah/snap['nav']*100:>+6.2f}%")
            rows.append(f"{self._chip(s['upl'])} {'Open P&L':<10}│{s['upl']:>+9,.0f} │")
            since = live_nav - IBKR_INITIAL_CAPITAL
            rows.append(f"{self._chip(since)} {'All-time':<10}│{since:>+9,.0f} │{since/IBKR_INITIAL_CAPITAL*100:>+6.1f}%")
            out.append(f"<b>🟢 IBKR · ${session_nav:,.0f}</b> <i>{tag}</i>\n" + self._tg_table(rows))
        except Exception as e:
            out.append(f"<b>🟢 IBKR</b> — ⚠️ {_h.escape(str(e))}")
        # ── ALPACA: same regular-hours discipline. equity drifts after hours, so when the
        # market is closed we rebuild the 4pm close as cash + Σ qty×dailyBar.close from the
        # data API (positions' intraday fields reset in the evening and can't be trusted).
        acct = self._alpaca_get("/v2/account")
        if acct:
            try:
                eq = float(acct.get("equity", 0)); le = float(acct.get("last_equity", 0))
                poss = self._alpaca_get("/v2/positions") or []
                rows, nav_shown, tag = [], eq, "live"
                if not mkt_open:
                    bars = self._alpaca_daily_bars([p["symbol"] for p in poss])
                    fresh = [b for b in bars.values() if b.get("bar_date") == today]
                    if poss and fresh and len(fresh) >= len(poss) * 0.8:   # today's bars exist → real close
                        close_eq = float(acct.get("cash", 0)) + sum(
                            float(p["qty"]) * (bars.get(p["symbol"], {}).get("close")
                                               or float(p["current_price"])) for p in poss)
                        nav_shown, tag = close_eq, "4pm close"
                        if le:
                            rows.append(f"{self._chip(close_eq-le)} {'Today':<10}│{close_eq-le:>+9,.0f} │{(close_eq-le)/le*100:>+6.2f}%")
                        ah = eq - close_eq
                        rows.append(f"{self._chip(ah)} {'After-hrs':<10}│{ah:>+9,.0f} │{ah/close_eq*100:>+6.2f}%")
                    else:                                                  # pre-market / no session today
                        tag = "pre-mkt"
                        if le:
                            rows.append(f"{self._chip(eq-le)} {'Overnight':<10}│{eq-le:>+9,.0f} │{(eq-le)/le*100:>+6.2f}%")
                elif le:
                    rows.append(f"{self._chip(eq-le)} {'Today':<10}│{eq-le:>+9,.0f} │{(eq-le)/le*100:>+6.2f}%")
                if poss:
                    upl_al = sum(float(p['unrealized_pl']) for p in poss)
                    rows.append(f"{self._chip(upl_al)} {'Open P&L':<10}│{upl_al:>+9,.0f} │")
                out.append(f"\n<b>🔵 ALPACA · ${nav_shown:,.0f}</b> <i>{tag}</i>\n" + self._tg_table(rows))
            except Exception as e:
                out.append(f"\n<b>🔵 ALPACA</b> — ⚠️ {_h.escape(str(e))}")
        return "\n".join(out)

    async def _cmd_daily(self):
        """NEW: today's move, position by position, for BOTH books. Prices from the
        Alpaca data API daily bars (correct during market hours AND after the close;
        also prices the IBKR book, which has no cheap per-symbol daily-change source)."""
        import html as _h
        mkt_open = self.is_market_open()
        out = [self._tg_header("📅 TODAY BY POSITION", mkt_open)]

        def table(entries, bars, nav):
            """entries: [(sym, qty)] → (header_suffix, aligned rows) sorted by day-$ impact."""
            rows, total, missing = [], 0.0, []
            for sym, qty in entries:
                b = bars.get(sym) or {}
                px = (b.get("px") if mkt_open else b.get("close")) or b.get("px")
                prev = b.get("prev")
                if not px or not prev:
                    missing.append(sym)
                    continue
                d = (px - prev) * qty
                total += d
                rows.append((sym, (px / prev - 1) * 100, d))
            rows.sort(key=lambda r: -r[2])
            rule = "   " + "─" * 6 + "┼" + "─" * 7 + "┼" + "─" * 9
            body = [f"   {'SYM':<6}│{'DAY%':>6} │{'DAY$':>9}", rule]
            body += [f"{self._chip(d)} {sym:<6}│{pct:>+6.1f} │{d:>+9,.0f}" for sym, pct, d in rows]
            body.append(rule)
            body.append(f"{self._chip(total)} {'TOTAL':<6}│{(total/nav*100 if nav else 0):>+6.2f} │{total:>+9,.0f}")
            note = f"\n<i>no data: {', '.join(missing)}</i>" if missing else ""
            pct_day = total / nav * 100 if nav else 0
            return f"{self._chip(total)} {pct_day:+.2f}% today", self._tg_table(body) + note

        # collect both books first so one data-API call prices everything
        ib_entries, ib_nav, ib_err = [], 0, None
        try:
            s = await self._ibkr_snapshot()
            ib_entries = [(it.contract.symbol, it.position) for it in s["items"]]
            ib_nav = s["nav"]
        except Exception as e:
            ib_err = _h.escape(str(e))
        poss = self._alpaca_get("/v2/positions") or []
        al_entries = [(p["symbol"], float(p["qty"])) for p in poss]
        acct = self._alpaca_get("/v2/account") or {}
        al_nav = float(acct.get("equity", 0) or 0)
        bars = self._alpaca_daily_bars([sym for sym, _ in ib_entries + al_entries])

        if ib_err:
            out.append(f"<b>🟢 IBKR</b> — ⚠️ {ib_err}")
        elif ib_entries:
            day, tbl = table(ib_entries, bars, ib_nav)
            out.append(f"<b>🟢 IBKR · ${ib_nav:,.0f} · {day}</b>\n{tbl}")
        if al_entries:
            day, tbl = table(al_entries, bars, al_nav)
            out.append(f"\n<b>🔵 ALPACA · ${al_nav:,.0f} · {day}</b>\n{tbl}")
        elif not ib_entries and not ib_err:
            out.append("no positions in either book")
        out.append(f"<i>{'live prices' if mkt_open else 'session close vs prev close'}</i>")
        return "\n".join(out)

    async def _cmd_status(self):
        socket_ok, data_ok = await self._real_connectivity()
        ib_state = ("✅ connected" if (socket_ok and data_ok)
                    else "🔴 socket up but DATA FEED DOWN — try /reconnect" if socket_ok
                    else "❌ disconnected")
        try:
            h = requests.get(SIGNAL_HEALTH_URL, timeout=5).json()
            sig = "✅ fresh" if not h.get("is_stale") else "⚠️ STALE"
        except Exception:
            sig = "❌ unreachable"
        vs, rv = (self.compute_vol_scale() if VOL_SCALING else (1.0, None))
        left = max(0, REBAL_DAYS - self._trading_days_since_rebal)
        return (self._tg_header("🩺 STATUS", self.is_market_open()) + "\n" +
                f"IBKR — {ib_state} <i>({self.account_id})</i>\n"
                f"Signals — {sig}\n"
                f"Rebalance — day {self._trading_days_since_rebal}/{REBAL_DAYS}, next in {left}d\n"
                f"Vol-scale — {vs:.2f}" + (f" <i>(realized vol {rv:.0%})</i>" if rv else " <i>(ramp-up)</i>"))

    def _cmd_signals(self):
        try:
            data = requests.get(SIGNAL_URL, timeout=10).json()
            sigs = data.get("signals", []) if isinstance(data, dict) else data
            buys = sorted([s for s in sigs if s.get("signal") == "BUY"],
                          key=lambda s: -s.get("probability", 0))
            rows = [f"{'#':>2} │ {'SYM':<6}│{'CONV':>5}", "─" * 3 + "┼" + "─" * 7 + "┼" + "─" * 5]
            rows += [f"{i:>2} │ {s['symbol']:<6}│{s.get('probability', 0):>5.2f}"
                     for i, s in enumerate(buys[:15], 1)]
            return (self._tg_header("🎯 TOP SIGNALS") +
                    f"{len(buys)} BUY signals · top {min(15, len(buys))}\n\n"
                    + self._tg_table(rows))
        except Exception as e:
            import html as _h
            return f"⚠️ signals error: {_h.escape(str(e))}"

    def _cmd_rebal(self):
        left = max(0, REBAL_DAYS - self._trading_days_since_rebal)
        done = min(self._trading_days_since_rebal, REBAL_DAYS)
        bar = "▰" * done + "▱" * (REBAL_DAYS - done)
        return (self._tg_header("🔄 REBALANCE") + "\n" +
                self._tg_table([bar]) + "\n"
                f"Day <b>{self._trading_days_since_rebal}</b> of {REBAL_DAYS} — next in <b>{left}</b> trading days\n"
                f"<i>last: {self._last_rebal_date} · both books rebalance together</i>")

    async def _cmd_data(self):
        try:
            from data_freshness_check import check_all, format_report
            checks, _ = await asyncio.to_thread(check_all)
            return format_report(checks)
        except Exception as e:
            return f"data check error: {e}"

    async def _real_connectivity(self):
        """True data-path test: isConnected() can return True on an Error-1100
        zombie (socket up, IBKR uplink dead). reqCurrentTime actually round-trips
        to IBKR's servers, so a timeout means the data feed is really down."""
        socket_ok = self.ib.isConnected()
        data_ok = False
        if socket_ok:
            for attempt in range(2):  # retry once — a single transient timeout shouldn't read as "down"
                try:
                    await asyncio.wait_for(self.ib.reqCurrentTimeAsync(), timeout=6)
                    data_ok = True
                    break
                except Exception:
                    await asyncio.sleep(0.5)
        return socket_ok, data_ok

    async def _cmd_connection(self):
        socket_ok, data_ok = await self._real_connectivity()
        hb = time.time() - getattr(self, "_last_progress", time.time())
        if socket_ok and data_ok:
            return (f"🟢 CONNECTION HEALTHY\nSocket: connected\nData feed: LIVE (round-trip OK)\n"
                    f"Loop heartbeat: {hb:.0f}s ago\nAccount: {self.account_id}")
        if socket_ok and not data_ok:
            return ("🔴 DATA FEED DOWN (Error-1100 state)\nSocket: 'connected' but NOT responding —\n"
                    "the Gateway lost its uplink to IBKR (usually: you're logged in elsewhere).\n"
                    "→ Log out of IBKR on phone/web, then send /reconnect. If it persists, the\n"
                    "Gateway itself needs a restart (which needs your 2FA).")
        return ("🔴 DISCONNECTED\nSocket: down. The engine auto-reconnects; send /reconnect to force it.")

    def _gateway_port_open(self):
        """True if the IB Gateway API port accepts a TCP connection (Gateway logged
        in). If False, the Gateway itself is down — restarting the engine won't help."""
        import socket
        try:
            with socket.create_connection((IB_HOST, IB_PORT), timeout=5):
                return True
        except Exception:
            return False

    def _restart_gateway(self):
        """Restart the IB Gateway docker container -> re-triggers IBKR login/2FA.
        Engine runs as 'ubuntu' (in the docker group). Returns (ok, message)."""
        import subprocess
        try:
            r = subprocess.run(["docker", "restart", IB_GATEWAY_CONTAINER],
                               capture_output=True, text=True, timeout=90)
            if r.returncode == 0:
                return True, (r.stdout or "").strip()
            return False, ((r.stderr or r.stdout) or "").strip()[:300]
        except Exception as e:
            return False, str(e)[:300]

    async def _connect_with_retry(self):
        """Startup connect that RETRIES (with the same 2FA escalation as a mid-session
        disconnect) instead of fatal-crashing into a ~1/sec PM2 restart loop when the
        Gateway is down. Stays alive so /reconnect stays reachable."""
        attempt = 0
        while True:
            try:
                await self.connect()
                self._on_reconnect()   # clears outage state; confirms recovery if escalated
                return
            except Exception as e:
                attempt += 1
                self._last_progress = time.time()   # keep the watchdog calm
                self._on_disconnect_detected()      # silent <15min; one 2FA alert after
                log.error(f"Startup connect failed (attempt {attempt}): {e} — retrying in 30s")
                await asyncio.sleep(30)

    async def _ibkr_alive(self):
        """Actively verify IBKR is REALLY working: Gateway port open AND a live server
        request returns. A dead login (you logged in elsewhere) keeps the port open but
        times out on data — only a Gateway relogin fixes that. Uses a throwaway read-only
        connection (separate clientId) so it never disturbs the engine's own connection."""
        if not self._gateway_port_open():
            return False
        probe = IB()
        try:
            await asyncio.wait_for(
                probe.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID + 90,
                                   readonly=True, timeout=8), timeout=10)
            await asyncio.wait_for(probe.reqCurrentTimeAsync(), timeout=6)  # only returns if the session is live
            return True
        except Exception:
            return False
        finally:
            try:
                probe.disconnect()
            except Exception:
                pass

    async def _cmd_reconnect(self):
        # "Reconnect any issue": probe whether IBKR is REALLY working (port open AND data
        # flowing), not just whether the port is open. A dead login (logged in elsewhere)
        # leaves the port open but times out on data — the only fix is a Gateway relogin.
        # So restart the Gateway whenever IBKR isn't truly alive (port closed OR session
        # dead); only do a plain engine restart when IBKR genuinely is alive.
        if not await self._ibkr_alive():
            ok, msg = await asyncio.to_thread(self._restart_gateway)
            if ok:
                return ("🔄 IBKR was down (Gateway closed or login dead) — restarting the\n"
                        "Gateway now. 📲 Approve the IBKR Mobile (IB Key) 2FA on your phone if\n"
                        "it asks. The engine auto-connects once it's back — /connection in ~90s.")
            return (f"⚠️ Gateway restart FAILED:\n{msg}\n"
                    f"Restart it manually on the server: docker restart {IB_GATEWAY_CONTAINER}")
        # IBKR is genuinely healthy -> engine-side stale socket -> clean engine restart.
        threading.Timer(2.0, lambda: os._exit(1)).start()
        return ("🔄 IBKR is healthy — restarting just the engine for a clean reconnect —\n"
                "give it ~45s, then send /connection to confirm.")

    async def _telegram_poll_loop(self):
        """Long-poll Telegram for /commands and reply. Read-only; never crashes the engine.
        Uses asyncio.to_thread so blocking HTTP never stalls the trading loop."""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
            return
        base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        offset = None
        try:  # skip backlog so a restart doesn't replay old commands
            r = await asyncio.to_thread(requests.get, f"{base}/getUpdates", params={"timeout": 0}, timeout=10)
            ups = r.json().get("result", [])
            if ups:
                offset = ups[-1]["update_id"] + 1
        except Exception:
            pass
        # seed the allowlist file with the primary chat so it exists + is easy to extend
        af = Path(__file__).resolve().parent / "data" / "telegram_allowed.json"
        if not af.exists() and TELEGRAM_CHAT:
            try:
                import json
                json.dump([str(TELEGRAM_CHAT)], open(af, "w"))
            except Exception:
                pass
        log.info("Telegram command bot started (read-only, multi-user allowlist)")
        while self.running:
            try:
                params = {"timeout": 25}
                if offset is not None:
                    params["offset"] = offset
                r = await asyncio.to_thread(requests.get, f"{base}/getUpdates", params=params, timeout=35)
                allowed = self._tg_allowed_chats()
                for up in r.json().get("result", []):
                    offset = up["update_id"] + 1
                    msg = up.get("message") or up.get("edited_message") or {}
                    chat_id = str((msg.get("chat") or {}).get("id", ""))
                    text = (msg.get("text") or "").strip()
                    if not text.startswith("/"):
                        continue
                    if chat_id not in allowed:
                        # tell them their ID so an admin can authorize them, and log it
                        log.info(f"Telegram: unauthorized chat {chat_id} sent '{text}'")
                        await asyncio.to_thread(
                            self._tg_send_raw,
                            f"🔒 Not authorized to use this bot.\nYour chat ID is: {chat_id}\nAsk the owner to add it.",
                            chat_id)
                        continue
                    try:
                        reply = await self._handle_command(text, chat_id)
                    except Exception as e:
                        reply = f"⚠️ command error: {e}"
                    await asyncio.to_thread(self._tg_send_raw, reply, chat_id)
            except Exception as e:
                log.warning(f"Telegram poll error: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Main loop."""
        # Init outage tracking + start the Telegram bot BEFORE connecting, so /reconnect
        # stays reachable even if the Gateway is down at startup (when you most need it).
        self._disconnect_since = None   # outage tracking for escalating alerts
        self._escalated = False
        self._last_progress = time.time()
        self.running = True             # set BEFORE telegram task: its loop is `while self.running`
        asyncio.create_task(self._telegram_poll_loop())
        # Robust startup connect: retry with backoff + 2FA escalation instead of fatal
        # crash-looping (the ~1/sec PM2 restart loop that spams alerts) when Gateway down.
        await self._connect_with_retry()

        log.info("=" * 60)
        log.info("  IBKR Trading Engine v12 Started")
        log.info(f"  Account: {self.account_id}")
        log.info(f"  Port: {IB_PORT} ({'LIVE' if IB_PORT == 4001 else 'PAPER'})")
        log.info(f"  Max positions: {MAX_POSITIONS}")
        log.info(f"  Position cap: {POSITION_CAP:.0%}")
        log.info(f"  Trailing stop: {TRAILING_STOP:.0%}")
        log.info(f"  Leverage: {LEVERAGE:.1f}x")
        log.info(f"  Rebalance: every 20 trading days")
        log.info(f"  Rebalance check interval: {REBALANCE_INTERVAL}s")
        log.info(f"  Short sleeve: {'ENABLED' if SHORT_ENABLED else 'DISABLED'}")
        if SHORT_ENABLED:
            log.info(f"  Short allocation: {SHORT_ALLOCATION:.0%}")
            log.info(f"  Short max positions: {SHORT_MAX_POSITIONS}")
            log.info(f"  Short hold days: {SHORT_HOLD_DAYS}")
        log.info("=" * 60)

        summary = await self.get_account_summary()
        log.info(f"NAV: ${summary.get('NetLiquidation', 0):,.0f}")
        log.info(f"Cash: ${summary.get('TotalCashValue', 0):,.0f}")
        self.record_nav(summary.get("NetLiquidation", 0))  # seed NAV history for vol-scaling
        # If the market is already closed and today's close snapshot isn't captured yet,
        # seed it now so /portfolio and /pnl show it immediately (refreshed exactly at 4pm).
        try:
            if not self.is_market_open():
                snap = self._load_close_snapshot()
                if not snap or snap.get("date") != datetime.now().date().isoformat():
                    await self._record_close_snapshot()
        except Exception:
            pass
        if VOL_SCALING:
            vs, rv = self.compute_vol_scale()
            log.info(f"Vol-scaling: {'ON' if VOL_SCALING else 'off'} target={VOL_TARGET:.0%} "
                     f"lookback={VOL_LOOKBACK}d | current scale={vs:.2f}"
                     + (f" (realized vol {rv:.0%})" if rv else " (ramp-up: <20 NAV days)"))

        self.running = True
        cycle = 0
        self._start_watchdog()          # force-restart if the loop ever hangs (Error 1100 etc.)
        # (outage tracking + Telegram bot already started at the top of run())

        # One-time self-test: if the sentinel file exists, run every command once
        # (logs + sends results), then delete the sentinel so it never repeats.
        selftest = Path(__file__).resolve().parent / "data" / ".tg_selftest"
        if selftest.exists():
            try:
                selftest.unlink()
            except Exception:
                pass
            for c in ["/status", "/connection", "/portfolio", "/pnl", "/daily", "/positions", "/alpaca", "/signals", "/rebal", "/data", "/help"]:
                try:
                    # render as the OWNER chat so /help includes the owner-only section
                    out = await self._handle_command(c, TELEGRAM_CHAT)
                    log.info(f"SELFTEST {c} ->\n{out}")
                    await asyncio.to_thread(self._tg_send_raw, out)
                except Exception as e:
                    log.error(f"SELFTEST {c} FAILED: {e}")

        while self.running:
            try:
                cycle += 1
                self._last_progress = time.time()   # watchdog heartbeat — proves the loop is alive

                # Connection health: check the socket every cycle AND the real DATA FEED
                # about once a minute. A dead login (you logged in elsewhere) keeps the
                # socket up but the feed dead, so socket-only checks miss it — meaning no
                # alert fired when the market was closed. _real_connectivity round-trips to
                # IBKR (retries once so a transient blip won't false-fire).
                if (cycle % 6 == 1) or not self.ib.isConnected():
                    socket_ok, data_ok = await self._real_connectivity()
                else:
                    socket_ok, data_ok = True, True
                if not (socket_ok and data_ok):
                    log.warning(f"IB unhealthy (socket={socket_ok} data={data_ok}) — reconnecting...")
                    self._on_disconnect_detected()  # silent for brief blips; escalates (2FA alert) if >15 min
                    try:
                        await self.connect()
                        self.ib.reqGlobalCancel()
                        await asyncio.sleep(2)
                        log.info("Reconnected successfully — cancelled pending orders")
                        self._on_reconnect()
                    except Exception as ce:
                        log.error(f"Reconnect failed: {ce}")
                        await asyncio.sleep(60)
                        continue

                if not self.is_market_open():
                    if cycle % 60 == 1:  # log once per ~10 min
                        log.info("Market closed — waiting...")
                    # Send daily end-of-day summary at 4:05 PM ET
                    if not hasattr(self, '_eod_sent_today'):
                        self._eod_sent_today = None
                    from zoneinfo import ZoneInfo
                    now_et = datetime.now(ZoneInfo("US/Eastern"))
                    today_str = now_et.strftime("%Y-%m-%d")
                    if now_et.hour == 16 and now_et.minute >= 5 and self._eod_sent_today != today_str:
                        self._eod_sent_today = today_str
                        try:
                            summary = await self.get_account_summary()
                            nav = summary.get("NetLiquidation", 0)
                            hist = self._load_nav_history()
                            prev = [p for p in hist if p[0] != today_str]      # yesterday's close BEFORE recording today
                            self.record_nav(nav, at_close=True)  # authoritative 4pm close mark
                            await self._record_close_snapshot()  # close mark for /portfolio + /pnl
                            n_pos = len(self.positions)
                            msg = f"📊 Daily Close ({today_str})\nNAV ${nav:,.0f}"
                            if prev and prev[-1][1]:
                                d = nav - prev[-1][1]
                                msg += f" | {'🟩' if d >= 0 else '🟥'} Today ${d:+,.0f} ({d/prev[-1][1]*100:+.2f}%)"
                            msg += f" | {n_pos} pos"
                            try:  # top movers of the day (by $ impact on the book)
                                s = await self._ibkr_snapshot()
                                bars = self._alpaca_daily_bars([it.contract.symbol for it in s["items"]])
                                mv = []
                                for it in s["items"]:
                                    b = bars.get(it.contract.symbol) or {}
                                    if b.get("close") and b.get("prev"):
                                        mv.append((it.contract.symbol, (b["close"] - b["prev"]) * it.position,
                                                   (b["close"] / b["prev"] - 1) * 100))
                                if mv:
                                    mv.sort(key=lambda x: x[1])
                                    lo, hi = mv[0], mv[-1]
                                    msg += (f"\n🏆 {hi[0]} ${hi[1]:+,.0f} ({hi[2]:+.1f}%)"
                                            f"\n💥 {lo[0]} ${lo[1]:+,.0f} ({lo[2]:+.1f}%)")
                            except Exception:
                                pass
                            msg += "\nSend /daily for the full per-position breakdown."
                            send_telegram(msg)
                        except Exception:
                            pass
                    await asyncio.sleep(10)
                    continue

                # One-time daily "market OPEN" announcement (parallel to the Alpaca engine).
                from zoneinfo import ZoneInfo
                _td = datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d")
                if getattr(self, "_open_announced", None) != _td:
                    self._open_announced = _td
                    try:
                        await self.update_positions()
                        nav = (await self.get_account_summary()).get("NetLiquidation", 0)
                        left = max(0, REBAL_DAYS - self._trading_days_since_rebal)
                        send_telegram(
                            f"🟢 IBKR market OPEN ({_td}) — engine live, holding "
                            f"{len(self.positions)} positions, NAV ${nav:,.0f}.\n"
                            f"Monitoring trailing stops; next rebalance in {left} trading days.")
                    except Exception:
                        send_telegram(f"🟢 IBKR market OPEN ({_td}) — engine live.")

                # Check trailing stops every cycle (long positions)
                await self.check_trailing_stops()

                # Check short exits every cycle
                if SHORT_ENABLED:
                    await self.check_short_exits()

                # Rebalance periodically
                should_rebalance = (
                    self.last_rebalance is None or
                    (datetime.now() - self.last_rebalance).total_seconds() >= REBALANCE_INTERVAL
                )

                if should_rebalance:
                    # Stamp now so we re-evaluate at most every REBALANCE_INTERVAL — rebalance()
                    # only updates last_rebalance on a REAL rebalance, so without this the check
                    # fires every cycle. The 20-trading-day gate inside still controls actual trades.
                    self.last_rebalance = datetime.now()
                    log.info(f"--- Cycle {cycle}: Rebalance check ---")
                    await self.rebalance()

                    # Process short events
                    if SHORT_ENABLED:
                        await self.process_short_events()

                    # Log status
                    summary = await self.get_account_summary()
                    nav = summary.get("NetLiquidation", 0)
                    cash = summary.get("TotalCashValue", 0)
                    log.info(f"NAV: ${nav:,.0f} | Cash: ${cash:,.0f} | "
                             f"Long: {len(self.positions)} | Short: {len(self.short_positions)}")

                await asyncio.sleep(10)

            except KeyboardInterrupt:
                log.info("Shutting down...")
                self.running = False
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                # Auto-reconnect if connection dropped
                if not self.ib.isConnected():
                    log.warning("IB connection lost — reconnecting in 30s...")
                    self._on_disconnect_detected()  # silent for brief blips; escalates if >15 min
                    await asyncio.sleep(30)
                    try:
                        await self.connect()
                        # Cancel any pending orders from before disconnect
                        self.ib.reqGlobalCancel()
                        await asyncio.sleep(2)
                        log.info("Reconnected successfully — cancelled pending orders")
                        self._on_reconnect()
                    except Exception as ce:
                        log.error(f"Reconnect failed: {ce}")
                else:
                    send_telegram(f"⚠️ IBKR Engine error: {e}")
                    await asyncio.sleep(30)

        self.ib.disconnect()
        log.info("IBKR Engine stopped")


if __name__ == "__main__":
    engine = IBKREngine()
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(engine.run())
    except KeyboardInterrupt:
        log.info("Stopped by user")
    except Exception as e:
        log.error(f"Fatal error: {e}")
        send_telegram(f"🔴 IBKR Engine CRASHED: {e}")
