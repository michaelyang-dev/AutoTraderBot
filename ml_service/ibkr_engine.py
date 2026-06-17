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

    def record_nav(self, nav):
        """Append today's NAV once per calendar day; keep last 70 days."""
        try:
            import json
            today = datetime.now().date().isoformat()
            hist = self._load_nav_history()
            if hist and hist[-1][0] == today:
                hist[-1] = [today, nav]      # update today's value
            else:
                hist.append([today, nav])
            hist = hist[-70:]
            with open(NAV_HISTORY_FILE, "w") as f:
                json.dump(hist, f)
        except Exception:
            pass

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
        navs = [h[1] for h in self._load_nav_history()][-(VOL_LOOKBACK + 1):]
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
        """Send a plain reply to a Telegram chat (defaults to the primary chat)."""
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          json={"chat_id": chat_id or TELEGRAM_CHAT, "text": text}, timeout=10)
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

    async def _handle_command(self, text):
        cmd = text.split()[0].lower().lstrip("/").split("@")[0]
        if cmd in ("start", "help"):
            return ("🤖 AutoTrader bot — read-only commands:\n"
                    "/portfolio — NAV, cash, leverage (both accounts)\n"
                    "/positions — IBKR holdings + P&L\n"
                    "/alpaca — Alpaca paper holdings + P&L\n"
                    "/pnl — gains (unrealized + since funding)\n"
                    "/status — system health\n"
                    "/connection — real connection test (is the data feed live?)\n"
                    "/reconnect — force a clean engine reconnect\n"
                    "/signals — current top picks\n"
                    "/rebal — rebalance schedule\n"
                    "/data — data freshness check")
        handlers = {"portfolio": self._cmd_portfolio, "positions": self._cmd_positions,
                    "alpaca": self._cmd_alpaca, "pnl": self._cmd_pnl, "status": self._cmd_status,
                    "signals": self._cmd_signals, "rebal": self._cmd_rebal, "data": self._cmd_data,
                    "connection": self._cmd_connection, "reconnect": self._cmd_reconnect}
        h = handlers.get(cmd)
        if not h:
            return f"Unknown command: /{cmd}. Try /help"
        out = h()
        return await out if asyncio.iscoroutine(out) else out

    async def _cmd_portfolio(self):
        out = ["📊 PORTFOLIO\n"]
        try:
            s = await self._ibkr_snapshot()
            out.append(f"🟢 IBKR LIVE\nNAV ${s['nav']:,.0f} | Cash ${s['cash']:,.0f}\n"
                       f"{len(s['items'])} positions | {s['lev']:.2f}x leverage")
        except Exception as e:
            out.append(f"🟢 IBKR LIVE — error: {e}")
        acct = self._alpaca_get("/v2/account"); poss = self._alpaca_get("/v2/positions")
        if acct:
            eq = float(acct.get("equity", 0)); cash = float(acct.get("cash", 0))
            gross = sum(abs(float(p["market_value"])) for p in poss) if poss else 0
            out.append(f"\n🔵 ALPACA PAPER\nNAV ${eq:,.0f} | Cash ${cash:,.0f}\n"
                       f"{len(poss) if poss else 0} positions | {(gross/eq if eq else 0):.2f}x leverage")
        else:
            out.append("\n🔵 ALPACA PAPER — unavailable")
        return "\n".join(out)

    async def _cmd_positions(self):
        s = await self._ibkr_snapshot()
        items = sorted(s["items"], key=lambda it: -it.marketValue)
        lines = [f"📈 IBKR POSITIONS ({len(items)}) — unrealized ${s['upl']:+,.0f}\n"]
        for it in items:
            cost = it.marketValue - it.unrealizedPNL
            pct = (it.unrealizedPNL / cost * 100) if cost else 0
            lines.append(f"{it.contract.symbol:5} ${it.marketValue:,.0f}  {it.unrealizedPNL:+,.0f} ({pct:+.0f}%)")
        return "\n".join(lines)

    def _cmd_alpaca(self):
        poss = self._alpaca_get("/v2/positions")
        if poss is None:
            return "🔵 ALPACA — unavailable"
        poss = sorted(poss, key=lambda p: -float(p["market_value"]))
        upl = sum(float(p["unrealized_pl"]) for p in poss)
        lines = [f"🔵 ALPACA POSITIONS ({len(poss)}) — unrealized ${upl:+,.0f}\n"]
        for p in poss[:30]:
            lines.append(f"{p['symbol']:5} ${float(p['market_value']):,.0f}  "
                         f"{float(p['unrealized_pl']):+,.0f} ({float(p['unrealized_plpc'])*100:+.0f}%)")
        return "\n".join(lines)

    async def _cmd_pnl(self):
        out = ["💰 P&L\n"]
        try:
            s = await self._ibkr_snapshot()
            nav = s["nav"]
            line = f"🟢 IBKR LIVE\nNAV ${nav:,.0f}\n"
            # daily change from persisted NAV history (most recent prior trading day)
            hist = self._load_nav_history()
            today = datetime.now().date().isoformat()
            prev = [p for p in hist if p[0] != today]
            if prev and prev[-1][1]:
                d = nav - prev[-1][1]
                line += f"Today ${d:+,.0f} ({d/prev[-1][1]*100:+.2f}%)\n"
            else:
                line += "Today: n/a (building NAV history)\n"
            line += f"Unrealized ${s['upl']:+,.0f}\n"
            since = nav - IBKR_INITIAL_CAPITAL
            line += f"Since ${IBKR_INITIAL_CAPITAL/1000:.0f}K funding: ${since:+,.0f} ({since/IBKR_INITIAL_CAPITAL*100:+.1f}%)"
            out.append(line)
        except Exception as e:
            out.append(f"🟢 IBKR — error: {e}")
        acct = self._alpaca_get("/v2/account"); poss = self._alpaca_get("/v2/positions")
        if acct:
            eq = float(acct.get("equity", 0)); le = float(acct.get("last_equity", 0))
            line = f"\n🔵 ALPACA PAPER\nNAV ${eq:,.0f}\n"
            if le:
                d = eq - le
                line += f"Today ${d:+,.0f} ({d/le*100:+.2f}%)\n"
            if poss:
                line += f"Unrealized ${sum(float(p['unrealized_pl']) for p in poss):+,.0f}"
            out.append(line.rstrip())
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
        return (f"🩺 STATUS\n"
                f"IBKR: {ib_state} ({self.account_id})\n"
                f"Signal server: {sig}\n"
                f"Rebalance: {self._trading_days_since_rebal}/{REBAL_DAYS} days since (last {self._last_rebal_date})\n"
                f"Vol-scale: {vs:.2f}" + (f" (realized vol {rv:.0%})" if rv else " (ramp-up)"))

    def _cmd_signals(self):
        try:
            data = requests.get(SIGNAL_URL, timeout=10).json()
            sigs = data.get("signals", []) if isinstance(data, dict) else data
            buys = sorted([s for s in sigs if s.get("signal") == "BUY"],
                          key=lambda s: -s.get("probability", 0))
            lines = [f"🎯 TOP SIGNALS ({len(buys)} BUY)\n"]
            for i, s in enumerate(buys[:15], 1):
                lines.append(f"{i:2}. {s['symbol']:5} ({s.get('probability', 0):.2f})")
            return "\n".join(lines)
        except Exception as e:
            return f"signals error: {e}"

    def _cmd_rebal(self):
        left = max(0, REBAL_DAYS - self._trading_days_since_rebal)
        return (f"🔄 REBALANCE\nLast: {self._last_rebal_date}\n"
                f"Days since: {self._trading_days_since_rebal}/{REBAL_DAYS}\n"
                f"Next: in {left} trading days")

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

    async def _cmd_reconnect(self):
        # send the reply, then exit ~2s later so PM2 restarts us with a clean reconnect
        threading.Timer(2.0, lambda: os._exit(1)).start()
        return ("🔄 Restarting the engine for a clean reconnect — give it ~45s, then send\n"
                "/connection to confirm. (Note: if the GATEWAY's uplink is broken, a restart\n"
                "won't fix it — you'll need to log out elsewhere + the Gateway restarted.)")

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
                        reply = await self._handle_command(text)
                    except Exception as e:
                        reply = f"⚠️ command error: {e}"
                    await asyncio.to_thread(self._tg_send_raw, reply, chat_id)
            except Exception as e:
                log.warning(f"Telegram poll error: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Main loop."""
        await self.connect()

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
        if VOL_SCALING:
            vs, rv = self.compute_vol_scale()
            log.info(f"Vol-scaling: {'ON' if VOL_SCALING else 'off'} target={VOL_TARGET:.0%} "
                     f"lookback={VOL_LOOKBACK}d | current scale={vs:.2f}"
                     + (f" (realized vol {rv:.0%})" if rv else " (ramp-up: <20 NAV days)"))

        self.running = True
        cycle = 0
        self._disconnect_since = None   # outage tracking for escalating alerts
        self._escalated = False
        self._start_watchdog()          # force-restart if the loop ever hangs (Error 1100 etc.)

        # Start the read-only Telegram command bot (concurrent task)
        asyncio.create_task(self._telegram_poll_loop())

        # One-time self-test: if the sentinel file exists, run every command once
        # (logs + sends results), then delete the sentinel so it never repeats.
        selftest = Path(__file__).resolve().parent / "data" / ".tg_selftest"
        if selftest.exists():
            try:
                selftest.unlink()
            except Exception:
                pass
            for c in ["/status", "/connection", "/portfolio", "/pnl", "/positions", "/alpaca", "/signals", "/rebal", "/data"]:
                try:
                    out = await self._handle_command(c)
                    log.info(f"SELFTEST {c} ->\n{out}")
                    await asyncio.to_thread(self._tg_send_raw, out)
                except Exception as e:
                    log.error(f"SELFTEST {c} FAILED: {e}")

        while self.running:
            try:
                cycle += 1
                self._last_progress = time.time()   # watchdog heartbeat — proves the loop is alive

                # Check IB connection health every cycle — reconnect if dropped
                if not self.ib.isConnected():
                    log.warning("IB connection lost in main loop — reconnecting...")
                    self._on_disconnect_detected()  # silent for brief blips; escalates if >15 min
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
                            self.record_nav(nav)  # build NAV history for vol-scaling
                            n_pos = len(self.positions)
                            signals = self.fetch_signals()
                            n_signals = len(signals) if signals else 0
                            send_telegram(
                                f"📊 IBKR Daily Summary ({today_str})\n"
                                f"NAV: ${nav:,.0f} | Positions: {n_pos}/{n_signals} signals\n"
                                f"Port: {IB_PORT} ({'LIVE' if IB_PORT == 4001 else 'PAPER'})"
                            )
                        except Exception:
                            pass
                    await asyncio.sleep(10)
                    continue

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
