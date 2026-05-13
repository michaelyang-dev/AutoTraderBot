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
    - 25% trailing stop per position
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

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
IB_HOST = "127.0.0.1"
IB_PORT = 4002          # paper trading
IB_CLIENT_ID = 1
SIGNAL_URL = "http://localhost:5001/signals"
SIGNAL_HEALTH_URL = "http://localhost:5001/health"

# Strategy parameters (must match backtest)
MAX_POSITIONS = 8
POSITION_CAP = 0.25     # v11: 25% max per position (matches backtest cap=0.25)
TRAILING_STOP = 0.35    # v11: 35% trailing stop
LEVERAGE = 1.50         # v11: 1.5x leverage via IBKR margin
REBALANCE_INTERVAL = 600  # check every 10 minutes
MIN_TRADE_PCT = 0.02    # don't trade if delta < 2% of portfolio

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
        requests.post(url, json={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=10)
    except:
        pass


class IBKREngine:
    def __init__(self):
        self.ib = IB()
        self.positions = {}       # symbol -> {qty, avg_cost, market_value, peak_price}
        self.trailing_peaks = {}  # symbol -> peak price since entry
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
        # Request delayed market data (type 3) — no subscription needed for paper
        self.ib.reqMarketDataType(3)
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
        """Fetch BUY signals from signal server."""
        try:
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
        except:
            pass
        # Fallback: check time
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("US/Eastern"))
        if now.weekday() >= 5:
            return False
        return now.hour >= 9 and (now.hour < 16 or (now.hour == 9 and now.minute >= 30))

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

        log.info(f"SELL {qty} {symbol}: {status} filled={filled} @ ${price:.2f} ({reason})")
        send_telegram(f"📉 SELL {qty} {symbol} @ ${price:.2f} | {reason}")

        # Clean up tracking
        if symbol in self.trailing_peaks:
            del self.trailing_peaks[symbol]

        return trade

    async def buy_position(self, symbol, qty, reason="signal"):
        """Buy a position."""
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        order = MarketOrder("BUY", qty)
        order.tif = "DAY"
        trade = self.ib.placeOrder(contract, order)
        await asyncio.sleep(3)

        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice

        log.info(f"BUY {qty} {symbol}: {status} filled={filled} @ ${price:.2f} ({reason})")
        send_telegram(f"📈 BUY {qty} {symbol} @ ${price:.2f} | {reason}")

        return trade

    async def rebalance(self):
        """Rebalance portfolio to match signal server picks."""
        signals = self.fetch_signals()
        if not signals:
            log.warning("No signals available — skipping rebalance")
            return

        # SAFETY: Close any accidental short positions first
        await self.update_positions()
        for sym, pos in list(self.positions.items()):
            if pos["qty"] < 0:
                log.warning(f"EMERGENCY COVER: {sym} has short position ({pos['qty']} shares)")
                # Use SMART routing (not exchange-specific) to avoid precautionary rejections
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

        # 1. SELL positions not in signals (enforce 20-day min hold)
        MIN_HOLD_DAYS = 20
        for sym in held_symbols - target_symbols:
            pos = self.positions[sym]
            # Check min hold period before rebalance exit
            entry_date = self.trailing_peaks.get(f"{sym}_entry")
            if entry_date:
                days_held = (datetime.now() - entry_date).days
                if days_held < MIN_HOLD_DAYS:
                    log.info(f"HOLD {sym}: dropped from top-8 but min-hold active ({days_held}d / {MIN_HOLD_DAYS}d)")
                    continue
            log.info(f"Selling {sym} — no longer in top-{MAX_POSITIONS}")
            await self.sell_position(sym, pos["qty"], "dropped_from_signals")

        # Wait for sells to settle
        if held_symbols - target_symbols:
            await asyncio.sleep(2)
            await self.update_positions()
            portfolio_value = await self.get_portfolio_value()

        # 2. Compute target weights (with 1.5x leverage)
        target_weight = LEVERAGE / MAX_POSITIONS  # 1.5x / 8 = 18.75% per position
        target_value_per_position = portfolio_value * target_weight

        # Cap at POSITION_CAP
        if target_weight > POSITION_CAP:
            target_value_per_position = portfolio_value * POSITION_CAP

        # 3. BUY new positions / adjust existing
        for sig in signals:
            sym = sig["symbol"]
            contract = Stock(sym, "SMART", "USD")

            price = await self.get_market_price(contract)
            if not price or price <= 0:
                log.warning(f"Cannot get price for {sym} — skipping")
                continue

            current_qty = self.positions.get(sym, {}).get("qty", 0)
            current_value = current_qty * price
            target_qty = int(target_value_per_position / price)
            delta_qty = target_qty - current_qty

            # Skip if delta is too small
            delta_value = abs(delta_qty * price)
            if delta_value < portfolio_value * MIN_TRADE_PCT:
                continue

            if delta_qty > 0:
                # BUY
                await self.buy_position(sym, delta_qty, "rebalance")
                # Set trailing peak and entry date
                self.trailing_peaks[sym] = price
                if f"{sym}_entry" not in self.trailing_peaks:
                    self.trailing_peaks[f"{sym}_entry"] = datetime.now()
            elif delta_qty < 0:
                # TRIM (sell excess)
                await self.sell_position(sym, abs(delta_qty), "trim_overweight")

        self.last_rebalance = datetime.now()
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
            except:
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

    async def run(self):
        """Main loop."""
        await self.connect()

        log.info("=" * 60)
        log.info("  IBKR Trading Engine Started")
        log.info(f"  Account: {self.account_id}")
        log.info(f"  Max positions: {MAX_POSITIONS}")
        log.info(f"  Position cap: {POSITION_CAP:.0%}")
        log.info(f"  Trailing stop: {TRAILING_STOP:.0%}")
        log.info(f"  Rebalance interval: {REBALANCE_INTERVAL}s")
        log.info(f"  Short sleeve: {'ENABLED' if SHORT_ENABLED else 'DISABLED'}")
        if SHORT_ENABLED:
            log.info(f"  Short allocation: {SHORT_ALLOCATION:.0%}")
            log.info(f"  Short max positions: {SHORT_MAX_POSITIONS}")
            log.info(f"  Short hold days: {SHORT_HOLD_DAYS}")
        log.info("=" * 60)

        summary = await self.get_account_summary()
        log.info(f"NAV: ${summary.get('NetLiquidation', 0):,.0f}")
        log.info(f"Cash: ${summary.get('TotalCashValue', 0):,.0f}")

        self.running = True
        cycle = 0

        while self.running:
            try:
                cycle += 1

                if not self.is_market_open():
                    if cycle % 60 == 1:  # log once per ~10 min
                        log.info("Market closed — waiting...")
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
                    (datetime.now() - self.last_rebalance).seconds >= REBALANCE_INTERVAL
                )

                if should_rebalance:
                    log.info(f"--- Cycle {cycle}: Rebalancing ---")
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
