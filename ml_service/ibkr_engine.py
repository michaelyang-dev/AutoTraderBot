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
POSITION_CAP = 0.15     # 15% max per position
TRAILING_STOP = 0.25    # 25% trailing stop
REBALANCE_INTERVAL = 600  # check every 10 minutes
MIN_TRADE_PCT = 0.02    # don't trade if delta < 2% of portfolio

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")


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

    async def connect(self):
        """Connect to IB Gateway."""
        log.info(f"Connecting to IB Gateway at {IB_HOST}:{IB_PORT}...")
        self.ib = IB()
        await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, readonly=False, timeout=20)
        accounts = self.ib.managedAccounts()
        self.account_id = accounts[0] if accounts else None
        log.info(f"Connected. Account: {self.account_id}")
        send_telegram(f"🟢 IBKR Engine connected. Account: {self.account_id}")

    def get_account_summary(self):
        """Get account NAV and buying power."""
        summary = {}
        for item in self.ib.accountSummary():
            if item.tag in ["NetLiquidation", "TotalCashValue", "BuyingPower"]:
                summary[item.tag] = float(item.value)
        return summary

    def get_portfolio_value(self):
        """Get current portfolio NAV."""
        summary = self.get_account_summary()
        return summary.get("NetLiquidation", 0)

    def update_positions(self):
        """Refresh positions from IBKR."""
        self.positions = {}
        for pos in self.ib.positions():
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
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)
        if price and price > 0 and not util.isNan(price):
            return price
        # Fallback: last price
        if ticker.last and ticker.last > 0:
            return ticker.last
        if ticker.close and ticker.close > 0:
            return ticker.close
        return None

    async def check_trailing_stops(self):
        """Check and execute trailing stops."""
        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            return

        self.update_positions()

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
        """Sell a position."""
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        order = MarketOrder("SELL", abs(qty))
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
        self.ib.qualifyContracts(contract)
        order = MarketOrder("BUY", qty)
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

        portfolio_value = self.get_portfolio_value()
        if portfolio_value <= 0:
            log.error("Portfolio value is 0 — skipping rebalance")
            return

        self.update_positions()

        target_symbols = set(s["symbol"] for s in signals)
        held_symbols = set(self.positions.keys())

        # 1. SELL positions not in signals
        for sym in held_symbols - target_symbols:
            pos = self.positions[sym]
            log.info(f"Selling {sym} — no longer in top-{MAX_POSITIONS}")
            await self.sell_position(sym, pos["qty"], "dropped_from_signals")

        # Wait for sells to settle
        if held_symbols - target_symbols:
            await asyncio.sleep(2)
            self.update_positions()
            portfolio_value = self.get_portfolio_value()

        # 2. Compute target weights
        target_weight = 1.0 / MAX_POSITIONS  # equal weight
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
                # Set trailing peak
                self.trailing_peaks[sym] = price
            elif delta_qty < 0:
                # TRIM (sell excess)
                await self.sell_position(sym, abs(delta_qty), "trim_overweight")

        self.last_rebalance = datetime.now()
        self.update_positions()
        log.info(f"Rebalance complete. Positions: {list(self.positions.keys())}")

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
        log.info("=" * 60)

        summary = self.get_account_summary()
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

                # Check trailing stops every cycle
                await self.check_trailing_stops()

                # Rebalance periodically
                should_rebalance = (
                    self.last_rebalance is None or
                    (datetime.now() - self.last_rebalance).seconds >= REBALANCE_INTERVAL
                )

                if should_rebalance:
                    log.info(f"--- Cycle {cycle}: Rebalancing ---")
                    await self.rebalance()

                    # Log status
                    summary = self.get_account_summary()
                    nav = summary.get("NetLiquidation", 0)
                    cash = summary.get("TotalCashValue", 0)
                    log.info(f"NAV: ${nav:,.0f} | Cash: ${cash:,.0f} | Positions: {len(self.positions)}")

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
