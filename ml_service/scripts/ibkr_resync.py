#!/usr/bin/env python3
"""One-time IBKR resync: sell positions not on Alpaca, buy Alpaca positions missing from IBKR."""
import asyncio
import requests
from ib_insync import *

LEVERAGE = 1.5
POSITION_CAP = 0.15  # v12: must match backtest
ALPACA_HOLDS = {"EA", "EOG", "FISV", "GILD", "GPN", "MRK", "MU", "SNDK"}


async def resync():
    ib = IB()
    await ib.connectAsync("127.0.0.1", 4002, clientId=99, timeout=20)
    ib.reqMarketDataType(3)  # delayed data

    # Current IBKR positions
    positions = await ib.reqPositionsAsync()
    held = {}
    for p in positions:
        if p.position != 0:
            held[p.contract.symbol] = {"qty": int(p.position), "contract": p.contract}

    print("Current IBKR positions:")
    for sym in sorted(held):
        print(f"  {sym}: {held[sym]['qty']} shares")

    # Get signal probabilities
    r = requests.get("http://localhost:5001/signals")
    sigs = r.json()["signals"]
    prob_map = {s["symbol"]: s["probability"] for s in sigs}

    # Compute targets
    ibkr_syms = set(held.keys())
    to_sell = ibkr_syms - ALPACA_HOLDS
    to_buy = ALPACA_HOLDS - ibkr_syms
    print(f"\nSELL (not in Alpaca): {sorted(to_sell)}")
    print(f"BUY (in Alpaca, not IBKR): {sorted(to_buy)}")

    # Get NAV
    summary = {}
    items = await ib.accountSummaryAsync()
    for item in items:
        if item.tag in ["NetLiquidation", "TotalCashValue", "BuyingPower"]:
            summary[item.tag] = float(item.value)
    nav = summary.get("NetLiquidation", 0)
    cash = summary.get("TotalCashValue", 0)
    print(f"\nIBKR NAV: ${nav:,.0f}, Cash: ${cash:,.0f}")

    # Signal-proportional targets
    total_prob = sum(prob_map.get(s, 0.01) for s in ALPACA_HOLDS)
    targets = {}
    for sym in ALPACA_HOLDS:
        prob = prob_map.get(sym, 0.01)
        w = min((prob / total_prob) * LEVERAGE, POSITION_CAP)
        targets[sym] = nav * w

    print("\nTarget positions:")
    for sym in sorted(targets, key=targets.get, reverse=True):
        print(f"  {sym}: ${targets[sym]:,.0f} (prob={prob_map.get(sym, 0):.4f})")
    print(f"  Total: ${sum(targets.values()):,.0f} ({sum(targets.values())/nav:.2f}x)")

    # === EXECUTE SELLS ===
    for sym in sorted(to_sell):
        qty = held[sym]["qty"]
        contract = Stock(sym, "SMART", "USD")
        await ib.qualifyContractsAsync(contract)
        order = MarketOrder("SELL", abs(qty))
        order.tif = "DAY"
        trade = ib.placeOrder(contract, order)
        print(f"\n  SELL {sym}: {abs(qty)} shares")
        await asyncio.sleep(3)
        print(f"    Status: {trade.orderStatus.status}, filled: {trade.orderStatus.filled}")

    # Wait for sells to settle
    if to_sell:
        print("\nWaiting 5s for sells to settle...")
        await asyncio.sleep(5)
        # Refresh NAV
        items = await ib.accountSummaryAsync()
        for item in items:
            if item.tag == "NetLiquidation":
                nav = float(item.value)
            if item.tag == "TotalCashValue":
                cash = float(item.value)
        print(f"Post-sell NAV: ${nav:,.0f}, Cash: ${cash:,.0f}")

        # Recompute targets with new NAV
        total_prob = sum(prob_map.get(s, 0.01) for s in ALPACA_HOLDS)
        targets = {}
        for sym in ALPACA_HOLDS:
            prob = prob_map.get(sym, 0.01)
            w = min((prob / total_prob) * LEVERAGE, POSITION_CAP)
            targets[sym] = nav * w

    # === EXECUTE BUYS + REBALANCE ===
    # Refresh positions after sells
    await asyncio.sleep(2)
    positions = await ib.reqPositionsAsync()
    current = {}
    for p in positions:
        if p.position != 0:
            current[p.contract.symbol] = int(p.position)

    for sym in sorted(ALPACA_HOLDS):
        contract = Stock(sym, "SMART", "USD")
        await ib.qualifyContractsAsync(contract)

        # Get current price
        ticker = ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(2)
        price = ticker.marketPrice()
        if not price or price <= 0 or price != price:  # NaN check
            price = ticker.close
        if not price or price <= 0:
            print(f"\n  SKIP {sym}: cannot get price")
            ib.cancelMktData(contract)
            continue

        current_qty = current.get(sym, 0)
        current_value = current_qty * price
        target_value = targets[sym]
        target_qty = int(target_value / price)
        delta = target_qty - current_qty

        if abs(delta) < 2:
            print(f"  {sym}: OK (have {current_qty}, target {target_qty}, delta {delta})")
            ib.cancelMktData(contract)
            continue

        side = "BUY" if delta > 0 else "SELL"
        order = MarketOrder(side, abs(delta))
        order.tif = "DAY"
        trade = ib.placeOrder(contract, order)
        print(f"\n  {side} {sym}: {abs(delta)} shares @ ~${price:.2f} (have {current_qty} -> target {target_qty})")
        await asyncio.sleep(3)
        print(f"    Status: {trade.orderStatus.status}, filled: {trade.orderStatus.filled}")
        ib.cancelMktData(contract)

    # Final summary
    await asyncio.sleep(5)
    positions = await ib.reqPositionsAsync()
    print("\n=== FINAL IBKR POSITIONS ===")
    for p in sorted(positions, key=lambda x: x.contract.symbol):
        if p.position != 0:
            print(f"  {p.contract.symbol}: {int(p.position)} shares, value=${p.marketValue:,.0f}")

    items = await ib.accountSummaryAsync()
    for item in items:
        if item.tag == "NetLiquidation":
            print(f"\nFinal NAV: ${float(item.value):,.0f}")

    ib.disconnect()
    print("\nResync complete.")


if __name__ == "__main__":
    asyncio.run(resync())
