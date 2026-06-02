#!/usr/bin/env python3
"""
Manual rebalance: sell oversized old positions, buy all BUY signals
at signal-proportional weights. Matches exactly how the backtest sizes positions.
"""
import requests
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALPACA_BASE = None  # read from .env
LEVERAGE = 1.5
MAX_POS_PCT = 0.15  # v12: must match backtest
COST_BUFFER = 0.98  # keep 2% cash buffer

def get_headers():
    global ALPACA_BASE
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    key = secret = ""
    if os.path.exists(env_file):
        for line in open(env_file):
            line = line.strip()
            if line.startswith("ALPACA_API_KEY="):
                key = line.split("=", 1)[1].strip()
            if line.startswith("ALPACA_SECRET_KEY="):
                secret = line.split("=", 1)[1].strip()
            if line.startswith("ALPACA_BASE_URL="):
                ALPACA_BASE = line.split("=", 1)[1].strip()
    if not ALPACA_BASE:
        ALPACA_BASE = "https://paper-api.alpaca.markets"
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

def main():
    headers = get_headers()

    # 1. Get current account
    acct = requests.get(f"{ALPACA_BASE}/v2/account", headers=headers).json()
    portfolio_value = float(acct["portfolio_value"])
    cash = float(acct["cash"])
    buying_power = float(acct["buying_power"])
    print(f"Portfolio: ${portfolio_value:,.0f}, Cash: ${cash:,.0f}, Buying Power: ${buying_power:,.0f}")

    # 2. Get current positions
    positions = requests.get(f"{ALPACA_BASE}/v2/positions", headers=headers).json()
    current = {}
    for p in positions:
        current[p["symbol"]] = {
            "qty": int(float(p["qty"])),
            "value": float(p["market_value"]),
            "price": float(p["current_price"]),
        }
    print(f"Current positions: {len(current)}")

    # 3. Get target signals
    sigs = requests.get("http://localhost:5001/signals").json()["signals"]
    buys = sorted([s for s in sigs if s["signal"] == "BUY"], key=lambda x: -x["probability"])
    total_prob = sum(s["probability"] for s in buys)
    print(f"BUY signals: {len(buys)}, total prob: {total_prob:.2f}")

    # 4. Compute targets (signal-proportional, matches backtest)
    targets = {}
    for s in buys:
        w = min((s["probability"] / total_prob) * LEVERAGE, MAX_POS_PCT)
        targets[s["symbol"]] = w * portfolio_value

    total_target = sum(targets.values())
    print(f"Total target: ${total_target:,.0f} ({total_target/portfolio_value:.2f}x leverage)")

    # 5. Compute trades needed
    sells = []
    buys_needed = []

    # Sell positions not in targets
    for sym in list(current.keys()):
        if sym not in targets:
            sells.append(("SELL_ALL", sym, current[sym]["qty"], current[sym]["price"]))

    # Trim oversized positions
    for sym in current:
        if sym in targets:
            curr_val = current[sym]["value"]
            tgt_val = targets[sym]
            if curr_val > tgt_val * 1.05:  # >5% over target
                excess = curr_val - tgt_val
                shares = int(excess / current[sym]["price"])
                if shares > 0:
                    sells.append(("TRIM", sym, shares, current[sym]["price"]))

    # Buy new / top-up undersized
    for sym in targets:
        tgt_val = targets[sym]
        curr_val = current.get(sym, {}).get("value", 0)
        if curr_val < tgt_val * 0.90:  # >10% under target
            shortfall = tgt_val - curr_val
            buys_needed.append((sym, shortfall, tgt_val))

    # 6. Execute sells FIRST (frees cash)
    print(f"\n--- SELLING ({len(sells)} orders) ---")
    for action, sym, qty, price in sells:
        print(f"  {action} {sym}: {qty} shares @ ~${price:.2f}")
        try:
            resp = requests.post(f"{ALPACA_BASE}/v2/orders", headers=headers, json={
                "symbol": sym, "qty": str(qty), "side": "sell",
                "type": "market", "time_in_force": "day"
            })
            if resp.status_code in (200, 201):
                print(f"    -> Order submitted")
            else:
                print(f"    -> ERROR: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            print(f"    -> FAILED: {e}")
        time.sleep(0.5)

    # Wait for sells to settle
    if sells:
        print("\nWaiting 10s for sells to settle...")
        time.sleep(10)
        acct = requests.get(f"{ALPACA_BASE}/v2/account", headers=headers).json()
        portfolio_value = float(acct["portfolio_value"])
        cash = float(acct["cash"])
        buying_power = float(acct["buying_power"])
        print(f"Post-sell: Portfolio=${portfolio_value:,.0f}, Cash=${cash:,.0f}, BP=${buying_power:,.0f}")

        # Recompute targets with new portfolio value
        for s in buys:
            w = min((s["probability"] / total_prob) * LEVERAGE, MAX_POS_PCT)
            targets[s["symbol"]] = w * portfolio_value

        # Recompute buys needed
        positions = requests.get(f"{ALPACA_BASE}/v2/positions", headers=headers).json()
        current = {}
        for p in positions:
            current[p["symbol"]] = {
                "qty": int(float(p["qty"])),
                "value": float(p["market_value"]),
                "price": float(p["current_price"]),
            }

        buys_needed = []
        for sym in targets:
            tgt_val = targets[sym]
            curr_val = current.get(sym, {}).get("value", 0)
            if curr_val < tgt_val * 0.90:
                shortfall = tgt_val - curr_val
                buys_needed.append((sym, shortfall, tgt_val))

    # 7. Execute buys (in order of signal strength)
    buys_needed.sort(key=lambda x: -x[2])  # largest target first
    print(f"\n--- BUYING ({len(buys_needed)} orders) ---")

    # Refresh buying power
    acct = requests.get(f"{ALPACA_BASE}/v2/account", headers=headers).json()
    available = float(acct["buying_power"]) * COST_BUFFER

    for sym, shortfall, tgt_val in buys_needed:
        if available < 500:
            print(f"  SKIP {sym} -- no buying power left (${available:.0f})")
            continue

        # Get current price
        try:
            snap = requests.get(f"{ALPACA_BASE}/v2/snapshot/{sym}", headers=headers)
            if snap.status_code == 200:
                price = float(snap.json().get("latestTrade", {}).get("p", 0))
            else:
                price = 0
        except:
            price = 0

        if price <= 0:
            # Try from current positions
            price = current.get(sym, {}).get("price", 0)
        if price <= 0:
            print(f"  SKIP {sym} -- cannot get price")
            continue

        alloc = min(shortfall, available)
        shares = int(alloc / price)
        if shares <= 0:
            print(f"  SKIP {sym} -- alloc ${alloc:.0f} < 1 share @ ${price:.2f}")
            continue

        print(f"  BUY {sym}: {shares} shares @ ~${price:.2f} (${shares*price:,.0f}, target ${tgt_val:,.0f})")
        try:
            resp = requests.post(f"{ALPACA_BASE}/v2/orders", headers=headers, json={
                "symbol": sym, "qty": str(shares), "side": "buy",
                "type": "market", "time_in_force": "day"
            })
            if resp.status_code in (200, 201):
                print(f"    -> Order submitted")
                available -= shares * price
            else:
                print(f"    -> ERROR: {resp.status_code} {resp.text[:100]}")
        except Exception as e:
            print(f"    -> FAILED: {e}")
        time.sleep(0.5)

    # 8. Final state
    time.sleep(5)
    positions = requests.get(f"{ALPACA_BASE}/v2/positions", headers=headers).json()
    acct = requests.get(f"{ALPACA_BASE}/v2/account", headers=headers).json()
    print(f"\n--- FINAL STATE ---")
    print(f"Portfolio: ${float(acct['portfolio_value']):,.0f}, Cash: ${float(acct['cash']):,.0f}")
    print(f"Positions: {len(positions)}")
    for p in sorted(positions, key=lambda x: -float(x["market_value"])):
        mv = float(p["market_value"])
        pct = mv / float(acct["portfolio_value"]) * 100
        tgt_pct = targets.get(p["symbol"], 0) / float(acct["portfolio_value"]) * 100
        print(f"  {p['symbol']:6s}: ${mv:>10,.0f} ({pct:.1f}%) target={tgt_pct:.1f}%")

if __name__ == "__main__":
    main()
