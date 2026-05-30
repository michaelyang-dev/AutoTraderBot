#!/usr/bin/env python3
"""Buy missing positions that the manual rebalance couldn't get prices for."""
import requests, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
key = secret = base = ""
for line in open(env_file):
    line = line.strip()
    if line.startswith("ALPACA_API_KEY="): key = line.split("=", 1)[1].strip()
    if line.startswith("ALPACA_SECRET_KEY="): secret = line.split("=", 1)[1].strip()
    if line.startswith("ALPACA_BASE_URL="): base = line.split("=", 1)[1].strip()

headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

acct = requests.get(base + "/v2/account", headers=headers).json()
pv = float(acct["portfolio_value"])
bp = float(acct["buying_power"])
print("Portfolio: %s, BP: %s" % ("{:,.0f}".format(pv), "{:,.0f}".format(bp)))

positions = requests.get(base + "/v2/positions", headers=headers).json()
held = set(p["symbol"] for p in positions)
print("Holding: %d positions" % len(held))

sigs = requests.get("http://localhost:5001/signals").json()["signals"]
buys = [s for s in sigs if s["signal"] == "BUY"]
total_prob = sum(s["probability"] for s in buys)

missing = []
for s in sorted(buys, key=lambda x: -x["probability"]):
    if s["symbol"] not in held:
        w = min((s["probability"] / total_prob) * 1.5, 0.25)
        missing.append((s["symbol"], w * pv, s["probability"]))

print("Missing: %d" % len(missing))

available = bp * 0.95
for sym, target, prob in missing:
    if available < 500:
        print("  SKIP %s -- no BP" % sym)
        continue

    # Get price via latest quote
    price = 0
    try:
        r = requests.get(base + "/v2/stocks/%s/quotes/latest" % sym, headers=headers)
        if r.status_code == 200:
            q = r.json().get("quote", {})
            price = float(q.get("ap", 0) or q.get("bp", 0) or 0)
    except:
        pass

    if price <= 0:
        try:
            r = requests.get(base + "/v2/stocks/%s/bars/latest" % sym, headers=headers, params={"feed": "iex"})
            if r.status_code == 200:
                price = float(r.json().get("bar", {}).get("c", 0))
        except:
            pass

    if price <= 0:
        print("  SKIP %s -- no price" % sym)
        continue

    alloc = min(target, available)
    shares = int(alloc / price)
    if shares <= 0:
        continue

    cost = shares * price
    print("  BUY %s: %d shares @ $%.2f ($%s, target $%s)" % (sym, shares, price, "{:,.0f}".format(cost), "{:,.0f}".format(target)))
    try:
        resp = requests.post(base + "/v2/orders", headers=headers, json={
            "symbol": sym, "qty": str(shares), "side": "buy",
            "type": "market", "time_in_force": "day"
        })
        if resp.status_code in (200, 201):
            print("    -> OK")
            available -= cost
        else:
            print("    -> ERR %d: %s" % (resp.status_code, resp.text[:100]))
    except Exception as e:
        print("    -> FAIL: %s" % e)
    time.sleep(0.5)

time.sleep(5)
positions = requests.get(base + "/v2/positions", headers=headers).json()
acct = requests.get(base + "/v2/account", headers=headers).json()
print("\nFinal: %d positions, Portfolio $%s" % (len(positions), "{:,.0f}".format(float(acct["portfolio_value"]))))
for p in sorted(positions, key=lambda x: -float(x["market_value"])):
    mv = float(p["market_value"])
    print("  %s: $%s (%.1f%%)" % (p["symbol"], "{:,.0f}".format(mv), mv/float(acct["portfolio_value"])*100))
