#!/usr/bin/env python3
"""
DRY-RUN by default: compute the orders that would make the Alpaca paper book mirror IBKR's
current holdings (same names, same weights, same leverage). Pass --execute to actually place them.

Reads:
  ml_service/data/ibkr_close_snapshot.json   (IBKR holdings: symbol, qty, value, nav)
  /tmp/acct.json  /tmp/pos.json              (Alpaca account + positions, curl'd by caller)
"""
import json, os, sys, time
import urllib.request

EXECUTE = "--execute" in sys.argv

IB = json.load(open("ml_service/data/ibkr_close_snapshot.json"))
ib_nav = IB["nav"]
ibw = {p["symbol"]: {"w": p["value"] / ib_nav, "px": (p["value"] / p["qty"] if p["qty"] else 0)}
       for p in IB["positions"]}

acct = json.load(open("/tmp/acct.json"))
eq = float(acct["equity"])
pos = json.load(open("/tmp/pos.json"))
ap = {p["symbol"]: {"qty": float(p["qty"]), "px": float(p["current_price"]), "mv": float(p["market_value"])}
      for p in pos}

ib_gross = sum(v["w"] for v in ibw.values())
al_gross = sum(p["mv"] for p in ap.values()) / eq
print("IBKR  NAV    ${:,.0f}  | {} names | {:.2f}x gross".format(ib_nav, len(ibw), ib_gross))
print("Alpaca equity ${:,.0f}  | {} names | {:.2f}x gross".format(eq, len(ap), al_gross))
print()
print("{:<6}{:>7}{:>11}{:>11}{:>14}".format("sym", "IBKRwt", "target$", "cur$", "action"))

orders = []   # (symbol, side, qty)
buys = sells = 0.0
for s, info in sorted(ibw.items(), key=lambda kv: -kv[1]["w"]):
    px = ap[s]["px"] if s in ap else info["px"]
    tgt = info["w"] * eq
    tq = round(tgt / px) if px else 0
    cq = ap[s]["qty"] if s in ap else 0
    dq = int(tq - cq)
    tag = "  <-- MISSING" if s not in ap else ""
    if dq > 0:
        orders.append((s, "buy", dq));  buys += dq * px; act = "BUY {}".format(dq)
    elif dq < 0:
        orders.append((s, "sell", -dq)); sells += -dq * px; act = "SELL {}".format(-dq)
    else:
        act = "hold"
    print("{:<6}{:>6.1f}%{:>11,.0f}{:>11,.0f}{:>14}{}".format(s, info["w"] * 100, tgt, cq * px, act, tag))

for s in [x for x in ap if x not in ibw]:
    q = int(ap[s]["qty"])
    orders.append((s, "sell", q)); sells += ap[s]["mv"]
    print("{:<6}{:>7}{:>11,.0f}{:>11,.0f}{:>14}  <-- EXTRA".format(s, "-", 0, ap[s]["mv"], "SELL ALL " + str(q)))

new_gross = (sum(p["mv"] for p in ap.values()) + buys - sells) / eq
print()
print("Total BUY ~${:,.0f} | SELL ~${:,.0f} | net ${:+,.0f}".format(buys, sells, buys - sells))
print("Resulting Alpaca gross ~{:.2f}x  (target IBKR {:.2f}x)".format(new_gross, ib_gross))
print("{} orders total".format(len(orders)))

if not EXECUTE:
    print("\n[DRY RUN] re-run with --execute to place these orders")
    sys.exit(0)

# ---- execute ----
BASE = "https://paper-api.alpaca.markets"
H = {"APCA-API-KEY-ID": os.environ["AK"], "APCA-API-SECRET-KEY": os.environ["AS"],
     "Content-Type": "application/json"}
ok = fail = 0
# sells first (free up buying power), then buys
for s, side, qty in sorted(orders, key=lambda o: 0 if o[1] == "sell" else 1):
    body = json.dumps({"symbol": s, "qty": str(qty), "side": side,
                       "type": "market", "time_in_force": "day"}).encode()
    req = urllib.request.Request(BASE + "/v2/orders", data=body, headers=H, method="POST")
    try:
        r = json.load(urllib.request.urlopen(req, timeout=20))
        print("  OK   {} {} {} -> {}".format(side, qty, s, r.get("status")))
        ok += 1
    except Exception as e:
        msg = e.read().decode()[:120] if hasattr(e, "read") else str(e)[:120]
        print("  FAIL {} {} {} -> {}".format(side, qty, s, msg))
        fail += 1
    time.sleep(0.15)
print("\nplaced {} ok, {} failed".format(ok, fail))
