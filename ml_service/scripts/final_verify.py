#!/usr/bin/env python3
"""Final comprehensive verification of live strategy."""
import requests, os, sys, time
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 70)
    print("FINAL STRATEGY + SYSTEM VERIFICATION")
    print("=" * 70)

    # Signals
    sigs = requests.get("http://localhost:5001/signals").json()["signals"]
    buys = sorted([s for s in sigs if s["signal"] == "BUY"], key=lambda x: -x["probability"])
    total_prob = sum(s["probability"] for s in buys)
    prob_map = {s["symbol"]: s["probability"] for s in buys}

    # Alpaca
    key = secret = base = ""
    for line in open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")):
        l = line.strip()
        if l.startswith("ALPACA_API_KEY="): key = l.split("=", 1)[1].strip()
        if l.startswith("ALPACA_SECRET_KEY="): secret = l.split("=", 1)[1].strip()
        if l.startswith("ALPACA_BASE_URL="): base = l.split("=", 1)[1].strip()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    acct = requests.get(base + "/v2/account", headers=headers).json()
    pv = float(acct["portfolio_value"])
    cash = float(acct["cash"])
    positions = requests.get(base + "/v2/positions", headers=headers).json()

    # Fundamentals
    fund_path = Path("data/wrds/compustat_fundamentals_quarterly.parquet")
    latest = None
    if fund_path.exists():
        fund = pd.read_parquet(fund_path)
        fund["datadate"] = pd.to_datetime(fund["datadate"])
        latest = fund.sort_values("datadate").groupby("tic").last()

    cache = Path("data/massive_cache")

    # === CHECK 1: Per-stock sleeve analysis ===
    print("\nCHECK 1: Per-stock sleeve analysis")
    print("%-6s %6s %-10s %s" % ("Sym", "Prob", "Sleeves", "Key metrics"))
    print("-" * 75)

    for s in buys:
        sym = s["symbol"]
        prob = s["probability"]
        ret252 = dist200 = vol60 = None
        bars = 0
        f = cache / ("%s_adj.parquet" % sym)
        if f.exists():
            df = pd.read_parquet(f)
            c = df["close"]
            bars = len(c)
            if bars >= 252: ret252 = c.iloc[-1] / c.iloc[-252] - 1
            if bars >= 200:
                sma = c.rolling(200).mean().iloc[-1]
                dist200 = (c.iloc[-1] - sma) / sma
            if bars >= 60: vol60 = c.pct_change().tail(60).std()

        roe = gm = None
        if latest is not None and sym in latest.index:
            row = latest.loc[sym]
            ceq, niq = row.get("ceqq", np.nan), row.get("niq", np.nan)
            saleq, cogsq = row.get("saleq", np.nan), row.get("cogsq", np.nan)
            if not pd.isna(ceq) and ceq > 0 and not pd.isna(niq): roe = niq * 4 / ceq
            if not pd.isna(saleq) and saleq > 0 and not pd.isna(cogsq): gm = (saleq - cogsq) / saleq

        sleeves = []
        if dist200 is not None and dist200 > 0 and ret252 is not None and ret252 > 0: sleeves.append("MOM")
        if (roe is not None and roe > 0.05 and gm is not None and gm > 0.15
                and dist200 is not None and not np.isnan(dist200) and dist200 > -0.15):
            sleeves.append("VAL")
        sleeves.append("LV")

        m = []
        if ret252 is not None: m.append("12m=%+.0f%%" % (ret252 * 100))
        if dist200 is not None: m.append("sma=%+.0f%%" % (dist200 * 100))
        if vol60 is not None: m.append("vol=%.1f%%" % (vol60 * 100))
        if roe is not None: m.append("roe=%+.0f%%" % (roe * 100))
        if bars < 252: m.append("BARS=%d!" % bars)
        print("%-6s %6.4f %-10s %s" % (sym, prob, "+".join(sleeves), ", ".join(m)))

    # === CHECK 2: Position sizing ===
    print("\nCHECK 2: Position sizing (actual vs target)")
    print("%-6s %8s %8s %6s" % ("Sym", "Actual%", "Target%", "OK?"))
    print("-" * 35)
    max_dev = 0
    for p in sorted(positions, key=lambda x: -float(x["market_value"])):
        sym = p["symbol"]
        mv = float(p["market_value"])
        actual = mv / pv * 100
        prob = prob_map.get(sym, 0)
        target = min((prob / total_prob) * 1.5, 0.25) * 100 if prob > 0 else 0
        diff = abs(actual - target)
        max_dev = max(max_dev, diff)
        ok = "OK" if diff < 3.0 else "DRIFT"
        print("%-6s %7.1f%% %7.1f%% %6s" % (sym, actual, target, ok))
    print("Max deviation: %.1f%%" % max_dev)

    # === CHECK 3: Data freshness ===
    print("\nCHECK 3: Data freshness")
    for name, path in [
        ("SP1500", "data/sp1500_members.json"),
        ("VIX", "data/enhanced_data/vix_cache.parquet"),
        ("Massive (SPY)", "data/massive_cache/SPY_adj.parquet"),
        ("Compustat", "data/wrds/compustat_fundamentals_quarterly.parquet"),
        ("IBES", "data/wrds/ibes_summary_latest.parquet"),
        ("FF UMD", "data/wrds/fama_french_5factors_momentum_daily.parquet"),
    ]:
        p = Path(path)
        if p.exists():
            age = (time.time() - p.stat().st_mtime) / 86400
            unit = "d" if age >= 1 else "h"
            val = age if age >= 1 else age * 24
            print("  %s: %.0f%s old" % (name, val, unit))
        else:
            print("  %s: MISSING!" % name)

    # === CHECK 4: No ortex ===
    print("\nCHECK 4: No ortex in system")
    h = requests.get("http://localhost:5001/health").json()
    ortex_found = any("ortex" in e.lower() for e in h.get("enhanced_data", []))
    print("  Enhanced data: %s" % ", ".join(h.get("enhanced_data", [])))
    print("  Ortex present: %s" % ("YES - PROBLEM!" if ortex_found else "NO - CLEAN"))

    # Check .env
    env_keys = []
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    for line in open(env_path):
        l = line.strip()
        if "=" in l and not l.startswith("#"):
            env_keys.append(l.split("=")[0])
    has_ortex = any("ORTEX" in k for k in env_keys)
    has_norgate = any("NORGATE" in k for k in env_keys)
    has_av = any("ALPHA_VANTAGE" in k for k in env_keys)
    print("  .env ORTEX key: %s" % ("FOUND - REMOVE!" if has_ortex else "CLEAN"))
    print("  .env NORGATE key: %s" % ("FOUND - REMOVE!" if has_norgate else "CLEAN"))
    print("  .env ALPHA_VANTAGE key: %s" % ("FOUND - REMOVE!" if has_av else "CLEAN"))
    print("  Active API keys: %s" % ", ".join(env_keys))

    # === CHECK 5: Journal DB ===
    print("\nCHECK 5: Journal DB")
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "journal.db")
    db = sqlite3.connect(db_path)
    filled = db.execute("SELECT COUNT(*) FROM trades WHERE status='filled'").fetchone()[0]
    pending = db.execute("SELECT COUNT(*) FROM trades WHERE status='pending'").fetchone()[0]
    snaps = db.execute("SELECT COUNT(*) FROM daily_snapshots").fetchone()[0]
    last_snap = db.execute("SELECT date, portfolio_value, day_pnl FROM daily_snapshots ORDER BY date DESC LIMIT 1").fetchone()
    print("  Filled trades: %d" % filled)
    print("  Pending trades: %d" % pending)
    print("  Daily snapshots: %d" % snaps)
    if last_snap:
        print("  Last snapshot: %s, $%s, P&L $%s" % (last_snap[0], "{:,.0f}".format(last_snap[1]), "{:+,.0f}".format(last_snap[2])))
    db.close()

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    leverage = (pv - cash) / pv
    print("Portfolio: $%s" % "{:,.0f}".format(pv))
    print("Cash: $%s" % "{:,.0f}".format(cash))
    print("Leverage: %.2fx (target 1.50x)" % leverage)
    print("Positions: %d held, %d BUY signals" % (len(positions), len(buys)))
    print("Max sizing deviation: %.1f%%" % max_dev)
    print("Ortex removed: %s" % ("YES" if not ortex_found and not has_ortex else "NO"))
    print("Snapshots recording: %s" % ("YES" if snaps > 0 else "NO"))
    print("Journal working: %s" % ("YES" if filled > 0 and pending == 0 else "CHECK"))

    # Match check
    held_syms = set(p["symbol"] for p in positions)
    buy_syms = set(s["symbol"] for s in buys)
    held_not_buy = held_syms - buy_syms
    buy_not_held = buy_syms - held_syms
    if held_not_buy:
        print("Held but not BUY: %s (min-hold, will sell)" % sorted(held_not_buy))
    if buy_not_held:
        print("BUY but not held: %s (will buy when cash frees)" % sorted(buy_not_held))
    if not held_not_buy and not buy_not_held:
        print("Positions PERFECTLY match BUY signals")

    all_ok = (not ortex_found and not has_ortex and snaps > 0
              and filled > 0 and pending == 0 and max_dev < 5
              and 1.3 < leverage < 1.7 and len(positions) >= 20)
    print("\nVERDICT: %s" % ("ALL SYSTEMS GO" if all_ok else "ISSUES FOUND - CHECK ABOVE"))


if __name__ == "__main__":
    main()
