#!/usr/bin/env python3
"""
Daily performance check: compare live Alpaca returns to SPY benchmark.
Run manually or add to cron after market close.

Usage: python3 scripts/performance_check.py
"""
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "journal.db"


def main():
    print("=" * 60)
    print(f"LIVE PERFORMANCE CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Check daily snapshots from journal
    if DB_PATH.exists():
        db = sqlite3.connect(str(DB_PATH))
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT 30"
        ).fetchall()

        if rows:
            print(f"\nDaily snapshots ({len(rows)} days):")
            print(f"{'Date':<12} {'Portfolio':>12} {'Day P&L':>10} {'Day %':>8} {'SPY':>8} {'Alpha':>8} {'Regime':<8}")
            print("-" * 72)

            cumulative_pnl = 0
            cumulative_spy = 0
            for row in reversed(rows):
                day_pnl = row["day_pnl"] or 0
                day_pct = (row["day_pnl_pct"] or 0) * 100
                spy_day = (row["spy_day_pct"] or 0) * 100 if row["spy_day_pct"] else 0
                alpha = day_pct - spy_day
                cumulative_pnl += day_pnl
                cumulative_spy += spy_day

                print(
                    f"{row['date']:<12} "
                    f"${row['portfolio_value']:>11,.0f} "
                    f"{'+'if day_pnl>=0 else ''}{day_pnl:>9,.0f} "
                    f"{'+'if day_pct>=0 else ''}{day_pct:>6.2f}% "
                    f"{'+'if spy_day>=0 else ''}{spy_day:>6.2f}% "
                    f"{'+'if alpha>=0 else ''}{alpha:>6.2f}% "
                    f"{row['regime'] or '':<8}"
                )

            if len(rows) >= 2:
                first = rows[-1]
                last = rows[0]
                total_ret = (last["portfolio_value"] / first["portfolio_value"] - 1) * 100
                print(f"\nPeriod return: {total_ret:+.2f}%")
                print(f"Cumulative day P&L: ${cumulative_pnl:+,.0f}")

                # Drawdown
                peak = max(r["portfolio_value"] for r in rows)
                dd = (last["portfolio_value"] / peak - 1) * 100
                print(f"Peak: ${peak:,.0f}, Current: ${last['portfolio_value']:,.0f}, Drawdown: {dd:.2f}%")
        else:
            print("\nNo daily snapshots yet (table empty).")
            print("Snapshots are recorded at market close (~3:55 PM ET).")
            print("Check back after today's close.")

        db.close()
    else:
        print(f"\nJournal DB not found at {DB_PATH}")

    # 2. Check current positions vs signals
    try:
        import requests
        r = requests.get("http://localhost:5001/signals", timeout=5)
        sigs = r.json()["signals"]
        top8 = sorted(sigs, key=lambda x: -x["probability"])[:8]
        buy_syms = set(s["symbol"] for s in top8)

        print(f"\nCurrent top-8 BUY signals: {sorted(buy_syms)}")
    except Exception as e:
        print(f"\nSignal server unavailable: {e}")

    # 3. SPY comparison from price data
    spy_file = DATA_DIR / "massive_cache" / "SPY_adj.parquet"
    if spy_file.exists():
        spy = pd.read_parquet(spy_file)
        c = spy["close"]
        print(f"\nSPY last 5 days:")
        for i in range(-5, 0):
            if abs(i) < len(c):
                d = c.index[i]
                ret = (c.iloc[i] / c.iloc[i - 1] - 1) * 100
                print(f"  {str(d.date())}: ${c.iloc[i]:.2f} ({ret:+.2f}%)")


if __name__ == "__main__":
    main()
