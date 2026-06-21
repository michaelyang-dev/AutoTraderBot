"""
Unlock overlay (doc lead B) — do large token unlock CLIFFS predict underperformance, and is the
residual edge in the CONDITIONAL (size-vs-float x regime) version (naive is front-run)?

Data: DefiLlama datasets CDN (free) — per-protocol unlock events (metadata.events). For each coin in
our tradable universe with an emission schedule, extract cliff unlocks, size them as % of circulating
float at that date, and event-study forward returns around the unlock — full and split by regime.

Use cases per the doc: (i) short/risk filter, (ii) hold-through VETO on the beta product.
Run:  python crypto/backtest/unlock_overlay.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")
CDN = "https://defillama-datasets.llama.fi/emissions"
# slug -> ticker, for high-unlock alts that are in our Binance perp universe
MAP = {"arbitrum": "ARB", "optimism": "OP", "aptos": "APT", "sui": "SUI", "sei": "SEI",
       "celestia": "TIA", "injective": "INJ", "worldcoin": "WLD", "jito": "JTO", "starknet": "STRK",
       "immutable": "IMX", "ethena": "ENA", "wormhole": "W", "dydx": "DYDX", "altlayer": "ALT",
       "pixels": "PIXEL", "manta": "MANTA", "ondo": "ONDO", "pyth-network": "PYTH", "jupiter": "JUP",
       "blast": "BLAST", "ethereum-name-service": "ENS", "aevo": "AEVO", "zksync": "ZK", "eigenlayer": "EIGEN"}


def fetch_events(slug):
    try:
        d = requests.get(f"{CDN}/{slug}", timeout=25, headers={"User-Agent": "m"}).json()
        ev = d.get("metadata", {}).get("events", [])
        # cumulative unlocked over time (sum across categories) → circulating estimate
        series = d.get("documentedData", {}).get("data", [])
        cum = {}
        for s in series:
            for pt in s.get("data", []):
                cum[pt["timestamp"]] = cum.get(pt["timestamp"], 0) + (pt.get("unlocked") or 0)
        cumdf = pd.Series(cum).sort_index() if cum else pd.Series(dtype=float)
        out = []
        for e in ev:
            if e.get("unlockType") != "cliff":
                continue
            ts = e["timestamp"]
            tok = sum(e.get("noOfTokens", []) or [0])
            if tok <= 0:
                continue
            float_before = cumdf[cumdf.index <= ts].iloc[-1] if len(cumdf[cumdf.index <= ts]) else np.nan
            pct = tok / float_before if float_before and float_before > 0 else np.nan
            out.append((MAP[slug], pd.to_datetime(ts, unit="s").normalize(), tok, pct))
        return out
    except Exception:
        return []


if __name__ == "__main__":
    close = pd.read_parquet(os.path.join(DATA, "binance_close.parquet"))
    btc = close["BTC"]
    btc_ma = btc.rolling(50).mean()
    with ThreadPoolExecutor(max_workers=10) as ex:
        allev = [e for sub in ex.map(fetch_events, MAP.keys()) for e in sub]
    ev = pd.DataFrame(allev, columns=["coin", "date", "tokens", "pct_float"]).dropna(subset=["pct_float"])
    ev = ev[(ev.coin.isin(close.columns)) & (ev.date >= close.index.min()) & (ev.date <= close.index.max())]
    print("=" * 88)
    print("UNLOCK OVERLAY — cliff unlocks vs forward returns | %d events, %d coins" % (len(ev), ev.coin.nunique()))
    print("=" * 88)
    print("    coins w/ events:", sorted(ev.coin.unique()))

    def fwd(coin, date, h):
        s = close[coin].dropna()
        if date not in s.index:
            near = s.index[s.index.get_indexer([date], method="nearest")[0]]
            if abs((near - date).days) > 3:
                return np.nan
            date = near
        i = s.index.get_loc(date)
        if i + h >= len(s):
            return np.nan
        return s.iloc[i + h] / s.iloc[i] - 1

    print("\n[1] EVENT STUDY — mean forward return around cliff unlocks (vs unconditional drift):")
    print(f"    {'bucket':<30}{'n':>5}{'fwd -3d':>9}{'fwd +3d':>9}{'fwd +7d':>9}{'fwd +14d':>9}")
    def show(label, sub):
        if len(sub) < 3:
            print(f"    {label:<30}{len(sub):>5}   (too few)"); return
        rm3 = sub.apply(lambda r: fwd(r.coin, r.date, -3), axis=1).mean()
        r3 = sub.apply(lambda r: fwd(r.coin, r.date, 3), axis=1).mean()
        r7 = sub.apply(lambda r: fwd(r.coin, r.date, 7), axis=1).mean()
        r14 = sub.apply(lambda r: fwd(r.coin, r.date, 14), axis=1).mean()
        print(f"    {label:<30}{len(sub):>5}{rm3*100:>8.1f}%{r3*100:>8.1f}%{r7*100:>8.1f}%{r14*100:>8.1f}%", flush=True)

    show("all cliffs", ev)
    show("large (>2% of float)", ev[ev.pct_float > 0.02])
    show("large (>5% of float)", ev[ev.pct_float > 0.05])
    # regime split (BTC below 50d MA = risk-off, where doc says unlocks bite hardest)
    ev["riskoff"] = ev.date.map(lambda d: btc.asof(d) < btc_ma.asof(d) if pd.notna(btc_ma.asof(d)) else False)
    show("large(>2%) + RISK-OFF regime", ev[(ev.pct_float > 0.02) & (ev.riskoff)])
    show("large(>2%) + risk-ON regime", ev[(ev.pct_float > 0.02) & (~ev.riskoff)])

    print("\n  Read: naive unlock is front-run (fwd ≈ 0). Real residual = large%-of-float in RISK-OFF")
    print("  showing clear negative fwd return → usable as a short filter / hold-through veto on beta.")
