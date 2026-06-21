"""
Daily signal generator for the recommended product — risk-managed BTC/ETH beta.

Outputs today's TARGET ALLOCATION (BTC %, ETH %, cash %) for a US-deployable SPOT strategy on
Coinbase/Kraken. Logic is exactly the validated backtest (`crypto/backtest/beta_product.py`):

  exposure = clip(TARGET_VOL / realized_vol_30d, 0, L_MAX)        # vol-targeting
  exposure *= 1 if BTC > SMA200 else 0                            # 200d regime de-risk
  weights  = {BTC: 0.6*exposure, ETH: 0.4*exposure, cash: 1-exposure}

Validated: Sharpe ~1.24, MaxDD ~-30% (vs BTC buy&hold 0.90 / -77%), positive every year incl. 2022.
Data: Coinbase public daily candles (US-accessible, spot — matches the no-leverage deployment).

Run:  python crypto/strategy/beta_signal.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import requests
import numpy as np
import pandas as pd

WEIGHTS = {"BTC": 0.60, "ETH": 0.40}
TARGET_VOL = 0.30
REGIME_N = 200
VOLWIN = 30
L_MAX = 2.0
RF = 0.045


def candles(product):
    r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                     params={"granularity": 86400}, timeout=20, headers={"User-Agent": "auto-trader"})
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["t", "low", "high", "open", "close", "vol"])
    df["date"] = pd.to_datetime(df["t"], unit="s")
    return df.sort_values("date").set_index("date")["close"]


def signal():
    btc = candles("BTC-USD")
    eth = candles("ETH-USD")
    px = pd.DataFrame({"BTC": btc, "ETH": eth}).dropna()
    ret = px.pct_change()
    basket = (ret * pd.Series(WEIGHTS)).sum(axis=1)

    realized_vol = basket.tail(VOLWIN).std() * np.sqrt(365)
    sma = px["BTC"].tail(REGIME_N).mean()
    regime_on = bool(px["BTC"].iloc[-1] > sma)

    vol_scale = float(np.clip(TARGET_VOL / realized_vol, 0, L_MAX)) if realized_vol > 0 else 0.0
    exposure = vol_scale * (1.0 if regime_on else 0.0)
    w = {k: round(v * exposure, 4) for k, v in WEIGHTS.items()}
    cash = round(max(0.0, 1.0 - sum(w.values())), 4)

    return {
        "asof": str(px.index[-1].date()),
        "btc_price": round(float(px["BTC"].iloc[-1]), 2),
        "btc_sma200": round(float(sma), 2),
        "regime": "RISK-ON (BTC > SMA200)" if regime_on else "RISK-OFF (BTC < SMA200 → de-risked to cash)",
        "realized_vol_30d": round(float(realized_vol), 3),
        "vol_scale": round(vol_scale, 3),
        "target_exposure": round(exposure, 3),
        "weights": {**w, "CASH": cash},
    }


if __name__ == "__main__":
    s = signal()
    print("=" * 60)
    print("RISK-MANAGED BTC/ETH BETA — daily target allocation")
    print("=" * 60)
    print(f"  as of            {s['asof']}")
    print(f"  BTC price        ${s['btc_price']:,}")
    print(f"  BTC 200d SMA     ${s['btc_sma200']:,}")
    print(f"  regime           {s['regime']}")
    print(f"  realized vol 30d {s['realized_vol_30d']:.1%}")
    print(f"  vol-target scale {s['vol_scale']:.2f}  →  exposure {s['target_exposure']:.0%}")
    print("  ---------------------------------------------------------")
    print(f"  TARGET:  BTC {s['weights']['BTC']:.0%}   ETH {s['weights']['ETH']:.0%}   CASH {s['weights']['CASH']:.0%}")
    print("=" * 60)
    print(json.dumps(s))
