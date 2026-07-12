"""
EDGAR OVERLAY SHADOW DIFF — phase-2 gate before any live wiring.

Compares, per symbol and feature, the CURRENT stale Compustat-derived value (exactly what
the live signal_builder computes today) against the EDGAR overlay's fresh validated value.
Reports the delta distribution and — the part that actually matters for trading — which
symbols would FLIP one of the sleeve screens/filters:

    value sleeve filters:  ROE > 5%, gross_margin > 15%, debt_to_equity < 3.0
    momentum quality boost: ROE > 15%
(thresholds per signal_builder/multi_strategy_engine — display-level shadow; the full
signal-diff runs after integration, behind the flag.)

A large flip-count is NOT an error — fresh data SHOULD change some names (that's the
point). The check is for SANITY: values in plausible ranges, no absurd jumps (>5x),
flip rate consistent with one quarter of earnings drift (~a few %).

Run on AWS after the patcher: venv/bin/python scripts/edgar_overlay_shadow.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent.parent
DATA = ML / "data"

SCREENS = {
    "roe":            [("value_roe>5%", 0.05, ">"), ("mom_roe>15%", 0.15, ">")],
    "gross_margin":   [("value_gm>15%", 0.15, ">")],
    "debt_to_equity": [("value_de<3", 3.0, "<")],
}


def stale_features():
    fund = pd.read_parquet(DATA / "wrds" / "compustat_fundamentals_quarterly.parquet")
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    last = fund.sort_values("datadate").drop_duplicates("tic", keep="last").set_index("tic")
    out = pd.DataFrame(index=last.index)
    out["roe"] = 4 * last["niq"] / last["seqq"].replace(0, np.nan)
    out["gross_margin"] = (last["saleq"] - last["cogsq"]) / last["saleq"].replace(0, np.nan)
    out["debt_to_equity"] = (last["dlttq"].fillna(0) + last["dlcq"].fillna(0)) / last["seqq"].replace(0, np.nan)
    out["datadate"] = last["datadate"]
    return out


def main():
    ov = json.load(open(DATA / "edgar_feature_overlay.json"))
    stale = stale_features()
    print(f"overlay generated {ov['generated']} | compustat max datadate {ov['compustat_max_datadate']}")
    print(f"symbols with fresh features: {len(ov['features'])}\n")
    for feat in SCREENS:
        rows = []
        for sym, feats in ov["features"].items():
            if feat not in feats or sym not in stale.index:
                continue
            old = stale.loc[sym, feat]
            new = feats[feat]["value"]
            if pd.isna(old):
                continue
            rows.append((sym, old, new, feats[feat]["period_end"]))
        if not rows:
            print(f"{feat}: no fresh values yet (expected before Q2 filings land)")
            continue
        df = pd.DataFrame(rows, columns=["sym", "old", "new", "period"]).set_index("sym")
        d = df["new"] - df["old"]
        print(f"── {feat}: {len(df)} fresh values ──")
        print(f"   delta: median {d.median():+.4f} | p5 {d.quantile(.05):+.4f} | p95 {d.quantile(.95):+.4f}")
        absurd = df[(df['new'].abs() > 5 * df['old'].abs().clip(lower=0.01)) & (df['new'].abs() > 0.5)]
        print(f"   absurd jumps (>5x & large): {len(absurd)}" + (f" -> {list(absurd.index[:5])}" if len(absurd) else ""))
        for name, thr, op in SCREENS[feat]:
            flips = ((df["old"] > thr) != (df["new"] > thr)) if op == ">" else ((df["old"] < thr) != (df["new"] < thr))
            flipped = df[flips]
            print(f"   screen {name}: {flips.mean():.1%} flip ({flips.sum()} names)"
                  + (f" e.g. {list(flipped.index[:6])}" if flips.sum() else ""))
        print()


if __name__ == "__main__":
    main()
