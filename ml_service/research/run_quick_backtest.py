"""
Quick Backtest Runner — triggered by Grafana dashboard API
Reads predictions.parquet, runs a simple long-only ML backtest with
configurable parameters, saves results to walkforward/ directory.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
WF_DIR = DATA_DIR / "walkforward"


def run_backtest(threshold, max_positions, position_pct, hold_days, label):
    preds_path = DATA_DIR / "predictions.parquet"
    if not preds_path.exists():
        return []

    preds = pd.read_parquet(preds_path)
    preds["date"] = pd.to_datetime(preds["date"])
    # Use prob_ensemble as the signal probability
    preds = preds.rename(columns={"prob_ensemble": "probability", "fwd_ret": "forward_return"})
    preds["year"] = preds["date"].dt.year

    INITIAL = 100_000.0
    results = []

    for year in sorted(preds["year"].unique()):
        yr_preds = preds[preds["year"] == year].copy()
        if yr_preds.empty:
            continue

        dates = sorted(yr_preds["date"].unique())
        cash = INITIAL
        positions = []
        portfolio_values = []
        trade_returns = []

        for date in dates:
            day_preds = yr_preds[yr_preds["date"] == date].sort_values(
                "probability", ascending=False
            )

            # Close expired positions
            new_positions = []
            for pos in positions:
                pos["days"] += 1
                if pos["days"] >= hold_days:
                    row = day_preds[day_preds["symbol"] == pos["symbol"]]
                    if len(row) > 0:
                        fwd = row["forward_return"].values[0]
                    else:
                        fwd = 0.0
                    pnl = fwd
                    trade_returns.append(pnl)
                    cash += pos["value"] * (1 + pnl)
                else:
                    new_positions.append(pos)
            positions = new_positions

            # Open new positions
            open_slots = max_positions - len(positions)
            if open_slots > 0:
                signals = day_preds[day_preds["probability"] >= threshold].head(open_slots)
                for _, sig in signals.iterrows():
                    alloc = min(cash * position_pct, cash * 0.95)
                    if alloc < 500 or cash < alloc:
                        continue
                    positions.append({
                        "symbol": sig["symbol"],
                        "value": alloc,
                        "days": 0,
                    })
                    cash -= alloc

            total = cash + sum(p["value"] for p in positions)
            portfolio_values.append(total)

        if not portfolio_values:
            continue

        vals = pd.Series(portfolio_values)
        daily_ret = vals.pct_change().dropna()
        final = vals.iloc[-1]
        cagr = (final / INITIAL) - 1
        std = daily_ret.std()
        sharpe = (daily_ret.mean() / std * np.sqrt(252)) if std > 0 else 0
        down_std = daily_ret[daily_ret < 0].std()
        sortino = (daily_ret.mean() / down_std * np.sqrt(252)) if down_std > 0 else 0
        max_dd = ((vals - vals.cummax()) / vals.cummax()).min()
        wins = sum(1 for r in trade_returns if r > 0)
        losses = sum(1 for r in trade_returns if r <= 0)
        win_rate = wins / max(wins + losses, 1)
        gross_profit = sum(r for r in trade_returns if r > 0)
        gross_loss = abs(sum(r for r in trade_returns if r < 0))
        pf = gross_profit / max(gross_loss, 1e-6)
        avg_ret = float(np.mean(trade_returns)) if trade_returns else 0

        results.append({
            "label": f"{label}_{year}",
            "year": int(year),
            "cagr": round(float(cagr), 6),
            "sharpe": round(float(sharpe), 4),
            "sortino": round(float(sortino), 4),
            "max_dd": round(float(max_dd), 6),
            "n_trades": len(trade_returns),
            "win_rate": round(float(win_rate), 4),
            "profit_factor": round(float(pf), 4),
            "avg_trade_ret": round(float(avg_ret), 6),
            "final_value": round(float(final), 2),
            "alpha": 0,
            "beta": 0,
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--position-pct", type=float, default=0.12)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--label", type=str, default="custom")
    args = parser.parse_args()

    results = run_backtest(
        args.threshold, args.max_positions, args.position_pct, args.hold_days, args.label
    )

    WF_DIR.mkdir(parents=True, exist_ok=True)
    output_path = WF_DIR / f"{args.label}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"status": "done", "years": len(results), "file": str(output_path)}))
