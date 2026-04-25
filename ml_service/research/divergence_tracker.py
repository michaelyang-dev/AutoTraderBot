#!/usr/bin/env python3
"""
Live vs Backtest Divergence Tracker
====================================
Compares actual ML trade outcomes (from the journal DB) against what the
backtester's predictions file expected for the same symbol+date.

Usage:
    python3 divergence_tracker.py                    # all ML trades
    python3 divergence_tracker.py --since 2026-04-01 # recent only
    python3 divergence_tracker.py --export report.csv
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH  = Path(__file__).resolve().parent.parent / "data" / "journal.db"
PRED_PATH = DATA_DIR / "predictions.parquet"


def load_journal_trades(db_path, since=None):
    """Load closed ML round-trips (buy+sell) from the journal."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            b.symbol,
            b.submitted_at   AS buy_date,
            b.fill_price     AS entry_price,
            b.signal_prob    AS ml_confidence,
            b.ml_rank,
            b.regime         AS entry_regime,
            s.submitted_at   AS sell_date,
            s.fill_price     AS exit_price,
            s.exit_reason,
            s.hold_days,
            s.realized_pnl,
            s.realized_pnl_pct
        FROM trades b
        JOIN trades s ON s.entry_trade_id = b.id
        WHERE b.side = 'buy'
          AND b.strategy = 'ml'
          AND b.status = 'filled'
          AND s.side = 'sell'
          AND s.status = 'filled'
    """
    params = []
    if since:
        query += " AND b.submitted_at >= ?"
        params.append(since)

    query += " ORDER BY b.submitted_at"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    # Parse dates — extract just the date portion
    df["buy_date_dt"] = pd.to_datetime(df["buy_date"], utc=True).dt.tz_localize(None).dt.normalize()
    df["sell_date_dt"] = pd.to_datetime(df["sell_date"], utc=True).dt.tz_localize(None).dt.normalize()
    df["actual_return"] = df["realized_pnl_pct"]
    return df


def load_predictions(pred_path):
    """Load the predictions parquet used by the backtester."""
    preds = pd.read_parquet(pred_path)
    preds["date"] = pd.to_datetime(preds["date"]).dt.normalize()
    return preds


def compute_divergence(trades_df, preds_df):
    """
    Match each live trade to its backtester prediction and compute divergence.

    Two comparison modes:
    - If the trade date falls within the predictions parquet range, compare
      actual return vs the parquet's fwd_ret (the true 10-day forward return
      computed from historical prices — the "ground truth" the backtester used).
    - If the trade is too recent for the parquet, we still report the trade
      with its live signal_prob but mark predicted_return as N/A.
    """
    pred_max_date = preds_df["date"].max()
    results = []

    for _, trade in trades_df.iterrows():
        sym = trade["symbol"]
        buy_date = trade["buy_date_dt"]

        predicted_return = np.nan
        predicted_prob = np.nan

        # Only attempt parquet match if trade is within prediction date range
        if buy_date <= pred_max_date:
            match = preds_df[(preds_df["symbol"] == sym) & (preds_df["date"] == buy_date)]
            if match.empty:
                for offset in [pd.Timedelta(days=-1), pd.Timedelta(days=1)]:
                    match = preds_df[(preds_df["symbol"] == sym)
                                     & (preds_df["date"] == buy_date + offset)]
                    if not match.empty:
                        break
            if not match.empty:
                predicted_return = match.iloc[0]["fwd_ret"]
                predicted_prob = match.iloc[0]["prob_ensemble"]

        actual_ret = trade["actual_return"] if trade["actual_return"] is not None else np.nan

        results.append({
            "symbol": sym,
            "buy_date": trade["buy_date"],
            "sell_date": trade["sell_date"],
            "exit_reason": trade["exit_reason"],
            "hold_days": trade["hold_days"],
            "entry_regime": trade["entry_regime"],
            "ml_confidence_live": trade["ml_confidence"],
            "ml_confidence_pred": predicted_prob,
            "actual_return": actual_ret,
            "predicted_return": predicted_return,
            "divergence": actual_ret - predicted_return if not (np.isnan(actual_ret) or np.isnan(predicted_return)) else np.nan,
            "direction_match": (
                (actual_ret > 0) == (predicted_return > 0)
                if not (np.isnan(actual_ret) or np.isnan(predicted_return))
                else None
            ),
            "has_prediction": not np.isnan(predicted_return),
        })

    return pd.DataFrame(results)


def print_report(div_df):
    """Print a human-readable divergence report."""
    if div_df.empty:
        print("No ML trades found.")
        return

    print("=" * 80)
    print("LIVE vs BACKTEST DIVERGENCE REPORT")
    print("=" * 80)

    n = len(div_df)
    has_ret = div_df.dropna(subset=["actual_return"])
    matched = div_df[div_df.get("has_prediction", False)] if "has_prediction" in div_df.columns else div_df.dropna(subset=["predicted_return"])
    n_matched = len(matched)

    print(f"\nTrades: {n} total, {len(has_ret)} with returns, {n_matched} matched to predictions")

    # ── Section 1: Live trade quality (always available) ──
    if not has_ret.empty:
        print(f"\n{'─' * 40}")
        print("LIVE TRADE QUALITY")
        live_wins = (has_ret["actual_return"] > 0).sum()
        live_losses = (has_ret["actual_return"] <= 0).sum()
        print(f"  Win rate:             {live_wins}/{len(has_ret)} ({live_wins / len(has_ret) * 100:.0f}%)")
        print(f"  Avg return:           {has_ret['actual_return'].mean() * 100:+.2f}%")
        print(f"  Median return:        {has_ret['actual_return'].median() * 100:+.2f}%")
        winners = has_ret[has_ret["actual_return"] > 0]
        losers = has_ret[has_ret["actual_return"] <= 0]
        if not winners.empty:
            print(f"  Avg winner:           {winners['actual_return'].mean() * 100:+.2f}%")
        if not losers.empty:
            print(f"  Avg loser:            {losers['actual_return'].mean() * 100:+.2f}%")
        avg_hold = has_ret["hold_days"].dropna().mean()
        if not np.isnan(avg_hold):
            print(f"  Avg hold days:        {avg_hold:.1f}")

        # Confidence calibration: do higher-confidence picks do better?
        conf_col = has_ret.dropna(subset=["ml_confidence_live"])
        if len(conf_col) >= 4:
            print(f"\n{'─' * 40}")
            print("CONFIDENCE CALIBRATION")
            print("  (Do higher-confidence ML picks produce better returns?)")
            median_conf = conf_col["ml_confidence_live"].median()
            high = conf_col[conf_col["ml_confidence_live"] >= median_conf]
            low = conf_col[conf_col["ml_confidence_live"] < median_conf]
            print(f"  High conf (>= {median_conf:.2f}):  n={len(high):3d}  avg={high['actual_return'].mean() * 100:+.2f}%  win={( high['actual_return'] > 0).sum()}/{len(high)}")
            print(f"  Low  conf (<  {median_conf:.2f}):  n={len(low):3d}  avg={low['actual_return'].mean() * 100:+.2f}%  win={(low['actual_return'] > 0).sum()}/{len(low)}")
            if len(conf_col) >= 6:
                corr = conf_col["ml_confidence_live"].corr(conf_col["actual_return"])
                print(f"  Confidence-return corr: {corr:.3f}")

        # By exit reason
        print(f"\n{'─' * 40}")
        print("BY EXIT REASON")
        for reason, grp in has_ret.groupby("exit_reason"):
            if reason is None:
                continue
            avg_ret = grp["actual_return"].mean() * 100
            wr = (grp["actual_return"] > 0).sum()
            print(f"  {str(reason):20s}  n={len(grp):3d}  avg={avg_ret:+.2f}%  win={wr}/{len(grp)}")

        # By regime
        regimes = has_ret.dropna(subset=["entry_regime"])
        if not regimes.empty:
            print(f"\n{'─' * 40}")
            print("BY REGIME")
            for regime, grp in regimes.groupby("entry_regime"):
                avg_ret = grp["actual_return"].mean() * 100
                wr = (grp["actual_return"] > 0).sum()
                print(f"  {regime:12s}  n={len(grp):3d}  avg={avg_ret:+.2f}%  win={wr}/{len(grp)}")

    # ── Section 2: Prediction divergence (only when matched) ──
    if n_matched > 0:
        print(f"\n{'─' * 40}")
        print("PREDICTION DIVERGENCE (live vs backtester fwd_ret)")
        print(f"  Live avg return:      {matched['actual_return'].mean() * 100:+.2f}%")
        print(f"  Predicted avg return: {matched['predicted_return'].mean() * 100:+.2f}%")
        print(f"  Avg divergence:       {matched['divergence'].mean() * 100:+.2f}%")
        print(f"  Divergence std:       {matched['divergence'].std() * 100:.2f}%")

        dir_matches = matched["direction_match"].sum()
        print(f"  Direction match:      {dir_matches}/{n_matched} ({dir_matches / n_matched * 100:.0f}%)")

        valid = matched.dropna(subset=["actual_return", "predicted_return"])
        if len(valid) >= 3:
            corr = valid["actual_return"].corr(valid["predicted_return"])
            print(f"  Return correlation:   {corr:.3f}")
    else:
        print(f"\n  (No prediction matches — predictions parquet may need regeneration)")
        print(f"  Run: python3 signal_server.py --rebuild-predictions to update")

    # ── Per-trade detail ──
    show_df = has_ret if not has_ret.empty else div_df
    print(f"\n{'─' * 40}")
    print("TRADE DETAILS (most recent first)")
    print(f"  {'Symbol':8s} {'Buy Date':12s} {'Hold':>4s} {'Exit':20s} {'Live':>8s} {'Conf':>6s} {'Pred':>8s} {'Div':>8s}")
    for _, r in show_df.sort_values("buy_date", ascending=False).head(40).iterrows():
        buy_d = str(r["buy_date"])[:10]
        hold = str(int(r["hold_days"])) if pd.notna(r.get("hold_days")) else "?"
        exit_r = str(r.get("exit_reason") or "?")[:20]
        live = f"{r['actual_return'] * 100:+.2f}%" if pd.notna(r.get("actual_return")) else "N/A"
        conf = f"{r['ml_confidence_live'] * 100:.0f}%" if pd.notna(r.get("ml_confidence_live")) else "?"
        pred = f"{r['predicted_return'] * 100:+.2f}%" if pd.notna(r.get("predicted_return")) else "N/A"
        div = f"{r['divergence'] * 100:+.2f}%" if pd.notna(r.get("divergence")) else "---"
        print(f"  {r['symbol']:8s} {buy_d:12s} {hold:>4s} {exit_r:20s} {live:>8s} {conf:>6s} {pred:>8s} {div:>8s}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Live vs Backtest Divergence Tracker")
    parser.add_argument("--since", default=None, help="Only include trades after this date (YYYY-MM-DD)")
    parser.add_argument("--db", default=None, help="Path to journal.db (default: data/journal.db)")
    parser.add_argument("--predictions", default=None, help="Path to predictions parquet")
    parser.add_argument("--export", default=None, help="Export results to CSV")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_PATH
    pred_path = Path(args.predictions) if args.predictions else PRED_PATH

    if not db_path.exists():
        print(f"Journal DB not found: {db_path}")
        sys.exit(1)
    if not pred_path.exists():
        print(f"Predictions file not found: {pred_path}")
        sys.exit(1)

    print(f"Journal:     {db_path}")
    print(f"Predictions: {pred_path}")

    trades = load_journal_trades(db_path, since=args.since)
    if trades.empty:
        print("\nNo closed ML trades found in journal.")
        sys.exit(0)

    preds = load_predictions(pred_path)
    print(f"Loaded {len(trades)} ML round-trips, {len(preds)} prediction rows")

    div_df = compute_divergence(trades, preds)
    print_report(div_df)

    if args.export:
        div_df.to_csv(args.export, index=False)
        print(f"Exported to {args.export}")


if __name__ == "__main__":
    main()
