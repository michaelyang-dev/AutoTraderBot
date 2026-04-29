"""
Backtest Utility Functions
==========================
Pure-computation helpers shared across backtest scripts and the CLI.
No matplotlib or heavy plotting dependencies — only numpy, pandas, scipy.

Functions:
    load_predictions()  — load predictions.parquet
    calc_metrics()      — CAGR, Sharpe, Sortino, max DD, win rate, etc.
    calc_alpha_beta()   — OLS alpha (annualised) and beta vs SPY
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATA_DIR  = Path(__file__).resolve().parent / "data"
PRED_FILE = DATA_DIR / "predictions.parquet"


# ── Load predictions ─────────────────────────────────────────────────────────
def load_predictions():
    print(f"Loading {PRED_FILE} ...")
    if not PRED_FILE.exists():
        sys.exit("ERROR: predictions.parquet not found — run train_model.py first.")
    df = pd.read_parquet(PRED_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["fwd_ret"]).sort_values(["date", "symbol"])
    print(f"  {len(df):,} rows  |  {df['symbol'].nunique()} symbols  |  "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    return df


# ── Metrics ──────────────────────────────────────────────────────────────────
def calc_metrics(values: pd.Series, trades: list, years: float, label: str) -> dict:
    values = values.dropna()
    daily_ret = values.pct_change().dropna()

    cagr      = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    sharpe    = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    down_ret  = daily_ret[daily_ret < 0]
    sortino   = (daily_ret.mean() / down_ret.std() * np.sqrt(252)) if len(down_ret) > 0 else np.nan
    peak      = values.cummax()
    drawdown  = (values - peak) / peak
    max_dd    = drawdown.min()

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate      = len(wins) / len(trades) if trades else np.nan
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else np.inf
    avg_trade_ret = float(np.mean(trades)) if trades else np.nan

    return dict(
        label=label, cagr=cagr, sharpe=sharpe, sortino=sortino,
        max_dd=max_dd, n_trades=len(trades), win_rate=win_rate,
        profit_factor=profit_factor, avg_trade_ret=avg_trade_ret,
        final_value=values.iloc[-1],
    )


def calc_alpha_beta(strat_vals: pd.Series, spy_vals: pd.Series) -> tuple[float, float]:
    """OLS alpha (annualised) and beta vs SPY."""
    strat_ret = strat_vals.pct_change().dropna()
    spy_ret   = spy_vals.pct_change().dropna()
    combined  = pd.concat([strat_ret, spy_ret], axis=1, join="inner")
    combined.columns = ["strat", "spy"]
    combined = combined.dropna()
    if len(combined) < 20:
        return np.nan, np.nan
    slope, intercept, *_ = stats.linregress(combined["spy"], combined["strat"])
    alpha_daily = intercept
    alpha_ann   = (1 + alpha_daily) ** 252 - 1
    return alpha_ann, slope
