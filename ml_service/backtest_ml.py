"""
ML Strategy Backtest
====================
Simulates live trading using the 108K+ out-of-sample predictions produced by
train_model.py.  Compares the ML strategy against three benchmarks:

  1. SPY buy-and-hold
  2. Equal-weight universe buy-and-hold
  3. Random stock picking — 1,000 Monte Carlo simulations

Outputs
-------
  - Full performance metrics printed to stdout
  - Monthly / annual return breakdown
  - data/ml_equity_curve.png  (equity curve + drawdown + annual bars)

Run with:
    python3 backtest_ml.py
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CASH      = 100_000.0
MAX_POSITIONS     = 6
POSITION_PCT      = 0.12        # 12% of portfolio per position (at full confidence)
THRESHOLDS        = [0.50, 0.52, 0.55, 0.57, 0.60]   # sweep these thresholds
SLIPPAGE          = 0.0005      # 0.05% per leg (one-way)
HOLD_DAYS         = 10          # trading days to hold each position
N_RANDOM_SIMS     = 1_000       # Monte Carlo random-picker simulations

# ── SPY idle-cash parking ─────────────────────────────────────────────────────
SPY_IDLE_RESERVE_PCT   = 0.30   # always keep 30% of portfolio as cash reserve
SPY_IDLE_THRESHOLD_PCT = 0.20   # park idle cash when it exceeds 20% of portfolio
SPY_IDLE_INVEST_PCT    = 0.85   # invest 85% of idle cash into SPY

DATA_DIR   = Path(__file__).resolve().parent / "data"
PRED_FILE  = DATA_DIR / "predictions.parquet"
OUTPUT_IMG = DATA_DIR / "ml_equity_curve.png"


# ── Load predictions ──────────────────────────────────────────────────────────
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


# ── Fetch benchmark prices ────────────────────────────────────────────────────
def fetch_benchmarks(start: str, end: str, universe_syms: list[str]):
    print("Fetching benchmark prices from Yahoo Finance ...")
    syms = list(set(["SPY"] + universe_syms))
    raw = yf.download(syms, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    print(f"  SPY: {len(close)} trading days fetched")
    return close


# ── Metrics ───────────────────────────────────────────────────────────────────
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


# ── Core simulation ───────────────────────────────────────────────────────────
def run_simulation(signals_by_date: dict, all_dates: list,
                   signal_fn, years: float,
                   label: str = "Strategy", verbose: bool = True,
                   spy_prices: dict = None):
    """
    signal_fn(date, held_syms) → [(symbol, prob, fwd_ret), ...] sorted desc by prob
    spy_prices: {date → float} — when provided, idle cash is parked in SPY.

    Returns (portfolio_series, trades_list)
    """
    n_dates         = len(all_dates)
    cash            = float(INITIAL_CASH)
    idle_spy_shares = 0.0           # shares of SPY held as cash substitute
    # positions: {sym: {cost, fwd_ret, exit_idx, entry_idx, prob}}
    positions: dict = {}
    port_vals: list = []
    trades:    list = []

    for i, date in enumerate(all_dates):
        spy_px = spy_prices.get(date) if spy_prices is not None else None
        if spy_px is not None and (np.isnan(spy_px) or spy_px <= 0):
            spy_px = None

        # ── 1. Close positions reaching their 10-day hold ──
        for sym in [s for s, p in positions.items() if p["exit_idx"] == i]:
            pos    = positions.pop(sym)
            gross  = pos["cost"] * (1.0 + pos["fwd_ret"])
            net    = gross * (1.0 - SLIPPAGE)
            cash  += net
            trades.append((net - pos["cost"]) / pos["cost"])

        # ── 2. Buy new signals ──
        slots    = MAX_POSITIONS - len(positions)
        held     = set(positions.keys())
        day_sigs = signal_fn(date, held, signals_by_date.get(date, []))

        # ── 2a. Release idle SPY before buying ML picks ──
        if spy_px and idle_spy_shares > 0 and day_sigs and slots > 0:
            proceeds = idle_spy_shares * spy_px * (1.0 - SLIPPAGE)
            cash    += proceeds
            idle_spy_shares = 0.0

        for sym, prob, fwd_ret in day_sigs[:slots]:
            if np.isnan(fwd_ret):
                continue
            # ML confidence scaling: 0.55→60%, 0.80→100% (matches live executor)
            ml_mult    = min(1.0, max(0.60, prob * 1.6 - 0.28))
            port_est   = cash + sum(p["cost"] for p in positions.values())
            target_val = port_est * POSITION_PCT * ml_mult
            cost       = min(target_val, cash * 0.95)
            if cost < 50.0:
                continue
            cash -= cost * (1.0 + SLIPPAGE)
            exit_idx    = min(i + HOLD_DAYS, n_dates - 1)
            positions[sym] = dict(cost=cost, fwd_ret=fwd_ret,
                                   exit_idx=exit_idx, entry_idx=i, prob=prob)

        # ── 3. Park idle cash in SPY ──
        if spy_px:
            ml_pos_val = sum(
                p["cost"] * (1.0 + p["fwd_ret"] * (i - p["entry_idx"]) / HOLD_DAYS)
                for p in positions.values()
            )
            est_port  = cash + idle_spy_shares * spy_px + ml_pos_val
            reserved  = est_port * SPY_IDLE_RESERVE_PCT
            idle_cash = cash - reserved
            if idle_cash > est_port * SPY_IDLE_THRESHOLD_PCT:
                invest          = min(idle_cash * SPY_IDLE_INVEST_PCT, cash * 0.95)
                new_shares      = invest / spy_px
                cash           -= invest * (1.0 + SLIPPAGE)
                idle_spy_shares += new_shares

        # ── 4. Mark-to-market (ML positions + idle SPY) ──
        port_val = cash + (idle_spy_shares * spy_px if spy_px else 0)
        for sym, pos in positions.items():
            days_held  = i - pos["entry_idx"]
            interp_ret = pos["fwd_ret"] * days_held / HOLD_DAYS
            port_val  += pos["cost"] * (1.0 + interp_ret)

        port_vals.append(port_val)

    series = pd.Series(port_vals, index=pd.DatetimeIndex(all_dates), name=label)
    if verbose:
        m = calc_metrics(series, trades, years, label)
        _print_metrics(m)
    return series, trades


def _print_metrics(m: dict):
    print(f"\n{'─'*55}")
    print(f"  {m['label']}")
    print(f"{'─'*55}")
    print(f"  Final value    : ${m['final_value']:>12,.0f}")
    print(f"  CAGR           : {m['cagr']:>+8.2%}")
    print(f"  Sharpe ratio   : {m['sharpe']:>8.3f}")
    print(f"  Sortino ratio  : {m['sortino']:>8.3f}")
    print(f"  Max drawdown   : {m['max_dd']:>8.1%}")
    if not np.isnan(m["win_rate"]):
        print(f"  Total trades   : {m['n_trades']:>8,}")
        print(f"  Win rate       : {m['win_rate']:>8.1%}")
        pf = m['profit_factor']
        print(f"  Profit factor  : {pf:>8.2f}" if np.isfinite(pf) else f"  Profit factor  :      ∞")
        print(f"  Avg trade ret  : {m['avg_trade_ret']:>+8.2%}")


# ── Signal functions ──────────────────────────────────────────────────────────
def make_ml_signal_fn(threshold: float):
    """Factory — returns a signal function filtered by the given threshold."""
    def fn(date, held, day_sigs):
        return [(s, p, r) for s, p, r in day_sigs
                if p > threshold and s not in held]
    return fn


def make_random_signal_fn(rng: np.random.Generator):
    def fn(date, held, day_sigs):
        avail = [(s, p, r) for s, p, r in day_sigs if s not in held and not np.isnan(r)]
        if not avail:
            return []
        idx = rng.choice(len(avail), size=min(MAX_POSITIONS, len(avail)), replace=False)
        # Use prob=0.80 → ml_mult=1.0 so sizing is comparable to full ML confidence
        return [(avail[j][0], 0.80, avail[j][2]) for j in idx]
    return fn


# ── Monthly / annual return tables ────────────────────────────────────────────
def print_period_returns(series: pd.Series, label: str):
    monthly  = series.resample("ME").last().pct_change().dropna()
    annual   = series.resample("YE").last().pct_change().dropna()

    print(f"\n{'═'*55}")
    print(f"  {label} — Annual Returns")
    print(f"{'═'*55}")
    for date, ret in annual.items():
        bar  = "█" * int(abs(ret) * 100)
        sign = "+" if ret >= 0 else ""
        print(f"  {date.year}  {sign}{ret:6.1%}  {bar}")

    print(f"\n  {label} — Monthly Returns ({monthly.index[0].year}–{monthly.index[-1].year})")
    # Pivot to year × month grid
    tbl = monthly.copy()
    tbl.index = pd.MultiIndex.from_arrays([tbl.index.year, tbl.index.month])
    grid = tbl.unstack(level=1)
    grid.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print(grid.applymap(lambda x: f"{x:+.1%}" if pd.notna(x) else "   —  ").to_string())


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_results(sweep_results, spy_bh, ew_bh, rand_mean, rand_lo, rand_hi, annual_spy):
    """Plot equity curves for all thresholds + benchmarks, drawdowns, and annual bars."""
    fig = plt.figure(figsize=(14, 10), facecolor="#0a1628")
    gs  = fig.add_gridspec(3, 2, height_ratios=[3, 1.2, 1.4], hspace=0.38, wspace=0.28)

    C = dict(spy="#0ea5e9", ew="#f59e0b", rand="#8899aa",
             bg="#0a1628", panel="#0d1f3c", text="#c8d6e5")

    # Gradient colours for thresholds (green family, darkest = strictest)
    THRESH_COLORS = ["#34d399", "#10b981", "#059669", "#047857", "#065f46"]

    def style_ax(ax):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text"], labelsize=8)
        ax.spines[:].set_color("#1e3050")
        ax.grid(True, color="#1e3050", linewidth=0.4, linestyle="--")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # ── Top: Equity curves ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)

    ax1.fill_between(rand_mean.index, rand_lo * INITIAL_CASH / rand_lo.iloc[0],
                     rand_hi * INITIAL_CASH / rand_hi.iloc[0],
                     color=C["rand"], alpha=0.15, label="_rand_ci")
    ax1.plot(rand_mean.index, rand_mean * INITIAL_CASH / rand_mean.iloc[0],
             color=C["rand"], lw=1.0, ls="--", alpha=0.7, label="Random (median ±1σ)")
    ax1.plot(spy_bh.index, spy_bh,  color=C["spy"], lw=1.5, alpha=0.85, label="SPY Buy & Hold")
    ax1.plot(ew_bh.index,  ew_bh,   color=C["ew"],  lw=1.5, alpha=0.85, label="Equal-Weight B&H")

    for (thresh, ml_vals, m), color in zip(sweep_results, THRESH_COLORS):
        lw = 2.2 if thresh == THRESHOLDS[-1] else 1.4
        ax1.plot(ml_vals.index, ml_vals, color=color, lw=lw,
                 label=f"ML >{thresh:.2f} ({m['cagr']:+.1%} CAGR)")

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.set_title("ML Threshold Sweep — Equity Curves vs Benchmarks", color=C["text"],
                  fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Portfolio Value", color=C["text"], fontsize=9)
    leg = ax1.legend(loc="upper left", fontsize=8, framealpha=0.25,
                     labelcolor=C["text"], facecolor=C["panel"])
    for line in leg.get_lines():
        line.set_linewidth(2)

    # ── Mid-left: Drawdown comparison ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    ax2.plot(spy_bh.index, (spy_bh - spy_bh.cummax()) / spy_bh.cummax() * 100,
             color=C["spy"], lw=1.2, alpha=0.7, label="SPY")
    for (thresh, ml_vals, _), color in zip(sweep_results, THRESH_COLORS):
        dd = (ml_vals - ml_vals.cummax()) / ml_vals.cummax() * 100
        lw = 1.8 if thresh == THRESHOLDS[-1] else 1.0
        ax2.plot(dd.index, dd, color=color, lw=lw, label=f">{thresh:.2f}")
    ax2.set_title("Drawdown (%)", color=C["text"], fontsize=9, fontweight="bold")
    ax2.set_ylabel("%", color=C["text"], fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.legend(fontsize=7, labelcolor=C["text"], facecolor=C["panel"], framealpha=0.25)

    # ── Mid-right: Final values bar chart ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    labels = [f">{t:.2f}" for t, _, _ in sweep_results] + ["SPY", "EW", "Rand"]
    values = [m["final_value"] for _, _, m in sweep_results] + \
             [spy_bh.iloc[-1], ew_bh.iloc[-1], rand_mean.iloc[-1] * INITIAL_CASH / rand_mean.iloc[0]]
    colors = THRESH_COLORS + [C["spy"], C["ew"], C["rand"]]
    bars   = ax3.bar(labels, values, color=colors, width=0.6, alpha=0.85)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax3.set_title("Final Portfolio Value", color=C["text"], fontsize=9, fontweight="bold")
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f"${val/1e3:.0f}k", ha="center", va="bottom",
                 color=C["text"], fontsize=7.5, fontweight="bold")
    ax3.axhline(INITIAL_CASH, color="#4a6080", lw=0.8, ls="--", alpha=0.6)
    ax3.tick_params(axis="x", labelsize=7.5)

    # ── Bottom: Annual returns — lowest threshold vs SPY ──────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    style_ax(ax4)
    # Show two extremes: strictest (0.60) vs most permissive (0.50)
    best_thresh, best_vals, _ = sweep_results[0]   # 0.50
    base_thresh, base_vals, _ = sweep_results[-1]  # 0.60
    annual_best = best_vals.resample("YE").last().pct_change().dropna()
    annual_best.index = annual_best.index.year
    annual_base = base_vals.resample("YE").last().pct_change().dropna()
    annual_base.index = annual_base.index.year

    years_idx = np.arange(len(annual_spy))
    w = 0.25
    ax4.bar(years_idx - w,   annual_best.reindex(annual_spy.index, fill_value=np.nan) * 100,
            w, color=THRESH_COLORS[0], alpha=0.85, label=f"ML >{best_thresh:.2f}")
    ax4.bar(years_idx,       annual_base.reindex(annual_spy.index, fill_value=np.nan) * 100,
            w, color=THRESH_COLORS[-1], alpha=0.85, label=f"ML >{base_thresh:.2f}")
    ax4.bar(years_idx + w,   annual_spy * 100,
            w, color=C["spy"], alpha=0.7, label="SPY")
    ax4.axhline(0, color="#4a6080", lw=0.6)
    ax4.set_xticks(years_idx)
    ax4.set_xticklabels([str(y) for y in annual_spy.index], fontsize=8, color=C["text"])
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax4.set_title(f"Annual Returns: ML >{best_thresh:.2f} vs ML >{base_thresh:.2f} vs SPY",
                  color=C["text"], fontsize=9, fontweight="bold")
    ax4.set_ylabel("%", color=C["text"], fontsize=8)
    ax4.legend(fontsize=8, labelcolor=C["text"], facecolor=C["panel"], framealpha=0.3)

    fig.suptitle("AutoTrader ML — Probability Threshold Sweep", color=C["text"],
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"\nEquity curve saved → {OUTPUT_IMG}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.perf_counter()
    print("=" * 65)
    print("  ML Strategy Backtest — Threshold Sweep")
    print(f"  Testing thresholds: {THRESHOLDS}")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_predictions()
    universe_syms = sorted(df["symbol"].unique().tolist())
    all_dates     = sorted(df["date"].unique().tolist())
    n_dates       = len(all_dates)
    years         = (all_dates[-1] - all_dates[0]).days / 365.25

    print(f"\n  {n_dates} trading days over {years:.1f} years")
    print(f"  Universe: {len(universe_syms)} symbols")

    # Pre-build per-date signal lists (sorted by prob desc) — fast lookup
    df_sigs = df[["date", "symbol", "prob", "fwd_ret"]].copy()
    signals_by_date: dict = {}
    for date, grp in df_sigs.groupby("date"):
        grp_s = grp.sort_values("prob", ascending=False)
        signals_by_date[date] = list(
            grp_s[["symbol", "prob", "fwd_ret"]].itertuples(index=False, name=None)
        )

    # ── Fetch benchmark prices ─────────────────────────────────────────────────
    start_str = pd.Timestamp(all_dates[0]).strftime("%Y-%m-%d")
    end_str   = (pd.Timestamp(all_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    close_px  = fetch_benchmarks(start_str, end_str, universe_syms)

    # Align to our simulation dates
    close_px = close_px.reindex(
        pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates]), method="ffill"
    )

    # SPY buy-and-hold
    spy_px  = close_px["SPY"].dropna()
    spy_bh  = spy_px / spy_px.iloc[0] * INITIAL_CASH

    # Equal-weight buy-and-hold (all universe symbols)
    ew_daily = close_px[universe_syms].pct_change().mean(axis=1).fillna(0)
    ew_bh    = (1 + ew_daily).cumprod() * INITIAL_CASH

    # Benchmark metrics (aligned to first ML series index — same for all)
    spy_series = spy_bh.copy()
    ew_series  = ew_bh.copy()
    m_spy = calc_metrics(spy_series, [], years, "SPY Buy & Hold")
    m_ew  = calc_metrics(ew_series,  [], years, "Equal-Weight B&H")

    annual_spy = spy_bh.resample("YE").last().pct_change().dropna()
    annual_spy.index = annual_spy.index.year

    # ── Build SPY price lookup dict for idle-cash parking ─────────────────────
    spy_px_dict = close_px["SPY"].to_dict()   # {Timestamp → float}

    # ── Threshold sweep — pure ML (no idle-SPY) ───────────────────────────────
    print("\n" + "=" * 65)
    print("  Running threshold sweep (pure ML) ...")
    sweep_results     = []    # pure ML results
    sweep_results_spy = []    # ML + SPY idle results

    for thresh in THRESHOLDS:
        print(f"\n  ► threshold >{thresh:.2f} ...", end="  ", flush=True)
        ml_vals, ml_trades = run_simulation(
            signals_by_date, all_dates,
            signal_fn=make_ml_signal_fn(thresh),
            years=years, label=f"ML >{thresh:.2f}", verbose=False,
        )
        m = calc_metrics(ml_vals, ml_trades, years, f"ML >{thresh:.2f}")
        alpha, beta = calc_alpha_beta(ml_vals, spy_bh.reindex(ml_vals.index, method="ffill"))
        m["alpha"] = alpha
        m["beta"]  = beta
        sweep_results.append((thresh, ml_vals, m))
        print(f"CAGR {m['cagr']:+.2%}  Sharpe {m['sharpe']:.2f}  "
              f"Trades {m['n_trades']:,}  WR {m['win_rate']:.1%}")

    # ── Threshold sweep — ML + SPY idle cash parking ──────────────────────────
    print("\n" + "=" * 65)
    print("  Running threshold sweep (ML + SPY idle) ...")

    for thresh in THRESHOLDS:
        print(f"\n  ► threshold >{thresh:.2f} (+ SPY idle) ...", end="  ", flush=True)
        label_spy = f"ML+SPY >{thresh:.2f}"
        ml_spy_vals, ml_spy_trades = run_simulation(
            signals_by_date, all_dates,
            signal_fn=make_ml_signal_fn(thresh),
            years=years, label=label_spy, verbose=False,
            spy_prices=spy_px_dict,
        )
        m = calc_metrics(ml_spy_vals, ml_spy_trades, years, label_spy)
        alpha, beta = calc_alpha_beta(ml_spy_vals, spy_bh.reindex(ml_spy_vals.index, method="ffill"))
        m["alpha"] = alpha
        m["beta"]  = beta
        sweep_results_spy.append((thresh, ml_spy_vals, m))
        print(f"CAGR {m['cagr']:+.2%}  Sharpe {m['sharpe']:.2f}  "
              f"Trades {m['n_trades']:,}  WR {m['win_rate']:.1%}")

    # ── Random strategy — 1,000 Monte Carlo simulations ───────────────────────
    print(f"\n{'─'*55}")
    print(f"  Running {N_RANDOM_SIMS} random-picking simulations ...")
    rand_results = []
    t_rand = time.perf_counter()
    ref_index = sweep_results[0][1].index    # use first threshold series as date index

    for seed in range(N_RANDOM_SIMS):
        if seed % 200 == 0:
            elapsed = time.perf_counter() - t_rand
            print(f"    {seed}/{N_RANDOM_SIMS}  ({elapsed:.0f}s elapsed)", end="\r")
        rng   = np.random.default_rng(seed)
        rand_vals, _ = run_simulation(
            signals_by_date, all_dates,
            signal_fn=make_random_signal_fn(rng),
            years=years, label="", verbose=False,
        )
        rand_results.append(rand_vals.values)

    rand_matrix    = np.array(rand_results)
    rand_median    = np.median(rand_matrix, axis=0)
    rand_p25       = np.percentile(rand_matrix, 25, axis=0)
    rand_p75       = np.percentile(rand_matrix, 75, axis=0)
    rand_mean_vals = pd.Series(rand_median / rand_median[0], index=ref_index)
    rand_lo_vals   = pd.Series(rand_p25   / rand_p25[0],    index=ref_index)
    rand_hi_vals   = pd.Series(rand_p75   / rand_p75[0],    index=ref_index)
    rand_cagr      = (rand_median[-1] / rand_median[0]) ** (1 / years) - 1
    print(f"\n  Random median CAGR : {rand_cagr:+.2%}")

    # ── Helper: print a sweep table ───────────────────────────────────────────
    col_w = [14, 9, 8, 13, 10, 10, 8, 8]
    headers  = ("Strategy",   "CAGR",  "Sharpe", "Max Drawdown", "Trades", "Win Rate", "Prft F.", "Alpha")
    dividers = tuple("─" * w for w in col_w)
    sep_row  = tuple("─" * (w - 1) for w in col_w)

    def fmt_row(vals):
        return "  " + "".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    def print_sweep_table(results, title):
        print(f"\n{'═'*85}")
        print(f"  {title}")
        print(f"{'═'*85}")
        print(fmt_row(headers))
        print(fmt_row(dividers))
        for thresh, _, m in results:
            pf   = m["profit_factor"]
            pf_s = f"{pf:.2f}" if np.isfinite(pf) else "∞"
            print(fmt_row((
                f">{thresh:.2f}",
                f"{m['cagr']:+.2%}",
                f"{m['sharpe']:.3f}",
                f"{m['max_dd']:.1%}",
                f"{m['n_trades']:,}",
                f"{m['win_rate']:.1%}",
                pf_s,
                f"{m['alpha']:+.2%}",
            )))
        print(fmt_row(sep_row))
        print(fmt_row(("SPY B&H",     f"{m_spy['cagr']:+.2%}", f"{m_spy['sharpe']:.3f}", f"{m_spy['max_dd']:.1%}", "—", "—", "—", "baseline")))
        print(fmt_row(("EW B&H",      f"{m_ew['cagr']:+.2%}",  f"{m_ew['sharpe']:.3f}",  f"{m_ew['max_dd']:.1%}",  "—", "—", "—", "—")))
        print(fmt_row(("Random",      f"{rand_cagr:+.2%}",     "—", "—", "—", "—", "—", "—")))

    print_sweep_table(sweep_results,     "PURE ML — THRESHOLD SWEEP")
    print_sweep_table(sweep_results_spy, "ML + SPY IDLE — THRESHOLD SWEEP")

    # ── Head-to-head at optimal threshold (0.55) ──────────────────────────────
    opt_pure = next((r for r in sweep_results     if r[0] == 0.55), sweep_results[2])
    opt_spy  = next((r for r in sweep_results_spy if r[0] == 0.55), sweep_results_spy[2])

    print(f"\n{'═'*85}")
    print("  HEAD-TO-HEAD @ THRESHOLD 0.55 — Pure ML vs ML + SPY Idle")
    print(f"{'═'*85}")
    col2_w = [22, 15, 15]
    h2 = ("Metric", "Pure ML >0.55", "ML + SPY Idle >0.55")
    print("  " + "".join(str(v).ljust(w) for v, w in zip(h2, col2_w)))
    print("  " + "".join(("─" * (w-1)).ljust(w) for w in col2_w))
    pm, sm = opt_pure[2], opt_spy[2]
    rows2 = [
        ("CAGR",           f"{pm['cagr']:+.2%}",   f"{sm['cagr']:+.2%}"),
        ("Sharpe",         f"{pm['sharpe']:.3f}",   f"{sm['sharpe']:.3f}"),
        ("Sortino",        f"{pm['sortino']:.3f}",  f"{sm['sortino']:.3f}"),
        ("Max Drawdown",   f"{pm['max_dd']:.1%}",   f"{sm['max_dd']:.1%}"),
        ("Final Value",    f"${pm['final_value']:,.0f}", f"${sm['final_value']:,.0f}"),
        ("Alpha vs SPY",   f"{pm['alpha']:+.2%}",   f"{sm['alpha']:+.2%}"),
        ("Total Trades",   f"{pm['n_trades']:,}",   f"{sm['n_trades']:,}"),
        ("Win Rate",       f"{pm['win_rate']:.1%}", f"{sm['win_rate']:.1%}"),
    ]
    for r in rows2:
        print("  " + "".join(str(v).ljust(w) for v, w in zip(r, col2_w)))

    # ── Per-threshold annual return tables (ML + SPY idle) ────────────────────
    for thresh, ml_vals, _ in sweep_results_spy:
        print_period_returns(ml_vals, f"ML+SPY >{thresh:.2f}")

    # ── Plot (uses ML + SPY idle sweep) ───────────────────────────────────────
    plot_results(
        sweep_results = sweep_results_spy,
        spy_bh        = spy_series,
        ew_bh         = ew_series,
        rand_mean     = rand_mean_vals,
        rand_lo       = rand_lo_vals,
        rand_hi       = rand_hi_vals,
        annual_spy    = annual_spy,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
