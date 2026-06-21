"""
Leverage / Kelly with a PATH-DEPENDENT ABSORBING MARGIN BARRIER.
================================================================
Per spec: don't use moments/Gaussian. Take the REAL deployed-strategy 1x daily
NAV (overlays ON: vol-scaling + SPY<SMA200 de-risk, enhanced/SI cleared, 25-yr
universe incl. GFC), block-bootstrap it to preserve crash clustering, and for each
candidate gross leverage L simulate a margin account with:
  - fixed dollar debt between rebalances (leverage drifts up as equity falls),
  - debt accruing margin interest,
  - maintenance-margin liquidation checked EVERY DAY (absorbing — once liquidated
    the path holds residual cash and cannot compound back),
  - leverage reset to target L at each rebalance.
Report g(L)=E[log(terminal equity)]/yr, ruin probability, and pctiles — plus the
closed-form barrier x*(L) and the DETERMINISTIC historical path at each L.

Barrier (maintenance m): liquidate when equity < m*assets. With fixed debt,
asset-drawdown trigger x*(L) = 1 - (L-1)/((1-m)*L).

Run: cd ml_service && ./venv/bin/python research/alpha_kelly_leverage.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import pandas as pd
from main_production_backtest import FastBacktester

# ---- config of the deployed strategy (overlays ON) ----
DEPLOYED = dict(universe="sp1500", mom_w=0.50, val_w=0.35, lv_w=0.15, sec_w=0.0,
                top_n=5, cap=0.15, rebal_days=20, use_rp=False, trailing_stop=0.40,
                bear_weights={"mom": 0.10, "val": 0.30, "s5": 0.50, "s3": 0.10},
                trend_scale={"bear": 0.40, "caution": 0.75},
                vol_scaling=True, vol_target=0.20, vol_lookback=40)
START, END = "2002-01-01", "2025-12-31"

# ---- margin-account parameters ----
MAINT = 0.25          # maintenance margin (Reg-T). 0.15 sensitivity reported too.
MARGIN_RATE = 0.065   # annual borrow cost on debt (IBKR small-acct-ish)
REBAL = 20            # leverage reset cadence (trading days) = strategy rebalance
LEVS = [1.00, 1.10, 1.25, 1.40, 1.49, 1.60, 1.75, 2.00]
N_PATHS = 4000
BLOCK = 42            # block-bootstrap length (~2 months) to keep crash clustering


def barrier_x(L, m=MAINT):
    if L <= 1.0:
        return 1.0
    return 1.0 - (L - 1) / ((1 - m) * L)


def simulate(R, L, m=MAINT, rate=MARGIN_RATE, rebal=REBAL):
    """R: (n_paths, n_days) daily asset returns. Returns terminal equity per path
    and a ruin mask. Vectorized across paths, looped over days."""
    n, T = R.shape
    i_d = (1 + rate) ** (1 / 252) - 1
    E = np.ones(n)
    A = np.full(n, L)          # assets = L per $1 equity
    D = np.full(n, L - 1.0)    # debt
    ruined = np.zeros(n, dtype=bool)
    for t in range(T):
        live = ~ruined
        A[live] *= (1 + R[live, t])
        D[live] *= (1 + i_d)
        E[live] = A[live] - D[live]
        # maintenance check (daily) — liquidate if equity < m*assets
        breach = live & (E < m * A)
        if breach.any():
            # liquidate: pay debt, keep residual equity as cash (absorbing)
            E[breach] = np.maximum(A[breach] - D[breach], 0.0)
            ruined[breach] = True
            A[breach] = E[breach]
            D[breach] = 0.0
        # rebalance: reset to target leverage on live paths
        if (t + 1) % rebal == 0:
            live = ~ruined
            A[live] = L * E[live]
            D[live] = (L - 1.0) * E[live]
    E = np.maximum(E, 1e-9)
    return E, ruined


def block_bootstrap(r, n_paths, n_days, block, seed=7):
    rng = np.random.default_rng(seed)
    m = len(r)
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, m, size=(n_paths, n_blocks))
    out = np.empty((n_paths, n_blocks * block))
    for b in range(block):
        out[:, b::block] = r[(starts + b) % m]
    return out[:, :n_days]


def det_path(r, L, m=MAINT, rate=MARGIN_RATE, rebal=REBAL):
    """Deterministic historical path at leverage L (single sequence)."""
    E, A, D = 1.0, L, L - 1.0
    i_d = (1 + rate) ** (1 / 252) - 1
    ruined_day = None
    eq = []
    for t, rt in enumerate(r):
        if ruined_day is None:
            A *= (1 + rt); D *= (1 + i_d); E = A - D
            if E < m * A:
                E = max(A - D, 0.0); ruined_day = t; A, D = E, 0.0
            elif (t + 1) % rebal == 0:
                A, D = L * E, (L - 1.0) * E
        eq.append(E)
    eq = np.array(eq)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return eq[-1], ruined_day, dd


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester(universe_path="data/wrds/sp1500_universe_2000.pkl")
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets",
              "_revenue_surprise", "_beat_streak", "_earnings_signals",
              "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    print(f"[loaded {time.time()-t0:.0f}s]")

    res = bt.run(START, END, DEPLOYED)
    nav = res["daily_values"].dropna()
    r = nav.pct_change().dropna().values
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    print(f"\n=== DEPLOYED 1x NAV: {nav.index[0].date()}..{nav.index[-1].date()} "
          f"({yrs:.1f}y, {len(r)} days) ===")
    print(f"  CAGR {res['cagr']*100:.1f}%  Sharpe {res['sharpe']:.2f}  "
          f"MaxDD {res['max_dd']*100:.1f}%  Vol {res['vol']*100:.1f}%")

    # closed-form barriers
    print("\n=== closed-form liquidation triggers (maint=25%) ===")
    for L in LEVS:
        print(f"  L={L:.2f}  liquidation at asset-drawdown {barrier_x(L)*100:5.1f}%")

    # deterministic historical paths
    print(f"\n=== DETERMINISTIC historical path at each L (real {nav.index[0].year}-{nav.index[-1].year} sequence) ===")
    for L in LEVS:
        term, rd, dd = det_path(r, L)
        cagr = term ** (1 / yrs) - 1
        flag = f"LIQUIDATED day {rd} ({nav.index[min(rd,len(nav)-1)].date()})" if rd is not None else "survived"
        print(f"  L={L:.2f}  終 {term:7.2f}x  CAGR {cagr*100:6.1f}%  pathMaxDD {dd*100:6.1f}%  [{flag}]")

    # Monte-Carlo with block bootstrap
    print(f"\n=== MONTE-CARLO (block bootstrap b={BLOCK}, {N_PATHS} paths, "
          f"{yrs:.0f}y horizon, maint=25%, borrow={MARGIN_RATE*100:.1f}%) ===")
    R = block_bootstrap(r, N_PATHS, len(r), BLOCK)
    print(f"  {'L':>5} {'g(f)/yr':>9} {'medTerm':>9} {'p5Term':>8} {'p50 CAGR':>9} {'ruin%':>7}")
    for L in LEVS:
        E, ruined = simulate(R, L)
        g = np.mean(np.log(E)) / yrs
        med = np.median(E); p5 = np.percentile(E, 5)
        med_cagr = med ** (1 / yrs) - 1
        print(f"  {L:5.2f} {g*100:8.2f}% {med:8.2f}x {p5:7.3f}x {med_cagr*100:8.1f}% {ruined.mean()*100:6.1f}%")

    # maint=15% (portfolio-margin) sensitivity on g(f)/ruin
    print(f"\n=== sensitivity: maint=15% (portfolio margin) ===")
    print(f"  {'L':>5} {'g(f)/yr':>9} {'ruin%':>7}")
    for L in LEVS:
        E, ruined = simulate(R, L, m=0.15)
        g = np.mean(np.log(E)) / yrs
        print(f"  {L:5.2f} {g*100:8.2f}% {ruined.mean()*100:6.1f}%")
    print(f"\n[total {time.time()-t0:.0f}s]")
