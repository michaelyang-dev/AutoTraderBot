"""
CREDIT DE-RISK GATE — live signal module.
Research-validated + OOS-confirmed (see research/THREAD_B_FINDINGS.md, THREAD_T_FINDINGS.md,
WALKFORWARD_OOS_FINDINGS.md, LIVEMIRROR_FINDINGS.md). Deployed 2026-07-18.

WHAT: when the HY credit spread (FRED BAMLH0A0HYM2) is in the top 5% of its own expanding
history, halve the gross-leverage target until stress passes. Fires rarely; ~free when idle
(full-engine: +0.3pp CAGR 26yr); cuts a 2008-style MaxDD by ~7pp (−63.5%→−56.6%), and
+16.6pp on the walk-forward OOS book. De-risk ONLY — never levers up (Kelly asymmetry).

EXACT-PARITY MATH (matches livemirror_backtest._stress_pctile_map, the validated engine):
  percentile = (latest >= all history INCLUDING latest).mean(), min 252 obs.
  The backtest's shift(1) (act on yesterday's close) is satisfied structurally live:
  FRED publishes BAMLH0A0HYM2 with a 1-day lag, so the newest value in the maintained
  history IS yesterday's close when the pre-market cron runs.
  Gate applies at REBALANCE sizing only (exactly the validated cadence — the backtest's
  +6.9pp figure is with rebalance-cadence gating, not daily).

DATA: seed research/_credit_signal.parquet (hy_oas 1996->, from WRDS
fred_interest_rates_spreads_daily spliced with FRED) + daily FRED appends ->
data/credit_signal_live.parquet. The LIVE feed must be FRED (WRDS freezes over summer).

The engine calls gate_status() at rebalance: pure read of the maintained file, no network
in the trading path. FAIL-SAFE: any error or insufficient history -> derisk 1.0.
"""
import os, io, ssl, urllib.request, datetime
import pandas as pd

ML = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(ML, "research", "_credit_signal.parquet")   # hy_oas 1996-> (+ baa_aaa)
LIVE = os.path.join(ML, "data", "credit_signal_live.parquet")   # maintained by daily cron
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

PCT = 0.95          # de-risk when spread >= this expanding percentile (validated p95)
DERISK = 0.0        # gross-leverage multiplier while gated. 0.5 validated 2026-07 on the single
                    # 20-day book; 0.00 (fully flat) validated 2026-09 on the tranched book
                    # (EXP-053: monotone 0.00 > 0.25 > 0.50 on CAGR, Sharpe AND MaxDD, 26yr;
                    # re-verified on the rebuilt v2 universes, FINAL_RECOMMENDATION.md §0).
                    # With 4 tranches the gate flattens ONE book per 5 sessions, so the
                    # de-risk is gradual by construction; a 0.00 on the single book would
                    # have been a one-day full liquidation, which is why 0.5 was chosen then.
MIN_OBS = 252       # matches backtest min_periods
STALE_DAYS = 7      # gate_status flags staleness beyond this (weekends/holidays are fine)


def _fred_series(sid):
    """Fetch a FRED series (recent window suffices — we only append to the seed)."""
    try:
        raw = urllib.request.urlopen(FRED.format(sid=sid), timeout=30).read().decode()
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE   # proxy fallback
        raw = urllib.request.urlopen(FRED.format(sid=sid), timeout=30, context=ctx).read().decode()
    df = pd.read_csv(io.StringIO(raw)); df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"]); df[sid] = pd.to_numeric(df[sid], errors="coerce")
    return df.dropna().set_index("date")[sid]


def load_history():
    """LIVE file if present, else the seed. Returns DataFrame with hy_oas column."""
    return pd.read_parquet(LIVE if os.path.exists(LIVE) else SEED)


def update_history():
    """Daily cron: append the latest FRED HY OAS to the maintained history. Idempotent."""
    base = load_history()[["hy_oas"]]
    hy = _fred_series("BAMLH0A0HYM2").rename("hy_oas")
    merged = pd.concat([base["hy_oas"].dropna(), hy])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame({"hy_oas": merged})
    os.makedirs(os.path.dirname(LIVE), exist_ok=True)
    out.to_parquet(LIVE)
    return out


def _pctile(s):
    """EXACT backtest parity: (latest >= all values incl. latest).mean(), min 252 obs."""
    s = s.dropna()
    if len(s) < MIN_OBS:
        return None
    return float((s.iloc[-1] >= s).mean())


def gate_status(history=None):
    """Pure read -> dict(derisk, pctile, latest, last_date, stale, ok). FAIL-SAFE: any
    problem -> derisk 1.0 with ok=False so the engine can alert without changing sizing."""
    try:
        h = history if history is not None else load_history()
        s = h["hy_oas"].dropna()
        p = _pctile(s)
        if p is None:
            return dict(derisk=1.0, pctile=None, latest=None, last_date=None,
                        stale=True, ok=False, why="insufficient history")
        last = s.index[-1].date()
        age = (datetime.date.today() - last).days
        return dict(derisk=(DERISK if p >= PCT else 1.0), pctile=p,
                    latest=float(s.iloc[-1]), last_date=str(last),
                    stale=(age > STALE_DAYS), ok=True, why=None)
    except Exception as e:
        return dict(derisk=1.0, pctile=None, latest=None, last_date=None,
                    stale=True, ok=False, why=f"{type(e).__name__}: {e}")


def compute_derisk(history=None):
    """Convenience: just the gross-leverage multiplier (1.0 or DERISK). Fail-safe 1.0."""
    return gate_status(history)["derisk"]


if __name__ == "__main__":   # daily pre-market cron entrypoint
    try:
        h = update_history()
        st = gate_status(h)
        print(f"credit_gate: HY-OAS {st['latest']:.2f} ({st['last_date']}) "
              f"pctile={st['pctile']:.2%} -> derisk={st['derisk']:.2f} "
              f"{'DE-RISK ON' if st['derisk'] < 1 else 'normal'}"
              f"{' [STALE]' if st['stale'] else ''}")
    except Exception as e:
        print(f"credit_gate UPDATE FAILED: {type(e).__name__}: {e}")
        raise SystemExit(1)
