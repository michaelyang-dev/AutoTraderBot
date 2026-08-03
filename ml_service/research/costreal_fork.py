"""
Fork of LiveMirrorBacktester with a PER-NAME, PER-TRADE realistic cost model.

The production/livemirror backtester charges a flat cost_frac = (COST_BPS 5 + SLIPPAGE_BPS 5)
/1e4 = 10 bps ONE WAY (i.e. 20 bps round trip) on every dollar traded.

This fork replaces that scalar with cost_frac(symbol, date, dollars) =
    half_spread        = SPREAD_MULT * 0.5 * quoted_spread_bps(sym, date)      [CRSP DlyBid/DlyAsk]
  + market impact      = K * sqrt(dollars / ADV20$(sym,date)) * dailyvol(sym,date)
  + commission         = clip(shares*0.005, 1.0, 0.01*dollars) / dollars       [IBKR Pro fixed]
(the impact form + K=0.1 are taken verbatim from research/cost_model_comparison.py)

Implemented by source-transforming LiveMirrorBacktester.run so the strategy logic is
byte-identical to the committed one (no transcription risk).
"""
import inspect
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
ML = "/Users/michaelslyanggmail.com/Downloads/auto-trader 2/ml_service"
sys.path.insert(0, ML)
sys.path.insert(0, os.path.join(ML, "research"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import livemirror_backtest as LM  # noqa: E402

import sys as _s, os as _o
_s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
_s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
from costreal_cache import SP, ensure_cache  # noqa: E402
ensure_cache()

K_IMPACT = 0.1
COMM_PER_SHARE = 0.005
COMM_MIN = 1.0
COMM_MAX_FRAC = 0.01

# ---- source transform -------------------------------------------------------
_src = inspect.getsource(LM.LiveMirrorBacktester.run)
_reps = [
    # 3 stock liquidation sites (trailing stop, signal-exit, exit-not-targeted)
    ('cash += holdings[sym]["shares"] * px * (1 - cost_frac)',
     'cash += holdings[sym]["shares"] * px * (1 - self._cf(sym, date, holdings[sym]["shares"] * px))'),
    # rebalance deltas
    ('cost = abs(d_val) * cost_frac',
     'cost = abs(d_val) * self._cf(sym, date, abs(d_val))'),
]
for a, b in _reps:
    assert a in _src, a
    _src = _src.replace(a, b)
assert _src.count("self._cf(") == 4, _src.count("self._cf(")
# park-ETF lines keep the flat cost_frac (bond ETF, not part of the question)
_src = _src.replace("def run(", "def run_costed(", 1)
_ns = {}
import textwrap  # noqa: E402
exec(compile(textwrap.dedent(_src), "<costfork>", "exec"), vars(LM), _ns)


class CostedBacktester(LM.LiveMirrorBacktester):
    run_costed = _ns["run_costed"]

    # ---- liquidity tables -------------------------------------------------
    def load_liquidity(self):
        d = pd.read_parquet(SP + "liq.parquet")
        d = d.dropna(subset=["Ticker"])
        keep = set(self.prices.columns)
        d = d[d["Ticker"].isin(keep)]
        d["date"] = pd.to_datetime(d["YYYYMMDD"], format="%Y%m%d")
        dv = d["DlyPrcVol"].where(d["DlyPrcVol"] > 0, d["DlyClose"].abs() * d["DlyVol"])
        d["dvol"] = dv
        mid = (d["DlyBid"] + d["DlyAsk"]) / 2
        d["spr"] = ((d["DlyAsk"] - d["DlyBid"]) / mid * 1e4).where(
            (d["DlyAsk"] > d["DlyBid"]) & (mid > 0))
        d.loc[d["spr"] > 2000, "spr"] = np.nan
        # ticker collisions across PERMNOs -> keep the most liquid row
        d = d.sort_values("dvol").drop_duplicates(["Ticker", "date"], keep="last")
        dvp = d.pivot(index="date", columns="Ticker", values="dvol").sort_index()
        self.ADV = dvp.rolling(20, min_periods=5).mean()
        self.SPR = d.pivot(index="date", columns="Ticker", values="spr").sort_index().ffill(limit=10)
        self.PXV = d.pivot(index="date", columns="Ticker", values="DlyClose").abs().sort_index()
        self.VOL = self.prices.pct_change().rolling(20, min_periods=10).std()
        self._adv_i = {d_: i for i, d_ in enumerate(self.ADV.index)}
        self._advv = self.ADV.values
        self._sprv = self.SPR.values
        self._pxvv = self.PXV.values
        self._col = {c: i for i, c in enumerate(self.ADV.columns)}
        print(f"[liq] ADV {self.ADV.shape}  spread coverage "
              f"{self.SPR.notna().mean().mean():.1%}")

    # ---- the cost function ------------------------------------------------
    def _cf(self, sym, date, dollars):
        base = (LM.COST_BPS + LM.SLIPPAGE_BPS) / 10000.0
        if self.cost_mode == "flat":
            self._trades.append((date, sym, dollars, base, np.nan, np.nan))
            return base
        i = self._adv_i.get(date)
        j = self._col.get(sym)
        adv = spr = px = np.nan
        if i is not None and j is not None:
            adv = self._advv[i, j]
            spr = self._sprv[i, j]
            px = self._pxvv[i, j]
        # fallbacks: pessimistic medians for a small-cap name
        if not np.isfinite(spr):
            spr = self.fallback_spread_bps
        if not np.isfinite(adv) or adv <= 0:
            adv = self.fallback_adv
        if not np.isfinite(px) or px <= 0:
            px = 50.0
        half = self.spread_mult * 0.5 * spr / 1e4
        v = self.VOL[sym].get(date, np.nan) if sym in self.VOL.columns else np.nan
        if not np.isfinite(v):
            v = 0.02
        impact = K_IMPACT * np.sqrt(max(dollars, 0.0) / adv) * v
        shares = max(dollars, 1e-9) / px
        comm = min(max(shares * COMM_PER_SHARE, COMM_MIN), COMM_MAX_FRAC * max(dollars, 1e-9))
        commf = comm / max(dollars, 1e-9)
        extra = self.extra_bps
        if self.smallcap_extra_bps and not self._in_sp500(sym, date):
            extra += self.smallcap_extra_bps
        tot = half + impact + commf + extra / 1e4
        tot = min(tot, 0.05)
        self._trades.append((date, sym, dollars, tot, half, impact + commf))
        return float(tot)

    def _in_sp500(self, sym, date):
        """PIT S&P500 membership (the names the buggy live path was restricted to)."""
        c = getattr(self, "_sp5cache", None)
        if c is None:
            c = self._sp5cache = {}
            self._sp5keys = sorted(self.sp500_mem.keys())
        s = c.get(date)
        if s is None:
            import bisect
            i = bisect.bisect_right(self._sp5keys, date) - 1
            s = c[date] = set(self.sp500_mem[self._sp5keys[i]]) if i >= 0 else set()
        return sym in s

    def run_cost(self, start, end, config, cost_mode="real", spread_mult=1.0,
                 extra_bps=0.0, smallcap_extra_bps=0.0,
                 fallback_spread_bps=25.0, fallback_adv=3e6):
        self.cost_mode = cost_mode
        self.spread_mult = spread_mult
        self.extra_bps = extra_bps
        self.smallcap_extra_bps = smallcap_extra_bps
        self.fallback_spread_bps = fallback_spread_bps
        self.fallback_adv = fallback_adv
        self._trades = []
        m = self.run_costed(start, end, config)
        m["trades"] = pd.DataFrame(self._trades, columns=[
            "date", "sym", "dollars", "cf", "half_spread", "impact_comm"])
        return m


def clear_deployed(bt):
    bt.uni._fin_growth = {}; bt.uni._ev = {}; bt.uni._estimates = {}
    bt.uni._price_targets = {}; bt.uni._revenue_surprise = {}
    bt.uni._beat_streak = {}; bt.uni._earnings_signals = {}
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []
    bt.uni._short_interest_rank = {}; bt.uni._si_change_rank = {}


def stats_of(vals):
    dr = vals.pct_change().dropna()
    yrs = max((vals.index[-1] - vals.index[0]).days / 365.25, 1)
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / yrs) - 1
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    dd = ((vals - vals.cummax()) / vals.cummax()).min()
    return cagr, dr.std() * np.sqrt(252), sh, dd
