"""
Fast Backtester (uses exact production strategy code)
=====================================================
Uses the SAME strategy1_momentum_reversal from multi_strategy_engine.py.
Speed comes from fixing the O(N²) ROE lookup bug, not from approximation.

~16s per backtest on SP1500 (was 175s before the fix).

Usage:
    from main_production_backtest import FastBacktester
    bt = FastBacktester()
    result = bt.run("2018-01-01", "2025-12-31", config)
"""

import numpy as np
import pandas as pd
import pickle
import time
import logging
from pathlib import Path

from wrds_universe import WRDSUniverse
from wrds_data_provider import SP500Membership, WRDSDataProvider
from strategies.multi_strategy_engine import (
    strategy1_momentum_reversal, strategy3_sector_rotation,
    strategy5_lowvol_quality, INITIAL_CASH, COST_BPS,
    strategy_value, PROD_WEIGHTS_BEAR, PROD_WEIGHTS_CRASH,
)

# Map shared production weights (live keys) to the short keys this backtest uses.
_WKEY = {"s1_momentum": "mom", "s7_value": "val", "s5_lowvol": "s5", "s3_sector": "s3"}
def _short_weights(w):
    return {_WKEY[k]: v for k, v in w.items() if k in _WKEY}

log = logging.getLogger("main_production_backtest")

SLIPPAGE_BPS = 5


class FastBacktester:
    """Backtester using exact production strategy code."""

    def __init__(self, universe_path="data/wrds/complete_sp1500_universe.pkl"):
        t0 = time.time()

        with open(universe_path, "rb") as f:
            data = pickle.load(f)

        # Build WRDSUniverse (same object the production strategy uses)
        sp500_provider = SP500Membership()
        fred_rates = WRDSDataProvider().fred_rates

        self.uni = WRDSUniverse(
            prices_df=data["prices_df"],
            features_by_date=data["features_by_date"],
            sp500_provider=sp500_provider,
            fred_rates=fred_rates,
            sector_map={},
        )
        self.uni._fin_growth = data["fin_growth"]
        self.uni._ev = data["ev_data"]
        self.uni._estimates = data["estimates_data"]
        self.uni._earnings_signals = data["earnings_signals"]
        self.uni._revenue_surprise = data["revenue_surprise"]
        self.uni._beat_streak = data["beat_streak"]
        self.uni._price_targets = data["price_targets"]

        self.prices = data["prices_df"]
        self.features_by_date = data["features_by_date"]
        self.sp500_mem = data["sp500_mem"]
        self.sp400_mem = data["sp400_mem"]
        self.sp600_mem = data["sp600_mem"]

        # Pre-build SP1500 membership cache
        self._sp1500_cache = {}
        self._original_get_sp500 = self.uni.get_sp500

        # Fama-French UMD
        ff = WRDSDataProvider().fama_french
        self.umd_20d = ff["umd"].rolling(20).sum()

        # GLD/VIXM
        sec_info = pd.read_parquet("data/wrds/crsp_security_info.parquet",
                                   columns=["PERMNO", "Ticker"])
        etf_permnos = sec_info[sec_info["Ticker"].isin(["GLD", "VIXM"])]["PERMNO"].unique().tolist()
        df = pd.read_parquet("data/wrds/crsp_daily_stock_full.parquet",
                             columns=["PERMNO", "DlyCalDt", "DlyPrc", "DlyRet", "Ticker"],
                             filters=[("PERMNO", "in", etf_permnos)])
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
        etf_prices = {}
        for ticker, group in df.groupby("Ticker"):
            g = group.sort_values("DlyCalDt").drop_duplicates(subset=["DlyCalDt"], keep="last")
            ret = g["DlyRet"].fillna(0).values
            raw = g["DlyPrc"].abs().values
            n = len(ret)
            adj = np.empty(n)
            adj[-1] = raw[-1]
            for i in range(n - 2, -1, -1):
                adj[i] = adj[i + 1] / (1 + ret[i + 1]) if ret[i + 1] != 0 and not np.isnan(ret[i + 1]) else adj[i + 1]
            etf_prices[ticker] = pd.Series(adj, index=g["DlyCalDt"].values)
        self.etf_df = pd.DataFrame(etf_prices)
        self.etf_df.index = pd.to_datetime(self.etf_df.index)

        # Short interest (Compustat) — pre-compute percentile ranks per month
        self._si_ranks_by_month = {}
        self._si_change_ranks_by_month = {}
        self._si_months = []
        try:
            si = pd.read_parquet("data/wrds/compustat_short_interest.parquet",
                                 columns=["tic", "datadate", "shortintadj"])
            si["datadate"] = pd.to_datetime(si["datadate"])
            si = si.dropna(subset=["shortintadj"])
            si = si[si["shortintadj"] > 0]
            si = si.sort_values(["tic", "datadate"])

            # SI level ranks (per month)
            for month, grp in si.groupby(si["datadate"].dt.to_period("M")):
                start = month.start_time
                latest = grp.sort_values("datadate").groupby("tic")["shortintadj"].last()
                ranks = latest.rank(pct=True)
                self._si_ranks_by_month[start] = ranks.to_dict()
            self._si_months = sorted(self._si_ranks_by_month.keys())

            # SI change ranks (shorts covering = bullish)
            si["si_prev"] = si.groupby("tic")["shortintadj"].shift(2)
            si["si_change"] = (si["shortintadj"] - si["si_prev"]) / si["si_prev"]
            si_chg = si.dropna(subset=["si_change"])
            for month, grp in si_chg.groupby(si_chg["datadate"].dt.to_period("M")):
                start = month.start_time
                latest = grp.sort_values("datadate").groupby("tic")["si_change"].last()
                # Negate: most negative change (covering) = highest rank
                ranks = (-latest).rank(pct=True)
                self._si_change_ranks_by_month[start] = ranks.to_dict()

            log.info(f"Short interest loaded: {len(self._si_months)} months (level + change)")
        except Exception as e:
            log.warning(f"Could not load short interest: {e}")

        log.info(f"FastBacktester loaded in {time.time() - t0:.1f}s")

    def _get_sp1500(self, date):
        if date not in self._sp1500_cache:
            members = set()
            for mem in [self.sp500_mem, self.sp400_mem, self.sp600_mem]:
                if date in mem:
                    members.update(mem[date])
                else:
                    prior = [d for d in mem.keys() if d <= date]
                    if prior:
                        members.update(mem[max(prior)])
            self._sp1500_cache[date] = members
        return self._sp1500_cache[date]

    # NOTE: the value sleeve now lives in multi_strategy_engine.strategy_value()
    # (shared with live signal_builder — single source of truth). Imported at top.

    def _apply_rp(self, picks, date, power=1.0):
        if not picks:
            return picks
        vol60 = self.uni.get_feature_map(date, "vol_60d")
        inv = {}
        for s in picks:
            v = vol60.get(s)
            if v and not np.isnan(v) and v > 0.01:
                inv[s] = (1.0 / v) ** power
            else:
                inv[s] = 1.0
        t = sum(inv.values())
        return {s: v / t for s, v in inv.items()} if t > 0 else picks

    def run(self, start="2018-01-01", end="2025-12-31", config=None):
        if config is None:
            config = {}

        # Set universe
        universe = config.get("universe", "sp1500")
        if universe == "sp1500":
            self.uni.get_sp500 = self._get_sp1500
        else:
            self.uni.get_sp500 = self._original_get_sp500

        trading_dates = [d for d in sorted(self.prices.index)
                         if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        if not trading_dates:
            return None

        cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000
        mom_w = config.get("mom_w", 0.60)
        val_w = config.get("val_w", 0.15)
        lv_w = config.get("lv_w", 0.15)
        sec_w = config.get("sec_w", 0.10)
        top_n = config.get("top_n", 8)
        cap = config.get("cap", 0.15)
        use_rp = config.get("use_rp", True)
        rp_power = config.get("rp_power", 1.0)
        gld_pct = config.get("gld_pct", 0.0)
        vixm_pct = config.get("vixm_pct", 0.0)
        rebal_days = config.get("rebal_days", 10)
        trailing_stop = config.get("trailing_stop", None)  # e.g., -0.20
        vol_scaling = config.get("vol_scaling", False)
        vol_target = config.get("vol_target", 0.20)
        vol_lookback = config.get("vol_lookback", 40)
        recent_rets = []

        cash = INITIAL_CASH
        holdings = {}
        port_values = []
        last_targets = {}
        s4_active = {}
        gld_shares = 0
        vixm_shares = 0
        self._stop_events = []  # research: populated when config["log_stops"] is set
        self._gross_traded = 0.0  # research: cumulative $ traded (turnover measurement)

        if vixm_pct > 0 and trading_dates[0] in self.etf_df.index and "VIXM" in self.etf_df.columns:
            vp = self.etf_df.loc[trading_dates[0], "VIXM"]
            if pd.notna(vp) and vp > 0:
                vixm_shares = (INITIAL_CASH * vixm_pct) / vp
                cash -= INITIAL_CASH * vixm_pct

        for day_idx, date in enumerate(trading_dates):
            today = {}
            if date in self.prices.index:
                row = self.prices.loc[date]
                for sym in list(holdings.keys()):
                    v = row.get(sym)
                    if v is not None and not np.isnan(v):
                        today[sym] = v
                for sym in row.dropna().index:
                    today[sym] = row[sym]

            # Per-position trailing stop
            if trailing_stop:
                for sym in list(holdings):
                    px = today.get(sym)
                    if px:
                        if "peak_px" not in holdings[sym]:
                            holdings[sym]["peak_px"] = px
                        if px > holdings[sym]["peak_px"]:
                            holdings[sym]["peak_px"] = px
                        dd = (px - holdings[sym]["peak_px"]) / holdings[sym]["peak_px"]
                        if dd < -abs(trailing_stop):
                            if config.get("log_stops"):
                                self._stop_events.append({
                                    "date": date, "sym": sym, "stop_px": px,
                                    "peak_px": holdings[sym]["peak_px"],
                                    "entry_px": holdings[sym]["entry_px"],
                                })
                            cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                            del holdings[sym]

            eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                for s, h in holdings.items())
            gld_val = 0
            if gld_shares > 0 and date in self.etf_df.index and "GLD" in self.etf_df.columns:
                gp = self.etf_df.loc[date, "GLD"]
                if pd.notna(gp):
                    gld_val = gld_shares * gp
            vixm_val = 0
            if vixm_shares > 0 and date in self.etf_df.index and "VIXM" in self.etf_df.columns:
                vp = self.etf_df.loc[date, "VIXM"]
                if pd.notna(vp):
                    vixm_val = vixm_shares * vp
            total_val = eq_val + gld_val + vixm_val

            # Track returns for vol scaling
            if len(port_values) > 0:
                prev = port_values[-1][1]
                if prev > 0:
                    recent_rets.append(total_val / prev - 1)
                    if len(recent_rets) > vol_lookback:
                        recent_rets.pop(0)

            if day_idx % rebal_days != 0:
                port_values.append((date, total_val))
                continue

            # ── Update short interest ranks for this date (pre-computed, O(1)) ─────
            if self._si_months:
                midx = np.searchsorted(self._si_months, date, side="right") - 1
                if midx >= 0:
                    self.uni._short_interest_rank = self._si_ranks_by_month.get(
                        self._si_months[midx], {})
                    self.uni._si_change_rank = self._si_change_ranks_by_month.get(
                        self._si_months[midx], {})
                else:
                    self.uni._short_interest_rank = {}
                    self.uni._si_change_rank = {}

            # ── Update point-in-time enhanced data (if provided) ────
            pit_earnings = config.get("_pit_earnings_snapshots")
            if pit_earnings:
                # Find latest snapshot before current date
                prior_dates = [d for d in pit_earnings.keys() if d <= date]
                if prior_dates:
                    snap = pit_earnings[max(prior_dates)]
                    # Update fin_growth with point-in-time eps/rev surprise
                    pit_fg = {}
                    for sym, eps_s in snap.get('eps_surprise', {}).items():
                        rev_s = snap.get('rev_surprise', {}).get(sym, 0)
                        pit_fg[sym] = {"rev_growth": rev_s, "eps_growth": eps_s}
                    self.uni._fin_growth = pit_fg
                    # Update beat streak and revenue surprise
                    self.uni._beat_streak = snap.get('beat_streak', {})
                    self.uni._revenue_surprise = snap.get('rev_surprise', {})

            # ── Strategy signals (EXACT production code) ─────────
            t1 = strategy1_momentum_reversal(date, self.uni, day_idx,
                                             top_n=top_n, rebal_days=rebal_days)
            if t1 is None:
                t1 = last_targets.get("mom", {})

            members = self.uni.get_sp500(date)
            t_val = strategy_value(self.uni, date, members, top_n=10)
            t3 = strategy3_sector_rotation(date, self.uni, day_idx)
            t5 = strategy5_lowvol_quality(date, self.uni, day_idx)
            if t3 is None:
                t3 = last_targets.get("s3", {})
            if t5 is None:
                t5 = last_targets.get("s5", {})
            last_targets.update({"mom": t1, "val": t_val, "s3": t3, "s5": t5})

            # UMD regime
            nu = self.umd_20d.loc[:date]
            in_crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < -0.05
            if in_crash:
                ew = _short_weights(PROD_WEIGHTS_CRASH)  # shared with live
            else:
                ew = {"mom": mom_w, "val": val_w, "s5": lv_w, "s3": sec_w}

            # Breadth
            fdate = self.features_by_date.get(date, {})
            above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
            total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
            breadth = above / max(total_f, 1)
            blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))
            bear = config.get("bear_weights", _short_weights(PROD_WEIGHTS_BEAR))  # shared default (10/30/50/10)
            blended = {n: ew[n] * blend + bear.get(n, 0) * (1 - blend) for n in ew}

            combined = {}
            for name, cap_pct in blended.items():
                tgt = last_targets.get(name, {})
                if use_rp and name in ("mom", "val"):
                    tgt = self._apply_rp(tgt, date, power=rp_power)
                for sym, w in tgt.items():
                    if w > 0:
                        combined[sym] = combined.get(sym, 0) + w * cap_pct

            # Vol scaling: reduce positions when realized vol is high
            if vol_scaling and len(recent_rets) >= 20:
                realized_vol = np.std(recent_rets) * np.sqrt(252)
                if realized_vol > 0.01:
                    vol_scale = min(1.5, max(0.3, vol_target / realized_vol))
                    combined = {s: w * vol_scale for s, w in combined.items()}

            longs = {s: w for s, w in combined.items() if w > 0}
            for sym in list(longs):
                if longs[sym] > cap:
                    longs[sym] = cap
            gross = sum(longs.values())
            if gross > 1.0:
                for sym in longs:
                    longs[sym] /= gross
            combined = {s: w for s, w in longs.items() if w >= 0.005}

            eq_pct = 1.0 - vixm_pct - (gld_pct if gld_pct > 0 else 0)

            # Market trend exposure scaling: reduce exposure when SPY < SMA200
            trend_scale = config.get("trend_scale", None)
            if trend_scale and "SPY" in self.prices.columns:
                spy_px = today.get("SPY", 0)
                spy_hist = self.prices["SPY"].loc[:date].dropna()
                if len(spy_hist) >= 200:
                    spy_sma200 = spy_hist.tail(200).mean()
                    spy_sma50 = spy_hist.tail(50).mean()
                    if spy_px < spy_sma200:
                        eq_pct *= trend_scale.get("bear", 0.5)
                    elif spy_px < spy_sma50:
                        eq_pct *= trend_scale.get("caution", 0.75)

            target_d = {s: w * total_val * eq_pct for s, w in combined.items()}

            for sym in list(holdings):
                if sym not in target_d:
                    px = today.get(sym, holdings[sym]["entry_px"])
                    self._gross_traded += holdings[sym]["shares"] * px
                    cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                    del holdings[sym]

            for sym, tgt in target_d.items():
                px = today.get(sym)
                if not px or px <= 0:
                    continue
                cur = holdings[sym]["shares"] * px if sym in holdings else 0
                delta = tgt - cur
                if abs(delta) < total_val * 0.003:
                    continue
                self._gross_traded += abs(delta)
                cost = abs(delta) * cost_frac
                if delta > 0 and cash >= delta:
                    shares = (delta - cost) / px
                    if sym in holdings:
                        holdings[sym]["shares"] += shares
                    else:
                        holdings[sym] = {"shares": shares, "entry_px": px, "peak_px": px}
                    cash -= delta
                elif delta < 0 and sym in holdings:
                    sell = min(abs(delta) / px, holdings[sym]["shares"])
                    cash += sell * px - cost
                    holdings[sym]["shares"] -= sell
                    if holdings[sym]["shares"] < 0.01:
                        del holdings[sym]

            # GLD trend
            buy_gld = False
            if gld_pct > 0 and "GLD" in self.etf_df.columns:
                gld_px = self.etf_df["GLD"].loc[:date].dropna()
                if len(gld_px) >= 252:
                    buy_gld = gld_px.iloc[-1] > gld_px.iloc[-252:].mean()
            if gld_shares > 0 and not buy_gld:
                if date in self.etf_df.index:
                    gp = self.etf_df.loc[date, "GLD"]
                    if pd.notna(gp) and gp > 0:
                        cash += gld_shares * gp * (1 - cost_frac)
                gld_shares = 0
            if buy_gld and gld_shares == 0:
                gld_amt = total_val * gld_pct
                if date in self.etf_df.index and cash >= gld_amt:
                    gp = self.etf_df.loc[date, "GLD"]
                    if pd.notna(gp) and gp > 0:
                        gld_shares = (gld_amt - gld_amt * cost_frac) / gp
                        cash -= gld_amt

            eq_val = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                for s, h in holdings.items())
            gld_val = 0
            if gld_shares > 0 and date in self.etf_df.index and "GLD" in self.etf_df.columns:
                gp = self.etf_df.loc[date, "GLD"]
                if pd.notna(gp):
                    gld_val = gld_shares * gp
            total_val = eq_val + gld_val + vixm_val
            port_values.append((date, max(total_val, 0)))

        # Metrics
        vals = pd.Series([v for _, v in port_values],
                         index=pd.DatetimeIndex([d for d, _ in port_values]))
        years = (vals.index[-1] - vals.index[0]).days / 365.25
        if years <= 0:
            years = 1
        dr = vals.pct_change().dropna()
        cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
        sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        sd = dr[dr < 0].std()
        sortino = dr.mean() / sd * np.sqrt(252) if sd > 0 else 0
        peak = vals.cummax()
        max_dd = ((vals - peak) / peak).min()
        vol = dr.std() * np.sqrt(252)
        spy = self.prices["SPY"].reindex(vals.index, method="ffill").dropna()
        spy = spy / spy.iloc[0] * INITIAL_CASH
        spy_cagr = (spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1
        yearly = {}
        for year in range(int(start[:4]), int(end[:4]) + 1):
            mask = (vals.index >= f"{year}-01-01") & (vals.index <= f"{year}-12-31")
            yv = vals[mask]
            if len(yv) > 10:
                yr = (yv.iloc[-1] / yv.iloc[0]) - 1
                ydr = yv.pct_change().dropna()
                ys = ydr.mean() / ydr.std() * np.sqrt(252) if ydr.std() > 0 else 0
                ydd = ((yv - yv.cummax()) / yv.cummax()).min()
                yearly[year] = {"cagr": yr, "sharpe": ys, "max_dd": ydd}
        return {
            "cagr": cagr, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd, "vol": vol, "alpha": cagr - spy_cagr,
            "final": vals.iloc[-1], "yearly": yearly,
            "daily_values": vals,  # daily NAV series for proper statistical tests
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    bt = FastBacktester()

    configs = [
        ("v10 PROD: 85/15 t8 r10 trail25 G2", {"universe": "sp1500", "mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0, "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25, "gld_pct": 0.02}),
        ("v10 NO-HEDGE: 85/15 t8 r10 trail25", {"universe": "sp1500", "mom_w": 0.85, "val_w": 0.15, "lv_w": 0.0, "sec_w": 0.0, "top_n": 8, "rebal_days": 10, "trailing_stop": 0.25}),
    ]

    print(f"\n{'Config':<40} {'CAGR':>6} {'Shrp':>5} {'DD':>6} {'Time':>5}")
    print("-" * 65)

    for name, cfg in configs:
        t0 = time.time()
        m = bt.run("2018-01-01", "2025-12-31", cfg)
        t1 = time.time() - t0
        if m:
            print(f"  {name:<40} {m['cagr']:>+5.1%} {m['sharpe']:>5.2f} {m['max_dd']:>5.1%} {t1:>4.0f}s")
            for year in range(2018, 2026):
                y = m["yearly"].get(year, {})
                print(f"    {year}: {y.get('cagr', 0):+.0%}")
