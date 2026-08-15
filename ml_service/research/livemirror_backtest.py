"""
LIVE-MIRROR BACKTESTER — everything the live IBKR book actually does, in one run().

Mirrors live reality that the standard FastBacktester approximates via a return-overlay:
  - start capital $50,000 (config initial_capital)
  - gross notional = LEVERAGE * NAV (1.00x or 1.49x), applied to the dollar targets
  - INTEGER shares, ROUNDED DOWN (floor) — no fractional (IBKR API constraint)
  - real margin: cash goes negative (debit); FINANCING charged daily on the debit at 6.3%/yr
  - CREDIT GATE wired INSIDE the loop: causal expanding-percentile HY-OAS; when in the top
    (1-pct) tail, scale gross leverage by derisk (e.g. 0.5). Acts next day (shift 1).
  - vol-scaling / 40% stops / 20d rebal / bear-weights / UMD crash — all the live stack, run
    JOINTLY (not stacked overlays) so interactions are real.

Only the SIZING/execution/financing/gate is changed; ALL strategy signal code is inherited
verbatim from FastBacktester.run (copied below, edits marked  # LIVEMIRROR).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd  # noqa: E402
from main_production_backtest import FastBacktester, INITIAL_CASH, COST_BPS, SLIPPAGE_BPS  # noqa: E402
from strategies.multi_strategy_engine import (  # noqa: E402
    strategy1_momentum_reversal, strategy3_sector_rotation, strategy5_lowvol_quality,
    strategy_value, PROD_WEIGHTS_BEAR, PROD_WEIGHTS_CRASH, UMD_CRASH_THRESHOLD,
)
from main_production_backtest import _short_weights  # noqa: E402


class LiveMirrorBacktester(FastBacktester):

    def _sector_map(self):
        """ticker -> 2-digit SIC (industry group), most-recent record per ticker. Cached."""
        if getattr(self, "_secmap", None) is not None:
            return self._secmap
        si = pd.read_parquet("data/wrds/crsp_security_info.parquet", columns=["Ticker", "SICCD"])
        si = si.dropna(subset=["Ticker", "SICCD"])
        si["sic2"] = (pd.to_numeric(si["SICCD"], errors="coerce") // 100)
        si = si.dropna(subset=["sic2"])
        self._secmap = {t: int(s) for t, s in zip(si["Ticker"], si["sic2"])}
        return self._secmap

    def _credit_pctile_map(self, trading_dates, col="hy_oas"):
        """date -> causal expanding percentile of credit spread (acts t+1). Used for BOTH
        the down-gate (pctile>=0.95 -> de-risk) and the bull-calm regime (pctile<=calm)."""
        cr = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "research", "_credit_signal.parquet"))[col].dropna()
        idx = pd.DatetimeIndex(trading_dates)
        s = cr.reindex(idx.union(cr.index)).sort_index().ffill().reindex(idx)
        pr = s.expanding(min_periods=252).apply(lambda a: (a[-1] >= a).mean(), raw=True).shift(1)
        return {d: (float(v) if pd.notna(v) else 0.5) for d, v in zip(idx, pr.values)}

    # ---------------- research: NARROW "broken company" exclusion ----------------
    def _sp500_pit(self, date):
        """PIT S&P500 membership set (used to identify the newly-exposed mid/small caps)."""
        mem = self.sp500_mem
        if date in mem:
            return set(mem[date])
        keys = getattr(self, "_sp500_keys", None)
        if keys is None:
            keys = self._sp500_keys = sorted(mem.keys())
        import bisect
        i = bisect.bisect_right(keys, date) - 1
        return set(mem[keys[i]]) if i >= 0 else set()

    _BROKEN_FEATS = ("revenue_growth_yoy", "eps_growth_yoy", "net_margin", "roe", "gp_assets")

    def _is_broken(self, sym, fm, spec):
        """True = positive EVIDENCE of fundamental breakage. Missing data -> not broken
        (unless spec['missing']=='drop'). Rules are exclusion screens, never rankings."""
        rule = spec["rule"]
        g = {k: fm[k].get(sym) for k in self._BROKEN_FEATS}
        need = {"rev_and_eps": ("revenue_growth_yoy", "eps_growth_yoy"),
                "rev_and_margin": ("revenue_growth_yoy", "net_margin"),
                "net_margin_neg": ("net_margin",),
                "roe_neg": ("roe",),
                "gp_floor": ("gp_assets",),
                "triple": ("revenue_growth_yoy", "net_margin", "roe")}[rule]
        if any(g[k] is None for k in need):
            return spec.get("missing", "keep") == "drop"
        if rule == "rev_and_eps":
            return g["revenue_growth_yoy"] < spec.get("rev_thr", 0.0) and g["eps_growth_yoy"] < 0.0
        if rule == "rev_and_margin":
            return g["revenue_growth_yoy"] < spec.get("rev_thr", 0.0) and g["net_margin"] < 0.0
        if rule == "net_margin_neg":
            return g["net_margin"] < spec.get("thr", 0.0)
        if rule == "roe_neg":
            return g["roe"] < spec.get("thr", 0.0)
        if rule == "gp_floor":
            return g["gp_assets"] < spec.get("thr", 0.10)
        if rule == "triple":
            return (g["revenue_growth_yoy"] < 0.0 and g["net_margin"] < 0.0 and g["roe"] < 0.0)
        return False

    def _apply_broken_filter(self, tgt, date, keep_n, spec):
        """Drop broken names from a (larger) momentum pool, keep top-`keep_n` by score."""
        fm = {k: self.uni.get_feature_map(date, k) for k in self._BROKEN_FEATS}
        small_only = spec.get("smallcap_only", False)
        big = self._sp500_pit(date) if small_only else set()
        ranked = sorted(tgt, key=tgt.get, reverse=True)
        keep, dropped = [], []
        for s in ranked:
            if small_only and s in big:
                keep.append(s); continue
            (dropped if self._is_broken(s, fm, spec) else keep).append(s)
        self._brk_seen += min(len(ranked), keep_n)
        base_top = set(ranked[:keep_n])
        self._brk_dropped += len(base_top & set(dropped))
        if not keep:
            keep = ranked
        keep = keep[:keep_n]
        if set(keep) != base_top:
            self._brk_bind_dates += 1
        tot = sum(tgt[s] for s in keep)
        return {s: tgt[s] / tot for s in keep} if tot > 0 else {s: tgt[s] for s in keep}

    def _hi52(self):
        """Rolling 252-session max of the price matrix, cached. Trailing and INCLUSIVE of the
        current bar -- same convention as every other feature here (see BUGS A1)."""
        if getattr(self, "_hi52_cache", None) is None:
            self._hi52_cache = self.prices.rolling(252, min_periods=200).max()
        return self._hi52_cache

    def _rescore(self, date, syms, kind):
        """Alternative momentum ORDERINGS (research I-09/I-10). EXP-003 established that the
        ranking carries 100% of the selection edge and the filters carry none, so this varies
        only the ordering -- every filter, boost and regime rule in strategy1 still produced
        the pool these names came from.

        All variants are causal: they read the same trailing feature panel the sleeve reads.
        Returns {sym: score}; larger is better. Missing inputs drop the name."""
        fm = self.uni.get_feature_map
        r252 = fm(date, "ret_252d"); r126 = fm(date, "ret_126d")
        r60 = fm(date, "ret_60d"); r20 = fm(date, "ret_20d")
        v60 = fm(date, "vol_60d")
        out = {}
        if kind == "orig":
            # CONTROL: keep the deployed ordering exactly, changing only the weighting to
            # equal. Isolates "is the ordering good" from "is signal-proportional sizing good".
            return {s: 1.0 for s in syms}
        if kind == "multi":
            # rank-average of three skip-month horizons. Any single lookback is arbitrary and
            # its sampling error is large; averaging RANKS is variance reduction on the
            # estimator, not a new bet. Claim: the ensemble beats every member (falsifiable).
            legs = []
            for long_, lbl in ((r252, "12-1"), (r126, "6-1"), (r60, "3-1")):
                d = {s: long_[s] - r20[s] for s in syms if s in long_ and s in r20}
                if len(d) >= 3:
                    order = sorted(d, key=d.get)
                    legs.append({s: i / (len(order) - 1) for i, s in enumerate(order)})
            if not legs:
                return {}
            common = set.intersection(*[set(l) for l in legs])
            return {s: float(np.mean([l[s] for l in legs])) for s in common}
        if kind == "hi52":
            # George-Hwang: proximity to the 52-week high. Anchoring is a documented
            # behavioural mechanism, and nearness is far more robust to a single outlier month
            # than a raw 12-month return. Same anomaly, less noisy estimator.
            hi = self._hi52()
            if date not in hi.index:
                return {}
            row = hi.loc[date]
            for s in syms:
                px = self.prices.at[date, s] if s in self.prices.columns else None
                h = row.get(s)
                if px and h and np.isfinite(px) and np.isfinite(h) and h > 0:
                    out[s] = float(px / h)
            return out
        if kind == "voladj":
            # risk-adjusted momentum: (12-1) / vol. AUDIT01 found vol_60d the single strongest
            # feature in the whole 29-feature panel (IC -0.050, stronger than any return
            # feature), and the sleeve harvests it only via a soft x1.15 nudge.
            for s in syms:
                if s in r252 and s in r20 and s in v60 and v60[s] > 0.01:
                    out[s] = (r252[s] - r20[s]) / v60[s]
            return out
        if kind == "blend":
            a = self._rescore(date, syms, "multi")
            b = self._rescore(date, syms, "hi52")
            if not a or not b:
                return a or b
            common = set(a) & set(b)
            if len(common) < 3:
                return {}
            for d in (a, b):
                pass
            ra = sorted(common, key=lambda s: a[s]); rb = sorted(common, key=lambda s: b[s])
            pa = {s: i / (len(ra) - 1) for i, s in enumerate(ra)}
            pb = {s: i / (len(rb) - 1) for i, s in enumerate(rb)}
            return {s: 0.5 * pa[s] + 0.5 * pb[s] for s in common}
        raise KeyError(f"unknown mom_rescore {kind!r}")

    def _stress_pctile_map(self, trading_dates, col, win=None):
        """date -> causal expanding percentile of a _macro_stress column (hy_oas/rates_vol/
        curve), acts t+1. `win` re-derives rates_vol at a custom window from DGS10 if given."""
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research")
        ms = pd.read_parquet(os.path.join(base, "_macro_stress.parquet"))
        if col not in ms.columns:
            # extra signals built from data already in the universe pickle but never used for
            # leverage (see research/build_stress2.py): baa_aaa, term_inv, umd_crash, mkt_vol.
            ms2 = pd.read_parquet(os.path.join(base, "_macro_stress2.parquet"))
            if col not in ms2.columns:
                raise KeyError(f"stress column {col!r} not in _macro_stress or _macro_stress2")
            ms = ms2
        s = ms[col].dropna()
        idx = pd.DatetimeIndex(trading_dates)
        s = s.reindex(idx.union(s.index)).sort_index().ffill().reindex(idx)
        pr = s.expanding(min_periods=252).apply(lambda a: (a[-1] >= a).mean(), raw=True).shift(1)
        return {d: (float(v) if pd.notna(v) else 0.5) for d, v in zip(idx, pr.values)}

    def run(self, start="2018-01-01", end="2025-12-31", config=None):
        if config is None:
            config = {}
        universe = config.get("universe", "sp1500")
        self.uni.get_sp500 = self._get_sp1500 if universe == "sp1500" else self._original_get_sp500
        trading_dates = [d for d in sorted(self.prices.index)
                         if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        if not trading_dates:
            return None

        # ---- LIVEMIRROR params ----
        capital = float(config.get("initial_capital", INITIAL_CASH))
        leverage = float(config.get("leverage", 1.0))
        integer_shares = bool(config.get("integer_shares", False))
        fin_rate = float(config.get("financing_rate", 0.0))
        # ---- research: TIME-VARYING FINANCING (research/build_financing_curve.py) --------
        # Every backtest here charges a FLAT 6.3%/yr on the margin debit across 2001-2025.
        # That is roughly right today (IBKR Pro small-balance = benchmark + ~1.5%, fed funds
        # ~4.3%) and badly wrong historically: fed funds was ~0-0.25% through most of
        # 2009-2015 and again in 2020-21. Measured: the flat rate OVERCHARGES by 3.05pp/yr on
        # the debit over 2001-2025 (actual mean 3.25%) and 2.44pp/yr over 2018-2025.
        # It therefore biases every leverage conclusion AGAINST leverage. financing_curve=True
        # charges DFF_t + 1.5pp instead. DFF is published same-day, so no look-ahead.
        _fin_ser = None
        if config.get("financing_curve"):
            _fp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "research", "_fin_rate.parquet")
            _fs = pd.read_parquet(_fp)["fin_rate"].dropna()
            _idx = pd.DatetimeIndex(trading_dates)
            _fin_ser = _fs.reindex(_idx.union(_fs.index)).sort_index().ffill().reindex(_idx)
            _fin_ser = _fin_ser.ffill().bfill()
        credit_pct = config.get("credit_pct", None)      # down-gate percentile (e.g. 0.95)
        credit_derisk = config.get("credit_derisk", 0.5)
        crmap = self._credit_pctile_map(trading_dates, config.get("credit_col", "hy_oas")) \
            if (credit_pct or config.get("bull_weights") or config.get("bull_lever")) else {}
        # composite gate: OR over macro-stress columns (e.g. ["hy_oas","rates_vol"])
        gate_cols = config.get("gate_cols")
        gate_pct = config.get("gate_pct", 0.95)
        gate_derisk = config.get("gate_derisk", 0.5)
        gate_maps = {c: self._stress_pctile_map(trading_dates, c) for c in gate_cols} if gate_cols else {}
        # strong-bull regime knobs
        bull_w = config.get("bull_weights")              # short-key weight dict for strong bull
        bull_breadth = config.get("bull_breadth", 0.60)
        bull_need_trend = config.get("bull_need_trend", True)
        bull_credit_calm = config.get("bull_credit_calm", None)   # pctile <= this = calm
        bull_lever = config.get("bull_lever", 1.0)       # extra leverage mult in strong bull

        # cost_mult lets a variant be stress-tested at 2x/3x transaction cost. The
        # off-cadence work adds turnover, so any benefit must survive costs being worse than
        # modelled — the backtest already charges 10 bps/leg vs 6.10 measured live, but a
        # finding that only works at the modelled cost is not a finding.
        def _gate_on(d):
            """Is the de-risk gate firing on date d? (causal maps, already shifted t+1)"""
            if gate_cols:
                return any(gate_maps[c].get(d, 0.5) >= gate_pct for c in gate_cols)
            if credit_pct:
                return crmap.get(d, 0.5) >= credit_pct
            return False

        cost_frac = (COST_BPS + SLIPPAGE_BPS) / 10000 * float(config.get("cost_mult", 1.0))
        mom_w = config.get("mom_w", 0.60); val_w = config.get("val_w", 0.15)
        lv_w = config.get("lv_w", 0.15); sec_w = config.get("sec_w", 0.10)
        top_n = config.get("top_n", 8); cap = config.get("cap", 0.15)
        use_rp = config.get("use_rp", True); rp_power = config.get("rp_power", 1.0)
        rebal_days = config.get("rebal_days", 10)
        # research: STRESS-CONDITIONAL REBALANCE CADENCE. The session's one robust result is
        # that applying a rule 20 sessions late destroys it. The same lateness applies to the
        # BOOK: during a crisis the holdings are up to `rebal_days` stale. rebal_stress uses a
        # shorter cadence while the credit gate is ON. None = current behaviour exactly.
        rebal_stress = config.get("rebal_stress")
        # research (I-01): REBALANCE PHASE. The rebalance date carries no information, yet the
        # measured per-start CAGR sigma is 7.07pp (8yr) -- that spread IS phase risk, taken
        # uncompensated. `rebal_phase` shifts which day of the rebal_days cycle trades, so K
        # sub-books can be run on different phases and combined. phase 0 == previous behaviour
        # BIT-FOR-BIT. The sleeves gate internally on `day_idx % their_rebal_days`, so they are
        # handed the PHASE-ADJUSTED index (`_didx`) -- otherwise a phase of e.g. 4 makes
        # strategy5 (rebal_days=10) return None forever and the lowvol sleeve silently empties.
        _ph = int(config.get("rebal_phase", 0)) % max(int(rebal_days), 1)
        _dsr = rebal_days      # days since rebalance; seeded so day 0 rebalances (== day_idx % rd == 0)
        trailing_stop = config.get("trailing_stop", None)
        stop_mode = config.get("stop_mode", "flat")       # flat | vol   (research I-05)
        stop_k = float(config.get("stop_k", 1.2))         # threshold = k * annualised name vol
        stop_lo = float(config.get("stop_lo", 0.25))
        stop_hi = float(config.get("stop_hi", 0.60))
        stop_feat = config.get("stop_feat", "vol_60d")
        vol_scaling = config.get("vol_scaling", False)
        vol_target = config.get("vol_target", 0.20); vol_lookback = config.get("vol_lookback", 40)
        recent_rets = []

        # ---- research: DYNAMIC-LEVERAGE framework (2026-08-13) --------------------------
        # Deployed behaviour has TWO separate lags, and prior leverage-timing research only
        # ever varied the rule, never these:
        #   MEASUREMENT lag — realized vol is a 40d SIMPLE mean, so a vol spike keeps the
        #     estimate elevated ~40 sessions after it has actually passed.
        #   APPLICATION lag — vol_scale is applied ONLY inside the rebalance block, i.e. once
        #     every `rebal_days` (20). Vol can collapse the day after a rebalance and the book
        #     stays de-levered for 20 more sessions.
        # These knobs let both be varied independently. All default to CURRENT behaviour, so
        # an unset config reproduces the deployed engine exactly.
        vol_mode = config.get("vol_mode", "simple")      # simple | ewma
        vol_ewma_lam = float(config.get("vol_ewma_lambda", 0.94))
        vol_up_lb = config.get("vol_up_lookback")        # asym: window used when vol is FALLING
        vol_dn_lb = config.get("vol_dn_lookback")        # asym: window used when vol is RISING
        lev_recheck = config.get("lev_recheck_every")    # off-cadence gross re-scale, in days
        lev_band = float(config.get("lev_band", 0.0))    # only act if |target/current - 1| > band

        # ---- research (I-03): EX-ANTE, HOLDINGS-BASED portfolio-vol estimate -------------
        # Deployed vol_scale divides the target by the trailing 40d REALISED vol of the book.
        # After a rebalance rotates into five different names that estimate still describes the
        # OLD book for up to 40 sessions. This is NOT another timing rule -- every timing rule
        # in DYNAMIC_LEVERAGE_FINDINGS is dead and stays dead. It is an ESTIMATOR-LAG fix:
        # measure the risk of the book you actually hold. The DYN program only ever varied the
        # WINDOW on portfolio returns; it never changed WHAT is being estimated.
        exante = config.get("exante_vol")                 # None | "pure" | "blend"
        exante_rho = float(config.get("exante_rho", 0.35))
        exante_feat = config.get("exante_vol_feat", "vol_60d")
        _ante = {"w": None, "date": None}                 # book the estimate should describe

        def _exante_vol(date_, wmap):
            """Predicted annualized vol of the CURRENT book from per-name vols plus a single
            average pairwise correlation (single-factor approximation):

                sigma_p^2 = rho * (sum_i w_i sigma_i)^2 + (1 - rho) * sum_i (w_i sigma_i)^2

            rho is held FIXED because correlation is slow-moving while book composition is
            fast -- the whole point is to react to composition. Sensitivity to rho is swept in
            the experiment. Names with no vol estimate are dropped and the rest renormalised."""
            if not wmap:
                return None
            vm = self.uni.get_feature_map(date_, exante_feat)
            ws, ss = [], []
            for _s2, _w2 in wmap.items():
                _v2 = vm.get(_s2)
                if _v2 is None or not np.isfinite(_v2) or _v2 <= 0:
                    continue
                ws.append(float(_w2)); ss.append(float(_v2))
            if len(ws) < 3:
                return None
            ws = np.asarray(ws); ss = np.asarray(ss)
            t2 = ws.sum()
            if t2 <= 0:
                return None
            ws = ws / t2
            lin = float((ws * ss).sum())
            quad = float(((ws * ss) ** 2).sum())
            var = exante_rho * lin ** 2 + (1.0 - exante_rho) * quad
            return var ** 0.5 if var > 0 else None

        def _rv(rets, lb=None, mode=None):
            """Annualized realized vol from the shadow (unlevered) return series."""
            r = list(rets)
            if lb:
                r = r[-int(lb):]
            # min sample scales with the requested window. A hard `< 20` floor silently
            # returned None for every short-window request, which made the asymmetric arm
            # run completely UNSCALED (avgGross 1.4902, deltas exactly 0.000) while looking
            # like a legitimate result. Caught 2026-08-13.
            if len(r) < max(10, min(20, int(0.75 * (lb or 20)))):
                return None
            a = np.asarray(r, dtype=float)
            m = (mode or vol_mode)
            if m == "ewma":
                lam = vol_ewma_lam
                w = lam ** np.arange(len(a) - 1, -1, -1)
                w = w / w.sum()
                mu = float((w * a).sum())
                var = float((w * (a - mu) ** 2).sum())
            else:
                var = float(a.var())
            return (var ** 0.5) * np.sqrt(252)

        # ---- research: LEVERAGE POLICIES — fundamentally DIFFERENT ideas of when to be
        # levered, not just different estimators of the same idea. Inverse-vol targeting is
        # ONE theory ("size to constant risk"); if that theory is simply wrong for this
        # strategy, retuning its window can never fix it. Each policy returns a multiplier in
        # [0.3, vol_scale_cap]; every one is compared against a CONSTANT-leverage arm matched
        # on realised average gross, because any policy that merely lowers average exposure
        # will look good against 1.49x for reasons that have nothing to do with timing.
        shadow_hist = []          # shadow (unlevered) NAV path — self-referential policies
        def _policy_mult():
            pol = config.get("lev_policy", "invvol")
            cap_ = config.get("vol_scale_cap", 1.5)
            clamp = lambda x: float(min(cap_, max(0.3, x)))

            if pol == "constant":
                # CONTROL: no timing at all. The bar every policy must clear.
                return 1.0

            if pol == "eqtrend":
                # THEORY: the strategy's own equity curve trends. Be levered while the shadow
                # NAV is above its own moving average, cut when below. Self-referential, so it
                # reacts to the strategy's actual P&L rather than to market vol — which is the
                # complaint about inverse-vol: a 40d vol window stays elevated long after the
                # book has resumed making money.
                n = int(config.get("eq_ma", 50))
                if len(shadow_hist) < n + 1:
                    return None
                ma = float(np.mean(shadow_hist[-n:]))
                lo = float(config.get("eq_low", 0.60))
                return clamp(1.0 if shadow_hist[-1] > ma else lo)

            if pol == "ddstate":
                # THEORY: risk of ruin is path-dependent, so lever off DISTANCE FROM PEAK
                # rather than volatility. Full size at highs, taper linearly into drawdown.
                if len(shadow_hist) < 20:
                    return None
                pk = max(shadow_hist)
                dd = (shadow_hist[-1] - pk) / pk if pk > 0 else 0.0
                tol = float(config.get("dd_tol", 0.25))
                return clamp(1.0 + dd / tol)          # dd is negative -> scales down

            if pol == "volofvol":
                # THEORY: what hurts is UNSTABLE vol, not high vol. A calm-but-high-vol regime
                # is survivable; a regime where vol itself is jumping is where gaps happen.
                w = int(config.get("vov_win", 20))
                if len(recent_rets) < w * 2:
                    return None
                a = np.asarray(recent_rets, dtype=float)
                roll = [float(a[i - w:i].std()) for i in range(w, len(a) + 1)]
                if len(roll) < 10:
                    return None
                vov = float(np.std(roll[-w:])) if len(roll) >= w else float(np.std(roll))
                base = float(np.mean(roll[-w:])) if len(roll) >= w else float(np.mean(roll))
                if base <= 0:
                    return None
                ratio = vov / base                     # coefficient of variation of vol
                thr = float(config.get("vov_thr", 0.35))
                return clamp(1.0 if ratio < thr else float(config.get("vov_low", 0.6)))

            if pol == "recovery":
                # THEORY: the cost is being LATE to re-lever. After a de-lever, ramp exposure
                # back on a fixed schedule instead of waiting for a trailing vol window to
                # decay. Deliberately ignores whether vol has actually fallen.
                if len(shadow_hist) < 20:
                    return None
                pk = max(shadow_hist)
                dd = (shadow_hist[-1] - pk) / pk if pk > 0 else 0.0
                trig = float(config.get("rec_trigger", -0.10))
                if dd <= trig:
                    return clamp(float(config.get("rec_low", 0.5)))
                ramp = int(config.get("rec_ramp", 20))
                since = 0
                for v in reversed(shadow_hist):
                    if (v - pk) / pk <= trig:
                        break
                    since += 1
                return clamp(min(1.0, 0.5 + 0.5 * since / max(ramp, 1)))

            return "invvol"

        def _vol_scale_now():
            """vol_scale under the configured estimator. None => leave leverage unchanged."""
            if not vol_scaling or len(recent_rets) < 20:
                return None
            _pm = _policy_mult()
            if _pm != "invvol":
                return _pm                       # policy answered (possibly None = hold)
            if exante and _ante["w"]:
                _ev = _exante_vol(_ante["date"], _ante["w"])
                if _ev and _ev > 0:
                    if exante == "blend":
                        _rv0 = _rv(recent_rets, vol_lookback)
                        if _rv0:
                            _ev = 0.5 * _ev + 0.5 * _rv0
                    return min(config.get("vol_scale_cap", 1.5), max(0.3, vol_target / _ev))
            if vol_up_lb and vol_dn_lb:
                # ASYMMETRIC: compare a short window to a long one to detect the DIRECTION of
                # vol, then estimate on the short window when vol is falling (re-lever
                # promptly) and the long window when rising (de-lever conservatively).
                short = _rv(recent_rets, min(vol_up_lb, vol_dn_lb))
                long_ = _rv(recent_rets, max(vol_up_lb, vol_dn_lb))
                if short is None or long_ is None:
                    return None
                rv = short if short < long_ else long_
            else:
                rv = _rv(recent_rets, vol_lookback)
            if not rv or rv <= 0:
                return None
            return min(config.get("vol_scale_cap", 1.5), max(0.3, vol_target / rv))
        # --------------------------------------------------------------------------------

        cash = capital
        holdings = {}
        port_values = []
        last_targets = {}
        sig_exit_every = config.get("signal_exit_every")   # research: mid-cycle exit check cadence (days)
        sig_exit_grace = config.get("signal_exit_grace")   # tolerate mom top-K membership (None = top_n)
        # research: DE-RISK PARKING — when the leverage target < 1.0x NAV, park the idle
        # fraction in a bond ETF (total-return series) instead of 0%-cash. Sold when the
        # target returns to >= 1.0x. Trades only at rebalance (cadence-consistent).
        park_etf = config.get("park_etf")                  # "TLT" | "IEF" | "SHY" | None
        park_px_ser = None
        if park_etf:
            _b = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "research", "_bond_etf_tr.parquet"))[park_etf].dropna()
            _idx = pd.DatetimeIndex(trading_dates)
            park_px_ser = _b.reindex(_idx.union(_b.index)).sort_index().ffill().reindex(_idx)
        park_sh = 0.0
        self._park_days = 0

        def park_val(d):
            if park_sh <= 0 or park_px_ser is None:
                return 0.0
            v = park_px_ser.get(d)
            return park_sh * float(v) if pd.notna(v) else 0.0
        self._sigexit_count = 0
        self._levadj_count = 0        # research: off-cadence leverage re-scales performed
        last_derisk = 1.0             # last credit-gate multiplier applied at a rebalance
        self._brk_seen = self._brk_dropped = self._brk_bind_dates = self._brk_reb = 0
        self._fin_paid = 0.0
        self._gross_path = []   # (date, realized gross / NAV) diagnostic
        prev_sh = {}; prev_px = {}; prev_gross = 0.0   # UNLEVERED vol-signal tracking
        shadow_nav = capital; shadow_gross_val = 0.0   # unlevered-book NAV (vol-scale feedback, as live)

        def shr(dollars, px):
            n = dollars / px
            return float(np.floor(n)) if integer_shares else n

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

            # ---- vol signal = UNLEVERED-book NAV return (leverage-independent, with the
            #      vol-scale feedback equilibrium live uses: vol-scale the strategy, THEN
            #      leverage). Matches the validated overlay's avg gross ~1.19 at 1.49x. ----
            if prev_gross > 0:
                pdr = sum(prev_sh[s] * (today.get(s, prev_px[s]) - prev_px[s]) for s in prev_sh) / prev_gross
                shadow_pnl = shadow_gross_val * pdr
                sret = shadow_pnl / shadow_nav if shadow_nav > 0 else 0.0
                recent_rets.append(sret)
                if len(recent_rets) > vol_lookback:
                    recent_rets.pop(0)
                shadow_nav += shadow_pnl
                shadow_hist.append(shadow_nav)
                shadow_gross_val *= (1 + pdr)

            # trailing stop (identical to live)
            # research (I-05): VOL-NORMALISED STOP. A flat 40% stop is a ~2-sigma annual event
            # on a 20%-vol name and ordinary noise on a 60%-vol one, so it applies a DIFFERENT
            # confidence level to every holding -- firing on high-vol names for no informational
            # reason and never protecting low-vol ones. stop_mode="vol" sets the threshold to
            # clamp(stop_k * name vol, stop_lo, stop_hi) using the panel's causal vol feature.
            # Unset => flat `trailing_stop`, bit-for-bit unchanged.
            _stop_vm = (self.uni.get_feature_map(date, stop_feat)
                        if (trailing_stop and stop_mode == "vol") else None)
            if trailing_stop:
                for sym in list(holdings):
                    px = today.get(sym)
                    if px:
                        _thr = abs(trailing_stop)
                        if _stop_vm is not None:
                            _nv = _stop_vm.get(sym)
                            if _nv is not None and np.isfinite(_nv) and _nv > 0:
                                _thr = min(stop_hi, max(stop_lo, stop_k * float(_nv)))
                        if "peak_px" not in holdings[sym]:
                            holdings[sym]["peak_px"] = px
                        if px > holdings[sym]["peak_px"]:
                            holdings[sym]["peak_px"] = px
                        dd = (px - holdings[sym]["peak_px"]) / holdings[sym]["peak_px"]
                        if dd < -_thr:
                            cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                            del holdings[sym]

            # ---- LIVEMIRROR: daily financing on the margin debit ----
            _fr = float(_fin_ser.get(date, fin_rate)) if _fin_ser is not None else fin_rate
            if _fr and cash < 0:
                chg = (-cash) * (_fr / 252.0)
                cash -= chg; self._fin_paid += chg

            nav = cash + sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in holdings.items()) + park_val(date)

            def _snapshot():
                prev_sh.clear(); prev_px.clear()
                g = 0.0
                for s, h in holdings.items():
                    px = today.get(s, h["entry_px"])
                    prev_sh[s] = h["shares"]; prev_px[s] = px; g += h["shares"] * px
                return g

            if rebal_stress:
                _cad = rebal_stress if _gate_on(date) else rebal_days
                _is_reb = _dsr >= _cad
            else:
                _is_reb = (day_idx >= _ph and (day_idx - _ph) % rebal_days == 0)  # _ph=0 => unchanged
            _didx = day_idx - _ph                          # what the sleeves' own gates see
            if not _is_reb:
                _dsr += 1
                # ---- mid-cycle SIGNAL-EXIT (research): sell a holding when NO sleeve
                # currently wants it (user idea: don't wait out the 20d cadence). Cash
                # waits until the next rebalance (recycle tested-dead). grace = tolerate
                # names still inside the mom top-K even if outside the top-5 picks. ----
                if sig_exit_every and holdings and day_idx % sig_exit_every == 0 and day_idx > 0:
                    K = sig_exit_grace if sig_exit_grace else top_n
                    m1 = strategy1_momentum_reversal(date, self.uni, 0, top_n=K,
                                                     rebal_days=rebal_days) or {}
                    mem = self.uni.get_sp500(date)
                    mv = strategy_value(self.uni, date, mem, top_n=10) or {}
                    m3 = strategy3_sector_rotation(date, self.uni, 0) or {}
                    m5 = strategy5_lowvol_quality(date, self.uni, 0) or {}
                    wanted = set(m1) | set(mv) | set(m3) | set(m5)
                    for sym in list(holdings):
                        if sym not in wanted:
                            px = today.get(sym, holdings[sym]["entry_px"])
                            cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                            del holdings[sym]
                            self._sigexit_count += 1
                    nav = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                     for s, h in holdings.items()) + park_val(date)

                # ---- research: OFF-CADENCE LEVERAGE RE-SCALE (APPLICATION-lag test) ----
                # Deployed, vol_scale is applied ONLY on rebalance days, so a vol collapse the
                # day after a rebalance leaves the book de-levered for up to `rebal_days` more
                # sessions. This re-scales EVERY position by a single factor to hit the current
                # vol target — no name changes, just leverage, which is what a live engine
                # could actually do between rebalances. Full transaction costs are charged and
                # shares stay integral, so any benefit has to survive the turnover it creates.
                # `lev_band` suppresses churn on trivial adjustments.
                if (lev_recheck and holdings and day_idx > 0
                        and day_idx % lev_recheck == 0):
                    if exante:
                        _ante["w"] = {s_: h_["shares"] * (today.get(s_) or h_["entry_px"])
                                      for s_, h_ in holdings.items()}
                        _ante["date"] = date
                    _vs = _vol_scale_now()
                    # The CREDIT GATE is also rebalance-only in the deployed engine, so in a
                    # crisis it can act up to `rebal_days` LATE — and credit spreads blow out
                    # in days, so that is precisely when lateness is most expensive. With
                    # gate_offcadence the gate is re-evaluated here too, on the same causal
                    # expanding-percentile maps (already shifted t+1, so no look-ahead).
                    _dr = last_derisk
                    if config.get("gate_offcadence"):
                        if gate_cols:
                            _dr = gate_derisk if any(
                                gate_maps[c].get(date, 0.5) >= gate_pct for c in gate_cols) else 1.0
                        elif credit_pct:
                            _dr = credit_derisk if crmap.get(date, 0.5) >= credit_pct else 1.0
                        if _vs is None:
                            _vs = 1.0        # gate can act even when vol-scaling is idle
                    if _vs is not None:
                        _cur = sum(h["shares"] * today.get(s, h["entry_px"])
                                   for s, h in holdings.items())
                        _tgt = nav * leverage * _vs * _dr
                        if _cur > 0 and abs(_tgt / _cur - 1.0) > lev_band:
                            _k = _tgt / _cur
                            for _s, _h in list(holdings.items()):
                                _px = today.get(_s)
                                if not _px or _px <= 0:
                                    continue
                                _tsh = shr(_h["shares"] * _px * _k, _px)
                                _d = _tsh - _h["shares"]
                                if abs(_d * _px) < max(nav * 0.003, 1e-9):
                                    continue
                                _cost = abs(_d * _px) * cost_frac
                                cash -= _d * _px + _cost
                                _h["shares"] = _tsh
                                if _h["shares"] < (1 if integer_shares else 0.01):
                                    del holdings[_s]
                            self._levadj_count += 1
                            nav = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                             for s, h in holdings.items()) + park_val(date)

                prev_gross = _snapshot()
                port_values.append((date, nav))
                continue
            _dsr = 1        # reset counts the rebalance day itself; 0 fires one day late

            # short interest ranks (identical)
            if self._si_months:
                midx = np.searchsorted(self._si_months, date, side="right") - 1
                if midx >= 0:
                    self.uni._short_interest_rank = self._si_ranks_by_month.get(self._si_months[midx], {})
                    self.uni._si_change_rank = self._si_change_ranks_by_month.get(self._si_months[midx], {})
                else:
                    self.uni._short_interest_rank = {}; self.uni._si_change_rank = {}

            # ---- strategy signals (EXACT production code) ----
            mom_qual = config.get("mom_quality_filter")   # e.g. "gp_assets"/"roe": keep top-N by quality
            mom_rescore = config.get("mom_rescore")       # research I-09/I-10: alternative ORDERING
            sector_cap = config.get("sector_cap")         # max names per 2-digit SIC among top_n
            mom_brk = config.get("mom_broken")            # research: narrow broken-company EXCLUSION
            pool = config.get("mom_pool", config.get("mom_quality_pool", 2.0))
            _tn = (int(round(top_n * pool)) if (mom_qual or sector_cap or mom_brk or mom_rescore)
                   else top_n)
            # ---- F1 AUDIT REPLICA: live mom/s5 sleeves select from SP500-only membership
            #      (PIT via self.sp500_mem); value/breadth stay SP1500. mom_pool_sp500=True ----
            _restore_pool = None
            if config.get("mom_pool_sp500"):
                _restore_pool = self.uni.get_sp500
                _mem = self.sp500_mem
                _keys = sorted(_mem.keys())
                def _sp500_pit(d, __m=_mem, __k=_keys):
                    if d in __m:
                        return set(__m[d])
                    import bisect
                    i = bisect.bisect_right(__k, d) - 1
                    return set(__m[__k[i]]) if i >= 0 else set()
                self.uni.get_sp500 = _sp500_pit
            t1 = strategy1_momentum_reversal(date, self.uni, _didx, top_n=_tn, rebal_days=rebal_days)
            if t1 is None:
                t1 = last_targets.get("mom", {})
            elif mom_brk:
                if len(t1) > top_n:
                    t1 = self._apply_broken_filter(t1, date, top_n, mom_brk)
                self._brk_reb += 1
            elif mom_rescore and len(t1) > top_n:
                _sc = self._rescore(date, list(t1.keys()), mom_rescore)
                if _sc:
                    _rk = sorted(_sc, key=_sc.get, reverse=True)[:top_n]
                    # WEIGHTING MUST MATCH THE SELECTOR. Inheriting t1's weights would size the
                    # variant's picks by the ORIGINAL momentum score -- and when the pool is
                    # wide, a name chosen by e.g. 52w-high proximity can carry a near-zero
                    # momentum score, so it would be selected and then given ~no capital. The
                    # book silently collapses to 3-4 names and the variant is tested unfairly.
                    # Default is EQUAL weight; "orig" reproduces the earlier (unfair) behaviour.
                    if config.get("mom_rescore_weight", "equal") == "orig":
                        _tt = sum(t1[s] for s in _rk)
                        t1 = ({s: t1[s] / _tt for s in _rk} if _tt > 0
                              else {s: 1.0 / len(_rk) for s in _rk})
                    else:
                        t1 = {s: 1.0 / len(_rk) for s in _rk}
                else:
                    t1 = dict(sorted(t1.items(), key=lambda kv: kv[1], reverse=True)[:top_n])
                    _tt = sum(t1.values())
                    t1 = {s: w / _tt for s, w in t1.items()} if _tt > 0 else t1
            elif mom_qual and len(t1) > top_n:
                qmap = self.uni.get_feature_map(date, mom_qual)
                ranked = sorted(t1.keys(),
                                key=lambda s: (qmap.get(s) if qmap.get(s) is not None else -1e18), reverse=True)
                keep = ranked[:top_n]
                tot = sum(t1[s] for s in keep)
                t1 = {s: t1[s] / tot for s in keep} if tot > 0 else {s: t1[s] for s in keep}
            elif sector_cap and len(t1) > top_n:
                smap = self._sector_map()
                ranked = sorted(t1, key=t1.get, reverse=True)
                picked, cnt = [], {}
                for s in ranked:
                    sec = smap.get(s, -1)
                    if cnt.get(sec, 0) < sector_cap:
                        picked.append(s); cnt[sec] = cnt.get(sec, 0) + 1
                    if len(picked) >= top_n:
                        break
                for s in ranked:                      # backfill if cap left us short
                    if len(picked) >= top_n:
                        break
                    if s not in picked:
                        picked.append(s)
                tot = sum(t1[s] for s in picked)
                t1 = {s: t1[s] / tot for s in picked} if tot > 0 else {s: t1[s] for s in picked}
            t5 = strategy5_lowvol_quality(date, self.uni, _didx)   # s5 under the (possibly SP500) pool
            if _restore_pool is not None:
                self.uni.get_sp500 = _restore_pool                   # value/breadth back to SP1500
            members = self.uni.get_sp500(date)
            t_val = strategy_value(self.uni, date, members, top_n=10)
            t3 = strategy3_sector_rotation(date, self.uni, _didx)
            if t3 is None:
                t3 = last_targets.get("s3", {})
            if t5 is None:
                t5 = last_targets.get("s5", {})
            last_targets.update({"mom": t1, "val": t_val, "s3": t3, "s5": t5})

            nu = self.umd_20d.loc[:date]
            in_crash = len(nu) > 0 and pd.notna(nu.iloc[-1]) and nu.iloc[-1] < UMD_CRASH_THRESHOLD
            # research hook: crash_weights override (used by the s3-ETF live-parity A/B —
            # live DROPS sector-ETF picks because the served signal list only emits SP1500
            # members, so its effective crash/bear weights have s3 removed + renormalized).
            _cw = config.get("crash_weights")
            ew = (dict(_cw) if _cw else _short_weights(PROD_WEIGHTS_CRASH)) if in_crash \
                else {"mom": mom_w, "val": val_w, "s5": lv_w, "s3": sec_w}

            # ---- vol-managed momentum (Barroso): scale the mom sleeve by target/own-vol ----
            vmm = config.get("vol_managed_mom")   # target annualized vol for the momentum book
            if vmm and t1:
                syms = [s for s in t1 if s in self.prices.columns]
                if len(syms) >= 2:
                    px = self.prices[syms].loc[:date].tail(127)
                    pr = px.pct_change().dropna(how="all")
                    if len(pr) >= 60:
                        port = pr.mean(axis=1)   # equal-weight momentum book proxy
                        rv = port.std() * np.sqrt(252)
                        if rv and rv > 0.01:
                            sc = min(config.get("vmm_cap", 1.0), vmm / rv)
                            ew = dict(ew); ew["mom"] = ew["mom"] * sc

            fdate = self.features_by_date.get(date, {})
            above = sum(1 for fd in fdate.values() if fd.get("dist_sma50", 0) > 0)
            total_f = sum(1 for fd in fdate.values() if "dist_sma50" in fd)
            breadth = above / max(total_f, 1)
            blend = min(1.0, max(0.0, (breadth - 0.35) / 0.25))
            bear = config.get("bear_weights", _short_weights(PROD_WEIGHTS_BEAR))
            blended = {n: ew[n] * blend + bear.get(n, 0) * (1 - blend) for n in ew}

            # ---- STRONG-BULL regime: tilt harder into momentum when breadth is high AND
            #      SPY uptrend AND credit calm (and not in a UMD crash). Multi-state mix. ----
            cr_pct = crmap.get(date, 0.5)
            strong_bull = False
            if (bull_w or bull_lever > 1.0) and not in_crash and breadth >= bull_breadth:
                up = True
                if bull_need_trend and "SPY" in self.prices.columns:
                    sh = self.prices["SPY"].loc[:date].dropna()
                    up = len(sh) >= 200 and sh.iloc[-1] > sh.iloc[-200:].mean()
                calm = (bull_credit_calm is None) or (cr_pct <= bull_credit_calm)
                strong_bull = up and calm
            if strong_bull and bull_w:
                blended = dict(bull_w)

            combined = {}
            for name, cap_pct in blended.items():
                tgt = last_targets.get(name, {})
                if use_rp and name in ("mom", "val"):
                    tgt = self._apply_rp(tgt, date, power=rp_power)
                for sym, w in tgt.items():
                    if w > 0:
                        combined[sym] = combined.get(sym, 0) + w * cap_pct

            # Uses the SHARED estimator so vol_mode / lookback / asymmetry apply identically
            # here and in the off-cadence re-scale. With no research knobs set, _vol_scale_now()
            # is simple 40d vol == the previous inline computation (parity-checked).
            _ante["w"], _ante["date"] = dict(combined), date   # book the ex-ante est. describes
            _vs_reb = _vol_scale_now()
            _vs_applied = 1.0
            if _vs_reb is not None:
                _vs_applied = _vs_reb
                combined = {s: w * _vs_reb for s, w in combined.items()}

            longs = {s: w for s, w in combined.items() if w > 0}
            for sym in list(longs):
                if longs[sym] > cap:
                    longs[sym] = cap
            gross = sum(longs.values())
            if gross > 1.0:
                for sym in longs:
                    longs[sym] /= gross
            combined = {s: w for s, w in longs.items() if w >= 0.005}
            shadow_gross_val = sum(combined.values()) * shadow_nav   # redeploy unlevered shadow

            # ---- LIVEMIRROR: leverage * de-risk gate (down) * bull-lever (up) ----
            if gate_cols:
                off = any(gate_maps[c].get(date, 0.5) >= gate_pct for c in gate_cols)
                derisk = gate_derisk if off else 1.0
            else:
                derisk = credit_derisk if (credit_pct and cr_pct >= credit_pct) else 1.0
            last_derisk = derisk        # off-cadence re-scale must not undo the credit gate
            lev_t = leverage * derisk
            if strong_bull and bull_lever > 1.0:
                lev_t *= bull_lever
            if config.get("live_sizing"):
                # ---- FAITHFUL PORT of ibkr_engine._calibrate_quantities ----
                # The default branch below multiplies `combined` by the budget AS-IS, so any
                # under-allocation (sleeve returning few names, per-position cap binding,
                # integer-rounding drag, cash drag) is PRESERVED — measured avg realized gross
                # 1.1164x at nominal 1.49x. Live does not do that: it renormalises
                # w = prob/total_prob so the weights sum to exactly 1, then closed-loops a
                # multiplier over 4 passes until ACTUAL gross (after cap + whole-share
                # truncation) hits nav * leverage * vol_scale * credit_derisk. It therefore
                # deploys the full target every time and recovers the rounding drag.
                # Net effect measured 2026-08-11: live runs 1.33x the exposure the backtest
                # actually held (+33.5%). live_sizing=True reproduces the LIVE behaviour so the
                # two can be compared at equal footing instead of assumed equivalent.
                _cap = float(config.get("live_position_cap", 0.15))   # ibkr POSITION_CAP
                _ceil = float(config.get("live_lev_ceiling", 1.80))   # ibkr LEVERAGE ceiling
                _tot_w = sum(combined.values())
                _base = ({s: w / _tot_w for s, w in combined.items()} if _tot_w > 0
                         else dict(combined))
                # vol_scale MUST enter through the multiplier here. It was applied to
                # `combined` above, but renormalising to sum-1 divides it out again — live
                # does target_gross = nav * LEV * vol_scale * derisk, i.e. it scales the
                # MULTIPLIER, never the weights. Without this the whole vol overlay is
                # silently disabled under live_sizing (caught 2026-08-13: every policy in
                # threadDYN realised avgGross ~1.49 regardless of vol).
                _tgt_gross = nav * lev_t * _vs_applied
                _m = lev_t * _vs_applied
                target_d = {}
                for _p in range(4):
                    target_d, _g = {}, 0.0
                    for _s, _w0 in _base.items():
                        _px = today.get(_s)
                        if not _px or _px <= 0:
                            continue
                        _w = min(_w0 * _m, _cap)
                        _q = shr(nav * _w, _px)
                        if _q > 0:
                            target_d[_s] = _q * _px
                            _g += _q * _px
                    if _p == 3 or _g <= 0:
                        break
                    _m = min(_m * (_tgt_gross / _g), _ceil)
            else:
                budget = nav * lev_t
                target_d = {s: w * budget for s, w in combined.items()}

            # ---- research (I-23): PARTIAL-ADJUSTMENT REBALANCE --------------------------
            # The deployable approximation of phase tranching (EXP-001). Rather than K
            # sub-books on K phases -- which needs new live code to net K target books inside
            # one IBKR account -- keep ONE book, rebalance more often, and move only
            # `move_frac` of the way to each new target. Same intent: no single arbitrary
            # session determines the book. It is NOT mathematically the same as tranching
            # (it smooths toward a MOVING target instead of averaging independent sub-books),
            # so it has to be measured, never assumed equivalent.
            # Names dropped from the target must stay in target_d at their residual value,
            # otherwise the exit loop below liquidates them in FULL and the partial move is
            # applied to buys only -- which would look like a free lunch.
            # move_frac = 1.0 is the untouched path, bit-for-bit.
            _mf = float(config.get("move_frac", 1.0))
            if _mf < 1.0:
                _cur_val = {s: h["shares"] * (today.get(s) or h["entry_px"])
                            for s, h in holdings.items()}
                _blend = {}
                for _s in set(target_d) | set(_cur_val):
                    _c0 = _cur_val.get(_s, 0.0)
                    _t0 = target_d.get(_s, 0.0)
                    _v = _c0 + _mf * (_t0 - _c0)
                    if _v > 0:
                        _blend[_s] = _v
                target_d = _blend

            # exit names no longer targeted (integer -> sell whole position)
            for sym in list(holdings):
                if sym not in target_d:
                    px = today.get(sym, holdings[sym]["entry_px"])
                    cash += holdings[sym]["shares"] * px * (1 - cost_frac)
                    del holdings[sym]

            # rebalance to integer share targets (margin allowed; no cash>=delta block)
            for sym, tgt in target_d.items():
                px = today.get(sym)
                if not px or px <= 0:
                    continue
                cur_sh = holdings[sym]["shares"] if sym in holdings else 0.0
                tgt_sh = shr(tgt, px)
                d_sh = tgt_sh - cur_sh
                d_val = d_sh * px
                if abs(d_val) < max(nav * 0.003, 1e-9):
                    continue
                cost = abs(d_val) * cost_frac
                if d_sh > 0:
                    cash -= d_val + cost
                    if sym in holdings:
                        holdings[sym]["shares"] += d_sh
                    else:
                        holdings[sym] = {"shares": d_sh, "entry_px": px, "peak_px": px}
                else:
                    cash += (-d_val) - cost
                    holdings[sym]["shares"] += d_sh
                    if holdings[sym]["shares"] < (1 if integer_shares else 0.01):
                        cash += 0
                        del holdings[sym]

            # ---- research: DE-RISK PARKING trade (rebalance-cadence only) ----
            if park_px_ser is not None:
                ppx = park_px_ser.get(date)
                if pd.notna(ppx) and ppx > 0:
                    nav_now = cash + sum(h["shares"] * today.get(s, h["entry_px"])
                                         for s, h in holdings.items()) + park_sh * float(ppx)
                    park_target = max(0.0, nav_now * (1.0 - lev_t))   # idle fraction when de-levered
                    delta = park_target - park_sh * float(ppx)
                    if abs(delta) > max(nav_now * 0.003, 1e-9):
                        if delta > 0:
                            spend = min(delta, max(0.0, cash - 1.0))   # never borrow to park
                            q = float(np.floor(spend / ppx)) if integer_shares else spend / ppx
                            if q > 0:
                                cash -= q * float(ppx) * (1 + cost_frac)
                                park_sh += q
                        else:
                            q = min(park_sh, float(np.ceil(-delta / ppx)) if integer_shares else -delta / ppx)
                            if q > 0:
                                cash += q * float(ppx) * (1 - cost_frac)
                                park_sh -= q
                    if park_sh > 0:
                        self._park_days += 1

            nav = cash + sum(h["shares"] * today.get(s, h["entry_px"]) for s, h in holdings.items()) + park_val(date)
            prev_gross = _snapshot()   # end-of-rebal snapshot for tomorrow's vol signal
            self._gross_path.append((date, prev_gross / nav if nav > 0 else 0))
            port_values.append((date, max(nav, 0)))

        vals = pd.Series([v for _, v in port_values],
                         index=pd.DatetimeIndex([d for d, _ in port_values]))
        years = max((vals.index[-1] - vals.index[0]).days / 365.25, 1)
        dr = vals.pct_change().dropna()
        cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
        sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        peak = vals.cummax(); max_dd = ((vals - peak) / peak).min()
        gp = pd.Series([g for _, g in self._gross_path])
        return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd,
                "vol": dr.std() * np.sqrt(252), "final": vals.iloc[-1],
                "fin_paid": self._fin_paid, "avg_gross": float(gp.mean()) if len(gp) else 0,
                "levadj": self._levadj_count,
                "brk_seen": self._brk_seen, "brk_dropped": self._brk_dropped,
                "brk_bind_dates": self._brk_bind_dates, "brk_reb": self._brk_reb,
                "daily_values": vals}
