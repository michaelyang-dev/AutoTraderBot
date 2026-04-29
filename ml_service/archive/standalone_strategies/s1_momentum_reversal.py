"""
Strategy 1: Momentum + Reversal Ensemble (40% of capital)
=========================================================
Combines four cross-sectional equity factors:
  1. 12-1 momentum (11-month return skipping recent month)
  2. 5-day short-term reversal (negative of recent 5d return)
  3. Earnings revisions momentum (60-day change in analyst EPS estimates / price)
  4. Quality (gross profitability / total assets)

Each signal is z-scored cross-sectionally, then averaged equally.
Top 50 stocks by composite, equal-weighted, 20-day hold.

Regime overlay: reduce to top 25 when VIX > 30 or term structure < 0.95.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from strategies.base_strategy import BaseStrategy


class MomentumReversalEnsemble(BaseStrategy):

    @property
    def name(self):
        return "momentum_reversal_ensemble"

    @property
    def capital_pct(self):
        return 0.40

    @property
    def rebalance_days(self):
        return 20

    def compute_targets(self, date, universe):
        members = universe.sp500_members
        if len(members) < 50:
            return {}

        scores = {}

        # --- Signal 1: 12-1 Momentum ---
        # 252-day return minus 21-day return (skip recent month)
        ret_252 = universe.get_sp500_features("ret_252d")
        ret_21 = {}
        for sym in members:
            r21 = universe.get_feature(sym, "ret_20d")
            if r21 is not None and not np.isnan(r21):
                ret_21[sym] = r21
        mom_12_1 = {}
        for sym in members:
            r252 = ret_252.get(sym)
            r21 = ret_21.get(sym)
            if r252 is not None and r21 is not None and not np.isnan(r252) and not np.isnan(r21):
                mom_12_1[sym] = r252 - r21

        # --- Signal 2: 5-day Short-Term Reversal ---
        ret_5d = universe.get_sp500_features("ret_5d")
        reversal_5d = {sym: -v for sym, v in ret_5d.items() if not np.isnan(v)}

        # --- Signal 3: Earnings Revisions Momentum ---
        # Use eps_surprise_last as proxy (actual FMP estimate changes need
        # historical estimates which we approximate via filing-date-aligned data)
        eps_surprise = universe.get_sp500_features("eps_surprise_last")
        # Normalize by a rough price proxy (use PE as indicator of price level)
        earn_rev = {}
        for sym in members:
            surp = eps_surprise.get(sym)
            if surp is not None and not np.isnan(surp):
                earn_rev[sym] = surp  # already surprise-scaled

        # --- Signal 4: Quality (gross margin as proxy for profitability/assets) ---
        gross_margin = universe.get_sp500_features("gross_margin")
        quality = {sym: v for sym, v in gross_margin.items() if not np.isnan(v)}

        # --- Cross-sectional z-score each signal ---
        signals = {
            "mom_12_1": mom_12_1,
            "reversal_5d": reversal_5d,
            "earn_rev": earn_rev,
            "quality": quality,
        }

        z_scores = {}
        for sig_name, sig_dict in signals.items():
            if len(sig_dict) < 20:
                continue
            vals = np.array(list(sig_dict.values()))
            mu, sigma = vals.mean(), vals.std()
            if sigma < 1e-10:
                continue
            for sym, v in sig_dict.items():
                if sym not in z_scores:
                    z_scores[sym] = []
                z_scores[sym].append((v - mu) / sigma)

        # --- Composite: average of available z-scores ---
        composite = {}
        for sym, zs in z_scores.items():
            if len(zs) >= 2:  # need at least 2 signals
                composite[sym] = np.mean(zs)

        if not composite:
            return {}

        # --- Regime overlay ---
        vix = universe.regime.get("vix", 20)
        vix_ts = universe.regime.get("vix_term_structure", 1.0)
        stress = vix > 30 or vix_ts < 0.95
        top_n = 25 if stress else 50

        # --- Select top N ---
        sorted_syms = sorted(composite.keys(), key=lambda s: composite[s], reverse=True)
        selected = sorted_syms[:top_n]

        weight = 1.0 / len(selected)
        return {sym: weight for sym in selected}
