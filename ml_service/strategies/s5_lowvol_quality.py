"""
Strategy 5: Low-Volatility Quality Tilt (15% of capital)
=========================================================
Defensive smart-beta: tilt toward low volatility + high quality.

Signals:
  1. Low volatility: 60-day realized vol (lower is better)
  2. Quality - profitability: gross margin (higher is better)
  3. Quality - leverage: debt-to-equity (lower is better)

Z-scored and averaged. Top 75 by composite, market-cap weighted.
Quarterly rebalance. Always on (no regime overlay).
"""

import numpy as np
import pandas as pd

from strategies.base_strategy import BaseStrategy


class LowVolQuality(BaseStrategy):

    TOP_N = 75

    @property
    def name(self):
        return "lowvol_quality"

    @property
    def capital_pct(self):
        return 0.15

    @property
    def rebalance_days(self):
        return 63  # quarterly

    def compute_targets(self, date, universe):
        members = universe.sp500_members
        if len(members) < 100:
            return {}

        # --- Signal 1: Low volatility (lower is better) ---
        vol_60d = universe.get_sp500_features("vol_60d")
        # Invert so higher z-score = lower vol
        inv_vol = {sym: -v for sym, v in vol_60d.items()
                   if not np.isnan(v) and v > 0}

        # --- Signal 2: Gross margin (higher is better) ---
        gross_margin = universe.get_sp500_features("gross_margin")
        quality_prof = {sym: v for sym, v in gross_margin.items()
                        if not np.isnan(v)}

        # --- Signal 3: Debt-to-equity (lower is better) ---
        dte = universe.get_sp500_features("debt_to_equity")
        # Invert so higher z-score = lower leverage
        inv_dte = {sym: -v for sym, v in dte.items()
                   if not np.isnan(v) and v >= 0}

        # --- Z-score each signal ---
        signals = {
            "inv_vol": inv_vol,
            "quality_prof": quality_prof,
            "inv_dte": inv_dte,
        }

        z_scores = {}
        for sig_name, sig_dict in signals.items():
            if len(sig_dict) < 30:
                continue
            vals = np.array(list(sig_dict.values()))
            mu, sigma = vals.mean(), vals.std()
            if sigma < 1e-10:
                continue
            for sym, v in sig_dict.items():
                if sym not in z_scores:
                    z_scores[sym] = []
                z_scores[sym].append((v - mu) / sigma)

        # --- Composite: average z-scores ---
        composite = {}
        for sym, zs in z_scores.items():
            if len(zs) >= 2:
                composite[sym] = np.mean(zs)

        if not composite:
            return {}

        # --- Select top 75 ---
        sorted_syms = sorted(composite.keys(),
                              key=lambda s: composite[s], reverse=True)
        selected = sorted_syms[:self.TOP_N]

        # --- Market-cap weighting proxy ---
        # Use inverse of vol as a rough cap proxy (larger companies = lower vol)
        # Without actual market cap data, equal-weight is safer
        # But we can use price level as a very rough proxy
        weights = {}
        px_sum = 0
        for sym in selected:
            close = universe.get_close(sym, lookback=1)
            if len(close) > 0:
                px = close.iloc[-1]
                if px > 0 and not np.isnan(px):
                    weights[sym] = px
                    px_sum += px

        if px_sum <= 0:
            # Fallback to equal weight
            w = 1.0 / len(selected)
            return {sym: w for sym in selected}

        # Normalize
        return {sym: px / px_sum for sym, px in weights.items()}
