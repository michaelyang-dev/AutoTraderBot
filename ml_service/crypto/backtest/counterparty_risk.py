"""
Counterparty / exchange-insolvency risk — what you can ACTUALLY do about it, quantified.

It can't be eliminated, but a stack of concrete mitigations turns 'lose everything' into 'lose a
bounded, survivable, partly-insurable amount'. Each layer reduces the tail; we show the tail-aware
Sharpe and effective CAGR after each.

Mitigation stack:
  0. naive            all capital on HL                      -> insolvency = total loss
  1. structure        SPOT leg (bigger) on regulated CEX /   -> only the perp MARGIN sits on HL
                      self-custody; only margin on HL
  2. venue-split x2   short legs split HL + dYdX             -> lose only the failed venue's half
  3. venue-split x3   HL + dYdX + a 3rd                      -> lose only a third
  4. + insurance      on-chain cover (e.g. Nexus Mutual) on  -> tail capped at the premium cost
                      the residual HL exposure
Run:  python crypto/backtest/counterparty_risk.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from improved_carry import load
from carry_optimize import carry as carry_opt

MARGIN_PER_NOTIONAL = 0.45        # HL margin to survive a ~+40% squeeze (from risk_controls)


def stats_with_tail(d, counterparty_loss, carry_haircut=0.0):
    """inject one counterparty event (loss = fraction of capital) and an annual carry haircut (insurance)."""
    x = d.dropna().copy() * (1 - carry_haircut)        # insurance premium reduces carry
    nav = (1 + x).cumprod(); yrs = (x.index[-1] - x.index[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    xt = x.copy(); xt.loc[nav.idxmax()] -= counterparty_loss
    nav2 = (1 + xt).cumprod()
    return cagr, xt.mean() / xt.std() * np.sqrt(365), ((nav2 - nav2.cummax()) / nav2.cummax()).min()


if __name__ == "__main__":
    F, P = load()
    d, _ = carry_opt(F, P, top_k=8, rebal=30, weighting="funding", rt_cost=0.0060, keep_mult=2.0, min_keep=0.02)

    print("=" * 96)
    print("COUNTERPARTY RISK — the mitigation stack, quantified (carry base, 1x)")
    print("=" * 96)
    print(f"  {'mitigation layer':<42}{'HL exposure':>13}{'insolvency tail':>17}{'CAGR':>8}{'tail-Sharpe':>13}")
    # insolvency loss = fraction of capital on the failing venue
    layers = [
        ("0. naive: all capital on HL",            1.00, 0.0),
        ("1. structure: spot off-HL, only margin", MARGIN_PER_NOTIONAL, 0.0),
        ("2. + venue-split x2 (HL + dYdX)",         MARGIN_PER_NOTIONAL / 2, 0.0),
        ("3. + venue-split x3",                     MARGIN_PER_NOTIONAL / 3, 0.0),
        ("4. + insurance on residual (~3%/yr)",     MARGIN_PER_NOTIONAL / 3, 0.03),
    ]
    for lab, hl_exp, ins in layers:
        # with insurance, the tail is capped (you're reimbursed) -> residual ~ a small deductible
        tail = hl_exp if ins == 0 else 0.05
        cg, sh, md = stats_with_tail(d, tail, carry_haircut=ins)
        print(f"  {lab:<42}{hl_exp*100:>11.0f}%{-tail*100:>16.0f}%{cg*100:>7.0f}%{sh:>13.2f}", flush=True)

    print("\n  THE STACK:  -100% naive  ->  -45% (spot off-HL)  ->  -22% (2 venues)  ->  -15% (3 venues)")
    print("              ->  ~-5% deductible if insured (for ~3%/yr premium).")
    print("\n  None of this needs more data — it's pure structure + risk management. The biggest single win")
    print("  is #1: keep the LONG SPOT leg (the bulk of capital) on a regulated CEX or self-custody, and")
    print("  only the small perp MARGIN on Hyperliquid. That alone cuts the tail from -100% to ~-45%.")
