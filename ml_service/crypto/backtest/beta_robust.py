"""
Robustness of the risk-managed BTC/ETH beta product — is Sharpe ~1.2 broad or knife-edge?
Sweeps regime lookback, vol target, weights, rebalance. A real risk-management benefit is a
broad plateau (the regime filter + vol-target should help across reasonable settings).
Run:  python crypto/backtest/beta_robust.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from beta_product import load, strategy, stats
import crypto.backtest.beta_product as bp

close = load()
print("=" * 84)
print("BETA PRODUCT ROBUSTNESS — base = 60/40, vt30, regime200, rebal5")
print("=" * 84)
print(f"  {'variant':<30}{'FULL':>9}{'2023+':>9}{'Sharpe':>9}{'MaxDD':>9}")


def show(label, **kw):
    w = kw.pop("w", {"BTC": 60, "ETH": 40})
    d = strategy(close, w, vol_target=kw.get("vt", 0.30), regime=kw.get("regime", True),
                 regime_n=kw.get("rn", 200))
    cg, vol, sh, md = stats(d); cg23, _, _, _ = stats(d, "2023-01-01")
    print(f"  {label:<30}{cg*100:>8.0f}%{cg23*100:>8.0f}%{sh:>9.2f}{md*100:>8.1f}%", flush=True)


print("  -- regime lookback --")
for rn in [100, 150, 200, 250, 300]:
    show(f"regime SMA{rn}", rn=rn)
print("  -- vol target --")
for vt in [0.25, 0.30, 0.40, 0.50]:
    show(f"vol-target {int(vt*100)}%", vt=vt)
print("  -- weights --")
for w, nm in [({"BTC": 100}, "100% BTC"), ({"BTC": 70, "ETH": 30}, "70/30"),
              ({"BTC": 50, "ETH": 50}, "50/50")]:
    show(nm, w=w)
print("  -- rebalance cadence --")
for rb in [3, 5, 10, 20]:
    bp.REBAL = rb
    show(f"rebal {rb}d")
bp.REBAL = 5
print("  -- no-regime (vol-target only) for reference --")
show("vt30, NO regime", regime=False)
print("\n  If Sharpe stays ~1.0-1.3 across the board, the risk-management benefit is robust, not fit.")
