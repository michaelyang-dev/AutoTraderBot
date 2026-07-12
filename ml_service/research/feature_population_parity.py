"""
FEATURE-POPULATION PARITY CHECK — the audit divergence #61 taught us to run.

Code parity is not enough: the eps-surprise boost hid because the SHARED sleeve code
consumed a feature that was populated live (IBES) but EMPTY in the backtest pkl —
identical code, silently different inputs. This script enumerates every feature the
sleeves actually consume (get_feature_map calls in multi_strategy_engine.py) and reports
its population rate in the backtest feature store. Any consumed feature near 0% is a
live-vs-backtest data divergence candidate: either populate it in the pkl or verify the
live side leaves it empty too (as done for eps_surprise_last, disabled 2026-07-12).

Run after any pkl rebuild or sleeve change:  python3 research/feature_population_parity.py
"""
import os
import pickle
import re
import sys

os.environ["OMP_NUM_THREADS"] = "1"
ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML)

PKL = os.path.join(ML, "data/wrds/complete_sp1500_universe.pkl")
ENGINE = os.path.join(ML, "strategies/multi_strategy_engine.py")


def main():
    src = open(ENGINE).read()
    consumed = sorted(set(re.findall(r'get_feature_map\([a-z_]+,\s*"([a-z_0-9]+)"', src)))
    print(f"features consumed by sleeves ({len(consumed)}): {consumed}\n")

    d = pickle.load(open(PKL, "rb"))
    fbd = d["features_by_date"]
    dates = sorted(fbd.keys())
    checks = [dates[-1], dates[len(dates) // 2]]
    print(f"{'feature':<22}" + "".join(f"{str(dt.date()):>14}" for dt in checks) + "   verdict")
    problems = []
    for feat in consumed:
        rates = []
        for dt in checks:
            fd = fbd[dt]
            n = len(fd)
            have = sum(1 for v in fd.values()
                       if feat in v and v[feat] == v[feat])   # non-NaN
            rates.append(have / n if n else 0)
        worst = min(rates)
        verdict = "OK" if worst > 0.5 else ("LOW" if worst > 0.05 else "EMPTY — DIVERGENCE RISK")
        if worst <= 0.05:
            problems.append(feat)
        print(f"{feat:<22}" + "".join(f"{r:>13.0%} " for r in rates) + f"  {verdict}")
    print()
    if problems:
        print(f"⚠ EMPTY-in-backtest features consumed by sleeves: {problems}")
        print("  For each: confirm the LIVE builder also leaves it empty (parity), or the")
        print("  live-only behavior is validated. eps_surprise_last resolved 2026-07-12.")
    else:
        print("✓ no empty consumed features — data-population parity holds")


if __name__ == "__main__":
    main()
