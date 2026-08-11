"""thread GM3 — is the value sleeve's absurd-gm "edge" a signal or a concentration bet?

GM2 showed the gm bound costs -0.91pp (8yr) / -0.32pp (26yr) via the VALUE sleeve, negative
on all 5 starts. Before accepting that as a real edge being removed, establish what the
corrupt names actually DO inside the sleeve.

The mechanism matters: gm carries weight 0.25 in the value score, so a gm of 4561 produces a
score ~2,400x the best legitimate name. The sleeve's 2/N cap does not contain that (it is
renormalized away — see the strategy_value docstring), so ONE corrupt name can take ~93% of
the sleeve. If that is what happens, arm A's value sleeve is not a 10-name value portfolio,
it is a 1-name bet on whichever company had a broken revenue denominator that month, and its
+0.9pp is one realized draw from a very wide distribution, not a repeatable edge.

Measures, per rebalance date, both with and without the bound:
  - how often >=1 absurd-gm name is in the value top-10
  - the weight share those names capture
  - the effective number of positions (1/sum w^2) -> concentration
  - how many DISTINCT corrupt names drive the whole period
  - forward 20d return of corrupt picks vs the legitimate names they displaced

Run:  python3 research/threadGM3_value_census.py
"""
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import strategies.multi_strategy_engine as M  # noqa: E402
from main_production_backtest import FastBacktester  # noqa: E402

PERIODS = [
    ("8yr 2018-25", "data/wrds/complete_sp1500_universe.pkl", "2018-01-02", "2025-12-31"),
    ("26yr 2001-25", "data/wrds/sp1500_universe_2000.pkl", "2001-01-02", "2025-12-31"),
]
_SHIPPED = M._sane_gross_margin
BOUND = {"on": True}


def _guard(gm_map):
    return _SHIPPED(gm_map) if BOUND["on"] else gm_map


M._sane_gross_margin = _guard


def eff_n(w):
    """Effective number of positions = 1/sum(w^2). 10 equal names -> 10; one name -> 1."""
    if not w:
        return 0.0
    a = np.array(list(w.values()), dtype=float)
    s = a.sum()
    if s <= 0:
        return 0.0
    a = a / s
    return float(1.0 / np.sum(a ** 2))


def main():
    for pname, path, start, end in PERIODS:
        print("\n" + "=" * 96, flush=True)
        print(f"{pname} | VALUE-sleeve absurd-gm census", flush=True)
        print("=" * 96, flush=True)
        t0 = time.time()
        bt = FastBacktester(universe_path=path)
        uni = bt.uni
        print(f"(loaded {time.time() - t0:.0f}s)", flush=True)

        dates = [d for d in sorted(uni._feat_by_date) if str(start) <= str(d)[:10] <= str(end)]
        rebals = dates[::20]
        px = uni.prices

        n_dirty_dates = 0
        share_hist, effn_dirty, effn_clean = [], [], []
        names = Counter()
        name_w = defaultdict(list)
        fwd_dirty, fwd_displaced = [], []

        for i, d in enumerate(rebals):
            try:
                members = list(uni.get_sp500(d))
            except Exception:
                continue
            if not members:
                continue
            gm_raw = uni.get_feature_map(d, "gross_margin", members)
            absurd = {s for s, v in gm_raw.items()
                      if v is not None and not np.isnan(v) and not (-1.0 <= v <= 1.0)}
            if not absurd:
                continue

            BOUND["on"] = False
            w_dirty = M.strategy_value(uni, d, members, top_n=10) or {}
            BOUND["on"] = True
            w_clean = M.strategy_value(uni, d, members, top_n=10) or {}
            if not w_dirty or not w_clean:
                continue

            hit = {s: w for s, w in w_dirty.items() if s in absurd}
            if hit:
                n_dirty_dates += 1
                share_hist.append(sum(hit.values()))
                effn_dirty.append(eff_n(w_dirty))
                effn_clean.append(eff_n(w_clean))
                for s, w in hit.items():
                    names[s] += 1
                    name_w[s].append(w)

                # forward 20d: corrupt picks vs the legitimate names they displaced
                if i + 1 < len(rebals):
                    d2 = rebals[i + 1]
                    def fwd(sym):
                        try:
                            a, b = px.loc[d, sym], px.loc[d2, sym]
                            if a and b and a > 0 and not (np.isnan(a) or np.isnan(b)):
                                return b / a - 1
                        except Exception:
                            pass
                        return None
                    for s in hit:
                        r = fwd(s)
                        if r is not None:
                            fwd_dirty.append(r)
                    for s in set(w_clean) - set(w_dirty):
                        r = fwd(s)
                        if r is not None:
                            fwd_displaced.append(r)

        nr = len(rebals)
        print(f"\n  rebalance dates examined      : {nr}")
        print(f"  dates where a corrupt name is IN the value top-10: {n_dirty_dates} "
              f"({n_dirty_dates / max(nr,1):.1%})")
        if share_hist:
            sh = np.array(share_hist)
            print(f"\n  WEIGHT SHARE captured by corrupt names on those dates:")
            print(f"    mean {sh.mean():.1%} | median {np.median(sh):.1%} | "
                  f"p90 {np.percentile(sh,90):.1%} | max {sh.max():.1%}")
            print(f"    dates where they took >50% of the sleeve: "
                  f"{int((sh>0.5).sum())} ({(sh>0.5).mean():.1%})")
            print(f"    dates where they took >90% of the sleeve: "
                  f"{int((sh>0.9).sum())} ({(sh>0.9).mean():.1%})")
            print(f"\n  CONCENTRATION (effective # positions, 10 = fully diversified):")
            print(f"    with corrupt names : {np.mean(effn_dirty):.2f}")
            print(f"    with bound applied : {np.mean(effn_clean):.2f}")
            print(f"\n  DISTINCT corrupt names driving the entire period: {len(names)}")
            print(f"    top by appearances:")
            for s, c in names.most_common(10):
                print(f"      {s:<8} {c:>4} dates   avg weight {np.mean(name_w[s]):.1%}")
            top3 = sum(c for _, c in names.most_common(3))
            tot = sum(names.values())
            print(f"    top-3 names = {top3}/{tot} ({top3/max(tot,1):.0%}) of all corrupt slots")
        if fwd_dirty and fwd_displaced:
            fd, fp = np.array(fwd_dirty), np.array(fwd_displaced)
            print(f"\n  FORWARD 20d RETURN:")
            print(f"    corrupt picks     n={len(fd):<5} mean {fd.mean():+.2%}  "
                  f"median {np.median(fd):+.2%}  std {fd.std():.2%}")
            print(f"    displaced legit   n={len(fp):<5} mean {fp.mean():+.2%}  "
                  f"median {np.median(fp):+.2%}  std {fp.std():.2%}")
            print(f"    edge (corrupt - legit): {(fd.mean()-fp.mean())*100:+.2f}pp per 20d, "
                  f"but corrupt-pick vol is {fd.std()/max(fp.std(),1e-9):.1f}x the legit names")
        del bt
    BOUND["on"] = True


if __name__ == "__main__":
    main()
