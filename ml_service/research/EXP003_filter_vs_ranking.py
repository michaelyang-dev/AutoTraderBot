"""EXP-003 (IDEAS I-21) — WHERE DOES THE EDGE ACTUALLY LIVE: THE FILTERS OR THE RANKING?

WHY THIS IS THE MOST IMPORTANT QUESTION IN THE PROGRAM RIGHT NOW
---------------------------------------------------------------
AUDIT02's shuffle test says random picks from the same PIT membership, run through the
identical chassis, return +6.36% / 0.37 over 26 years against the deployed +10.87% / 0.50.
So the whole of "stock selection" is worth ~+4.5pp CAGR / +0.13 Sharpe through a full cycle
(it is worth +20pp / +0.58 on the 8yr, i.e. the recent window flatters it ~4x).

But "selection" is two very different things bolted together:

  FILTER  — dist_sma200 > 0, valid 12-1 momentum, the bear-regime sector exclusion, the value
            sleeve's quality screens. These are cheap, stable, and mechanical.
  RANKING — the composite score that orders the survivors and picks the top 5.

Every remaining hour of this program should go wherever the edge actually is. If the ranking
is worth ~nothing over a random draw from the filtered pool, then years of cross-sectional
scoring work has been rediscovering a trend filter, and the productive direction is risk /
construction / filtering, not better scores. If the ranking carries it, the opposite.

DESIGN
------
  A  DEPLOYED          filter + ranking
  B  FILTER-ONLY       real eligible pool from the real sleeves, then a STICKY RANDOM draw
                       inside it, equal weight
  C  NO-FILTER         sticky random draw from raw PIT membership

     ranking value = A - B          filter value = B - C

STICKY is the whole reason this is a fair test. A naive random draw re-picks the entire book
every rebalance -- ~100% turnover against the deployed book's partial turnover -- so a plain
A-vs-random gap silently prices turnover, not ranking. The sticky draw keeps any name that is
still eligible and only replaces the ones that dropped out, which reproduces momentum's
persistence WITHOUT using its ordering.

The eligible pool is obtained by calling the REAL sleeve with an enormous top_n, so every
filter, boost-driven exclusion and regime rule is preserved exactly. Nothing is reimplemented.

Run:  python3 research/EXP003_filter_vs_ranking.py [8yr|26yr]
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "research"))

import numpy as np  # noqa: E402
import livemirror_backtest as LM  # noqa: E402
from livemirror_backtest import LiveMirrorBacktester  # noqa: E402
from evalkit import DEPLOYED, HORIZONS, END, starts, stat  # noqa: E402

HZ = sys.argv[1] if len(sys.argv) > 1 else "26yr"
PATH = HORIZONS[HZ]["path"]
STARTS = starts(HZ)
REPS = 2
BIG = 100_000          # "give me the whole eligible pool"

_S1, _S3, _S5, _SV = (LM.strategy1_momentum_reversal, LM.strategy3_sector_rotation,
                      LM.strategy5_lowvol_quality, LM.strategy_value)


class Sticky:
    """Keeps prior picks that are still eligible; replaces only the drop-outs at random."""

    def __init__(self, seed):
        self.rs = np.random.RandomState(seed)
        self.held = {}
        self.last = {}

    def pick(self, key, date, pool, k):
        # a run restarts whenever we see a date at or before the previous one
        if self.last.get(key) is not None and date <= self.last[key]:
            self.held[key] = []
        self.last[key] = date
        pool = sorted(pool)
        if len(pool) < k:
            return {s: 1.0 / len(pool) for s in pool} if pool else {}
        ps = set(pool)
        keep = [s for s in self.held.get(key, []) if s in ps]
        avail = [s for s in pool if s not in keep]
        need = k - len(keep)
        if need > 0 and avail:
            idx = self.rs.choice(len(avail), size=min(need, len(avail)), replace=False)
            keep = keep + [avail[i] for i in idx]
        keep = keep[:k]
        self.held[key] = keep
        return {s: 1.0 / len(keep) for s in keep} if keep else {}


def install(mode, seed):
    """mode: 'filter' -> sticky random inside the real eligible pool
             'nofilter' -> sticky random inside raw PIT membership"""
    st = Sticky(seed)

    def s1(date, uni, di, top_n=8, rebal_days=10, **kw):
        if di % rebal_days != 0:
            return None
        if mode == "filter":
            pool = list(_S1(date, uni, 0, top_n=BIG, rebal_days=1, **kw) or {})
        else:
            pool = list(uni.get_sp500(date))
        return st.pick("s1", date, pool, top_n)

    def sv(uni, date, members, top_n=10):
        pool = (list(_SV(uni, date, members, top_n=BIG) or {}) if mode == "filter"
                else list(members))
        return st.pick("sv", date, pool, top_n)

    def s5(date, uni, di, top_n=10, rebal_days=10):
        if di % rebal_days != 0:
            return None
        pool = (list(_S5(date, uni, 0, top_n=BIG, rebal_days=1) or {}) if mode == "filter"
                else list(uni.get_sp500(date)))
        return st.pick("s5", date, pool, top_n)

    LM.strategy1_momentum_reversal = s1
    LM.strategy_value = sv
    LM.strategy5_lowvol_quality = s5
    LM.strategy3_sector_rotation = lambda date, uni, di, **kw: {}


def restore():
    (LM.strategy1_momentum_reversal, LM.strategy3_sector_rotation,
     LM.strategy5_lowvol_quality, LM.strategy_value) = _S1, _S3, _S5, _SV


def main():
    t0 = time.time()
    bt = LiveMirrorBacktester(universe_path=PATH)
    print(f"\n{'='*88}\nEXP-003 FILTER vs RANKING — {HZ}, {len(STARTS)} starts, "
          f"{REPS} random reps\n{'='*88}", flush=True)

    # --- sanity: how big is the eligible pool the filters leave? -------------------
    d = sorted(bt.uni._feat_by_date.keys())[len(bt.uni._feat_by_date) // 2]
    bt.uni.get_sp500 = bt._get_sp1500
    mem = bt.uni.get_sp500(d)
    p1 = _S1(d, bt.uni, 0, top_n=BIG, rebal_days=1) or {}
    p5 = _S5(d, bt.uni, 0, top_n=BIG, rebal_days=1) or {}
    pv = _SV(bt.uni, d, mem, top_n=BIG) or {}
    print(f"  pool sizes @{d.date()}:  members {len(mem)}  mom-eligible {len(p1)}  "
          f"lowvol-eligible {len(p5)}  value-eligible {len(pv)}", flush=True)
    if min(len(p1), len(p5), len(pv)) < 20:
        print("  ABORT — a pool is degenerate; the sleeve did not honour top_n=BIG.", flush=True)
        return

    res = {"A": [], "B": [], "C": []}
    for si, st_ in enumerate(STARTS):
        m = bt.run(st_, END, dict(DEPLOYED))
        res["A"].append(stat(m["daily_values"]))
        for tag, mode in (("B", "filter"), ("C", "nofilter")):
            for rep in range(REPS):
                install(mode, seed=9000 + 100 * rep + (0 if tag == "B" else 1))
                try:
                    mm = bt.run(st_, END, dict(DEPLOYED))
                    res[tag].append(stat(mm["daily_values"]))
                finally:
                    restore()
        print(f"  [{si+1:2d}/{len(STARTS)}] {st_}  A {res['A'][-1]['cagr']:+7.2%}   "
              f"({time.time()-t0:.0f}s)", flush=True)

    def col(t, k):
        return np.array([r[k] for r in res[t]])

    print(f"\n  {'arm':<28}{'n':>4}{'meanCAGR':>10}{'sdCAGR':>9}{'meanShrp':>10}"
          f"{'sdShrp':>8}{'meanDD':>9}", flush=True)
    for t, name in (("A", "A DEPLOYED (filter+rank)"),
                    ("B", "B FILTER only (sticky rnd)"),
                    ("C", "C NO FILTER (sticky rnd)")):
        c, s, d_ = col(t, "cagr"), col(t, "sharpe"), col(t, "dd")
        print(f"  {name:<28}{len(c):>4}{c.mean():>+10.2%}{c.std(ddof=1):>9.2%}"
              f"{s.mean():>10.3f}{s.std(ddof=1):>8.3f}{d_.mean():>9.1%}", flush=True)

    ca, cb, cc = col("A", "cagr"), col("B", "cagr"), col("C", "cagr")
    sa, sb, sc = col("A", "sharpe"), col("B", "sharpe"), col("C", "sharpe")

    def sem(x, y):
        return np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))

    print(f"\n  {'DECOMPOSITION':<30}{'dCAGR':>10}{'SEM':>8}{'dSharpe':>10}{'SEM':>8}", flush=True)
    print(f"  {'RANKING value  (A - B)':<30}{(ca.mean()-cb.mean())*100:>+9.2f}p"
          f"{sem(ca, cb)*100:>8.2f}{sa.mean()-sb.mean():>+10.3f}{sem(sa, sb):>8.3f}", flush=True)
    print(f"  {'FILTER  value  (B - C)':<30}{(cb.mean()-cc.mean())*100:>+9.2f}p"
          f"{sem(cb, cc)*100:>8.2f}{sb.mean()-sc.mean():>+10.3f}{sem(sb, sc):>8.3f}", flush=True)
    print(f"  {'TOTAL selection (A - C)':<30}{(ca.mean()-cc.mean())*100:>+9.2f}p"
          f"{sem(ca, cc)*100:>8.2f}{sa.mean()-sc.mean():>+10.3f}{sem(sa, sc):>8.3f}", flush=True)
    tot = ca.mean() - cc.mean()
    if abs(tot) > 1e-9:
        print(f"\n  share of total selection edge from RANKING: "
              f"{(ca.mean()-cb.mean())/tot:.0%}   from FILTER: "
              f"{(cb.mean()-cc.mean())/tot:.0%}", flush=True)
    print(f"\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
