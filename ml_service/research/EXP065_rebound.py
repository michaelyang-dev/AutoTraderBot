"""EXP-065 — THE REBOUND PROBLEM: can the deployed package re-lever faster after a crash without giving up the
drawdown protection? (owner, 2026-10-16: "increase CAGR without increasing the drawdown much, fix the rebound")

Mechanism of the cost (EXP-064): after a crash the 40-day realised NAV vol stays elevated for ~40 sessions, each book
is rebuilt at a low vol-scale on its own day (every 20 sessions) and the deployed overlay only ever trims, so the book
is still de-levered when the V-recovery comes (2009, 2020, 2026). Arms, all on the deployed package (cap 0.15,
overlay_down, equal-weight momentum, 80/15/5 @ 1.49x), EXP-059 harness, its reps parsed verbatim:
  pkg_overlay_all   weekly PROMPT re-lever of every book to today's vol x gate target (both directions) — overlay_all
                    instead of the de-risk-only overlay_down (was not dead on the old base; never run on the package)
  pkg_semivol       vol estimate = DOWNSIDE semi-deviation x sqrt(2) (40d): a sharp rally does not count as risk
  pkg_ewma15        vol estimate = EWMA (half-life 15 sessions) over up to 120 sessions: forgets a spike faster
  pkg_recov         recovery kicker: when the book is >= 15% below its NAV peak AND 10d vol < 40d vol (vol is falling),
                    scale on the 10d vol — faster re-lever ONLY in a post-crash recovery
  pkg_all_semivol   overlay_all + semivol (prompt re-lever on a rally-blind vol)
  pkg_all_recov     overlay_all + recovery kicker
Protocol: stage 1 = 8 starts both horizons; survivors (dSharpe >= 0 on both, no horizon worse on MaxDD by > 1pp)
go to stage 2 = 24 starts + GATE059-style checks. Base = cap0.15_package (cached). Nothing here touches live.
Run from ml_service/: python3 research/EXP065_rebound.py <8yr|26yr> <stage1|stage2> [run|analyse] [arm ...]
"""
import ast, inspect, os, sys, textwrap, time
import numpy as np, pandas as pd
HZ, STAGE = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "run"
ONLY = sys.argv[4:]
sys.argv = ["x", HZ]
import EXP057_final_on_v2 as E
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
from main_production_backtest import FastBacktester

BASE_CACHE = f"research/_v2_{HZ}/exp059"
CACHE = f"research/_v2_{HZ}/exp065"; os.makedirs(CACHE, exist_ok=True)
STARTS = E.STARTS[0::3] if STAGE == "stage1" else E.STARTS
PKG = dict(E.FIN, leverage=1.49, cap=0.15, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05)
ALL = dict(PKG, overlay_down=False, overlay_all=True)
ARMS = {
    "pkg_overlay_all": ALL,
    "pkg_semivol": dict(PKG, vol_est="semi"),
    "pkg_ewma15": dict(PKG, vol_est="ewma", ewma_hl=15),
    "pkg_recov": dict(PKG, vol_est="recov", recov_dd=0.15),
    "pkg_all_semivol": dict(ALL, vol_est="semi"),
    "pkg_all_recov": dict(ALL, vol_est="recov", recov_dd=0.15),
}

EXTRA = [
    ("                    rv = float(np.std(rr)) * np.sqrt(252)\n",
     "                    rv = float(np.std(rr)) * np.sqrt(252)\n"
     "                    _ve = cfg.get('vol_est')\n"
     "                    if _ve == 'semi':\n"
     "                        _neg = rr[rr < 0]\n"
     "                        if len(_neg) >= 5: rv = float(np.sqrt(np.mean(np.square(_neg)) * 2.0) * np.sqrt(252))\n"
     "                    elif _ve == 'ewma':\n"
     "                        _hl = float(cfg.get('ewma_hl', 15)); _h = np.array(navhist[-121:]); _r = np.diff(_h) / _h[:-1]\n"
     "                        _w = 0.5 ** (np.arange(len(_r))[::-1] / _hl); rv = float(np.sqrt(np.sum(_w * _r * _r) / np.sum(_w)) * np.sqrt(252))\n"
     "                    elif _ve == 'recov':\n"
     "                        _dd = navhist[-1] / max(navhist) - 1\n"
     "                        _r10 = np.diff(np.array(navhist[-11:])) / np.array(navhist[-11:-1]); _rv10 = float(np.std(_r10)) * np.sqrt(252)\n"
     "                        if _dd <= -float(cfg.get('recov_dd', 0.15)) and _rv10 < rv: rv = _rv10\n"
     "                    self._vs_log = getattr(self, '_vs_log', []); self._vs_log.append(rv)\n"),
]


def install():
    tree = ast.parse(open("research/EXP059_frontier.py").read())
    eng = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_engine")
    node = next(n for n in ast.walk(eng) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "reps")
    reps = ast.literal_eval(node.value)
    src = inspect.getsource(CleanRoom.run)
    for a, b in reps + EXTRA:
        assert src.count(a) == 1, a[:90]
        src = src.replace(a, b)
    V.END = E.END; ns = dict(V.__dict__); assert ns["END"] == E.END
    exec(compile(textwrap.dedent(src), "<exp065>", "exec"), ns); CleanRoom.run = ns["run"]


def run(names):
    install(); cr = CleanRoom(FastBacktester(universe_path=E.PATH)); t0 = time.time()
    for nm in names:
        f = f"{CACHE}/{STAGE}_{nm}.parquet"
        if os.path.exists(f): continue
        curves = {}
        for s_ in STARTS:
            t1 = time.time(); r = cr.run(s_, ARMS[nm]); curves[s_] = r["curve"]
            c, sh, dd = E.st(r["curve"]); print(f"    {nm:<16} {s_}  {c:+.2%} / {sh:.3f} / {dd:.1%}  ({time.time() - t1:.0f}s)", flush=True)
        pd.DataFrame(curves).to_parquet(f); print(f"  {nm:<16} done {time.time() - t0:6.0f}s", flush=True)


def analyse(names):
    rng = np.random.default_rng(1)
    B = pd.read_parquet(f"{BASE_CACHE}/{STAGE}_cap0.15_package.parquet")
    cols = [c for c in B.columns if c in STARTS]; b = np.array([E.st(B[c]) for c in cols])
    vb = np.mean([B[c].dropna().pct_change().std() * np.sqrt(252) for c in cols])
    print(f"\n{HZ} {STAGE} ({len(cols)} starts) — PACKAGE (live) {b[:, 0].mean():+.2%} / {b[:, 1].mean():.3f} / {b[:, 2].mean():.1%}  vol {vb:.1%}")
    print(f"  {'arm':<17}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}{'vol':>7}{'dCAGR':>8}{'dShrp':>8}{'+Shrp':>7}{'dMaxDD':>8}{'+DD':>6}{'yrs>':>6}{'volmatchCAGR':>13}")
    for nm in names:
        f = f"{CACHE}/{STAGE}_{nm}.parquet"
        if not os.path.exists(f): continue
        A = pd.read_parquet(f); cc = [c for c in cols if c in A.columns]
        a = np.array([E.st(A[c]) for c in cc]); bb = np.array([E.st(B[c]) for c in cc]); d = a - bb
        va = np.mean([A[c].dropna().pct_change().std() * np.sqrt(252) for c in cc])
        yl = [E.yearly(B[c]) for c in cc]; ya = [E.yearly(A[c]) for c in cc]; ys = sorted(set().union(*[set(x) for x in yl]))
        dy = {y: np.mean([xa[y] - xb[y] for xa, xb in zip(ya, yl) if y in xa and y in xb]) for y in ys}
        wins = sum(1 for y in ys if dy[y] > 0)
        print(f"  {nm:<17}{a[:, 0].mean():>+9.2%}{a[:, 1].mean():>8.3f}{a[:, 2].mean():>8.1%}{va:>7.1%}{d[:, 0].mean() * 100:>+7.2f}p{d[:, 1].mean():>+8.3f}{int((d[:, 1] > 0).sum()):>4}/{len(cc)}{d[:, 2].mean() * 100:>+7.2f}p{int((d[:, 2] > 0).sum()):>3}/{len(cc)}{wins:>4}/{len(ys)}{a[:, 0].mean() * vb / va:>+13.2%}")
        print(f"  {'':<17}years: " + " ".join(f"{y}:{v * 100:+.1f}" for y, v in dy.items()))
        if STAGE == "stage2" and len(cc) >= 12:
            shd = []
            for _ in range(1000):
                c = cc[rng.integers(0, len(cc))]; rb_ = B[c].dropna().pct_change().dropna(); ra_ = A[c].dropna().pct_change().dropna()
                j = rb_.index.intersection(ra_.index); x_, y_ = rb_[j].values, ra_[j].values; Bk = 60
                idx = rng.integers(0, len(x_) - Bk, size=len(x_) // Bk + 1); sel = np.concatenate([np.arange(i, i + Bk) for i in idx])[:len(x_)]
                shd.append(y_[sel].mean() / y_[sel].std() * np.sqrt(252) - x_[sel].mean() / x_[sel].std() * np.sqrt(252))
            shd = np.array(shd); pos = {y: v for y, v in dy.items() if v > 0}
            conc = (max(pos.values()) / sum(pos.values())) if pos and sum(dy.values()) > 0 else float("nan")
            subs = []
            for lab, y0, y1 in E.SUBS:
                ds_ = [E.st(A[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"])[1] - E.st(B[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"])[1] for c in cc if len(A[c].dropna().loc[f"{y0}-01-01":f"{y1}-12-31"]) > 60]
                subs.append(f"{lab} {np.mean(ds_):+.3f}" if ds_ else f"{lab} n/a")
            print(f"  {'':<17}gate: bootstrap dSharpe {shd.mean():+.3f} 90% CI [{np.percentile(shd, 5):+.3f}, {np.percentile(shd, 95):+.3f}] P(>0) {np.mean(shd > 0):.0%} | top-year share {conc:.0%} | sub-periods {' '.join(subs)}")


if __name__ == "__main__":
    names = [a for a in ARMS if (not ONLY or a in ONLY)]
    if MODE == "run": run(names)
    analyse(names)
