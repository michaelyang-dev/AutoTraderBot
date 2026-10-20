"""EXP-066 — GOLD SLEEVE, SECOND IMPLEMENTATION (double-check before deployment, owner 2026-10-20).

EXP-060 found a 10-15% GLD sleeve passes the gate on the package (26yr: +0.3pp CAGR, +0.02 Sharpe 24/24, ~1pp shallower
MaxDD). Its harness gave gold the weight x_w x breadth-blend (gold scales down with the bull sleeves and is 0 in bear
and crash regimes). This re-implements the sleeve on the EXP-059 engine from scratch — same package, 24 starts — in
BOTH forms, so the deployment decision rests on two engines and the regime question is answered:
  gld{10,15}_blend  weight = g x blend (the validated EXP-060 form)
  gld{10,15}_const  weight = g in every regime (bull, bear and crash sleeves all scaled by 1-g)
GLD needs 200 sessions of history (from 2004-11 in the panel; cash before). Everything else = cap0.15_package.
Run from ml_service/: python3 research/EXP066_gold.py <8yr|26yr> [run|analyse]"""
import ast, inspect, os, sys, textwrap, time
import numpy as np, pandas as pd
HZ = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else "run"; sys.argv = ["x", HZ]
import EXP057_final_on_v2 as E
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
from main_production_backtest import FastBacktester
BASE = f"research/_v2_{HZ}/exp059/stage2_cap0.15_package.parquet"
CACHE = f"research/_v2_{HZ}/exp066"; os.makedirs(CACHE, exist_ok=True)
PKG = dict(E.FIN, leverage=1.49, cap=0.15, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05)
ARMS = {"gld10_blend": dict(PKG, gold_w=0.10, gold_mode="blend"), "gld15_blend": dict(PKG, gold_w=0.15, gold_mode="blend"),
        "gld10_const": dict(PKG, gold_w=0.10, gold_mode="const"), "gld15_const": dict(PKG, gold_w=0.15, gold_mode="const")}
EXTRA = [
    # the gold sleeve: GLD alone, once it has 200 sessions of history
    ("                last = {\"m\": m1, \"l\": m5, \"s\": m3}\n",
     "                last = {\"m\": m1, \"l\": m5, \"s\": m3}\n"
     "                _g = float(cfg.get('gold_w', 0.0)); mg = {}\n"
     "                if _g > 0 and 'GLD' in px.columns and px['GLD'].loc[:d].dropna().shape[0] >= 200: mg = {'GLD': 1.0}\n"),
    # weights: blend form = gold at g in the BULL set only (scales with the blend, 0 in bear/crash);
    #          const form = gold at g always, every other sleeve scaled by (1 - g)
    ("                w = {k: w[k] * bl + bw.get(k, 0) * (1 - bl) for k in w}\n",
     "                if _g > 0 and cfg.get('gold_mode', 'blend') == 'blend' and not crash:\n"
     "                    w = {k: v * (1 - _g) for k, v in w.items()}; w['gold'] = _g\n"
     "                w = {k: w[k] * bl + bw.get(k, 0) * (1 - bl) for k in w}\n"
     "                if _g > 0 and cfg.get('gold_mode', 'blend') == 'const':\n"
     "                    w = {k: v * (1 - _g) for k, v in w.items()}; w['gold'] = _g\n"
     "                w.setdefault('gold', 0.0)\n"),
    ("                for key, src in ((\"mom\", m1), (\"val\", mv), (\"s5\", m5), (\"s3\", m3)):\n",
     "                for key, src in ((\"mom\", m1), (\"val\", mv), (\"s5\", m5), (\"s3\", m3), (\"gold\", mg)):\n"),
]


def install():
    tree = ast.parse(open("research/EXP059_frontier.py").read())
    eng = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_engine")
    node = next(n for n in ast.walk(eng) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "reps")
    src = inspect.getsource(CleanRoom.run)
    for a, b in ast.literal_eval(node.value) + EXTRA:
        assert src.count(a) == 1, a[:90]
        src = src.replace(a, b)
    V.END = E.END; ns = dict(V.__dict__); assert ns["END"] == E.END
    exec(compile(textwrap.dedent(src), "<exp066>", "exec"), ns); CleanRoom.run = ns["run"]


def run():
    install(); cr = CleanRoom(FastBacktester(universe_path=E.PATH)); t0 = time.time()
    for nm, cfg in ARMS.items():
        f = f"{CACHE}/stage2_{nm}.parquet"
        if os.path.exists(f): continue
        curves = {}
        for s_ in E.STARTS:
            t1 = time.time(); curves[s_] = cr.run(s_, cfg)["curve"]
            print(f"    {nm:<12} {s_} ({time.time() - t1:.0f}s)", flush=True)
        pd.DataFrame(curves).to_parquet(f); print(f"  {nm:<12} done {time.time() - t0:6.0f}s", flush=True)


def analyse():
    B = pd.read_parquet(BASE); cols = [c for c in B.columns if c in E.STARTS]; b = np.array([E.st(B[c]) for c in cols])
    vb = np.mean([B[c].dropna().pct_change().std() * np.sqrt(252) for c in cols])
    print(f"\n{HZ} (24 starts) — PACKAGE {b[:, 0].mean():+.2%} / {b[:, 1].mean():.3f} / {b[:, 2].mean():.1%} vol {vb:.1%}")
    rng = np.random.default_rng(1)
    for nm in ARMS:
        f = f"{CACHE}/stage2_{nm}.parquet"
        if not os.path.exists(f): continue
        A = pd.read_parquet(f); a = np.array([E.st(A[c]) for c in cols]); d = a - b
        va = np.mean([A[c].dropna().pct_change().std() * np.sqrt(252) for c in cols])
        yl = [E.yearly(B[c]) for c in cols]; ya = [E.yearly(A[c]) for c in cols]; ys = sorted(set().union(*[set(x) for x in yl]))
        dy = {y: np.mean([xa[y] - xb[y] for xa, xb in zip(ya, yl) if y in xa and y in xb]) for y in ys}; wins = sum(1 for y in ys if dy[y] > 0)
        pos = {y: v for y, v in dy.items() if v > 0}; conc = max(pos.values()) / sum(pos.values()) if pos else float("nan")
        shd = []
        for _ in range(1000):
            c = cols[rng.integers(0, len(cols))]; rb = B[c].dropna().pct_change().dropna(); ra = A[c].dropna().pct_change().dropna(); j = rb.index.intersection(ra.index); x, y = rb[j].values, ra[j].values
            idx = rng.integers(0, len(x) - 60, size=len(x) // 60 + 1); sel = np.concatenate([np.arange(i, i + 60) for i in idx])[:len(x)]
            shd.append(y[sel].mean() / y[sel].std() * np.sqrt(252) - x[sel].mean() / x[sel].std() * np.sqrt(252))
        shd = np.array(shd)
        print(f"  {nm:<12} {a[:, 0].mean():+.2%} / {a[:, 1].mean():.3f} / {a[:, 2].mean():.1%} vol {va:.1%} | dCAGR {d[:, 0].mean() * 100:+.2f}pp ({int((d[:, 0] > 0).sum())}/24) dSharpe {d[:, 1].mean():+.3f} ({int((d[:, 1] > 0).sum())}/24) dMaxDD {d[:, 2].mean() * 100:+.2f}pp ({int((d[:, 2] > 0).sum())}/24) | vol-matched CAGR {a[:, 0].mean() * vb / va:+.2%} | years {wins}/{len(ys)} top-yr {conc:.0%} | bootstrap P {np.mean(shd > 0):.0%} CI [{np.percentile(shd, 5):+.3f},{np.percentile(shd, 95):+.3f}]")
        print(f"  {'':<12} years: " + " ".join(f"{y}:{v * 100:+.1f}" for y, v in dy.items()))


if __name__ == "__main__":
    if MODE == "run": run()
    analyse()
