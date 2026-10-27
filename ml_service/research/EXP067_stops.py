"""EXP-067 — STOP RULE: level, confirmation, partial exit and RE-ENTRY after a stop (owner 2026-10-27, after MXL was
stopped in all four books at the 10-22/23 closes and rebounded).

Prior: 30/35/45/50% and no-stop all worse than 40% (cycle 42, batch 8); vol-normalised stops worse; intraday peaks worse
(EXP-062). NOT yet tested: what happens AFTER a stop. Today a stopped name only returns when a book rebuilds (<= 5 sessions
for some book, 20 for the same book). Arms on the deployed package at the LIVE base leverage 1.65:
  base       stop 40% (deployed)
  nostop     no trailing stop (control, re-measured at 1.65)
  conf2      stop fires only on the 2nd consecutive close at/below the line (earnings-gap whipsaw filter)
  partial    sell half at 40%, the remainder at 50%
  reent_tr   the stopped book re-buys the name (same dollar value as at exit) on the next tranche day of ANOTHER book if the
             name is still in the latest signal set, within 20 sessions
  reent_rc   the stopped book re-buys when the close reclaims exit x 1.10 within 20 sessions and the name is still a signal
Everything else = cap0.15_package. Neutrality check: with default flags the patched engine must reproduce the reps-only
engine's curve EXACTLY on the first start (asserted before any arm runs).
Run from ml_service/: python3 research/EXP067_stops.py <8yr|26yr> <stage1|stage2> [run|analyse] [arm ...]"""
import ast, inspect, os, sys, textwrap, time
import numpy as np, pandas as pd
HZ = sys.argv[1]; STAGE = sys.argv[2]; MODE = sys.argv[3] if len(sys.argv) > 3 else "run"; ONLY = sys.argv[4:]; sys.argv = ["x", HZ]
import EXP057_final_on_v2 as E
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
from main_production_backtest import FastBacktester
STARTS = E.STARTS[0::3] if STAGE == "stage1" else E.STARTS
CACHE = f"research/_v2_{HZ}/exp067"; os.makedirs(CACHE, exist_ok=True)
PKG = dict(E.FIN, leverage=1.65, cap=0.15, overlay_down=True, mom_equal=True, mom_w=0.80, val_w=0.15, lv_w=0.05)
ARMS = {"base": dict(PKG),
        "nostop": dict(PKG, stop=0.0),
        "conf2": dict(PKG, stop_confirm=2),
        "partial": dict(PKG, stop_partial=0.5, stop2=0.50),
        "reent_tr": dict(PKG, stop_reentry="tranche"),
        "reent_rc": dict(PKG, stop_reentry="reclaim", reclaim_pct=0.10)}
EXTRA = [
    # state for confirmation / partial / re-entry (K is defined above this line in the clean room)
    ("        stop = float(cfg.get('stop', 0.40)); vol_stop_k = cfg.get('vol_stop'); stop_of = {}\n",
     "        stop = float(cfg.get('stop', 0.40)); vol_stop_k = cfg.get('vol_stop'); stop_of = {}\n"
     "        _conf = int(cfg.get('stop_confirm', 1)); _below = [dict() for _ in range(K)]\n"
     "        _part = float(cfg.get('stop_partial', 0.0)); _stop2 = float(cfg.get('stop2', 0.50)); _parted = [dict() for _ in range(K)]\n"
     "        _reent = cfg.get('stop_reentry'); _stopped = [dict() for _ in range(K)]; _reclaim = float(cfg.get('reclaim_pct', 0.10))\n"
     "        _rwin = int(cfg.get('reentry_window', 20)); last_base = {}\n"),
    # the stop block: confirmation count, partial exit, exit record for re-entry
    ("                        if p <= peaks[t][s] * (1.0 - stop_of.get(s, stop)):\n"
     "                            q = books[t].pop(s); peaks[t].pop(s, None)\n"
     "                            gross_ = q * p\n"
     "                            cash += gross_\n"
     "                            cash -= gross_ * cost_r          # DIFFERENCE 2: separate deduction\n",
     "                        _lvl = _stop2 if _parted[t].get(s) else stop_of.get(s, stop)\n"
     "                        if p <= peaks[t][s] * (1.0 - _lvl):\n"
     "                            _below[t][s] = _below[t].get(s, 0) + 1\n"
     "                            if _below[t][s] < _conf:\n"
     "                                continue\n"
     "                            q = books[t][s]\n"
     "                            if _part > 0 and not _parted[t].get(s) and q > 1:\n"
     "                                qs = max(1, min(int(round(q * _part)), q - 1))\n"
     "                                books[t][s] = q - qs; _parted[t][s] = True; _below[t].pop(s, None)\n"
     "                                gross_ = qs * p\n"
     "                            else:\n"
     "                                books[t].pop(s); peaks[t].pop(s, None); _parted[t].pop(s, None); _below[t].pop(s, None)\n"
     "                                gross_ = q * p\n"
     "                                if _reent: _stopped[t][s] = (i, p, q)\n"
     "                            cash += gross_\n"
     "                            cash -= gross_ * cost_r          # DIFFERENCE 2: separate deduction\n"
     "                        else:\n"
     "                            _below[t].pop(s, None)\n"),
    # re-entry after a stop, evaluated before the NAV mark (same dollar value as at exit; name must still be a signal)
    ("            # DIFFERENCE 1: NAV rebuilt from scratch, never carried\n",
     "            if _reent:\n"
     "                for t in range(K):\n"
     "                    for s, (i0, px0, q0) in list(_stopped[t].items()):\n"
     "                        if i - i0 > _rwin or s in books[t]:\n"
     "                            _stopped[t].pop(s, None); continue\n"
     "                        p = prc.get(s)\n"
     "                        if p is None or s not in last_base:\n"
     "                            continue\n"
     "                        go = ((_reent == 'reclaim' and p >= px0 * (1.0 + _reclaim))\n"
     "                              or (_reent == 'tranche' and i in sched and sched[i] != t))\n"
     "                        if not go:\n"
     "                            continue\n"
     "                        q = int(q0 * px0 / p)\n"
     "                        if q <= 0:\n"
     "                            _stopped[t].pop(s, None); continue\n"
     "                        cash -= q * p; cash -= q * p * cost_r\n"
     "                        books[t][s] = q; peaks[t][s] = p; _stopped[t].pop(s, None)\n"
     "            # DIFFERENCE 1: NAV rebuilt from scratch, never carried\n"),
    # remember the latest signal set for the re-entry membership test
    ("                base = {s: v / tot for s, v in comb.items()} if tot > 0 else {}\n",
     "                base = {s: v / tot for s, v in comb.items()} if tot > 0 else {}; last_base = base\n"),
]


def _compile(extra):
    tree = ast.parse(open("research/EXP059_frontier.py").read())
    eng = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_engine")
    node = next(n for n in ast.walk(eng) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "reps")
    src = inspect.getsource(CleanRoom.run)
    for a, b in ast.literal_eval(node.value) + extra:
        assert src.count(a) == 1, a[:90]
        src = src.replace(a, b)
    V.END = E.END; ns = dict(V.__dict__); assert ns["END"] == E.END
    exec(compile(textwrap.dedent(src), "<exp067>", "exec"), ns); return ns["run"]


def run():
    run_reps, run_full = _compile([]), _compile(EXTRA)
    cr = CleanRoom(FastBacktester(universe_path=E.PATH)); t0 = time.time()
    # NEUTRALITY: default flags must reproduce the reps-only engine exactly
    s0 = STARTS[0]; a = run_reps(cr, s0, PKG)["curve"]; b = run_full(cr, s0, PKG)["curve"]
    assert a.index.equals(b.index) and np.allclose(a.values, b.values, rtol=0, atol=1e-9), "patched engine is NOT neutral with default flags"
    print(f"  neutrality OK on {s0} ({time.time() - t0:.0f}s): identical curve with default flags", flush=True)
    CleanRoom.run = run_full
    for nm, cfg in ARMS.items():
        if ONLY and nm not in ONLY: continue
        f = f"{CACHE}/{STAGE}_{nm}.parquet"
        if os.path.exists(f): continue
        curves = {}
        for s_ in STARTS:
            t1 = time.time(); curves[s_] = cr.run(s_, cfg)["curve"]
            print(f"    {nm:<9} {s_} ({time.time() - t1:.0f}s)", flush=True)
        pd.DataFrame(curves).to_parquet(f); print(f"  {nm:<9} done {time.time() - t0:6.0f}s", flush=True)


def analyse():
    B = pd.read_parquet(f"{CACHE}/{STAGE}_base.parquet"); cols = [c for c in B.columns if c in STARTS]; b = np.array([E.st(B[c]) for c in cols])
    vb = np.mean([B[c].dropna().pct_change().std() * np.sqrt(252) for c in cols])
    print(f"\n{HZ} {STAGE} ({len(cols)} starts) — BASE stop 40% @1.65: {b[:, 0].mean():+.2%} / {b[:, 1].mean():.3f} / {b[:, 2].mean():.1%} vol {vb:.1%}")
    rng = np.random.default_rng(1)
    for nm in ARMS:
        if nm == "base": continue
        f = f"{CACHE}/{STAGE}_{nm}.parquet"
        if not os.path.exists(f): continue
        A = pd.read_parquet(f); a = np.array([E.st(A[c]) for c in cols]); d = a - b
        va = np.mean([A[c].dropna().pct_change().std() * np.sqrt(252) for c in cols])
        yl = [E.yearly(B[c]) for c in cols]; ya = [E.yearly(A[c]) for c in cols]; ys = sorted(set().union(*[set(x) for x in yl]))
        dy = {y: np.mean([xa[y] - xb[y] for xa, xb in zip(ya, yl) if y in xa and y in xb]) for y in ys}; wins = sum(1 for y in ys if dy[y] > 0)
        shd = []
        for _ in range(600):
            c = cols[rng.integers(0, len(cols))]; rb = B[c].dropna().pct_change().dropna(); ra = A[c].dropna().pct_change().dropna(); j = rb.index.intersection(ra.index); x, y = rb[j].values, ra[j].values
            idx = rng.integers(0, len(x) - 60, size=len(x) // 60 + 1); sel = np.concatenate([np.arange(i, i + 60) for i in idx])[:len(x)]
            shd.append(y[sel].mean() / y[sel].std() * np.sqrt(252) - x[sel].mean() / x[sel].std() * np.sqrt(252))
        shd = np.array(shd); n = len(cols)
        print(f"  {nm:<9} {a[:, 0].mean():+.2%} / {a[:, 1].mean():.3f} / {a[:, 2].mean():.1%} vol {va:.1%} | dCAGR {d[:, 0].mean() * 100:+.2f}pp ({int((d[:, 0] > 0).sum())}/{n}) dSharpe {d[:, 1].mean():+.3f} ({int((d[:, 1] > 0).sum())}/{n}) dMaxDD {d[:, 2].mean() * 100:+.2f}pp ({int((d[:, 2] > 0).sum())}/{n}) | years {wins}/{len(ys)} | P(dSharpe>0) {np.mean(shd > 0):.0%}")
        print(f"  {'':<9} years: " + " ".join(f"{y}:{v * 100:+.1f}" for y, v in dy.items()))


if __name__ == "__main__":
    if MODE == "run": run()
    analyse()
