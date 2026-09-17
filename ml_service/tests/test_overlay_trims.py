"""EXP-059 de-risk-only overlay: the pure trim function in ibkr_engine, and its parity with the clean-room
harness formula (research/EXP059_frontier.py overlay_down). Run: python3 tests/test_overlay_trims.py"""
import os, sys, random, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "ib_insync" not in sys.modules:
    _stub = types.ModuleType("ib_insync")
    _stub.__all__ = ["IB", "MarketOrder", "Position", "Stock", "util"]
    for _n in _stub.__all__:
        setattr(_stub, _n, type(_n, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules["ib_insync"] = _stub
os.environ.setdefault("IBKR_TRANCHES", "4")
import ibkr_engine as E  # noqa: E402
E.send_telegram = lambda m: None

fails = 0
def check(name, cond, detail=""):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond: fails += 1

F = E.IBKREngine._overlay_trims
px = {"A": 100.0, "B": 50.0, "C": 20.0, "D": 10.0}
books = {"0": {"A": 30, "B": 40}, "1": {"C": 200, "D": 300}, "2": {"A": 10, "C": 50}, "3": {}}
pos = {"A": 40, "B": 40, "C": 250, "D": 300}
nav = 15000.0

# book 0 gross = 3000+2000 = 5000 = 0.333x of 15000; target 1.34x -> f > 1 -> no trim (never up)
check("no trim when every book is below target", F(books, "9", px, nav, 1.34, pos) == [])
# target 0.20x -> target $3000; book0 f = 0.6 -> A 30->18 (-12), B 40->24 (-16); book1 gross 4000+3000=7000, f=0.4286 -> C 200->85 (-115), D 300->128 (-172); book2 gross 1000+1000=2000, f=1.5 -> no trim
t = F(books, "9", px, nav, 0.20, pos)
check("pro-rata int() trims on books above target", sorted(t) == sorted([("0","A",-12,"overlay_derisk"),("0","B",-16,"overlay_derisk"),("1","C",-115,"overlay_derisk"),("1","D",-172,"overlay_derisk")]), str(t))
check("skip_book is untouched", all(b != "0" for b, *_ in F(books, "0", px, nav, 0.20, pos)))
# threshold: book0 at 0.333x vs target 0.32x -> f = 0.96 >= 0.95 -> no trim; target 0.31x -> f=0.93 -> trim
check("no trim inside the 5% band", F({"0": books["0"]}, "9", px, nav, 0.32, pos) == [])
check("trim just outside the band", F({"0": books["0"]}, "9", px, nav, 0.31, pos) != [])
# min-trade band: gross 3000+30 = 3030, target 0.10x = $1500 -> f = 0.495: A 30->14 (-16 = $1600), D 3->1 (-2 = $20 < 0.3% x 15000 = $45 -> skipped)
check("tiny deltas skipped", F({"0": {"A": 30, "D": 3}}, "9", px, nav, 0.10, {"A": 30, "D": 3}) == [("0","A",-16,"overlay_derisk")])
# bounded by actual position minus other books: A held 40 at broker, book2 has 10 -> book0 can sell at most 30; ask for 12 -> fine; if broker only has 15 A -> 15-10 = 5 sellable
t = F(books, "9", px, nav, 0.20, {"A": 15, "B": 40, "C": 250, "D": 300})
check("sell bounded by broker position minus other books", ("0","A",-5,"overlay_derisk") in t, str(t))
# no price -> name skipped, and its book's gross ignores it: gross = 3000, target 0.10x = 1500 -> f = 0.5 -> A 30->15
t = F({"0": {"A": 30, "Z": 100}}, "9", px, nav, 0.10, {"A": 30, "Z": 100})
check("unpriced name skipped", t == [("0","A",-15,"overlay_derisk")], str(t))
check("zero/negative target -> nothing", F(books, "9", px, nav, 0.0, pos) == [] and F(books, "9", px, 0.0, 1.0, pos) == [])
check("never emits a buy", all(d < 0 for _, _, d, _ in F(books, "9", px, nav, 0.05, pos)))

# ---- parity with the clean-room formula (random books) ----
def cleanroom(books, skip, prc, tnav, lev_dr, thr=0.95):
    out = []
    for t2, bk in books.items():
        if t2 == skip or not bk: continue
        gross2 = sum(q2 * prc.get(s2, 0.0) for s2, q2 in bk.items()); tgt2 = tnav * lev_dr
        f2 = (tgt2 / gross2) if gross2 > 0 else None
        if f2 is not None and (tgt2 <= 0 or abs(f2 - 1) > 0.05) and f2 < thr:
            for s2 in list(bk):
                p2 = prc.get(s2)
                if not p2: continue
                q_old = bk[s2]; q_new = int(q_old * f2); dq2 = q_new - q_old
                if dq2 == 0 or abs(dq2 * p2) < tnav * 0.003: continue
                out.append((t2, s2, dq2))
    return sorted(out)
rng = random.Random(7); mism = 0
for _ in range(300):
    syms = [f"S{i}" for i in range(8)]; prc = {s: rng.uniform(5, 900) for s in syms}
    bks = {str(b): {s: rng.randint(1, 200) for s in rng.sample(syms, rng.randint(1, 6))} for b in range(4)}
    tn = rng.uniform(5000, 60000); lev = rng.uniform(0.0, 1.6)
    posq = {}
    for bk in bks.values():
        for s, q in bk.items(): posq[s] = posq.get(s, 0) + q
    mine = sorted((b, s, d) for b, s, d, _ in F(bks, "0", prc, tn, lev, posq))
    ref = cleanroom(bks, "0", prc, tn, lev)
    if mine != ref: mism += 1
check("parity with the clean-room overlay formula on 300 random books", mism == 0, f"{mism} mismatches")
print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAIL'}"); sys.exit(1 if fails else 0)
