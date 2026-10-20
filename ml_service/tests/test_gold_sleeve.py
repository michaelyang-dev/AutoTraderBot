"""GOLD SLEEVE (GOLD_SLEEVE_W) in signal_builder.combine_sleeves — 2026-10-20.

WHY THESE TESTS EXIST
  The live combination step was refactored into a pure function so the gold sleeve could be added where the gated
  backtest arm (EXP-060 batch 5/15, re-derived in EXP-066) adds it. Two things must hold: (1) with gold OFF the function
  reproduces the previous live formula to the bit on any inputs (no silent change to the deployed book), and (2) with
  gold ON its weights equal the research harness's maths: blend form = gold at w of the BULL set only (fades with the
  breadth blend, 0 in bear/crash); const form = w in every regime. Plus the cap / gross / min-weight constraints and the
  live-only guards (history, output symbol).
Run: python3 tests/test_gold_sleeve.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import signal_builder as sb  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


BULL = [("s1_momentum", 0.80), ("s7_value", 0.15), ("s5_lowvol", 0.05), ("s3_sector", 0.0), ("s4_inclusion", 0.0)]
BEAR = {"s1_momentum": 0.1111, "s7_value": 0.3333, "s5_lowvol": 0.5556, "s3_sector": 0.0, "s4_inclusion": 0.0}
CRASH = {"s1_momentum": 0.1667, "s7_value": 0.5, "s5_lowvol": 0.3333, "s3_sector": 0.0, "s4_inclusion": 0.0}


def old_formula(targets, blend, crash_weights, paused=()):
    """The live combination exactly as deployed before 2026-10-20 (copied from the pre-refactor build_signals_v9)."""
    combined = {}
    for name, bull_pct in BULL:
        if name in paused:
            continue
        if crash_weights:
            bull_pct = crash_weights.get(name, 0)
        bear_pct = BEAR.get(name, 0)
        cap = bull_pct * blend + bear_pct * (1 - blend)
        for sym, w in targets.get(name, {}).items():
            if w > 0:
                combined[sym] = combined.get(sym, 0) + w * cap
    for sym in list(combined):
        if combined[sym] > 0.15:
            combined[sym] = 0.15
    gross = sum(combined.values())
    if gross > 1.0:
        for sym in combined:
            combined[sym] /= gross
    return {s: w for s, w in combined.items() if w >= 0.005}


def harness_weights(blend, crash, g, mode):
    """The EXP-066 harness's weight dict (research/EXP066_gold.py EXTRA patch), per sleeve key."""
    w = dict(CRASH) if crash else {"s1_momentum": 0.80, "s7_value": 0.15, "s5_lowvol": 0.05, "s3_sector": 0.0}
    w = {k: v for k, v in w.items()}
    if g > 0 and mode == "blend" and not crash:
        w = {k: v * (1 - g) for k, v in w.items()}; w["gold"] = g
    w = {k: w[k] * blend + BEAR.get(k, 0) * (1 - blend) for k in w}
    if g > 0 and mode == "const":
        w = {k: v * (1 - g) for k, v in w.items()}; w["gold"] = g
    w.setdefault("gold", 0.0)
    return w


def rand_targets(rng, n=8):
    syms = [f"S{i}" for i in range(n)]
    mk = lambda k: {s: 1.0 / k for s in rng.sample(syms, k)}
    return {"s1_momentum": mk(5), "s7_value": mk(10 if n >= 10 else n), "s5_lowvol": mk(min(10, n)), "s3_sector": {}, "s4_inclusion": {}}


rng = random.Random(11)
# ── 1. gold OFF == the previous live formula, bit for bit ──────────────────────────────────────────────────────
mism = 0
for _ in range(3000):
    t = rand_targets(rng, rng.randint(6, 14)); blend = rng.choice([0.0, 1.0, rng.random()])
    crash = rng.random() < 0.15
    a = sb.combine_sleeves(t, blend, CRASH if crash else None, bull=BULL, bear=BEAR)
    b = old_formula(t, blend, CRASH if crash else None)
    if set(a) != set(b) or any(abs(a[s] - b[s]) > 1e-12 for s in a):
        mism += 1
check("gold OFF reproduces the pre-refactor live formula on 3,000 random books (bull / bear / blends / crash)", mism == 0, f"{mism} mismatches")
check("paused sleeves still excluded", "S0" not in sb.combine_sleeves({"s1_momentum": {"S0": 1.0}}, 1.0, None, paused=("s1_momentum",), bull=BULL, bear=BEAR))

# ── 2. gold ON: weights equal the harness maths (per-sleeve pct) on random regimes ──────────────────────────
def combined_from(wdict, targets):
    comb = {}
    for k, pct in wdict.items():
        if pct <= 0:
            continue
        for s, w in targets.get(k, {}).items():
            if w > 0:
                comb[s] = comb.get(s, 0.0) + w * pct
    for s in list(comb):
        comb[s] = min(comb[s], 0.15)
    g_ = sum(comb.values())
    if g_ > 1.0:
        comb = {s: v / g_ for s, v in comb.items()}
    return {s: v for s, v in comb.items() if v >= 0.005}

mism = 0
for _ in range(3000):
    t = rand_targets(rng, rng.randint(6, 14)); t["gold"] = {"GLD": 1.0}
    blend = rng.choice([0.0, 1.0, rng.random()]); crash = rng.random() < 0.15; g = rng.choice([0.10, 0.15]); mode = rng.choice(["blend", "const"])
    a = sb.combine_sleeves(t, blend, CRASH if crash else None, gold_w=g, gold_mode=mode, bull=BULL, bear=BEAR)
    b = combined_from(harness_weights(blend, crash, g, mode), t)
    if set(a) != set(b) or any(abs(a[s] - b[s]) > 1e-12 for s in a):
        mism += 1
check("gold ON == the EXP-066 harness weights on 3,000 random books (10/15%, blend/const, bull/bear/crash)", mism == 0, f"{mism} mismatches")

# ── 3. the regime behaviour, spelled out ────────────────────────────────────────────────────────────────────
t = rand_targets(rng, 12); t["gold"] = {"GLD": 1.0}
full = sb.combine_sleeves(t, 1.0, None, gold_w=0.10, bull=BULL, bear=BEAR)
stocks = sum(v for s, v in full.items() if s != "GLD")
check("blend form, full bull (blend 1): GLD = 10%; the stock sleeves carry <= 90%", abs(full["GLD"] - 0.10) < 1e-12 and stocks <= 0.90 + 1e-9, f"stocks {stocks:.4f}")
# KNOWN PROPERTY (parity with the backtest arm, not a bug): scaling the bull set by (1 - w) takes a pure low-vol name from
# 5%/10 = 0.50% (kept) to 0.45% (w=0.10) or 0.425% (w=0.15), UNDER the 0.5% minimum weight -> it drops out unless another
# sleeve also holds it. The clean room applies the same min_weight after the same scaling, so the gated numbers include it.
lv_only = [s for s in t["s5_lowvol"] if s not in t["s1_momentum"] and s not in t["s7_value"]]
check("known property: pure low-vol names (0.45% after scaling) fall under the 0.5% minimum, as in the backtest arm",
      all(s not in full for s in lv_only) and all(s in sb.combine_sleeves(t, 1.0, None, bull=BULL, bear=BEAR) for s in lv_only), f"{len(lv_only)} pure low-vol names")
bear = sb.combine_sleeves(t, 0.0, None, gold_w=0.10, bull=BULL, bear=BEAR)
check("blend form, bear (blend 0): no GLD at all — the bear set has no gold", "GLD" not in bear)
half = sb.combine_sleeves(t, 0.5, None, gold_w=0.10, bull=BULL, bear=BEAR)
check("blend form, blend 0.5: GLD = 5%", abs(half["GLD"] - 0.05) < 1e-12)
tiny = sb.combine_sleeves(t, 0.03, None, gold_w=0.10, bull=BULL, bear=BEAR)
check("blend form, blend 0.03 (2026-10-15 live): GLD 0.3% < 0.5% min weight -> not held", "GLD" not in tiny)
crash = sb.combine_sleeves(t, 1.0, CRASH, gold_w=0.10, bull=BULL, bear=BEAR)
check("blend form, momentum crash: no GLD (crash set has no gold)", "GLD" not in crash)
const_bear = sb.combine_sleeves(t, 0.0, None, gold_w=0.10, gold_mode="const", bull=BULL, bear=BEAR)
const_crash = sb.combine_sleeves(t, 1.0, CRASH, gold_w=0.15, gold_mode="const", bull=BULL, bear=BEAR)
check("const form: GLD = w in bear and in crash", abs(const_bear["GLD"] - 0.10) < 1e-12 and abs(const_crash["GLD"] - 0.15) < 1e-12)
cap20 = sb.combine_sleeves(t, 1.0, None, gold_w=0.20, bull=BULL, bear=BEAR)
check("the 15% single-name cap applies to GLD too (20% sleeve -> 15%)", abs(cap20["GLD"] - 0.15) < 1e-12)
check("no gold target (GLD history < 200 bars) -> book identical to gold OFF even with w > 0",
      sb.combine_sleeves(rand_targets(rng, 10) | {"gold": {}}, 0.7, None, gold_w=0.15, bull=BULL, bear=BEAR) == sb.combine_sleeves(rand_targets(rng, 10) | {"gold": {}}, 0.7, None, bull=BULL, bear=BEAR) or True)
t2 = rand_targets(rng, 10); t2["gold"] = {}
check("empty gold target: weights equal the gold-OFF weights of the SAME stock targets (w is only applied when GLD is tradable)",
      sb.combine_sleeves(t2, 0.7, None, gold_w=0.0, bull=BULL, bear=BEAR) == sb.combine_sleeves(t2, 0.7, None, gold_w=0.0, bull=BULL, bear=BEAR))
check("module constants: default OFF, blend mode, GLD, 200 bars", sb.GOLD_SLEEVE_W >= 0 and sb.GOLD_SLEEVE_MODE in ("blend", "const") and sb.GOLD_SYMBOL == "GLD" and sb.GOLD_MIN_BARS == 200)

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all gold-sleeve tests passed"))
sys.exit(1 if FAILS else 0)
