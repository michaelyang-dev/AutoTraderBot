"""Verify the CACHE DEPTH GUARD actually fires -- and, just as important, that it does NOT
fire on a healthy cache or on legitimately-short symbols. No API calls: fetches are stubbed."""
import os, sys, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd, numpy as np
import massive_data_provider as M

def mkbars(n, end="2026-08-18"):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"open": np.arange(n, dtype=float) + 100,
                         "high": np.arange(n, dtype=float) + 101,
                         "low": np.arange(n, dtype=float) + 99,
                         "close": np.arange(n, dtype=float) + 100,
                         "volume": np.full(n, 1e6)}, index=idx)

class Stub(M.MassiveDataProvider):
    def __init__(self, depth):
        self.depth = depth; self.calls = 0
        self.api_key = "x"; self.validate_vs_yfinance = False
    def fetch_ticker_bars(self, symbol, start, end, adjusted=True):
        self.calls += 1
        return mkbars(self.depth)
    def fetch_grouped_daily(self, date, adjusted=True):
        return {}

def run_case(name, cached_depth, fresh_depth, n_syms=100, short_syms=0):
    tmp = tempfile.mkdtemp()
    old = M.CACHE_DIR
    M.CACHE_DIR = __import__("pathlib").Path(tmp)
    M.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    syms = [f"S{i:03d}" for i in range(n_syms)]
    for i, s in enumerate(syms):
        d = 15 if i < short_syms else cached_depth      # a few genuine recent-IPO shorties
        mkbars(d).to_parquet(M.CACHE_DIR / f"{s}_adj.parquet")
    p = Stub(fresh_depth)
    out = p.fetch_bars_batch(syms, warmup_days=550)
    depths = sorted(len(v) for v in out.values() if len(v) > 0)
    med = depths[len(depths)//2]
    print(f"  {name:<44} refetched={p.calls:>4}  median_out={med:<5} "
          f"{'GUARD FIRED' if p.calls>0 else 'cache kept'}")
    M.CACHE_DIR = old
    shutil.rmtree(tmp, ignore_errors=True)
    return p.calls, med

print("=== CACHE DEPTH GUARD TESTS ===")
print("expected sessions for warmup_days=550 ~= 379; floor = 80% = 303\n")

c1, m1 = run_case("A. healthy cache (377 bars)", 377, 377)
c2, m2 = run_case("B. TRUNCATED cache (210 bars) <- the bug", 210, 377)
c3, m3 = run_case("C. healthy + 8 genuine short IPOs", 377, 377, short_syms=8)
c4, m4 = run_case("D. truncated AND vendor still short", 210, 215)

print("\n=== ASSERTIONS ===")
ok = True
def chk(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg); ok &= cond

chk(c1 == 0, "A: healthy cache is NOT refetched (no wasted API budget)")
chk(c2 == 100, "B: truncated cache IS discarded and every symbol refetched")
chk(m2 == 377, "B: post-fix data is full depth (200-bar window now satisfiable)")
chk(c3 == 0, "C: a few genuinely-short IPOs do NOT trigger a mass refetch")
chk(c4 == 100, "D: still refetches when truncated...")
chk(m4 == 215, "D: ...and surfaces that the vendor is the problem, without looping")
print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
sys.exit(0 if ok else 1)
