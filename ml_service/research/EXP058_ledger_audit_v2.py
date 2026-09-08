"""EXP-058 — LEDGER / LOOK-AHEAD / INDEPENDENT-RECOMPUTE AUDIT of the v2 backtests (user spec §2-3).
Instrumented clean-room runs (trade ledger, daily cash/NAV, data-access recorder) for LIVE and FINAL_1.25,
one used + one untouched start each; independent metric recomputation for ALL 24 cached starts (EXP057).
Run: python3 research/EXP058_ledger_audit_v2.py [8yr|26yr]"""
import os, sys, time, json, inspect, textwrap, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "research"))
HZ = sys.argv[1] if len(sys.argv) > 1 else "8yr"
import EXP057_final_on_v2 as E
from main_production_backtest import FastBacktester, COST_BPS, SLIPPAGE_BPS
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
R = []
def chk(name, ok, ev): R.append((name, "PASS" if ok else "FAIL", ev)); print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {ev}", flush=True)
def info(name, ev): R.append((name, "INFO", ev)); print(f"  [INFO] {name}: {ev}", flush=True)
_AUD = {"d": None, "log": []}
def _engine():
    src = inspect.getsource(CleanRoom.run); reps = [
        ("dd=float(((v - v.cummax()) / v.cummax()).min()))", "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v, ledger=ledger, daily=daily, capfrac=capfrac)"),
        ("        navs, last = [], {}\n", "        navs, last = [], {}; ledger, daily, capfrac = [], [], []\n"),
        ("                cash -= prev_debit * float(fr.get(d, 0.063)) / 252.0\n", "                _fin = prev_debit * float(fr.get(d, 0.063)) / 252.0; cash -= _fin\n            else:\n                _fin = 0.0\n"),
        ("                            cash -= gross_ * cost_r          # DIFFERENCE 2: separate deduction\n", "                            cash -= gross_ * cost_r; ledger.append((d, t, s, -q, p, gross_ * cost_r, 'stop'))\n"),
        ("                        cash -= abs(q * p) * cost_r\n", "                        cash -= abs(q * p) * cost_r; ledger.append((d, t, s, -q, p, abs(q * p) * cost_r, 'exit'))\n"),
        ("                    cash -= abs(dq * p) * cost_r\n", "                    cash -= abs(dq * p) * cost_r; ledger.append((d, t, s, dq, p, abs(dq * p) * cost_r, 'rebal')); capfrac.append((d, t, s, q * p / tnav))\n"),
        ("                t = sched[i]\n", "                t = sched[i]; _AUD['d'] = d\n"),
        ("            navhist.append(max(nav, 1e-9))\n", "            navhist.append(max(nav, 1e-9)); daily.append((d, cash, nav, _fin, sum(len(b) for b in books), sum(q * prc.get(s, lastpx.get(s, 0.0)) for b in books for s, q in b.items())))\n"),
    ]
    for a, b in reps: assert src.count(a) == 1, a; src = src.replace(a, b)
    V.END = E.END; ns = dict(V.__dict__); ns["_AUD"] = _AUD; assert ns["END"] == E.END; exec(compile(textwrap.dedent(src), "<p58>", "exec"), ns); CleanRoom.run = ns["run"]
def _record(uni, bt):
    """Wrap every data accessor the sleeves use; record the max data date touched per decision date."""
    def wrap(name, fn, kind):
        def w(*a, **k):
            out = fn(*a, **k); d = _AUD["d"]
            if d is not None:
                if kind == "feat": req = a[0]; got = req
                elif kind == "series": req = a[1]; got = out.index.max() if len(out) else req
                elif kind == "at": req = a[0]; got = req
                else: req = a[0]; got = max([x for x in bt.sp500_mem if x <= req], default=req)   # membership as-of key
                _AUD["log"].append((d, name, pd.Timestamp(req), pd.Timestamp(got)))
            return out
        return w
    uni.get_feature_map = wrap("get_feature_map", uni.get_feature_map, "feat"); uni.get_close_series = wrap("get_close_series", uni.get_close_series, "series"); uni.get_close_at = wrap("get_close_at", uni.get_close_at, "at")
    bt._get_sp1500 = wrap("get_sp1500(membership)", bt._get_sp1500, "mem")
def m_curve(v):   # method A (EXP057.st)
    return E.st(v)
def m_alt(v):     # method B: CAGR via compounded calendar-year returns; Sharpe from MONTHLY returns*sqrt(12); MaxDD by explicit loop
    v = v.dropna(); yrs = v.groupby(v.index.year); chain = 1.0; first = v.iloc[0]
    for y, seg in yrs: chain *= seg.iloc[-1] / first; first = seg.iloc[-1]
    n = (v.index[-1] - v.index[0]).days / 365.25; cagr = chain ** (1 / n) - 1
    mr = v.resample("ME").last().pct_change().dropna(); sh_m = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0
    pk = -1e18; mdd = 0.0
    for x in v.values:
        pk = max(pk, x); mdd = min(mdd, x / pk - 1)
    return cagr, sh_m, mdd
def main():
    t0 = time.time(); _engine(); print(f"\n{'='*100}\nEXP058 LEDGER/LOOK-AHEAD AUDIT {HZ}  END={E.END.date()}\n{'='*100}", flush=True)
    info("cost config", f"COST_BPS={COST_BPS} SLIPPAGE_BPS={SLIPPAGE_BPS} -> {(COST_BPS+SLIPPAGE_BPS)/1e4:.4%} of traded notional per side; financing = daily on previous day's margin debit at research/_fin_rate.parquet (broker rate)")
    bt = FastBacktester(universe_path=E.PATH); cr = CleanRoom(bt); _record(cr.uni, bt); cr.bt = bt
    starts = [E.STARTS[0], E.STARTS[13]]        # one previously-used (Jan), one untouched (Feb, year 2)
    for nm, cfg in [("LIVE", E.LIVE), ("FINAL_1.25", dict(E.FIN, leverage=1.25))]:
        for s0 in starts:
            _AUD["log"].clear(); res = cr.run(s0, cfg); L = pd.DataFrame(res["ledger"], columns=["date", "tranche", "sym", "dq", "px", "cost", "kind"]); D = pd.DataFrame(res["daily"], columns=["date", "cash", "nav", "fin", "npos", "gross"]).set_index("date")
            print(f"\n--- {nm} start {s0}: {len(L)} fills, {len(D)} days, final NAV {D.nav.iloc[-1]:,.0f} ---", flush=True)
            L.to_parquet(f"{E.CACHE}/ledger_{nm}_{s0}.parquet"); D.to_parquet(f"{E.CACHE}/daily_{nm}_{s0}.parquet")
            # §2 look-ahead
            lg = pd.DataFrame(_AUD["log"], columns=["decision", "accessor", "requested", "got"]); viol = lg[(lg.requested > lg.decision) | (lg.got > lg.decision)]
            samp = lg.groupby("decision").agg(max_requested=("requested", "max"), max_got=("got", "max"), calls=("accessor", "size")); samp = samp.iloc[np.linspace(0, len(samp)-1, 6).astype(int)]
            print("    decision date | max data date requested | max data date returned | accessor calls"); [print(f"    {i.date()}    {r.max_requested.date()}               {r.max_got.date()}              {r.calls}") for i, r in samp.iterrows()]
            mem = lg[lg.accessor.str.startswith("get_sp1500")]; print("    membership as-of: " + ", ".join(f"{a.date()}->{b.date()}" for a, b in zip(mem.decision.iloc[::max(1,len(mem)//4)], mem.got.iloc[::max(1,len(mem)//4)])))
            chk(f"{nm}/{s0} no data access beyond decision date (all sleeves+membership)", len(viol) == 0, f"{len(lg)} accesses over {lg.decision.nunique()} decision dates, {len(viol)} violations")
            # §2 trades / limits / cash
            dup = L.duplicated(subset=["date", "tranche", "sym", "kind"]).sum(); chk(f"{nm}/{s0} no duplicate fills", dup == 0, f"{dup} duplicate (date,tranche,sym,kind) rows of {len(L)}")
            cf = pd.DataFrame(res["capfrac"], columns=["date", "t", "sym", "frac"]); chk(f"{nm}/{s0} no position above the 15% tranche cap at entry", cf.frac.max() <= 0.15 * 1.001, f"max entry weight {cf.frac.max():.2%} of tranche NAV (cap 15%); positions/day median {int(D.npos.median())} max {D.npos.max()}")
            lev = (D.gross / D.nav); chk(f"{nm}/{s0} gross exposure never exceeds leverage limit", lev.max() <= cfg['leverage'] * 1.12, f"gross/NAV max {lev.max():.3f} mean {lev.mean():.3f} (target {cfg['leverage']}x; closed-loop tolerance)")
            info(f"{nm}/{s0} cash", f"min {D.cash.min():,.0f} (negative = margin debit, INTENDED under {cfg['leverage']}x; financed daily) ; total financing paid {D.fin.sum():,.0f}; total commissions+slippage {L.cost.sum():,.0f}")
            # §3 independent NAV reconstruction from the ledger
            cash_recon = cfg["initial_capital"] - (L.dq * L.px).sum() - L.cost.sum() - D.fin.sum(); chk(f"{nm}/{s0} ledger reconstructs final cash", abs(cash_recon - D.cash.iloc[-1]) < 1e-3 * abs(D.cash.iloc[-1]) + 1, f"ledger {cash_recon:,.2f} vs engine {D.cash.iloc[-1]:,.2f}")
            # §3 round-trips, win rate, turnover, trade count
            pos = {}; rt = []
            for r in L.sort_values(["date"]).itertuples():
                k = (r.tranche, r.sym); q0, c0 = pos.get(k, (0, 0.0))
                if r.dq > 0: pos[k] = (q0 + r.dq, c0 + r.dq * r.px + r.cost)
                else:
                    sold = -r.dq; avg = c0 / q0 if q0 else r.px; pnl = sold * r.px - sold * avg - r.cost; q1 = q0 - sold
                    if q1 <= 0: rt.append(pnl + 0); pos.pop(k, None)
                    else: pos[k] = (q1, c0 - sold * avg)
            rt = np.array(rt); yrs = (D.index[-1] - D.index[0]).days / 365.25; turn = (L.dq.abs() * L.px).sum() / D.nav.mean() / yrs
            wr = (rt > 0).mean() if len(rt) else float("nan"); info(f"{nm}/{s0} trades", f"{len(L)} fills, {len(rt)} closed round-trips, win rate {wr:.1%}, avg win {rt[rt>0].mean() if (rt>0).any() else 0:,.0f} avg loss {rt[rt<=0].mean() if (rt<=0).any() else 0:,.0f}, turnover {turn:.2f}x NAV/yr, {len(L)/yrs:.0f} fills/yr")
            chk(f"{nm}/{s0} win rate not implausible (<=70%)", wr <= 0.70, f"{wr:.1%}")
            a = m_curve(D.nav); b = m_alt(D.nav); info(f"{nm}/{s0} metrics A(daily curve) vs B(yearly-chain/monthly/loop)", f"CAGR {a[0]:+.2%} vs {b[0]:+.2%} | Sharpe(d) {a[1]:.3f} vs Sharpe(m) {b[1]:.3f} | MaxDD {a[2]:.1%} vs {b[2]:.1%}")
    # §3 for ALL 24 starts from the EXP057 cache: method A vs B
    print(f"\n--- independent recomputation, all 24 starts, from research/_v2_{HZ}/ ---", flush=True)
    for nm in ("LIVE", "FINAL_1.10", "FINAL_1.25", "FINAL_1.49"):
        f = f"{E.CACHE}/{nm}.parquet"
        if not os.path.exists(f): chk(f"{nm} cache", False, "missing"); continue
        C = pd.read_parquet(f); A = np.array([m_curve(C[c]) for c in C]); B = np.array([m_alt(C[c]) for c in C])
        chk(f"{nm} curves span the declared window", C.dropna(how="all").index.max() >= E.END - pd.Timedelta(days=5), f"first {C.dropna(how='all').index.min().date()} last {C.dropna(how='all').index.max().date()} (declared END {E.END.date()})")
        chk(f"{nm} CAGR A==B (24 starts)", np.abs(A[:,0]-B[:,0]).max() < 1e-9, f"A {A[:,0].mean():+.2%} B {B[:,0].mean():+.2%} max|diff| {np.abs(A[:,0]-B[:,0]).max():.2e}")
        chk(f"{nm} MaxDD A==B", np.abs(A[:,2]-B[:,2]).max() < 1e-9, f"A {A[:,2].mean():.1%} B {B[:,2].mean():.1%} max|diff| {np.abs(A[:,2]-B[:,2]).max():.2e}")
        info(f"{nm} Sharpe daily vs monthly estimator", f"daily {A[:,1].mean():.3f}  monthly {B[:,1].mean():.3f}  (different estimators; agreement within ~0.15 expected)"); chk(f"{nm} Sharpe estimators agree", abs(A[:,1].mean()-B[:,1].mean()) < 0.25, f"|diff| {abs(A[:,1].mean()-B[:,1].mean()):.3f}")
        chk(f"{nm} no too-good flags", A[:,1].max() < 3 and A[:,0].mean() < 0.60, f"max Sharpe {A[:,1].max():.2f} (<3), mean CAGR {A[:,0].mean():+.1%}, worst MaxDD {A[:,2].min():.1%}")
        yl = [E.yearly(C[c]) for c in C]; ys = sorted(set().union(*[set(x) for x in yl])); print(f"    {nm:<11} per-year mean across starts: " + " ".join(f"{y}:{np.mean([x[y] for x in yl if y in x])*100:+.0f}%" for y in ys), flush=True)
    print(f"\n{'='*100}\nEXP058 {HZ}: {sum(1 for r in R if r[1]=='PASS')} PASS / {sum(1 for r in R if r[1]=='FAIL')} FAIL / {sum(1 for r in R if r[1]=='INFO')} INFO ({time.time()-t0:.0f}s)")
    for nm, s_, ev in R:
        if s_ == "FAIL": print(f"  FAIL: {nm}: {ev}")
    print("="*100, flush=True)
if __name__ == "__main__": main()
