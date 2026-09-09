"""FINAL PARITY AUDIT (2026-09-08 evening), one universe load:
 (1) SHIFT test: decide on yesterday's data, execute at today's close (a leak dies; a real edge degrades gracefully).
 (2) LIVE vs BACKTEST like-for-like since the 2026-08-11 rebalance: clean room started 2026-08-11 with the LIVE config on the v2
     universe vs the account's NAV path (no deposits in the window) and its current 25 holdings.
 (3) SIGNAL PARITY on 2026-09-04: backtest sleeves + 70/21/9 combination on the CRSP/Compustat universe vs the live server's 25 BUYs."""
import os, sys, json, inspect, textwrap, numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.getcwd()); sys.path.insert(0, "research")
S = "/private/tmp/claude-501/-Users-michaelslyanggmail-com-Downloads-auto-trader-2/6f99a2a4-abb9-4cf8-828c-1cfdd3632fc3/scratchpad"
import EXP057_final_on_v2 as E
from main_production_backtest import FastBacktester, _short_weights
from strategies.multi_strategy_engine import strategy1_momentum_reversal, strategy5_lowvol_quality, strategy_value, PROD_WEIGHTS_BEAR
import VERIFY2_cleanroom as V
from VERIFY2_cleanroom import CleanRoom
src = inspect.getsource(CleanRoom.run)
reps = [("dd=float(((v - v.cummax()) / v.cummax()).min()))", "dd=float(((v - v.cummax()) / v.cummax()).min()), curve=v, books=books)"),
        # SHIFT: sleeves + breadth + UMD evaluated on the PREVIOUS session's data when cfg['shift']
        ("                m1 = strategy1_momentum_reversal(d, self.uni, di, top_n=cfg.get(\"top_n\", 5),\n",
         "                dsig = dates[i - 1] if (cfg.get('shift') and i > 0) else d\n                m1 = strategy1_momentum_reversal(dsig, self.uni, di, top_n=cfg.get(\"top_n\", 5),\n"),
        ("                m5 = strategy5_lowvol_quality(d, self.uni, di)\n                mem = self.uni.get_sp500(d)\n                mv = strategy_value(self.uni, d, mem, top_n=10)\n                m3 = strategy3_sector_rotation(d, self.uni, di)\n",
         "                m5 = strategy5_lowvol_quality(dsig, self.uni, di)\n                mem = self.uni.get_sp500(dsig)\n                mv = strategy_value(self.uni, dsig, mem, top_n=10)\n                m3 = strategy3_sector_rotation(dsig, self.uni, di)\n"),
        ("                nu = self.bt.umd_20d.loc[:d]\n", "                nu = self.bt.umd_20d.loc[:dsig]\n"),
        ("                fd = self.bt.features_by_date.get(d, {})\n", "                fd = self.bt.features_by_date.get(dsig, {})\n")]
for a, b in reps: assert src.count(a) == 1, a[:70]; src = src.replace(a, b)
V.END = pd.Timestamp("2026-09-04"); ns = dict(V.__dict__); exec(compile(textwrap.dedent(src), "<fp>", "exec"), ns); CleanRoom.run = ns["run"]
def st(v): return E.st(v)
bt = FastBacktester(universe_path=E.PATH); cr = CleanRoom(bt); p2t = pickle_p2t = None
import pickle
with open(E.PATH, "rb") as f: d_ = pickle.load(f); p2t = d_["permno_to_ticker"]; del d_
t2p = {}
for p, t in p2t.items(): t2p.setdefault(t, []).append(p)
print(f"\n{'#'*100}\n(1) SHIFT TEST — 8yr, 8 starts, decide on T-1 data, execute at T close\n{'#'*100}", flush=True)
STARTS = E.STARTS[::3]
for nm, cfg in (("LIVE", E.LIVE), ("FINAL_1.49", dict(E.FIN, leverage=1.49))):
    base = np.array([st(pd.read_parquet(f"{E.CACHE}/{nm}.parquet")[s].loc[:"2026-09-04"]) for s in STARTS])
    sh = np.array([[r["cagr"], r["sharpe"], r["dd"]] for r in (cr.run(s, dict(cfg, shift=True)) for s in STARTS)])
    print(f"  {nm:<11} normal {base[:,0].mean():+.2%} / {base[:,1].mean():.3f} / {base[:,2].mean():.1%}   SHIFTED {sh[:,0].mean():+.2%} / {sh[:,1].mean():.3f} / {sh[:,2].mean():.1%}   dCAGR {(sh[:,0]-base[:,0]).mean()*100:+.2f}pp dSharpe {(sh[:,1]-base[:,1]).mean():+.3f}", flush=True)
print(f"\n{'#'*100}\n(2) LIVE vs BACKTEST since 2026-08-11 (clean room start 2026-08-11, LIVE config, v2 universe)\n{'#'*100}", flush=True)
r = cr.run("2026-08-11", E.LIVE); c = r["curve"]; nav = json.load(open(f"{S}/ibkr_nav_history.json")); nav = pd.Series({pd.Timestamp(a): b for a, b in nav}).sort_index()
live = nav.loc["2026-08-11":"2026-09-04"]; bt_c = c.loc["2026-08-11":"2026-09-04"]
print(f"  live NAV  {live.index[0].date()} {live.iloc[0]:,.0f} -> {live.index[-1].date()} {live.iloc[-1]:,.0f}  = {live.iloc[-1]/live.iloc[0]-1:+.2%}  (no deposits in window)")
print(f"  backtest  {bt_c.index[0].date()} -> {bt_c.index[-1].date()} = {bt_c.iloc[-1]/bt_c.iloc[0]-1:+.2%}  (fresh $50k book, first rebalance 08-11 close)")
j = live.index.intersection(bt_c.index); lr, br = live.reindex(j).pct_change().dropna(), bt_c.reindex(j).pct_change().dropna()
print(f"  daily-return corr {np.corrcoef(lr, br)[0,1]:.2f} on {len(lr)} days; live vol {lr.std()*np.sqrt(252):.0%} vs backtest {br.std()*np.sqrt(252):.0%}")
snap = json.load(open(f"{S}/ibkr_close_snapshot.json")); live_names = {p["symbol"] for p in snap["positions"]}
bt_names = {p2t.get(s, s) for b in r["books"] for s in b}
print(f"  holdings on 09-04: live {len(live_names)} names, backtest {len(bt_names)}; overlap {len(live_names & bt_names)}: {sorted(live_names & bt_names)}")
print(f"  live-only: {sorted(live_names - bt_names)}\n  backtest-only: {sorted(bt_names - live_names)}")
print(f"\n{'#'*100}\n(3) SIGNAL PARITY 2026-09-04 — backtest sleeves on CRSP/Compustat vs live server BUYs (Polygon + Compustat/EDGAR)\n{'#'*100}", flush=True)
d = pd.Timestamp("2026-09-04"); uni = bt.uni; uni.get_sp500 = bt._get_sp1500
m1 = strategy1_momentum_reversal(d, uni, 0, top_n=5, rebal_days=20) or {}; m5 = strategy5_lowvol_quality(d, uni, 0) or {}; mem = uni.get_sp500(d); mv = strategy_value(uni, d, mem, top_n=10) or {}
fd = bt.features_by_date.get(d, {}); ab = sum(1 for x in fd.values() if x.get("dist_sma50", 0) > 0); tf = sum(1 for x in fd.values() if "dist_sma50" in x); bl = min(1.0, max(0.0, (ab / max(tf, 1) - 0.35) / 0.25))
w = {"mom": .70, "val": .21, "s5": .09, "s3": 0.0}; bw = _short_weights(PROD_WEIGHTS_BEAR); w = {k: w[k] * bl + bw.get(k, 0) * (1 - bl) for k in w}
comb = {}
for key, srcm in (("mom", m1), ("val", mv), ("s5", m5)):
    for s, ww in srcm.items():
        if ww > 0: comb[s] = comb.get(s, 0.0) + ww * w[key]
comb = {s: min(v, 0.10) for s, v in comb.items() if v > 0}; g = sum(comb.values()); comb = {s: v / g for s, v in comb.items()} if g > 1 else comb; comb = {s: v for s, v in comb.items() if v >= 0.005}
bt_list = sorted(((p2t.get(s, s), v) for s, v in comb.items()), key=lambda x: -x[1])
sig = json.load(open(f"{S}/sig.json")); sig = sig.get("signals", sig); live_list = sorted([(x["symbol"], x["probability"]) for x in sig if x.get("signal") == "BUY"], key=lambda x: -x[1])
B, Lv = {s for s, _ in bt_list}, {s for s, _ in live_list}
print(f"  breadth blend {bl:.2f} | momentum top-5 backtest {sorted(p2t.get(s,s) for s in m1)} | live top-5 by prob {[s for s,_ in live_list[:5]]}")
print(f"  backtest {len(B)} names: {' '.join(f'{s}:{v:.3f}' for s, v in bt_list)}")
print(f"  live     {len(Lv)} names: {' '.join(f'{s}:{v:.2f}' for s, v in live_list)}")
print(f"  OVERLAP {len(B & Lv)}/{len(Lv)} | backtest-only {sorted(B - Lv)} | live-only {sorted(Lv - B)}")
rb = {s: i for i, (s, _) in enumerate(bt_list)}; rl = {s: i for i, (s, _) in enumerate(live_list)}; common = sorted(B & Lv)
if len(common) > 3:
    from scipy.stats import spearmanr; print(f"  rank correlation on the {len(common)} common names: {spearmanr([rb[s] for s in common], [rl[s] for s in common])[0]:.2f}")
