"""
Thread #1b — GDELT news-tone GO/NO-GO for the break/bounce classifier.
Price features gave AUC 0.55 (thread1). Does NEWS TONE in the gap window separate
structural breaks from overreaction bounces? Small throttled prototype:
sample balanced gap-down events (2021-2024), query GDELT DOC API for each
company's article tone in [gap-1d, gap+2d], compare tone break-vs-bounce + tone-only AUC.
HONEST: noisy name matching, generic tone (not stock-specific), small n -> sniff
test only. If tone clearly separates -> justify full GDELT build; if not -> news
(at least GDELT generic tone) doesn't rescue it.
"""
import sys, os, time, ssl, json, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import certifi
from main_production_backtest import FastBacktester
from sklearn.metrics import roc_auc_score

CTX = ssl.create_default_context(cafile=certifi.where())
GAP, BREAK_H, BREAK_THR = 0.10, 40, -0.10


def clear(bt):
    for k in ["_fin_growth", "_ev", "_estimates", "_price_targets", "_revenue_surprise",
              "_beat_streak", "_earnings_signals", "_short_interest_rank", "_si_change_rank"]:
        setattr(bt.uni, k, {})
    bt._si_ranks_by_month = {}; bt._si_change_ranks_by_month = {}; bt._si_months = []


def gdelt_tone(name, d0, d1, tries=3):
    """avg article tone + count for company `name` in [d0,d1]."""
    q = f'"{name}"'
    p = dict(query=q, mode="ToneChart", format="json",
             startdatetime=d0.strftime("%Y%m%d000000"),
             enddatetime=d1.strftime("%Y%m%d235959"))
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(p)
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}),
                                        context=CTX, timeout=30) as r:
                js = json.loads(r.read().decode("utf-8", "replace"))
            bins = js.get("tonechart", [])
            tot = sum(b["count"] for b in bins)
            if tot == 0:
                return None, 0
            mean = sum(b["bin"] * b["count"] for b in bins) / tot
            return mean, tot
        except Exception as e:
            if "429" in str(e):
                time.sleep(8 * (a + 1)); continue
            return None, -1
    return None, -1


if __name__ == "__main__":
    t0 = time.time()
    bt = FastBacktester(); clear(bt); bt.uni.get_sp500 = bt._get_sp1500
    si = pd.read_parquet("data/wrds/crsp_security_info.parquet", columns=["Ticker", "IssuerNm"])
    name_map = dict(si.dropna().drop_duplicates("Ticker").values)
    print(f"[loaded {time.time()-t0:.0f}s, {len(name_map)} names]")

    prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
    allidx = list(prices.index); di = {d: i for i, d in enumerate(allidx)}
    fbd = bt.features_by_date
    dates = [d for d in allidx if pd.Timestamp("2021-01-01") <= d <= pd.Timestamp("2024-10-01")]

    evs = []
    for d in dates:
        gi = di[d]
        if gi + BREAK_H >= len(allidx):
            continue
        sr = spy_r.loc[d]
        if pd.isna(sr):
            continue
        row = dr.loc[d]
        for sym in bt._get_sp1500(d):
            g = row.get(sym)
            if g is None or pd.isna(g) or (g - sr) >= -GAP:
                continue
            if sym not in name_map:
                continue
            seg = dr[sym].iloc[gi+1:gi+1+BREAK_H]
            if seg.isna().all():
                continue
            fwd = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+1+BREAK_H].sum()
            evs.append((d, sym, int(fwd < BREAK_THR)))
    ev = pd.DataFrame(evs, columns=["date", "sym", "brk"])
    rng = np.random.default_rng(11)
    samp = pd.concat([ev[ev.brk == 1].sample(min(30, (ev.brk==1).sum()), random_state=11),
                      ev[ev.brk == 0].sample(30, random_state=11)]).sample(frac=1, random_state=1)
    print(f"[{len(ev)} events; querying GDELT for {len(samp)} sampled (throttled)]")

    rows = []
    for i, r in enumerate(samp.itertuples()):
        nm = name_map[r.sym]
        d0 = r.date - pd.Timedelta(days=1); d1 = r.date + pd.Timedelta(days=2)
        tone, cnt = gdelt_tone(nm, d0, d1)
        rows.append({"sym": r.sym, "name": nm, "brk": r.brk, "tone": tone, "narts": cnt})
        time.sleep(5)
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(samp)} done ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_parquet("research/_gdelt_proto.parquet")
    cov = res[res["tone"].notna()]
    print(f"\n=== GDELT tone prototype ({len(cov)}/{len(res)} events had coverage) ===")
    if len(cov) >= 10:
        for b, g in cov.groupby("brk"):
            print(f"  {'BREAK' if b else 'BOUNCE'}: avg tone {g['tone'].mean():+.2f}  "
                  f"median {g['tone'].median():+.2f}  avg #arts {g['narts'].mean():.0f}  n={len(g)}")
        if cov["brk"].nunique() > 1:
            # tone-only AUC (more-negative tone => more likely break => use -tone)
            print(f"  tone-only AUC for break: {roc_auc_score(cov['brk'], -cov['tone']):.3f}")
    else:
        print("  insufficient GDELT coverage on these (mostly smaller-cap) names -> "
              "generic GDELT tone is too sparse; full build would need richer per-stock news.")
    print(f"\n[total {time.time()-t0:.0f}s]")
