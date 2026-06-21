"""
Thread #1b MINIMAL — directional GDELT sniff on LARGE-CAP gap-down events only
(dense coverage), wide 12s spacing, unbuffered. Anecdotal (small n) but tells us
if news tone even points the right way (breaks more negative than bounces).
"""
import sys, os, time, ssl, json, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd, certifi
from main_production_backtest import FastBacktester

CTX = ssl.create_default_context(cafile=certifi.where())
BIG = ["AAPL","MSFT","NVDA","META","GOOGL","AMZN","TSLA","NFLX","AMD","CRM","ADBE",
       "INTC","DIS","BA","NKE","PYPL","SBUX","QCOM","MU","UBER","SNAP","ROKU","ZM",
       "DOCU","PINS","SQ","SHOP","ABNB","COIN","PLTR","MRNA","LULU","FDX","TGT","WBA"]
NAMES = {"AAPL":"Apple","MSFT":"Microsoft","NVDA":"Nvidia","META":"Meta Platforms",
  "GOOGL":"Google","AMZN":"Amazon","TSLA":"Tesla","NFLX":"Netflix","AMD":"AMD",
  "CRM":"Salesforce","ADBE":"Adobe","INTC":"Intel","DIS":"Disney","BA":"Boeing",
  "NKE":"Nike","PYPL":"PayPal","SBUX":"Starbucks","QCOM":"Qualcomm","MU":"Micron",
  "UBER":"Uber","SNAP":"Snap","ROKU":"Roku","ZM":"Zoom","DOCU":"DocuSign",
  "PINS":"Pinterest","SQ":"Block Inc","SHOP":"Shopify","ABNB":"Airbnb","COIN":"Coinbase",
  "PLTR":"Palantir","MRNA":"Moderna","LULU":"Lululemon","FDX":"FedEx","TGT":"Target","WBA":"Walgreens"}


def gtone(name, d0, d1):
    p = dict(query=f'"{name}"', mode="ToneChart", format="json",
             startdatetime=d0.strftime("%Y%m%d000000"), enddatetime=d1.strftime("%Y%m%d235959"))
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(p)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent":"r"}), context=CTX, timeout=30) as r:
            js = json.loads(r.read().decode("utf-8","replace"))
        b = js.get("tonechart", []); tot = sum(x["count"] for x in b)
        return (sum(x["bin"]*x["count"] for x in b)/tot, tot) if tot else (None, 0)
    except Exception as e:
        return None, str(e)[:30]


bt = FastBacktester()
prices = bt.prices; dr = prices.pct_change(); spy_r = dr["SPY"]
ai = list(prices.index); di = {d:i for i,d in enumerate(ai)}
evs = []
for sym in BIG:
    if sym not in dr.columns: continue
    s = dr[sym]
    for d in [x for x in ai if pd.Timestamp("2021-01-01")<=x<=pd.Timestamp("2024-09-01")]:
        gi = di[d]
        if gi+40 >= len(ai): continue
        g = s.loc[d]; sr = spy_r.loc[d]
        if pd.isna(g) or pd.isna(sr) or (g-sr) >= -0.10: continue
        seg = s.iloc[gi+1:gi+41]
        if seg.isna().all(): continue
        fwd = seg.fillna(0).sum() - spy_r.iloc[gi+1:gi+41].sum()
        evs.append((d, sym, int(fwd < -0.10)))
ev = pd.DataFrame(evs, columns=["date","sym","brk"])
rng = np.random.default_rng(3)
samp = pd.concat([ev[ev.brk==1].sample(min(7,(ev.brk==1).sum()),random_state=3),
                  ev[ev.brk==0].sample(min(7,(ev.brk==0).sum()),random_state=3)])
print(f"[{len(ev)} large-cap gap events; querying {len(samp)}]", flush=True)
rows=[]
for r in samp.itertuples():
    tone,cnt = gtone(NAMES[r.sym], r.date-pd.Timedelta(days=1), r.date+pd.Timedelta(days=2))
    print(f"  {r.sym:5} {str(r.date.date())} brk={r.brk} tone={tone if tone is None else round(tone,2)} arts={cnt}", flush=True)
    rows.append({"sym":r.sym,"brk":r.brk,"tone":tone,"n":cnt}); time.sleep(12)
res=pd.DataFrame(rows); cov=res[res.tone.notna()]
print(f"\n=== {len(cov)}/{len(res)} covered ===", flush=True)
if len(cov)>=6:
    for b,g in cov.groupby("brk"):
        print(f"  {'BREAK' if b else 'BOUNCE'}: mean tone {g.tone.mean():+.2f} (n={len(g)})", flush=True)
