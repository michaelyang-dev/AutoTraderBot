"""
Attention / behavioral alt-data test (Da et al. "In Search of Attention" style),
using FREE Wikipedia pageviews as the attention proxy. ~50 attention-heavy large
caps. Abnormal attention = log(weekly views) - trailing-8wk mean (ASVI-style).
Test: does abnormal attention predict MARKET-RELATIVE forward returns (1wk price
pressure, then reversal)? Cross-sectional IC + spike event study. Free, orthogonal
to price/news/weather, genuinely untested here.
"""
import os, sys, ssl, urllib.request, certifi, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CTX = ssl.create_default_context(cafile=certifi.where())
TMAP = {
 "AAPL":"Apple_Inc.","TSLA":"Tesla,_Inc.","NVDA":"Nvidia","AMZN":"Amazon_(company)",
 "META":"Meta_Platforms","MSFT":"Microsoft","NFLX":"Netflix","AMD":"Advanced_Micro_Devices",
 "INTC":"Intel","DIS":"The_Walt_Disney_Company","BA":"Boeing","NKE":"Nike,_Inc.",
 "SBUX":"Starbucks","MCD":"McDonald's","KO":"Coca-Cola","PEP":"PepsiCo","WMT":"Walmart",
 "TGT":"Target_Corporation","HD":"The_Home_Depot","COST":"Costco","PYPL":"PayPal",
 "UBER":"Uber","ABNB":"Airbnb","COIN":"Coinbase_Global","PLTR":"Palantir_Technologies",
 "F":"Ford_Motor_Company","GM":"General_Motors","DAL":"Delta_Air_Lines","AAL":"American_Airlines_Group",
 "CCL":"Carnival_Corporation_%26_plc","MRNA":"Moderna","PFE":"Pfizer","JNJ":"Johnson_%26_Johnson",
 "JPM":"JPMorgan_Chase","BAC":"Bank_of_America","GS":"Goldman_Sachs","V":"Visa_Inc.",
 "MA":"Mastercard","CRM":"Salesforce","ORCL":"Oracle_Corporation","ADBE":"Adobe_Inc.",
 "QCOM":"Qualcomm","MU":"Micron_Technology","XOM":"ExxonMobil","CVX":"Chevron_Corporation",
 "GE":"General_Electric","CAT":"Caterpillar_Inc.","WFC":"Wells_Fargo","C":"Citigroup","T":"AT%26T",
}


def pv(article):
    u=(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
       f"all-access/all-agents/{article}/daily/20160101/20241231")
    for _ in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(u,headers={"User-Agent":"research/1.0 (r@e.com)"}),context=CTX,timeout=45) as r:
                j=json.loads(r.read())
            it=j.get("items",[])
            return pd.Series([x["views"] for x in it],
                             index=pd.to_datetime([x["timestamp"][:8] for x in it], format="%Y%m%d"))
        except Exception as e:
            if "404" in str(e): return None
            time.sleep(3)
    return None


cache="research/_wiki_pv.parquet"
if os.path.exists(cache):
    pvdf=pd.read_parquet(cache)
else:
    t0=time.time(); cols={}
    for tk,art in TMAP.items():
        s=pv(art)
        if s is not None and len(s)>500: cols[tk]=s
        time.sleep(0.3)
    pvdf=pd.DataFrame(cols); pvdf.to_parquet(cache)
    print(f"[fetched {pvdf.shape[1]} tickers' pageviews in {time.time()-t0:.0f}s]")
pvdf.index=pd.to_datetime(pvdf.index)
print(f"[pageviews: {pvdf.shape[1]} tickers, {pvdf.index.min().date()}..{pvdf.index.max().date()}]")

cl=pd.read_parquet("data/cached_close_prices.parquet"); cl.index=pd.to_datetime(cl.index)
dr=cl.pct_change()
tickers=[t for t in pvdf.columns if t in dr.columns]

# weekly abnormal attention (ASVI): log views, minus trailing-8wk mean
wv=np.log(pvdf[tickers].resample("W-FRI").sum()+1)
asvi=wv - wv.rolling(8, min_periods=4).mean().shift(1)   # no look-ahead
# weekly market-relative returns
wret=(1+dr[tickers]).resample("W-FRI").prod()-1
spy=(1+dr["SPY"]).resample("W-FRI").prod()-1
rel=wret.sub(spy, axis=0)

# cross-sectional IC: ASVI(t) vs forward rel return(t+1), (t+2..t+4 reversal)
ic1, ic4 = [], []
for i in range(8, len(asvi)-4):
    wk=asvi.index[i]
    a=asvi.loc[wk].dropna()
    if len(a)<15: continue
    f1=rel.iloc[i+1].reindex(a.index)
    f4=rel.iloc[i+1:i+5].sum().reindex(a.index)   # next 4 weeks (reversal window)
    d1=pd.concat([a,f1],axis=1).dropna(); d4=pd.concat([a,f4],axis=1).dropna()
    if len(d1)>12: ic1.append(spearmanr(d1.iloc[:,0],d1.iloc[:,1]).statistic)
    if len(d4)>12: ic4.append(spearmanr(d4.iloc[:,0],d4.iloc[:,1]).statistic)
ic1,ic4=np.array(ic1),np.array(ic4)
print(f"\n=== abnormal-attention (ASVI) cross-sectional IC ({len(ic1)} weeks) ===")
print(f"  fwd 1wk:   IC {ic1.mean():+.4f}  t {ic1.mean()/(ic1.std()/np.sqrt(len(ic1))+1e-9):+.2f}")
print(f"  fwd 2-4wk: IC {ic4.mean():+.4f}  t {ic4.mean()/(ic4.std()/np.sqrt(len(ic4))+1e-9):+.2f}  (reversal check)")

# spike event study: top-decile ASVI -> forward returns
flat=[]
for i in range(8,len(asvi)-4):
    for tk in tickers:
        a=asvi.iloc[i].get(tk)
        if pd.notna(a):
            flat.append((a, rel.iloc[i+1].get(tk), rel.iloc[i+1:i+5].sum().get(tk)))
fl=pd.DataFrame(flat,columns=["asvi","f1","f4"]).dropna()
hi=fl[fl.asvi>=fl.asvi.quantile(0.9)]
print(f"\n  attention SPIKE (top-decile ASVI): fwd1wk {hi.f1.mean()*100:+.2f}%  fwd2-4wk {hi.f4.mean()*100:+.2f}%  n={len(hi)}")
print(f"  baseline:                          fwd1wk {fl.f1.mean()*100:+.2f}%  fwd2-4wk {fl.f4.mean()*100:+.2f}%")
