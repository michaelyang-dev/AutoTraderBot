"""Carry sleeve from the individual-contracts term structure. Carry = annualized
roll yield = slope between the front and 2nd contract. Backwardation (front>2nd) =>
positive carry => go long. Signal comes from term structure; returns come from the
SAME validated continuous panel, so methodology matches every other sleeve."""
import numpy as np, pandas as pd
import futures_lib as fl

MONTH = {"F":1,"G":2,"H":3,"J":4,"K":5,"M":6,"N":7,"Q":8,"U":9,"V":10,"X":11,"Z":12}

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
eq = R["ES"].dropna()

ind = pd.read_parquet("../data/norgate/norgate_futures.parquet",
                      columns=["date","symbol","Close","Open Interest"])
ind["date"] = pd.to_datetime(ind["date"])
# parse ROOT-YYYYM
parts = ind["symbol"].str.extract(r"^(.+)-(\d{4})([FGHJKMNQUVXZ])$")
valid = parts[0].notna() & parts[1].notna() & parts[2].notna()
ind = ind[valid].copy(); parts = parts[valid]
ind["root"] = parts[0].str.lstrip("&").values
ind = ind[ind["root"].isin(mkts)]; parts = parts.loc[ind.index]
ind["expiry"] = pd.to_datetime(dict(year=parts[1].astype(int),
                                    month=parts[2].map(MONTH), day=15), errors="coerce")
ind = ind.dropna(subset=["expiry"])
ind["dte"] = (ind["expiry"] - ind["date"]).dt.days
# liquid, near contracts only (front + a few) — keeps front/2nd identification clean
ind = ind[(ind["dte"] > 5) & (ind["dte"] < 400) & (ind["Open Interest"] > 0) & (ind["Close"] > 0)]
print(f"[carry: {len(ind):,} contract-days, {ind['root'].nunique()} markets]")

ind = ind.sort_values(["date","root","expiry"])
ind["rk"] = ind.groupby(["date","root"]).cumcount()
front = ind[ind.rk == 0][["date","root","Close","expiry"]].rename(columns={"Close":"cf","expiry":"ef"})
second = ind[ind.rk == 1][["date","root","Close","expiry"]].rename(columns={"Close":"cs","expiry":"es"})
m = front.merge(second, on=["date","root"])
gap = (m["es"] - m["ef"]).dt.days.clip(lower=20)
m["carry"] = (m["cf"] - m["cs"]) / m["cs"] * (365.0 / gap)   # annualized roll yield

carry = m.pivot_table(index="date", columns="root", values="carry").reindex(CCB.index).ffill(limit=10)
carry = carry.reindex(columns=R.columns)

# smooth slightly (term structure is noisy day to day), then signal
csig = carry.rolling(5, min_periods=1).mean()
sig_carry = np.sign(csig)                    # long backwardated, short contango

carry_pnl, _ = fl.backtest(fl.vol_target(sig_carry, R), R)

print("\n=== CARRY sleeve ===")
fl.stats(carry_pnl, "carry (sign)")
fl.stats(carry_pnl[carry_pnl.index.year >= 2015], "carry (2015-2026)")
fl.decade_table(carry_pnl)
fl.crisis_table(carry_pnl, eq, "carry")
j = carry_pnl.index.intersection(eq.index)
print(f"\n  corr(carry, equity): {np.corrcoef(carry_pnl[j], eq[j])[0,1]:+.2f}")

# save sleeve returns + the per-market carry SIGNAL for combined-book / capacity tests
pd.DataFrame({"carry": carry_pnl}).to_parquet("_sleeve_carry.parquet")
sig_carry.to_parquet("_sleeve_carry_sig.parquet")
print("  [saved _sleeve_carry.parquet, _sleeve_carry_sig.parquet]")
