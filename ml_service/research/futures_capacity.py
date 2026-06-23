"""THE GO/NO-GO TEST. A 12%-vol diversified futures book is great on paper, but it must
be built from INTEGER contracts with real notionals. At small AUM the minimum 1-contract
(even micro) is huge vs the per-market target -> rounding destroys the diversification
(esp BONDS, the key hedge, which have NO micro and ~$100k notional). This computes the
realized Sharpe vs AUM curve with realistic tiered costs + integer micro/full contracts.

USD-approx point multipliers (notional = price * mult). MICRO = micro $ per point or None.
"""
import numpy as np, pandas as pd
import futures_lib as fl

# (full_mult, micro_mult_or_None, cost_bps_per_side)  -- USD-approx
SPEC = {
 # US equity (have micros)
 "ES":(50,5,1.5),"NQ":(20,2,1.5),"YM":(5,0.5,2),"RTY":(50,5,2),"EMD":(100,None,4),
 # global equity (no micro)
 "FDAX":(27,None,3),"FESX":(11,None,2),"FCE":(11,None,4),"FSMI":(11,None,5),"FTDX":(22,None,6),
 "HSI":(6.4,None,4),"NKD":(5,None,3),"NIY":(4.5,None,5),"SNK":(5,None,6),"SCN":(1,None,8),
 "KOS":(13,None,5),"SSG":(11,None,8),"SXF":(150,None,5),
 # US rates (NO micro for price futures; micro-yield are separate instruments)
 "ZT":(2000,None,1.5),"ZF":(1000,None,1.5),"ZN":(1000,None,1.5),"TN":(1000,None,2),
 "ZB":(1000,None,1.5),"UB":(1000,None,2),"ZQ":(4167,None,2),"SR3":(2500,None,2),
 # global rates (no micro)
 "FGBS":(1080,None,2),"FGBM":(1080,None,2),"FGBL":(1080,None,2),"FGBX":(1080,None,4),
 "FOAT":(1080,None,3),"FBTP":(1080,None,4),"CGB":(750,None,4),"LLG":(1000,None,4),"SJB":(1000,None,5),
 # FX (micro M6x for majors)
 "6E":(125000,12500,1.5),"6J":(12500000,1250000,2),"6B":(62500,6250,2),"6A":(100000,10000,2),
 "6C":(100000,None,2),"6S":(125000,None,3),"6N":(100000,None,3),"6M":(500000,None,4),"DX":(1000,None,3),
 # energy (MCL micro crude)
 "CL":(1000,100,2),"BRN":(1000,None,2),"NG":(10000,None,3),"HO":(42000,None,3),"RB":(42000,None,3),"GAS":(100,None,4),
 # metals (MGC/SIL/MHG micros)
 "GC":(100,10,2),"SI":(5000,1000,3),"HG":(25000,2500,3),"PA":(100,None,6),"PL":(50,None,5),
 # grains/softs/livestock (no micro)
 "ZC":(50,None,3),"ZS":(50,None,3),"ZW":(50,None,3),"KE":(50,None,5),"MWE":(50,None,6),
 "ZL":(600,None,3),"ZM":(100,None,3),"ZO":(50,None,8),"ZR":(2000,None,8),
 "SB":(1120,None,4),"KC":(375,None,5),"CC":(10,None,5),"CT":(500,None,4),"OJ":(150,None,8),
 "LE":(400,None,5),"GF":(500,None,6),"HE":(400,None,5),
 "VX":(1000,None,4),
}
DEFAULT = (100000, None, 10)   # unknown -> assume large full-size, illiquid (untradeable small)

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
vol = R.ewm(span=60, min_periods=20).std()

# combined per-market signal (conviction trend + xs-mom + carry sign), vol-targeted
def conv(L):
    mom = CCB.diff(L) / NON.shift(L).abs(); return np.tanh(mom / (vol * np.sqrt(L)))
tr = sum(conv(L) for L in [21, 63, 252]) / 3
ramom = (CCB.diff(252) / NON.shift(252).abs()) / vol
xs = np.tanh(ramom.sub(ramom.median(axis=1), axis=0) / ramom.std())
# proper book: 3 sleeves (trend on diversifying univ, xs on all, carry on all),
# combined as the element-wise nanmean of per-market vol-target positions.
# Exclude STIR (SR3/SO3/ZQ): ultra-low vol -> pathological leverage/turnover. Cap lev 7x.
STIR = {"SR3", "SO3", "ZQ"}
keep = [m for m in R.columns if m not in STIR]
R = R[keep]; NON = NON[keep]; CCB = CCB[keep]
div = [m for m in keep if m not in fl.EQUITY]
carry_sig = pd.read_parquet("_sleeve_carry_sig.parquet").reindex(index=CCB.index, columns=keep)
pos_tr = fl.vol_target(tr[div], R[div], lev_cap=7).reindex(columns=keep)
pos_xs = fl.vol_target(xs[keep], R, lev_cap=7)
pos_ca = fl.vol_target(carry_sig, R, lev_cap=7)
eff_pos = pd.DataFrame(np.nanmean(np.stack([pos_tr.values, pos_xs.values, pos_ca.values]), axis=0),
                       index=pos_xs.index, columns=keep)

# overlay scalar to 12% book vol (rolling, no lookahead)
act = eff_pos.notna() & R.notna()
book_raw = (eff_pos * R).where(act).mean(axis=1)
sc = (0.12 / (book_raw.rolling(252, min_periods=60).std().shift(1) * np.sqrt(252))).clip(upper=3).fillna(1)
N = act.sum(axis=1).clip(lower=1)
omega = eff_pos.mul(sc, axis=0).div(N, axis=0)        # ideal per-market weight (frac of AUM notional)

mult = pd.Series({m: SPEC.get(m, DEFAULT)[0] for m in CCB.columns})
micro = pd.Series({m: (SPEC.get(m, DEFAULT)[1] or SPEC.get(m, DEFAULT)[0]) for m in CCB.columns})
cost = pd.Series({m: SPEC.get(m, DEFAULT)[2] for m in CCB.columns}) / 1e4
notional_micro = NON.mul(micro, axis=1)               # $ per (smallest) contract over time

# weekly rebalance
wk = omega.resample("W-FRI").last()
NON_wk = NON.resample("W-FRI").last()
notm_wk = notional_micro.resample("W-FRI").last()

def realized(A, rounding=True, cost_mult=1.0):
    contracts = (wk * A) / notm_wk
    if rounding:
        contracts = np.round(contracts).fillna(0)
    rw = (contracts * notm_wk / A).reindex(R.index).ffill()          # realized weight, daily
    turn = (contracts.diff().abs() * notm_wk / A)                     # weekly turnover (notional frac)
    cst = (turn * cost).sum(axis=1).reindex(R.index).fillna(0) * cost_mult
    ret = (rw.shift(1) * R).sum(axis=1) - cst
    held = (contracts.abs() >= 1).sum(axis=1)                         # markets actually held
    gross = rw.abs().sum(axis=1)
    return ret.dropna(), held, gross

def sh(r):
    r = r[r.index.year >= 2010].dropna()                             # modern era (realistic)
    return r.mean() / r.std() * np.sqrt(252) if r.std() else 0

print("=== Realized Sharpe (2010-26) vs AUM, integer micro/full contracts ===")
ig, _, g0 = realized(1e9, rounding=False, cost_mult=0)
inet, _, _ = realized(1e9, rounding=False, cost_mult=1)
print(f"  IDEAL gross (no cost, no rounding)   Sharpe {sh(ig):5.2f}")
print(f"  IDEAL net  (realistic costs)         Sharpe {sh(inet):5.2f}   gross {g0[g0.index.year>=2010].mean():.1f}x   <- cost drag")
print("  --- with integer contracts at each AUM (realistic costs) ---")
for A in [30e3, 100e3, 300e3, 1e6, 3e6, 10e6, 50e6]:
    ret, held, gross = realized(A)
    h = held[held.index.year >= 2010].mean()
    print(f"  AUM ${A/1e3:>7.0f}k        Sharpe {sh(ret):5.2f}   markets held ~{h:4.0f}   gross {gross[gross.index.year>=2010].mean():.2f}x")

# diagnostic: at $30k, what CAN you hold right now?
print("\n=== at $30k today: target vs achievable per market (recent) ===")
A = 30e3
last = wk.index[-1]
tgt = (wk.loc[last] * A)
con = np.round(tgt / notm_wk.loc[last]).fillna(0)
have = con[con.abs() >= 1]
print(f"  markets with >=1 contract: {len(have)} of {(wk.loc[last].abs()>1e-6).sum()} targeted")
for m in have.index:
    print(f"    {m}: {int(have[m]):+d} contract(s)  (1 contract = ${notm_wk.loc[last,m]:,.0f} notional, "
          f"target was ${tgt[m]:,.0f})")
bonds = [m for m in ["ZT","ZF","ZN","TN","ZB","UB","FGBL","FGBM","FGBS","FOAT","CGB"] if m in con.index]
print(f"  BONDS held (the key crisis hedge): {[m for m in bonds if abs(con.get(m,0))>=1] or 'NONE'}")
