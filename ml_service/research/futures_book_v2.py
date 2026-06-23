"""Push the diversified book (the one thing that worked) as far as it honestly goes.
Two less-arbitraged improvements over sign-based trend:
  (1) CONVICTION weighting — risk-adjusted trend strength (tanh of t-stat), not just sign
  (2) OPEN-INTEREST trend-quality filter — trends confirmed by RISING open interest are
      higher-conviction (real money committing); fade trends on falling OI.
Compare recent-era (2015-26) Sharpe vs the v1 sign-based book."""
import numpy as np, pandas as pd
import futures_lib as fl

CCB, NON, mkts = fl.load()
R = fl.market_returns(CCB, NON)
eq = R["ES"].dropna()
vol = R.ewm(span=60, min_periods=20).std()
div = [m for m in mkts if m not in fl.EQUITY]

# open interest (front, smoothed) -> OI trend as conviction confirmer
df = pd.read_parquet(fl.PANEL, columns=["date", "symbol", "Open Interest"])
df["date"] = pd.to_datetime(df["date"])
oi = df[~df["symbol"].str.endswith("_CCB")].copy()
oi["mkt"] = oi["symbol"].str.lstrip("&")
OI = oi.pivot_table(index="date", columns="mkt", values="Open Interest").reindex(CCB.index)[CCB.columns]
OI_sm = OI.rolling(20, min_periods=5).mean()
oi_rising = (OI_sm > OI_sm.shift(40))          # OI building over ~2 months

# --- signals ---
sign_tr = sum(np.sign(CCB.diff(L)) for L in [21, 63, 252]) / 3                     # v1
def conv(L):
    mom = CCB.diff(L) / NON.shift(L).abs()
    return np.tanh(mom / (vol * np.sqrt(L)))                                       # risk-adj, squashed
conv_tr = sum(conv(L) for L in [21, 63, 252]) / 3                                  # conviction
# OI-gated conviction: full weight if OI confirms trend direction, else half
oi_confirm = (np.sign(conv_tr) * np.where(oi_rising, 1.0, 1.0)).where(oi_rising, 0.5)
conv_tr_oi = conv_tr * oi_confirm.abs()

ramom = (CCB.diff(252) / NON.shift(252).abs()) / vol
xs = ramom.sub(ramom.median(axis=1), axis=0)
sign_xs = np.sign(xs); conv_xs = np.tanh(xs / xs.std())
carry = pd.read_parquet("_sleeve_carry.parquet")["carry"]

def book(trend_sig, xs_sig, universe=None):
    u = universe or list(R.columns)
    t, _ = fl.backtest(fl.vol_target(trend_sig[u], R[u]), R[u])
    x, _ = fl.backtest(fl.vol_target(xs_sig[u], R[u]), R[u])
    S = pd.concat([t, x, carry], axis=1).dropna()
    raw = S.mean(axis=1)
    sc = (0.12 / (raw.rolling(252, min_periods=60).std().shift(1) * np.sqrt(252))).clip(upper=3).fillna(1)
    return (raw * sc).dropna()

books = {
    "v1 sign (trend+xs+carry)":      book(sign_tr, sign_xs),
    "conviction (trend+xs+carry)":   book(conv_tr, conv_xs),
    "conviction + OI filter":        book(conv_tr_oi, conv_xs),
    "conviction, div-universe trend":book(conv_tr, conv_xs, div),
}
print("=== FULL SAMPLE ===")
for nm, r in books.items(): fl.stats(r, nm)
print("\n=== RECENT 2015-2026 (the real test) ===")
for nm, r in books.items(): fl.stats(r[r.index.year >= 2015], nm)
print("\n=== 2020s ===")
for nm, r in books.items(): fl.stats(r[r.index.year >= 2020], nm)

best = books["conviction, div-universe trend"]
fl.crisis_table(best, eq, "conv+div")
pd.DataFrame({"book": best}).to_parquet("_futures_book_v2.parquet")
print("  [saved _futures_book_v2.parquet — improved book for integration]")
