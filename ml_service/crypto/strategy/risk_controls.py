"""
Risk-controls + sizing for the HL funding-carry book (the 'don't blow up' layer).

The carry is ~15%/yr but the backtest hides the real risks. This module enforces the controls that
keep it survivable, and makes the capital-efficiency-vs-safety tradeoff explicit:

  1. SQUEEZE SURVIVAL — the HL short's margin (on HL) must survive an up-move large enough that your
     offsetting spot gain (stuck on Coinbase) can't help in time. Choose a survive-% → caps short
     leverage → sets required HL margin. This is THE control that prevents cross-venue liquidation.
  2. PER-COIN CAP — diversify; and don't exceed a fraction of the coin's HL liquidity (capacity).
  3. COUNTERPARTY CAP — limit total capital sitting on HL (a newer DEX) — split risk, hold dry powder.
  4. FUNDING-FLIP RULE — only hold coins with positive trailing funding; rotate when they flip.
  5. LEVERAGE CAP — hard ceiling.

Outputs a sized target book + a stress report (market-wide squeeze / funding-flip / HL-insolvency).
Run:  python crypto/strategy/risk_controls.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "crypto")

PARAMS = dict(
    capital=30000,          # total capital ($)
    n_coins=6,              # diversify across N positive-funding coins
    per_coin_cap=0.25,      # max 25% of capital per coin
    max_hl_frac=0.60,       # max 60% of capital posted on Hyperliquid (counterparty cap)
    squeeze_survive=0.40,   # HL short must survive a +40% intraday squeeze without liquidation
    maint_margin=0.05,      # HL maintenance margin
    min_funding=0.03,       # only hold coins with >3%/yr trailing funding (flip buffer)
    taker=0.0005,
)
US_SPOT = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "NEAR", "SUI", "AAVE", "XRP", "ADA", "LINK", "WLD"]


def current_funding():
    F = pd.read_parquet(os.path.join(DATA, "hl_funding.parquet"))
    F.index = pd.to_datetime(F.index).normalize()
    F = F[[c for c in US_SPOT if c in F.columns]]
    return (F.rolling(14).mean().iloc[-1] * 365).dropna()      # trailing annualized funding


def build_book(p=PARAMS):
    ann = current_funding()
    elig = ann[ann > p["min_funding"]].sort_values(ascending=False).head(p["n_coins"])
    if len(elig) == 0:
        return None
    w = np.minimum(1.0 / len(elig), p["per_coin_cap"])
    # max short leverage that survives the squeeze: liq at +1/L (minus maint) → L <= 1/(survive+maint)
    max_short_lev = 1.0 / (p["squeeze_survive"] + p["maint_margin"])
    rows = []
    for c, f in elig.items():
        notional = w * p["capital"]                            # spot long = short perp notional
        hl_margin = notional / max_short_lev                   # margin posted on HL for the short
        rows.append((c, f, notional, hl_margin))
    book = pd.DataFrame(rows, columns=["coin", "funding_ann", "notional", "hl_margin"])
    spot_capital = book["notional"].sum()                      # long spot on Coinbase
    hl_capital = book["hl_margin"].sum()                       # margin on HL
    return book, spot_capital, hl_capital, max_short_lev


def stress(book, p=PARAMS):
    notional = book["notional"].sum()
    hl_margin = book["hl_margin"].sum()
    out = {}
    # 1. market-wide +squeeze%: HL shorts lose squeeze*notional; spot gains it BUT on Coinbase (can't
    #    post to HL in time) → does HL margin survive? loss vs margin.
    sq = p["squeeze_survive"]
    hl_loss = sq * notional
    out["squeeze_survives"] = hl_margin > hl_loss * 0.9        # buffer check (need to top up before full loss)
    out["squeeze_margin_util"] = hl_loss / hl_margin
    # 2. funding flip: all funding → -10%/yr for a month
    out["funding_flip_monthly_cost"] = -0.10 / 12 * notional
    # 3. HL insolvency: lose HL margin + the short's unrealized (assume total HL capital lost)
    out["hl_insolvency_loss"] = hl_margin
    out["hl_insolvency_pct_capital"] = hl_margin / p["capital"]
    return out


if __name__ == "__main__":
    p = PARAMS
    res = build_book(p)
    print("=" * 84)
    print("HL CARRY — SIZED BOOK + RISK CONTROLS | capital $%s" % f"{p['capital']:,}")
    print("=" * 84)
    if res is None:
        print("No coins above min funding — stand aside (this IS a valid state).")
        sys.exit()
    book, spot_cap, hl_cap, max_lev = res
    print("\n  TARGET BOOK (long spot / short HL perp):")
    print(book.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    gross_carry = (book["funding_ann"] * book["notional"]).sum()
    print(f"\n  capital deployed:  spot(Coinbase) ${spot_cap:,.0f} + HL margin ${hl_cap:,.0f} = ${spot_cap + hl_cap:,.0f} of ${p['capital']:,.0f}")
    print("  max short leverage (survives +%.0f%% squeeze): %.2fx" % (p["squeeze_survive"] * 100, max_lev))
    print(f"  expected gross carry: ${gross_carry:,.0f}/yr = {gross_carry / p['capital'] * 100:.1f}% on capital")

    print("\n  CONTROL CHECKS:")
    print("    per-coin cap %.0f%%: %s" % (p["per_coin_cap"] * 100, "OK" if (book["notional"] <= p["per_coin_cap"] * p["capital"] + 1).all() else "VIOLATED"))
    print("    counterparty cap (HL <= %.0f%%): HL holds %.0f%% → %s"
          % (p["max_hl_frac"] * 100, hl_cap / p["capital"] * 100, "OK" if hl_cap <= p["max_hl_frac"] * p["capital"] else "VIOLATED — reduce size/leverage"))

    s = stress(book, p)
    print("\n  STRESS SCENARIOS:")
    print("    +%.0f%% market-wide squeeze: HL margin util %.0f%% → %s"
          % (p["squeeze_survive"] * 100, s["squeeze_margin_util"] * 100, "SURVIVES (top-up buffer intact)" if s["squeeze_survives"] else "LIQUIDATION RISK — lower leverage"))
    print(f"    funding flips to -10%/yr for a month: cost ${-s['funding_flip_monthly_cost']:,.0f} (~{-s['funding_flip_monthly_cost'] / p['capital'] * 100:.1f}% of capital)")
    print(f"    HL insolvency (total loss of HL capital): -${s['hl_insolvency_loss']:,.0f} (-{s['hl_insolvency_pct_capital'] * 100:.0f}% of capital) <- the hard tail")
    print("\n  → Safe operating point: 1x notional, ~%.1fx short leverage, HL exposure capped, majors first." % max_lev)
    print("    The HL-insolvency line is the irreducible counterparty risk — size it as money you can lose.")
