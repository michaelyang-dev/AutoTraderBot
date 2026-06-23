# Futures deep-dive — findings (Norgate gold-standard panel)

Data: Norgate continuous panel, **98 liquid global-macro markets, 1977–2026**, plus
27,261 individual contracts (term structure) + cash commodities. Return construction
validated: `ccb.diff()/nonadj.shift(1)` (difference back-adjusted; ES → 19.4% vol /
8.2% drift, exactly right). One consistent methodology across every test
(`futures_lib.py`): per-market vol-targeting, 1-day execution lag, 1.5bps costs,
12% portfolio-vol overlay.

## Sleeve results (vol-scaled to 12%)

| Sleeve | Full Sharpe | Recent (2015–26) | Crisis-alpha | Verdict |
|---|---|---|---|---|
| **Trend (TSMOM 1/3/12m)** | 1.18 | **0.06** | 2008 +49%, 2020 +20%, 2022 +10% | Convex crisis hedge; standalone DECAYED |
| Trend (diversifying only) | 1.18 | 0.22 | same | Dropping equity-index trend helps recent |
| **Cross-sec momentum** | 1.12 | **0.43** | 2000 +85%, 2022 +21% | Best recent; decayed least |
| **Carry (term structure)** | 0.92 | 0.19 | 2008 +22%, **2020 −6%, 2022 −5%** | Distinct (corr 0.14); earns in calm, NOT a crash hedge |
| Value (5y reversal) | −0.67 | −0.99 | — | **DEAD** (just anti-momentum). Don't use. |

## The prize — combined multi-strat book
**Trend_div + xsmom + carry (equal risk):**
- Full Sharpe **1.44**, MaxDD −24.5% (vs trend-alone 1.18 / −31.5%)
- **Recent (2015–26) Sharpe 0.43; 2020s Sharpe 0.55** — diversification largely fixes
  trend's calm-market bleed (carry is the MVP diversifier, corr 0.14 to trend)
- Crisis convexity PRESERVED: 2000 +137%, 2008 +48%, 2020 +14%, 2022 +10%; corr −0.14

## Integration with the live v12 equity book (2018–2025)
- corr(v12, futures book) ≈ **0.00** — genuinely diversifying
- **Overlay** (capital-efficient, futures on margin): modest — Sharpe 1.12→1.13,
  MaxDD −24%→−21% at 30–40% overlay; 2020 −24%→−21%, 2022 −20%→−19%
- **Reallocation** (move 25% capital to futures): MaxDD −24%→−16%, 2020 −16%, 2022
  −14%, but CAGR 21.6%→16.8% (you give up equity in a bull)
- HONEST: benefit is modest over 2018–25 because that window is a huge equity bull
  AND the futures book's weakest era, with no 2008-style crash. The value rests on
  TAIL protection (2008 +48% vs equity −45%), which the short window can't exercise.

## Dead ends (don't re-test)
- **Value (5y reversal):** dead.
- **Cross-asset regime signal to time the equity de-risk:** does NOT beat the simple
  SPY<200d rule (Sharpe 0.37 vs 0.43, worse DD). The live de-risk is already at frontier.

## Bottom line
The new data genuinely unlocks **one new thing**: a fundable diversified global-macro
futures book (trend+xsmom+carry), recent Sharpe ~0.4–0.55, real crisis convexity,
~0 correlation to the equity book. It's a legitimate diversifying return stream / tail
hedge — NOT a high-return engine (all premia decayed from their 80s–90s peaks).
Caveats: idealized (1.5bps, no market impact, daily rebal, market-selection
survivorship) — net recent Sharpe likely ~0.3–0.45. At $30K, the modest diversification
benefit is marginal vs ongoing futures data/exchange/commission costs; the case
strengthens with AUM. Live execution feasible via IBKR micros; live continuous series
buildable from IBKR and verifiable against this Norgate snapshot.

---

# Deeper dive — first-principles, not the factor zoo

**Pattern that emerged: you CANNOT use the futures panel to predict/time equities
(efficient market), but you CAN harvest it as a diversifier — and conviction-weighting
makes the book meaningfully better.**

- **Macro PCA (2000–26):** PC1 (16%) = risk axis (equities vs bonds), PC2 (10%) =
  dollar/liquidity. Real & economically interpretable — but no equity-predictive edge.
- **Crash-timing via cross-asset stress** (dollar + flight-to-quality + VIX): **DEAD.**
  "200d OR stress" is byte-identical to 200d alone; stress fires *later* than the 200d
  (2020: Feb-17 vs Feb-3). SPY<200d is at the frontier — don't complicate it.
- **Factor timing (momentum vs value) via macro state: DEAD.** IC≈0 (commod +0.03,
  rates −0.01, dollar −0.03, combined −0.007); regime-tilted Sharpe 0.32 < fixed 0.56.
  Factor timing fails OOS, as the literature warns.
- **Cross-asset lead-lag: DEAD** at the tradeable horizon. fwd-1d ICs all < 0.04;
  fwd-5d "notables" have inconsistent signs + overlap artifacts. Equities efficient.
- **★ IMPROVED BOOK — the real win.** Conviction-weighting (risk-adjusted trend
  *strength* via tanh, not just sign) + dropping equity-index trend:
  recent (2015–26) Sharpe **0.32→0.60**, 2020s **0.39→0.77**, MaxDD −25%→−23%, crisis
  convexity kept (2008 +46%, 2020 +9%, 2022 +9%). OI trend-quality filter ≈ neutral.
  (Haircut for in-sample variant selection → realistic net ~0.45–0.5.)
- **Integration with the improved book:** capital-efficient overlay (futures on margin)
  at 50% → v12 **Sharpe 1.04→1.20, CAGR 19.9%→24.2%**, drawdown flat-to-better, corr
  −0.07. Now adds *return AND Sharpe AND* crisis protection — not just modest DD.

**REVISED VERDICT:** the actionable output is the **conviction-weighted diversified
futures book as a capital-efficient overlay** on the equity strategy. It crossed from
"marginal" (v1) to "worth seriously considering": ~+0.16 Sharpe + crisis convexity at
zero capital cost (margin). Remaining honest caveats: idealized costs, $30K
capacity/contract-granularity, ongoing data/exchange fees. Case strengthens with AUM.
The equity strategy itself can't be improved by this data (timing & tilts are efficient);
the value is purely the diversifying overlay.

---

# GO/NO-GO: realistic costs + $30K capacity (futures_capacity.py)

Pushed book construction further (futures_book_v3.py): more trend horizons / risk-parity
sleeve weighting = **diminishing returns** (recent Sharpe 0.59→0.62 but worse DD; risk-
parity ≈ neutral). Conviction-weighting was the real win; we're at the construction
ceiling (~recent Sharpe 0.6, idealized).

Then the decisive stress test — integer micro/full contracts at real notionals + tiered
realistic costs (2–10bps/side by liquidity), STIR excluded, 7x lev cap, weekly rebal:

| | Sharpe (2010–26) | markets held | gross |
|---|---|---|---|
| IDEAL gross (no cost) | 0.90 | — | — |
| IDEAL net (realistic costs) | **0.34** | — | 2.6x |
| **AUM $30k** | **−0.34** | **0** | 0.05x |
| AUM $100k | −0.17 | 2 | 0.19x |
| AUM $300k | 0.17 | 7 | 0.38x |
| AUM $1M | 0.13 | 22 | 1.26x |
| AUM $3M | 0.45 | 43 | 1.63x |
| AUM $10M | 0.49 | 58 | 1.76x |

**Two decisive findings:**
1. **Realistic costs roughly HALVE the edge** (idealized 0.90 → net 0.34). The 1.5bps
   numbers overstate by ~2x. (Weekly rebal is conservative; monthly recovers some.)
2. **At $30K it is a hard NO-GO.** 0 of 41 targeted markets can hold even ONE contract;
   **no bonds** (the key crisis hedge has no micro, ~$100k notional); gross 0.05x = 95%
   uninvested. The diversified book is *physically unbuildable* at this size. It needs
   **~$2–3M+** to approximate the ideal (Sharpe ~0.45 at $3M); below ~$300k it's
   nonfunctional.

**FINAL:** the attractive overlay (v12 Sharpe 1.04→1.20) is real ON PAPER but
**unimplementable at $30K** — it requires ~$2–3M of capital to hold the contracts. Plus
realistic costs halve the standalone edge. **Decision: do NOT build at current size.**
Shelve as a validated, ready sleeve; revisit at ~$2–3M+ AUM. Norgate snapshot + scripts
are the reusable asset for that future build.
