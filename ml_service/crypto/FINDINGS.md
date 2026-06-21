# Crypto Strategy Research — Honest Findings

Status as of the survivorship-complete validation. Discipline: **no inflated numbers** — every
result below is on data audited for survivorship, look-ahead, and costless-exit bias.

## TL;DR

On the survivorship-complete Binance USDT-perp universe (the exact venue a live bot would trade),
**no systematic daily edge survives out-of-sample (2023+).** Momentum, reversal, funding carry,
funding-positioning, and trend-following all either decayed after the 2020–21 mania or were
artifacts of survivorship/universe construction. **Buy-and-hold BTC (Sharpe ≈ 0.90) is the
benchmark nothing has beaten.** The only legitimate improvement found is **vol-targeting**, which
cuts drawdown (−77% → −63%) at near-identical Sharpe — a real *risk-reduction* tool, not alpha.

## Data foundation (audited)

| Dataset | What | Survivorship |
|---|---|---|
| `binance_close/qvol.parquet` | 683 USDT perps incl. delisted, daily, 2020–26 | **Complete** — symbols from CDN bucket listing (incl. dead), delisted coins NaN-terminate (27/27), no back-fill (654/654) |
| `binance_funding.parquet` | 57 perp funding series, daily (sum of 8h) | Complete for tradable set |
| `binance_close/qvol_1h.parquet` | 40 liquid majors, hourly | For intraday tests |
| `crypto2_history*.csv` | CoinMarketCap mcap universe | **Incomplete** (deprecated) — only ~50 dead coins fetchable; *caused the +47% mirage* |

Audit script: `backtest/data_integrity.py`. Costless-exit bias tested via a delisting penalty
sweep (0/30/50/100%) — result **unchanged**, because the volume-ranked book sheds a coin when its
volume collapses, before it dies.

## What was tested and ruled out

| Strategy | File | Full | **2023+** | Verdict |
|---|---|---|---|---|
| Cross-sectional momentum (crypto2) | `alt_factors.py` | +132% | +85% | **MIRAGE** — survivorship + meme-exclusion |
| Cross-sectional momentum (Binance, complete) | `momentum_binance.py` | −18% | **−33%** | Dead. No filter (age/skip/universe) recovers it |
| Funding carry (collect funding) | `funding_carry_v2.py` | +29% | ~+1–3% | Decayed to ≈ cash |
| Basis / cash-and-carry | `basis_backtest.py` | flat | flat | Compressed to financing cost |
| Short-term reversal (1–3d) | `structural_edges.py` | −94% | −92% | Dead — crypto continues, doesn't revert; fees finish it |
| Funding-as-positioning | `structural_edges.py` | +16% | **−34%** | Worked pre-2023 only (mania) |
| Trend (equal-weight basket) | `trend_majors.py` | — | — | Worse DD than BTC B&H |
| Trend (risk-parity, vol-targeted) | `trend_v2.py` | +35% | +18% | **Best Sharpe 0.88 — still ties/loses to BTC's 0.90** |
| Intraday momentum / reversal (1–24h) | `intraday_edges.py` | varies | **all <0 OOS** | Dead — short horizon fee-killed; 12/24h positive only pre-2023 |
| Session / funding-settlement drift | `intraday_edges.py` | bps | bps | Real but tiny, fee-eaten |

**Exhausted across every frequency (1h→daily) and every signal class. Nothing is positive
out-of-sample (2023+) net of fees.** Liquid crypto is efficient; mania-era alpha is gone.

## Benchmark to beat

| | Full CAGR | 2023+ CAGR | Sharpe | MaxDD |
|---|---|---|---|---|
| **BTC buy & hold** | 43.7% | 54.9% | **0.90** | −76.7% |
| 50/50 BTC-ETH | 52.4% | 36.6% | 0.97 | −76.3% |
| **BTC vol-targeted 40%** | 32.9% | 46.2% | 0.85 | **−63.3%** |

## The recommended product — risk-managed BTC/ETH beta (a drawdown reducer, NOT a Sharpe-beater)

`backtest/beta_product.py`. 60/40 BTC-ETH, vol-target 30%, 200d regime de-risk (cash when BTC<SMA200).
**Audited deeply in `backtest/beta_audit.py` — and the audit corrected an earlier overstatement.**

⚠️ **Honest sub-period truth (the full-period Sharpe was inflated by the 2020-21 mania):**

| Period | Product Sharpe | BTC Sharpe | Product DD | BTC DD |
|---|---|---|---|---|
| Full 2020–26 | 1.24 | 0.90 | −30% | −77% |
| **2023+ (out-of-sample)** | **0.95** (≈0.78 w/ exec lag) | **1.16** | −30% | −50% |

**Out-of-sample, buy-and-hold BTC has a HIGHER Sharpe than the product.** The product is **not** a
risk-adjusted-return improvement. What it genuinely delivers OOS is **lower drawdown / tail risk**:
−30% vs −50%; in BTC's 10 worst months −3% vs −22%; positive in the 2025–26 downturn (+9% vs −16%).
The cost is ~half the upside (25% vs BTC 55% in the 2023+ bull). It's "hold less crypto in downtrends"
— 43% of days in cash, ~0.59× avg exposure. **Real risk management, not alpha, not a higher Sharpe.**

Stress-tested: survives fees to 100bps (0.95→0.76), independent re-impl matches to 0.0000, ~3%/yr of
return needs cash parked in T-bills. Robust across params (`beta_robust.py`).

**Deployment (US person):** at vt30 rarely wants >1x → **spot BTC/ETH on Coinbase/Kraken**, no
perps/leverage. Live signal: `strategy/beta_signal.py`. **Use it only if you specifically want
materially less drawdown and will accept materially less return** — otherwise just hold BTC/ETH.

## Alternative-data pilots (free — Deribit DVOL, Coin Metrics on-chain)

Tested whether alpha lives in *different* data (not more price data, which is competed away).

| Pilot | Signal | Result | Verdict |
|---|---|---|---|
| On-chain flows | exchange netflow, active addrs, NVT | netflow BTC 7d IC 0.139 OOS but timing book *loses* to buy&hold | Marginal — weak, parked |
| **Vol-risk-premium** | sell vol when IV>RV (Deribit DVOL) | premium REAL (IV>RV 71%); idealized Sharpe 1.65 **but** hardened (vega-MTM tail + 0.3–0.5% spread + stop) → **~0.75 Sharpe, below BTC** | Real but sub-BTC, short-vol tail, not worth it |

`vrp_pilot.py`, `vrp_hardened.py`, `onchain_pilot.py`. **Conclusion: no free-data alpha clears the bar,
and the paid avenues are negative-EV (order-book = unwinnable latency game; option data only refines a
sub-BTC strategy). Recommend ZERO further data spend.** The free-first discipline answered it for $0.

## ✅ The one real, deployable edge — Hyperliquid funding carry (a RISK PREMIUM, not alpha)

The exhaustive search found no costless alpha — but it found one genuine, US-deployable **risk
premium**. `xvenue_inspect.py`, `xvenue_carry.py`, `us_carry.py`.

**Why it exists (structural):** retail is structurally long-biased on perps and Hyperliquid is the
retail-heavy venue, so HL funding stays persistently rich — **14.5%/yr avg, positive 79% of days,
~2–3× Binance** (BTC 15% vs 7%, ETH 24% vs 8%), and the gap is one-sided 78–82% of the time (not a
spread that flips before capture). Ethena's USDe monetizes exactly this at $5B+.

**The US-deployable trade:** long spot (Coinbase/Kraken) + short perp on Hyperliquid → collect HL
funding − premium convergence. (The clean Binance-vs-HL perp spread is bigger but Binance perps are
NOT US-accessible.)

| US carry (long spot / short HL perp, 1x) | Net CAGR |
|---|---|
| 20bp round-trip | 16.7% full / 14.9% (2024+) |
| 40bp round-trip | 11.7% / 10.3% |
| **>80bp round-trip → NEGATIVE** — cost-sensitivity is the binding constraint |

**DATA-BUG NOTE (caught in audit):** the HL `funding_fetcher` had a `len(chunk)<500` pagination break
+ a rate-limit/empty-chunk confusion that silently truncated ~half the coins (SOL→2023, BNB→2024).
Both fixed; all carry numbers above are on the RE-PULLED clean panel (60 coins, all fresh-to-end).
Audit (`carry_audit.py`): look-ahead clean (lag 1/2/3 barely move), no back-fill/forward-fill, but the
universe = current top-OI so CAGR is an UPPER bound (faded/delisted coins absent). Avg HL funding 13.3%/yr.

**The headline Sharpe (10–12) and −1% DD are a LIE of the daily data** (the carry illusion). Real risks,
none in the backtest: (1) **intraday cross-venue liquidation** — your spot gain is on Coinbase while
the HL short's margin gets hit on HL in an up-squeeze; survivable at **1× with a fat HL margin buffer**,
deadly if levered up; (2) **funding flips** ~21% of days (rotate, eat slippage); (3) **HL counterparty/
solvency** (newer DEX); (4) **capacity** (HL alt liquidity). It's ~15% for warehousing retail's leverage
demand + liquidation/counterparty risk — market-neutral, diversifies the beta book.

**Deploy path:** validate live & small first (confirm funding is captured net of HL's hourly mechanics);
1× only; size the HL margin buffer for a ~40–50% intraday squeeze; auto-rotate on funding flip; majors
first (BTC/ETH/SOL), alts cautiously; cap HL counterparty exposure. Sizing/stress: `strategy/risk_controls.py`
(shows safe book + the irreducible HL-insolvency tail). A 2nd US-accessible perp venue (dYdX) shows a
large *current* HL-vs-dYdX funding spread (dYdX funding negative, HL +15%) but dYdX's indexer only serves
~2mo history so it's a promising snapshot, not yet backtested. `xvenue_inspect.py`, `dydx_funding_fetcher.py`.

## Secondary: unlock overlay (event-driven risk filter) — `unlock_overlay.py`

Free unlock data (DefiLlama datasets CDN). Event study on 2037 cliff unlocks across 12 alts confirms:
naive "short the unlock" is **front-run** (large unlocks −12% in the 3d *before*; −26% for >5% of float),
BUT the **conditional** residual is real — a large (>2% float) unlock landing in a **risk-off regime**
(BTC<50d MA) keeps falling **−8% over the next 14d**, vs +0.4% in risk-on. Use as an alt short filter /
hold-through veto. Modest (n≈52 risk-off events, volatile alt shorts) — an overlay, not a core strategy.

## Interpretation

This mirrors the equity research conclusion ("pure momentum is the efficient frontier; ML IC≈0.014").
Liquid crypto post-mania is **efficient at every frequency tested** — the easy factor alpha of
2020–21 is arbitraged away. A retail bot's honest edge is **not** manufactured alpha; it's
**disciplined, risk-managed beta**. We validated that rigorously and refused to ship a mirage.
