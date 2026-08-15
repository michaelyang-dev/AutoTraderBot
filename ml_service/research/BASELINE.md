# BASELINE — current audited champion

**Last updated 2026-08-14.** Update ONLY after a result clears the full audit gate in `LOG.md`.

## Champion config (= what is deployed live, v12)

```python
{"universe": "sp1500", "mom_w": 0.50, "val_w": 0.35, "lv_w": 0.15, "sec_w": 0.0,
 "top_n": 5, "rebal_days": 20, "trailing_stop": 0.40, "cap": 0.10, "use_rp": False,
 "vol_scaling": True, "vol_target": 0.15, "vol_lookback": 40, "vol_scale_cap": 1.0,
 "initial_capital": 50_000.0, "leverage": 1.49, "integer_shares": True,
 "financing_rate": 0.063}
```

Long-only US equity, SP1500 point-in-time membership. Three sleeves blended by market breadth:
momentum (12-1 skip-month, top-5, SMA200 filter), value, low-vol-quality. 20-session rebalance,
40% per-name trailing stop, 10%-of-book position cap, inverse-vol gross scaling to a 15%×1.49
target, 1.49× closed-loop leverage financed at 6.3%, integer shares, credit gate (HY-OAS ≥ p95
expanding → halve leverage) applied at rebalance.

Research harness: `research/livemirror_backtest.py` (`LiveMirrorBacktester`), which is the only
harness that models integer shares, margin debit, financing and the credit gate jointly.
`deployed_parity=True` is the default — backtest-only inputs (short interest, price targets,
analyst estimates, EV, earnings-surprise maps) are zeroed so the backtest cannot use data live
does not have.

## Audited metrics — quote these, with the error bars

Measured over **12 monthly starts** per horizon (`research/threadCANON_confidence.py`).
Never quote a point estimate from fewer than 12 starts.

| horizon | CAGR | σ per start | 95% CI on mean | Sharpe | MaxDD |
|---|---|---|---|---|---|
| **8yr 2018-2025** | **+22.75%** | 7.07pp | **±4.00pp** | **0.79 ±0.09** | −38.2% |
| **26yr 2001-2025** | **+11.26%** | 2.75pp | **±1.56pp** | **0.51 ±0.05** | −64.8% |

8yr range across entry month alone: **+15.03% … +34.55%**. Prefer the 26yr number when the two
horizons disagree — it is ~2.5× tighter and spans dot-com + GFC + COVID + 2022.

### 🔴 CORRECTION 2026-08-14 — the 26yr numbers above UNDERSTATE the strategy

Every figure in this repo charges a **flat 6.3%/yr** on the margin debit. Actual broker financing
(DFF + 1.5pp = IBKR Pro small-balance tier) averaged **3.25%** over 2001-2025 and **1.63%** over
2009-2015 (BUGS A5). Re-measured with the real curve (`financing_curve: True`, EXP-010, 12 starts):

| 26yr, 1.49× | flat 6.3% | **REAL DFF+1.5pp** | correction |
|---|---|---|---|
| CAGR | +11.26% | **+11.97%** | **+0.71pp** |
| Sharpe | 0.509 | **0.531** | +0.021 |
| MaxDD | −64.2% | −64.1% | ~0 |

The flat rate is retained as the DEFAULT so every previously logged comparison stays
reproducible, but **+11.97% / 0.531 is the honest 26yr figure.** The 8yr correction is smaller
(actual mean 3.86% vs 6.30%) and has not yet been re-measured.

### Leverage is a preference, not a free parameter (EXP-010, 26yr, REAL financing, 12 starts)

| leverage | CAGR | Sharpe | Sortino | MaxDD | worst | avgGross |
|---|---|---|---|---|---|---|
| 1.00× | +9.37% | **0.553** | **0.776** | **−48.2%** | −54.8% | 0.797 |
| 1.25× | +10.95% | 0.544 | 0.763 | −56.8% | −63.8% | 1.000 |
| **1.49× (deployed)** | **+11.97%** | 0.531 | 0.743 | −64.1% | −71.0% | 1.194 |
| 1.75× | +12.67% | 0.517 | 0.722 | −70.8% | −77.2% | 1.404 |

Paired vs 1.00×: 1.49× buys **+2.60pp CAGR** and costs **−0.022 Sharpe (0/12 starts) and
−15.87pp MaxDD**. Monotone in leverage on every axis. This is a genuine risk/return trade the
owner must choose, not a bug — but it is not a *free* parameter, and "1.49× ≈ optimal" (repo
canon, established pre-audit on few starts) is **not supported**: Sharpe and Sortino are
maximised at the LOWEST leverage tested, on 12/12 starts.

Sharpe is computed **without subtracting the risk-free rate** (`dr.mean()/dr.std()*√252`).
It is therefore overstated by roughly `rf/vol` ≈ 0.10–0.15 versus a textbook Sharpe. This is
consistent across all arms, so *deltas* are unaffected; absolute levels are not comparable to
published Sharpes.

## Falsification status of the baseline itself

See `research/AUDIT02_falsification.py` and `BUGS.md`. Summary (8yr, 4 starts):

| test | result |
|---|---|
| cost ×2 | +20.98% / 0.76 (−1.87pp CAGR) |
| cost ×3 | +19.16% / 0.71 (−3.69pp CAGR) |
| cost ×5 | see `_audit02_8yr.out` |
| shift +1 bar | see `_audit02_8yr.out` |
| shuffle (random picks) | see `_audit02_8yr.out` |
| drop best 5/10/20 days | see `_audit02_8yr.out` |

---

## ✅ AUDITED CANDIDATE — REBALANCE-PHASE TRANCHING (not deployed)

**First result in this program to pass the full audit gate.** `research/EXP001*.py`,
`EXP014_tranching_audit_gate.py`.

**What it is.** Split capital into K sub-books, each running the identical strategy but
rebalancing on a different phase of the 20-session cycle (K=4 → phases 0, 5, 10, 15). No new
signal, no new data, no forecast.

**Why it works.** The rebalance date carries zero information — it is a pure nuisance parameter.
Measured with start date AND capital held constant, the choice of phase moves 8yr CAGR by
**σ = 7.92pp** (26yr: 1.94pp), and the deployed book stakes 100% of capital on one arbitrary
phase. No phase is systematically better (per-phase means span 4pp against a 2.04pp SEM), which
is the required confirmation that this is a nuisance and not something to optimise.

| | dCAGR | dSharpe | dMaxDD | worst DD | **sd ratio** (theory 1/√4 = 0.500) |
|---|---|---|---|---|---|
| **26yr, $50k** | **+0.52pp** | **+0.031** | +0.28pp | **−71.1% → −64.5%** | **0.409** |
| 8yr, $50k | +2.87pp | +0.094 | +1.04pp | −45.5% → −44.5% | 0.571 |
| 8yr, **$33k real size** | +2.57pp | +0.086 | +0.2pp | −45.2% → −45.2% | 0.560 |
| 8yr, $33k, **cost ×3** | +2.48pp | +0.081 | — | — | 0.569 |

**Quote the 26yr.** The 8yr gain is ~4× larger only because phase risk is loudest there. Of the
8yr's +2.87pp, ~+1.1pp is arithmetic (recovered variance drag: the CAGR of an average path
exceeds the average of the paths' CAGRs) and the rest is within noise.

**Audit gate — all passed:**
- parity: `rebal_phase=0` ≡ untouched path, bit-for-bit
- capital control: effect is PHASE, not book size (−0.35pp default / **+0.51pp** live sizing)
- live-sizing control: holds under the closed-loop quantity calibration the engine uses
- **real account size $33k**: dSharpe +0.086 vs +0.094 at $50k
- **cost ×2 / ×3**: dSharpe +0.084 / **+0.081** — almost flat, as expected mechanically
  (tranching changes *when* dollars trade, not how many)
- **event concentration: 11-28%** of excess from top-5 days (reject threshold 50%)
- both horizons positive

**Sign consistency is 8/12, and that is expected** — base is one draw from the phase
distribution, K=4 is its average, so base wins ~half the starts by construction. The
pre-registered statistic was the sd ratio and it hit its theoretical value on both horizons.

**🔴 BLOCKER — engineering, not research.** `ibkr_engine` holds ONE book. K=4 requires four
target books netted into one IBKR account with per-tranche holdings tracked. The cheap
single-book approximation (partial adjustment) was tested and **destroys the strategy**
(EXP-007: −7 to −10pp CAGR), so this cost cannot be avoided. **K=2 captures ~half the benefit
(dSharpe +0.041 at $33k) for half the complexity** and is the sensible first step.

---

## ✅ AUDITED CANDIDATE 2 — REMOVE THE INVERSE-VOL OVERLAY (not deployed)

`research/EXP015_overlay_vs_constant.py`, `EXP017_constant_vs_deployed.py`,
`EXP018_constant_cost_audit.py`.

**The change:** `vol_scaling: False`, `leverage: 1.00` or `1.10`. A config change that **removes
code** — it deletes the 40-day vol estimator, the leverage churn it causes, and the stale-vol
failure mode. No new signal, data, or live machinery.

**Why:** at MATCHED average gross the overlay has **negative** timing skill on both horizons —
it costs −0.91pp CAGR / −0.024 Sharpe (26yr) and −0.99pp / −0.017 (8yr), and buys +2.89pp of
MaxDD on the 26yr but only +0.59pp on the 8yr, where it makes worst-case drawdown 1.24pp *worse*.
It is a ~1pp-per-year option that only pays in a genuine credit crisis.

**Paired per-start, 12 starts, REAL time-varying financing (which favours the deployed arm):**

| | 26yr dCAGR | 26yr dSharpe | 26yr dMaxDD | 8yr dCAGR | 8yr dSharpe | 8yr dMaxDD |
|---|---|---|---|---|---|---|
| **B const 1.00×** | −0.58pp | **+0.038 (12/12)** | **+6.40pp (12/12)** | −1.78pp | +0.035 (7/12) | **+5.60pp (12/12)** |
| **C const 1.10×** | **+0.14pp** | **+0.034 (12/12)** | **+2.49pp (12/12)** | −0.20pp | +0.026 (8/12) | +2.15pp (8/12) |

Absolute, 26yr: deployed +11.97% / 0.531 / −64.1% → **C: +12.11% / 0.564 / −61.6%**;
**B: +11.38% / 0.569 / −57.7%**.

**Audit — passed:** cost sensitivity (C's dSharpe **grows** +0.026→+0.028→+0.030 and dCAGR
improves −0.20→−0.00→+0.19pp as cost triples, confirming the gain is a turnover reduction);
event concentration (**28%**, threshold 50%); honest financing; not a knife-edge (1.00×/1.10×/
1.20× all positive, all 12/12 on 26yr Sharpe).

**⚠️ Does NOT fully clear the ≥9/12-on-both bar.** Decisive on the 26yr (12/12 Sharpe and
drawdown), **marginal at 8/12 on the 8yr**. Never negative on either horizon. Recorded as
marginal-to-strong, not a clean pass.

**The most sign-consistent result in the program:** drawdown improves in **12/12 starts on BOTH
horizons** (+6.40pp / +5.60pp at 1.00×). EXP-004 established drawdown is the axis where this
strategy is genuinely behind passive.

**Trade-off is the owner's:** 1.00× = maximum drawdown protection at −0.6 to −1.8pp CAGR;
1.10× = CAGR-neutral with smaller protection.

---

## Bar a challenger must clear

1. **≥12 monthly starts** per horizon, judged on **sign-consistency**, not the mean.
   6/12 = coin flip = dead. Target ≥9/12 on Sharpe and ≥8/12 on CAGR.
2. **Positive on BOTH horizons** (8yr and 26yr). One-horizon results have been wrong before
   (`umd_crash`: +0.026 8yr / −0.020 26yr).
3. **Matched-exposure control** for anything that changes gross exposure — score against a
   constant-leverage curve interpolated at the variant's own realised `avg_gross`, otherwise
   any de-risking rule looks like genius purely by running less.
4. **Event-concentration test** — what share of total excess comes from the top 5/10/20 days?
   >50% from 5 days = rejected. Start-consistency and walk-forward share a blind spot: every
   start window inside a horizon contains the *same* crisis.
5. **Walk-forward OOS** with annual re-selection, selection cost fully paid.
6. **Cost sensitivity** — must survive 2× modelled cost (modelled = 10 bps round trip;
   measured live = 6.10 bps, so 2× modelled ≈ 3.3× reality).
7. **Live-implementability** — it must be reproducible by `ibkr_engine` + `signal_builder`
   with data the live box actually has. Anything needing backtest-only maps is dead on arrival.
8. **Multiple-testing deflation** — report how many cells/configs were examined. A t of −2.18
   from 6 cells is p ≈ 0.18 after Bonferroni, not 0.03 (EXP-006).
9. **Hit rate inside the selected book, not pool-wide IC** — the momentum sleeve takes the top 5
   of a ~1,100-name eligible pool. A screen with a real pool-wide effect can still be worth
   nothing because it almost never touches a name that reaches the book (EXP-006). Evaluate
   screens on how often they change an actual pick.
