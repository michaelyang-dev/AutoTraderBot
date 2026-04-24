# Expanded Universe Short Signal Audit

Generated: 2026-04-23

## Executive Summary

**The expanded-universe short signal is an artifact of massive look-ahead bias, not a real trading edge.** The `is_former_sp500` feature tells the model — using 2026 knowledge — which stocks will be removed from the S&P 500 in the future. In 2015, 65% of the "former" stocks hadn't been removed yet. The model learns "stocks labeled former go down" and this label uses future information. Simple heuristics on the same universe (random, momentum, SMA) all produce **positive** forward returns (+0.3% to +1.9%), proving the universe itself is not shortable — only the leaked label makes it appear so. **STOP — do not proceed to B4.**

---

## 1. Universe Definition

**Methodology**: RETROSPECTIVE (uses future knowledge)

**Code reference**: [smallcap_short_pipeline.py:578](ml_service/smallcap_short_pipeline.py#L578)

```python
feat["is_former_sp500"] = 1.0 if sym in former_sp500 else 0.0
```

The `former_sp500` set is built at pipeline runtime (2026) using ALL historical S&P 500 removals. This flag is then applied as a **static constant** across all dates. A stock removed in 2024 is labeled `is_former_sp500=1.0` on every date from 2010 onward — including prediction dates in 2015 when the removal hasn't happened yet.

The `removal_dates` dict was computed ([line 110](ml_service/smallcap_short_pipeline.py#L110)) but **never used** — it was intended for point-in-time filtering but that logic was never implemented.

### Look-ahead contamination by prediction year

| Prediction Year | Former Stocks Not Yet Removed | % Leaked |
|----------------|-------------------------------|----------|
| 2015 | 208 / 318 | **65%** |
| 2016 | 179 / 318 | **56%** |
| 2017 | 151 / 318 | **47%** |
| 2018 | 128 / 318 | **40%** |
| 2019 | 108 / 318 | **34%** |
| 2020 | 93 / 318 | **29%** |
| 2021 | 74 / 318 | **23%** |
| 2022 | 57 / 318 | **18%** |
| 2023 | 42 / 318 | **13%** |
| 2024 | 25 / 318 | **8%** |
| 2025 | 6 / 318 | **2%** |

In 2015, the model is told "these 318 stocks will eventually leave the S&P 500" when only 110 actually had at the time. The remaining 208 are perfectly legitimate S&P 500 stocks whose future removal is unknowable.

**Assessment: BIASED — critical look-ahead violation**

---

## 2. Sample Sizes

| Year | Former Stocks | prob>0.40 obs | prob>0.40 unique symbols | prob>0.50 obs |
|------|--------------|---------------|-------------------------|---------------|
| 2015 | 120 | 567 | 12 | 307 |
| 2016 | 128 | 1,020 | 16 | 416 |
| 2017 | 128 | 777 | 14 | 400 |
| 2018 | 129 | 109 | 5 | 4 |
| 2019 | 118 | 984 | 16 | 94 |
| 2020 | 115 | 1,159 | 30 | 221 |
| 2021 | 117 | 226 | 13 | 88 |
| 2022 | 122 | 123 | 2 | 53 |
| 2023 | 121 | 123 | 3 | 94 |
| 2024 | 124 | 230 | 7 | 147 |
| 2025 | 127 | 503 | 13 | 112 |

**Totals:**
- prob>0.40: 5,821 observations across 2,118 trading days (48 unique symbols total)
- prob>0.50: 1,936 observations across 1,291 trading days

Sample size is adequate for statistics but **highly concentrated**: only 2-30 unique symbols per year are flagged high-confidence. The signal is about specific stocks, not a broad pattern.

**Assessment: Adequate sample size, but narrow stock concentration**

---

## 3. Simple Heuristic Comparison

All heuristics applied to the SAME "former S&P 500" universe, WITHOUT the ML model:

| Method | Avg 10d Fwd Ret | Notes |
|--------|----------------|-------|
| A) Random former SP500 | **+0.568%** | Just pick random stocks from the universe |
| B) Low momentum (bot 20% by ret_60d) | **+1.810%** | Contrarian bounce |
| C) Low volume (bot 20%) | **+0.331%** | Low-liquidity stocks |
| D) Below SMA200 (worst 5) | **+1.918%** | Contrarian bounce |
| E) All former SP500 equally | **+0.454%** | Equal-weight all former members |
| **ML model (prob>0.40, former)** | **-4.57%** | Uses `is_former_sp500` feature |
| **ML model (top-5, all stocks)** | **-0.67%** | Uses `is_former_sp500` feature |

**Every single heuristic produces positive forward returns on the same universe.** Random stocks from the former S&P 500 universe average +0.57% per 10-day trade. The universe itself is not shortable — former S&P 500 stocks go UP on average.

The only method producing negative returns is the ML model, which has access to the `is_former_sp500` feature — a look-ahead label. The ML model's -4.57% is not coming from fundamental deterioration patterns; it's coming from the leaked label telling the model which stocks will be removed from the index.

**Assessment: ML does NOT add value — the "signal" is entirely from the leaked feature**

---

## 4. Tradeability

For stocks appearing at prob>0.40 in the former SP500 universe (48 unique symbols, 38 with current market data):

| Metric | Value |
|--------|-------|
| Median market cap | $3.5B |
| Mean market cap | $6.3B |
| Min market cap | $1M |
| Market cap < $100M | 4/38 (11%) |
| Market cap < $1B | 7/38 (18%) |
| Market cap < $5B | 23/38 (61%) |
| Median daily dollar volume | $94M |
| Daily volume < $1M | 5/38 (13%) |
| Daily volume < $5M | 6/38 (16%) |
| Price < $5 | 4/38 (11%) |
| Price < $10 | 10/38 (26%) |

Tradeability is mixed. Median market cap ($3.5B) and volume ($94M/day) are adequate for institutional shorting. However, ~18% are micro/small-cap and ~11% are penny stocks that would be hard-to-borrow.

**Assessment: Mostly tradeable IF the signal were real (it is not)**

---

## 5. Feature Leakage

### Critical leak: `is_former_sp500`

This feature is in the FEATURE_COLS list ([smallcap_short_pipeline.py:652](ml_service/smallcap_short_pipeline.py#L652)) and is used as a training feature. The model learns that `is_former_sp500=1.0` predicts negative forward returns.

**Impact on prob_short distribution:**

| Year | Former SP500 mean prob_short | Current SP500 mean prob_short | Ratio |
|------|------------------------------|-------------------------------|-------|
| 2015 | 0.147 | 0.092 | 1.6x |
| 2020 | 0.203 | 0.099 | 2.1x |
| 2025 | 0.163 | 0.085 | 1.9x |

The model assigns 1.6-2.1x higher short probability to former SP500 stocks than current SP500 stocks. This is the direct effect of the leaked label.

### Other features: no additional leakage found

- Technical features (SMA, RSI, volatility): computed correctly from historical prices
- Fundamental features: computed using filing dates (point-in-time correct)
- Cross-sectional ranks: computed per-date (no forward leakage)

**Assessment: One critical leak (`is_former_sp500`), other features clean**

---

## 6. Final Verdict

### **Option C: Signal is BIASED (look-ahead/leakage)**

The expanded-universe short signal is an artifact of look-ahead bias, not a real trading edge.

**Bias mechanism:**
1. The pipeline labels stocks as "former S&P 500" using 2026 knowledge
2. This label is applied retrospectively to all historical dates
3. The model uses this label as a training feature
4. The model learns "former = will go down" (which is tautologically true — they were removed because they declined)
5. But this information is unknowable at prediction time

**Evidence:**
- 65% of "former" stocks haven't been removed yet when predicting 2015
- Simple heuristics on the same universe produce positive returns (+0.3% to +1.9%)
- The model assigns 1.6-2.1x higher short probability to former stocks — driven by the leaked label
- Removing the leaked label would eliminate the signal

**Recommendation: STOP B4 work. The short signal does not exist.**

The honest conclusion from all research (Phase 1, 1A, 1B, 1C, deep analysis, expanded universe) is:

> **Within any universe of US equities that can be defined without future knowledge, the ML model cannot predict which stocks will decline. The feature set — fundamentals, technicals, and market context — predicts upside dispersion but not downside dispersion. This is likely a structural property of US equity markets, not a model limitation.**
