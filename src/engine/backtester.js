// ══════════════════════════════════════════
//  BACKTESTER — Production-quality simulation engine
//  Signal-on-close → execute-at-next-open (T+1)
//  Prevents look-ahead bias via incremental history
// ══════════════════════════════════════════

import { RISK, UNIVERSE, INITIAL_CASH, getSector } from "../config/constants";
import { getSignals } from "./signalEngine";
import { atr, avgVolume, sma } from "./indicators";
import { computeRegime, REGIME_RECOVERY_DAYS } from "./regimeEngine";
import { computeTrendStatus } from "./trendEngine";

// ── Constants ──
const COMMISSION_PER_SHARE = 0.005;   // $0.005/share (Interactive Brokers-like)
const SLIPPAGE_PCT         = 0.0005;  // 0.05% market impact on fill
const MIN_VOLUME_FOR_FILL  = 10000;   // shares: below this, partial fill logic kicks in
const EARNINGS_BLACKOUT    = 2;       // skip buys N trading days before earnings
const MAX_POSITION_HOLD    = 60;      // force-close positions held > 60 trading days

// ── Trend-following constants ──
const TREND_ALLOC_PCT     = 0.05;  // fixed 5% allocation per trend position
const TREND_MAX_POSITIONS = 8;     // max simultaneous trend positions
const TREND_MAX_PORT_PCT  = 0.30;  // max 30% of portfolio in trend positions
const TREND_MAX_COMBINED  = 0.15;  // max 15% combined per symbol (consensus + trend)
const TREND_TRAILING_STOP = 0.10;  // 10% trailing stop for trend positions
const TREND_BREAK_DAYS    = 3;     // exit after N consecutive closes below 200-SMA

// ── Utility ──
function dateStr(d) { return d.toISOString().split("T")[0]; }
function addDays(d, n) { const r = new Date(d); r.setDate(r.getDate() + n); return r; }
function tradingDaysBetween(dates, from, to) {
  return dates.filter((d) => d >= from && d <= to);
}

/**
 * Build a sorted list of unique trading days from bar data.
 */
function buildTradingCalendar(daily) {
  const seen = new Set();
  for (const bars of Object.values(daily)) {
    bars.forEach((b) => seen.add(b.t.split("T")[0]));
  }
  return Array.from(seen).sort();
}

/**
 * Build per-symbol indexed bar maps for O(1) date lookups.
 * Returns { [sym]: { [date]: { o, h, l, c, v } } }
 */
function indexBars(daily) {
  const idx = {};
  for (const [sym, bars] of Object.entries(daily)) {
    idx[sym] = {};
    bars.forEach((b) => { idx[sym][b.t.split("T")[0]] = b; });
  }
  return idx;
}

/**
 * Get close-price history up to and including `date` for a symbol.
 * THROWS if `afterDate` prices are present — look-ahead bias guard.
 */
function getHistoryUpTo(barIdx, sym, date) {
  const symBars = barIdx[sym];
  if (!symBars) return [];
  const prices = [];
  for (const [d, bar] of Object.entries(symBars)) {
    if (d > date) continue; // strict: no future bars
    prices.push({ d, c: bar.c });
  }
  prices.sort((a, b) => (a.d < b.d ? -1 : 1));
  return prices.map((p) => p.c);
}

/**
 * Same for volumes.
 */
function getVolumeHistoryUpTo(barIdx, sym, date) {
  const symBars = barIdx[sym];
  if (!symBars) return [];
  const vols = [];
  for (const [d, bar] of Object.entries(symBars)) {
    if (d > date) continue;
    vols.push({ d, v: bar.v });
  }
  vols.sort((a, b) => (a.d < b.d ? -1 : 1));
  return vols.map((p) => p.v);
}

// ── Metrics computation ──

function computeMetrics(equityCurve, initialCash, trades, tradingDays, spyReturns = null) {
  if (equityCurve.length < 2) return null;

  const finalValue = equityCurve[equityCurve.length - 1].value;
  const years = tradingDays / 252;
  const cagr = years > 0 ? Math.pow(finalValue / initialCash, 1 / years) - 1 : 0;

  // Daily returns
  const returns = [];
  for (let i = 1; i < equityCurve.length; i++) {
    returns.push((equityCurve[i].value - equityCurve[i - 1].value) / equityCurve[i - 1].value);
  }

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  const sharpe = stdDev > 0 ? (mean / stdDev) * Math.sqrt(252) : 0;

  // Sortino (downside deviation only)
  const downsideReturns = returns.filter((r) => r < 0);
  const downsideVar = downsideReturns.length > 0
    ? downsideReturns.reduce((a, b) => a + b ** 2, 0) / downsideReturns.length : 0;
  const sortino = downsideVar > 0 ? (mean / Math.sqrt(downsideVar)) * Math.sqrt(252) : 0;

  // Max drawdown
  let peak = equityCurve[0].value;
  let maxDD = 0;
  let ddStart = equityCurve[0].date;
  let ddPeak = ddStart;
  let ddTrough = ddStart;
  let currentDDStart = ddStart;
  let currentPeak = peak;

  for (const pt of equityCurve) {
    if (pt.value > currentPeak) {
      currentPeak = pt.value;
      currentDDStart = pt.date;
    }
    const dd = (pt.value - currentPeak) / currentPeak;
    if (dd < maxDD) {
      maxDD = dd;
      ddPeak = currentDDStart;
      ddTrough = pt.date;
    }
    if (pt.value > peak) peak = pt.value;
  }

  const calmar = maxDD < 0 ? cagr / Math.abs(maxDD) : 0;

  // Trade statistics
  const closedTrades = trades.filter((t) => t.exitDate);
  const wins = closedTrades.filter((t) => t.pnl > 0);
  const losses = closedTrades.filter((t) => t.pnl <= 0);
  const winRate = closedTrades.length > 0 ? wins.length / closedTrades.length : 0;
  const avgWinPct = wins.length > 0
    ? wins.reduce((a, t) => a + t.pnlPct, 0) / wins.length : 0;
  const avgLossPct = losses.length > 0
    ? losses.reduce((a, t) => a + t.pnlPct, 0) / losses.length : 0;
  const grossProfit = wins.reduce((a, t) => a + t.pnl, 0);
  const grossLoss   = Math.abs(losses.reduce((a, t) => a + t.pnl, 0));
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;
  const avgHoldDays  = closedTrades.length > 0
    ? closedTrades.reduce((a, t) => a + (t.holdDays || 0), 0) / closedTrades.length : 0;

  // Monthly returns
  const monthlyMap = {};
  for (let i = 1; i < equityCurve.length; i++) {
    const month = equityCurve[i].date.slice(0, 7);
    if (!monthlyMap[month]) monthlyMap[month] = { start: equityCurve[i - 1].value, end: equityCurve[i].value };
    else monthlyMap[month].end = equityCurve[i].value;
  }
  const monthlyReturns = Object.entries(monthlyMap).map(([month, { start, end }]) => ({
    month,
    return: (end - start) / start,
  }));

  // Exit reason breakdown
  const exitReasons = {};
  closedTrades.forEach((t) => {
    exitReasons[t.exitReason] = (exitReasons[t.exitReason] || 0) + 1;
  });

  // Beta & Alpha vs SPY (Jensen's alpha, Rf = 0)
  let beta  = null;
  let alpha = null;
  if (spyReturns && spyReturns.length >= 10 && returns.length >= 10) {
    const n = Math.min(returns.length, spyReturns.length);
    const portR = returns.slice(-n);
    const spyR  = spyReturns.slice(-n);
    const spyMean  = spyR.reduce((a, b) => a + b, 0) / n;
    const portMean = portR.reduce((a, b) => a + b, 0) / n;
    let cov = 0, varSpy = 0;
    for (let i = 0; i < n; i++) {
      cov    += (portR[i] - portMean) * (spyR[i] - spyMean);
      varSpy += (spyR[i] - spyMean) ** 2;
    }
    cov    /= n;
    varSpy /= n;
    beta  = varSpy > 0 ? cov / varSpy : null;
    // Annualised Jensen's alpha: excess return not explained by beta * market return
    alpha = beta != null ? (portMean - beta * spyMean) * 252 : null;
  }

  return {
    cagr,
    sharpe,
    sortino,
    maxDrawdownPct: maxDD,
    calmar,
    winRate,
    avgWinPct,
    avgLossPct,
    profitFactor,
    avgHoldDays,
    totalTrades: closedTrades.length,
    grossProfit,
    grossLoss,
    netPnL: finalValue - initialCash,
    finalValue,
    monthlyReturns,
    exitReasons,
    drawdownPeriod: { peak: ddPeak, trough: ddTrough },
    tradingDays,
    years,
    beta,
    alpha,
  };
}

// ── Layer breakdown stats ──

function computeLayerStats(trades) {
  const closed      = trades.filter((t) => t.exitDate && t.pnl != null);
  const consTrades  = closed.filter((t) => !t.layer || t.layer === "consensus");
  const trendTrades = closed.filter((t) => t.layer === "trend");

  const stats = (ts) => {
    if (!ts.length) return null;
    const wins   = ts.filter((t) => t.pnl > 0);
    const losses = ts.filter((t) => t.pnl <= 0);
    return {
      count:       ts.length,
      winRate:     wins.length / ts.length,
      grossProfit: wins.reduce((a, t) => a + t.pnl, 0),
      grossLoss:   Math.abs(losses.reduce((a, t) => a + t.pnl, 0)),
      netPnl:      ts.reduce((a, t) => a + t.pnl, 0),
      avgPnlPct:   ts.reduce((a, t) => a + t.pnlPct, 0) / ts.length,
      avgHoldDays: ts.reduce((a, t) => a + (t.holdDays ?? 0), 0) / ts.length,
    };
  };
  return { consensus: stats(consTrades), trend: stats(trendTrades) };
}

// ── Core simulation loop ──

/**
 * Simulate trading over a date range using pre-built barIdx.
 * @param {object} barIdx       - { [sym]: { [date]: { o,h,l,c,v } } }
 * @param {string[]} calendar   - sorted trading day strings for the full dataset
 * @param {string} simStart     - first trading day to simulate (after warmup)
 * @param {string} simEnd       - last trading day to simulate
 * @param {object} earnings     - { [sym]: string[] } of earnings dates
 * @param {number} initialCash
 * @param {string} label        - "in-sample" / "out-of-sample" / "full"
 * @returns {{ equityCurve, trades, metrics, label }}
 */
function runSimulation(barIdx, calendar, simStart, simEnd, earnings, initialCash, label) {
  // Restrict calendar to simulation window
  const simDays = calendar.filter((d) => d >= simStart && d <= simEnd);
  if (simDays.length === 0) return { equityCurve: [], trades: [], metrics: null, label };

  // State
  let cash = initialCash;
  const positions = {};   // { [sym]: { shares, avgPrice, entryDate, entryDayIdx, peakPrice, stopLevel } }
  const trades = [];
  const equityCurve = [];
  const cooldowns = {};   // { [sym]: dayIdx }
  let pendingBuys = [];   // orders queued at close, executed at next open
  let pendingSells = [];  // same
  const COOLDOWN_DAYS = 3; // trading-day cooldown after loss exit

  // Trend layer state
  const trendPositions = {};   // { [sym]: { shares, avgPrice, entryDate, entryDayIdx, peakPrice } }
  const trendBreakCount = {};  // { [sym]: number } — consecutive closes below 200-SMA
  let pendingTrendBuys  = [];  // queued at close, executed at next open
  let pendingTrendSells = [];  // same

  // Regime state
  let prevRegime = "BULLISH";
  let consecutiveDaysAbove50 = 0;

  const getPortfolioValue = (dayDate) => {
    const consVal  = Object.entries(positions).reduce((sum, [sym, pos]) => {
      const bar = barIdx[sym]?.[dayDate];
      return sum + (bar ? bar.c * pos.shares : pos.avgPrice * pos.shares);
    }, 0);
    const trendVal = Object.entries(trendPositions).reduce((sum, [sym, pos]) => {
      const bar = barIdx[sym]?.[dayDate];
      return sum + (bar ? bar.c * pos.shares : pos.avgPrice * pos.shares);
    }, 0);
    return cash + consVal + trendVal;
  };

  const isOnCooldown = (sym, dayIdx) => {
    const lossDay = cooldowns[sym];
    return lossDay !== undefined && (dayIdx - lossDay) < COOLDOWN_DAYS;
  };

  const daysUntilEarnings = (sym, dayDate) => {
    const dates = earnings[sym];
    if (!dates) return Infinity;
    const future = dates.filter((d) => d > dayDate).sort();
    if (future.length === 0) return Infinity;
    // Count trading days
    const futureTradingDays = calendar.filter((d) => d > dayDate && d <= future[0]);
    return futureTradingDays.length;
  };

  for (let di = 0; di < simDays.length; di++) {
    const today = simDays[di];

    // ── Phase A: Execute yesterday's pending orders at today's OPEN ──
    for (const order of pendingBuys) {
      const bar = barIdx[order.sym]?.[today];
      if (!bar) continue;

      // Gap detection: if stock gapped down > 3% from prior close, skip
      const priorBar = di > 0 ? barIdx[order.sym]?.[simDays[di - 1]] : null;
      if (priorBar && (bar.o - priorBar.c) / priorBar.c < -0.03) continue;

      // Partial fill: if today's volume is < 10k shares, skip (illiquid)
      if (bar.v < MIN_VOLUME_FOR_FILL) continue;

      // Slippage: pay slightly above open
      const fillPrice = bar.o * (1 + SLIPPAGE_PCT);
      const commission = order.shares * COMMISSION_PER_SHARE;
      const totalCost  = order.shares * fillPrice + commission;

      if (totalCost > cash) {
        // Recalculate affordable shares
        const affordShares = Math.floor((cash - commission) / fillPrice);
        if (affordShares <= 0) continue;
        order.shares = affordShares;
        order.totalCost = affordShares * fillPrice + affordShares * COMMISSION_PER_SHARE;
      }

      cash -= order.shares * fillPrice + commission;
      positions[order.sym] = {
        shares:     order.shares,
        avgPrice:   fillPrice,
        entryDate:  today,
        entryDayIdx: di,
        peakPrice:  fillPrice,
      };
    }
    pendingBuys = [];

    // Execute pending sells at today's open
    for (const order of pendingSells) {
      const bar = barIdx[order.sym]?.[today];
      if (!bar || !positions[order.sym]) continue;

      const pos = positions[order.sym];
      // Slippage: sell slightly below open
      const fillPrice = bar.o * (1 - SLIPPAGE_PCT);
      const commission = pos.shares * COMMISSION_PER_SHARE;
      const revenue    = pos.shares * fillPrice - commission;
      const pnl        = revenue - pos.shares * pos.avgPrice;
      const pnlPct     = pnl / (pos.shares * pos.avgPrice);
      const holdDays   = di - pos.entryDayIdx;

      cash += revenue;
      trades.push({
        sym:        order.sym,
        side:       "sell",
        entryDate:  pos.entryDate,
        exitDate:   today,
        entryPrice: pos.avgPrice,
        exitPrice:  fillPrice,
        shares:     pos.shares,
        pnl,
        pnlPct,
        holdDays,
        exitReason: order.reason,
      });

      if (pnl < 0) cooldowns[order.sym] = di;
      delete positions[order.sym];
    }
    pendingSells = [];

    // ── Phase A (trend): Execute pending trend sells at today's open ──
    for (const order of pendingTrendSells) {
      const bar = barIdx[order.sym]?.[today];
      if (!bar || !trendPositions[order.sym]) continue;

      const pos        = trendPositions[order.sym];
      const fillPrice  = bar.o * (1 - SLIPPAGE_PCT);
      const commission = pos.shares * COMMISSION_PER_SHARE;
      const revenue    = pos.shares * fillPrice - commission;
      const pnl        = revenue - pos.shares * pos.avgPrice;
      const pnlPct     = pnl / (pos.shares * pos.avgPrice);
      const holdDays   = di - pos.entryDayIdx;

      cash += revenue;
      trades.push({
        sym:        order.sym,
        side:       "sell",
        entryDate:  pos.entryDate,
        exitDate:   today,
        entryPrice: pos.avgPrice,
        exitPrice:  fillPrice,
        shares:     pos.shares,
        pnl,
        pnlPct,
        holdDays,
        exitReason: order.reason,
        layer:      "trend",
      });

      if (pnl < 0) cooldowns[order.sym] = di;
      delete trendPositions[order.sym];
      delete trendBreakCount[order.sym];
    }
    pendingTrendSells = [];

    // ── Phase A (trend): Execute pending trend buys at today's open ──
    for (const order of pendingTrendBuys) {
      if (trendPositions[order.sym]) continue; // duplicate guard
      const bar = barIdx[order.sym]?.[today];
      if (!bar) continue;

      // Gap-down guard
      const priorBar = di > 0 ? barIdx[order.sym]?.[simDays[di - 1]] : null;
      if (priorBar && (bar.o - priorBar.c) / priorBar.c < -0.03) continue;
      if (bar.v < MIN_VOLUME_FOR_FILL) continue;

      const fillPrice  = bar.o * (1 + SLIPPAGE_PCT);
      const commission = order.shares * COMMISSION_PER_SHARE;
      let   shares     = order.shares;

      if (shares * fillPrice + commission > cash) {
        shares = Math.floor((cash - commission) / fillPrice);
        if (shares <= 0) continue;
      }

      cash -= shares * fillPrice + commission;
      trendPositions[order.sym] = {
        shares,
        avgPrice:    fillPrice,
        entryDate:   today,
        entryDayIdx: di,
        peakPrice:   fillPrice,
      };
    }
    pendingTrendBuys = [];

    // ── Phase B: Stop/TP check using today's OHLC ──
    for (const [sym, pos] of Object.entries(positions)) {
      const bar = barIdx[sym]?.[today];
      if (!bar) continue;

      // Update trailing peak
      if (bar.h > pos.peakPrice) pos.peakPrice = bar.h;

      const stopLevel    = pos.peakPrice * (1 - RISK.TRAILING_STOP_PCT);
      const tpLevel      = pos.avgPrice  * (1 + RISK.TAKE_PROFIT_PCT);
      const maxHoldLimit = pos.entryDayIdx + MAX_POSITION_HOLD;

      let exitReason = null;
      let exitPrice  = null;

      // Check if price gapped through stop at open
      if (bar.o <= stopLevel) {
        exitReason = "stop";
        exitPrice  = bar.o; // gap-down fill at open
      } else if (bar.l <= stopLevel) {
        exitReason = "stop";
        exitPrice  = stopLevel; // intra-day stop at exact level
      } else if (bar.h >= tpLevel) {
        exitReason = "takeProfit";
        exitPrice  = tpLevel;
      } else if (di >= maxHoldLimit) {
        exitReason = "maxHold";
        exitPrice  = bar.c;
      }

      if (exitReason) {
        const commission = pos.shares * COMMISSION_PER_SHARE;
        const revenue    = pos.shares * exitPrice - commission;
        const pnl        = revenue - pos.shares * pos.avgPrice;
        const pnlPct     = pnl / (pos.shares * pos.avgPrice);
        const holdDays   = di - pos.entryDayIdx;

        cash += revenue;
        trades.push({
          sym,
          side:       "sell",
          entryDate:  pos.entryDate,
          exitDate:   today,
          entryPrice: pos.avgPrice,
          exitPrice,
          shares:     pos.shares,
          pnl,
          pnlPct,
          holdDays,
          exitReason,
        });

        if (pnl < 0) cooldowns[sym] = di;
        delete positions[sym];
      }
    }

    // ── Phase B': Trend trailing stop (10%) using today's OHLC ──
    for (const [sym, pos] of Object.entries(trendPositions)) {
      const bar = barIdx[sym]?.[today];
      if (!bar) continue;

      if (bar.h > pos.peakPrice) pos.peakPrice = bar.h;
      const stopLevel = pos.peakPrice * (1 - TREND_TRAILING_STOP);

      let exitPrice  = null;
      let exitReason = null;
      if (bar.o <= stopLevel) {
        exitReason = "trendStop";
        exitPrice  = bar.o;
      } else if (bar.l <= stopLevel) {
        exitReason = "trendStop";
        exitPrice  = stopLevel;
      }

      if (exitReason) {
        const commission = pos.shares * COMMISSION_PER_SHARE;
        const revenue    = pos.shares * exitPrice - commission;
        const pnl        = revenue - pos.shares * pos.avgPrice;
        const pnlPct     = pnl / (pos.shares * pos.avgPrice);
        const holdDays   = di - pos.entryDayIdx;

        cash += revenue;
        trades.push({
          sym,
          side:       "sell",
          entryDate:  pos.entryDate,
          exitDate:   today,
          entryPrice: pos.avgPrice,
          exitPrice,
          shares:     pos.shares,
          pnl,
          pnlPct,
          holdDays,
          exitReason,
          layer:      "trend",
        });

        if (pnl < 0) cooldowns[sym] = di;
        delete trendPositions[sym];
        delete trendBreakCount[sym];
      }
    }

    // ── Phase C: Signal generation on today's close ──
    // Build price and volume history up to today (no future bars)
    const spyPrices = getHistoryUpTo(barIdx, "SPY", today);
    let regimeMultiplier = 1.0;

    if (spyPrices.length >= 200) {
      const { regime: rawRegime, sma50, consecutiveDaysAbove50: cda } = computeRegime(spyPrices);

      let effectiveRegime = rawRegime;
      if (prevRegime === "BEARISH" && rawRegime !== "BEARISH" && cda < REGIME_RECOVERY_DAYS) {
        effectiveRegime = "BEARISH";
      }
      prevRegime = effectiveRegime;

      if (effectiveRegime === "BEARISH") regimeMultiplier = 0;
      else if (effectiveRegime === "CAUTIOUS") regimeMultiplier = 0.75;
    }

    // Sell signals: queue sell orders for next open
    for (const sym of Object.keys(positions)) {
      const prices = getHistoryUpTo(barIdx, sym, today);
      if (prices.length < 35) continue;
      const analysis = getSignals(prices);
      if (analysis.consensus === "SELL" || analysis.consensus === "STRONG SELL") {
        pendingSells.push({ sym, reason: "signal" });
      }
    }

    // Buy signals: collect opportunities if regime allows
    if (regimeMultiplier > 0) {
      const opportunities = [];

      for (const { sym } of UNIVERSE) {
        if (positions[sym]) continue;
        if (isOnCooldown(sym, di)) continue;

        const prices = getHistoryUpTo(barIdx, sym, today);
        if (prices.length < 35) continue;
        const analysis = getSignals(prices);

        if (analysis.consensus !== "BUY" && analysis.consensus !== "STRONG BUY") continue;

        // Volume confirmation
        const vols = getVolumeHistoryUpTo(barIdx, sym, today);
        const volAvg = avgVolume(vols, 20);
        if (volAvg !== null) {
          const currVol = vols[vols.length - 1];
          if (currVol / volAvg < RISK.VOLUME_CONFIRM_RATIO) continue;
        }

        // Earnings blackout
        const daysToEarn = daysUntilEarnings(sym, today);
        if (daysToEarn <= EARNINGS_BLACKOUT) continue;

        opportunities.push({ sym, score: analysis.score, consensus: analysis.consensus, rsiVal: analysis.indicators.rsi });
      }

      // Rank and queue buys
      opportunities.sort((a, b) => b.score - a.score);
      const openCount = Object.keys(positions).length + pendingBuys.length;
      const slots = Math.max(0, RISK.MAX_OPEN_POSITIONS - openCount);

      const portValue = getPortfolioValue(today);

      for (const opp of opportunities.slice(0, slots)) {
        // Asset-class position limit (International / Commodity / Bond / Volatility only)
        const sector = getSector(opp.sym);
        const sectorLimit = RISK.SECTOR_MAX_POSITIONS[sector];
        if (sectorLimit !== undefined) {
          const held = Object.keys(positions).filter((s) => getSector(s) === sector).length
                     + pendingBuys.filter((b) => getSector(b.sym) === sector).length;
          if (held >= sectorLimit) continue;
        }

        const prices = getHistoryUpTo(barIdx, opp.sym, today);
        const stockAtr = atr(prices, 14);
        const currPrice = prices[prices.length - 1];
        const atrPct    = stockAtr ? stockAtr / currPrice : RISK.ATR_TARGET_PCT;
        const volScale  = RISK.ATR_TARGET_PCT / atrPct;
        const dynPct    = Math.max(
          RISK.MIN_POSITION_PCT,
          Math.min(RISK.MAX_POSITION_PCT, RISK.MAX_POSITION_PCT * volScale * regimeMultiplier)
        );

        const maxAlloc  = portValue * dynPct;
        const allocCash = Math.min(maxAlloc, cash * RISK.MAX_CASH_DEPLOY_PCT);
        if (allocCash < currPrice) continue;

        const shares = Math.floor(allocCash / currPrice);
        if (shares <= 0) continue;

        pendingBuys.push({ sym: opp.sym, shares, consensus: opp.consensus });

        // Log as a trade record (entry leg)
        trades.push({
          sym:        opp.sym,
          side:       "buy",
          entryDate:  today,
          exitDate:   null,
          entryPrice: currPrice,
          exitPrice:  null,
          shares,
          pnl:        null,
          pnlPct:     null,
          holdDays:   null,
          exitReason: null,
          signalDate: today,
          consensus:  opp.consensus,
          layer:      "consensus",
        });
      }
    }

    // ── Phase C': Trend layer — SMA-break exits + new entries ──

    // Exit: queue sell if price has closed below 200-SMA for TREND_BREAK_DAYS consecutive days
    for (const sym of Object.keys(trendPositions)) {
      if (pendingTrendSells.some((s) => s.sym === sym)) continue;
      const prices = getHistoryUpTo(barIdx, sym, today);
      if (prices.length < 200) continue;
      const ts = computeTrendStatus(prices);
      if (!ts) continue;

      if (!ts.priceAbove200) {
        trendBreakCount[sym] = (trendBreakCount[sym] ?? 0) + 1;
        if (trendBreakCount[sym] >= TREND_BREAK_DAYS) {
          pendingTrendSells.push({ sym, reason: "trendBreak" });
        }
      } else {
        trendBreakCount[sym] = 0; // price recovered — reset streak
      }
    }

    // Entry: open new trend positions when strong uptrend detected
    const activeTrendCount = Object.keys(trendPositions).length + pendingTrendBuys.length;
    if (activeTrendCount < TREND_MAX_POSITIONS) {
      const portValue     = getPortfolioValue(today);
      const trendPosValue = Object.entries(trendPositions).reduce((sum, [sym, pos]) => {
        const bar = barIdx[sym]?.[today];
        return sum + (bar ? bar.c * pos.shares : pos.avgPrice * pos.shares);
      }, 0);
      const trendPortPct = portValue > 0 ? trendPosValue / portValue : 0;

      if (trendPortPct < TREND_MAX_PORT_PCT) {
        for (const { sym } of UNIVERSE) {
          if (trendPositions[sym]) continue;
          if (pendingTrendBuys.some((b) => b.sym === sym)) continue;
          if (isOnCooldown(sym, di)) continue;

          const prices = getHistoryUpTo(barIdx, sym, today);
          if (prices.length < 200) continue;
          const ts = computeTrendStatus(prices);
          if (!ts?.isStrongUptrend) continue;

          // Earnings blackout — same guard as consensus
          const daysToEarn = daysUntilEarnings(sym, today);
          if (daysToEarn <= EARNINGS_BLACKOUT) continue;

          // Combined position limit: if consensus also holds this symbol
          const consPos = positions[sym];
          if (consPos) {
            const consPrice = barIdx[sym]?.[today]?.c ?? consPos.avgPrice;
            const consAlloc = (consPos.shares * consPrice) / portValue;
            if (consAlloc + TREND_ALLOC_PCT > TREND_MAX_COMBINED) continue;
          }

          const currPrice = prices[prices.length - 1];
          const allocCash = portValue * TREND_ALLOC_PCT;
          if (allocCash < currPrice) continue;

          const shares = Math.floor(allocCash / currPrice);
          if (shares <= 0) continue;

          pendingTrendBuys.push({ sym, shares });
          trades.push({
            sym,
            side:       "buy",
            entryDate:  today,
            exitDate:   null,
            entryPrice: currPrice,
            exitPrice:  null,
            shares,
            pnl:        null,
            pnlPct:     null,
            holdDays:   null,
            exitReason: null,
            signalDate: today,
            layer:      "trend",
          });

          if (Object.keys(trendPositions).length + pendingTrendBuys.length >= TREND_MAX_POSITIONS) break;
        }
      }
    }

    // ── Phase D: Record equity curve point + SPY close ──
    equityCurve.push({ date: today, value: getPortfolioValue(today), spyClose: barIdx["SPY"]?.[today]?.c ?? null });
  }

  // Force-close any remaining open positions at final day's close
  const lastDay = simDays[simDays.length - 1];
  for (const [sym, pos] of Object.entries(positions)) {
    const bar = barIdx[sym]?.[lastDay];
    const exitPrice = bar ? bar.c : pos.avgPrice;
    const commission = pos.shares * COMMISSION_PER_SHARE;
    const revenue    = pos.shares * exitPrice - commission;
    const pnl        = revenue - pos.shares * pos.avgPrice;
    const pnlPct     = pnl / (pos.shares * pos.avgPrice);

    // Update the open trade record
    const openRecord = trades.findLast((t) => t.sym === sym && t.exitDate === null && t.layer !== "trend");
    if (openRecord) {
      openRecord.exitDate   = lastDay;
      openRecord.exitPrice  = exitPrice;
      openRecord.pnl        = pnl;
      openRecord.pnlPct     = pnlPct;
      openRecord.holdDays   = simDays.length - 1 - (simDays.indexOf(pos.entryDate));
      openRecord.exitReason = "endOfPeriod";
    }

    cash += revenue;
  }

  // Force-close remaining trend positions at final day's close
  for (const [sym, pos] of Object.entries(trendPositions)) {
    const bar        = barIdx[sym]?.[lastDay];
    const exitPrice  = bar ? bar.c : pos.avgPrice;
    const commission = pos.shares * COMMISSION_PER_SHARE;
    const revenue    = pos.shares * exitPrice - commission;
    const pnl        = revenue - pos.shares * pos.avgPrice;
    const pnlPct     = pnl / (pos.shares * pos.avgPrice);

    const openRecord = trades.findLast((t) => t.sym === sym && t.exitDate === null && t.layer === "trend");
    if (openRecord) {
      openRecord.exitDate   = lastDay;
      openRecord.exitPrice  = exitPrice;
      openRecord.pnl        = pnl;
      openRecord.pnlPct     = pnlPct;
      openRecord.holdDays   = simDays.length - 1 - simDays.indexOf(pos.entryDate);
      openRecord.exitReason = "endOfPeriod";
    }

    cash += revenue;
  }

  // Build daily SPY return series aligned with the equity curve
  const spyReturns = [];
  for (let i = 1; i < equityCurve.length; i++) {
    const prev = equityCurve[i - 1].spyClose;
    const curr = equityCurve[i].spyClose;
    if (prev != null && curr != null && prev > 0) {
      spyReturns.push((curr - prev) / prev);
    } else {
      spyReturns.push(0); // keep arrays aligned; missing bars contribute 0
    }
  }

  const metrics    = computeMetrics(equityCurve, initialCash, trades, simDays.length, spyReturns);
  const layerStats = computeLayerStats(trades);
  return { equityCurve, trades, metrics, label, layerStats };
}

// ── Walk-forward testing ──

/**
 * Run rolling walk-forward windows.
 * In-sample: 2 years (504 trading days)
 * Out-of-sample: 6 months (126 trading days)
 * Rolls forward by one out-of-sample period each iteration.
 */
function runWalkForward(barIdx, calendar, simStart, simEnd, earnings) {
  const IN_SAMPLE_DAYS  = 504;
  const OUT_SAMPLE_DAYS = 126;

  const allSimDays = calendar.filter((d) => d >= simStart && d <= simEnd);
  const windows = [];

  let startIdx = 0;
  while (startIdx + IN_SAMPLE_DAYS + OUT_SAMPLE_DAYS <= allSimDays.length) {
    const isStart = allSimDays[startIdx];
    const isEnd   = allSimDays[startIdx + IN_SAMPLE_DAYS - 1];
    const oosStart = allSimDays[startIdx + IN_SAMPLE_DAYS];
    const oosEnd   = allSimDays[Math.min(startIdx + IN_SAMPLE_DAYS + OUT_SAMPLE_DAYS - 1, allSimDays.length - 1)];

    const inSample  = runSimulation(barIdx, calendar, isStart, isEnd, earnings, INITIAL_CASH, "in-sample");
    const outSample = runSimulation(barIdx, calendar, oosStart, oosEnd, earnings, INITIAL_CASH, "out-of-sample");

    windows.push({ isStart, isEnd, oosStart, oosEnd, inSample, outSample });
    startIdx += OUT_SAMPLE_DAYS;
  }

  // Aggregate OOS equity curve (chained: each window starts where prior ends)
  let chainedCash = INITIAL_CASH;
  const chainedCurve = [];
  const chainedTrades = [];

  for (const w of windows) {
    const scale = chainedCash / INITIAL_CASH;
    for (const pt of w.outSample.equityCurve) {
      chainedCurve.push({ date: pt.date, value: pt.value * scale });
    }
    if (w.outSample.equityCurve.length > 0) {
      chainedCash = w.outSample.equityCurve[w.outSample.equityCurve.length - 1].value * scale;
    }
    chainedTrades.push(...w.outSample.trades);
  }

  const oosDays = windows.reduce((sum, w) => sum + w.outSample.equityCurve.length, 0);
  const oosMetrics = chainedCurve.length > 0
    ? computeMetrics(chainedCurve, INITIAL_CASH, chainedTrades, oosDays)
    : null;

  return { windows, chainedCurve, oosMetrics };
}

// ── localStorage persistence ──

const LS_KEY_PREFIX = "backtest_run_";
const MAX_STORED_RUNS = 10;

export function saveRunToStorage(run) {
  try {
    const id = `${LS_KEY_PREFIX}${Date.now()}`;
    const existing = listStoredRuns();

    // Prune oldest if at limit
    if (existing.length >= MAX_STORED_RUNS) {
      const oldest = existing[0];
      localStorage.removeItem(oldest.key);
    }

    const payload = {
      id,
      savedAt: new Date().toISOString(),
      label: run.label,
      config: run.config,
      metrics: run.metrics,
      equityCurve: run.equityCurve,
      trades: run.trades.filter((t) => t.exitDate), // only closed trades to save space
      walkForward: run.walkForward ? {
        oosMetrics: run.walkForward.oosMetrics,
        windows: run.walkForward.windows.map((w) => ({
          isStart: w.isStart, isEnd: w.isEnd,
          oosStart: w.oosStart, oosEnd: w.oosEnd,
          inSampleMetrics: w.inSample.metrics,
          outSampleMetrics: w.outSample.metrics,
        })),
      } : null,
    };

    localStorage.setItem(id, JSON.stringify(payload));
    return id;
  } catch (e) {
    console.warn("Failed to save backtest run:", e.message);
    return null;
  }
}

export function listStoredRuns() {
  const runs = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!key.startsWith(LS_KEY_PREFIX)) continue;
    try {
      const data = JSON.parse(localStorage.getItem(key));
      runs.push({ key, id: data.id, savedAt: data.savedAt, label: data.label, metrics: data.metrics, config: data.config });
    } catch { /* skip corrupt entries */ }
  }
  return runs.sort((a, b) => a.savedAt.localeCompare(b.savedAt));
}

export function loadStoredRun(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

export function deleteStoredRun(key) {
  localStorage.removeItem(key);
}

// ── Main entry point ──

/**
 * Run a full backtest.
 * @param {object} params
 * @param {string} params.start - sim start (YYYY-MM-DD)
 * @param {string} params.end   - sim end (YYYY-MM-DD)
 * @param {boolean} params.walkForward - run walk-forward analysis
 * @param {function} params.onProgress - callback(pct, msg)
 * @returns {Promise<object>} result with equityCurve, trades, metrics, walkForward
 */
export async function runBacktest({ start, end, walkForward = false, onProgress }) {
  const report = (pct, msg) => { if (onProgress) onProgress(pct, msg); };

  report(0, "Checking data cache...");

  // Check cache status
  const statusRes = await fetch(`/api/backtest/cache-status?start=${start}&end=${end}`);
  const { allCached } = await statusRes.json();

  if (!allCached) {
    report(5, "Downloading historical data...");

    // Stream SSE progress
    await new Promise((resolve, reject) => {
      let sseTotal = 39; // fallback: 38 price symbols + 1 earnings
      const evtSource = new EventSource(`/api/backtest/fetch?start=${start}&end=${end}`);
      evtSource.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "start") {
          sseTotal = msg.total || sseTotal; // capture accurate total from server
        }
        if (msg.type === "progress") {
          const pct = 5 + Math.round((msg.completed / sseTotal) * 30);
          report(pct, `Fetching ${msg.sym} (${msg.status})`);
        }
        if (msg.type === "done") {
          evtSource.close();
          resolve();
        }
      };
      evtSource.onerror = () => { evtSource.close(); reject(new Error("SSE stream failed")); };
    });
  }

  report(37, "Loading cached data...");

  // Load the data
  const dataRes = await fetch(`/api/backtest/data?start=${start}&end=${end}`);
  if (!dataRes.ok) {
    const err = await dataRes.json();
    throw new Error(err.error || "Failed to load backtest data");
  }
  const { daily, earnings, warmupStart } = await dataRes.json();

  report(45, "Building trading calendar...");
  const calendar = buildTradingCalendar(daily);
  const barIdx   = indexBars(daily);

  report(50, "Running simulation...");
  const fullResult = runSimulation(barIdx, calendar, start, end, earnings, INITIAL_CASH, "full");

  let wfResult = null;
  if (walkForward) {
    report(75, "Running walk-forward analysis...");
    wfResult = runWalkForward(barIdx, calendar, start, end, earnings);
  }

  report(95, "Computing metrics...");

  const config = { start, end, walkForward, warmupStart };
  const label  = `${start} → ${end}`;

  const result = {
    label,
    config,
    equityCurve:        fullResult.equityCurve,
    trades:             fullResult.trades,
    metrics:            fullResult.metrics,
    walkForward:        wfResult,
    layerStats:  fullResult.layerStats,
  };

  report(100, "Done.");
  return result;
}
