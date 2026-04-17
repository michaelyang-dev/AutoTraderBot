// ══════════════════════════════════════════
//  TRADE EXECUTOR — Autonomous order management
// ══════════════════════════════════════════

import { RISK, UNIVERSE, getSector } from "../config/constants";
import { getSignals } from "./signalEngine";
import { atr, avgVolume } from "./indicators";
import { computeTrendStatus } from "./trendEngine";

function fmtVol(v) {
  return v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : `${(v / 1_000).toFixed(0)}k`;
}

/**
 * Run one full trading cycle:
 *  1. Check stop-loss / take-profit on existing positions
 *  2. Scan all stocks for signals
 *  3. Sell positions with SELL consensus
 *  4. Buy top-ranked BUY opportunities
 *
 * Returns updated state + array of log messages.
 * This is a pure function — no side effects.
 *
 * @param {object} params
 * @param {number} params.cash
 * @param {object} params.positions   - { [sym]: { shares, avgPrice, entryTick } }
 * @param {object} params.priceHist   - { [sym]: number[] }
 * @param {number} params.tick
 * @param {object} params.stats       - { buys, sells, wins, losses, totalPnL }
 * @param {object} params.cooldowns   - { [sym]: cycleNumber } — losing-exit cooldown map
 * @param {number} params.cycleNumber - monotonically increasing trade-cycle counter
 * @returns {{ cash, positions, stats, cooldowns, logs: {msg, type}[] }}
 */
export function executeTradingCycle({ cash, positions, priceHist, tick, stats, cooldowns = {}, cycleNumber = 0, volHist = {}, trendPositions = {}, trendBreakCounts = {} }) {
  let currentCash = cash;
  const currentPos = { ...positions };
  const currentStats = { ...stats };
  const currentCooldowns = { ...cooldowns };
  const currentTrendPos    = { ...trendPositions };
  const currentTrendBreaks = { ...trendBreakCounts };
  const logs = [];

  const isOnCooldown = (sym) => {
    const lossAt = currentCooldowns[sym];
    return lossAt !== undefined && (cycleNumber - lossAt) < RISK.LOSS_COOLDOWN_CYCLES;
  };

  const getPortValue = () => {
    return currentCash + Object.entries(currentPos).reduce((sum, [sym, p]) => {
      const prices = priceHist[sym];
      return sum + (prices ? prices[prices.length - 1] * p.shares : 0);
    }, 0);
  };

  // ── STEP 1: Trailing/Fixed Stop-loss & Take-profit ──
  Object.entries(currentPos).forEach(([sym, pos]) => {
    const prices = priceHist[sym];
    if (!prices) return;
    const curr = prices[prices.length - 1];

    // Update trailing peak price
    if (RISK.USE_TRAILING_STOP && curr > (pos.peakPrice ?? pos.avgPrice)) {
      currentPos[sym] = { ...pos, peakPrice: curr };
      pos = currentPos[sym];
    }

    // Determine if stop is triggered
    let stopTriggered = false;
    let stopMsg = "";
    if (RISK.USE_TRAILING_STOP) {
      const peak = pos.peakPrice ?? pos.avgPrice;
      const dropFromPeak = (curr - peak) / peak;
      if (dropFromPeak <= -RISK.TRAILING_STOP_PCT) {
        stopTriggered = true;
        stopMsg = `🛑 TRAIL-STOP ${sym}: ${pos.shares} shares @ $${curr.toFixed(2)} | Peak $${peak.toFixed(2)}, drop ${(dropFromPeak * 100).toFixed(1)}%`;
      }
    } else {
      const pnlPct = (curr - pos.avgPrice) / pos.avgPrice;
      if (pnlPct <= RISK.STOP_LOSS_PCT) {
        stopTriggered = true;
        const pnl = pos.shares * (curr - pos.avgPrice);
        stopMsg = `🛑 STOP-LOSS ${sym}: ${pos.shares} shares @ $${curr.toFixed(2)} | P&L: $${pnl.toFixed(2)}`;
      }
    }

    if (stopTriggered) {
      const revenue = pos.shares * curr;
      const pnl = revenue - pos.shares * pos.avgPrice;
      currentCash += revenue;
      delete currentPos[sym];
      currentStats.sells++;
      currentStats.losses++;
      currentStats.totalPnL += pnl;
      if (pnl < 0) currentCooldowns[sym] = cycleNumber;
      logs.push({ msg: stopMsg, type: "sell" });
    } else {
      const pnlPct = (curr - pos.avgPrice) / pos.avgPrice;
      if (pnlPct >= RISK.TAKE_PROFIT_PCT) {
        const revenue = pos.shares * curr;
        const pnl = revenue - pos.shares * pos.avgPrice;
        currentCash += revenue;
        delete currentPos[sym];
        currentStats.sells++;
        currentStats.wins++;
        currentStats.totalPnL += pnl;
        logs.push({
          msg: `🎯 TAKE-PROFIT ${sym}: ${pos.shares} shares @ $${curr.toFixed(2)} | P&L: +$${pnl.toFixed(2)}`,
          type: "profit",
        });
      }
    }
  });

  // ── STEP 2: Scan all stocks ──
  const opportunities = [];

  UNIVERSE.forEach(({ sym }) => {
    const prices = priceHist[sym];
    if (!prices || prices.length < 35) return;

    const analysis = getSignals(prices);

    // Sell on SELL consensus
    if (currentPos[sym] && (analysis.consensus === "STRONG SELL" || analysis.consensus === "SELL")) {
      const pos = currentPos[sym];
      const curr = prices[prices.length - 1];
      const revenue = pos.shares * curr;
      const pnl = revenue - pos.shares * pos.avgPrice;
      currentCash += revenue;
      delete currentPos[sym];
      currentStats.sells++;
      pnl >= 0 ? currentStats.wins++ : currentStats.losses++;
      currentStats.totalPnL += pnl;
      if (pnl < 0) currentCooldowns[sym] = cycleNumber;  // cooldown only on a losing exit
      logs.push({
        msg: `📉 SELL ${sym}: ${pos.shares} shares @ $${curr.toFixed(2)} | ${analysis.consensus} | P&L: ${pnl >= 0 ? "+" : ""}$${pnl.toFixed(2)}`,
        type: pnl >= 0 ? "profit" : "sell",
      });
    }

    // Collect buy candidates — skip if still in loss cooldown
    if (!currentPos[sym] && (analysis.consensus === "STRONG BUY" || analysis.consensus === "BUY")) {
      if (isOnCooldown(sym)) {
        const remaining = RISK.LOSS_COOLDOWN_CYCLES - (cycleNumber - currentCooldowns[sym]);
        logs.push({
          msg: `⏸ Skipping ${sym} — cooldown active, ${remaining} cycle${remaining !== 1 ? "s" : ""} remaining`,
          type: "system",
        });
      } else {
        opportunities.push({
          sym,
          score: analysis.score,
          price: prices[prices.length - 1],
          consensus: analysis.consensus,
          rsiVal: analysis.indicators.rsi,
        });
      }
    }
  });

  // ── STEP 3: Rank & execute buys ──
  opportunities.sort((a, b) => b.score - a.score);
  const openCount = Object.keys(currentPos).length;
  const slotsAvail = RISK.MAX_OPEN_POSITIONS - openCount;

  opportunities.slice(0, Math.max(0, slotsAvail)).forEach((opp) => {
    // Asset-class position limit (International / Commodity / Bond / Volatility only)
    const sector = getSector(opp.sym);
    const sectorLimit = RISK.SECTOR_MAX_POSITIONS[sector];
    if (sectorLimit !== undefined) {
      const held = Object.keys(currentPos).filter((s) => getSector(s) === sector).length;
      if (held >= sectorLimit) {
        logs.push({ msg: `⛔ Skipping ${opp.sym} — ${sector} limit reached (max ${sectorLimit})`, type: "system" });
        return;
      }
    }

    // Volume confirmation check
    const vols = volHist[opp.sym];
    const avg = avgVolume(vols, 20);
    if (avg !== null) {
      const currVol = vols[vols.length - 1];
      const ratio = currVol / avg;
      if (ratio < RISK.VOLUME_CONFIRM_RATIO) {
        logs.push({
          msg: `📊 Skipping ${opp.sym} — volume ${fmtVol(currVol)} below 1.5x average of ${fmtVol(avg)} (${ratio.toFixed(2)}x)`,
          type: "system",
        });
        return;
      }
    }

    const portValue = getPortValue();

    // ATR-based position sizing: scale allocation inversely with volatility.
    // High-ATR stocks (TSLA, NVDA) get smaller positions; low-ATR stocks (JPM) get larger ones.
    const stockAtr = atr(priceHist[opp.sym], 14);
    const atrPct = stockAtr ? stockAtr / opp.price : RISK.ATR_TARGET_PCT;
    const volatilityScale = RISK.ATR_TARGET_PCT / atrPct;
    const dynPositionPct = Math.max(
      RISK.MIN_POSITION_PCT,
      Math.min(RISK.MAX_POSITION_PCT, RISK.MAX_POSITION_PCT * volatilityScale)
    );

    const maxAlloc = portValue * dynPositionPct;
    const allocCash = Math.min(maxAlloc, currentCash * RISK.MAX_CASH_DEPLOY_PCT);
    if (allocCash < opp.price) return;

    const shares = Math.floor(allocCash / opp.price);
    if (shares <= 0) return;
    const cost = shares * opp.price;

    currentCash -= cost;
    currentPos[opp.sym] = { shares, avgPrice: opp.price, entryTick: tick, peakPrice: opp.price };
    currentStats.buys++;
    logs.push({
      msg: `📈 BUY ${opp.sym}: ${shares} shares @ $${opp.price.toFixed(2)} | ${opp.consensus} (score: ${opp.score.toFixed(1)}) | alloc ${(dynPositionPct * 100).toFixed(1)}% (ATR ${(atrPct * 100).toFixed(1)}%)`,
      type: "buy",
    });
  });

  // ── Trend layer: trailing stop (10%) ──
  Object.entries(currentTrendPos).forEach(([sym, pos]) => {
    const prices = priceHist[sym];
    if (!prices) return;
    const curr = prices[prices.length - 1];

    if (curr > (pos.peakPrice ?? pos.avgPrice)) {
      currentTrendPos[sym] = { ...pos, peakPrice: curr };
      pos = currentTrendPos[sym];
    }

    const peak = pos.peakPrice ?? pos.avgPrice;
    const dropFromPeak = (curr - peak) / peak;
    if (dropFromPeak <= -0.10) {
      const revenue = pos.shares * curr;
      const pnl = revenue - pos.shares * pos.avgPrice;
      currentCash += revenue;
      delete currentTrendPos[sym];
      delete currentTrendBreaks[sym];
      logs.push({
        msg: `🏔 TREND-STOP ${sym}: ${pos.shares} sh @ $${curr.toFixed(2)} | Peak $${peak.toFixed(2)}, drop ${(dropFromPeak * 100).toFixed(1)}%`,
        type: "sell",
      });
    }
  });

  // ── Trend layer: SMA-break exits (3 consecutive closes below 200-SMA) ──
  Object.keys(currentTrendPos).forEach((sym) => {
    const prices = priceHist[sym];
    if (!prices || prices.length < 200) return;
    const ts = computeTrendStatus(prices);
    if (!ts) return;

    if (!ts.priceAbove200) {
      currentTrendBreaks[sym] = (currentTrendBreaks[sym] ?? 0) + 1;
      if (currentTrendBreaks[sym] >= 3) {
        const pos = currentTrendPos[sym];
        const curr = prices[prices.length - 1];
        const revenue = pos.shares * curr;
        const pnl = revenue - pos.shares * pos.avgPrice;
        currentCash += revenue;
        delete currentTrendPos[sym];
        delete currentTrendBreaks[sym];
        logs.push({
          msg: `📉 TREND-BREAK ${sym}: 3 consecutive closes below 200-SMA — exiting | P&L: ${pnl >= 0 ? "+" : ""}$${pnl.toFixed(2)}`,
          type: "sell",
        });
      }
    } else {
      currentTrendBreaks[sym] = 0;
    }
  });

  // ── Trend layer: new entries (strong uptrend) ──
  const trendCount = Object.keys(currentTrendPos).length;
  if (trendCount < 8) {
    const portValue = currentCash + Object.entries(currentPos).reduce((sum, [sym, p]) => {
      const prices = priceHist[sym];
      return sum + (prices ? prices[prices.length - 1] * p.shares : 0);
    }, 0) + Object.entries(currentTrendPos).reduce((sum, [sym, p]) => {
      const prices = priceHist[sym];
      return sum + (prices ? prices[prices.length - 1] * p.shares : 0);
    }, 0);

    const trendPortPct = Object.entries(currentTrendPos).reduce((sum, [sym, p]) => {
      const prices = priceHist[sym];
      const curr = prices ? prices[prices.length - 1] : p.avgPrice;
      return sum + (curr * p.shares) / portValue;
    }, 0);

    if (trendPortPct < 0.30) {
      UNIVERSE.forEach(({ sym }) => {
        if (Object.keys(currentTrendPos).length >= 8) return;
        if (currentTrendPos[sym] || currentPos[sym]) return;
        const prices = priceHist[sym];
        if (!prices || prices.length < 200) return;
        const ts = computeTrendStatus(prices);
        if (!ts?.isStrongUptrend) return;

        const curr = prices[prices.length - 1];
        const allocCash = portValue * 0.05;
        if (allocCash < curr) return;
        const shares = Math.floor(allocCash / curr);
        if (shares <= 0 || shares * curr > currentCash) return;

        currentCash -= shares * curr;
        currentTrendPos[sym] = { shares, avgPrice: curr, entryTick: tick, peakPrice: curr };
        logs.push({
          msg: `🏔 TREND-BUY ${sym}: ${shares} sh @ $${curr.toFixed(2)} | Strong uptrend — ${ts.daysAbove200}/40 days above 200-SMA`,
          type: "buy",
        });
      });
    }
  }

  return {
    cash: currentCash,
    positions: currentPos,
    stats: currentStats,
    cooldowns: currentCooldowns,
    trendPositions:  currentTrendPos,
    trendBreakCounts: currentTrendBreaks,
    logs,
  };
}
