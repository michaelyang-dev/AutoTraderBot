// ══════════════════════════════════════════
//  LIVE TRADE EXECUTOR — Uses Alpaca paper trading API
//  Replaces the simulated tradeExecutor.js
// ══════════════════════════════════════════

import { RISK, MARKET_HOURS, UNIVERSE, getSector } from "../config/constants";
import { getSignals } from "./signalEngine";
import { atr, avgVolume } from "./indicators";
import { computeTrendStatus } from "./trendEngine";

function fmtVol(v) {
  return v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : `${(v / 1_000).toFixed(0)}k`;
}
import * as alpaca from "./alpacaClient";
import { getUpcomingEarnings } from "./fmpClient";

// ── Earnings cache — refresh at most once every 4 hours to protect FMP free tier ──
let _earningsCache = { data: {}, fetchedAt: 0 };
const EARNINGS_TTL_MS = 4 * 60 * 60 * 1000;

async function fetchEarnings(symbols) {
  if (Date.now() - _earningsCache.fetchedAt < EARNINGS_TTL_MS) {
    return _earningsCache.data;
  }
  try {
    const data = await getUpcomingEarnings(symbols);
    _earningsCache = { data, fetchedAt: Date.now() };
    return data;
  } catch {
    return _earningsCache.data; // serve stale cache rather than crashing
  }
}

/**
 * Calendar days between today (ET) and a "YYYY-MM-DD" earnings date.
 * Returns 0 if earnings is today, 1 if tomorrow, negative if in the past.
 */
function daysUntilEarnings(dateStr) {
  const nowET = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
  nowET.setHours(0, 0, 0, 0);
  const target = new Date(dateStr + "T00:00:00");
  return Math.round((target - nowET) / 86_400_000);
}

/**
 * Returns ET hours/minutes from an ISO timestamp.
 * Alpaca clock timestamps are always anchored to US/Eastern market time.
 */
function getETTime(isoTimestamp) {
  return new Date(
    new Date(isoTimestamp).toLocaleString("en-US", { timeZone: "America/New_York" })
  );
}

/**
 * How many minutes have elapsed since 9:30 AM ET and until 4:00 PM ET.
 * Only meaningful when market is_open.
 */
function marketWindowMins(clock) {
  const et = getETTime(clock.timestamp);
  const totalMins = et.getHours() * 60 + et.getMinutes();
  const minutesSinceOpen = totalMins - (9 * 60 + 30);  // minutes after 9:30 AM ET
  const minutesUntilClose = 16 * 60 - totalMins;        // minutes before 4:00 PM ET
  return { minutesSinceOpen, minutesUntilClose };
}

/**
 * Run one full live trading cycle:
 *  1. Fetch current account + positions from Alpaca
 *  2. Enforce market hours window (open/close buffers)
 *  3. Check trailing/fixed stop-loss and take-profit on existing positions
 *  4. Scan all stocks for signals using historical bars
 *  5. Execute buy/sell orders through Alpaca (skipped near close)
 *
 * @param {object} params
 * @param {object} params.priceHist      - { [sym]: number[] } — kept locally for indicators
 * @param {object} params.trailingPeaks  - { [sym]: peakPrice } — mutated in-place across cycles
 * @param {string} params.regime         - "BULLISH" | "CAUTIOUS" | "BEARISH"
 * @param {object} params.cooldowns      - { [sym]: cycleNumber } — mutated in-place across cycles
 * @param {number} params.cycleNumber    - current trade-cycle count (from the hook)
 * @returns {{ logs: {msg, type}[], account, positions }}
 */
// ── SPY idle-cash parking constants ──
const SPY_IDLE_RESERVE_PCT   = 0.30;   // always keep 30% of portfolio as cash reserve
const SPY_IDLE_THRESHOLD_PCT = 0.20;   // trigger parking when idle cash > 20% of portfolio
const SPY_IDLE_INVEST_PCT    = 0.85;   // park 85% of idle cash into SPY

export async function executeLiveTradingCycle({ priceHist, volHist = {}, trailingPeaks = {}, regime = "BULLISH", cooldowns = {}, cycleNumber = 0, trendPositions = {}, trendBreakCounts = {}, mlSignals = null, idleSpyRef = null }) {
  const logs = [];

  const isOnCooldown = (sym) => {
    const lossAt = cooldowns[sym];
    return lossAt !== undefined && (cycleNumber - lossAt) < RISK.LOSS_COOLDOWN_CYCLES;
  };

  try {
    // ── Fetch live state from Alpaca ──
    const [account, positions, clock] = await Promise.all([
      alpaca.getAccount(),
      alpaca.getPositions(),
      alpaca.getClock(),
    ]);

    if (!clock.is_open) {
      logs.push({
        msg: `🕐 Market is closed. Next open: ${new Date(clock.next_open).toLocaleString()}`,
        type: "system",
      });
      return { logs, account, positions };
    }

    // ── Market hours window check ──
    const { minutesSinceOpen, minutesUntilClose } = marketWindowMins(clock);

    if (minutesSinceOpen < MARKET_HOURS.OPEN_BUFFER_MINS) {
      logs.push({
        msg: `⏰ Opening buffer: ${(MARKET_HOURS.OPEN_BUFFER_MINS - minutesSinceOpen).toFixed(0)} min until trading begins (avoiding open volatility).`,
        type: "system",
      });
      return { logs, account, positions };
    }

    const skipNewBuys = minutesUntilClose < MARKET_HOURS.CLOSE_BUFFER_MINS;
    if (skipNewBuys) {
      logs.push({
        msg: `⏰ Close buffer: ${minutesUntilClose.toFixed(0)} min until close — stop-loss checks only, no new buys.`,
        type: "system",
      });
    }

    let cash = account.cash;
    const portfolioValue = account.portfolio_value;

    // ── Fetch upcoming earnings (cached — refreshes every 4 h) ──
    const allSymbols = [...new Set([...UNIVERSE.map(u => u.sym), ...positions.map(p => p.symbol)])];
    const earningsMap = await fetchEarnings(allSymbols);
    const earningsSymbols = Object.keys(earningsMap);
    if (earningsSymbols.length > 0) {
      logs.push({
        msg: `📅 Earnings next 7 days: ${earningsSymbols.map(s => `${s} (${earningsMap[s]})`).join(", ")}`,
        type: "system",
      });
    }

    // ── STEP 1: Trailing/Fixed Stop-loss & Take-profit on existing positions ──
    const closedSymbols = new Set();
    for (const pos of positions) {
      const { symbol, qty, current_price: curr, unrealized_pl, unrealized_plpc } = pos;
      // Idle-spy positions are a cash substitute — no stop-loss or take-profit
      if (symbol === "SPY" && idleSpyRef) continue;
      // Trend-managed positions have their own exit logic — skip consensus stop/TP
      if (trendPositions[symbol]) continue;

      // Update trailing peak
      if (RISK.USE_TRAILING_STOP) {
        if (!trailingPeaks[symbol] || curr > trailingPeaks[symbol]) {
          trailingPeaks[symbol] = curr;
        }
      }

      let stopTriggered = false;
      let stopMsg = "";

      if (RISK.USE_TRAILING_STOP) {
        const peak = trailingPeaks[symbol] ?? curr;
        const dropFromPeak = (curr - peak) / peak;
        if (dropFromPeak <= -RISK.TRAILING_STOP_PCT) {
          stopTriggered = true;
          stopMsg = `🛑 TRAIL-STOP ${symbol}: ${qty} shares @ $${curr.toFixed(2)} | Peak $${peak.toFixed(2)}, drop ${(dropFromPeak * 100).toFixed(1)}%`;
        }
      } else {
        if (unrealized_plpc <= RISK.STOP_LOSS_PCT) {
          stopTriggered = true;
          stopMsg = `🛑 STOP-LOSS ${symbol}: ${qty} shares @ $${curr.toFixed(2)} | P&L: $${unrealized_pl.toFixed(2)}`;
        }
      }

      if (stopTriggered) {
        try {
          await alpaca.closePosition(symbol);
          delete trailingPeaks[symbol];
          closedSymbols.add(symbol);
          // Cooldown only when we actually lost money (trailing stop can exit at a profit)
          if (unrealized_plpc < 0) cooldowns[symbol] = cycleNumber;
          logs.push({ msg: stopMsg, type: "sell" });
        } catch (err) {
          logs.push({ msg: `❌ Failed to close ${symbol}: ${err.message}`, type: "error" });
        }
      } else if (unrealized_plpc >= RISK.TAKE_PROFIT_PCT) {
        try {
          await alpaca.closePosition(symbol);
          delete trailingPeaks[symbol];
          closedSymbols.add(symbol);
          logs.push({
            msg: `🎯 TAKE-PROFIT ${symbol}: ${qty} shares @ $${curr.toFixed(2)} | P&L: +$${unrealized_pl.toFixed(2)}`,
            type: "profit",
          });
        } catch (err) {
          logs.push({ msg: `❌ Failed to close ${symbol}: ${err.message}`, type: "error" });
        }
      }
    }

    // ── STEP 1b: Earnings-eve exits — sell before tomorrow's announcement ──
    for (const pos of positions) {
      if (closedSymbols.has(pos.symbol)) continue;
      if (pos.symbol === "SPY" && idleSpyRef) continue; // idle-spy — no earnings exit
      const earningsDate = earningsMap[pos.symbol];
      if (!earningsDate) continue;
      const days = daysUntilEarnings(earningsDate);
      if (days === 1) {
        try {
          await alpaca.closePosition(pos.symbol);
          delete trailingPeaks[pos.symbol];
          closedSymbols.add(pos.symbol);
          logs.push({
            msg: `📅 EARNINGS SELL ${pos.symbol}: earnings tomorrow (${earningsDate}) — exiting to avoid overnight announcement risk`,
            type: "sell",
          });
        } catch (err) {
          logs.push({ msg: `❌ Earnings sell failed ${pos.symbol}: ${err.message}`, type: "error" });
        }
      }
    }

    // ── STEP 2: Scan for signals ──
    if (regime !== "BULLISH") {
      const spyPrice = priceHist.SPY?.[priceHist.SPY.length - 1];
      logs.push({
        msg: `🌡️ Regime: ${regime} | SPY $${spyPrice?.toFixed(2) ?? "N/A"} — ${regime === "BEARISH" ? "buys suspended" : "BUY + STRONG BUY allowed, 75% position size"}`,
        type: "system",
      });
    }

    // Build ML signal map — null means ML server is down → fallback to consensus
    const mlMap = {};
    if (mlSignals && mlSignals.length > 0) {
      for (const s of mlSignals) mlMap[s.symbol] = s;
    }
    const mlActive = mlSignals !== null && Object.keys(mlMap).length > 0;

    if (mlActive) {
      const buyCount = mlSignals.filter(s => s.signal === "BUY").length;
      logs.push({ msg: `🤖 ML active — ${buyCount} BUY signal${buyCount !== 1 ? "s" : ""} above 55% threshold`, type: "system" });
    } else {
      logs.push({ msg: "🔄 ML server offline — using consensus engine (fallback mode)", type: "system" });
    }

    const heldSymbols = new Set(positions.map((p) => p.symbol));
    const opportunities = [];

    for (const { sym } of UNIVERSE) {
      const prices = priceHist[sym];
      if (!prices || prices.length < 35) continue;

      const analysis = getSignals(prices);

      // Sell on SELL consensus — unchanged regardless of ML mode
      if (heldSymbols.has(sym) && !trendPositions[sym] && (analysis.consensus === "STRONG SELL" || analysis.consensus === "SELL")) {
        try {
          await alpaca.closePosition(sym);
          const posData = positions.find(p => p.symbol === sym);
          if (posData && posData.unrealized_plpc < 0) cooldowns[sym] = cycleNumber;
          logs.push({ msg: `📉 SELL ${sym}: ${analysis.consensus} — closing position`, type: "sell" });
        } catch (err) {
          logs.push({ msg: `❌ Sell failed ${sym}: ${err.message}`, type: "error" });
        }
      }

      if (heldSymbols.has(sym) || regime === "BEARISH") continue;

      if (isOnCooldown(sym)) {
        const remaining = RISK.LOSS_COOLDOWN_CYCLES - (cycleNumber - cooldowns[sym]);
        logs.push({ msg: `⏸ Skipping ${sym} — cooldown active, ${remaining} cycle${remaining !== 1 ? "s" : ""} remaining`, type: "system" });
        continue;
      }

      if (mlActive) {
        // ── ML primary decision ──
        const mlSig = mlMap[sym];
        if (mlSig && mlSig.signal === "BUY") {
          logs.push({ msg: `🤖 ML BUY ${sym} — confidence ${mlSig.probability.toFixed(2)}`, type: "buy" });
          opportunities.push({
            sym,
            score: mlSig.probability,
            price: prices[prices.length - 1],
            consensus: `ML BUY (${(mlSig.probability * 100).toFixed(0)}%)`,
            rsiVal: analysis.indicators.rsi,
            mlConf: mlSig.probability,
          });
        } else {
          // Only log near-misses (prob ≥ 0.45) to avoid flooding the feed
          if (mlSig && mlSig.probability >= 0.45) {
            logs.push({ msg: `🤖 ML SKIP ${sym} — confidence ${mlSig.probability.toFixed(2)} below threshold`, type: "system" });
          }
        }
      } else {
        // ── Consensus fallback ──
        const signalQualifies = analysis.consensus === "STRONG BUY" || analysis.consensus === "BUY";
        if (signalQualifies) {
          opportunities.push({
            sym,
            score: analysis.score,
            price: prices[prices.length - 1],
            consensus: analysis.consensus,
            rsiVal: analysis.indicators.rsi,
            mlConf: null,
          });
        }
      }
    }

    // ── STEP 3: Rank & execute buys ──
    if (skipNewBuys) {
      const updatedPositions = await alpaca.getPositions();
      const updatedAccount = await alpaca.getAccount();
      return { logs, account: updatedAccount, positions: updatedPositions };
    }

    opportunities.sort((a, b) => b.score - a.score);

    // ── STEP 3a: Sell idle SPY to free cash before ML buys ──
    // Don't count idle-spy toward position limit — only active ML/trend positions count
    const activePositionCount = positions.filter(p => p.symbol !== "SPY").length;
    const slotsAvail = RISK.MAX_OPEN_POSITIONS - activePositionCount;

    if (idleSpyRef && idleSpyRef.current > 0 && opportunities.length > 0) {
      const spyPrice = priceHist.SPY?.[priceHist.SPY.length - 1];
      if (spyPrice && spyPrice > 0) {
        const sharesToSell = idleSpyRef.current;
        const idleValue = sharesToSell * spyPrice;
        try {
          await alpaca.placeOrder({ symbol: "SPY", qty: sharesToSell, side: "sell", type: "market" });
          idleSpyRef.current = 0;
          logs.push({
            msg: `💼 [idle-spy] Selling ${sharesToSell} SPY shares ($${idleValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}) to fund ${opportunities.length} ML pick${opportunities.length !== 1 ? "s" : ""}`,
            type: "system",
          });
          // Refresh cash after selling idle SPY
          const freshAcct = await alpaca.getAccount();
          cash = freshAcct.cash;
        } catch (err) {
          logs.push({ msg: `❌ [idle-spy] Failed to sell SPY: ${err.message}`, type: "error" });
        }
      }
    }

    // Track sectors bought this cycle (positions snapshot is fixed at cycle start)
    const boughtThisCycle = {}; // { sector: count }

    for (const opp of opportunities.slice(0, Math.max(0, slotsAvail))) {
      // Asset-class position limit (International / Commodity / Bond / Volatility only)
      const sector = getSector(opp.sym);
      const sectorLimit = RISK.SECTOR_MAX_POSITIONS[sector];
      if (sectorLimit !== undefined) {
        const existingSectorCount = positions.filter((p) => getSector(p.symbol) === sector).length;
        const cycleCount = boughtThisCycle[sector] ?? 0;
        if (existingSectorCount + cycleCount >= sectorLimit) {
          logs.push({ msg: `⛔ Skipping ${opp.sym} — ${sector} limit reached (max ${sectorLimit})`, type: "system" });
          continue;
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
          continue;
        }
      }

      // Skip if earnings within 3 calendar days
      const earningsDate = earningsMap[opp.sym];
      if (earningsDate) {
        const days = daysUntilEarnings(earningsDate);
        if (days >= 0 && days <= 3) {
          logs.push({
            msg: `📅 SKIP ${opp.sym}: earnings in ${days} day(s) on ${earningsDate}`,
            type: "system",
          });
          continue;
        }
      }

      // ATR-based position sizing scaled by regime and ML confidence
      const stockAtr = atr(priceHist[opp.sym], 14);
      const atrPct = stockAtr ? stockAtr / opp.price : RISK.ATR_TARGET_PCT;
      const volatilityScale = RISK.ATR_TARGET_PCT / atrPct;
      const regimeMult = regime === "CAUTIOUS" ? 0.75 : 1.0;
      // ML confidence scaling: linear 0.55→60% of normal, 0.80→100% of normal
      // Formula: scale = conf*1.6 - 0.28, clamped to [0.60, 1.0]
      const mlMult = opp.mlConf != null
        ? Math.min(1.0, Math.max(0.60, opp.mlConf * 1.6 - 0.28))
        : 1.0;
      const dynPositionPct = Math.max(
        RISK.MIN_POSITION_PCT,
        Math.min(RISK.MAX_POSITION_PCT, RISK.MAX_POSITION_PCT * volatilityScale * regimeMult * mlMult)
      );

      const maxAlloc = portfolioValue * dynPositionPct;
      const allocCash = Math.min(maxAlloc, cash * RISK.MAX_CASH_DEPLOY_PCT);
      if (allocCash < opp.price) continue;

      const shares = Math.floor(allocCash / opp.price);
      if (shares <= 0) continue;

      try {
        const order = await alpaca.placeOrder({
          symbol: opp.sym,
          qty: shares,
          side: "buy",
          type: "market",
        });
        boughtThisCycle[sector] = (boughtThisCycle[sector] ?? 0) + 1;
        const mlNote = opp.mlConf != null ? `, ML ${(opp.mlConf * 100).toFixed(0)}% conf` : "";
        logs.push({
          msg: `📈 BUY ${opp.sym}: ${shares} shares | ${opp.consensus} (score: ${opp.score.toFixed(2)}) | alloc ${(dynPositionPct * 100).toFixed(1)}% (ATR ${(atrPct * 100).toFixed(1)}%${regime === "CAUTIOUS" ? ", cautious 75%" : ""}${mlNote}) | Order: ${order.status}`,
          type: "buy",
        });
      } catch (err) {
        logs.push({ msg: `❌ Buy failed ${opp.sym}: ${err.message}`, type: "error" });
      }
    }

    // ── STEP 4: Trend trailing stop (10%) ──
    for (const pos of positions) {
      const sym  = pos.symbol;
      const tPos = trendPositions[sym];
      if (!tPos || closedSymbols.has(sym)) continue;

      const curr = pos.current_price;
      if (curr > (tPos.peakPrice ?? tPos.entryPrice)) {
        tPos.peakPrice = curr;
      }
      const peak = tPos.peakPrice ?? tPos.entryPrice;
      const dropFromPeak = (curr - peak) / peak;

      if (dropFromPeak <= -0.10) {
        try {
          await alpaca.closePosition(sym);
          closedSymbols.add(sym);
          delete trendPositions[sym];
          delete trendBreakCounts[sym];
          logs.push({
            msg: `🏔 TREND-STOP ${sym}: @ $${curr.toFixed(2)} | Peak $${peak.toFixed(2)}, drop ${(dropFromPeak * 100).toFixed(1)}%`,
            type: "sell",
          });
        } catch (err) {
          logs.push({ msg: `❌ Trend-stop close failed ${sym}: ${err.message}`, type: "error" });
        }
      }
    }

    // ── STEP 5: Trend SMA-break exits (3 consecutive closes below 200-SMA) ──
    for (const sym of Object.keys(trendPositions)) {
      if (closedSymbols.has(sym)) continue;
      const prices = priceHist[sym];
      if (!prices || prices.length < 200) continue;
      const ts = computeTrendStatus(prices);
      if (!ts) continue;

      if (!ts.priceAbove200) {
        trendBreakCounts[sym] = (trendBreakCounts[sym] ?? 0) + 1;
        if (trendBreakCounts[sym] >= 3) {
          try {
            await alpaca.closePosition(sym);
            closedSymbols.add(sym);
            delete trendPositions[sym];
            delete trendBreakCounts[sym];
            logs.push({
              msg: `📉 TREND-BREAK ${sym}: 3 consecutive closes below 200-SMA — exiting`,
              type: "sell",
            });
          } catch (err) {
            logs.push({ msg: `❌ Trend-break close failed ${sym}: ${err.message}`, type: "error" });
          }
        } else {
          logs.push({
            msg: `⚠️ TREND ${sym}: day ${trendBreakCounts[sym]}/3 below 200-SMA`,
            type: "system",
          });
        }
      } else {
        trendBreakCounts[sym] = 0;
      }
    }

    // ── STEP 6: Trend new entries ──
    if (!skipNewBuys && regime !== "BEARISH") {
      const trendCount = Object.keys(trendPositions).length;
      if (trendCount < 8) {
        const trendPosValue = Object.values(trendPositions).reduce((sum, tp) => {
          const alpacaPos = positions.find((p) => p.symbol === tp.sym);
          return sum + (alpacaPos ? alpacaPos.market_value : 0);
        }, 0);
        const trendPortPct = portfolioValue > 0 ? trendPosValue / portfolioValue : 0;

        if (trendPortPct < 0.30) {
          for (const { sym } of UNIVERSE) {
            if (trendPositions[sym] || heldSymbols.has(sym)) continue;
            const prices = priceHist[sym];
            if (!prices || prices.length < 200) continue;
            const ts = computeTrendStatus(prices);
            if (!ts?.isStrongUptrend) continue;

            const currPrice = prices[prices.length - 1];
            const allocCash = portfolioValue * 0.05;
            if (allocCash < currPrice || allocCash > cash) continue;
            const shares = Math.floor(allocCash / currPrice);
            if (shares <= 0) continue;

            try {
              const order = await alpaca.placeOrder({ symbol: sym, qty: shares, side: "buy", type: "market" });
              trendPositions[sym] = { entryPrice: currPrice, peakPrice: currPrice };
              trendBreakCounts[sym] = 0;
              logs.push({
                msg: `🏔 TREND-BUY ${sym}: ${shares} sh | ${ts.daysAbove200}/40 days above 200-SMA | Order: ${order.status}`,
                type: "buy",
              });
              if (Object.keys(trendPositions).length >= 8) break;
            } catch (err) {
              logs.push({ msg: `❌ Trend buy failed ${sym}: ${err.message}`, type: "error" });
            }
          }
        }
      }
    }

    // ── STEP 7: Park idle cash in SPY ──
    if (idleSpyRef && !skipNewBuys) {
      try {
        const freshAcct    = await alpaca.getAccount();
        const freshCash    = freshAcct.cash;
        const freshPortVal = freshAcct.portfolio_value;
        const reservedCash = freshPortVal * SPY_IDLE_RESERVE_PCT;
        const idleCash     = freshCash - reservedCash;

        if (idleCash > freshPortVal * SPY_IDLE_THRESHOLD_PCT) {
          const parkAmount = idleCash * SPY_IDLE_INVEST_PCT;
          const spyPrice   = priceHist.SPY?.[priceHist.SPY.length - 1];
          if (spyPrice && spyPrice > 0 && parkAmount >= spyPrice) {
            const spyShares = Math.floor(parkAmount / spyPrice);
            if (spyShares > 0) {
              await alpaca.placeOrder({ symbol: "SPY", qty: spyShares, side: "buy", type: "market" });
              idleSpyRef.current += spyShares;
              logs.push({
                msg: `💼 [idle-spy] Parking $${parkAmount.toLocaleString("en-US", { maximumFractionDigits: 0 })} → ${spyShares} SPY @ $${spyPrice.toFixed(2)} | Total idle SPY: ${idleSpyRef.current} shares`,
                type: "system",
              });
            }
          }
        }
      } catch (err) {
        logs.push({ msg: `❌ [idle-spy] SPY park failed: ${err.message}`, type: "error" });
      }
    }

    // Refresh positions after trades
    const updatedPositions = await alpaca.getPositions();
    const updatedAccount = await alpaca.getAccount();

    return { logs, account: updatedAccount, positions: updatedPositions };
  } catch (err) {
    logs.push({ msg: `❌ Trading cycle error: ${err.message}`, type: "error" });
    return { logs, account: null, positions: [] };
  }
}
