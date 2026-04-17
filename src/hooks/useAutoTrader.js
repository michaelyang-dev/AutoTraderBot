// ══════════════════════════════════════════
//  useAutoTrader — Core simulation hook
// ══════════════════════════════════════════

import { useState, useEffect, useCallback, useRef } from "react";
import { INITIAL_CASH, UNIVERSE, RISK } from "../config/constants";
import { initPriceHistory, advancePrices, initVolumeHistory, advanceVolumes } from "../engine/priceEngine";
import { executeTradingCycle } from "../engine/tradeExecutor";
import { calcPortfolioValue } from "../utils/formatters";

export function useAutoTrader() {
  const [tick, setTick] = useState(0);
  const [speed, setSpeed] = useState(800);
  const [paused, setPaused] = useState(false);
  const [cash, setCash] = useState(INITIAL_CASH);
  const [positions, setPositions] = useState({});
  const [priceHist, setPriceHist] = useState(() => initPriceHistory(UNIVERSE));
  const [volHist, setVolHist] = useState(() => initVolumeHistory(UNIVERSE));
  const [portfolioHist, setPortfolioHist] = useState([{ tick: 0, value: INITIAL_CASH }]);
  const [activityLog, setActivityLog] = useState([]);
  const [tradeCount, setTradeCount] = useState({
    buys: 0,
    sells: 0,
    wins: 0,
    losses: 0,
    totalPnL: 0,
  });
  const [cooldowns, setCooldowns] = useState({});
  const [trendPositions, setTrendPositions] = useState({});
  const [trendBreakCounts, setTrendBreakCounts] = useState({});
  const cycleCountRef = useRef(0);

  // Add a message to the activity log
  const addLog = useCallback(
    (msg, type = "info") => {
      setActivityLog((prev) => [
        ...prev.slice(-80),
        { msg, type, tick, time: new Date().toLocaleTimeString() },
      ]);
    },
    [tick]
  );

  // Boot message
  useEffect(() => {
    addLog("🤖 AutoTrader initialized. Bot is running autonomously.", "system");
    addLog("📋 Scanning 12 stocks with 5-strategy consensus engine.", "system");
    addLog(
      `⚙️ Risk: max ${RISK.MAX_POSITION_PCT * 100}% per position, ${Math.abs(RISK.STOP_LOSS_PCT) * 100}% stop-loss, ${RISK.TAKE_PROFIT_PCT * 100}% take-profit.`,
      "system"
    );
  }, []);

  // Tick engine
  useEffect(() => {
    if (paused) return;
    const iv = setInterval(() => setTick((t) => t + 1), speed);
    return () => clearInterval(iv);
  }, [paused, speed]);

  // Advance prices each tick
  useEffect(() => {
    if (tick === 0) return;
    setPriceHist((prev) => advancePrices(prev, UNIVERSE, tick));
    setVolHist((prev) => advanceVolumes(prev, UNIVERSE, tick));
  }, [tick]);

  // Auto-trade on rebalance interval
  useEffect(() => {
    if (tick === 0 || tick % RISK.REBALANCE_INTERVAL !== 0) return;

    cycleCountRef.current++;
    const result = executeTradingCycle({
      cash,
      positions,
      priceHist,
      tick,
      stats: tradeCount,
      cooldowns,
      cycleNumber: cycleCountRef.current,
      volHist,
      trendPositions,
      trendBreakCounts,
    });

    setCash(result.cash);
    setPositions(result.positions);
    setTradeCount(result.stats);
    setCooldowns(result.cooldowns);
    setTrendPositions(result.trendPositions);
    setTrendBreakCounts(result.trendBreakCounts);
    result.logs.forEach((l) => addLog(l.msg, l.type));
  }, [tick]);

  // Track portfolio value
  useEffect(() => {
    const total = calcPortfolioValue(cash, positions, priceHist, trendPositions);
    setPortfolioHist((prev) => [...prev.slice(-150), { tick, value: total }]);
  }, [tick]);

  return {
    tick,
    speed,
    setSpeed,
    paused,
    setPaused,
    cash,
    positions,
    priceHist,
    volHist,
    portfolioHist,
    activityLog,
    tradeCount,
  };
}
