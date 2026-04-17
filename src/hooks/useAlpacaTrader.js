// ══════════════════════════════════════════
//  useAlpacaTrader — Live trading hook (Alpaca paper)
//  Drop-in replacement for useAutoTrader
//  Same interface, real market data + real paper orders
// ══════════════════════════════════════════

import { useState, useEffect, useCallback, useRef } from "react";
import { INITIAL_CASH, UNIVERSE, RISK } from "../config/constants";
import { screenStocks } from "../engine/stockScreener";
import { initLivePriceHistory, updateLivePrices } from "../engine/livePriceEngine";
import { executeLiveTradingCycle } from "../engine/liveTradeExecutor";
import { computeRegime, REGIME_RECOVERY_DAYS } from "../engine/regimeEngine";
import * as alpaca from "../engine/alpacaClient";

// How often to poll prices (ms) and trade (ms)
const PRICE_POLL_MS = 15000;     // every 15 seconds
const TRADE_CYCLE_MS = 60000;    // every 60 seconds

export function useAlpacaTrader() {
  const [tick, setTick] = useState(0);
  const [speed, setSpeed] = useState(PRICE_POLL_MS);
  const [paused, setPaused] = useState(false);
  const [cash, setCash] = useState(INITIAL_CASH);
  const [positions, setPositions] = useState({});
  const [priceHist, setPriceHist] = useState({});
  const [portfolioHist, setPortfolioHist] = useState([]);
  const [activityLog, setActivityLog] = useState([]);
  const [tradeCount, setTradeCount] = useState({
    buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0,
  });
  const [volHist, setVolHist] = useState({});
  const [connected, setConnected] = useState(false);
  const [marketOpen, setMarketOpen] = useState(false);
  const [error, setError] = useState(null);
  const [regime, setRegime] = useState("BULLISH");
  const [mlSignals, setMlSignals] = useState(null);   // array | null
  const [mlStatus, setMlStatus] = useState("down");   // "ok" | "stale" | "down"
  const [idleSpyShares, setIdleSpyShares] = useState(0); // shares parked in SPY as cash substitute
  const tradeCycleRef = useRef(0);
  const screenedRef = useRef(UNIVERSE);
  const idleSpySharesRef = useRef(0);                 // mutated in-place by executor
  const trailingPeaksRef    = useRef({});
  const prevRegimeRef       = useRef("BULLISH");
  const cooldownsRef        = useRef({});
  const volHistRef          = useRef({});
  const priceHistRef        = useRef({});
  const trendPositionsRef   = useRef({});
  const trendBreakCountsRef = useRef({});
  const mlSignalsRef        = useRef(null);   // passed to executor each cycle


  const addLog = useCallback((msg, type = "info") => {
    setActivityLog((prev) => [
      ...prev.slice(-100),
      { msg, type, tick: 0, time: new Date().toLocaleTimeString() },
    ]);
  }, []);

  // ── INIT: Connect to Alpaca + load historical data ──
  useEffect(() => {
    let cancelled = false;

    async function init() {
      addLog("🔌 Connecting to Alpaca paper trading...", "system");

      try {
        // Health check
        const health = await alpaca.checkHealth();
        if (!health.connected) {
          throw new Error("Cannot connect to Alpaca. Check your API keys.");
        }
        setConnected(true);
        addLog("✅ Connected to Alpaca paper trading account.", "system");

        // Fetch account
        const account = await alpaca.getAccount();
        setCash(account.cash);
        setPortfolioHist([{ tick: 0, value: account.portfolio_value }]);
        addLog(`💰 Account balance: $${account.portfolio_value.toFixed(2)}`, "system");

        // Fetch current positions
        const pos = await alpaca.getPositions();
        const posMap = {};
        pos.forEach((p) => {
          posMap[p.symbol] = {
            shares: p.qty,
            avgPrice: p.avg_entry_price,
            entryTick: 0,
          };
        });
        setPositions(posMap);
        if (pos.length > 0) {
          addLog(`📦 Loaded ${pos.length} existing position(s).`, "system");
        }

        // Check market clock
        const clock = await alpaca.getClock();
        setMarketOpen(clock.is_open);
        addLog(
          clock.is_open
            ? "🟢 Market is OPEN. Bot will trade live."
            : `🔴 Market is CLOSED. Next open: ${new Date(clock.next_open).toLocaleString()}`,
          "system"
        );

        // Run stock screener
        addLog("🔍 Running stock screener on full market...", "system");
        const screened = await screenStocks();
        if (screened && screened.length > 0) {
          addLog(`✅ Screener found ${screened.length} tradeable stocks.`, "system");
          // Store screened universe for use by trade executor
          screenedRef.current = screened;
        } else {
          addLog("⚠️ Screener unavailable, using default 12 stocks.", "system");
          screenedRef.current = UNIVERSE;
        }

        // Load historical bars for indicators
        addLog("📊 Loading historical bars for indicators...", "system");
        const { hist, volHist: initVolHist } = await initLivePriceHistory(screenedRef.current);
        if (!cancelled) {
          setPriceHist(hist);
          priceHistRef.current = hist;
          setVolHist(initVolHist);
          volHistRef.current = initVolHist;
          const loadedCount = Object.keys(hist).length;
          addLog(`✅ Loaded historical data for ${loadedCount} stocks. Bot is ready.`, "system");
          addLog(
            `⚙️ Config: ${RISK.MAX_POSITION_PCT * 100}% max position, ${Math.abs(RISK.STOP_LOSS_PCT) * 100}% SL, ${RISK.TAKE_PROFIT_PCT * 100}% TP, max ${RISK.MAX_OPEN_POSITIONS} positions.`,
            "system"
          );
        }
      } catch (err) {
        setError(err.message);
        addLog(`❌ Init failed: ${err.message}`, "error");
      }
    }

    init();
    return () => { cancelled = true; };
  }, []);

  // ── PRICE POLLING ──
  useEffect(() => {
    if (paused || !connected) return;

    const iv = setInterval(async () => {
      try {
        // Fetch prices, account, positions, and clock in parallel
        const [
          { hist: updatedHist, volHist: updatedVolHist },
          account,
          pos,
          clock,
          mlData,
        ] = await Promise.all([
          updateLivePrices(priceHistRef.current, volHistRef.current),
          alpaca.getAccount(),
          alpaca.getPositions(),
          alpaca.getClock(),
          alpaca.getMLSignals(),
        ]);

        // Update ML state — stale signals → fallback (pass null to executor)
        if (mlData && Array.isArray(mlData.signals) && !mlData.is_stale) {
          mlSignalsRef.current = mlData.signals;
          setMlSignals(mlData.signals);
          setMlStatus("ok");
        } else if (mlData && Array.isArray(mlData.signals) && mlData.is_stale) {
          mlSignalsRef.current = null;
          setMlSignals(mlData.signals);   // keep for display in scanner
          setMlStatus("stale");
        } else {
          mlSignalsRef.current = null;
          setMlSignals(null);
          setMlStatus("down");
        }

        setPriceHist(updatedHist);
        priceHistRef.current = updatedHist;
        setVolHist(updatedVolHist);
        volHistRef.current = updatedVolHist;
        setTick((t) => t + 1);

        setCash(account.cash);
        setPortfolioHist((prev) => [
          ...prev.slice(-200),
          { tick: prev.length, value: account.portfolio_value },
        ]);

        const posMap = {};
        pos.forEach((p) => {
          posMap[p.symbol] = {
            shares: p.qty,
            avgPrice: p.avg_entry_price,
            entryTick: 0,
          };
        });
        setPositions(posMap);

        setMarketOpen(clock.is_open);

        // ── Compute SPY market regime ──
        const spyPrices = updatedHist.SPY;
        if (spyPrices && spyPrices.length >= 200) {
          const { regime: rawRegime, sma50, sma200, consecutiveDaysAbove50 } = computeRegime(spyPrices);

          // Recovery rule: after BEARISH, require 3 consecutive days above 50-SMA
          let effectiveRegime = rawRegime;
          if (prevRegimeRef.current === "BEARISH" && rawRegime !== "BEARISH" && consecutiveDaysAbove50 < REGIME_RECOVERY_DAYS) {
            effectiveRegime = "BEARISH"; // still confirming recovery
          }

          if (effectiveRegime !== prevRegimeRef.current) {
            const emoji = { BULLISH: "🟢", CAUTIOUS: "🟡", BEARISH: "🔴" }[effectiveRegime];
            const spyPrice = spyPrices[spyPrices.length - 1];
            addLog(
              `${emoji} Regime change: ${prevRegimeRef.current} → ${effectiveRegime} | SPY $${spyPrice.toFixed(2)} | SMA50 $${sma50?.toFixed(2)} | SMA200 $${sma200?.toFixed(2)}`,
              "system"
            );
            if (effectiveRegime === "BEARISH") {
              addLog("🔴 BEARISH regime active — all new buys suspended until SPY recovers.", "system");
            } else if (effectiveRegime === "CAUTIOUS") {
              addLog("🟡 CAUTIOUS regime active — STRONG BUY only, position sizes halved.", "system");
            } else {
              addLog("🟢 BULLISH regime restored — resuming normal trading.", "system");
            }
          }

          prevRegimeRef.current = effectiveRegime;
          setRegime(effectiveRegime);
        }
      } catch (err) {
        console.error("Poll error:", err.message);
      }
    }, speed);

    return () => clearInterval(iv);
  }, [paused, connected, speed]);

  // ── TRADE CYCLE (runs less frequently than price polling) ──
  useEffect(() => {
    if (paused || !connected || !marketOpen) return;

    const iv = setInterval(async () => {
      tradeCycleRef.current++;
      addLog(`🔄 Trade cycle #${tradeCycleRef.current} — scanning...`, "system");

      const result = await executeLiveTradingCycle({ priceHist: priceHistRef.current, volHist: volHistRef.current, trailingPeaks: trailingPeaksRef.current, regime, cooldowns: cooldownsRef.current, cycleNumber: tradeCycleRef.current, trendPositions: trendPositionsRef.current, trendBreakCounts: trendBreakCountsRef.current, mlSignals: mlSignalsRef.current, idleSpyRef: idleSpySharesRef });
      result.logs.forEach((l) => addLog(l.msg, l.type));
      setIdleSpyShares(idleSpySharesRef.current);

      // Update trade counts from order history
      try {
        const orders = await alpaca.getOrders("filled", 100);
        const buys = orders.filter((o) => o.side === "buy").length;
        const sells = orders.filter((o) => o.side === "sell").length;
        setTradeCount((prev) => ({ ...prev, buys, sells }));
      } catch (err) {
        // non-critical
      }
    }, TRADE_CYCLE_MS);

    return () => clearInterval(iv);
  }, [paused, connected, marketOpen]);

  return {
    tick,
    speed,
    setSpeed: (s) => setSpeed(s),
    paused,
    setPaused,
    cash,
    positions,
    priceHist,
    portfolioHist,
    activityLog,
    tradeCount,
    volHist,
    // Extra live-only fields:
    connected,
    marketOpen,
    regime,
    error,
    mlSignals,     // array for display in MarketScanner (may be stale)
    mlStatus,      // "ok" | "stale" | "down" for Header badge
    idleSpyShares, // shares of SPY held as idle cash substitute
  };
}
