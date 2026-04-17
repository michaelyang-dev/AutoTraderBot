// ══════════════════════════════════════════
//  useAlpacaTrader — Thin polling client
//  Reads trading state from the Express server's trading engine.
//  All trading logic runs server-side (server/tradingEngine.js).
//  This hook just polls GET /api/trading-state every 5 seconds.
// ══════════════════════════════════════════

import { useState, useEffect, useRef } from "react";
import { INITIAL_CASH } from "../config/constants";

const STATE_POLL_MS = 5000;   // poll trading state every 5 seconds
const FEED_POLL_MS  = 5000;   // poll activity feed every 5 seconds
const API = "/api";

async function fetchJSON(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export function useAlpacaTrader() {
  const [tick, setTick] = useState(0);
  const [speed, setSpeed] = useState(STATE_POLL_MS);
  const [paused, setPaused] = useState(false);
  const [cash, setCash] = useState(INITIAL_CASH);
  const [portfolioValue, setPortfolioValue] = useState(INITIAL_CASH);
  const [initialPortfolioValue, setInitialPortfolioValue] = useState(null);
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
  const [mlSignals, setMlSignals] = useState(null);
  const [mlStatus, setMlStatus] = useState("down");
  const [idleSpyShares, setIdleSpyShares] = useState(0);

  const lastFeedLength = useRef(0);

  // ── Poll trading state from server ──
  useEffect(() => {
    if (paused) return;

    let cancelled = false;

    async function poll() {
      try {
        const state = await fetchJSON("/trading-state");
        if (cancelled) return;

        setTick(state.tick || 0);
        setCash(state.cash || 0);
        setPortfolioValue(state.portfolioValue || 0);
        setInitialPortfolioValue(state.initialPortfolioValue);
        setPositions(state.positions || {});
        setPriceHist(state.priceHist || {});
        setPortfolioHist(state.portfolioHist || []);
        setTradeCount(state.tradeCount || { buys: 0, sells: 0, wins: 0, losses: 0, totalPnL: 0 });
        setVolHist(state.volHist || {});
        setConnected(state.connected || false);
        setMarketOpen(state.marketOpen || false);
        setError(state.error || null);
        setRegime(state.regime || "BULLISH");
        setMlSignals(state.mlSignals || null);
        setMlStatus(state.mlStatus || "down");
        setIdleSpyShares(state.idleSpyShares || 0);
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setConnected(false);
        }
      }
    }

    // Poll immediately, then on interval
    poll();
    const iv = setInterval(poll, speed);
    return () => { cancelled = true; clearInterval(iv); };
  }, [paused, speed]);

  // ── Poll activity feed from server ──
  useEffect(() => {
    if (paused) return;

    let cancelled = false;

    async function pollFeed() {
      try {
        const feed = await fetchJSON("/activity-feed?limit=200");
        if (cancelled) return;
        // Only update if feed has grown (avoid re-renders on identical data)
        if (feed.length !== lastFeedLength.current) {
          lastFeedLength.current = feed.length;
          setActivityLog(feed);
        }
      } catch {
        // non-critical — state poll handles connectivity errors
      }
    }

    pollFeed();
    const iv = setInterval(pollFeed, FEED_POLL_MS);
    return () => { cancelled = true; clearInterval(iv); };
  }, [paused]);

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
    mlSignals,
    mlStatus,
    idleSpyShares,
    portfolioValue,
    initialPortfolioValue,
  };
}
