// ══════════════════════════════════════════
//  App.js — Supports both SIMULATED and LIVE modes
//  Toggle between modes with the selector at the top
//  Simulated: no API keys needed, fake prices
//  Live: connects to Alpaca paper trading
// ══════════════════════════════════════════

import React, { useState } from "react";
import { useAutoTrader } from "./hooks/useAutoTrader";
import { useAlpacaTrader } from "./hooks/useAlpacaTrader";
import { FONTS, globalCSS } from "./styles/theme";
import {
  Header,
  StatsBar,
  EquityChart,
  AllocationChart,
  PositionsPanel,
  ActivityFeed,
  MarketScanner,
  StrategyLegend,
  BacktestTab,
} from "./components";

function ModeSelector({ mode, setMode }) {
  return (
    <div style={{
      display: "flex", gap: 6, marginBottom: 14,
      padding: 4, background: "rgba(12,18,30,0.85)",
      borderRadius: 10, border: "1px solid rgba(40,55,80,0.4)",
      width: "fit-content",
    }}>
      {[
        { id: "sim",      label: "⚡ Simulated" },
        { id: "live",     label: "🔌 Alpaca Paper" },
        { id: "backtest", label: "🧪 Backtest" },
      ].map((m) => (
        <button
          key={m.id}
          onClick={() => setMode(m.id)}
          style={{
            padding: "8px 16px", borderRadius: 8, border: "none", cursor: "pointer",
            background: mode === m.id ? "rgba(16,185,129,0.15)" : "transparent",
            color: mode === m.id ? "#10b981" : "#4a6080",
            fontFamily: FONTS.mono, fontSize: 11, fontWeight: 600,
            transition: "all 0.2s",
          }}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

function ConnectionBanner({ connected, marketOpen, error }) {
  if (error) {
    return (
      <div style={{
        padding: "10px 16px", borderRadius: 10, marginBottom: 12,
        background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.25)",
        color: "#ef4444", fontSize: 11, fontFamily: FONTS.mono,
      }}>
        ❌ {error} — Check your .env file and make sure the server is running (npm run server)
      </div>
    );
  }

  if (!connected) {
    return (
      <div style={{
        padding: "10px 16px", borderRadius: 10, marginBottom: 12,
        background: "rgba(245,158,11,0.1)", border: "1px solid rgba(245,158,11,0.25)",
        color: "#f59e0b", fontSize: 11, fontFamily: FONTS.mono,
      }}>
        🔄 Connecting to Alpaca...
      </div>
    );
  }

  return (
    <div style={{
      padding: "8px 16px", borderRadius: 10, marginBottom: 12,
      background: marketOpen ? "rgba(16,185,129,0.08)" : "rgba(245,158,11,0.08)",
      border: `1px solid ${marketOpen ? "rgba(16,185,129,0.2)" : "rgba(245,158,11,0.2)"}`,
      color: marketOpen ? "#10b981" : "#f59e0b",
      fontSize: 10, fontFamily: FONTS.mono,
      display: "flex", alignItems: "center", gap: 8,
    }}>
      <span style={{ fontSize: 8 }}>{marketOpen ? "🟢" : "🟡"}</span>
      {marketOpen
        ? "Connected to Alpaca • Market OPEN • Bot is trading"
        : "Connected to Alpaca • Market CLOSED • Bot waiting for open"
      }
    </div>
  );
}

function Dashboard({ trader }) {
  return (
    <>
      <StatsBar
        cash={trader.cash}
        positions={trader.positions}
        priceHist={trader.priceHist}
        tradeCount={trader.tradeCount}
        idleSpyShares={trader.idleSpyShares ?? 0}
      />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
        <EquityChart
          portfolioHist={trader.portfolioHist}
          cash={trader.cash}
          positions={trader.positions}
          priceHist={trader.priceHist}
        />
        <AllocationChart
          positions={trader.positions}
          priceHist={trader.priceHist}
          cash={trader.cash}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <PositionsPanel
          positions={trader.positions}
          priceHist={trader.priceHist}
          tick={trader.tick}
        />
        <ActivityFeed activityLog={trader.activityLog} />
      </div>

      <MarketScanner priceHist={trader.priceHist} volHist={trader.volHist} positions={trader.positions} mlSignals={trader.mlSignals} />
      <StrategyLegend tradeCount={trader.tradeCount} />
    </>
  );
}

function SimApp() {
  const trader = useAutoTrader();
  return (
    <>
      <Header
        tick={trader.tick}
        speed={trader.speed}
        setSpeed={trader.setSpeed}
        paused={trader.paused}
        setPaused={trader.setPaused}
        regime={null}
      />
      <Dashboard trader={trader} />
    </>
  );
}

function LiveApp() {
  const trader = useAlpacaTrader();
  return (
    <>
      <Header
        tick={trader.tick}
        speed={trader.speed}
        setSpeed={trader.setSpeed}
        paused={trader.paused}
        setPaused={trader.setPaused}
        regime={trader.regime}
        mlStatus={trader.mlStatus}
      />
      <ConnectionBanner
        connected={trader.connected}
        marketOpen={trader.marketOpen}
        error={trader.error}
      />
      <Dashboard trader={trader} />
    </>
  );
}

export default function App() {
  const [mode, setMode] = useState("sim");

  return (
    <div style={{
      minHeight: "100vh", padding: "6px 8px",
      background: "radial-gradient(ellipse at 20% 0%, #0b1628 0%, #060b14 50%, #030508 100%)",
      color: "#c8d6e5", fontFamily: FONTS.mono, fontSize: 12,
    }}>
      <style>{globalCSS}</style>
      <ModeSelector mode={mode} setMode={setMode} />
      {mode === "sim"      && <SimApp />}
      {mode === "live"     && <LiveApp />}
      {mode === "backtest" && <BacktestTab />}
      <div style={{ textAlign: "center", marginTop: 8, fontSize: 9, color: "#1e3050", letterSpacing: 1.5 }}>
        ⚠ PAPER TRADING ONLY — NO REAL CAPITAL AT RISK — FOR EDUCATIONAL PURPOSES ONLY
      </div>
    </div>
  );
}
