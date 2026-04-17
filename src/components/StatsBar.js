import { DollarSign, TrendingUp, TrendingDown, Activity, Target, Zap } from "lucide-react";
import { RISK, INITIAL_CASH } from "../config/constants";
import { fmt, fmtPct, calcPortfolioValue } from "../utils/formatters";
import { card, FONTS } from "../styles/theme";

export function StatsBar({ cash, positions, priceHist, tradeCount, idleSpyShares = 0 }) {
  const totalValue = calcPortfolioValue(cash, positions, priceHist);
  const totalReturn = (totalValue - INITIAL_CASH) / INITIAL_CASH;
  const winRate =
    tradeCount.wins + tradeCount.losses > 0
      ? (tradeCount.wins / (tradeCount.wins + tradeCount.losses) * 100).toFixed(1)
      : "—";

  const spyPrice    = priceHist?.SPY?.[priceHist.SPY.length - 1] ?? 0;
  const idleSpyVal  = idleSpyShares * spyPrice;
  const idleSpyPct  = totalValue > 0 ? idleSpyVal / totalValue : 0;

  // Active positions excludes idle SPY (it's a cash substitute, not a trade slot)
  const activePositions = Object.keys(positions).filter(s => s !== "SPY").length;

  const stats = [
    { label: "Portfolio",  val: fmt(totalValue),  color: "#f0f4f8", icon: <DollarSign size={13} /> },
    { label: "Return",     val: fmtPct(totalReturn), color: totalReturn >= 0 ? "#10b981" : "#ef4444", icon: totalReturn >= 0 ? <TrendingUp size={13} /> : <TrendingDown size={13} /> },
    { label: "Cash",       val: fmt(cash),         color: "#0ea5e9", icon: <DollarSign size={13} /> },
    { label: "Positions",  val: `${activePositions} / ${RISK.MAX_OPEN_POSITIONS}`, color: "#8b5cf6", icon: <Target size={13} /> },
    { label: "Win Rate",   val: winRate + (winRate !== "—" ? "%" : ""), color: "#f59e0b", icon: <Zap size={13} /> },
    { label: "Total P&L",  val: fmt(tradeCount.totalPnL), color: tradeCount.totalPnL >= 0 ? "#10b981" : "#ef4444", icon: <Activity size={13} /> },
    ...(idleSpyShares > 0 ? [{
      label: "SPY Idle",
      val:   `${fmt(idleSpyVal)} (${fmtPct(idleSpyPct)})`,
      color: "#06b6d4",
      icon:  <DollarSign size={13} />,
    }] : []),
  ];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 8, marginBottom: 12 }}>
      {stats.map((s, i) => (
        <div key={i} style={{ ...card, padding: "10px 12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 5, color: "#4a6080", fontSize: 9, textTransform: "uppercase", letterSpacing: 1.5, marginBottom: 5 }}>
            <span style={{ color: s.color }}>{s.icon}</span>{s.label}
          </div>
          <div style={{ fontSize: 16, fontWeight: 800, color: s.color, fontFamily: FONTS.sans }}>{s.val}</div>
        </div>
      ))}
    </div>
  );
}
