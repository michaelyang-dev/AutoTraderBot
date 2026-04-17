import { STRATEGIES } from "../config/constants";

export function StrategyLegend({ tradeCount }) {
  return (
    <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
      {Object.entries(STRATEGIES).map(([k, s]) => (
        <div key={k} style={{
          padding: "6px 12px", borderRadius: 8,
          background: "rgba(12,18,30,0.7)", border: "1px solid rgba(40,55,80,0.3)",
          fontSize: 10, color: s.color, display: "flex", alignItems: "center", gap: 5,
        }}>
          <span>{s.icon}</span> {s.name}
        </div>
      ))}
      <div style={{
        padding: "6px 12px", borderRadius: 8,
        background: "rgba(12,18,30,0.7)", border: "1px solid rgba(40,55,80,0.3)",
        fontSize: 10, color: "#4a6080", marginLeft: "auto",
      }}>
        Trades: {tradeCount.buys}B / {tradeCount.sells}S • W/L: {tradeCount.wins}/{tradeCount.losses}
      </div>
    </div>
  );
}
