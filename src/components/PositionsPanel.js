import { Shield } from "lucide-react";
import { RISK } from "../config/constants";
import { fmt, fmtPct, getPositionEntries } from "../utils/formatters";
import { card, FONTS } from "../styles/theme";

export function PositionsPanel({ positions, priceHist, tick }) {
  const posEntries = getPositionEntries(positions, priceHist);

  return (
    <div style={card}>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#4a6080", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>
        <Shield size={11} style={{ marginRight: 4 }} /> Open Positions ({posEntries.length})
      </div>
      <div style={{ maxHeight: 300, overflowY: "auto" }}>
        {posEntries.length === 0 ? (
          <div style={{ color: "#2a3f5a", textAlign: "center", padding: 30, fontSize: 11 }}>
            {tick < 35 ? "🔍 Scanning... signals need ~35 ticks of data" : "No positions — waiting for signals"}
          </div>
        ) : (
          posEntries.map((p) => (
            <div key={p.sym} style={{ padding: "10px 0", borderBottom: "1px solid rgba(30,45,70,0.3)", animation: "fadeIn 0.3s ease" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <span style={{ fontWeight: 800, fontSize: 13, color: "#e0e8f0", fontFamily: FONTS.sans }}>{p.sym}</span>
                  <span style={{ fontSize: 9, color: "#4a6080", marginLeft: 6 }}>{p.shares} shares</span>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 13, fontWeight: 800, color: p.pnl >= 0 ? "#10b981" : "#ef4444", fontFamily: FONTS.sans }}>
                    {p.pnl >= 0 ? "+" : ""}{fmt(p.pnl)}
                  </div>
                  <div style={{ fontSize: 9, color: p.pnlPct >= 0 ? "#10b981" : "#ef4444" }}>
                    {fmtPct(p.pnlPct)}
                  </div>
                </div>
              </div>
              {/* Stop-loss / Take-profit progress bar */}
              <div style={{ marginTop: 6, height: 4, borderRadius: 2, background: "rgba(30,45,70,0.4)", position: "relative", overflow: "hidden" }}>
                <div style={{
                  position: "absolute", left: "0%",
                  width: `${Math.min(100, Math.max(0, ((p.pnlPct - RISK.STOP_LOSS_PCT) / (RISK.TAKE_PROFIT_PCT - RISK.STOP_LOSS_PCT)) * 100))}%`,
                  height: "100%", borderRadius: 2,
                  background: p.pnlPct >= 0
                    ? "linear-gradient(90deg, #0ea5e9, #10b981)"
                    : "linear-gradient(90deg, #ef4444, #f59e0b)",
                }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#3d5575", marginTop: 2 }}>
                <span>SL: {fmtPct(RISK.STOP_LOSS_PCT)}</span>
                <span>Entry: {fmt(p.avgPrice)}</span>
                <span>TP: {fmtPct(RISK.TAKE_PROFIT_PCT)}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
