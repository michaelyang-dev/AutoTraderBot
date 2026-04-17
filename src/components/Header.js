import { Radio } from "lucide-react";
import { SPEED_OPTIONS } from "../config/constants";
import { FONTS } from "../styles/theme";

const REGIME_STYLE = {
  BULLISH:  { bg: "rgba(16,185,129,0.12)",  border: "rgba(16,185,129,0.25)",  color: "#10b981", dot: "🟢" },
  CAUTIOUS: { bg: "rgba(245,158,11,0.12)",  border: "rgba(245,158,11,0.25)",  color: "#f59e0b", dot: "🟡" },
  BEARISH:  { bg: "rgba(239,68,68,0.12)",   border: "rgba(239,68,68,0.25)",   color: "#ef4444", dot: "🔴" },
};

const ML_STYLE = {
  ok:    { bg: "rgba(16,185,129,0.10)",  border: "rgba(16,185,129,0.25)",  color: "#10b981", dot: "#10b981", label: "ML" },
  stale: { bg: "rgba(245,158,11,0.10)",  border: "rgba(245,158,11,0.25)",  color: "#f59e0b", dot: "#f59e0b", label: "ML STALE" },
  down:  { bg: "rgba(239,68,68,0.08)",   border: "rgba(239,68,68,0.20)",   color: "#ef4444", dot: "#ef4444", label: "ML OFF" },
};

export function Header({ tick, speed, setSpeed, paused, setPaused, regime, mlStatus }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, flexWrap: "wrap", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 36, height: 36, borderRadius: 10,
          background: "linear-gradient(135deg, #10b981 0%, #0ea5e9 100%)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 16, color: "#fff", fontWeight: 900,
        }}>⚡</div>
        <div>
          <div style={{ fontSize: 16, fontWeight: 800, color: "#f0f4f8", fontFamily: FONTS.sans, letterSpacing: "-0.5px" }}>
            AutoTrader <span style={{ color: "#10b981" }}>Engine</span>
          </div>
          <div style={{ fontSize: 9, color: "#4a6080", letterSpacing: 2.5, textTransform: "uppercase" }}>
            Fully Autonomous • Paper Trading
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        {mlStatus && (() => {
          const s = ML_STYLE[mlStatus] ?? ML_STYLE.down;
          return (
            <div style={{
              display: "flex", alignItems: "center", gap: 5,
              padding: "5px 10px", borderRadius: 8, fontSize: 10, fontWeight: 700,
              background: s.bg, color: s.color, border: `1px solid ${s.border}`,
              letterSpacing: 0.5,
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: "50%",
                background: s.dot,
                boxShadow: mlStatus === "ok" ? `0 0 6px ${s.dot}` : "none",
                display: "inline-block",
              }} />
              {s.label}
            </div>
          );
        })()}
        {regime && (() => {
          const s = REGIME_STYLE[regime] ?? REGIME_STYLE.BULLISH;
          return (
            <div style={{
              padding: "5px 10px", borderRadius: 8, fontSize: 10, fontWeight: 700,
              background: s.bg, color: s.color, border: `1px solid ${s.border}`,
              letterSpacing: 0.5,
            }}>
              {s.dot} {regime}
            </div>
          );
        })()}
        <div style={{
          display: "flex", alignItems: "center", gap: 6, padding: "5px 12px",
          borderRadius: 20, fontSize: 10, fontWeight: 700,
          background: paused ? "rgba(239,68,68,0.12)" : "rgba(16,185,129,0.12)",
          color: paused ? "#ef4444" : "#10b981",
          border: `1px solid ${paused ? "rgba(239,68,68,0.2)" : "rgba(16,185,129,0.25)"}`,
        }}>
          <Radio size={10} style={{ animation: paused ? "none" : "pulse 1.5s infinite" }} />
          {paused ? "PAUSED" : "LIVE"}
        </div>

        <button onClick={() => setPaused((p) => !p)} style={{
          padding: "5px 14px", borderRadius: 8, border: "1px solid rgba(40,55,80,0.4)",
          background: "rgba(12,18,30,0.8)", color: "#8899aa", fontSize: 10,
          cursor: "pointer", fontFamily: FONTS.mono, fontWeight: 600,
        }}>
          {paused ? "▶ RESUME" : "❚❚ PAUSE"}
        </button>

        <select value={speed} onChange={(e) => setSpeed(+e.target.value)} style={{
          padding: "5px 10px", borderRadius: 8, border: "1px solid rgba(40,55,80,0.4)",
          background: "rgba(12,18,30,0.8)", color: "#8899aa", fontSize: 10,
          fontFamily: FONTS.mono, cursor: "pointer",
        }}>
          {SPEED_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>

        <div style={{
          padding: "5px 10px", borderRadius: 8, background: "rgba(16,185,129,0.08)",
          color: "#3d8b6e", fontSize: 10, fontWeight: 600,
          border: "1px solid rgba(16,185,129,0.15)",
        }}>
          T:{tick}
        </div>
      </div>
    </div>
  );
}
