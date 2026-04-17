import { useMemo } from "react";
import { Eye } from "lucide-react";
import { UNIVERSE, getSector, RISK } from "../config/constants";
import { getSignals } from "../engine/signalEngine";
import { avgVolume, weeklyTrend } from "../engine/indicators";
import { fmt, fmtPct } from "../utils/formatters";
import { card, FONTS, signalColor, consensusColor } from "../styles/theme";

export function MarketScanner({ priceHist, volHist = {}, positions, mlSignals = null }) {
  // Build ML probability lookup map
  const mlMap = useMemo(() => {
    if (!mlSignals) return {};
    return Object.fromEntries(mlSignals.map((s) => [s.symbol, s]));
  }, [mlSignals]);

  // Compute all signals once per priceHist/volHist change, not on every render.
  const rowData = useMemo(() => {
    const result = {};
    UNIVERSE.forEach(({ sym }) => {
      const prices = priceHist[sym];
      if (!prices || prices.length < 2) return;
      const vols = volHist[sym];
      const volAvg = avgVolume(vols, 20);
      result[sym] = {
        curr:     prices[prices.length - 1],
        prev:     prices[prices.length - 2],
        analysis: prices.length >= 35 ? getSignals(prices) : null,
        wt:       weeklyTrend(prices),
        volRatio: volAvg && vols ? vols[vols.length - 1] / volAvg : null,
      };
    });
    return result;
  }, [priceHist, volHist]);

  return (
    <div style={{ ...card, marginTop: 10 }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#4a6080", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>
        <Eye size={11} style={{ marginRight: 4 }} /> Market Scanner — All Signals
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(30,45,70,0.4)" }}>
              {["Symbol", "Price", "Change", "Vol", "RSI", "SMA", "MACD", "Boll", "Mom", "Consensus", "ML%"].map((h) => (
                <th key={h} style={{
                  padding: "8px 6px", textAlign: "left", fontSize: 9, color: "#3d5575",
                  fontWeight: 600, letterSpacing: 1, textTransform: "uppercase",
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {UNIVERSE.map(({ sym }) => {
              const row = rowData[sym];
              if (!row) return null;
              const { curr, prev, analysis, wt, volRatio } = row;
              const chg = (curr - prev) / prev;

              const volColor = volRatio === null ? "#2a3f5a"
                : volRatio >= RISK.VOLUME_CONFIRM_RATIO ? "#10b981"
                : volRatio >= 1.0 ? "#f59e0b"
                : "#ef4444";

              return (
                <tr key={sym} style={{ borderBottom: "1px solid rgba(30,45,70,0.15)" }}>
                  <td style={{ padding: "7px 6px", fontFamily: FONTS.sans }}>
                    <span style={{ fontWeight: 700, color: positions[sym] ? "#10b981" : "#c8d6e5" }}>
                      {sym}
                    </span>
                    {positions[sym] && <span style={{ fontSize: 8, color: "#10b981", marginLeft: 3 }}>●</span>}
                    <span style={{
                      display: "inline-block", marginLeft: 5, fontSize: 8, fontWeight: 600,
                      padding: "1px 4px", borderRadius: 3,
                      background: "rgba(30,45,70,0.6)", color: "#4a6080",
                      letterSpacing: 0.5, textTransform: "uppercase",
                    }}>
                      {getSector(sym)}
                    </span>
                    {wt && (
                      <span
                        title={`Weekly trend: ${wt.trend} (5w SMA ${wt.smaFast.toFixed(2)} vs 15w SMA ${wt.smaSlow.toFixed(2)})`}
                        style={{ marginLeft: 4, fontSize: 10, color: wt.trend === "BULLISH" ? "#10b981" : "#ef4444" }}
                      >
                        {wt.trend === "BULLISH" ? "▲" : "▼"}
                      </span>
                    )}
                  </td>
                  <td style={{ padding: "7px 6px", color: "#c8d6e5" }}>{fmt(curr)}</td>
                  <td style={{ padding: "7px 6px", color: chg >= 0 ? "#10b981" : "#ef4444" }}>{fmtPct(chg)}</td>
                  <td style={{ padding: "7px 6px", color: volColor, fontWeight: 700, fontSize: 10 }}>
                    {volRatio !== null ? `${volRatio.toFixed(1)}x` : "—"}
                  </td>
                  <td style={{
                    padding: "7px 6px",
                    color: analysis
                      ? analysis.indicators.rsi < 30 ? "#10b981" : analysis.indicators.rsi > 70 ? "#ef4444" : "#6a8099"
                      : "#2a3f5a",
                  }}>
                    {analysis ? analysis.indicators.rsi.toFixed(0) : "—"}
                  </td>
                  {["sma", "macd", "boll", "mom"].map((s) => (
                    <td key={s} style={{ padding: "7px 6px", color: signalColor(analysis?.signals[s]), fontSize: 10, fontWeight: 600 }}>
                      {analysis?.signals[s] || "—"}
                    </td>
                  ))}
                  <td style={{ padding: "7px 6px", fontWeight: 800, fontSize: 10, color: consensusColor(analysis?.consensus) }}>
                    {analysis?.consensus || "SCANNING"}
                  </td>
                  {(() => {
                    const ml = mlMap[sym];
                    if (!ml) return <td style={{ padding: "7px 6px", color: "#2a3f5a", fontSize: 10 }}>—</td>;
                    const prob = ml.probability;
                    const color = prob >= 0.60 ? "#10b981" : prob >= 0.50 ? "#f59e0b" : "#6a8099";
                    return (
                      <td style={{ padding: "7px 6px", fontSize: 10, fontWeight: 700, color }}>
                        {(prob * 100).toFixed(0)}%
                        {ml.signal === "BUY" && <span style={{ marginLeft: 3, fontSize: 8, color: "#10b981" }}>▲</span>}
                      </td>
                    );
                  })()}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
