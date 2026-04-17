import { useRef, useEffect } from "react";
import { Activity } from "lucide-react";
import { card } from "../styles/theme";

export function ActivityFeed({ activityLog }) {
  const logRef = useRef(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [activityLog]);

  const typeColor = (type) => {
    switch (type) {
      case "buy":    return "#10b981";
      case "sell":   return "#ef4444";
      case "profit": return "#f59e0b";
      case "system": return "#0ea5e9";
      default:       return "#6a8099";
    }
  };

  return (
    <div style={card}>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#4a6080", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>
        <Activity size={11} style={{ marginRight: 4 }} /> Live Activity Feed
      </div>
      <div ref={logRef} style={{ maxHeight: 300, overflowY: "auto" }}>
        {activityLog.length === 0 ? (
          <div style={{ color: "#2a3f5a", textAlign: "center", padding: 30, fontSize: 11 }}>
            Waiting for activity...
          </div>
        ) : (
          activityLog.map((l, i) => (
            <div
              key={i}
              style={{
                padding: "6px 0",
                borderBottom: "1px solid rgba(30,45,70,0.15)",
                animation: "fadeIn 0.3s ease",
                fontSize: 11,
                lineHeight: 1.5,
                color: typeColor(l.type),
              }}
            >
              <span style={{ fontSize: 8, color: "#3d5575", marginRight: 6 }}>{l.time}</span>
              {l.msg}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
