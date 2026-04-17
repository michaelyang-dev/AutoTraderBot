import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { PIE_COLORS } from "../config/constants";
import { fmt, getAllocationData } from "../utils/formatters";
import { card, FONTS } from "../styles/theme";

export function AllocationChart({ positions, priceHist, cash }) {
  const data = getAllocationData(positions, priceHist, cash);

  return (
    <div style={card}>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#4a6080", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>
        Capital Allocation
      </div>
      {data.length > 0 ? (
        <ResponsiveContainer width="100%" height={180}>
          <PieChart>
            <Pie data={data} cx="50%" cy="50%" innerRadius={45} outerRadius={72} paddingAngle={2} dataKey="value" stroke="none">
              {data.map((_, i) => (
                <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ background: "#0c1220", border: "1px solid #1e3050", borderRadius: 8, fontSize: 10, fontFamily: FONTS.mono }}
              formatter={(v) => fmt(v)}
            />
          </PieChart>
        </ResponsiveContainer>
      ) : (
        <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", color: "#2a3f5a" }}>
          Waiting for data...
        </div>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 4 }}>
        {data.map((d, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 9, color: "#6a8099" }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: PIE_COLORS[i % PIE_COLORS.length] }} />
            {d.name}
          </div>
        ))}
      </div>
    </div>
  );
}
