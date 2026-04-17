import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { INITIAL_CASH } from "../config/constants";
import { fmt, calcPortfolioValue } from "../utils/formatters";
import { card, FONTS } from "../styles/theme";

export function EquityChart({ portfolioHist, cash, positions, priceHist }) {
  const totalValue = calcPortfolioValue(cash, positions, priceHist);
  const totalReturn = (totalValue - INITIAL_CASH) / INITIAL_CASH;
  const strokeColor = totalReturn >= 0 ? "#10b981" : "#ef4444";

  return (
    <div style={card}>
      <div style={{ fontSize: 10, fontWeight: 700, color: "#4a6080", letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 10 }}>
        Portfolio Equity Curve
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={portfolioHist}>
          <defs>
            <linearGradient id="eqG" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={strokeColor} stopOpacity={0.25} />
              <stop offset="100%" stopColor={strokeColor} stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="tick" hide />
          <YAxis
            tick={{ fontSize: 9, fill: "#3d5575" }}
            axisLine={false}
            tickLine={false}
            domain={["auto", "auto"]}
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
            width={40}
          />
          <Tooltip
            contentStyle={{ background: "#0c1220", border: "1px solid #1e3050", borderRadius: 8, fontSize: 10, fontFamily: FONTS.mono }}
            formatter={(v) => [fmt(v), "Value"]}
            labelFormatter={(l) => `Tick ${l}`}
          />
          <Area type="monotone" dataKey="value" stroke={strokeColor} fill="url(#eqG)" strokeWidth={2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
