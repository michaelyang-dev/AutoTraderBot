// ══════════════════════════════════════════
//  BacktestTab — Full backtesting UI
// ══════════════════════════════════════════

import { useState, useCallback, useMemo } from "react";
import {
  AreaChart, Area, BarChart, Bar, ComposedChart, Line, XAxis, YAxis,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";
import { FlaskConical, Play, Download, Trash2, TrendingUp, TrendingDown, BarChart2, Clock, ChevronDown, ChevronUp } from "lucide-react";
import { card, FONTS } from "../styles/theme";
import {
  runBacktest, saveRunToStorage, listStoredRuns, loadStoredRun, deleteStoredRun,
} from "../engine/backtester";
import { INITIAL_CASH } from "../config/constants";

// ── Helpers ──
const pct = (v, d = 1) => v != null ? `${(v * 100).toFixed(d)}%` : "—";
const money = (v) => v != null ? `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—";
const round2 = (v) => v != null ? v.toFixed(2) : "—";

function MetricCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: "rgba(14,22,38,0.7)", borderRadius: 10,
      border: "1px solid rgba(40,55,80,0.35)", padding: "12px 14px",
    }}>
      <div style={{ fontSize: 9, color: "#3d5575", letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 800, color: color || "#c8d6e5", fontFamily: FONTS.mono }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 9, color: "#4a6080", marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

// ── Custom Tooltip ──
function EquityTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const v = payload[0].value;
  const gain = v - INITIAL_CASH;
  return (
    <div style={{
      background: "rgba(8,14,24,0.95)", border: "1px solid rgba(40,55,80,0.5)",
      borderRadius: 8, padding: "8px 12px", fontSize: 10, fontFamily: FONTS.mono,
    }}>
      <div style={{ color: "#4a6080", marginBottom: 3 }}>{label}</div>
      <div style={{ color: "#c8d6e5" }}>${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
      <div style={{ color: gain >= 0 ? "#10b981" : "#ef4444" }}>
        {gain >= 0 ? "+" : ""}{pct(gain / INITIAL_CASH)}
      </div>
    </div>
  );
}

// ── Equity chart ──
function EquityChart({ curve, benchmark }) {
  if (!curve?.length) return null;
  const data = curve.map((pt) => ({
    date: pt.date,
    value: Math.round(pt.value),
    bm: benchmark ? Math.round(INITIAL_CASH * (1 + (benchmark.find((b) => b.date === pt.date)?.ret || 0))) : null,
  }));

  const min = Math.min(...data.map((d) => d.value));
  const max = Math.max(...data.map((d) => d.value));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ComposedChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="btGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#10b981" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis dataKey="date" tick={{ fontSize: 8, fill: "#3d5575" }} tickLine={false}
          axisLine={false} interval="preserveStartEnd" tickFormatter={(d) => d.slice(0, 7)} />
        <YAxis tick={{ fontSize: 8, fill: "#3d5575" }} tickLine={false} axisLine={false}
          tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
          domain={[Math.floor(min * 0.98 / 1000) * 1000, Math.ceil(max * 1.02 / 1000) * 1000]} />
        <Tooltip content={<EquityTooltip />} />
        <ReferenceLine y={INITIAL_CASH} stroke="rgba(100,120,160,0.3)" strokeDasharray="3 3" />
        <Area type="monotone" dataKey="value" stroke="#10b981" strokeWidth={1.5}
          fill="url(#btGradient)" dot={false} />
        {benchmark && (
          <Line type="monotone" dataKey="bm" stroke="#f59e0b" strokeWidth={1}
            dot={false} strokeDasharray="4 2" />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ── Monthly returns heatmap ──
function MonthlyHeatmap({ monthlyReturns }) {
  if (!monthlyReturns?.length) return null;

  const years = [...new Set(monthlyReturns.map((m) => m.month.slice(0, 4)))].sort();
  const months = ["01","02","03","04","05","06","07","08","09","10","11","12"];
  const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

  const lookup = {};
  monthlyReturns.forEach((m) => { lookup[m.month] = m.return; });

  const cellColor = (r) => {
    if (r == null) return "rgba(20,32,52,0.4)";
    if (r > 0.05)  return "rgba(16,185,129,0.85)";
    if (r > 0.02)  return "rgba(16,185,129,0.55)";
    if (r > 0)     return "rgba(16,185,129,0.25)";
    if (r > -0.02) return "rgba(239,68,68,0.25)";
    if (r > -0.05) return "rgba(239,68,68,0.55)";
    return "rgba(239,68,68,0.85)";
  };

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontSize: 9, fontFamily: FONTS.mono }}>
        <thead>
          <tr>
            <th style={{ padding: "3px 8px", color: "#3d5575", textAlign: "left" }}>Year</th>
            {monthNames.map((m) => (
              <th key={m} style={{ padding: "3px 6px", color: "#3d5575", textAlign: "center", minWidth: 36 }}>{m}</th>
            ))}
            <th style={{ padding: "3px 8px", color: "#3d5575", textAlign: "center" }}>YTD</th>
          </tr>
        </thead>
        <tbody>
          {years.map((yr) => {
            let ytd = 1;
            return (
              <tr key={yr}>
                <td style={{ padding: "3px 8px", color: "#4a6080", fontWeight: 600 }}>{yr}</td>
                {months.map((mo) => {
                  const key = `${yr}-${mo}`;
                  const r = lookup[key];
                  if (r != null) ytd *= (1 + r);
                  return (
                    <td key={mo} title={r != null ? pct(r) : ""} style={{
                      padding: "3px 6px", textAlign: "center", borderRadius: 3,
                      background: cellColor(r), color: r != null ? "#c8d6e5" : "#1e3050",
                    }}>
                      {r != null ? pct(r, 0) : ""}
                    </td>
                  );
                })}
                <td style={{
                  padding: "3px 8px", textAlign: "center", fontWeight: 700,
                  color: ytd >= 1 ? "#10b981" : "#ef4444",
                }}>
                  {pct(ytd - 1, 1)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Trade list ──
function TradeTable({ trades }) {
  const [page, setPage] = useState(0);
  const PER_PAGE = 20;
  const closed = useMemo(() => trades.filter((t) => t.exitDate).sort((a, b) => b.exitDate?.localeCompare(a.exitDate)), [trades]);
  const totalPages = Math.ceil(closed.length / PER_PAGE);
  const visible = closed.slice(page * PER_PAGE, page * PER_PAGE + PER_PAGE);

  return (
    <div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(30,45,70,0.4)" }}>
              {["Symbol","Entry","Exit","Reason","Shares","Entry $","Exit $","P&L","P&L %","Hold"].map((h) => (
                <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontSize: 9, color: "#3d5575",
                  fontWeight: 600, letterSpacing: 0.8, textTransform: "uppercase" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.map((t, i) => (
              <tr key={i} style={{ borderBottom: "1px solid rgba(30,45,70,0.1)" }}>
                <td style={{ padding: "5px 8px", fontWeight: 700, color: "#c8d6e5" }}>{t.sym}</td>
                <td style={{ padding: "5px 8px", color: "#4a6080" }}>{t.entryDate}</td>
                <td style={{ padding: "5px 8px", color: "#4a6080" }}>{t.exitDate}</td>
                <td style={{ padding: "5px 8px", color: "#6a8099", fontSize: 9 }}>{t.exitReason}</td>
                <td style={{ padding: "5px 8px", color: "#4a6080" }}>{t.shares}</td>
                <td style={{ padding: "5px 8px", color: "#c8d6e5" }}>${t.entryPrice?.toFixed(2)}</td>
                <td style={{ padding: "5px 8px", color: "#c8d6e5" }}>${t.exitPrice?.toFixed(2)}</td>
                <td style={{ padding: "5px 8px", fontWeight: 700, color: t.pnl >= 0 ? "#10b981" : "#ef4444" }}>
                  {t.pnl >= 0 ? "+" : ""}${t.pnl?.toFixed(0)}
                </td>
                <td style={{ padding: "5px 8px", color: t.pnlPct >= 0 ? "#10b981" : "#ef4444" }}>
                  {t.pnlPct >= 0 ? "+" : ""}{pct(t.pnlPct)}
                </td>
                <td style={{ padding: "5px 8px", color: "#4a6080" }}>{t.holdDays}d</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div style={{ display: "flex", gap: 8, marginTop: 8, justifyContent: "center" }}>
          <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0}
            style={{ ...btnStyle, opacity: page === 0 ? 0.3 : 1 }}>← Prev</button>
          <span style={{ fontSize: 10, color: "#4a6080", lineHeight: "28px" }}>
            {page + 1} / {totalPages}
          </span>
          <button onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))} disabled={page === totalPages - 1}
            style={{ ...btnStyle, opacity: page === totalPages - 1 ? 0.3 : 1 }}>Next →</button>
        </div>
      )}
    </div>
  );
}

// ── Walk-forward summary ──
function WalkForwardPanel({ wf }) {
  if (!wf) return null;
  const { windows, oosMetrics } = wf;

  return (
    <div>
      <div style={{ fontSize: 10, color: "#4a6080", marginBottom: 10 }}>
        Chained out-of-sample performance across {windows.length} windows
      </div>
      {oosMetrics && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 14 }}>
          <MetricCard label="OOS CAGR" value={pct(oosMetrics.cagr)} color={oosMetrics.cagr >= 0 ? "#10b981" : "#ef4444"} />
          <MetricCard label="OOS Sharpe" value={round2(oosMetrics.sharpe)} color={oosMetrics.sharpe >= 1 ? "#10b981" : "#f59e0b"} />
          <MetricCard label="OOS Max DD" value={pct(oosMetrics.maxDrawdownPct)} color="#ef4444" />
          <MetricCard label="OOS Win Rate" value={pct(oosMetrics.winRate)} />
        </div>
      )}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(30,45,70,0.4)" }}>
              {["IS Period","OOS Period","IS CAGR","OOS CAGR","IS Sharpe","OOS Sharpe","IS WinR","OOS WinR"].map((h) => (
                <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontSize: 9,
                  color: "#3d5575", letterSpacing: 0.8, textTransform: "uppercase" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {windows.map((w, i) => {
              const im = w.inSample.metrics;
              const om = w.outSample.metrics;
              const cagrDelta = om?.cagr != null && im?.cagr != null ? om.cagr - im.cagr : null;
              return (
                <tr key={i} style={{ borderBottom: "1px solid rgba(30,45,70,0.1)" }}>
                  <td style={{ padding: "5px 8px", color: "#4a6080", fontSize: 9 }}>{w.isStart} → {w.isEnd}</td>
                  <td style={{ padding: "5px 8px", color: "#4a6080", fontSize: 9 }}>{w.oosStart} → {w.oosEnd}</td>
                  <td style={{ padding: "5px 8px", color: (im?.cagr ?? 0) >= 0 ? "#10b981" : "#ef4444" }}>{pct(im?.cagr)}</td>
                  <td style={{ padding: "5px 8px", fontWeight: 700, color: (om?.cagr ?? 0) >= 0 ? "#10b981" : "#ef4444" }}>
                    {pct(om?.cagr)}
                    {cagrDelta != null && (
                      <span style={{ fontSize: 8, color: cagrDelta >= 0 ? "#10b981" : "#ef4444", marginLeft: 4 }}>
                        ({cagrDelta >= 0 ? "+" : ""}{pct(cagrDelta, 0)})
                      </span>
                    )}
                  </td>
                  <td style={{ padding: "5px 8px", color: "#4a6080" }}>{round2(im?.sharpe)}</td>
                  <td style={{ padding: "5px 8px", color: "#4a6080" }}>{round2(om?.sharpe)}</td>
                  <td style={{ padding: "5px 8px", color: "#4a6080" }}>{pct(im?.winRate)}</td>
                  <td style={{ padding: "5px 8px", color: "#4a6080" }}>{pct(om?.winRate)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Compare panel ──
// Rows define which metric to show, how to format it, and which direction is "better".
const COMPARE_ROWS = [
  { key: "cagr",           label: "CAGR",               fmt: (v) => pct(v),                   better: "higher" },
  { key: "sharpe",         label: "Sharpe Ratio",        fmt: (v) => round2(v),                better: "higher" },
  { key: "sortino",        label: "Sortino Ratio",       fmt: (v) => round2(v),                better: "higher" },
  { key: "maxDrawdownPct", label: "Max Drawdown",        fmt: (v) => pct(v),                   better: "higher" }, // less negative = better
  { key: "calmar",         label: "Calmar Ratio",        fmt: (v) => round2(v),                better: "higher" },
  { key: "winRate",        label: "Win Rate",            fmt: (v) => pct(v),                   better: "higher" },
  { key: "profitFactor",   label: "Profit Factor",       fmt: (v) => isFinite(v) ? round2(v) : "∞", better: "higher" },
  { key: "avgWinPct",      label: "Avg Win %",           fmt: (v) => pct(v),                   better: "higher" },
  { key: "avgLossPct",     label: "Avg Loss %",          fmt: (v) => pct(v),                   better: "higher" }, // less negative = better
  { key: "avgHoldDays",    label: "Avg Hold Days",       fmt: (v) => v?.toFixed(1) + "d",      better: null     },
  { key: "totalTrades",    label: "Total Trades",        fmt: (v) => String(v ?? "—"),          better: null     },
  { key: "netPnL",         label: "Net P&L",             fmt: (v) => money(v),                 better: "higher" },
  { key: "beta",           label: "Beta vs SPY",         fmt: (v) => round2(v),                better: "lower"  },
  { key: "alpha",          label: "Alpha vs SPY (ann.)", fmt: (v) => pct(v),                   better: "higher" },
  { key: "tradingDays",    label: "Trading Days",        fmt: (v) => String(v ?? "—"),          better: null     },
];

function ComparePanel({ runA, runB }) {
  const mA = runA.metrics;
  const mB = runB.metrics;

  const winner = (row, a, b) => {
    if (row.better === null || a == null || b == null) return null;
    if (row.better === "higher") return a > b ? "A" : b > a ? "B" : null;
    return a < b ? "A" : b < a ? "B" : null; // "lower is better"
  };

  const cellColor = (which, w) => {
    if (!w) return "#4a6080";
    return w === which ? "#10b981" : "#ef4444";
  };

  return (
    <div style={{ ...card, padding: 14, marginTop: 10 }}>
      <div style={{ fontSize: 9, color: "#3d5575", letterSpacing: 1.2, textTransform: "uppercase", marginBottom: 12 }}>
        Side-by-Side Comparison
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 10 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(30,45,70,0.5)" }}>
              <th style={{ padding: "7px 10px", textAlign: "left", fontSize: 9, color: "#3d5575",
                fontWeight: 600, letterSpacing: 0.8, textTransform: "uppercase" }}>Metric</th>
              {[runA, runB].map((r, i) => (
                <th key={i} style={{ padding: "7px 10px", textAlign: "right", fontSize: 9,
                  color: "#c8d6e5", fontWeight: 700, maxWidth: 160 }}>
                  {r.label || r.config?.start + " → " + r.config?.end}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {COMPARE_ROWS.map((row) => {
              const vA = mA?.[row.key];
              const vB = mB?.[row.key];
              const w  = winner(row, vA, vB);
              return (
                <tr key={row.key} style={{ borderBottom: "1px solid rgba(30,45,70,0.12)" }}>
                  <td style={{ padding: "6px 10px", color: "#4a6080", fontSize: 9,
                    letterSpacing: 0.5, textTransform: "uppercase" }}>{row.label}</td>
                  <td style={{ padding: "6px 10px", textAlign: "right", fontWeight: w === "A" ? 800 : 500,
                    color: cellColor("A", w), fontFamily: FONTS.mono }}>
                    {vA != null ? row.fmt(vA) : "—"}
                    {w === "A" && <span style={{ marginLeft: 5, fontSize: 8 }}>✓</span>}
                  </td>
                  <td style={{ padding: "6px 10px", textAlign: "right", fontWeight: w === "B" ? 800 : 500,
                    color: cellColor("B", w), fontFamily: FONTS.mono }}>
                    {vB != null ? row.fmt(vB) : "—"}
                    {w === "B" && <span style={{ marginLeft: 5, fontSize: 8 }}>✓</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Stored runs list ──
function SavedRunsPanel({ onLoad }) {
  const [runs, setRuns] = useState(() => listStoredRuns());
  const [selected, setSelected] = useState([]);   // up to 2 keys
  const [compareData, setCompareData] = useState(null);

  const handleDelete = (key) => {
    deleteStoredRun(key);
    setRuns(listStoredRuns());
    setSelected((s) => s.filter((k) => k !== key));
    setCompareData(null);
  };

  const handleSelect = (key) => {
    setSelected((prev) => {
      if (prev.includes(key)) return prev.filter((k) => k !== key);
      if (prev.length >= 2)   return [prev[1], key]; // replace oldest selection
      return [...prev, key];
    });
    setCompareData(null);
  };

  const handleCompare = () => {
    if (selected.length !== 2) return;
    const a = loadStoredRun(selected[0]);
    const b = loadStoredRun(selected[1]);
    if (a && b) setCompareData({ a, b });
  };

  if (runs.length === 0) {
    return (
      <div style={{ color: "#3d5575", fontSize: 10, padding: "12px 0", textAlign: "center" }}>
        No saved runs yet. Run a backtest and it will appear here.
      </div>
    );
  }

  return (
    <div>
      {/* Action bar */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 10 }}>
        <span style={{ fontSize: 9, color: "#3d5575" }}>
          {selected.length === 0 && "Select up to 2 runs to compare"}
          {selected.length === 1 && "Select 1 more run to compare"}
          {selected.length === 2 && "Ready to compare"}
        </span>
        {selected.length === 2 && (
          <button onClick={handleCompare} style={primaryBtnStyle}>Compare Selected</button>
        )}
        {selected.length > 0 && (
          <button onClick={() => { setSelected([]); setCompareData(null); }} style={btnStyle}>Clear</button>
        )}
      </div>

      {/* Run list */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {runs.slice().reverse().map((r) => {
          const isSelected = selected.includes(r.key);
          const selIdx = selected.indexOf(r.key);
          return (
            <div key={r.key} style={{
              display: "flex", alignItems: "center", gap: 10,
              background: isSelected ? "rgba(16,185,129,0.07)" : "rgba(14,22,38,0.5)", borderRadius: 8,
              border: `1px solid ${isSelected ? "rgba(16,185,129,0.3)" : "rgba(40,55,80,0.3)"}`,
              padding: "8px 12px", cursor: "pointer",
            }}
              onClick={() => handleSelect(r.key)}
            >
              {/* Selection badge */}
              <div style={{
                width: 20, height: 20, borderRadius: "50%", flexShrink: 0,
                display: "flex", alignItems: "center", justifyContent: "center",
                background: isSelected ? "rgba(16,185,129,0.2)" : "rgba(30,45,70,0.4)",
                border: `1px solid ${isSelected ? "#10b981" : "rgba(40,55,80,0.4)"}`,
                fontSize: 9, fontWeight: 800, color: isSelected ? "#10b981" : "#3d5575",
              }}>
                {isSelected ? selIdx + 1 : ""}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700, fontSize: 10, color: "#c8d6e5" }}>
                  {r.label || (r.config?.start + " → " + r.config?.end)}
                </div>
                <div style={{ fontSize: 9, color: "#3d5575", marginTop: 2 }}>
                  {new Date(r.savedAt).toLocaleString()}
                  {r.metrics && ` · CAGR ${pct(r.metrics.cagr)} · Sharpe ${round2(r.metrics.sharpe)} · MaxDD ${pct(r.metrics.maxDrawdownPct)}`}
                  {r.metrics?.alpha != null && ` · Alpha ${pct(r.metrics.alpha)} · Beta ${round2(r.metrics.beta)}`}
                </div>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); onLoad(loadStoredRun(r.key)); }}
                style={btnStyle}
              >Load</button>
              <button
                onClick={(e) => { e.stopPropagation(); handleDelete(r.key); }}
                style={{ ...btnStyle, color: "#ef4444", borderColor: "rgba(239,68,68,0.2)" }}
              >
                <Trash2 size={10} />
              </button>
            </div>
          );
        })}
      </div>

      {/* Compare result */}
      {compareData && <ComparePanel runA={compareData.a} runB={compareData.b} />}
    </div>
  );
}

// ── Shared button style ──
const btnStyle = {
  padding: "6px 12px", borderRadius: 7, border: "1px solid rgba(40,55,80,0.4)",
  background: "rgba(14,22,38,0.6)", color: "#4a6080", cursor: "pointer",
  fontFamily: FONTS.mono, fontSize: 10, fontWeight: 600,
};

const primaryBtnStyle = {
  ...btnStyle,
  background: "rgba(16,185,129,0.15)", color: "#10b981",
  border: "1px solid rgba(16,185,129,0.3)",
};

// ── Section toggle ──
function Section({ title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ marginBottom: 12 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{ ...btnStyle, width: "100%", textAlign: "left", display: "flex",
          justifyContent: "space-between", alignItems: "center", marginBottom: open ? 8 : 0 }}
      >
        <span>{title}</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && children}
    </div>
  );
}

// ── Results panel ──
function ResultsPanel({ result }) {
  const { metrics: m, equityCurve, trades, walkForward, config, layerStats } = result;
  if (!m) return <div style={{ color: "#ef4444", fontSize: 10 }}>No results to display.</div>;

  const exitReasonData = Object.entries(m.exitReasons || {}).map(([reason, count]) => ({ reason, count }));
  const holdData = useMemo(() => {
    const bins = { "1-5d": 0, "6-10d": 0, "11-20d": 0, "21-40d": 0, "40+d": 0 };
    trades.filter((t) => t.holdDays != null).forEach((t) => {
      if (t.holdDays <= 5)       bins["1-5d"]++;
      else if (t.holdDays <= 10) bins["6-10d"]++;
      else if (t.holdDays <= 20) bins["11-20d"]++;
      else if (t.holdDays <= 40) bins["21-40d"]++;
      else                       bins["40+d"]++;
    });
    return Object.entries(bins).map(([label, count]) => ({ label, count }));
  }, [trades]);

  return (
    <div>
      {/* Top metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 12 }}>
        <MetricCard label="CAGR" value={pct(m.cagr)} color={m.cagr >= 0 ? "#10b981" : "#ef4444"}
          sub={`${money(m.finalValue)} final`} />
        <MetricCard label="Sharpe" value={round2(m.sharpe)}
          color={m.sharpe >= 2 ? "#10b981" : m.sharpe >= 1 ? "#f59e0b" : "#ef4444"}
          sub={`Sortino ${round2(m.sortino)}`} />
        <MetricCard label="Max Drawdown" value={pct(m.maxDrawdownPct)} color="#ef4444"
          sub={`Calmar ${round2(m.calmar)}`} />
        <MetricCard label="Win Rate" value={pct(m.winRate)}
          color={m.winRate >= 0.55 ? "#10b981" : m.winRate >= 0.45 ? "#f59e0b" : "#ef4444"}
          sub={`${m.totalTrades} trades`} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 12 }}>
        <MetricCard label="Net P&L" value={money(m.netPnL)} color={m.netPnL >= 0 ? "#10b981" : "#ef4444"} />
        <MetricCard label="Profit Factor" value={isFinite(m.profitFactor) ? round2(m.profitFactor) : "∞"}
          color={m.profitFactor >= 1.5 ? "#10b981" : m.profitFactor >= 1 ? "#f59e0b" : "#ef4444"} />
        <MetricCard label="Avg Win" value={pct(m.avgWinPct)} color="#10b981"
          sub={`Avg Loss ${pct(m.avgLossPct)}`} />
        <MetricCard label="Avg Hold" value={`${m.avgHoldDays?.toFixed(1)}d`}
          sub={`${m.tradingDays} trading days`} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8, marginBottom: 12 }}>
        <MetricCard label="Beta vs SPY"
          value={m.beta != null ? round2(m.beta) : "—"}
          color={m.beta != null ? (m.beta < 0.8 ? "#10b981" : m.beta < 1.3 ? "#f59e0b" : "#ef4444") : "#4a6080"}
          sub={m.beta != null ? (m.beta < 1 ? "Less volatile than market" : "More volatile than market") : "Needs SPY data"} />
        <MetricCard label="Alpha vs SPY (ann.)"
          value={m.alpha != null ? pct(m.alpha) : "—"}
          color={m.alpha != null ? (m.alpha > 0 ? "#10b981" : "#ef4444") : "#4a6080"}
          sub={m.alpha != null ? (m.alpha > 0 ? "Outperforms SPY on risk-adj." : "Underperforms SPY") : "Needs SPY data"} />
        <MetricCard label="Gross Profit" value={money(m.grossProfit)} color="#10b981" />
        <MetricCard label="Gross Loss" value={money(m.grossLoss)} color="#ef4444" />
      </div>

      {/* Equity curve */}
      <Section title="Portfolio Equity Curve">
        <div style={{ ...card, padding: 12, marginBottom: 4 }}>
          <EquityChart curve={equityCurve} />
        </div>
      </Section>

      {/* Monthly heatmap */}
      <Section title="Monthly Returns Heatmap">
        <div style={{ ...card, padding: 12, overflowX: "auto" }}>
          <MonthlyHeatmap monthlyReturns={m.monthlyReturns} />
        </div>
      </Section>

      {/* Exit reason + hold duration */}
      <Section title="Trade Analysis">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
          <div style={{ ...card, padding: 12 }}>
            <div style={{ fontSize: 9, color: "#3d5575", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>
              Exit Reasons
            </div>
            <ResponsiveContainer width="100%" height={130}>
              <BarChart data={exitReasonData} margin={{ top: 0, right: 0, bottom: 0, left: -10 }}>
                <XAxis dataKey="reason" tick={{ fontSize: 8, fill: "#3d5575" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 8, fill: "#3d5575" }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "#0c1222", border: "1px solid #1e3050", fontSize: 10, fontFamily: FONTS.mono }} />
                <Bar dataKey="count" fill="#6366f1" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ ...card, padding: 12 }}>
            <div style={{ fontSize: 9, color: "#3d5575", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>
              Hold Duration Distribution
            </div>
            <ResponsiveContainer width="100%" height={130}>
              <BarChart data={holdData} margin={{ top: 0, right: 0, bottom: 0, left: -10 }}>
                <XAxis dataKey="label" tick={{ fontSize: 8, fill: "#3d5575" }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 8, fill: "#3d5575" }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "#0c1222", border: "1px solid #1e3050", fontSize: 10, fontFamily: FONTS.mono }} />
                <Bar dataKey="count" fill="#06b6d4" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Section>

      {/* Walk-forward */}
      {walkForward && (
        <Section title={`Walk-Forward Analysis (${walkForward.windows?.length} windows)`} defaultOpen={true}>
          <div style={{ ...card, padding: 12 }}>
            <WalkForwardPanel wf={walkForward} />
          </div>
        </Section>
      )}

      {/* Layer Breakdown */}
      {layerStats && (layerStats.consensus || layerStats.trend) && (
        <Section title="Layer Breakdown: Consensus vs Trend">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              { key: "consensus", label: "Consensus Layer", color: "#10b981" },
              { key: "trend",     label: "Trend Layer",     color: "#06b6d4" },
            ].map(({ key, label, color }) => {
              const s = layerStats[key];
              if (!s) return (
                <div key={key} style={{ ...card, padding: 12 }}>
                  <div style={{ fontSize: 9, color: "#3d5575", letterSpacing: 1, textTransform: "uppercase", marginBottom: 8 }}>{label}</div>
                  <div style={{ fontSize: 10, color: "#3d5575" }}>No trades</div>
                </div>
              );
              const pf = s.grossLoss > 0 ? s.grossProfit / s.grossLoss : s.grossProfit > 0 ? Infinity : 0;
              return (
                <div key={key} style={{ ...card, padding: 12 }}>
                  <div style={{ fontSize: 9, color: color, letterSpacing: 1, textTransform: "uppercase", marginBottom: 10, fontWeight: 700 }}>{label}</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <MetricCard label="Trades"        value={String(s.count)} color={color} />
                    <MetricCard label="Win Rate"      value={pct(s.winRate)}  color={s.winRate >= 0.5 ? "#10b981" : "#ef4444"} />
                    <MetricCard label="Net P&L"       value={money(s.netPnl)} color={s.netPnl >= 0 ? "#10b981" : "#ef4444"} />
                    <MetricCard label="Profit Factor" value={isFinite(pf) ? round2(pf) : "∞"} color={pf >= 1.5 ? "#10b981" : pf >= 1 ? "#f59e0b" : "#ef4444"} />
                    <MetricCard label="Avg P&L %"     value={pct(s.avgPnlPct)} color={s.avgPnlPct >= 0 ? "#10b981" : "#ef4444"} />
                    <MetricCard label="Avg Hold"      value={`${s.avgHoldDays?.toFixed(1)}d`} />
                  </div>
                </div>
              );
            })}
          </div>
        </Section>
      )}

      {/* Trade log */}
      <Section title={`Trade Log (${trades.filter((t) => t.exitDate).length} closed trades)`} defaultOpen={false}>
        <div style={{ ...card, padding: 12 }}>
          <TradeTable trades={trades} />
        </div>
      </Section>
    </div>
  );
}

// ── Main component ──
export function BacktestTab() {
  const today = new Date().toISOString().split("T")[0];
  const twoYearsAgo = new Date(Date.now() - 2 * 365 * 24 * 60 * 60 * 1000).toISOString().split("T")[0];

  const [startDate, setStartDate] = useState(twoYearsAgo);
  const [endDate, setEndDate] = useState(today);
  const [useWalkForward, setUseWalkForward] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("results");

  const handleRun = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    setProgress(0);
    setProgressMsg("Starting...");

    try {
      const res = await runBacktest({
        start: startDate,
        end: endDate,
        walkForward: useWalkForward,
        onProgress: (pct, msg) => { setProgress(pct); setProgressMsg(msg); },
      });
      setResult(res);
      // Auto-save
      saveRunToStorage(res);
      setActiveTab("results");
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
      setProgress(0);
      setProgressMsg("");
    }
  }, [startDate, endDate, useWalkForward]);

  const handleLoad = useCallback((run) => {
    if (run) { setResult(run); setActiveTab("results"); }
  }, []);

  return (
    <div style={{ ...card, marginTop: 10 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: "#4a6080", letterSpacing: 1.5, textTransform: "uppercase" }}>
          <FlaskConical size={11} style={{ marginRight: 4 }} /> Backtester
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {["results","saved"].map((t) => (
            <button key={t} onClick={() => setActiveTab(t)} style={{
              ...btnStyle,
              color: activeTab === t ? "#10b981" : "#4a6080",
              background: activeTab === t ? "rgba(16,185,129,0.1)" : "rgba(14,22,38,0.4)",
              border: `1px solid ${activeTab === t ? "rgba(16,185,129,0.25)" : "rgba(40,55,80,0.3)"}`,
            }}>
              {t === "results" ? "Results" : "Saved Runs"}
            </button>
          ))}
        </div>
      </div>

      {/* Config bar */}
      <div style={{
        display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap",
        background: "rgba(10,16,28,0.5)", borderRadius: 10,
        border: "1px solid rgba(30,45,70,0.4)", padding: "10px 14px",
        marginBottom: 14,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <label style={{ fontSize: 9, color: "#3d5575", letterSpacing: 0.8, textTransform: "uppercase" }}>From</label>
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)}
            style={{ ...inputStyle }} />
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <label style={{ fontSize: 9, color: "#3d5575", letterSpacing: 0.8, textTransform: "uppercase" }}>To</label>
          <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)}
            style={{ ...inputStyle }} />
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked={useWalkForward} onChange={(e) => setUseWalkForward(e.target.checked)}
            style={{ accentColor: "#10b981" }} />
          <span style={{ fontSize: 9, color: "#4a6080", letterSpacing: 0.8, textTransform: "uppercase" }}>
            Walk-Forward
          </span>
        </label>
        <button
          onClick={handleRun}
          disabled={running}
          style={{
            ...primaryBtnStyle,
            opacity: running ? 0.6 : 1,
            cursor: running ? "not-allowed" : "pointer",
            display: "flex", alignItems: "center", gap: 5,
          }}
        >
          <Play size={10} />
          {running ? "Running..." : "Run Backtest"}
        </button>
      </div>

      {/* Progress bar */}
      {running && (
        <div style={{ marginBottom: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 9, color: "#4a6080" }}>{progressMsg}</span>
            <span style={{ fontSize: 9, color: "#4a6080" }}>{progress}%</span>
          </div>
          <div style={{ height: 3, background: "rgba(30,45,70,0.4)", borderRadius: 2 }}>
            <div style={{
              height: "100%", borderRadius: 2,
              width: `${progress}%`, background: "#10b981",
              transition: "width 0.3s ease",
            }} />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: "10px 14px", borderRadius: 8, marginBottom: 12,
          background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)",
          color: "#ef4444", fontSize: 10,
        }}>
          {error}
        </div>
      )}

      {/* Content */}
      {activeTab === "results" && (
        result
          ? <ResultsPanel result={result} />
          : !running && (
            <div style={{ textAlign: "center", padding: "40px 0", color: "#3d5575", fontSize: 10 }}>
              Configure a date range above and click Run Backtest to start.
            </div>
          )
      )}
      {activeTab === "saved" && <SavedRunsPanel onLoad={handleLoad} />}
    </div>
  );
}

const inputStyle = {
  padding: "5px 10px", borderRadius: 7, border: "1px solid rgba(40,55,80,0.4)",
  background: "rgba(10,16,28,0.8)", color: "#c8d6e5",
  fontFamily: FONTS.mono, fontSize: 10,
};
