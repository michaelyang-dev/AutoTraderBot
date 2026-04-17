// ══════════════════════════════════════════
//  SHARED STYLES & THEME
// ══════════════════════════════════════════

export const FONTS = {
  mono: "'IBM Plex Mono', 'Fira Code', 'SF Mono', monospace",
  sans: "'DM Sans', 'Satoshi', system-ui, sans-serif",
};

export const card = {
  borderRadius: 14,
  background: "rgba(12,18,30,0.85)",
  border: "1px solid rgba(40,55,80,0.4)",
  padding: 16,
  backdropFilter: "blur(10px)",
};

export const globalCSS = `
  *, *::before, *::after { box-sizing: border-box; }
  html, body, #root { margin: 0; padding: 0; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 4px; }
`;

export const signalColor = (s) => {
  if (!s) return "#2a3f5a";
  if (s === "BUY" || s === "BULLISH") return "#10b981";
  if (s === "SELL" || s === "BEARISH") return "#ef4444";
  return "#6a8099";
};

export const consensusColor = (c) => {
  if (!c) return "#2a3f5a";
  if (c.includes("BUY")) return "#10b981";
  if (c.includes("SELL")) return "#ef4444";
  return "#6a8099";
};
