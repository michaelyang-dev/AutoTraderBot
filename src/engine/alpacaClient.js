// ══════════════════════════════════════════
//  ALPACA CLIENT — Frontend API calls
//  All requests go to our Express server (localhost:3001)
//  which securely proxies to Alpaca's API
// ══════════════════════════════════════════

const API = "/api"; // proxied via package.json "proxy" field

async function request(path, options = {}) {
  try {
    const res = await fetch(`${API}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  } catch (err) {
    console.error(`API Error [${path}]:`, err.message);
    throw err;
  }
}

// ── Account ──
export const getAccount = () => request("/account");

// ── Positions ──
export const getPositions = () => request("/positions");
export const closePosition = (symbol) =>
  request(`/positions/${symbol}`, { method: "DELETE" });
export const closeAllPositions = () =>
  request("/positions", { method: "DELETE" });

// ── Orders ──
export const placeOrder = ({ symbol, qty, side, type = "market", time_in_force = "day", limit_price }) =>
  request("/orders", {
    method: "POST",
    body: JSON.stringify({ symbol, qty, side, type, time_in_force, limit_price }),
  });
export const getOrders = (status = "all", limit = 50) =>
  request(`/orders?status=${status}&limit=${limit}`);
export const cancelAllOrders = () =>
  request("/orders", { method: "DELETE" });

// ── Market Data ──
export const getQuote = (symbol) => request(`/quote/${symbol}`);
export const getSnapshots = (symbols) =>
  request(`/snapshots?symbols=${symbols.join(",")}`);
export const getBars = (symbol, timeframe = "1Day", limit = 60) =>
  request(`/bars/${symbol}?timeframe=${timeframe}&limit=${limit}`);

// ── Clock ──
export const getClock = () => request("/clock");

// ── Health ──
export const checkHealth = () => request("/health");

// ── ML Signal Server (localhost:5001 — direct, not proxied through Express) ──
const ML_API = "http://localhost:5001";

async function mlRequest(path) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${ML_API}${path}`, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    // Return null so callers can cleanly fall back — don't throw
    console.warn(`ML API [${path}]: ${err.message}`);
    return null;
  }
}

export const getMLSignals = () => mlRequest("/signals");
export const getMLHealth  = () => mlRequest("/health");
