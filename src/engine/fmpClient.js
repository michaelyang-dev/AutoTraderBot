// ══════════════════════════════════════════
//  FMP CLIENT — Financial Modeling Prep API
//  Proxied through Express server so the key never reaches the browser
// ══════════════════════════════════════════

const API = "/api";

async function request(path) {
  try {
    const res = await fetch(`${API}${path}`, {
      headers: { "Content-Type": "application/json" },
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  } catch (err) {
    console.error(`FMP Error [${path}]:`, err.message);
    throw err;
  }
}

/**
 * Fetch upcoming earnings dates for the given symbols.
 * Returns { [symbol]: "YYYY-MM-DD" } for symbols that have earnings
 * in the next 7 calendar days. Symbols with no upcoming earnings are omitted.
 *
 * @param {string[]} symbols
 * @returns {Promise<Record<string, string>>}
 */
export const getUpcomingEarnings = (symbols) =>
  request(`/earnings?symbols=${symbols.join(",")}`);

