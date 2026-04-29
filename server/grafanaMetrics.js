// ══════════════════════════════════════════════════════════════════════
//  Grafana Metrics API — JSON endpoints for Grafana dashboards
//
//  Exposes trade journal, portfolio, signal, and system health data
//  via a clean REST API that Grafana's Infinity datasource can query.
//
//  All endpoints return arrays of objects (Grafana table format) or
//  single objects (Grafana JSON format).
//
//  Mount with: require("./grafanaMetrics")(app)
// ══════════════════════════════════════════════════════════════════════

const path = require("path");
const Database = require("better-sqlite3");
const http = require("http");

const DB_PATH = path.join(__dirname, "..", "data", "journal.db");
const ML_SERVER = "http://localhost:5001";

// ── Helpers ─────────────────────────────────────────────────────────

function getDb() {
  return new Database(DB_PATH, { readonly: true });
}

function fetchLocal(url, timeoutMs = 3000) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, { timeout: timeoutMs }, (res) => {
      let raw = "";
      res.on("data", (c) => (raw += c));
      res.on("end", () => {
        try { resolve(JSON.parse(raw)); }
        catch { resolve(null); }
      });
    });
    req.on("error", () => resolve(null));
    req.on("timeout", () => { req.destroy(); resolve(null); });
  });
}

// ══════════════════════════════════════════════════════════════════════
//  MOUNT ROUTES
// ══════════════════════════════════════════════════════════════════════

module.exports = function mountGrafanaRoutes(app) {

  // ── Portfolio equity curve (daily_snapshots) ──────────────────────
  app.get("/api/grafana/equity", (req, res) => {
    try {
      const db = getDb();
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT date, portfolio_value, cash, equity, positions_count,
               day_pnl, day_pnl_pct, regime, spy_close, spy_day_pct,
               peak_value, drawdown_pct
        FROM daily_snapshots
        WHERE date >= ?
        ORDER BY date ASC
      `).all(since);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Current portfolio summary ────────────────────────────────────
  app.get("/api/grafana/portfolio", async (req, res) => {
    try {
      const db = getDb();

      // Latest snapshot
      const latest = db.prepare(`
        SELECT * FROM daily_snapshots ORDER BY date DESC LIMIT 1
      `).get();

      // All-time stats from trades
      const tradeStats = db.prepare(`
        SELECT
          COUNT(*) as total_trades,
          SUM(CASE WHEN side='sell' AND realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
          SUM(CASE WHEN side='sell' AND realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
          SUM(CASE WHEN side='sell' THEN realized_pnl ELSE 0 END) as total_realized_pnl,
          AVG(CASE WHEN side='sell' THEN realized_pnl_pct END) as avg_return_pct,
          AVG(CASE WHEN side='sell' THEN hold_days END) as avg_hold_days,
          AVG(slippage_bps) as avg_slippage_bps
        FROM trades WHERE status='filled'
      `).get();

      // Open positions: count + total value + unrealized P&L
      const positions = db.prepare(`
        SELECT COUNT(*) as count,
               SUM(current_value) as total_current_value,
               SUM(cost_basis) as total_cost_basis,
               SUM(unrealized_pnl) as total_unrealized_pnl
        FROM positions
      `).get();

      // Peak and drawdown
      const peak = db.prepare(`
        SELECT MAX(portfolio_value) as all_time_high FROM daily_snapshots
      `).get();

      // Today's realized P&L from sells (use ET timezone, not UTC)
      const today = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
      const todayPnl = db.prepare(`
        SELECT COALESCE(SUM(realized_pnl), 0) as day_realized_pnl
        FROM trades
        WHERE status='filled' AND side='sell' AND DATE(submitted_at) = ?
      `).get(today);

      db.close();

      // Fetch live account data from Alpaca via the existing /api/account handler
      const account = await fetchLocal("http://localhost:3001/api/account", 5000);

      const positionsValue = positions?.total_current_value || 0;
      const cash = account?.cash || latest?.cash || 0;
      const portfolio_value = account?.equity || latest?.portfolio_value || positionsValue;
      const ath = peak?.all_time_high || portfolio_value;
      const drawdown_from_ath = ath > 0 ? (portfolio_value / ath - 1) : 0;
      const sellTrades = (tradeStats?.wins || 0) + (tradeStats?.losses || 0);

      // Day P&L: prefer snapshot, fall back to today's realized + unrealized change
      const day_pnl = latest?.day_pnl ?? (todayPnl?.day_realized_pnl || 0);

      // Wrap in array — Infinity plugin handles string fields better from arrays
      res.json([{
        portfolio_value,
        cash,
        day_pnl,
        day_pnl_pct: latest?.day_pnl_pct || 0,
        regime: latest?.regime || "UNKNOWN",
        spy_day_pct: latest?.spy_day_pct || 0,
        positions_count: positions?.count || 0,
        total_unrealized_pnl: positions?.total_unrealized_pnl || 0,
        positions_value: positionsValue,
        all_time_high: ath,
        drawdown_from_ath,
        total_trades: tradeStats?.total_trades || 0,
        wins: tradeStats?.wins || 0,
        losses: tradeStats?.losses || 0,
        win_rate: sellTrades > 0 ? (tradeStats.wins / sellTrades) : 0,
        total_realized_pnl: tradeStats?.total_realized_pnl || 0,
        avg_return_pct: tradeStats?.avg_return_pct || 0,
        avg_hold_days: tradeStats?.avg_hold_days || 0,
        avg_slippage_bps: tradeStats?.avg_slippage_bps || 0,
        last_snapshot_date: latest?.date || null,
      }]);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Open positions ───────────────────────────────────────────────
  app.get("/api/grafana/positions", (req, res) => {
    try {
      const db = getDb();
      const rows = db.prepare(`
        SELECT symbol, qty, strategy, avg_cost, current_price,
               current_value, unrealized_pnl, unrealized_pnl_pct,
               days_held
        FROM positions
        ORDER BY unrealized_pnl DESC
      `).all();
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Trade history ────────────────────────────────────────────────
  app.get("/api/grafana/trades", (req, res) => {
    try {
      const db = getDb();
      const limit = parseInt(req.query.limit) || 200;
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT submitted_at, symbol, side, qty, strategy,
               fill_price, regime, signal_prob,
               realized_pnl, realized_pnl_pct,
               hold_days, exit_reason, slippage_bps
        FROM trades
        WHERE status='filled' AND submitted_at >= ?
        ORDER BY submitted_at DESC
        LIMIT ?
      `).all(since, limit);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Strategy performance breakdown ───────────────────────────────
  app.get("/api/grafana/strategy-performance", (req, res) => {
    try {
      const db = getDb();
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT
          strategy,
          COUNT(*) as total_trades,
          SUM(CASE WHEN side='sell' AND realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
          SUM(CASE WHEN side='sell' AND realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
          ROUND(100.0 * SUM(CASE WHEN side='sell' AND realized_pnl > 0 THEN 1 ELSE 0 END)
            / MAX(SUM(CASE WHEN side='sell' THEN 1 ELSE 0 END), 1), 1) as win_rate,
          ROUND(SUM(CASE WHEN side='sell' THEN realized_pnl ELSE 0 END), 2) as total_pnl,
          ROUND(AVG(CASE WHEN side='sell' THEN realized_pnl_pct END) * 100, 2) as avg_return_pct,
          ROUND(AVG(CASE WHEN side='sell' THEN hold_days END), 1) as avg_hold_days,
          ROUND(AVG(slippage_bps), 1) as avg_slippage_bps
        FROM trades
        WHERE status='filled' AND submitted_at >= ?
        GROUP BY strategy
        ORDER BY total_pnl DESC
      `).all(since);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Daily P&L time series ────────────────────────────────────────
  app.get("/api/grafana/daily-pnl", (req, res) => {
    try {
      const db = getDb();
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT
          DATE(fill_time) as date,
          COUNT(*) as trades,
          SUM(CASE WHEN side='sell' THEN realized_pnl ELSE 0 END) as realized_pnl,
          SUM(CASE WHEN side='sell' AND realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
          SUM(CASE WHEN side='sell' AND realized_pnl <= 0 THEN 1 ELSE 0 END) as losses
        FROM trades
        WHERE status='filled' AND fill_time IS NOT NULL AND fill_time >= ?
        GROUP BY DATE(fill_time)
        ORDER BY date ASC
      `).all(since);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Trade distribution by symbol ─────────────────────────────────
  app.get("/api/grafana/symbol-distribution", (req, res) => {
    try {
      const db = getDb();
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT
          symbol,
          COUNT(*) as trade_count,
          SUM(CASE WHEN side='sell' THEN realized_pnl ELSE 0 END) as total_pnl,
          ROUND(AVG(CASE WHEN side='sell' THEN realized_pnl_pct END) * 100, 2) as avg_return_pct
        FROM trades
        WHERE status='filled' AND submitted_at >= ?
        GROUP BY symbol
        ORDER BY total_pnl DESC
        LIMIT 30
      `).all(since);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Exit reason breakdown ────────────────────────────────────────
  app.get("/api/grafana/exit-reasons", (req, res) => {
    try {
      const db = getDb();
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT
          COALESCE(exit_reason, 'unknown') as exit_reason,
          COUNT(*) as count,
          ROUND(SUM(realized_pnl), 2) as total_pnl,
          ROUND(AVG(realized_pnl_pct) * 100, 2) as avg_return_pct,
          ROUND(AVG(hold_days), 1) as avg_hold_days
        FROM trades
        WHERE status='filled' AND side='sell' AND submitted_at >= ?
        GROUP BY exit_reason
        ORDER BY count DESC
      `).all(since);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── System events ────────────────────────────────────────────────
  app.get("/api/grafana/events", (req, res) => {
    try {
      const db = getDb();
      const limit = parseInt(req.query.limit) || 100;
      const severity = req.query.severity;
      const type = req.query.type;

      let sql = `SELECT id, event_type, severity, message,
                        portfolio_value_at_event, created_at
                 FROM events WHERE 1=1`;
      const params = [];

      if (severity) { sql += ` AND severity = ?`; params.push(severity); }
      if (type) { sql += ` AND event_type = ?`; params.push(type); }
      sql += ` ORDER BY created_at DESC LIMIT ?`;
      params.push(limit);

      const rows = db.prepare(sql).all(...params);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Circuit breaker state ────────────────────────────────────────
  app.get("/api/grafana/circuit-breaker", (req, res) => {
    try {
      const fs = require("fs");
      const cbPath = path.join(__dirname, "..", "ml_service", "data", "circuit_breaker_state.json");
      if (fs.existsSync(cbPath)) {
        const data = JSON.parse(fs.readFileSync(cbPath, "utf8"));
        res.json(data);
      } else {
        res.json({ active: false, daily_loss_pct: 0, weekly_loss_pct: 0, peak_drawdown_pct: 0 });
      }
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Signal server health (v9.6 factor strategy) ─────────────────
  app.get("/api/grafana/ml-health", async (req, res) => {
    try {
      const health = await fetchLocal(`${ML_SERVER}/health`);
      if (health) {
        res.json({
          status: health.status || "unknown",
          model_loaded: health.model_loaded || false,
          model_version: health.model_version || "unknown",
          feature_count: health.feature_count || 0,
          cached_signals: health.cached_signals || 0,
          last_update: health.last_update || null,
          is_stale: health.is_stale || true,
          market_hours: health.market_hours || false,
          regime: health.regime || "UNKNOWN",
          ml_mode: health.ml_mode || "unknown",
          breadth: health.breadth || 0,
        });
      } else {
        res.json({
          status: "offline",
          model_loaded: false,
          cached_signals: 0,
          is_stale: true,
        });
      }
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Signal quality over time ─────────────────────────────────────
  app.get("/api/grafana/signal-quality", (req, res) => {
    try {
      const db = getDb();
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT
          DATE(signal_time) as date,
          COUNT(*) as total_signals,
          SUM(acted_on) as acted_on_count,
          ROUND(AVG(probability), 4) as avg_probability,
          SUM(CASE WHEN is_top_5=1 THEN 1 ELSE 0 END) as top_5_count,
          regime
        FROM signals
        WHERE signal_time >= ?
        GROUP BY DATE(signal_time), regime
        ORDER BY date DESC
        LIMIT 500
      `).all(since);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Cumulative P&L series ────────────────────────────────────────
  app.get("/api/grafana/cumulative-pnl", (req, res) => {
    try {
      const db = getDb();
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT date, portfolio_value, spy_close
        FROM daily_snapshots
        WHERE date >= ?
        ORDER BY date ASC
      `).all(since);
      db.close();

      if (rows.length === 0) return res.json([]);

      const initial = rows[0].portfolio_value;
      const spyInitial = rows[0].spy_close;

      const result = rows.map((r) => ({
        date: r.date,
        portfolio_return_pct: ((r.portfolio_value / initial) - 1) * 100,
        spy_return_pct: spyInitial ? ((r.spy_close / spyInitial) - 1) * 100 : 0,
        portfolio_value: r.portfolio_value,
      }));

      res.json(result);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Regime timeline ──────────────────────────────────────────────
  app.get("/api/grafana/regime-timeline", (req, res) => {
    try {
      const db = getDb();
      const rows = db.prepare(`
        SELECT date, regime, portfolio_value, spy_close
        FROM daily_snapshots
        ORDER BY date ASC
      `).all();
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Weekly performance rollup ────────────────────────────────────
  app.get("/api/grafana/weekly-rollup", (req, res) => {
    try {
      const db = getDb();
      const since = req.query.since || "2020-01-01";
      const rows = db.prepare(`
        SELECT
          strftime('%Y-W%W', date) as week,
          MIN(date) as week_start,
          MAX(date) as week_end,
          MIN(portfolio_value) as min_value,
          MAX(portfolio_value) as max_value,
          SUM(day_pnl) as week_pnl,
          AVG(positions_count) as avg_positions,
          GROUP_CONCAT(DISTINCT regime) as regimes
        FROM daily_snapshots
        WHERE date >= ?
        GROUP BY strftime('%Y-W%W', date)
        ORDER BY week DESC
        LIMIT 52
      `).all(since);
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Slippage analysis ────────────────────────────────────────────
  app.get("/api/grafana/slippage", (req, res) => {
    try {
      const db = getDb();
      const rows = db.prepare(`
        SELECT
          DATE(fill_time) as date,
          COUNT(*) as fills,
          ROUND(AVG(slippage_bps), 2) as avg_slippage_bps,
          ROUND(MAX(slippage_bps), 2) as max_slippage_bps,
          ROUND(SUM(ABS(slippage_bps * fill_price * filled_qty / 10000)), 2) as slippage_cost
        FROM trades
        WHERE status='filled' AND fill_time IS NOT NULL AND slippage_bps IS NOT NULL
        GROUP BY DATE(fill_time)
        ORDER BY date DESC
        LIMIT 200
      `).all();
      db.close();
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ══════════════════════════════════════════════════════════════════
  //  BACKTESTING ENDPOINTS
  // ══════════════════════════════════════════════════════════════════

  const fs = require("fs");
  const WF_DIR = path.join(__dirname, "..", "ml_service", "data", "walkforward");

  // ── List available backtest configs ──────────────────────────────
  app.get("/api/grafana/backtest/configs", (req, res) => {
    try {
      const files = fs.readdirSync(WF_DIR).filter(f => f.endsWith("_results.json"));
      const configs = files.map(f => {
        const name = f.replace("_results.json", "");
        return { config: name, file: f };
      });
      res.json(configs);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Yearly backtest results for a config ────────────────────────
  app.get("/api/grafana/backtest/yearly", (req, res) => {
    try {
      const config = req.query.config || "config_live_production";
      const filePath = path.join(WF_DIR, `${config}_results.json`);
      if (!fs.existsSync(filePath)) {
        return res.json([]);
      }
      const data = JSON.parse(fs.readFileSync(filePath, "utf8"));
      const rows = data.map(r => ({
        year: r.year,
        cagr_pct: Math.round(r.cagr * 10000) / 100,
        sharpe: Math.round(r.sharpe * 100) / 100,
        sortino: Math.round(r.sortino * 100) / 100,
        max_dd_pct: Math.round(r.max_dd * 10000) / 100,
        win_rate_pct: Math.round(r.win_rate * 10000) / 100,
        profit_factor: Math.round(r.profit_factor * 100) / 100,
        avg_trade_ret_pct: Math.round(r.avg_trade_ret * 10000) / 100,
        trades: r.n_trades,
        alpha_pct: Math.round(r.alpha * 10000) / 100,
        beta: Math.round(r.beta * 100) / 100,
        final_value: Math.round(r.final_value),
      }));
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Summary stats across all years ──────────────────────────────
  app.get("/api/grafana/backtest/summary", (req, res) => {
    try {
      const config = req.query.config || "config_live_production";
      const filePath = path.join(WF_DIR, `${config}_results.json`);
      if (!fs.existsSync(filePath)) {
        return res.json([]);
      }
      const data = JSON.parse(fs.readFileSync(filePath, "utf8"));
      const n = data.length;
      const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

      const cagrs = data.map(r => r.cagr);
      const sharpes = data.map(r => r.sharpe);
      const maxDDs = data.map(r => r.max_dd);
      const winRates = data.map(r => r.win_rate);
      const alphas = data.map(r => r.alpha);
      const trades = data.map(r => r.n_trades);

      // Compound growth: chain yearly returns
      const compounded = data.reduce((acc, r) => acc * (1 + r.cagr), 1);
      const totalReturn = (compounded - 1);

      const profitableYears = data.filter(r => r.cagr > 0).length;

      res.json([{
        years: n,
        avg_cagr_pct: Math.round(avg(cagrs) * 10000) / 100,
        avg_sharpe: Math.round(avg(sharpes) * 100) / 100,
        avg_win_rate_pct: Math.round(avg(winRates) * 10000) / 100,
        worst_dd_pct: Math.round(Math.min(...maxDDs) * 10000) / 100,
        avg_alpha_pct: Math.round(avg(alphas) * 10000) / 100,
        total_trades: trades.reduce((a, b) => a + b, 0),
        total_return_pct: Math.round(totalReturn * 10000) / 100,
        profitable_years: profitableYears,
        win_year_pct: Math.round((profitableYears / n) * 10000) / 100,
        best_year_cagr_pct: Math.round(Math.max(...cagrs) * 10000) / 100,
        worst_year_cagr_pct: Math.round(Math.min(...cagrs) * 10000) / 100,
      }]);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Compare two configs side by side ────────────────────────────
  app.get("/api/grafana/backtest/compare", (req, res) => {
    try {
      const configs = (req.query.configs || "config_live_production,config_backtester_default").split(",");
      const results = [];

      for (const config of configs) {
        const filePath = path.join(WF_DIR, `${config.trim()}_results.json`);
        if (!fs.existsSync(filePath)) continue;
        const data = JSON.parse(fs.readFileSync(filePath, "utf8"));
        const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
        const compounded = data.reduce((acc, r) => acc * (1 + r.cagr), 1) - 1;

        results.push({
          config: config.trim().replace(/^config_/, "").replace(/^isolated_/, "iso: "),
          years: data.length,
          avg_cagr_pct: Math.round(avg(data.map(r => r.cagr)) * 10000) / 100,
          avg_sharpe: Math.round(avg(data.map(r => r.sharpe)) * 100) / 100,
          total_return_pct: Math.round(compounded * 10000) / 100,
          worst_dd_pct: Math.round(Math.min(...data.map(r => r.max_dd)) * 10000) / 100,
          avg_win_rate_pct: Math.round(avg(data.map(r => r.win_rate)) * 10000) / 100,
          total_trades: data.reduce((a, r) => a + r.n_trades, 0),
          avg_alpha_pct: Math.round(avg(data.map(r => r.alpha)) * 10000) / 100,
        });
      }
      res.json(results);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── CAGR by year as time series (for bar chart) ─────────────────
  app.get("/api/grafana/backtest/cagr-by-year", (req, res) => {
    try {
      const config = req.query.config || "config_live_production";
      const filePath = path.join(WF_DIR, `${config}_results.json`);
      if (!fs.existsSync(filePath)) return res.json([]);
      const data = JSON.parse(fs.readFileSync(filePath, "utf8"));
      const rows = data.map(r => ({
        year: String(r.year),
        cagr_pct: Math.round(r.cagr * 10000) / 100,
      }));
      res.json(rows);
    } catch (err) {
      res.status(500).json({ error: err.message });
    }
  });

  // ── Run a new backtest ────────────────────────────────────────────
  let backtestRunning = false;
  let backtestStatus = { state: "idle", message: "", started_at: null, finished_at: null, config: null };

  app.post("/api/grafana/backtest/run", (req, res) => {
    if (backtestRunning) {
      return res.json({ error: "Backtest already running", status: backtestStatus });
    }

    const { exec } = require("child_process");
    const threshold = req.body.threshold || 0.52;
    const maxPositions = req.body.max_positions || 6;
    const positionPct = req.body.position_pct || 0.12;
    const holdDays = req.body.hold_days || 10;
    const label = req.body.label || `custom_t${threshold}_p${maxPositions}`;

    backtestRunning = true;
    backtestStatus = { state: "running", message: `Running backtest: ${label}`, started_at: new Date().toISOString(), finished_at: null, config: label };

    const pythonBin = path.join(__dirname, "..", "ml_service", "venv", "bin", "python3");
    const scriptPath = path.join(__dirname, "..", "ml_service", "run_quick_backtest.py");
    const scriptDir = path.join(__dirname, "..", "ml_service");

    const cmd = `${pythonBin} ${scriptPath} --threshold ${threshold} --max-positions ${maxPositions} --position-pct ${positionPct} --hold-days ${holdDays} --label ${label}`;

    exec(cmd, { cwd: scriptDir, timeout: 300000 }, (err, stdout, stderr) => {
      backtestRunning = false;
      if (err) {
        backtestStatus = { state: "error", message: stderr || err.message, started_at: backtestStatus.started_at, finished_at: new Date().toISOString(), config: label };
      } else {
        backtestStatus = { state: "done", message: `Backtest ${label} complete`, started_at: backtestStatus.started_at, finished_at: new Date().toISOString(), config: label };
      }
    });

    res.json({ status: "started", label, params: { threshold, maxPositions, positionPct, holdDays } });
  });

  app.get("/api/grafana/backtest/status", (req, res) => {
    res.json(backtestStatus);
  });

  // ── Health check for Grafana itself ──────────────────────────────
  app.get("/api/grafana/health", (req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString() });
  });
};
