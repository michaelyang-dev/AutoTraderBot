#!/usr/bin/env node
// ══════════════════════════════════════════════════════════════════════
//  Journal CLI — Query and report on trade journal data
//
//  Usage:
//    node scripts/journal.js stats [--since DATE]
//    node scripts/journal.js strategy [--since DATE]
//    node scripts/journal.js trades [--last N] [--symbol X] [--strategy Y] [--status Z]
//    node scripts/journal.js position [SYMBOL]
//    node scripts/journal.js day YYYY-MM-DD
//    node scripts/journal.js events [--last N] [--type T] [--severity S]
//    node scripts/journal.js signals [--symbol X] [--since DATE]
// ══════════════════════════════════════════════════════════════════════

const path = require("path");
const journal = require("../db/journal");
const Table = require("cli-table3");
const chalk = require("chalk");
const yargs = require("yargs/yargs");
const { hideBin } = require("yargs/helpers");

// ── Helpers ──────────────────────────────────────────────────────────

function initJournal() {
  const dbPath = path.join(__dirname, "..", "data", "journal.db");
  journal.initDb(dbPath);
}

function pnlColor(val) {
  if (val == null) return chalk.gray("—");
  const n = parseFloat(val);
  if (n > 0) return chalk.green(`+$${n.toFixed(2)}`);
  if (n < 0) return chalk.red(`-$${Math.abs(n).toFixed(2)}`);
  return chalk.white("$0.00");
}

function pctColor(val) {
  if (val == null) return chalk.gray("—");
  const n = parseFloat(val) * 100;
  if (n > 0) return chalk.green(`+${n.toFixed(2)}%`);
  if (n < 0) return chalk.red(`${n.toFixed(2)}%`);
  return chalk.white("0.00%");
}

function statusColor(s) {
  if (s === "filled") return chalk.green(s);
  if (s === "pending") return chalk.yellow(s);
  if (s === "partial") return chalk.yellow(s);
  if (s === "canceled" || s === "rejected") return chalk.red(s);
  return s;
}

function shortDate(iso) {
  if (!iso) return chalk.gray("—");
  return iso.replace("T", " ").slice(0, 19);
}

function defaultSince() {
  const d = new Date();
  d.setDate(d.getDate() - 30);
  return d.toISOString().slice(0, 10);
}

// ── Commands ─────────────────────────────────────────────────────────

function cmdStats(argv) {
  initJournal();
  const since = argv.since || defaultSince();
  const stats = journal.getStrategyStats(since);

  if (stats.length === 0) {
    console.log(chalk.yellow(`No trades found since ${since}`));
    return;
  }

  console.log(chalk.bold(`\n📊 Strategy Stats (since ${since})\n`));

  const table = new Table({
    head: ["Strategy", "Trades", "Buys", "Sells", "Wins", "Losses", "Win%", "Total P&L", "Avg P&L", "Avg Hold", "Avg Slip"],
    style: { head: ["cyan"] },
  });

  for (const s of stats) {
    const winRate = s.sells > 0 ? ((s.wins / s.sells) * 100).toFixed(1) + "%" : "—";
    table.push([
      chalk.bold(s.strategy),
      s.total_trades,
      s.buys,
      s.sells,
      chalk.green(s.wins || 0),
      chalk.red(s.losses || 0),
      winRate,
      pnlColor(s.total_pnl),
      pnlColor(s.avg_pnl),
      s.avg_hold_days != null ? `${Math.round(s.avg_hold_days)}d` : "—",
      s.avg_slippage_bps != null ? `${Math.round(s.avg_slippage_bps)}bps` : "—",
    ]);
  }

  console.log(table.toString());
}

function cmdStrategy(argv) {
  initJournal();
  const since = argv.since || defaultSince();
  const stats = journal.getStrategyStats(since);

  if (stats.length === 0) {
    console.log(chalk.yellow(`No trades found since ${since}`));
    return;
  }

  console.log(chalk.bold(`\n📈 Strategy Breakdown (since ${since})\n`));

  for (const s of stats) {
    const winRate = s.sells > 0 ? ((s.wins / s.sells) * 100).toFixed(1) : "N/A";
    console.log(chalk.bold.underline(`  ${s.strategy}`));
    console.log(`    Trades: ${s.total_trades} (${s.buys} buys, ${s.sells} sells)`);
    console.log(`    Win/Loss: ${chalk.green(s.wins || 0)}/${chalk.red(s.losses || 0)} (${winRate}%)`);
    console.log(`    Total P&L: ${pnlColor(s.total_pnl)}  |  Avg P&L: ${pnlColor(s.avg_pnl)}`);
    console.log(`    Avg Hold: ${s.avg_hold_days != null ? Math.round(s.avg_hold_days) + " days" : "—"}`);
    console.log(`    Avg Slippage: ${s.avg_slippage_bps != null ? Math.round(s.avg_slippage_bps) + " bps" : "—"}`);
    console.log();
  }
}

function cmdTrades(argv) {
  initJournal();
  let trades;

  if (argv.symbol) {
    trades = journal.getTradesBySymbol(argv.symbol, argv.last || 20);
  } else if (argv.strategy) {
    trades = journal.getTradesByStrategy(argv.strategy, argv.since || "2000-01-01");
    if (argv.last) trades = trades.slice(0, argv.last);
  } else {
    // Use date range: last N trades via raw query
    const db = journal.getDb();
    const limit = argv.last || 20;
    let query = "SELECT * FROM trades";
    const conditions = [];
    const params = {};

    if (argv.status) {
      conditions.push("status = @status");
      params.status = argv.status;
    }
    if (argv.since) {
      conditions.push("submitted_at >= @since");
      params.since = argv.since;
    }

    if (conditions.length > 0) query += " WHERE " + conditions.join(" AND ");
    query += " ORDER BY submitted_at DESC LIMIT @limit";
    params.limit = limit;

    trades = db.prepare(query).all(params);
  }

  if (trades.length === 0) {
    console.log(chalk.yellow("No trades found matching criteria."));
    return;
  }

  console.log(chalk.bold(`\n📋 Trades (${trades.length} results)\n`));

  const table = new Table({
    head: ["ID", "Time", "Symbol", "Side", "Qty", "Strategy", "Price", "Fill", "Slip", "P&L", "Status"],
    style: { head: ["cyan"] },
    colWidths: [6, 21, 8, 6, 6, 12, 10, 10, 7, 12, 10],
  });

  for (const t of trades) {
    const sideStr = t.side === "buy" ? chalk.green("BUY") : chalk.red("SELL");
    table.push([
      t.id,
      shortDate(t.submitted_at),
      chalk.bold(t.symbol),
      sideStr,
      t.qty,
      t.strategy || "—",
      t.intended_price ? `$${parseFloat(t.intended_price).toFixed(2)}` : "—",
      t.fill_price ? `$${parseFloat(t.fill_price).toFixed(2)}` : "—",
      t.slippage_bps != null ? `${t.slippage_bps}bps` : "—",
      pnlColor(t.realized_pnl),
      statusColor(t.status),
    ]);
  }

  console.log(table.toString());
}

function cmdPosition(argv) {
  initJournal();

  if (argv.symbol) {
    const pos = journal.getDb().prepare("SELECT * FROM positions WHERE symbol = ?").get(argv.symbol.toUpperCase());
    if (!pos) {
      console.log(chalk.yellow(`No position found for ${argv.symbol.toUpperCase()}`));
      return;
    }
    console.log(chalk.bold(`\n📌 Position: ${pos.symbol}\n`));
    console.log(`  Strategy:    ${pos.strategy || "—"}`);
    console.log(`  Qty:         ${pos.qty}`);
    console.log(`  Avg Cost:    $${parseFloat(pos.avg_cost).toFixed(2)}`);
    console.log(`  Cost Basis:  $${parseFloat(pos.cost_basis).toFixed(2)}`);
    console.log(`  Current:     ${pos.current_price ? "$" + parseFloat(pos.current_price).toFixed(2) : "—"}`);
    console.log(`  Value:       ${pos.current_value ? "$" + parseFloat(pos.current_value).toFixed(2) : "—"}`);
    console.log(`  Unrealized:  ${pnlColor(pos.unrealized_pnl)} (${pctColor(pos.unrealized_pnl_pct)})`);
    console.log(`  Days Held:   ${pos.days_held || "—"}`);
    console.log(`  Stop Loss:   ${pos.stop_loss ? "$" + parseFloat(pos.stop_loss).toFixed(2) : "—"}`);
    console.log(`  Take Profit: ${pos.take_profit ? "$" + parseFloat(pos.take_profit).toFixed(2) : "—"}`);
    console.log(`  Trail Stop:  ${pos.trailing_stop ? "$" + parseFloat(pos.trailing_stop).toFixed(2) : "—"}`);
    console.log(`  High Since:  ${pos.highest_price_since_entry ? "$" + parseFloat(pos.highest_price_since_entry).toFixed(2) : "—"}`);
    console.log(`  Opened:      ${shortDate(pos.opened_at)}`);
    console.log();
    return;
  }

  // List all positions
  const positions = journal.getOpenPositions();
  if (positions.length === 0) {
    console.log(chalk.yellow("No open positions."));
    return;
  }

  console.log(chalk.bold(`\n📌 Open Positions (${positions.length})\n`));

  const table = new Table({
    head: ["Symbol", "Qty", "Avg Cost", "Current", "Value", "P&L", "P&L%", "Days", "Strategy"],
    style: { head: ["cyan"] },
  });

  for (const p of positions) {
    table.push([
      chalk.bold(p.symbol),
      p.qty,
      `$${parseFloat(p.avg_cost).toFixed(2)}`,
      p.current_price ? `$${parseFloat(p.current_price).toFixed(2)}` : "—",
      p.current_value ? `$${parseFloat(p.current_value).toFixed(2)}` : "—",
      pnlColor(p.unrealized_pnl),
      pctColor(p.unrealized_pnl_pct),
      p.days_held || "—",
      p.strategy || "—",
    ]);
  }

  console.log(table.toString());
}

function cmdDay(argv) {
  initJournal();
  const date = argv.date || new Date().toISOString().slice(0, 10);

  const snap = journal.getDailySnapshots(date, date);
  if (snap.length === 0) {
    console.log(chalk.yellow(`No snapshot found for ${date}`));
    return;
  }

  const s = snap[0];
  console.log(chalk.bold(`\n📅 Daily Snapshot: ${s.date}\n`));
  console.log(`  Portfolio Value: $${parseFloat(s.portfolio_value).toFixed(2)}`);
  console.log(`  Cash:            $${parseFloat(s.cash).toFixed(2)}`);
  console.log(`  Equity:          ${s.equity ? "$" + parseFloat(s.equity).toFixed(2) : "—"}`);
  console.log(`  Positions:       ${s.positions_count}`);
  console.log(`  Day P&L:         ${pnlColor(s.day_pnl)} (${pctColor(s.day_pnl_pct)})`);
  console.log(`  Regime:          ${s.regime || "—"}`);
  console.log(`  SPY Close:       ${s.spy_close ? "$" + parseFloat(s.spy_close).toFixed(2) : "—"}`);
  console.log(`  SPY Day%:        ${s.spy_day_pct != null ? (parseFloat(s.spy_day_pct) * 100).toFixed(2) + "%" : "—"}`);
  console.log(`  Peak Value:      ${s.peak_value ? "$" + parseFloat(s.peak_value).toFixed(2) : "—"}`);
  console.log(`  Drawdown:        ${s.drawdown_pct != null ? (parseFloat(s.drawdown_pct) * 100).toFixed(2) + "%" : "—"}`);
  console.log(`  Circuit Breaker: ${s.circuit_breaker_state || "—"}`);

  if (s.strategy_pnl_json) {
    console.log(chalk.bold(`\n  Strategy P&L:`));
    const strats = JSON.parse(s.strategy_pnl_json);
    for (const [k, v] of Object.entries(strats)) {
      console.log(`    ${k}: ${pnlColor(v)}`);
    }
  }

  if (s.slot_usage_json) {
    console.log(chalk.bold(`\n  Slot Usage:`));
    const slots = JSON.parse(s.slot_usage_json);
    for (const [k, v] of Object.entries(slots)) {
      console.log(`    ${k}: ${v}`);
    }
  }
  console.log();
}

function cmdEvents(argv) {
  initJournal();
  const limit = argv.last || 20;
  let events;

  if (argv.type) {
    events = journal.getEventsByType(argv.type, limit);
  } else {
    events = journal.getRecentEvents(limit);
  }

  if (argv.severity) {
    events = events.filter(e => e.severity === argv.severity);
  }

  if (events.length === 0) {
    console.log(chalk.yellow("No events found."));
    return;
  }

  console.log(chalk.bold(`\n⚡ Events (${events.length})\n`));

  const table = new Table({
    head: ["ID", "Time", "Type", "Severity", "Message", "Value"],
    style: { head: ["cyan"] },
    colWidths: [6, 21, 18, 10, 40, 14],
    wordWrap: true,
  });

  for (const e of events) {
    const sevColor = e.severity === "critical" ? chalk.red.bold
      : e.severity === "error" ? chalk.red
      : e.severity === "warning" ? chalk.yellow
      : chalk.white;

    table.push([
      e.id,
      shortDate(e.created_at),
      e.event_type,
      sevColor(e.severity),
      e.message || "—",
      e.portfolio_value_at_event ? `$${parseFloat(e.portfolio_value_at_event).toFixed(0)}` : "—",
    ]);
  }

  console.log(table.toString());
}

function cmdSignals(argv) {
  initJournal();
  const since = argv.since || defaultSince();

  let signals;
  if (argv.symbol) {
    signals = journal.getDb().prepare(
      "SELECT * FROM signals WHERE symbol = @symbol AND signal_time >= @since ORDER BY signal_time DESC LIMIT @limit"
    ).all({ symbol: argv.symbol.toUpperCase(), since, limit: argv.last || 50 });
  } else {
    signals = journal.getDb().prepare(
      "SELECT * FROM signals WHERE signal_time >= @since ORDER BY signal_time DESC LIMIT @limit"
    ).all({ since, limit: argv.last || 50 });
  }

  if (signals.length === 0) {
    console.log(chalk.yellow(`No signals found since ${since}`));
    return;
  }

  console.log(chalk.bold(`\n🔔 Signals (${signals.length})\n`));

  const table = new Table({
    head: ["ID", "Time", "Symbol", "Prob", "Rank", "Top5", "Price", "Regime", "Acted"],
    style: { head: ["cyan"] },
  });

  for (const s of signals) {
    const probStr = s.probability != null ? `${(parseFloat(s.probability) * 100).toFixed(1)}%` : "—";
    table.push([
      s.id,
      shortDate(s.signal_time),
      chalk.bold(s.symbol),
      probStr,
      s.rank || "—",
      s.is_top_5 ? chalk.green("✓") : "—",
      s.price_at_signal ? `$${parseFloat(s.price_at_signal).toFixed(2)}` : "—",
      s.regime || "—",
      s.acted_on ? chalk.green("✓") : chalk.gray("✗"),
    ]);
  }

  console.log(table.toString());
}

// ── CLI Setup ────────────────────────────────────────────────────────

yargs(hideBin(process.argv))
  .scriptName("journal")
  .usage("$0 <command> [options]")

  .command("stats", "Strategy performance stats", (y) => {
    y.option("since", { type: "string", describe: "Start date (YYYY-MM-DD)", default: defaultSince() });
  }, cmdStats)

  .command("strategy", "Strategy breakdown (detailed)", (y) => {
    y.option("since", { type: "string", describe: "Start date (YYYY-MM-DD)", default: defaultSince() });
  }, cmdStrategy)

  .command("trades", "List trades", (y) => {
    y.option("last", { type: "number", describe: "Number of trades", default: 20 });
    y.option("symbol", { type: "string", describe: "Filter by symbol" });
    y.option("strategy", { type: "string", describe: "Filter by strategy" });
    y.option("status", { type: "string", describe: "Filter by status" });
    y.option("since", { type: "string", describe: "Start date" });
  }, cmdTrades)

  .command("position [symbol]", "Show position(s)", (y) => {
    y.positional("symbol", { type: "string", describe: "Symbol (omit for all)" });
  }, cmdPosition)

  .command("day [date]", "Daily snapshot", (y) => {
    y.positional("date", { type: "string", describe: "Date (YYYY-MM-DD)", default: new Date().toISOString().slice(0, 10) });
  }, cmdDay)

  .command("events", "List events", (y) => {
    y.option("last", { type: "number", describe: "Number of events", default: 20 });
    y.option("type", { type: "string", describe: "Filter by event_type" });
    y.option("severity", { type: "string", describe: "Filter by severity" });
  }, cmdEvents)

  .command("signals", "List ML signals", (y) => {
    y.option("symbol", { type: "string", describe: "Filter by symbol" });
    y.option("since", { type: "string", describe: "Start date", default: defaultSince() });
    y.option("last", { type: "number", describe: "Number of signals", default: 50 });
  }, cmdSignals)

  .demandCommand(1, "Please specify a command")
  .strict()
  .help()
  .argv;
