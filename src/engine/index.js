// Engine barrel export — Simulated + Live
export { simPrice, initPriceHistory, advancePrices } from "./priceEngine";
export { sma, ema, rsi, macd, bollinger } from "./indicators";
export { getSignals } from "./signalEngine";
export { executeTradingCycle } from "./tradeExecutor";

// Alpaca live trading
export * as alpacaClient from "./alpacaClient";
export { executeLiveTradingCycle } from "./liveTradeExecutor";
export { initLivePriceHistory, updateLivePrices } from "./livePriceEngine";
