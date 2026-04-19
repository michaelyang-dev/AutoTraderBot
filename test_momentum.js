// Quick test to verify momentum strategy works
const Alpaca = require('@alpacahq/alpaca-trade-api');
require('dotenv').config();

async function test() {
  const alpaca = new Alpaca({
    keyId: process.env.ALPACA_API_KEY,
    secretKey: process.env.ALPACA_SECRET_KEY,
    paper: true,
  });

  const UNIVERSE_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'UNH'];
  
  const priceHist = {};
  const volHist = {};
  
  console.log('Fetching 75 days of data...');
  const end = new Date();
  const start = new Date(end.getTime() - 100 * 24 * 60 * 60 * 1000);
  
  for (const sym of UNIVERSE_SYMBOLS) {
    try {
      const bars = [];
      for await (const bar of alpaca.getBarsV2(sym, {
        start: start.toISOString(),
        end: end.toISOString(),
        timeframe: '1Day',
      })) {
        bars.push(bar);
      }
      priceHist[sym] = bars.map(b => b.ClosePrice);
      volHist[sym] = bars.map(b => b.Volume);
      console.log(`${sym}: ${bars      console.log(`${sym}: ${be)      console.log(`${sym}: ${bars      consolege}`);
    }
  }
  
  console.log('\nCalculating momentu  console.log('\nCalculatikin  console.log('\nCalcuma  console.log('\nCalculatis =   console.log('\    if (!prices || pric  console.log('\nCalculating momentu  consol (  console.log('ength - 1] - prices[prices.length - 63]) / prices[prices.length - 63];
    return { sym, ret };
  }).filter(x => x).sort((a, b) => b.ret - a.  }).filter(x => x.log('Top momentum picks:');
  rankings.slice(0, 5).forEach((r, i) => {
    console.log(`  ${i+1}. ${r.sym}: ${(r.ret * 100).toFixed(1)}%`);
  });
}

ttst().catch(console.error);
