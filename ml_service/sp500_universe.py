"""
S&P 500 Universe + Infrastructure ETFs
=======================================
Central universe definition for the trading system.
Fetches current S&P 500 constituents from FMP and combines with infrastructure ETFs.

Usage:
    from sp500_universe import get_stock_symbols, get_etf_symbols, get_full_universe

Run standalone to refresh the cached S&P 500 list:
    python3 sp500_universe.py
"""

import json
import os
import ssl
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

CACHE_FILE = Path(__file__).resolve().parent / "data" / "sp500_constituents.json"

# SSL context (Mac Python sometimes lacks certs)
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Infrastructure ETFs (always included, never traded with fundamentals) ────
ETF_SYMBOLS = [
    # Broad market
    "SPY", "QQQ", "IWM",
    # Sector ETFs
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC",
    # International
    "EWZ", "EWJ", "FXI", "INDA", "EFA", "EEM", "VGK", "VWO", "IEFA",
    # Commodities
    "GLD", "SLV", "USO", "DBC", "CPER",
    # Bonds
    "TLT", "IEF", "SHY", "HYG", "LQD",
    # Volatility / Macro
    "VIXY", "UUP",
]

# Cross-asset symbols needed for feature engineering (may overlap with ETFs)
CROSS_ASSET = ["SPY", "VIXY", "TLT"]

# ── Fallback S&P 500 list (used when FMP API is unavailable) ─────────────────
_FALLBACK_SP500 = [
    "AAPL","ABBV","ABT","ACN","ADBE","ADI","ADM","ADP","ADSK","AEE","AEP","AES",
    "AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALK","ALL","ALLE","AMAT","AMCR",
    "AMD","AME","AMGN","AMP","AMT","AMZN","ANET","ANSS","AON","AOS","APA","APD",
    "APH","APTV","ARE","ATO","ATVI","AVB","AVGO","AVY","AWK","AXP","AZO",
    "BA","BAC","BAX","BBWI","BBY","BDX","BEN","BF.B","BIO","BIIB","BK","BKNG",
    "BKR","BLK","BMY","BR","BRK.B","BRO","BSX","BWA","BXP",
    "C","CAG","CAH","CAT","CB","CBOE","CBRE","CCI","CCL","CDAY","CDNS","CDW",
    "CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CINF","CL","CLX","CMA",
    "CMCSA","CME","CMG","CMI","CMS","CNC","CNP","COF","COO","COP","COST","CPB",
    "CPRT","CPT","CRL","CRM","CSCO","CSGP","CSX","CTAS","CTLT","CTRA","CTSH",
    "CTVA","CVS","CVX","CZR",
    "D","DAL","DD","DE","DFS","DG","DGX","DHI","DHR","DIS","DISH","DLTR",
    "DOV","DOW","DPZ","DRI","DTE","DUK","DVA","DVN","DXC",
    "EA","EBAY","ECL","ED","EFX","EIX","EL","EMN","EMR","ENPH","EOG","EPAM",
    "EQIX","EQR","EQT","ES","ESS","ETN","ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE","EXR",
    "F","FANG","FAST","FBHS","FCX","FDS","FDX","FE","FFIV","FIS","FISV","FITB",
    "FLT","FMC","FOX","FOXA","FRC","FRT",
    "FTNT","FTV",
    "GD","GE","GEHC","GEN","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL",
    "GPC","GPN","GRMN","GS","GWW",
    "HAL","HAS","HBAN","HCA","HOLX","HON","HPE","HPQ","HRL","HSIC","HST","HSY",
    "HUM","HWM",
    "IBM","ICE","IDXX","IEX","IFF","ILMN","INCY","INTC","INTU","INVH","IP","IPG",
    "IQV","IR","IRM","ISRG","IT","ITW","IVZ",
    "J","JBHT","JCI","JKHY","JNJ","JNPR","JPM",
    "K","KDP","KEY","KEYS","KHC","KIM","KLAC","KMB","KMI","KMX","KO","KR",
    "L","LDOS","LEN","LH","LHX","LIN","LKQ","LLY","LMT","LNC","LNT","LOW",
    "LRCX","LUMN","LUV","LVS","LW","LYB","LYV",
    "MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META",
    "MGM","MHK","MKC","MKTX","MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC",
    "MPWR","MRK","MRNA","MRO","MS","MSCI","MSFT","MSI","MTB","MTCH","MTD","MU",
    "NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE","NOC","NOW","NRG","NSC",
    "NTAP","NTRS","NUE","NVDA","NVR","NWL","NWS","NWSA",
    "O","ODFL","OGN","OKE","OMC","ON","ORCL","ORLY","OTIS","OXY",
    "PARA","PAYC","PAYX","PCAR","PCG","PEAK","PEG","PEP","PFE","PFG","PG","PGR",
    "PH","PHM","PKG","PKI","PLD","PM","PNC","PNR","PNW","POOL","PPG","PPL",
    "PRU","PSA","PSX","PTC","PVH","PWR","PXD",
    "QCOM","QRVO",
    "RCL","RE","REG","REGN","RF","RHI","RJF","RL","RMD","ROK","ROL","ROP","ROST","RSG",
    "RTX",
    "SBAC","SBNY","SBUX","SCHW","SEE","SHW","SIVB","SJM","SLB","SNA","SNPS",
    "SO","SPG","SPGI","SRE","STE","STT","STX","STZ","SWK","SWKS","SYF","SYK","SYY",
    "T","TAP","TDG","TDY","TECH","TEL","TER","TFC","TFX","TGT","TMO","TMUS",
    "TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT","TYL",
    "UAL","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB",
    "V","VFC","VICI","VLO","VMC","VRSK","VRSN","VRTX","VTR","VTRS","VZ",
    "WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC","WHR","WM","WMB","WMT",
    "WRB","WRK","WST","WTW","WY","WYNN",
    "XEL","XOM","XRAY","XYL",
    "YUM",
    "ZBH","ZBRA","ZION","ZTS",
]


def fetch_sp500_from_fmp() -> list[str]:
    """Fetch current S&P 500 constituents from FMP API."""
    if not FMP_API_KEY:
        print("  FMP_API_KEY not set — using fallback list")
        return list(_FALLBACK_SP500)

    url = f"https://financialmodelingprep.com/stable/sp500-constituent?apikey={FMP_API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "auto-trader/1.0"})
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list) and len(data) > 400:
            symbols = sorted(set(item.get("symbol", "") for item in data if item.get("symbol")))
            print(f"  Fetched {len(symbols)} S&P 500 constituents from FMP")
            return symbols
        print(f"  FMP returned unexpected data ({len(data) if isinstance(data, list) else type(data).__name__}) — using fallback")
        return list(_FALLBACK_SP500)
    except Exception as e:
        print(f"  FMP fetch failed: {e} — using fallback")
        return list(_FALLBACK_SP500)


def get_stock_symbols(use_cache: bool = True) -> list[str]:
    """Get S&P 500 stock symbols (cached to disk)."""
    if use_cache and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) > 400:
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    symbols = fetch_sp500_from_fmp()

    # Save cache
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(symbols, f)
    except OSError:
        pass

    return symbols


def get_etf_symbols() -> list[str]:
    """Get infrastructure ETF symbols."""
    return list(ETF_SYMBOLS)


def get_full_universe() -> list[str]:
    """Get full trading universe: S&P 500 + ETFs (deduplicated)."""
    stocks = get_stock_symbols()
    return sorted(set(stocks + ETF_SYMBOLS))


def get_cross_asset() -> list[str]:
    """Get cross-asset reference symbols."""
    return list(CROSS_ASSET)


def get_all_symbols() -> list[str]:
    """Get all symbols: SP1500 + ETFs + cross-asset references (deduplicated).

    v10: Loads SP1500 membership from sp1500_members.json (weekly Wikipedia scrape).
    Falls back to SP500-only if the file is missing.
    """
    sp1500_file = Path(__file__).resolve().parent / "data" / "sp1500_members.json"
    members = []
    try:
        with open(sp1500_file) as f:
            data = json.load(f)
        members = data.get("sp500", []) + data.get("sp400", []) + data.get("sp600", [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if len(members) < 1000:
        # Fallback to SP500-only
        members = get_full_universe()

    return sorted(set(members + ETF_SYMBOLS + CROSS_ASSET))


if __name__ == "__main__":
    print("=" * 60)
    print("S&P 500 Universe Builder")
    print("=" * 60)

    stocks = get_stock_symbols(use_cache=False)
    etfs = get_etf_symbols()
    full = get_full_universe()

    print(f"\n  S&P 500 stocks: {len(stocks)}")
    print(f"  Infrastructure ETFs: {len(etfs)}")
    print(f"  Full universe (deduplicated): {len(full)}")
    print(f"  Cache file: {CACHE_FILE}")

    # Show overlap
    overlap = set(stocks) & set(etfs)
    if overlap:
        print(f"  Overlap (in both S&P 500 and ETF list): {sorted(overlap)}")

    print(f"\n  First 20 stocks: {stocks[:20]}")
    print(f"  ETFs: {etfs}")
    print(f"\n  Done.")
