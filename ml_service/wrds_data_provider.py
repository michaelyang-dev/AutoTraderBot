"""
WRDS Data Provider
==================
Loads CRSP, Compustat, and FRED data from WRDS parquet files.
Handles identifier mapping (PERMNO ↔ ticker ↔ GVKEY) with point-in-time
resolution to handle ticker reuse and renames.

This module is for BACKTESTING AND RESEARCH ONLY — not live production.
Live trading uses FMP + Massive/Polygon.

Usage:
    from wrds_data_provider import WRDSDataProvider
    wp = WRDSDataProvider()
    members = wp.get_sp500_members("2024-01-02")  # → set of tickers
    prices = wp.load_sp500_prices("2018-01-01", "2025-12-31")  # → DataFrame
    fundamentals = wp.load_fundamentals(permnos, "2018-01-01", "2025-12-31")
"""

import bisect
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger("wrds_data_provider")

WRDS_DIR = Path(__file__).resolve().parent / "data" / "wrds"


# ─── Identifier Mapping ──────────────────────────────────────────────────────

class TickerMapper:
    """Point-in-time bidirectional mapping: PERMNO ↔ ticker ↔ GVKEY.

    CRSP uses PERMNO (permanent security ID, never changes).
    Tickers change (FB→META) and get reused (FB was a different company in 1976-1983).
    Compustat uses GVKEY, linked to PERMNO via the link table.

    All lookups require a date because the same ticker can refer to
    different securities at different times.
    """

    def __init__(self, wrds_dir: Path = WRDS_DIR):
        log.info("Building TickerMapper...")

        # ── PERMNO ↔ Ticker mapping (from crsp_security_info) ────────────
        si = pd.read_parquet(
            wrds_dir / "crsp_security_info.parquet",
            columns=["PERMNO", "Ticker", "SecInfoStartDt", "SecInfoEndDt"],
        )
        si["SecInfoStartDt"] = pd.to_datetime(si["SecInfoStartDt"]).dt.date
        si["SecInfoEndDt"] = pd.to_datetime(si["SecInfoEndDt"]).dt.date
        si = si.dropna(subset=["Ticker", "SecInfoStartDt", "SecInfoEndDt"])
        si["PERMNO"] = si["PERMNO"].astype(int)

        # Build ticker→[(start, end, permno)] sorted by start date
        self._ticker_to_ranges: dict[str, list[tuple[date, date, int]]] = {}
        for _, row in si.iterrows():
            ticker = row["Ticker"]
            entry = (row["SecInfoStartDt"], row["SecInfoEndDt"], row["PERMNO"])
            self._ticker_to_ranges.setdefault(ticker, []).append(entry)
        for ticker in self._ticker_to_ranges:
            self._ticker_to_ranges[ticker].sort(key=lambda x: x[0])

        # Build permno→[(start, end, ticker)] sorted by start date
        self._permno_to_ranges: dict[int, list[tuple[date, date, str]]] = {}
        for _, row in si.iterrows():
            permno = row["PERMNO"]
            entry = (row["SecInfoStartDt"], row["SecInfoEndDt"], row["Ticker"])
            self._permno_to_ranges.setdefault(permno, []).append(entry)
        for permno in self._permno_to_ranges:
            self._permno_to_ranges[permno].sort(key=lambda x: x[0])

        log.info("  TickerMapper: %d unique tickers, %d unique PERMNOs",
                 len(self._ticker_to_ranges), len(self._permno_to_ranges))

        # ── GVKEY ↔ PERMNO mapping (from compustat_crsp_link) ────────────
        link = pd.read_parquet(
            wrds_dir / "compustat_crsp_link.parquet",
            columns=["gvkey", "LPERMNO", "LINKPRIM", "LINKTYPE", "LINKDT", "LINKENDDT"],
        )
        # Filter to primary, valid link types
        link = link[
            (link["LINKPRIM"] == "P") &
            (link["LINKTYPE"].isin(["LC", "LU"]))
        ].copy()
        link["LINKDT"] = pd.to_datetime(link["LINKDT"], errors="coerce").dt.date
        # LINKENDDT can be NaT or non-date values like "E" (still active) — set to far future
        link["LINKENDDT"] = pd.to_datetime(link["LINKENDDT"], errors="coerce").dt.date
        link["LINKDT"] = link["LINKDT"].fillna(date(1900, 1, 1))
        link["LINKENDDT"] = link["LINKENDDT"].fillna(date(2099, 12, 31))
        link["LPERMNO"] = link["LPERMNO"].astype(int)

        # gvkey→[(start, end, permno)]
        self._gvkey_to_ranges: dict[str, list[tuple[date, date, int]]] = {}
        for _, row in link.iterrows():
            gvkey = str(row["gvkey"])
            entry = (row["LINKDT"], row["LINKENDDT"], row["LPERMNO"])
            self._gvkey_to_ranges.setdefault(gvkey, []).append(entry)

        # permno→[(start, end, gvkey)]
        self._permno_to_gvkey_ranges: dict[int, list[tuple[date, date, str]]] = {}
        for _, row in link.iterrows():
            permno = row["LPERMNO"]
            entry = (row["LINKDT"], row["LINKENDDT"], str(row["gvkey"]))
            self._permno_to_gvkey_ranges.setdefault(permno, []).append(entry)

        log.info("  TickerMapper: %d GVKEY→PERMNO links loaded", len(link))

    def ticker_to_permno(self, ticker: str, dt: date) -> Optional[int]:
        """Which PERMNO held this ticker on this date?
        Returns None if not found. Handles ticker reuse."""
        ranges = self._ticker_to_ranges.get(ticker)
        if not ranges:
            return None
        for start, end, permno in ranges:
            if start <= dt <= end:
                return permno
        return None

    def permno_to_ticker(self, permno: int, dt: date) -> Optional[str]:
        """What ticker did this PERMNO trade under on this date?"""
        ranges = self._permno_to_ranges.get(permno)
        if not ranges:
            return None
        for start, end, ticker in ranges:
            if start <= dt <= end:
                return ticker
        # Fallback: return the most recent ticker before the date
        for start, end, ticker in reversed(ranges):
            if start <= dt:
                return ticker
        return ranges[-1][2] if ranges else None

    def gvkey_to_permno(self, gvkey: str, dt: date) -> Optional[int]:
        """Map Compustat GVKEY to CRSP PERMNO on this date."""
        ranges = self._gvkey_to_ranges.get(gvkey)
        if not ranges:
            return None
        for start, end, permno in ranges:
            if start <= dt <= end:
                return permno
        return None

    def permno_to_gvkey(self, permno: int, dt: date) -> Optional[str]:
        """Map CRSP PERMNO to Compustat GVKEY on this date."""
        ranges = self._permno_to_gvkey_ranges.get(permno)
        if not ranges:
            return None
        for start, end, gvkey in ranges:
            if start <= dt <= end:
                return gvkey
        return None

    def batch_permno_to_ticker(self, permnos: list[int], dt: date) -> dict[int, str]:
        """Resolve multiple PERMNOs to tickers on a given date."""
        return {p: t for p in permnos if (t := self.permno_to_ticker(p, dt)) is not None}


# ─── SP500 Membership ────────────────────────────────────────────────────────

class SP500Membership:
    """Ground-truth SP500 membership from CRSP constituent daily file.

    Pre-builds a {date → set(PERMNO)} lookup for O(1) access.
    Also stores {date → {PERMNO: ticker}} for ticker resolution.
    """

    def __init__(self, wrds_dir: Path = WRDS_DIR):
        log.info("Loading SP500 membership from CRSP constituent daily...")

        df = pd.read_parquet(
            wrds_dir / "crsp_sp500_constituent_daily.parquet",
            columns=["PERMNO", "Ticker", "DlyCalDt"],
        )
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"]).dt.date
        df["PERMNO"] = df["PERMNO"].astype(int)

        # Build {date: {permno: ticker}}
        self._membership: dict[date, dict[int, str]] = {}
        for dt, group in df.groupby("DlyCalDt"):
            self._membership[dt] = dict(zip(group["PERMNO"], group["Ticker"]))

        # Sorted dates for nearest-date lookups
        self._dates = sorted(self._membership.keys())

        # All PERMNOs that were ever in SP500 (for filtering price data)
        self._all_permnos = set()
        for members in self._membership.values():
            self._all_permnos.update(members.keys())

        log.info("  SP500 membership: %d trading days, %d unique PERMNOs ever",
                 len(self._dates), len(self._all_permnos))

    def get_members_permno(self, dt: date) -> set[int]:
        """Return set of PERMNOs in SP500 on this date."""
        members = self._membership.get(dt)
        if members:
            return set(members.keys())
        # Find nearest prior trading day
        idx = bisect.bisect_right(self._dates, dt) - 1
        if idx >= 0:
            return set(self._membership[self._dates[idx]].keys())
        return set()

    def get_members_ticker(self, dt: date) -> set[str]:
        """Return set of tickers in SP500 on this date."""
        members = self._membership.get(dt)
        if members:
            return set(members.values())
        idx = bisect.bisect_right(self._dates, dt) - 1
        if idx >= 0:
            return set(self._membership[self._dates[idx]].values())
        return set()

    def get_members_map(self, dt: date) -> dict[int, str]:
        """Return {PERMNO: ticker} for SP500 on this date."""
        members = self._membership.get(dt)
        if members:
            return dict(members)
        idx = bisect.bisect_right(self._dates, dt) - 1
        if idx >= 0:
            return dict(self._membership[self._dates[idx]])
        return {}

    def get_all_permnos(self) -> set[int]:
        """All PERMNOs that were ever in SP500 (for data filtering)."""
        return self._all_permnos

    def get_trading_dates(self, start: date, end: date) -> list[date]:
        """Trading dates in range."""
        i = bisect.bisect_left(self._dates, start)
        j = bisect.bisect_right(self._dates, end)
        return self._dates[i:j]


# ─── Price Data ──────────────────────────────────────────────────────────────

def load_sp500_prices(
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    sp500_membership: Optional[SP500Membership] = None,
    wrds_dir: Path = WRDS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CRSP daily prices for SP500 stocks in date range.

    Returns:
        prices: DataFrame[date × ticker] of adjusted close prices
        open_prices: DataFrame[date × ticker] of open prices
        volumes: DataFrame[date × ticker] of volumes

    Uses column-selective + PERMNO-filtered parquet reading to keep
    memory manageable (the full file is 110M rows / 5.77 GB).
    """
    if sp500_membership is None:
        sp500_membership = SP500Membership(wrds_dir)

    permnos = list(sp500_membership.get_all_permnos())

    # Also need key ETFs that aren't SP500 constituents but are used by the strategy
    # (SPY for benchmark, sector ETFs for strategy3, QQQ/IWM for macro)
    ETF_TICKERS = [
        "SPY", "QQQ", "IWM",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC",
        "TLT", "GLD", "VXX", "HYG",
    ]
    # Resolve ETF tickers to PERMNOs via the security info file
    sec_info = pd.read_parquet(
        wrds_dir / "crsp_security_info.parquet",
        columns=["PERMNO", "Ticker"],
    )
    etf_permnos = sec_info[sec_info["Ticker"].isin(ETF_TICKERS)]["PERMNO"].unique().tolist()
    permnos = list(set(permnos + etf_permnos))

    log.info("Loading CRSP prices for %d PERMNOs (SP500 + ETFs), %s to %s...",
             len(permnos), start, end)

    # Column-selective read with PERMNO filter
    columns = [
        "PERMNO", "Ticker", "DlyCalDt",
        "DlyClose", "DlyOpen", "DlyHigh", "DlyLow",
        "DlyPrc", "DlyRet", "DlyVol", "ShrOut",
    ]

    df = pd.read_parquet(
        wrds_dir / "crsp_daily_stock_full.parquet",
        columns=columns,
        filters=[
            ("PERMNO", "in", permnos),
        ],
    )
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    df = df[(df["DlyCalDt"] >= start) & (df["DlyCalDt"] <= end)].copy()
    df["PERMNO"] = df["PERMNO"].astype(int)

    log.info("  Loaded %d rows for %d PERMNOs", len(df), df["PERMNO"].nunique())

    # CRITICAL: DlyClose is NOT split-adjusted in CRSP.
    # DlyRet IS adjusted (includes split/dividend factors).
    # We must reconstruct adjusted close prices from DlyRet.
    #
    # Strategy: for each PERMNO, anchor to the LAST known raw price
    # and work backward using cumulative returns to build adjusted prices.
    df["ticker"] = df["Ticker"]
    df = df.sort_values(["ticker", "DlyCalDt"])
    df = df.dropna(subset=["ticker"])
    df = df.drop_duplicates(subset=["DlyCalDt", "ticker"], keep="last")

    # Build adjusted close prices per ticker using DlyRet
    log.info("  Reconstructing split-adjusted prices from DlyRet...")
    adjusted_prices = {}
    adjusted_opens = {}

    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("DlyCalDt").copy()
        ret = g["DlyRet"].fillna(0).values
        raw_close = g["DlyPrc"].abs().values
        raw_open = g["DlyOpen"].fillna(g["DlyPrc"].abs()).values

        # Anchor: last raw price = last adjusted price (most recent, no future splits)
        # Work backward: adj_close[i] = adj_close[i+1] / (1 + ret[i+1])
        n = len(ret)
        adj_close = np.empty(n)
        adj_close[-1] = raw_close[-1]
        for i in range(n - 2, -1, -1):
            if ret[i + 1] != 0 and not np.isnan(ret[i + 1]):
                adj_close[i] = adj_close[i + 1] / (1 + ret[i + 1])
            else:
                adj_close[i] = adj_close[i + 1]

        # Adjusted open: scale by same ratio as close adjustment
        raw_c = np.where(raw_close > 0, raw_close, np.nan)
        adj_ratio = np.where(np.isnan(raw_c), 1.0, adj_close / raw_c)
        adj_open = raw_open * adj_ratio

        adjusted_prices[ticker] = pd.Series(adj_close, index=g["DlyCalDt"].values)
        adjusted_opens[ticker] = pd.Series(adj_open, index=g["DlyCalDt"].values)

    # Pivot to wide format: date × ticker
    prices = pd.DataFrame(adjusted_prices)
    open_prices = pd.DataFrame(adjusted_opens)
    volumes = df.pivot(index="DlyCalDt", columns="ticker", values="DlyVol")

    prices.index.name = "date"
    open_prices.index.name = "date"
    volumes.index.name = "date"

    log.info("  Price matrix: %d dates × %d tickers", *prices.shape)
    return prices, open_prices, volumes


# ─── Delisting Returns ───────────────────────────────────────────────────────

def load_delistings(wrds_dir: Path = WRDS_DIR) -> pd.DataFrame:
    """Load delisting returns.

    Returns DataFrame with columns: PERMNO, DelistingDt, DelRet, DelReasonType
    """
    df = pd.read_parquet(
        wrds_dir / "crsp_delistings.parquet",
        columns=["PERMNO", "DelistingDt", "DelRet", "DelReasonType"],
    )
    df["DelistingDt"] = pd.to_datetime(df["DelistingDt"])
    df["PERMNO"] = df["PERMNO"].astype(int)
    df = df[df["DelRet"].notna()].copy()
    log.info("Loaded %d delistings with returns", len(df))
    return df


# ─── Compustat Fundamentals ──────────────────────────────────────────────────

def load_compustat_quarterly(
    permnos: Optional[set[int]] = None,
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    wrds_dir: Path = WRDS_DIR,
) -> pd.DataFrame:
    """Load Compustat quarterly fundamentals for given PERMNOs.

    Returns DataFrame with key financial columns, filtered to primary links
    and point-in-time using rdq (report date of quarterly earnings).

    Columns returned:
        LPERMNO, datadate, rdq, tic, conm,
        saleq, niq, epsfxq, atq, seqq, dlttq, dlcq, cheq, cogsq,
        oibdpq, cshoq, revtq, xsgaq, txtq, dpq
    """
    key_cols = [
        "GVKEY", "LPERMNO", "LPERMCO", "LINKPRIM", "LINKTYPE",
        "tic", "conm", "datadate", "rdq", "fyearq", "fqtr",
        # Income statement
        "saleq", "revtq", "niq", "ibq", "epsfxq", "epspxq",
        "cogsq", "xsgaq", "oibdpq", "oiadpq",
        # Balance sheet
        "atq", "actq", "seqq", "ceqq", "ltq", "lctq",
        "dlttq", "dlcq", "cheq", "invtq", "rectq",
        # Cash flow
        "dpq", "txtq",
        # Shares
        "cshoq",
        # Valuation
        "prccq", "mkvaltq",
    ]

    # Only read columns that exist in the file
    schema = pq.read_schema(wrds_dir / "compustat_quarterly.parquet")
    available = {schema.field(i).name for i in range(len(schema))}
    cols_to_read = [c for c in key_cols if c in available]

    df = pd.read_parquet(
        wrds_dir / "compustat_quarterly.parquet",
        columns=cols_to_read,
    )

    # Filter to primary links
    df = df[
        (df["LINKPRIM"] == "P") &
        (df["LINKTYPE"].isin(["LC", "LU"]))
    ].copy()

    df["LPERMNO"] = df["LPERMNO"].astype(int)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["rdq"] = pd.to_datetime(df["rdq"])

    # Date filter
    df = df[(df["datadate"] >= start) & (df["datadate"] <= end)]

    # PERMNO filter
    if permnos is not None:
        df = df[df["LPERMNO"].isin(permnos)]

    # For rows missing rdq, estimate as datadate + 45 days (median lag is 33)
    rdq_missing = df["rdq"].isna()
    if rdq_missing.any():
        df.loc[rdq_missing, "rdq"] = df.loc[rdq_missing, "datadate"] + pd.Timedelta(days=45)
        log.info("  Estimated rdq for %d/%d rows (%.1f%%)",
                 rdq_missing.sum(), len(df), rdq_missing.mean() * 100)

    # Sort by PERMNO + datadate for time-series operations
    df = df.sort_values(["LPERMNO", "datadate"]).reset_index(drop=True)

    log.info("Loaded %d Compustat quarterly rows for %d PERMNOs",
             len(df), df["LPERMNO"].nunique())
    return df


# ─── Fama-French Factors ─────────────────────────────────────────────────────

def load_fama_french(wrds_dir: Path = WRDS_DIR) -> pd.DataFrame:
    """Load Fama-French 5 factors + momentum daily."""
    df = pd.read_parquet(wrds_dir / "fama_french_5factors_momentum_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ─── FRED Interest Rates ─────────────────────────────────────────────────────

def load_fred_rates(wrds_dir: Path = WRDS_DIR) -> pd.DataFrame:
    """Load FRED interest rates and credit spreads."""
    df = pd.read_parquet(wrds_dir / "fred_interest_rates_spreads_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ─── Main Provider Class ─────────────────────────────────────────────────────

class WRDSDataProvider:
    """Unified interface to all WRDS data.

    Lazily loads components on first access to avoid loading everything upfront.
    """

    def __init__(self, wrds_dir: Path = WRDS_DIR):
        self.wrds_dir = wrds_dir
        self._mapper: Optional[TickerMapper] = None
        self._sp500: Optional[SP500Membership] = None
        self._ff: Optional[pd.DataFrame] = None
        self._fred: Optional[pd.DataFrame] = None

    @property
    def mapper(self) -> TickerMapper:
        if self._mapper is None:
            self._mapper = TickerMapper(self.wrds_dir)
        return self._mapper

    @property
    def sp500(self) -> SP500Membership:
        if self._sp500 is None:
            self._sp500 = SP500Membership(self.wrds_dir)
        return self._sp500

    @property
    def fama_french(self) -> pd.DataFrame:
        if self._ff is None:
            self._ff = load_fama_french(self.wrds_dir)
        return self._ff

    @property
    def fred_rates(self) -> pd.DataFrame:
        if self._fred is None:
            self._fred = load_fred_rates(self.wrds_dir)
        return self._fred

    def get_sp500_members(self, dt) -> set[str]:
        """Get SP500 ticker set on a date (string or date object)."""
        if isinstance(dt, str):
            dt = datetime.strptime(dt, "%Y-%m-%d").date()
        elif isinstance(dt, datetime):
            dt = dt.date()
        elif isinstance(dt, pd.Timestamp):
            dt = dt.date()
        return self.sp500.get_members_ticker(dt)

    def load_prices(self, start="2015-01-01", end="2025-12-31"):
        """Load SP500 price matrices. Returns (close, open, volume) DataFrames."""
        return load_sp500_prices(start, end, self.sp500, self.wrds_dir)

    def load_fundamentals(self, permnos=None, start="2015-01-01", end="2025-12-31"):
        """Load Compustat quarterly fundamentals."""
        if permnos is None:
            permnos = self.sp500.get_all_permnos()
        return load_compustat_quarterly(permnos, start, end, self.wrds_dir)

    def load_delistings(self):
        """Load delisting returns."""
        return load_delistings(self.wrds_dir)


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    wp = WRDSDataProvider()

    # Test SP500 membership
    members = wp.get_sp500_members("2024-01-02")
    print(f"\nSP500 on 2024-01-02: {len(members)} members")
    print(f"  AAPL in SP500: {'AAPL' in members}")
    print(f"  Sample: {sorted(members)[:10]}")

    # Test ticker mapper
    m = wp.mapper
    dt = date(2024, 1, 2)
    for ticker in ["AAPL", "META", "TSLA"]:
        permno = m.ticker_to_permno(ticker, dt)
        back = m.permno_to_ticker(permno, dt) if permno else "N/A"
        print(f"  {ticker} → PERMNO {permno} → {back}")

    # Test FB→META transition
    fb_2021 = m.ticker_to_permno("FB", date(2021, 1, 1))
    meta_2023 = m.ticker_to_permno("META", date(2023, 1, 1))
    print(f"  FB(2021) → PERMNO {fb_2021}")
    print(f"  META(2023) → PERMNO {meta_2023}")
    print(f"  Same security: {fb_2021 == meta_2023}")

    print("\nDone.")
