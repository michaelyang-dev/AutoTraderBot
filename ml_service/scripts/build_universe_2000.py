#!/usr/bin/env python3
"""
Build SP1500 Universe Pickle from 2000 onwards
===============================================
Uses raw WRDS CRSP + Compustat + IBES data to build a complete
backtesting universe with point-in-time membership, features, and
enhanced data. Same format as complete_sp1500_universe.pkl but
starting from 2000 instead of 2016.

Output: data/wrds/sp1500_universe_2000.pkl
"""
import os
import re
import sys
import time
import pickle
import logging

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_universe")

WRDS = Path("data/wrds")
# Date range is overridable so this ONE audited builder can produce BOTH canonical universes
# (they must share identical construction, or their numbers are not comparable):
#   26yr:  START=2000-01-01  -> data/wrds/sp1500_universe_2000.pkl
#    8yr:  START=2016-06-01  -> data/wrds/complete_sp1500_universe.pkl
START = os.environ.get("BUILD_UNIVERSE_START", "2000-01-01")
END = os.environ.get("BUILD_UNIVERSE_END", "2025-12-31")
# Output path is overridable so a rebuild can be written side-by-side and diffed against the
# incumbent before anything replaces it:  BUILD_UNIVERSE_OUT=/path/to.pkl python3 build_universe_2000.py
OUTPUT = Path(os.environ.get("BUILD_UNIVERSE_OUT", str(WRDS / "sp1500_universe_2000.pkl")))


def load_crsp_prices():
    """Load CRSP daily prices as a PERMNO-keyed total-return matrix, then expose each series
    under the membership ticker convention.

    ⚠️ WHY PERMNO (rewritten 2026-07-27): the previous version grouped prices by CRSP's
    `Ticker` column, which is the ticker AS OF THAT ROW'S DATE. Two failure modes followed,
    both measured on the ticker-keyed build:
      1. TICKER CHANGES — a company that renamed (AEOS->AEO, and ~8,261 PERMNOs changed ticker
         at least once) had its history split across two columns, so its pre-rename prices were
         invisible under the modern ticker the membership file uses. This is why index-member
         coverage ramped 82%(2001)->99%(2025) and 88.5%(2016)->99.3%(2025): the further back
         you looked, the more members had no reachable price history. Departure rate among the
         missing was 73% vs 60% among the present => residual SURVIVORSHIP skew.
      2. TICKER REUSE — 940 of 3,642 series (25.8%) had >1 PERMNO active in-window, so one
         column spliced two different companies (AA = Alcoa 2000-2016 then Arconic 2016-2025).
         Momentum computed across such a splice is meaningless.
    PERMNO is CRSP's permanent security identifier: stable across renames, never reused. We
    build one series per PERMNO, then map it to the membership symbol so the sleeves can find
    it. Where several PERMNOs claim the same membership symbol we keep the one with the most
    overlap with that symbol's own ticker history (the reuse case), so no splicing occurs.
    """
    log.info("Loading CRSP daily stock data (PERMNO-keyed)...")
    t0 = time.time()
    cols = ["PERMNO", "Ticker", "DlyCalDt", "DlyRet", "DlyPrc"]
    df = pd.read_parquet(WRDS / "crsp_daily_stock_full.parquet", columns=cols)
    log.info("  Raw CRSP: %d rows (%.1fs)" % (len(df), time.time() - t0))

    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
    df = df.dropna(subset=["DlyCalDt", "DlyPrc", "PERMNO"])
    df = df[(df["DlyCalDt"] >= START) & (df["DlyCalDt"] <= END)]
    log.info("  After date/price filter: %d rows" % len(df))

    # --- membership symbols we need (suffix + bankruptcy-Q normalised, as before) ---
    _strip = lambda s: re.sub(r"-\d+$", "", str(s))
    wanted = set()
    for mem_file in ["sp500_membership_history.parquet", "sp400_membership_history.parquet",
                     "sp600_membership_history.parquet"]:
        mf = WRDS / mem_file
        if mf.exists():
            mdf = pd.read_parquet(mf, columns=["symbol"])
            for s in mdf["symbol"].dropna().unique():
                b = _strip(s)
                wanted.add(b)
                if len(b) == 5 and b.endswith("Q"):
                    wanted.add(b[:-1])
    ETFS = ["SPY", "GLD", "VIXM", "SH", "XLK", "XLF", "XLE", "XLV", "XLI",
            "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]
    wanted.update(ETFS)
    log.info("  membership symbols wanted: %d" % len(wanted))

    # --- PERMNO -> which wanted symbol does it represent, and for how many days? ---
    df["tick"] = df["Ticker"].astype(str)
    cand = df[df["tick"].isin(wanted)]
    pair = cand.groupby(["PERMNO", "tick"]).size().reset_index(name="days")
    # a PERMNO maps to the symbol it traded under most (handles mid-window renames);
    # a symbol claimed by several PERMNOs goes to the PERMNO with the most days (no splice).
    pair = pair.sort_values("days", ascending=False)
    permno_to_sym, sym_to_permno = {}, {}
    for _, r in pair.iterrows():
        p, s = int(r["PERMNO"]), r["tick"]
        if p in permno_to_sym or s in sym_to_permno:
            continue
        permno_to_sym[p] = s
        sym_to_permno[s] = p
    log.info("  resolved %d PERMNO<->symbol pairs (1:1, no splicing)" % len(permno_to_sym))

    keep = df[df["PERMNO"].isin(permno_to_sym)]
    log.info("  rows for resolved securities: %d" % len(keep))

    # --- total-return back-adjusted series per PERMNO (uses the FULL history of that
    #     security, including dates when it traded under a different ticker) ---
    log.info("  Building price matrix...")
    t1 = time.time()
    prices, count = {}, 0
    for permno, g in keep.groupby("PERMNO"):
        sym = permno_to_sym[int(permno)]
        g = g.sort_values("DlyCalDt").drop_duplicates(subset=["DlyCalDt"], keep="last")
        ret = pd.to_numeric(g["DlyRet"], errors="coerce").values
        raw = pd.to_numeric(g["DlyPrc"], errors="coerce").abs().values
        # CRSP writes DlyPrc = 0 to mean "no valid price that day" — it is NOT a real quote.
        # Treat it as missing, otherwise anchoring the back-adjustment on a 0 propagates zero
        # through the whole series (measured: 666,293 zero cells / 621 columns, some entirely
        # zero, when these securities were first included by the delisting fix).
        raw = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan)
        n_ = len(ret)
        pos = np.flatnonzero(np.isfinite(raw))
        if n_ == 0 or pos.size == 0:
            continue
        anchor = pos[-1]                      # last VALID price anchors the total-return chain
        adj = np.full(n_, np.nan)
        adj[anchor] = raw[anchor]
        for i in range(anchor + 1, n_):       # tail after the last valid quote: carry by return
            r = ret[i]
            adj[i] = adj[i - 1] * (1.0 + r) if (np.isfinite(r) and r > -1.0) else np.nan
        for i in range(anchor - 1, -1, -1):
            r = ret[i + 1]
            if np.isfinite(r) and r != -1.0:
                adj[i] = adj[i + 1] / (1.0 + r)
            else:
                # ⚠️ DELISTING-LOSS FIX (2026-07-27). CRSP leaves DlyRet NaN on the final
                # delisting row, and the old code then did `adj[i] = adj[i+1]`, i.e. treated
                # that day as a ZERO return — silently deleting the collapse. Measured on SVB
                # (PERMNO 11786): 2023-03-10 close $39.37 -> 2023-03-13 delist $0.40, a -99%
                # move that simply vanished, so a held bankruptcy realised about -86% instead
                # of ~-100%. Fall back to the RAW PRICE RATIO, which recovers the true move.
                pr_next, pr_cur = raw[i + 1], raw[i]
                if np.isfinite(pr_next) and np.isfinite(pr_cur) and pr_next > 0 and pr_cur > 0:
                    adj[i] = adj[i + 1] * (pr_cur / pr_next)
                else:
                    adj[i] = adj[i + 1]
        prices[sym] = pd.Series(adj, index=g["DlyCalDt"].values)
        count += 1
        if count % 1000 == 0:
            log.info("  Processed %d/%d securities..." % (count, len(permno_to_sym)))

    prices_df = pd.DataFrame(prices)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()
    log.info("  Price matrix: %s (%.1fs)" % (str(prices_df.shape), time.time() - t1))
    return prices_df, keep, permno_to_sym


def load_sp1500_membership():
    """Load point-in-time SP500/SP400/SP600 membership."""
    log.info("Loading SP1500 membership history...")

    sp500_mem = {}
    sp400_mem = {}
    sp600_mem = {}

    for name, filename, mem_dict in [
        ("SP500", "sp500_membership_history.parquet", sp500_mem),
        ("SP400", "sp400_membership_history.parquet", sp400_mem),
        ("SP600", "sp600_membership_history.parquet", sp600_mem),
    ]:
        f = WRDS / filename
        if not f.exists():
            log.warning("  %s not found" % filename)
            continue

        df = pd.read_parquet(f)
        log.info("  %s: %d rows, cols=%s, index=%s" % (name, len(df), list(df.columns), df.index.name))

        # The membership files have date as INDEX, "Index Constituent" as flag, "symbol" as ticker.
        # ⚠️ CRITICAL: a row is NOT proof of membership — every (date, ticker) pair carries an
        # "Index Constituent" flag that is 1 (in the index that day) or 0 (NOT in it). The
        # original code here assumed "each row = one date-ticker pair where the ticker is in
        # the index" and grouped ALL symbols per date, which silently pulled in every 0-flag
        # row. Effect at 2001-01-02: SP500 866 / SP400 1136 / SP600 1508 = 3510 names instead
        # of the true 500/400/600 = 1500 (2.3x too many), and it is LOOK-AHEAD — 99.8% of the
        # spurious names have 0-flag rows BEFORE they ever joined the index, so the backtest
        # could hold tomorrow's index members today. FIXED 2026-07-26; filtering flag==1
        # reproduces the true index sizes exactly. The WRDS data was always correct.
        # Some tickers have suffixes like "AAI-199908" — strip the suffix.
        if "symbol" in df.columns:
            # Reset index to get date as column
            df = df.reset_index()
            date_col = df.columns[0]  # First column after reset is the date index
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])

            # Filter date range
            df = df[(df[date_col] >= pd.Timestamp(START)) & (df[date_col] <= pd.Timestamp(END))]

            # THE FIX: keep only rows actually flagged as index constituents.
            _flag = next((c for c in df.columns if c.strip().lower() == "index constituent"), None)
            if _flag is None:
                raise RuntimeError(
                    f"{filename}: no 'Index Constituent' column — refusing to build a "
                    f"membership map without it (that produced the 2.3x look-ahead universe)."
                )
            _before = df["symbol"].nunique()
            df = df[pd.to_numeric(df[_flag], errors="coerce") == 1]
            log.info("  %s: constituent filter %d -> %d unique symbols" %
                     (name, _before, df["symbol"].nunique()))

            # Clean ticker: strip suffixes like "-199908"
            df["clean_sym"] = df["symbol"].str.replace(r'-\d+$', '', regex=True)
            # ...and normalise bankruptcy tickers to their live root (SIVBQ -> SIVB) so the
            # MEMBERSHIP map uses the same convention as the CRSP price matrix. Without this
            # the price fix above is useless: get_sp1500() would hand the sleeves "SIVBQ",
            # which has no price series, so the name stays untradeable and the survivorship
            # hole remains. Only 5-char symbols ending in Q (the standard convention).
            _q = df["clean_sym"].str.len().eq(5) & df["clean_sym"].str.endswith("Q")
            df.loc[_q, "clean_sym"] = df.loc[_q, "clean_sym"].str[:-1]

            # Group by date — each date's members are all symbols present on that date
            for date, grp in df.groupby(date_col):
                mem_dict[date] = set(grp["clean_sym"].dropna().unique())

        log.info("  %s: %d dates loaded" % (name, len(mem_dict)))

    return sp500_mem, sp400_mem, sp600_mem


def compute_features(prices_df):
    """Compute features_by_date dict from price matrix."""
    log.info("Computing features...")
    t0 = time.time()

    trading_dates = prices_df.index
    features_by_date = {}

    # Precompute returns and indicators for ALL tickers at once (vectorized)
    log.info("  Computing returns (vectorized)...")
    ret_5d = prices_df.pct_change(5, fill_method=None)
    ret_10d = prices_df.pct_change(10, fill_method=None)
    ret_20d = prices_df.pct_change(20, fill_method=None)
    ret_60d = prices_df.pct_change(60, fill_method=None)
    ret_120d = prices_df.pct_change(120, fill_method=None)
    ret_126d = prices_df.pct_change(126, fill_method=None)
    ret_252d = prices_df.pct_change(252, fill_method=None)

    log.info("  Computing volatility...")
    daily_ret = prices_df.pct_change(fill_method=None)
    vol_10d = daily_ret.rolling(10).std() * np.sqrt(252)
    vol_20d = daily_ret.rolling(20).std() * np.sqrt(252)
    vol_60d = daily_ret.rolling(60).std() * np.sqrt(252)

    log.info("  Computing SMAs...")
    sma50 = prices_df.rolling(50).mean()
    sma200 = prices_df.rolling(200).mean()
    dist_sma50 = (prices_df - sma50) / sma50
    dist_sma200 = (prices_df - sma200) / sma200

    log.info("  Computing MACD...")
    ema12 = prices_df.ewm(span=12).mean()
    ema26 = prices_df.ewm(span=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()

    log.info("  Computing RSI...")
    delta = prices_df.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))

    log.info("  Building features_by_date dict (vectorized)...")
    # Stack all feature matrices into one dict per date
    feat_names = ["ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d", "ret_126d", "ret_252d",
                  "vol_10d", "vol_20d", "vol_60d", "dist_sma50", "dist_sma200", "rsi_14",
                  "macd_line", "macd_signal"]
    feat_dfs = [ret_5d, ret_10d, ret_20d, ret_60d, ret_120d, ret_126d, ret_252d,
                vol_10d, vol_20d, vol_60d, dist_sma50, dist_sma200, rsi_14,
                macd_line, macd_signal]

    valid_dates = [d for d in trading_dates if d >= pd.Timestamp(START)]
    log.info("    Processing %d dates..." % len(valid_dates))

    for i, date in enumerate(valid_dates):
        if i % 500 == 0:
            log.info("    Date %d/%d: %s" % (i, len(valid_dates), date.date()))

        # Get all non-null tickers for this date (vectorized)
        px_row = prices_df.loc[date]
        valid_mask = px_row.notna() & (px_row > 0)
        valid_syms = valid_mask[valid_mask].index

        # Check ret_20d is not NaN (basic requirement)
        r20_row = ret_20d.loc[date]
        r20_valid = r20_row[valid_syms].notna()
        valid_syms = r20_valid[r20_valid].index

        if len(valid_syms) == 0:
            features_by_date[date] = {}
            continue

        date_feats = {}
        for sym in valid_syms:
            feats = {}
            for fname, fdf in zip(feat_names, feat_dfs):
                val = fdf.loc[date, sym]
                feats[fname] = val if not pd.isna(val) else np.nan
            date_feats[sym] = feats

        features_by_date[date] = date_feats

    log.info("  Features computed: %d dates (%.1fs)" % (len(features_by_date), time.time() - t0))
    return features_by_date


def load_fundamentals():
    """Load Compustat fundamentals and compute ROE, margins, etc."""
    log.info("Loading Compustat fundamentals...")
    fund = pd.read_parquet(WRDS / "compustat_quarterly.parquet")
    fund["datadate"] = pd.to_datetime(fund["datadate"])

    # Point-in-time: use rdq (report date) if available
    if "rdq" in fund.columns:
        fund["avail_date"] = pd.to_datetime(fund["rdq"], errors="coerce")
        fund["avail_date"] = fund["avail_date"].fillna(fund["datadate"] + pd.Timedelta(days=90))
    else:
        fund["avail_date"] = fund["datadate"] + pd.Timedelta(days=90)

    fund = fund.sort_values(["tic", "avail_date"])
    log.info("  Compustat: %d rows, %d tickers" % (len(fund), fund["tic"].nunique()))
    return fund


def main():
    log.info("=" * 60)
    log.info("BUILDING SP1500 UNIVERSE FROM 2000")
    log.info("=" * 60)
    t_total = time.time()

    # 1. Load prices
    prices_df, crsp_raw, permno_to_sym = load_crsp_prices()
    if prices_df is None:
        log.error("Failed to load prices")
        return

    # 2. Load membership
    sp500_mem, sp400_mem, sp600_mem = load_sp1500_membership()

    # 3. Compute features
    features_by_date = compute_features(prices_df)

    # 4. Load fundamentals (for value strategy)
    fund = load_fundamentals()

    # ⚠️ FUNDAMENTALS PERMNO JOIN (2026-07-27). Compustat rows were previously matched to the
    # price matrix by ticker (`tic`), which suffers the same rename/reuse failure as prices did:
    # measured fundamentals coverage ramped 66%(2001) -> 88%(2025), i.e. the further back you
    # looked the more names silently lost roe/gross_margin. The VALUE sleeve is 35% of the book
    # and REQUIRES both, so early-period value picks were drawn from a shrunken, survivor-skewed
    # subset. compustat_quarterly carries LPERMNO (100% populated), so we relabel each Compustat
    # row with the universe symbol its PERMNO maps to; everything downstream stays keyed by
    # symbol and is unchanged. Rows with no PERMNO match keep their original ticker.
    if "LPERMNO" in fund.columns:
        _mapped = fund["LPERMNO"].map(permno_to_sym)
        _n_before = fund["tic"].isin(set(permno_to_sym.values())).sum()
        fund["tic"] = _mapped.fillna(fund["tic"])
        log.info("  fundamentals relabelled via LPERMNO: %d/%d rows now carry a universe symbol "
                 "(was %d by ticker)" % (_mapped.notna().sum(), len(fund), _n_before))

    # Compute fundamental features per ticker per quarter
    # and merge into features_by_date using point-in-time
    log.info("Merging fundamentals into features (point-in-time)...")
    t1 = time.time()

    # Build lookup: for each ticker, sorted list of (avail_date, fundamentals)
    # Include ALL features that the old pickle has
    fund_lookup = {}
    for tic, grp in fund.groupby("tic"):
        records = []
        prev_sale = np.nan
        prev_eps = np.nan
        prev_atq = np.nan
        prev_ceq = np.nan
        prev_sstk = np.nan
        prev_prstkc = np.nan
        for _, row in grp.sort_values("avail_date").iterrows():
            ad = row["avail_date"]
            if pd.isna(ad):
                continue
            # Use SAME Compustat fields as wrds_universe.py (seqq, oibdpq, epsfxq)
            seqq = row.get("seqq", np.nan)  # stockholders' equity (NOT ceqq)
            niq = row.get("niq", np.nan)
            saleq = row.get("saleq", np.nan)
            cogsq = row.get("cogsq", np.nan)
            dlttq = row.get("dlttq", np.nan)
            dlcq = row.get("dlcq", np.nan)
            oibdpq = row.get("oibdpq", np.nan)  # operating income BEFORE depreciation
            atq = row.get("atq", np.nan)  # total assets
            gpq = row.get("gpq", np.nan)  # gross profit
            epsfxq = row.get("epsfxq", np.nan)  # diluted EPS (NOT epspiq)

            # Core ratios (match wrds_universe.py exactly)
            roe = (niq / seqq * 4) if not pd.isna(seqq) and seqq > 0 and not pd.isna(niq) else np.nan
            gm = ((saleq - cogsq) / saleq) if not pd.isna(saleq) and saleq > 0 and not pd.isna(cogsq) else np.nan
            de = ((float(dlttq if not pd.isna(dlttq) else 0) + float(dlcq if not pd.isna(dlcq) else 0)) / seqq) if not pd.isna(seqq) and seqq > 0 else np.nan

            # Operating margin = oibdpq / saleq (before depreciation, matches old pickle)
            op_margin = (oibdpq / saleq) if not pd.isna(oibdpq) and not pd.isna(saleq) and saleq > 0 else np.nan
            net_margin = (niq / saleq) if not pd.isna(niq) and not pd.isna(saleq) and saleq > 0 else np.nan

            # Growth (YoY — compare to 4 quarters ago, using epsfxq like old pickle)
            rev_growth = ((saleq / prev_sale) - 1) if not pd.isna(saleq) and not pd.isna(prev_sale) and prev_sale > 0 else np.nan
            eps_growth = ((epsfxq - prev_eps) / abs(prev_eps)) if not pd.isna(epsfxq) and not pd.isna(prev_eps) and abs(prev_eps) > 0.01 else np.nan

            # Asset growth
            asset_growth = ((atq / prev_atq) - 1) if not pd.isna(atq) and not pd.isna(prev_atq) and prev_atq > 0 else np.nan

            # GP/Assets (profitability factor)
            gp_assets = (float(gpq if not pd.isna(gpq) else (saleq - cogsq if not pd.isna(saleq) and not pd.isna(cogsq) else np.nan)) / atq) if not pd.isna(atq) and atq > 0 else np.nan

            # Net issuance (equity change — proxy for buybacks vs issuance)
            net_iss = np.nan
            if not pd.isna(seqq) and not pd.isna(prev_ceq) and prev_ceq > 0:
                net_iss = (seqq - prev_ceq) / prev_ceq

            fdata = {
                "roe": roe, "gross_margin": gm, "debt_to_equity": de,
                "operating_margin": op_margin, "net_margin": net_margin,
                "revenue_growth_yoy": rev_growth, "eps_growth_yoy": eps_growth,
                "asset_growth": asset_growth, "gp_assets": gp_assets,
                "net_issuance": net_iss,
            }

            records.append((ad, fdata))

            # Store for next quarter's growth calc (lag by 4 for YoY)
            if len(records) >= 4:
                prev_sale = records[-4][1].get("_saleq", np.nan)
                prev_eps = records[-4][1].get("_epsfxq", np.nan)
                prev_atq = records[-4][1].get("_atq", np.nan)
                prev_ceq = records[-4][1].get("_seqq", np.nan)
            # Store raw values for future YoY computation
            fdata["_saleq"] = saleq
            fdata["_epsfxq"] = epsfxq
            fdata["_atq"] = atq
            fdata["_seqq"] = seqq

        if records:
            fund_lookup[tic] = sorted(records, key=lambda x: x[0])

    # For each date in features_by_date, look up latest available fundamentals
    dates = sorted(features_by_date.keys())
    for i, date in enumerate(dates):
        if i % 500 == 0:
            log.info("  Fundamentals merge %d/%d: %s" % (i, len(dates), date.date()))

        for sym, feats in features_by_date[date].items():
            if sym in fund_lookup:
                # Binary search for latest available before date
                records = fund_lookup[sym]
                latest = None
                for ad, fdata in records:
                    if ad <= date:
                        latest = fdata
                    else:
                        break
                if latest:
                    feats.update(latest)

    log.info("  Fundamentals merged (%.1fs)" % (time.time() - t1))

    # 5. Build enhanced data from Compustat (matching wrds_universe.py logic)
    log.info("Building enhanced data from Compustat...")
    fin_growth = {}
    ev_data = {}
    estimates_data = {}
    earnings_signals = {}
    revenue_surprise = {}
    beat_streak = {}
    price_targets = {}

    for tic, grp in fund.groupby("tic"):
        cq_sorted = grp.sort_values("datadate")
        if len(cq_sorted) < 5:
            continue

        # fin_growth, ev_data, estimates_data: INTENTIONALLY LEFT EMPTY
        # These are snapshot-based (use latest quarter for ALL dates) which
        # introduces look-ahead bias. The old pickle had the same issue but
        # was built with data only through Q3 2024. Building with 2026 data
        # gives the strategy future information.
        #
        # Only earnings_signals below are properly point-in-time (date-keyed).

        # earnings_signals: per-date EPS/revenue surprise + beat streak
        beat_count = 0
        for i in range(4, len(cq_sorted)):
            row = cq_sorted.iloc[i]
            prev = cq_sorted.iloc[i - 4]
            rdq = row.get("rdq")
            if pd.isna(rdq):
                continue

            eps_cur = row.get("epsfxq")
            eps_prev = prev.get("epsfxq")
            rev_cur = row.get("saleq")
            rev_prev = prev.get("saleq")

            eps_surp = np.nan
            rev_surp = np.nan
            beat = False

            if pd.notna(eps_cur) and pd.notna(eps_prev) and abs(eps_prev) > 0.01:
                eps_surp = (eps_cur - eps_prev) / abs(eps_prev)
                beat = eps_cur > eps_prev
            if pd.notna(rev_cur) and pd.notna(rev_prev) and rev_prev > 0:
                rev_surp = (rev_cur - rev_prev) / rev_prev

            if beat:
                beat_count += 1
            else:
                beat_count = 0

            rdq_ts = pd.Timestamp(rdq)
            if rdq_ts not in earnings_signals:
                earnings_signals[rdq_ts] = {}
            earnings_signals[rdq_ts][tic] = {
                "eps_surprise": eps_surp,
                "rev_surprise": rev_surp,
                "beat_streak": beat_count,
            }

            # Also store latest revenue_surprise and beat_streak per ticker
            if not np.isnan(rev_surp):
                revenue_surprise[tic] = rev_surp
            beat_streak[tic] = beat_count

    log.info("  Enhanced data: %d fin_growth, %d ev_data, %d estimates, %d earnings dates" %
             (len(fin_growth), len(ev_data), len(estimates_data), len(earnings_signals)))

    # 6. Load Fama-French
    log.info("Loading Fama-French factors...")
    ff = pd.read_parquet(WRDS / "fama_french_5factors_momentum_daily.parquet")
    ff["date"] = pd.to_datetime(ff["date"])
    ff = ff.set_index("date").sort_index()

    # 7. Load FRED rates
    fred_rates = None
    fred_file = WRDS / "fred_interest_rates_spreads_daily.parquet"
    if fred_file.exists():
        fred_rates = pd.read_parquet(fred_file)
        fred_rates["date"] = pd.to_datetime(fred_rates["date"])
        fred_rates = fred_rates.set_index("date").sort_index()
        log.info("  FRED rates: %d rows" % len(fred_rates))

    # 8. Build ETF prices for GLD, VIXM, SH etc.
    log.info("Building ETF price matrix...")
    etf_syms = ["GLD", "VIXM", "SH", "SPY", "XLK", "XLF", "XLE", "XLV", "XLI",
                "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]
    etf_prices = {}
    for sym in etf_syms:
        if sym in prices_df.columns:
            etf_prices[sym] = prices_df[sym]
    etf_df = pd.DataFrame(etf_prices)
    log.info("  ETF matrix: %s" % str(etf_df.shape))

    # 9. Save pickle
    log.info("Saving pickle to %s..." % OUTPUT)
    data = {
        "prices_df": prices_df,
        "features_by_date": features_by_date,
        "sp500_mem": sp500_mem,
        "sp400_mem": sp400_mem,
        "sp600_mem": sp600_mem,
        "fin_growth": fin_growth,
        "ev_data": ev_data,
        "estimates_data": estimates_data,
        "earnings_signals": earnings_signals,
        "revenue_surprise": revenue_surprise,
        "beat_streak": beat_streak,
        "price_targets": price_targets,
        "fama_french": ff,
        "fred_rates": fred_rates,
        "etf_df": etf_df,
    }

    with open(OUTPUT, "wb") as f:
        pickle.dump(data, f, protocol=4)

    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    log.info("Saved! Size: %.0f MB" % size_mb)
    log.info("Total time: %.0f minutes" % ((time.time() - t_total) / 60))

    # Quick validation
    log.info("\nValidation:")
    log.info("  Prices: %s" % str(prices_df.shape))
    log.info("  Feature dates: %d" % len(features_by_date))
    log.info("  SP500 membership dates: %d" % len(sp500_mem))
    log.info("  SP400 membership dates: %d" % len(sp400_mem))
    log.info("  SP600 membership dates: %d" % len(sp600_mem))
    sample_date = dates[len(dates) // 2]
    log.info("  Sample date %s: %d tickers with features" % (sample_date.date(), len(features_by_date[sample_date])))


if __name__ == "__main__":
    main()
