"""S&P 1500 membership scraper: SSGA ETF holdings first, Wikipedia fallback, last-good last.

WHY THIS TEST EXISTS (2026-09-30)
  Wikipedia's S&P 600 page had not applied the September 2026 rebalance (15 adds / 11 deletes vs SPSM's
  holdings) and that changed live picks (ATRC/AXTI vs LGND/VICR). The scraper now reads SPY/MDY/SPSM
  holdings with a standard-library xlsx parser (the box has no openpyxl). No network: every fetch is stubbed.
Run: python3 tests/test_sp1500_scrape.py
"""
import io
import itertools
import json
import os
import string
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scrape_sp1500 as S  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def make_xlsx(rows, inline_first=False):
    """Minimal SSGA-shaped xlsx built with zipfile (shared strings; optionally one inline string)."""
    shared, idx = [], {}

    def sid(s):
        if s not in idx:
            idx[s] = len(shared); shared.append(s)
        return idx[s]
    xml_rows = []
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, val in enumerate(row):
            if val is None:
                continue
            ref = f"{string.ascii_uppercase[c]}{r}"
            if inline_first and r == 7 and c == 1:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(val)}</t></is></c>')
            else:
                cells.append(f'<c r="{ref}" t="s"><v>{sid(val)}</v></c>')
        xml_rows.append(f'<row r="{r}">{"".join(cells)}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sheet = f'<?xml version="1.0"?><worksheet {ns}><sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    sst = f'<?xml version="1.0"?><sst {ns}>' + "".join(f"<si><t>{escape(s)}</t></si>" for s in shared) + "</sst>"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)
    return buf.getvalue()


print("SSGA holdings parser")
rows = [["Fund Name:", "SPDR Portfolio S&P 600"], ["Ticker Symbol:", "SPSM"], ["Holdings:", "As of 28-Sep-2026"], [],
        ["Name", "Ticker", "Identifier", "Weight"],
        ["FORMFACTOR INC", "FORM", "x", "0.5"],
        ["PATHWARD FINANCIAL INC", "CASH", "x", "0.1"],          # a REAL ticker named CASH
        ["RUSH ENTERPRISES INC CL B", "RUSHB", "x", "0.1"],
        ["MOOG INC CLASS A", "MOG.A", "x", "0.1"],
        ["BERKSHIRE HATHAWAY INC CL B", "BRK/B", "x", "0.1"],
        ["SSI US GOV MONEY MARKET CLASS", "-", "x", "0.2"],
        ["US DOLLAR", "-", "x", "0.0"],
        ["OMNIAB INC 12.5 EARNOUT", "2200963D", "x", "0.0"],
        ["E-MINI RUSS 2000 DEC26", "RTYZ6", "x", "0.0"]]
t, asof = S._ssga_constituents(make_xlsx(rows, inline_first=False))
check("As-of text read from the header block", asof == "As of 28-Sep-2026", asof)
check("equities kept incl. CASH (Pathward), RUSHB, MOG.A; BRK/B normalised to BRK.B",
      t == sorted(["FORM", "CASH", "RUSHB", "MOG.A", "BRK.B"]), t)
check("cash / money-market / earn-out / futures lines dropped", not {"-", "2200963D", "RTYZ6", "US DOLLAR"} & set(t))
t2, _ = S._ssga_constituents(make_xlsx(rows, inline_first=True))
check("inline-string cells parse the same", t2 == t, t2)

print("source selection")
letters = ["".join(p) for p in itertools.product(string.ascii_uppercase, repeat=3)]
L = {"sp500": letters[:503], "sp400": letters[1000:1400], "sp600": letters[2000:2601]}
etf_600 = L["sp600"][15:] + letters[3000:3015]                      # 15 adds / 15 deletes: 97.5% overlap
hold = {"spy": L["sp500"], "mdy": L["sp400"], "spsm": etf_600}


def holdings_xlsx(tickers):
    return make_xlsx([["Holdings:", "As of 28-Sep-2026"], [], ["Name", "Ticker"]] + [["N", x] for x in tickers])


def wiki_html(tickers):
    body = "".join(f"<tr><td>{x}</td><td>n</td></tr>" for x in tickers)
    return f'<table id="constituents"><tr><th>Symbol</th></tr>{body}</table>'


def run(wiki_ok=True, ssga=hold, prev=None):
    tmp = Path(tempfile.mkdtemp(prefix="scrape_test_"))
    S.OUTPUT_FILE = tmp / "sp1500_members.json"; S.DATA_DIR = tmp
    if prev:
        S.OUTPUT_FILE.write_text(json.dumps(prev))
    url2name = {v: k for k, v in S.URLS.items()}

    def _fetch(url):
        if not wiki_ok:
            raise IOError("wikipedia down")
        return wiki_html(L[url2name[url]])

    def _fetch_bytes(url):
        etf = url.split("holdings-daily-us-en-")[1].split(".xlsx")[0]
        if ssga is None or etf not in ssga:
            raise IOError("ssga down")
        return holdings_xlsx(ssga[etf])
    S._fetch, S._fetch_bytes = _fetch, _fetch_bytes
    return S.scrape()


d = run()
check("ETF holdings chosen when sane (sp600 = SPSM list incl. its 15 adds)", d["sp600"] == sorted(etf_600) and d["source"]["sp600"].startswith("ssga:SPSM"))
check("sp500 / sp400 from SPY / MDY", d["sp500"] == sorted(L["sp500"]) and d["source"]["sp400"].startswith("ssga:MDY"))
d = run(ssga=None)
check("ETF download fails -> Wikipedia", d["sp600"] == sorted(L["sp600"]) and d["source"]["sp600"] == "wikipedia")
bad = dict(hold); bad["spsm"] = letters[5000:5601]                   # plausible size, ~0% overlap = bad parse
d = run(ssga=bad)
check("ETF list with <90% overlap is rejected -> Wikipedia", d["sp600"] == sorted(L["sp600"]) and d["source"]["sp600"] == "wikipedia")
small = dict(hold); small["mdy"] = L["sp400"][:300]                  # out of the expected count range
d = run(ssga=small)
check("ETF list outside the expected count range is rejected -> Wikipedia", d["source"]["sp400"] == "wikipedia")
prev = {"updated": "2026-09-28 06:00:00", **{k: sorted(v) for k, v in L.items()}}
d = run(wiki_ok=False, ssga=hold, prev=prev)
check("Wikipedia down: ETF list validated against the last-good list and used", d["source"]["sp600"].startswith("ssga:SPSM"))
d = run(wiki_ok=False, ssga=None, prev=prev)
check("both down -> last-good list, marked stale", d["sp600"] == prev["sp600"] and "sp600" in d["stale"])
check("output keeps the three list keys every reader uses", all(isinstance(d[k], list) for k in ("sp500", "sp400", "sp600")))

print(f"\n{len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else " — all sp1500 scrape tests passed"))
sys.exit(1 if FAILS else 0)
