"""
EDGAR OVERLAY RECONCILIATION — measure the REALIZED accuracy of the freshness overlay
against Compustat truth, so "98.9%" stops being a one-time pre-flip holdout and becomes
a standing, self-checking number. Two independent legs:

1. GO-FORWARD (the real proof, per the 2026-07-15 design): every patched value the
   overlay serves is ARCHIVED with its period_end. When a later WRDS Compustat upload
   finally contains that quarter, we diff served-vs-truth. This is the honest measure —
   it scores the ACTUAL extrapolated value we shipped (jump-guard and all), not a
   re-derivation. It starts accumulating the day this runs; nothing to wait for beyond
   the next upload.

2. WALK-FORWARD (no-wait proxy): because we already hold prior quarterly uploads, we can
   replay history now — resolve specs on quarters T-1.. and score the extraction of each
   held-out quarter T against the Compustat value we already own. Same engine as
   gate_calibration_test.py; summarized here for the standing report so one artifact
   carries both the realized and the replayed accuracy.

Writes data/edgar_reconcile_report.json (+ archives to data/edgar_overlay_archive/).
Alerts Telegram if realized roe accuracy drops below ROE_ALERT_ACC. Never mutates the
overlay or any live path. Cron: right after the daily patch.
  40 18 * * 1-5  (patch) ; 55 18 * * 1-5  venv/bin/python scripts/edgar_overlay_reconcile.py
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "scripts"))
from edgar_fundamentals_patch import (  # noqa: E402
    CONCEPTS, DATA, FEATURES, FLOWS, FUND, GATE_DEPTH, ZERO_OK, close_enough,
    fetch_facts, item_anchors, load_cik_map, spec_value)

OVERLAY = DATA / "edgar_feature_overlay.json"
ARCHIVE = DATA / "edgar_overlay_archive"
REPORT = DATA / "edgar_reconcile_report.json"
LEDGER = DATA / "edgar_reconcile_ledger.json"    # every served value once its truth lands
LIVE_FEATURES = ["roe"]                          # only features actually consumed live
ROE_ALERT_ACC = 0.97                             # realized roe accuracy floor -> alert
MATCH_DAYS = 20                                  # period_end <-> datadate tolerance (4-4-5)
N_TARGETS = 6                                    # walk-forward holdout quarters per symbol
FEAT_TOL = lambda tv: max(0.02, 0.05 * abs(tv))  # feature-level agreement band


# ── truth lookup: recompute the Compustat feature value from the latest upload ──────────
def _truth_feature(fund_by_tic, sym, feat, period_end):
    """The feature value implied by the CURRENT Compustat upload at the quarter nearest
    period_end (within MATCH_DAYS). None if that quarter is not yet in the upload."""
    g = fund_by_tic.get(sym)
    if g is None:
        return None
    pe = pd.Timestamp(period_end)
    within = g[(g["datadate"] - pe).abs() <= pd.Timedelta(days=MATCH_DAYS)]
    if within.empty:
        return None
    row = within.iloc[(within["datadate"] - pe).abs().argmin()]
    inputs, formula = FEATURES[feat]
    vals = {}
    for it in inputs:
        v = row.get(it)
        vals[it] = 0.0 if (pd.isna(v) and it in ZERO_OK) else v
    if any(pd.isna(vals[it]) for it in inputs):
        return None
    try:
        fv = formula(vals)
    except (ZeroDivisionError, TypeError):
        return None
    return fv if fv is not None and np.isfinite(fv) else None


# ── leg 1: archive today's overlay, then reconcile every archived value whose truth landed
def go_forward(fund_by_tic):
    ARCHIVE.mkdir(exist_ok=True)
    stamp = None
    if OVERLAY.exists():
        cur = json.load(open(OVERLAY))
        stamp = cur.get("generated", "").split()[0] or None
        if stamp:
            dest = ARCHIVE / f"overlay_{stamp}.json"
            if not dest.exists():
                shutil.copy(OVERLAY, dest)

    ledger = json.load(open(LEDGER)) if LEDGER.exists() else {"reconciled": {}}
    done = ledger["reconciled"]                 # key "sym|feat|period_end" -> already scored
    stats = {f: {"scored": 0, "ok": 0, "err_bps": []} for f in LIVE_FEATURES}
    for arch in sorted(ARCHIVE.glob("overlay_*.json")):
        feats = json.load(open(arch)).get("features", {})
        for sym, fmap in feats.items():
            for feat, e in fmap.items():
                if feat not in LIVE_FEATURES:
                    continue
                pe = e.get("period_end")
                key = f"{sym}|{feat}|{pe}"
                if key in done:
                    continue
                truth = _truth_feature(fund_by_tic, sym, feat, pe)
                if truth is None:
                    continue                     # truth not in the upload yet; retry later
                served = float(e["value"])
                ok = abs(served - truth) <= FEAT_TOL(truth)
                s = stats[feat]
                s["scored"] += 1
                s["ok"] += int(ok)
                s["err_bps"].append(round(1e4 * (served - truth), 1))
                done[key] = {"served": round(served, 6), "truth": round(float(truth), 6),
                             "ok": bool(ok), "archived": arch.name}
    json.dump(ledger, open(LEDGER, "w"), indent=1)
    out = {}
    for feat, s in stats.items():
        n = s["scored"]
        out[feat] = {
            "newly_reconciled": n,
            "cumulative_reconciled": sum(1 for k in done if k.split("|")[1] == feat),
            "accuracy": round(s["ok"] / n, 4) if n else None,
            "median_abs_err_bps": round(float(np.median(np.abs(s["err_bps"]))), 1) if n else None,
        }
    return out


# ── leg 2: walk-forward replay on the history we already own (compact calibration) ──────
def walk_forward(fund_by_tic, cik_map, syms, depth_gate=1):
    res = {f: [0, 0] for f in LIVE_FEATURES}     # [scored, ok]
    for sym in syms:
        g = fund_by_tic.get(sym)
        if g is None or len(g) < 5:
            continue
        cik = cik_map.get(sym.upper().replace(".", "-")) or cik_map.get(sym.upper())
        if cik is None:
            continue
        facts = fetch_facts(cik)
        if facts is None:
            continue
        rows = list(g.itertuples())
        for tgt_i in range(max(4, len(rows) - N_TARGETS), len(rows)):
            target = rows[tgt_i]
            anchors = rows[:tgt_i][::-1]
            resolved = {}
            for feat in LIVE_FEATURES:
                inputs, _ = FEATURES[feat]
                for it in inputs:
                    if it in resolved:
                        continue
                    depth = GATE_DEPTH.get(it, 1)
                    if len(anchors) < depth:
                        resolved[it] = None
                        continue
                    hit = None
                    for spec in CONCEPTS[it]:
                        if all(close_enough(spec_value(facts, spec, a.datadate, it in FLOWS)[0],
                                            getattr(a, it)) for a in anchors[:depth]):
                            hit = spec
                            break
                    if hit is None and it in ZERO_OK:
                        a0 = anchors[0]
                        if pd.isna(getattr(a0, it)) or abs(getattr(a0, it)) <= 2.0:
                            hit = "ZERO"
                    resolved[it] = hit
            for feat in LIVE_FEATURES:
                inputs, formula = FEATURES[feat]
                if any(resolved.get(it) is None for it in inputs):
                    continue
                vals, tvals, ok = {}, {}, True
                for it in inputs:
                    if resolved[it] == "ZERO":
                        vals[it] = 0.0
                    else:
                        v, _ = spec_value(facts, resolved[it], target.datadate, it in FLOWS)
                        if v is None:
                            ok = False
                            break
                        vals[it] = v
                    tv = getattr(target, it)
                    tvals[it] = 0.0 if pd.isna(tv) else tv
                if not ok:
                    continue
                fv, tvv = formula(vals), formula(tvals)
                if fv is None or tvv is None or not np.isfinite(fv) or not np.isfinite(tvv):
                    continue
                res[feat][0] += 1
                res[feat][1] += int(abs(fv - tvv) <= FEAT_TOL(tvv))
    return {f: {"scored": n, "accuracy": round(ok / n, 4) if n else None}
            for f, (n, ok) in res.items()}


def _alert(msg):
    try:
        import requests
        tok = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat = os.environ.get("TELEGRAM_CHAT_ID") or "8746062845"
        if tok:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": msg}, timeout=10)
    except Exception:
        pass


def main():
    t0 = time.time()
    fund = pd.read_parquet(FUND)
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    fund_by_tic = {t: g.sort_values("datadate") for t, g in fund.groupby("tic")}
    members = json.load(open(DATA / "sp1500_members.json"))
    syms = sorted(set(members["sp500"] + members["sp400"] + members["sp600"]))
    cik_map = load_cik_map()

    gf = go_forward(fund_by_tic)
    wf = walk_forward(fund_by_tic, cik_map, syms)

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "compustat_max_datadate": str(fund["datadate"].max().date()),
        "go_forward": gf,        # realized: served value vs the upload that later revealed truth
        "walk_forward": wf,      # replayed: extraction vs Compustat truth we already own
        "runtime_s": round(time.time() - t0, 1),
    }
    json.dump(report, open(REPORT, "w"), indent=1)

    print(f"===== EDGAR OVERLAY RECONCILIATION ({report['runtime_s']}s) =====")
    print(f"  compustat_max_datadate: {report['compustat_max_datadate']}")
    for feat in LIVE_FEATURES:
        g, w = gf[feat], wf[feat]
        print(f"  {feat}: go-forward realized acc={g['accuracy']} "
              f"(n={g['newly_reconciled']} new / {g['cumulative_reconciled']} cum, "
              f"median|err|={g['median_abs_err_bps']}bps) | "
              f"walk-forward acc={w['accuracy']} (n={w['scored']})")

    roe_gf = gf.get("roe", {}).get("accuracy")
    if roe_gf is not None and roe_gf < ROE_ALERT_ACC and gf["roe"]["newly_reconciled"] >= 20:
        _alert(f"⚠️ EDGAR overlay roe REALIZED accuracy {roe_gf:.1%} < "
               f"{ROE_ALERT_ACC:.0%} on {gf['roe']['newly_reconciled']} newly-landed quarters. "
               f"Check scripts/edgar_overlay_reconcile.py report.")
    print(f"  report -> {REPORT}")


if __name__ == "__main__":
    main()
