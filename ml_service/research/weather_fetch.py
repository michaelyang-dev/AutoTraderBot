"""
Fetch daily mean temperature for major US metros (Open-Meteo free historical archive,
ERA5), 2016-2024, and build a population-weighted NATIONAL daily series with heating/
cooling degree days (base 65F). Saves research/_weather.parquet.
"""
import ssl, urllib.request, certifi, json, time
import numpy as np
import pandas as pd

CTX = ssl.create_default_context(cafile=certifi.where())
# (name, lat, lon, metro population weight ~millions)
CITIES = [
    ("NYC", 40.71, -74.01, 20.1), ("LA", 34.05, -118.24, 13.2), ("Chicago", 41.85, -87.65, 9.6),
    ("Dallas", 32.78, -96.80, 7.6), ("Houston", 29.76, -95.37, 7.1), ("DC", 38.90, -77.04, 6.3),
    ("Miami", 25.76, -80.19, 6.1), ("Philadelphia", 39.95, -75.16, 6.2), ("Atlanta", 33.75, -84.39, 6.1),
    ("Phoenix", 33.45, -112.07, 4.9), ("Boston", 42.36, -71.06, 4.9), ("SF", 37.77, -122.42, 4.7),
    ("Minneapolis", 44.98, -93.27, 3.7), ("Detroit", 42.33, -83.05, 4.3), ("Seattle", 47.61, -122.33, 4.0),
]


def fetch(lat, lon):
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&start_date=2016-01-01&end_date=2024-12-31"
           "&daily=temperature_2m_mean&temperature_unit=fahrenheit&timezone=America%2FNew_York")
    for _ in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research"}), context=CTX, timeout=60) as r:
                j = json.loads(r.read())
            return pd.Series(j["daily"]["temperature_2m_mean"], index=pd.to_datetime(j["daily"]["time"]))
        except Exception as e:
            print("  retry", e); time.sleep(5)
    return None


if __name__ == "__main__":
    t0 = time.time()
    temps, wts = {}, {}
    for nm, la, lo, w in CITIES:
        s = fetch(la, lo)
        if s is not None:
            temps[nm] = s; wts[nm] = w
            print(f"  {nm}: {len(s)} days ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(temps)
    W = pd.Series(wts); W = W / W.sum()
    natl = (df[W.index] * W).sum(axis=1)   # population-weighted national mean temp
    out = pd.DataFrame({"temp": natl})
    out["hdd"] = (65 - out["temp"]).clip(lower=0)
    out["cdd"] = (out["temp"] - 65).clip(lower=0)
    out.index.name = "date"
    out.to_parquet("research/_weather.parquet")
    print(f"\n[DONE: {len(out)} days {out.index.min().date()}..{out.index.max().date()}; "
          f"mean temp {out['temp'].mean():.0f}F, annual HDD {out['hdd'].sum()/9:.0f}, CDD {out['cdd'].sum()/9:.0f}]")
