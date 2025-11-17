# fred_datacollection.py
# ------------------------------------------------------------
# FRED macro data collector & event builder
#  - Loads API key from .env (if present)
#  - Builds event-style dataset for a hardwired date window
#  - Calendar mode: event dates from FRED release calendar
#  - Event-day values: ALFRED "as-of" vintage (vintage_dates=release_date)
#  - Keeps initial-vintage (output_type=4) as a fallback only
#  - Saves first-release (event-day, as-published) and latest (revised)
#  - Output: a single Parquet with all events
# ------------------------------------------------------------

from __future__ import annotations

import pandas as pd
import numpy as np  # optional but often used alongside pandas
import os
import sys
import re
import json
import time
import math
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Any

# --- bootstrap: ensure dependencies are present ---
def _ensure(pkgs: List[str]) -> None:
    import_map = {"python-dotenv": "dotenv"}
    for p in pkgs:
        mod = import_map.get(p, p.replace("-", "_"))
        try:
            __import__(mod)
        except Exception:
            print(f"[setup] Installing: {p} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])

_ensure(["requests", "pandas", "python-dotenv", "pyarrow"])

import requests
import pandas as pd
from dotenv import load_dotenv

FRED_BASE = "https://api.stlouisfed.org/fred"
load_dotenv()

# Window (example)
HW_START_DATE: Optional[date] = date(2011, 6, 17)
HW_END_DATE:   Optional[date] = date(2025, 6, 16)

HW_API_KEY: Optional[str] = None
HW_DEBUG: bool = True

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HW_EVENTS_OUT: str = os.path.join(
    _SCRIPT_DIR,
    f"macro_events_{(HW_START_DATE.isoformat() if HW_START_DATE else 'start')}_{(HW_END_DATE.isoformat() if HW_END_DATE else 'end')}.parquet",
)

def resolve_api_key() -> str:
    key = (os.getenv("FRED_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "No FRED API key found.\n"
            "Add FRED_API_KEY=YOUR_32_CHAR_KEY to a .env file or the environment."
        )
    if not re.fullmatch(r"[a-z0-9]{32}", key):
        raise SystemExit("FRED API key must be 32 chars, lowercase alphanumeric.")
    return key

# ------------------------------------------------------------
# Series specs
# ------------------------------------------------------------
@dataclass
class SeriesSpec:
    label: str
    field: str
    series_id: str
    units: str = "lin"               # lin|pc1|pch|chg
    frequency: Optional[str] = None
    notes: Optional[str] = None
    typical_time_et: Optional[str] = None

def build_specs() -> List[SeriesSpec]:
    specs: List[SeriesSpec] = []
    specs += [
        # PPI
        SeriesSpec("PPI", "ppi_headline_yoy", "PPIACO", units="pc1", typical_time_et="08:30"),
        SeriesSpec("PPI", "ppi_headline_mom", "PPIACO", units="pch", typical_time_et="08:30"),
        # Housing Starts
        SeriesSpec("HousingStarts", "housing_starts_mom", "HOUST", units="pch", typical_time_et="08:30"),
        SeriesSpec("HousingStarts", "housing_starts_yoy", "HOUST", units="pc1", typical_time_et="08:30"),
        # JOLTS
        SeriesSpec("JOLTS", "jolts_job_openings_mom", "JTSJOL", units="pch", typical_time_et="10:00"),
        SeriesSpec("JOLTS", "jolts_job_openings_yoy", "JTSJOL", units="pc1", typical_time_et="10:00"),
        # Michigan Sentiment
        SeriesSpec("MichiganSentiment", "michigan_sentiment_mom", "UMCSENT", units="pch", typical_time_et="10:00"),
        SeriesSpec("MichiganSentiment", "michigan_sentiment_yoy", "UMCSENT", units="pc1", typical_time_et="10:00"),
        # Trade Prices
        SeriesSpec("TradePrices", "import_prices_mom", "IR", units="pch", typical_time_et="08:30"),
        SeriesSpec("TradePrices", "import_prices_yoy", "IR", units="pc1", typical_time_et="08:30"),
        SeriesSpec("TradePrices", "export_prices_mom", "IQ", units="pch", typical_time_et="08:30"),
        SeriesSpec("TradePrices", "export_prices_yoy", "IQ", units="pc1", typical_time_et="08:30"),
        # Retail Sales
        SeriesSpec("RetailSales", "retail_sales_mom", "RSXFS", units="pch", typical_time_et="08:30"),
        SeriesSpec("RetailSales", "retail_sales_yoy", "RSXFS", units="pc1", typical_time_et="08:30"),
        # Industrial Production
        SeriesSpec("IndustrialProduction", "industrial_production_mom", "INDPRO", units="pch", typical_time_et="09:15"),
        SeriesSpec("IndustrialProduction", "industrial_production_yoy", "INDPRO", units="pc1", typical_time_et="09:15"),
        # CPI
        SeriesSpec("CPI", "cpi_headline_mom", "CPIAUCSL", units="pch", typical_time_et="08:30"),
        SeriesSpec("CPI", "cpi_headline_yoy", "CPIAUCSL", units="pc1", typical_time_et="08:30"),
        # Unemployment Rate
        SeriesSpec("UnemploymentRate", "unemployment_rate", "UNRATE", units="lin", typical_time_et="08:30"),
        # GDP
        SeriesSpec("GDP", "gdp_qoq", "GDP", units="pch", typical_time_et="08:30"),
        SeriesSpec("GDP", "gdp_yoy", "GDP", units="pc1", typical_time_et="08:30"),
        # FOMC Rate Decision (DFEDTARU/L only; FEDFUNDS handled separately)
        SeriesSpec("FOMC", "fed_funds_target_upper", "DFEDTARU", units="lin", typical_time_et="14:00"),
        SeriesSpec("FOMC", "fed_funds_target_lower", "DFEDTARL", units="lin", typical_time_et="14:00"),
        # Initial Jobless Claims (weekly)
        SeriesSpec("InitialJoblessClaims", "initial_claims_level", "ICSA", units="lin", typical_time_et="08:30"),
        SeriesSpec("InitialJoblessClaims", "initial_claims_wow_change", "ICSA", units="chg", typical_time_et="08:30"),
        # Durable Goods Orders
        SeriesSpec("DurableGoodsOrders", "durable_goods_new_orders_mom", "DGORDER", units="pch", typical_time_et="08:30"),
        SeriesSpec("DurableGoodsOrders", "durable_goods_new_orders_yoy", "DGORDER", units="pc1", typical_time_et="08:30"),
        # Nonfarm Payrolls
        SeriesSpec("NonfarmPayrolls", "nfp_payroll_change", "PAYEMS", units="chg", typical_time_et="08:30"),
        SeriesSpec("NonfarmPayrolls", "unemployment_rate", "UNRATE", units="lin", typical_time_et="08:30"),
        # PCE
        SeriesSpec("PCE", "pce_price_index_mom", "PCEPI", units="pch", typical_time_et="08:30"),
        SeriesSpec("PCE", "pce_price_index_yoy", "PCEPI", units="pc1", typical_time_et="08:30"),
        SeriesSpec("PCE", "core_pce_mom", "PCEPILFE", units="pch", typical_time_et="08:30"),
        SeriesSpec("PCE", "core_pce_yoy", "PCEPILFE", units="pc1", typical_time_et="08:30"),
        # (Optional) If you really want FEDFUNDS around, give it its own label so it doesn't
        # pollute the 2pm FOMC event. Comment out if you don't need it in events.
        # SeriesSpec("FedFundsDaily", "fed_funds_effective", "FEDFUNDS", units="lin", typical_time_et="09:00"),
    ]
    return specs

# Canonical series per label for the release calendar
CANONICAL_SERIES = {
    "CPI":"CPIAUCSL","PPI":"PPIACO","RetailSales":"RSXFS","IndustrialProduction":"INDPRO",
    "HousingStarts":"HOUST","TradePrices":"IR","DurableGoodsOrders":"DGORDER",
    "PCE":"PCEPI","NonfarmPayrolls":"PAYEMS","UnemploymentRate":"UNRATE",
    "JOLTS":"JTSJOL","MichiganSentiment":"UMCSENT","GDP":"GDP",
    "InitialJoblessClaims":"ICSA","FOMC":"DFEDTARU",
    # "FedFundsDaily":"FEDFUNDS",
}

# ------------------------------------------------------------
# FRED fetchers
# ------------------------------------------------------------
def _fred_get_series(api_key: str, series_id: str, observation_start: str, *,
                     observation_end: Optional[str] = None,
                     units: str = "lin",
                     frequency: Optional[str] = None) -> pd.Series:
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": observation_start, "units": units,
    }
    if observation_end: params["observation_end"] = observation_end
    if frequency:       params["frequency"] = frequency
    url = f"{FRED_BASE}/series/observations"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    obs = data.get("observations", [])
    if not obs:
        return pd.Series(dtype=float)
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    def _to_float(x: str) -> float:
        try: return float(x)
        except Exception: return float("nan")
    df["value"] = df["value"].map(_to_float)
    s = df.set_index("date")["value"].sort_index()
    s.name = series_id
    return s

def _fred_get_series_initial_release(api_key: str, series_id: str, observation_start: str, *,
                                     observation_end: Optional[str] = None,
                                     units: str = "lin",
                                     frequency: Optional[str] = None) -> pd.Series:
    def _fetch_once(rt_start: str, rt_end: str) -> pd.DataFrame:
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
                  "observation_start": observation_start, "units": units,
                  "output_type": 4, "realtime_start": rt_start, "realtime_end": rt_end}
        if observation_end: params["observation_end"] = observation_end
        if frequency:       params["frequency"] = frequency
        url = f"{FRED_BASE}/series/observations"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json() or {}
        arr = js.get("observations", [])
        if not arr: return pd.DataFrame(columns=["date","value"])
        dfi = pd.DataFrame(arr)
        return dfi[[c for c in ["date","value"] if c in dfi.columns]]

    needs_chunk = series_id in {"DFEDTARU","DFEDTARL"}
    frames: List[pd.DataFrame] = []
    if needs_chunk:
        today = pd.Timestamp.today().date()

        # choose sensible year bounds 
        obs_year = pd.to_datetime(observation_start).year if observation_start else today.year
        end_year = (pd.to_datetime(observation_end).year if observation_end else today.year)
        start_year = max(2010, obs_year)  

        for y in range(start_year, end_year + 1):
            rt_start = pd.Timestamp(year=y, month=1, day=1).date()
            rt_end   = min(pd.Timestamp(year=y, month=12, day=31).date(), today)  
            if rt_start > rt_end:
                continue
            try:
                frames.append(_fetch_once(rt_start.isoformat(), rt_end.isoformat()))
            except requests.HTTPError as e:
                if getattr(e, "response", None) and e.response.status_code == 400:
                    if HW_DEBUG:
                        print(f"  · no vintage for {series_id} in {y}, skipping")
                    continue
                raise
    else:
        # Unchanged behavior for non-DFEDTAR* series; keep the broad vintage sweep
        # but handle a 400 gracefully by falling back to a modern window.
        try:
            frames.append(_fetch_once("1776-07-04", "9999-12-31"))
        except requests.HTTPError as e:
            if getattr(e, "response", None) and e.response.status_code == 400:
                frames.append(_fetch_once("1990-01-01", pd.Timestamp.today().date().isoformat()))
            else:
                raise

    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames, ignore_index=True)
    if df.empty or "date" not in df.columns or "value" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    def _to_float(x: str) -> float:
        try: return float(x)
        except Exception: return float("nan")
    df["value"] = df["value"].map(_to_float)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    s = df.set_index("date")["value"].sort_index()
    s.name = series_id
    return s

_ALFRED_VINTAGE_CACHE: Dict[str, Dict[date, Optional[float]]] = {}

def _fred_get_series_as_of_vintage(api_key: str, series_id: str,
                                   observation_dates: List[date], *,
                                   vintage_date: date,
                                   units: str = "lin",
                                   frequency: Optional[str] = None,
                                   retries: int = 2,
                                   sleep_sec: float = 0.22) -> Dict[date, Optional[float]]:
    """Return mapping {obs_date -> value_as_of_vintage} using vintage_dates=<release_date>."""
    if not observation_dates:
        return {}
    obs_sorted = sorted(observation_dates)
    obs_start = obs_sorted[0].isoformat()
    obs_end   = obs_sorted[-1].isoformat()
    cache_key = f"v:{series_id}:{vintage_date.isoformat()}:{obs_start}:{obs_end}:{units}"
    if cache_key in _ALFRED_VINTAGE_CACHE:
        return _ALFRED_VINTAGE_CACHE[cache_key]

    url = f"{FRED_BASE}/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
              "observation_start": obs_start, "observation_end": obs_end,
              "vintage_dates": vintage_date.isoformat(), "output_type": 1, "units": units}
    if frequency: params["frequency"] = frequency

    last_err = None
    for attempt in range(1, retries + 2):
        try:
            now = time.time()
            last = getattr(_fred_get_series_as_of_vintage, "_last_call", 0.0)
            wait = sleep_sec - (now - last)
            if wait > 0: time.sleep(wait)
            r = requests.get(url, params=params, timeout=30)
            _fred_get_series_as_of_vintage._last_call = time.time()
            r.raise_for_status()
            js = r.json() or {}
            arr = js.get("observations", []) or []
            out = {d: None for d in observation_dates}
            for item in arr:
                dstr = item.get("date")
                if not dstr: continue
                try: dd = pd.to_datetime(dstr).date()
                except Exception: continue
                if dd in out:
                    try: out[dd] = float(item.get("value"))
                    except Exception: out[dd] = None
            _ALFRED_VINTAGE_CACHE[cache_key] = out
            if HW_DEBUG:
                got = sum(1 for v in out.values() if v is not None)
                print(f"  ↳ vintage {series_id} {obs_start}->{obs_end} @ {vintage_date}: {got} values")
            return out
        except Exception as exc:
            last_err = exc
            if attempt <= retries:
                time.sleep(0.5 * attempt)
                continue
            _ALFRED_VINTAGE_CACHE[cache_key] = {d: None for d in observation_dates}
            if HW_DEBUG:
                print(f"  ⚠ vintage failed {series_id} @ {vintage_date}: {last_err}")
            return {d: None for d in observation_dates}

def collect_initial_release_data(*, api_key: str, observation_start: str,
                                 observation_end: Optional[str]) -> pd.DataFrame:
    specs = build_specs()
    unique_series: Dict[str, SeriesSpec] = {}
    for spec in specs:
        if spec.series_id not in unique_series:
            unique_series[spec.series_id] = spec
    frames: Dict[str, pd.Series] = {}
    for sid, spec in unique_series.items():
        try:
            s = _fred_get_series_initial_release(api_key=api_key, series_id=sid,
                                                 observation_start=observation_start,
                                                 observation_end=observation_end,
                                                 units="lin", frequency=spec.frequency)
            if not s.empty:
                s.name = sid
                frames[sid] = s
                print(f"  ↳ initial base {sid}")
        except Exception as exc:
            print(f"  ⚠ initial-release failed base {sid}: {exc}")
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    df = pd.concat(frames.values(), axis=1).sort_index()
    df.index.name = "date"
    return df

def collect_fred_data(*, api_key: Optional[str],
                      observation_start_override: str,
                      observation_end_override: Optional[str] = None) -> pd.DataFrame:
    key = api_key or resolve_api_key()
    if not observation_start_override:
        raise RuntimeError("observation_start_override is required.")
    start_date = observation_start_override
    specs = build_specs()

    frames: Dict[str, pd.Series] = {}
    errors: List[str] = []
    print(f"[info] Fetching series from FRED start={start_date} end={observation_end_override or '(open)'}")
    for spec in specs:
        try:
            s = _fred_get_series(api_key=key, series_id=spec.series_id,
                                 observation_start=start_date,
                                 observation_end=observation_end_override,
                                 units=spec.units, frequency=spec.frequency)
            s.name = spec.field
            frames[spec.field] = s
            print(f"  ✓ {spec.field} ({spec.series_id})")
        except Exception as e:
            errors.append(f"{spec.field} ({spec.series_id}): {e}")
            print(f"  ⚠ {spec.field} ({spec.series_id}): {e}")

    if not frames:
        raise RuntimeError("No series fetched.")
    if errors:
        print("[warn] Some series could not be fetched and were skipped.")
    df = pd.concat(frames.values(), axis=1).sort_index()
    df.index.name = "date"
    print(f"[done] Wide matrix shape: {df.shape}")
    return df

# ------------------------------------------------------------
# Date helpers (freq-aware previous periods)
# ------------------------------------------------------------
def _month_start(dt: date) -> date:
    return date(dt.year, dt.month, 1)

def _prev_month_start(dt: date) -> date:
    m = dt.month - 1; y = dt.year
    if m == 0: m = 12; y -= 1
    return date(y, m, 1)

def _quarter_start_from_date(dt: date) -> date:
    q = (dt.month - 1) // 3
    qm = q * 3 + 1
    return date(dt.year, qm, 1)

def _prev_quarter_start(dt: date) -> date:
    q = (dt.month - 1) // 3
    prev_q = (q - 1) % 4
    year = dt.year - (1 if q == 0 else 0)
    month = prev_q * 3 + 1
    return date(year, month, 1)

def _prev_saturday(dt: date) -> date:
    d = dt - timedelta(days=1)
    while d.weekday() != 5:
        d -= timedelta(days=1)
    return d

def _map_release_to_observation(label: str, rel_date: date) -> date:
    L = (label or "").lower()
    # Quarterly (GDP)
    if L == "gdp":
        return _prev_quarter_start(rel_date)
    # Weekly (Initial Claims)
    if L == "initialjoblessclaims":
        return _prev_saturday(rel_date)
    # Michigan = current month
    if L == "michigansentiment":
        return _month_start(rel_date)
    # JOLTS = t-2 months
    if L == "jolts":
        return _prev_month_start(_prev_month_start(rel_date))
    # FOMC = meeting day
    if L == "fomc":
        return rel_date
    # Default monthly previous month
    return _prev_month_start(rel_date)

def _prev_obs_for_label(label: str, obs: date) -> date:
    L = (label or "").lower()
    if L == "gdp": return _prev_quarter_start(obs)
    if L == "initialjoblessclaims": return obs - timedelta(days=7)
    # monthly default
    return _prev_month_start(obs)

def _prev_year_obs_for_label(label: str, obs: date) -> Optional[date]:
    L = (label or "").lower()
    if L == "gdp":
        # same quarter last year: month is already quarter start
        return date(obs.year - 1, obs.month, 1)
    if L == "initialjoblessclaims":
        return obs - timedelta(days=364)  # approx 52 weeks
    # monthly default
    return date(obs.year - 1, obs.month, 1)

# ------------------------------------------------------------
# Release calendar helpers
# ------------------------------------------------------------
def _fred_get_series_release(api_key: str, series_id: str) -> Optional[dict]:
    url = f"{FRED_BASE}/series/release"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    try:
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()
        data = r.json() or {}
        rel = (data.get("releases") or data.get("release"))
        if isinstance(rel, dict):
            return {"release_id": rel.get("id"), "release_name": rel.get("name")}
        if isinstance(rel, list) and rel:
            it = rel[0]; return {"release_id": it.get("id"), "release_name": it.get("name")}
    except Exception:
        return None
    return None

def _fred_get_release_dates(api_key: str, release_id: int, start_date: date, end_date: date) -> List[date]:
    url = f"{FRED_BASE}/release/dates"
    collected: List[date] = []
    offset = 0
    page = 1000
    while True:
        params = {"release_id": release_id, "api_key": api_key, "file_type": "json",
                  "limit": page, "offset": offset, "order_by": "release_date", "sort_order": "desc"}
        try:
            r = requests.get(url, params=params, timeout=30); r.raise_for_status()
            data = r.json() or {}
        except Exception:
            break
        arr = data.get("release_dates") or []
        if not arr: break
        reached_earliest = False
        for item in arr:
            d = item.get("date") or item.get("release_date")
            if not d: continue
            try: dd = pd.to_datetime(d).date()
            except Exception: continue
            if dd < start_date: reached_earliest = True
            if start_date <= dd <= end_date: collected.append(dd)
        if reached_earliest: break
        offset += page
        if offset > 10000: break
    return sorted(set(collected)) if collected else []

# ------------------------------------------------------------
# Event builders
# ------------------------------------------------------------
def generate_event_rows(df: pd.DataFrame, *, api_key: Optional[str] = None) -> pd.DataFrame:
    """Observation-date mode (no calendar). Keeps behavior for parity/testing."""
    specs = build_specs()
    rows: List[Dict[str, Any]] = []

    total_dates = len(df)
    processed = 0
    start_ts = time.time()
    print(f"[progress] building events over {total_dates} observation dates...")

    for dt, row in df.iterrows():
        processed += 1
        if processed % 50 == 0 or processed == total_dates:
            elapsed = time.time() - start_ts
            per_row = elapsed / processed if processed else 0.0
            remaining = max(0.0, (total_dates - processed) * per_row)
            print(f"[progress] {processed}/{total_dates} | ETA ~{int(remaining//60)}m{int(remaining%60):02d}s")

        rel_date = dt.date()
        vals_latest = {}
        for fld in df.columns:
            v = row.get(fld)
            if pd.notna(v):
                try: vals_latest[fld] = float(v)
                except Exception: pass
        if not vals_latest: continue

        vals_first = dict(vals_latest)  # passthrough in this mode

        expected = set(df.columns)
        merged = {fld: (vals_first.get(fld, vals_latest.get(fld)))
                  for fld in expected if (fld in vals_first or fld in vals_latest)}

        rows.append({
            "event_id": f"E_macro_allreports_{rel_date.isoformat()}",
            "event_type": "Core Macro Report",
            "release_date": rel_date,
            "release_time": "08:30",
            "release_tz": "ET",
            "release_time_source": "typical",
            "series_id": "ALL",
            "values_first_release": vals_first,
            "values_latest": vals_latest,
            "values": merged,
            "source": "FRED/ALFRED",
            "confidence": 1.0,
            "release_date_source": "observation",
            "is_partial_first": len(vals_first) < len(expected),
            "is_partial_latest": len(vals_latest) < len(expected),
        })

    cols = ["event_id","event_type","release_date","release_time","release_tz","release_time_source",
            "series_id","values_first_release","values_latest","values","source","confidence",
            "release_date_source","is_partial_first","is_partial_latest"]
    if not rows: return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["release_date","event_type"]).reset_index(drop=True)

def generate_event_rows_via_release_calendar(
    df: pd.DataFrame,
    *,
    api_key: str,
    initial_release_df: Optional[pd.DataFrame] = None,
    window_start: Optional[date] = None,
    window_end: Optional[date] = None,
    fomc_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calendar-based events.
    - release_date: from FRED calendar
    - values_first_release: **as-of release day** numbers (primary), falling back to initial vintage
    - values_latest: today's snapshot from df
    - values: merged (prefer values_first_release, else latest)
    """
    specs = build_specs()
    spec_by_field: Dict[str, SeriesSpec] = {s.field: s for s in specs}

    data_start = pd.to_datetime(df.index.min()).date()
    data_end   = pd.to_datetime(df.index.max()).date()
    start_d = window_start or data_start
    end_d   = window_end   or data_end

    rows: List[Dict[str, Any]] = []
    release_cache: Dict[str, int] = {}
    dates_cache: Dict[int, List[date]] = {}

    # group fields by label
    groups: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        label = spec.label
        if label not in groups:
            groups[label] = {"fields": [], "series_ids": set(), "time_et": spec.typical_time_et or "08:30"}
        groups[label]["fields"].append(spec.field)
        groups[label]["series_ids"].add(spec.series_id)

    # Build events per label
    for label, meta in groups.items():
        # Choose canonical series for calendar
        sid_calendar = CANONICAL_SERIES.get(label, next(iter(meta["series_ids"])))
        # FOMC: ensure FEDFUNDS is not part of the event bundle
        fields_for_label = [f for f in meta["fields"]
                            if not (label == "FOMC" and spec_by_field[f].series_id == "FEDFUNDS")]

        # =====================================================
        # CUSTOM FOMC HANDLING (CSV REQUIRED; no calendar fallback)
        # =====================================================
        if label == "FOMC":
            if not fomc_csv_path or not os.path.exists(fomc_csv_path):
                print(f"[ERROR] FOMC CSV required but missing: {fomc_csv_path}")
                # Skip producing any FOMC events if file absent
                continue
            try:
                fomc_df = pd.read_csv(fomc_csv_path, parse_dates=["date"])
            except Exception as e:
                print(f"[FOMC override] failed to read CSV '{fomc_csv_path}': {e}")
                fomc_df = pd.DataFrame(columns=["date"])  # proceed empty
            fomc_dates = [d.date() for d in fomc_df.get("date", []) if start_d <= d.date() <= end_d]
            print(f"[FOMC override] Using {len(fomc_dates)} official FOMC decision dates from CSV")

            last_upper: Optional[float] = None
            last_lower: Optional[float] = None
            for rel_date in sorted(fomc_dates):
                vals_latest: Dict[str, float] = {}
                vals_first: Dict[str, float] = {}

                # Latest snapshot from wide df (if present exactly on rel_date)
                for fld in fields_for_label:
                    ts_rel = pd.Timestamp(rel_date)
                    if fld in df.columns and ts_rel in df.index:
                        v = df.at[ts_rel, fld]
                        if pd.notna(v):
                            try:
                                vals_latest[fld] = float(v)
                            except Exception:
                                pass

                # Event-day vintage fetch
                for fld in fields_for_label:
                    spec = spec_by_field[fld]
                    vintage_map = _fred_get_series_as_of_vintage(
                        api_key=api_key,
                        series_id=spec.series_id,
                        observation_dates=[rel_date],
                        vintage_date=rel_date,
                        units="lin",
                        frequency=spec.frequency,
                    )
                    raw_v = vintage_map.get(rel_date)
                    if raw_v is not None and not (isinstance(raw_v, float) and math.isnan(raw_v)):
                        try:
                            vals_first[fld] = float(raw_v)
                        except Exception:
                            pass

                # Synthesize no-change if both empty
                if not vals_latest and not vals_first:
                    if last_upper is not None and last_lower is not None:
                        vals_latest = vals_first = {
                            "fed_funds_target_upper": last_upper,
                            "fed_funds_target_lower": last_lower,
                        }
                        src = "synthetic_no_change"
                    else:
                        vals_latest = vals_first = {
                            "fed_funds_target_upper": None,
                            "fed_funds_target_lower": None,
                        }
                        src = "synthetic_initial"
                else:
                    src = "FOMC CSV"

                # Update rolling memory of last known targets
                up = vals_latest.get("fed_funds_target_upper") or vals_first.get("fed_funds_target_upper")
                lo = vals_latest.get("fed_funds_target_lower") or vals_first.get("fed_funds_target_lower")
                if up is not None:
                    last_upper = up
                if lo is not None:
                    last_lower = lo

                merged_values: Dict[str, float] = {**vals_first, **vals_latest}
                rows.append({
                    "event_id": f"E_macro_FOMC_{rel_date.isoformat()}",
                    "event_type": "FOMC Decision",
                    "release_date": rel_date,
                    "release_time": meta["time_et"],
                    "release_tz": "ET",
                    "release_time_source": "typical",
                    "series_id": sid_calendar,
                    "values_first_release": vals_first,
                    "values_latest": vals_latest,
                    "values": merged_values,
                    "source": src,
                    "confidence": 1.0,
                    "release_date_source": "csv_fomc",
                    "is_partial_first": len(vals_first) < len(fields_for_label),
                    "is_partial_latest": len(vals_latest) < len(fields_for_label),
                })
            continue  # skip normal calendar logic for FOMC
        # =====================================================
        # END CUSTOM FOMC CSV (mandatory) OVERRIDE
        # =====================================================

        # calendar lookup
        if sid_calendar not in release_cache:
            ri = _fred_get_series_release(api_key, sid_calendar)
            if not ri or not ri.get("release_id"):
                print(f"[warn] release_id not found for label='{label}' series_id='{sid_calendar}' -> skip group")
                continue
            release_cache[sid_calendar] = int(ri["release_id"])
        rel_id = release_cache[sid_calendar]
        if rel_id not in dates_cache:
            dates_cache[rel_id] = _fred_get_release_dates(api_key, rel_id, start_d, end_d)
            if not dates_cache[rel_id]:
                print(f"[warn] no release dates for label='{label}' release_id={rel_id} in {start_d}->{end_d}")
        rel_dates = [d for d in dates_cache.get(rel_id, []) if start_d <= d <= end_d]
        if not rel_dates:
            print(f"[warn] label='{label}' has zero release dates after filtering")
            continue

        for rel_date in rel_dates:
            # Build values_latest using wide (today's snapshot) based on mapped obs dates
            vals_latest: Dict[str, float] = {}
            obs_by_field: Dict[str, date] = {}
            for fld in fields_for_label:
                spec = spec_by_field[fld]
                # base mapping by label + special cases
                obs = _map_release_to_observation(label, rel_date)
                if spec.series_id == "JTSJOL":
                    obs = _prev_month_start(_prev_month_start(rel_date))
                elif spec.series_id == "UMCSENT":
                    obs = _month_start(rel_date)
                elif spec.series_id == "ICSA":
                    obs = _prev_saturday(rel_date)
                elif spec.series_id in {"DFEDTARU","DFEDTARL"}:
                    obs = rel_date
                obs_by_field[fld] = obs
                ts = pd.Timestamp(obs)
                if ts in df.index:
                    v = df.at[ts, fld] if fld in df.columns else None
                    if pd.notna(v):
                        try: vals_latest[fld] = float(v)
                        except Exception: pass

            if not vals_latest:
                # no data for this date in the wide matrix; skip
                continue

            # Build values_first_release as **as-of release day** using ALFRED vintage.
            vals_first: Dict[str, float] = {}
            for fld in fields_for_label:
                spec = spec_by_field[fld]
                sid = spec.series_id
                obs = obs_by_field[fld]

                # what legs do we need?
                need_prev = (spec.units in {"pch","chg"})
                need_yoy  = (spec.units == "pc1")

                # compute comparison dates by frequency
                prev_obs = _prev_obs_for_label(label, obs) if need_prev else None
                yoy_obs  = _prev_year_obs_for_label(label, obs) if need_yoy else None

                # primary: as-of vintage (same vintage for current + comps)
                obs_dates_needed = [obs]
                if prev_obs: obs_dates_needed.append(prev_obs)
                if yoy_obs:  obs_dates_needed.append(yoy_obs)
                vintage_map = _fred_get_series_as_of_vintage(
                    api_key=api_key,
                    series_id=sid,
                    observation_dates=[d for d in obs_dates_needed if d is not None],
                    vintage_date=rel_date,
                    units="lin",
                    frequency=spec.frequency,
                )

                def _coerce(x):
                    try:
                        return float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None
                    except Exception:
                        return None

                raw = _coerce(vintage_map.get(obs))
                prev_raw = _coerce(vintage_map.get(prev_obs)) if prev_obs else None
                yoy_raw  = _coerce(vintage_map.get(yoy_obs))  if yoy_obs else None

                # fallback to initial vintage (output_type=4) if as-of missing
                if raw is None and initial_release_df is not None and sid in initial_release_df.columns:
                    ts = pd.Timestamp(obs)
                    if ts in initial_release_df.index:
                        raw = _coerce(initial_release_df.at[ts, sid])
                if prev_obs and prev_raw is None and initial_release_df is not None and sid in initial_release_df.columns:
                    ts = pd.Timestamp(prev_obs)
                    if ts in initial_release_df.index:
                        prev_raw = _coerce(initial_release_df.at[ts, sid])
                if yoy_obs and yoy_raw is None and initial_release_df is not None and sid in initial_release_df.columns:
                    ts = pd.Timestamp(yoy_obs)
                    if ts in initial_release_df.index:
                        yoy_raw = _coerce(initial_release_df.at[ts, sid])

                # assemble by units
                v_first = None
                try:
                    if spec.units == "lin":
                        if raw is not None:
                            v_first = raw
                    elif spec.units == "pch":
                        if raw is not None and prev_raw not in (None, 0.0):
                            v_first = 100.0 * (raw / prev_raw - 1.0)
                    elif spec.units == "pc1":
                        if raw is not None and yoy_raw not in (None, 0.0):
                            v_first = 100.0 * (raw / yoy_raw - 1.0)
                    elif spec.units == "chg":
                        if raw is not None and prev_raw is not None:
                            v_first = raw - prev_raw
                except Exception:
                    v_first = None

                if v_first is not None:
                    vals_first[fld] = v_first

            expected_fields = set(fields_for_label)
            is_partial_first = len(vals_first) < len(expected_fields)
            is_partial_latest = len(vals_latest) < len(expected_fields)

            # merged values: prefer event-day (vals_first), else latest
            merged_values: Dict[str, float] = {}
            for fld in expected_fields:
                if fld in vals_first:
                    merged_values[fld] = vals_first[fld]
                elif fld in vals_latest:
                    merged_values[fld] = vals_latest[fld]

            rows.append({
                "event_id": f"E_macro_{label}_{rel_date.isoformat()}",
                "event_type": f"{label} Release",
                "release_date": rel_date,
                "release_time": meta["time_et"],
                "release_tz": "ET",
                "release_time_source": "typical",
                "series_id": sid_calendar,
                "values_first_release": vals_first,
                "values_latest": vals_latest,
                "values": merged_values,
                "source": "FRED calendar",
                "confidence": 1.0,
                "release_date_source": "calendar",
                "is_partial_first": bool(is_partial_first),
                "is_partial_latest": bool(is_partial_latest),
            })

    cols = ["event_id","event_type","release_date","release_time","release_tz","release_time_source",
            "series_id","values_first_release","values_latest","values",
            "source","confidence","release_date_source","is_partial_first","is_partial_latest"]
    if not rows:
        return pd.DataFrame(columns=cols)
    ev = pd.DataFrame(rows)

    # diagnostics (detailed per-label + global)
    total_missing_fields = 0
    events_with_incomplete_values = 0
    per_label_stats: Dict[str, Dict[str, int]] = {}

    for label, meta in groups.items():
        expected_fields = set(meta["fields"])
        exp_n = len(expected_fields)
        # select events for this label (event_type is "{label} Release")
        mask = ev["event_type"].str.startswith(f"{label} ")
        subset = ev[mask]
        if subset.empty:
            continue
        total = len(subset)
        fully_using_latest = 0
        partial_first = 0
        missing_fields = 0
        incomplete_events = 0

        for _, r in subset.iterrows():
            v_first = r.get("values_first_release") or {}
            v_values = r.get("values") or {}
            n_first = len(v_first) if isinstance(v_first, dict) else 0
            n_values = len(v_values) if isinstance(v_values, dict) else 0

            if n_first == 0:
                fully_using_latest += 1
            elif 0 < n_first < exp_n:
                partial_first += 1

            if n_values < exp_n:
                incomplete_events += 1
                missing_fields += (exp_n - n_values)

        per_label_stats[label] = {
            "total_events": total,
            "fully_using_latest": fully_using_latest,
            "partial_first": partial_first,
            "missing_fields_in_values": missing_fields,
            "events_with_incomplete_values": incomplete_events,
        }

        total_missing_fields += missing_fields
        events_with_incomplete_values += incomplete_events

    # print per-label summaries
    for lbl, stats in per_label_stats.items():
        print(
            f"[summary:{lbl}] events={stats['total_events']} | fully_using_latest={stats['fully_using_latest']} "
            f"| partial_first={stats['partial_first']} | missing_fields_in_values={stats['missing_fields_in_values']} "
            f"| events_with_incomplete_values={stats['events_with_incomplete_values']}"
        )

    # global summary for the values column
    print(f"[summary] total missing fields in 'values' column: {total_missing_fields} | events with incomplete 'values': {events_with_incomplete_values}")

    return ev.sort_values(["release_date", "event_type"]).reset_index(drop=True)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
def _save_events(ev: pd.DataFrame, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    p = out_path.lower()
    if p.endswith(".parquet"):
        ev2 = ev.copy()
        if "release_date" in ev2.columns:
            ev2["release_date"] = pd.to_datetime(ev2["release_date"])
        for col in ["values_first_release","values_latest","values"]:
            if col in ev2.columns:
                ev2[col] = ev2[col].apply(lambda x: json.dumps(x))
        try:
            ev2.to_parquet(out_path, engine="pyarrow", index=False)
        except Exception:
            ev2.to_parquet(out_path, index=False)
        print(f"[save] Parquet -> {out_path}")
        return out_path
    elif p.endswith(".csv"):
        ev2 = ev.copy()
        for col in ["values_first_release","values_latest","values"]:
            if col in ev2.columns:
                ev2[col] = ev2[col].apply(lambda x: json.dumps(x))
        ev2.to_csv(out_path, index=False)
        print(f"[save] CSV -> {out_path}")
        return out_path
    else:
        jsonl = out_path if p.endswith(".jsonl") else out_path.rsplit(".", 1)[0] + ".jsonl"
        with open(jsonl, "w", encoding="utf-8") as f:
            for _, r in ev.iterrows():
                obj = {
                    "event_id": r["event_id"],
                    "event_type": r["event_type"],
                    "release_date": pd.to_datetime(r["release_date"]).date().isoformat(),
                    "release_time": r["release_time"],
                    "release_tz": r["release_tz"],
                    "release_time_source": r.get("release_time_source", "typical"),
                    "series_id": r["series_id"],
                    "values_first_release": r.get("values_first_release", {}),
                    "values_latest": r.get("values_latest", {}),
                    "values": r.get("values", {}),
                    "source": r["source"],
                    "confidence": float(r["confidence"]),
                    "release_date_source": r.get("release_date_source", "unknown"),
                    "is_partial_first": bool(r.get("is_partial_first", False)),
                    "is_partial_latest": bool(r.get("is_partial_latest", False)),
                }
                f.write(json.dumps(obj) + "\n")
        print(f"[save] JSONL -> {jsonl}")
        return jsonl

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    start_date_env = date(2011, 6, 17)
    end_date_env   = date(2025, 6, 16)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    events_out = os.path.join(out_dir, "macro_events_2011-06-17_2025-06-16_allreports.parquet")

    print("[run] hardwired config active")
    api_key = resolve_api_key()

    # buffer for YoY transforms
    buffer_days = 400
    obs_start = (start_date_env - timedelta(days=buffer_days)).isoformat()

    print(f"[run] mode=calendar OUT={events_out}")
    print(f"[run] window: {start_date_env} -> {end_date_env}")
    print(f"[run] fetch observation_start={obs_start}")

    # latest snapshot matrix
    wide = collect_fred_data(api_key=api_key,
                             observation_start_override=obs_start,
                             observation_end_override=end_date_env.isoformat())
    wide = wide[wide.index <= pd.Timestamp(end_date_env)]

    # initial-vintage matrix (fallback only)
    print("[info] Fetching initial-release matrices (output_type=4) for fallback)...")
    initial_wide = collect_initial_release_data(api_key=api_key,
                                                observation_start=obs_start,
                                                observation_end=end_date_env.isoformat())
    if not initial_wide.empty:
        initial_wide = initial_wide[initial_wide.index <= pd.Timestamp(end_date_env)]
    else:
        initial_wide = None

    print("[info] Building event-style dataset (FRED release calendar w/ as-of vintages + FOMC CSV override)...")
    fomc_csv_path = os.path.join(out_dir, "fomc_statement_dates.csv")
    events = generate_event_rows_via_release_calendar(
        wide,
        api_key=api_key,
        initial_release_df=initial_wide,
        window_start=start_date_env,
        window_end=end_date_env,
        fomc_csv_path=fomc_csv_path,
    )
    events = events[(events["release_date"] >= start_date_env) & (events["release_date"] <= end_date_env)]

    saved_path = _save_events(events, events_out)

    try:
        counts = events["release_date_source"].fillna("unknown").value_counts(dropna=False) if "release_date_source" in events.columns else pd.Series()
        cal = int(counts.get("calendar", 0))
        csv_fomc = int(counts.get("csv_fomc", 0))
        total = len(events)
        pct_cal = (cal / total * 100.0) if total else 0.0
        pct_csv_fomc = (csv_fomc / total * 100.0) if total else 0.0
        print(f"[summary] Events: {total} | calendar: {cal} ({pct_cal:.1f}%) | csv_fomc: {csv_fomc} ({pct_csv_fomc:.1f}%)")
    except Exception:
        pass

    print(f"[done] All set. Output -> {saved_path}")

if __name__ == "__main__":
    main()
