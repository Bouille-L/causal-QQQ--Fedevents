#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Layer 2.3 Event Formatter (single-file version)
Reads:
        - FRED macro events parquet
Outputs:
        - events_unified.parquet (strict timing, FRED-only schema)

Current behavior (notes):
    - STRICT timing for FRED: rows without a parsed release_time are dropped.
    - Date+time parsing and localization to America/New_York (ET) are done
        in a single helper (`to_et_ts`) that accepts either (date, time) or
        a timestamp-like input and returns a tz-aware pandas.Timestamp.
    - JSON-like columns (values, values_first_release, values_latest) are
        parsed into Python dict/list objects via `to_obj` for downstream use.
    - A compact `dedupe_hash` is produced (SHA-256 prefix) from stable
        identifiers (currently hashed from `event_id`) to support dedupe/upsert.
    - Logging is enabled at DEBUG level for diagnostics; parse failures are
        logged and yield pd.NaT/None as appropriate.
"""

import pandas as pd
import numpy as np
import hashlib
import json, ast
import logging
from zoneinfo import ZoneInfo
from pathlib import Path



# --------- USER PATHS (edit as needed) ---------
# NOTE: Adjust the file path below to match your local setup.
data = pd.read_parquet("C:/Users/yourname/path/to/file.parquet")
FRED_PARQUET   = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer2\macro_events_2011-06-17_2025-06-16_allreports.parquet"
OUTPUT_PARQUET = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer2\events_unified_E1.parquet"
# -----------------------------------------------

ET = ZoneInfo("America/New_York")

# Basic logger for this script (users can reconfigure from their application)
# Default to DEBUG during development so parse failures and diagnostics are visible.
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
# Ensure root logger is at DEBUG so debug messages are consistently emitted
logging.getLogger().setLevel(logging.DEBUG)

def to_et_ts(date_or_ts, time_str=None):
    """
    Parse inputs and return a tz-aware pandas.Timestamp in America/New_York (ET).

    Behavior:
    - If called with (date, time): require both and parse them as an ET wall-clock time.
      If parsing fails or time is missing/invalid, return pd.NaT.
    - If called with a single timestamp-like `date_or_ts` and time_str is None:
      - If it's tz-aware, convert to ET.
      - If it's naive, localize it to ET (treat the wall-clock as ET).
      - If it's a string, try to parse then treat as naive or tz-aware accordingly.

    Returns: pandas.Timestamp tz-aware in America/New_York, or pd.NaT on failure.
    """
    # Quick NA check
    if pd.isna(date_or_ts) and pd.isna(time_str):
        logger.debug("to_et_ts: both inputs NA (date_or_ts=%r, time_str=%r) -> returning NaT", date_or_ts, time_str)
        return pd.NaT

    # If time_str provided, we expect date_or_ts to be a date and time_str to be a time
    if time_str is not None:
        # Strict: require non-empty time
        if pd.isna(date_or_ts) or pd.isna(time_str) or str(time_str).strip() == "":
            logger.warning("to_et_ts: missing/empty time for date %r -> returning NaT (strict policy)", date_or_ts)
            return pd.NaT

        # parse date part
        dts = pd.to_datetime(str(date_or_ts), errors='coerce')
        if pd.isna(dts):
            logger.warning("to_et_ts: failed to parse date %r -> returning NaT", date_or_ts)
            return pd.NaT
        d = dts.date()

        # normalize HH:MM -> HH:MM:SS
        t = str(time_str).strip()
        if ":" not in t:
            logger.warning("to_et_ts: invalid time format %r for date %r -> returning NaT", time_str, date_or_ts)
            return pd.NaT
        parts = t.split(":")
        if len(parts) == 2:
            t = t + ":00"

        parsed = pd.to_datetime(f"{d.isoformat()} {t}", errors='coerce')
        if pd.isna(parsed):
            logger.warning("to_et_ts: failed to parse combined datetime '%s %s' -> returning NaT", d.isoformat(), t)
            return pd.NaT
        ts = parsed
    else:
        # single-argument mode: try to interpret date_or_ts as a timestamp-like value
        try:
            ts = pd.to_datetime(date_or_ts, errors='coerce')
        except Exception:
            return pd.NaT
        if pd.isna(ts):
            logger.warning("to_et_ts: failed to parse timestamp-like input %r -> returning NaT", date_or_ts)
            return pd.NaT

    # At this point `ts` is a pandas.Timestamp (may be naive or tz-aware)
    try:
        # pandas Timestamp exposes tz attribute; if tz is not None it's tz-aware
        if getattr(ts, 'tz', None) is not None:
            # tz-aware: convert to ET
            return ts.tz_convert(ET)
        else:
            # naive: localize to ET (treat wall-clock as ET)
            return ts.tz_localize(ET)
    except Exception:
        # If conversion/localization fails, log and try a fallback: parse string then localize
        logger.exception("to_et_ts: tz conversion/localization failed for %r; attempting fallback parse", ts)
        try:
            parsed = pd.to_datetime(ts, errors='coerce')
            if pd.isna(parsed):
                logger.warning("to_et_ts: fallback parse produced NaT for %r -> returning NaT", ts)
                return pd.NaT
            return parsed.tz_localize(ET)
        except Exception:
            logger.exception("to_et_ts: fallback localization also failed for %r -> returning NaT", ts)
            return pd.NaT


def to_obj(x):
    """
    Ensure JSON-like columns become Python objects (dict/list), not strings.
    Leaves dict/list/NaN untouched. Tries JSON first, then literal_eval.
    """
    if isinstance(x, (dict, list)) or pd.isna(x):
        return x
    s = str(x)
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            logger.debug("to_obj: failed to parse value as JSON or Python literal: %r", s)
            return None

# ---------- Load FRED ----------
fred = pd.read_parquet(FRED_PARQUET)

fred_out = pd.DataFrame()
fred_out["event_id"]    = fred["event_id"]
fred_out["event_domain"]= "macro"
fred_out["event_type"]  = fred["event_type"]
fred_out["series_id"]   = fred.get("series_id")


fred_out["time_et"] = fred.apply(
    lambda r: to_et_ts(r["release_date"], r.get("release_time")),
    axis=1
)

# JSON normalize
# Parse raw JSON-like columns from the original FRED parquet (keep raw objects)
raw_values = fred.get("values").map(to_obj)
fred_out["values"] = fred.get("values").map(to_obj)


# ---------- QC (FRED-only) ----------
cols = [
    "event_id","event_domain","event_type","series_id",
    "time_et","values"
]

events = fred_out[cols].copy()

# STRICT: drop rows missing critical fields (incl. time_et)
before_count = len(events)
missing_time_et = events[events["time_et"].isna()].shape[0]
missing_values = events[events["values"].isna()].shape[0]
events = events.dropna(subset=["event_id","event_type","time_et","values"])
after_count = len(events)
dropped_total = before_count - after_count
print(f"[info] Rows before strict dropna: {before_count}")
print(f"[info] Rows with missing time_et (pd.NaT): {missing_time_et}")
print(f"[info] Rows with missing values: {missing_values}")
print(f"[info] Rows dropped by strict dropna: {dropped_total}")

# Sort by time
events = events.sort_values("time_et").reset_index(drop=True)

# Print event counts by type
print("\nEvent count by type:")
print(events["event_type"].value_counts())

# Print first 5 and last 4 events
print("\nFirst 5 events:")
print(events.head(5))
print("\nLast 5 events:")
print(events.tail(5))

# Small diagnostics (optional)
if len(events):
    try:
        fred_sample  = fred_out["values"].dropna().iloc[0]
        print("FRED values example type:", type(fred_sample))
    except Exception:
        pass

# Write output
Path(OUTPUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)

# Prevent parquet engine from expanding per-row dicts into a struct with the union
# of all keys. Serialize JSON-like object columns to compact JSON strings so the
# original compact dict/list is preserved per-row. On read, callers can json.loads
# these back to objects if desired.
def _serialize_obj_to_json(x):
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, separators=(',', ':'), allow_nan=True)
        except Exception:
            # Fallback: use python repr if json fails
            return repr(x)
    return x

ser = events.copy()
for col in ("values", "values_first_release", "values_latest"):
    if col in ser.columns:
        ser[col] = ser[col].map(_serialize_obj_to_json)

ser.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Wrote unified events parquet to: {OUTPUT_PARQUET}")
print(f"Total rows: {len(events):,}")
print("\nfred_out columns:", fred_out.columns.tolist())
print("events columns:", events.columns.tolist())
