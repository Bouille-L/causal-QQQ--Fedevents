#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Layer 3 — Stage C: Final Alignment
----------------------------------
Inputs
  • Layer 1C cleaned option features:  Options_features_L1_VF.parquet
  • Layer 3 Stage B reactions:         event_reactions.parquet

Output
  • events_aligned_L3.parquet  (one row per event with covariates + outcomes)
  • events_aligned_L3_summary.txt (quick counts / coverage diagnostics)

What it does
  1) Loads cleaned daily option features (by date) and event reactions (by event).
  2) Joins events to the *previous trading session* features via `feat_date`.
  3) Keeps only the tenors we decided to retain (D10, D30, D60, D>60).
  4) Exports an analysis-ready table for Layer 4 causal work.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import argparse
import logging
import os
import sys
import pytz
from datetime import datetime, time, timedelta

# -------- Paths --------
L1C_FEATURES = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\Options_features_L1_VF.parquet"
EVENTS_PARQUET = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer2\events_unified_E1.parquet"
MINUTE_PARQUET = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\QQQ_1min_2011-01_2025-07.parquet"


OUT_PARQUET  = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\\Treatment_dataset_Fedmacr-_event\events_aligned_L3_E1.parquet"
OUT_SUMMARY  = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\\Treatment_dataset_Fedmacr-_event\events_aligned_L3_summary_E1.txt"
OUT_CONFIG   = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\\Treatment_dataset_Fedmacr-_event\layer3_stageC_policy_used_E1.json"



# ----- Logging -----
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# ----- Policy defaults -----
PRICE_T0_TOLERANCE_MIN = 0


# ----------------- Helpers copied/adapted from build_event_reactions_E1.py -----------------
def logret(a, b):
    if a is None or b is None:
        return np.nan
    try:
        if a <= 0 or b <= 0:
            return np.nan
        return float(np.log(b) - np.log(a))
    except Exception:
        return np.nan


def ensure_et_aware(ts):
    """Return a pandas.Timestamp that is timezone-aware in US/Eastern.

    - If ts is parseable and timezone-aware: convert to US/Eastern and return.
    - If ts is parseable and timezone-naive: assume it's already ET and localize to US/Eastern.
    - If parsing fails or ts is null: return None.
    """
    if pd.isna(ts):
        return None
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return None
    if getattr(t, 'tzinfo', None) is None:
        try:
            return t.tz_localize(pytz.timezone("US/Eastern"))
        except Exception:
            py_dt = t.to_pydatetime()
            return pytz.timezone("US/Eastern").localize(py_dt)
    else:
        return t.tz_convert(pytz.timezone("US/Eastern"))


class MinutePriceLookup:
    def __init__(self, minute_df: pd.DataFrame, tolerance_min: int):
        self.df = minute_df
        self.tol = tolerance_min

    def get_price_at(self, ts_et: pd.Timestamp):
        ts_et = ensure_et_aware(ts_et)
        if ts_et is None:
            return None
        pos = self.df["timestamp"].searchsorted(ts_et, side="right") - 1
        if pos < 0:
            return None
        row_ts = self.df.iloc[pos]["timestamp"]
        delta_minutes = (ts_et - row_ts).total_seconds() / 60.0
        if delta_minutes > self.tol:
            return None
        return float(self.df.iloc[pos]["price"]) if "price" in self.df.columns else None

    def get_last_before(self, ts_et: pd.Timestamp):
        ts_et = ensure_et_aware(ts_et)
        if ts_et is None:
            return None
        pos = self.df["timestamp"].searchsorted(ts_et, side="right") - 1
        if pos < 0:
            return None
        return float(self.df.iloc[pos]["price"]) if "price" in self.df.columns else None


def load_minute_prices(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Minute parquet not found: {path}")
    df = pd.read_parquet(path)
    if not all(c in df.columns for c in ["date", "time", "price"]):
        raise ValueError("Minute parquet must have columns date,time,price")
    ts = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(pytz.timezone("US/Eastern"))
    else:
        ts = ts.dt.tz_convert(pytz.timezone("US/Eastern"))
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "price"]]


SESSION_PRE    = "pre_market"
SESSION_REG    = "regular"
SESSION_AFTER  = "after_hours"


def classify_session(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts)
    hhmm = ts.time()
    if time(4, 0) <= hhmm < time(9, 30):
        return SESSION_PRE
    if time(9, 30) <= hhmm < time(16, 0):
        return SESSION_REG
    if time(16, 0) <= hhmm < time(20, 0):
        return SESSION_AFTER
   

def trading_ceiling(date_like, trading_index):
    d = pd.to_datetime(date_like).normalize()
    pos = trading_index.searchsorted(d)
    if pos >= len(trading_index):
        return None
    return trading_index[pos].date()


def trading_shift(date_like, trading_index, n: int):
    d = pd.to_datetime(date_like).normalize()
    pos = trading_index.searchsorted(d)
    if pos < len(trading_index) and trading_index[pos].date() == d.date():
        anchor_pos = pos
    else:
        anchor_pos = trading_index.searchsorted(d)
    tgt = anchor_pos + n
    if tgt < 0 or tgt >= len(trading_index):
        return None
    return trading_index[tgt].date()


def build_reactions(events_path: str, minute_path: str, tolerance_min: int) -> pd.DataFrame:
    """Load events and compute session-based reactions; return DataFrame ready to align (with feat_date).

    Note: daily prices are derived from the minute parquet; the legacy daily file is not used.
    """
      

    minute_df = load_minute_prices(minute_path)
    minute_lookup = MinutePriceLookup(minute_df, tolerance_min)

    # Build daily Open/Close from minute ticks (Close defined as last price at or before 16:00 ET)
    tmp = minute_df.copy()
    # Date as calendar day (midnight), convert to ET then drop tz to match downstream expectations
    tmp["Date"] = tmp["timestamp"].dt.tz_convert(pytz.timezone("US/Eastern")).dt.normalize().dt.tz_localize(None)

    # mask for intraday minutes between 09:30 and 16:00 (market session)
    session_end = tmp["Date"] + pd.Timedelta(hours=16)
    
    # Close = last price at or before 16:00 ET
    closes = tmp[tmp["timestamp"].dt.tz_convert(pytz.timezone("US/Eastern")).dt.tz_localize(None) <= session_end].groupby("Date", sort=True).agg(Close=("price", "last"))

     # build daily_px from closes only (no Open)
    daily_px = closes.reset_index()
    daily_px["Date"] = pd.to_datetime(daily_px["Date"]).dt.normalize()
    daily_px = daily_px.set_index("Date").sort_index()
    trading_index = daily_px.index

    ev = pd.read_parquet(events_path)

    keep_cols = [
        "event_id", "event_domain", "event_type", "series_id", "time_et", "values", 
    ]
    cols = [c for c in keep_cols if c in ev.columns]
    ev = ev[cols].copy()

    # compute anchors
    anchors = []
    for _, r in ev.iterrows():
        d = r.get("time_et")
        t = ensure_et_aware(d)
        if t is None:
            anchors.append(None)
            continue
        # event calendar date in ET
        et_date = t.tz_convert(pytz.timezone("US/Eastern")).date()
        if t.time() >= time(20, 0):
            anchor = trading_ceiling(pd.Timestamp(et_date) + pd.Timedelta(days=1), trading_index)
        else:
            anchor = trading_ceiling(pd.Timestamp(et_date), trading_index)
        anchors.append(anchor)
    ev["anchor_day"] = anchors

    # feat_date = previous trading day
    feat_dates = []
    feat_lags = []
    for _, r in ev.iterrows():
        ad = r["anchor_day"]
        if ad is None:
            feat_dates.append(None)
            feat_lags.append(np.nan)
            continue
        feat = trading_shift(ad, trading_index, -1)
        feat_dates.append(feat)
        feat_lags.append(1 if feat is not None else np.nan)
    ev["feat_date"] = feat_dates
    ev["feat_lag_days"] = feat_lags

    # helper accessors
    def get_close(d):
        if d is None:
            return None
        ts = pd.to_datetime(d)
        if ts not in daily_px.index:
            return None
        v = daily_px.loc[ts, "Close"]
        return None if pd.isna(v) else float(v)
    
    # initialize columns
    ev["event_session"] = pd.Series([None]*len(ev), dtype="object")
    ev["price_t0"] = np.nan
    ev["price_t0_missing"] = 0
    ev["price_t0_age_min"] = np.nan
    ev["price_t0_source"] = pd.Series([None]*len(ev), dtype="object")
    ev["pre_effect_log"] = np.nan
    ev["post_effect_log"] = np.nan
    # ev["pre_post_identity_diff"] = np.nan  # removed: not required per new policy
    ev["aux_gap_next_open_log"] = np.nan
    ev["drop_flag"] = 0
    

    total_events = len(ev)
    processed = 0
    for i, r in ev.iterrows():
        ad = r["anchor_day"]
        if ad is None:
            continue
        prev = trading_shift(ad, trading_index, -1)
        c_prev = get_close(prev)
        c0 = get_close(ad)

        ts = r.get("time_et")
        if pd.isna(ts):
            continue
        t = ensure_et_aware(ts)
        if t is None:
            continue
        session = classify_session(t)

        d0 = pd.to_datetime(ad)
        d_prev = pd.to_datetime(prev) if prev else None
        d_plus1 = trading_shift(ad, trading_index, 1)

        close_d_prev = c_prev
        close_d0 = c0
        close_d_plus1 = get_close(d_plus1)

        # compute price_t0: prefer minute price at event moment (with tolerance).
        # Conservative policy: for trading calendar days use minute-only (strict).
        # For non-trading calendar days, if minute tick within tolerance is absent,
        # fall back to the previous trading-day Close (conservative).
        event_cal_day = t.tz_convert(pytz.timezone("US/Eastern")).normalize()
        is_trading_day = event_cal_day in daily_px.index

        # try exact-minute (subject to tolerance) and record age/source if found
        pt0 = None
        pt0_age = np.nan
        pt0_source = None
        pos = minute_lookup.df["timestamp"].searchsorted(t, side="right") - 1
        if pos >= 0:
            tick_ts = minute_lookup.df.iloc[pos]["timestamp"]
            pt0_age = (t - tick_ts).total_seconds() / 60.0
            if pt0_age <= minute_lookup.tol:
                pt0 = float(minute_lookup.df.iloc[pos]["price"]) if "price" in minute_lookup.df.columns else None
                pt0_source = "minute"

        # Fallback for non-trading calendar days: use previous trading-day Close
        if pt0 is None and (not is_trading_day) and (close_d_prev is not None):
            pt0 = close_d_prev
            pt0_source = "prev_close"

        # record session (mark non-trading calendar days)
        ev.at[i, "event_session"] = session if is_trading_day else "non_trading"

        ev.at[i, "price_t0"] = pt0 if pt0 is not None else np.nan
        # price_t0_missing indicates the absence of a minute-level price
        ev.at[i, "price_t0_missing"] = 0 if (pt0_source == "minute") else 1
        ev.at[i, "price_t0_age_min"] = pt0_age if not np.isnan(pt0_age) else np.nan
        ev.at[i, "price_t0_source"] = pt0_source

        # compute pre/post effects using only closes + minute price_t0
        pre_eff = np.nan
        post_eff = np.nan

        if pt0 is not None and close_d_prev is not None:
            pre_eff = logret(close_d_prev, pt0)

        if session in {SESSION_PRE, SESSION_REG}:
            if pt0 is not None and close_d0 is not None:
                post_eff = logret(close_d0, pt0)
        elif session == SESSION_AFTER:
            if pt0 is not None and close_d_plus1 is not None:
                post_eff = logret(close_d_plus1, pt0)

        # write results for every event
        ev.at[i, "pre_effect_log"] = pre_eff
        ev.at[i, "post_effect_log"] = post_eff
        # increment processed counter once we've computed results for this row
        processed += 1

    # Summary prints so caller can see how many rows were processed
    print(f"Processed events: {processed:,} / {total_events:,} ({0.0 if total_events==0 else 100.0*processed/total_events:.1f}%)")

    # Drop events flagged for missing next open
    ev = ev[ev["drop_flag"] == 0].copy()
    returned = len(ev)
    print(f"Returned rows after drop_flag filter: {returned:,} / {total_events:,}")

    # normalize feat_date to naive midnight US/Eastern for safe joining downstream (same as original)
    if "feat_date" in ev.columns:
        try:
            s = pd.to_datetime(ev["feat_date"], errors="coerce")
            if s.dt.tz is not None:
                s = s.dt.tz_convert(pytz.timezone("US/Eastern")).dt.normalize().dt.tz_localize(None)
            else:
                s = s.dt.normalize()
            ev["feat_date"] = s
        except Exception:
            pass

    return ev

# -----------------------------------------------------------------------------------------------

# -------- Feature selection policy --------
KEEP_TENORS = ["D10", "D30", "D60", "D>60"]

FEATURE_KEEP_PREFIXES = [
    "ATM_IV_", "RR25_", "VOLMIX_", "TURNOVER_", "REL_SPREAD_", "OI_SUM_",
    "PCR_DOLLAR_OTM_", "PCR_VEGA_OTM_", "TS_7_30", "TS_30_60",
    "SKEW_DECAY_30D", "PCR_OI", "NET_OI","DOLLAR_VOL_" 
]

EVENT_KEEP_COLS = [
    "event_id","event_domain","event_type","time_et","anchor_day",
    "feat_date","feat_lag_days","values",
    # Refined methodology only
    "event_session","price_t0","price_t0_missing","pre_effect_log","post_effect_log"
]

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.drop_duplicates("Date").set_index("Date").sort_index()

    keep_cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in FEATURE_KEEP_PREFIXES) and ("_D" not in c):
            keep_cols.append(c)
            continue
        for t in KEEP_TENORS:
            if c.endswith("_" + t) and any(c.startswith(p) for p in FEATURE_KEEP_PREFIXES):
                keep_cols.append(c)
                break
    keep_cols = list(dict.fromkeys(keep_cols))
    return df[keep_cols].copy()

def load_reactions(path: str) -> pd.DataFrame:
    ev = pd.read_parquet(path)
    ev["feat_date"] = pd.to_datetime(ev["feat_date"]).dt.normalize()
    keep = [c for c in EVENT_KEEP_COLS if c in ev.columns]
    return ev[keep].copy()

def align_events_to_features(ev: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    return ev.merge(
        feats.reset_index().rename(columns={"Date": "feat_date"}),
        on="feat_date", how="left", validate="m:1"
    )

def quick_summary(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Total events: {len(df):,}")
    for col in ["pre_effect_log","post_effect_log"]:
        if col in df.columns:
            nonnull = df[col].notna().sum()
            lines.append(f"{col:>12}: non-null {nonnull:,} ({nonnull/len(df):.1%})")
    if "price_t0" in df.columns:
        nonnull = df["price_t0"].notna().sum()
        lines.append(f"{ 'price_t0':>18}: non-null {nonnull:,} ({nonnull/len(df):.1%})")
    if "price_t0_missing" in df.columns:
        miss = int(df['price_t0_missing'].sum())
        lines.append(f"price_t0_missing events: {miss} ({miss/len(df):.1%})")
    for probe in ["ATM_IV_D10","ATM_IV_D30","ATM_IV_D60","ATM_IV_D>60",
                  "RR25_D30","PCR_DOLLAR_OTM_D30","REL_SPREAD_D30","OI_SUM_D30"]:
        if probe in df.columns:
            nonnull = df[probe].notna().sum()
            lines.append(f"{probe:>18}: non-null {nonnull:,} ({nonnull/len(df):.1%})")
    if "event_domain" in df.columns:
        mix = df["event_domain"].value_counts(dropna=False).to_dict()
        lines.append(f"event_domain mix: {mix}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', default=EVENTS_PARQUET)
    parser.add_argument('--minute', default=MINUTE_PARQUET)
    parser.add_argument('--features', default=L1C_FEATURES)
    parser.add_argument('--out-dir', default=Path(OUT_PARQUET).parent)
    
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = load_features(args.features)
    # Build reactions from events + minute + daily data (no intermediate parquet written)
    ev = build_reactions(args.events, args.minute, PRICE_T0_TOLERANCE_MIN)

    aligned = align_events_to_features(ev, feats)

    # ---- Preview ----
    print("\n📌 First 5 events after alignment:")
    print(aligned.head(5).to_string(index=False))
    print("\n📌 Last 5 events after alignment:")
    print(aligned.tail(5).to_string(index=False))

    out_parquet = out_dir / Path(OUT_PARQUET).name
    summary_path = out_dir / Path(OUT_SUMMARY).name
    config_path = out_dir / Path(OUT_CONFIG).name

    # Remove internal/monitoring columns from final saved file per user request
    cols_to_exclude = [
        "anchor_day",
        "feat_date",
        "feat_lag_days",
        "event_session",
        "price_t0",
        "price_t0_missing",
        "price_t0_age_min",
        "price_t0_source",
        "aux_gap_next_open_log",
    ]
    cols_present = [c for c in cols_to_exclude if c in aligned.columns]
    if cols_present:
        aligned = aligned.drop(columns=cols_present)

    # Ensure time_et is the first column in the saved file for ease of reading
    if "time_et" in aligned.columns:
        cols = aligned.columns.tolist()
        cols = [c for c in cols if c != "time_et"]
        aligned = aligned[["time_et"] + cols]

    aligned.to_parquet(out_parquet, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(quick_summary(aligned))

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "KEEP_TENORS": KEEP_TENORS,
            "FEATURE_KEEP_PREFIXES": FEATURE_KEEP_PREFIXES,
            "EVENT_KEEP_COLS": EVENT_KEEP_COLS,
            "price_t0_tolerance_minutes": PRICE_T0_TOLERANCE_MIN,
            "minute_source": args.minute,
        }, f, indent=2)

    print(f"\n✅ Saved aligned dataset → {out_parquet}")
    print(f"🧾 Summary               → {summary_path}")
    print(f"ℹ️  Policy used          → {config_path}")
    
    # New: print counts of NaN / empty in pre/post effect columns (use already-computed `aligned`)
    for col in ["pre_effect_log", "post_effect_log"]:
        if col in aligned.columns:
            total = len(aligned)
            na_count = aligned[col].isna().sum()
            empty_count = 0
            if aligned[col].dtype == object:
                empty_count = (aligned[col].astype(str) == "").sum()
            missing_total = na_count + empty_count
            pct = 0.0 if total == 0 else 100.0 * missing_total / total
            print(f"{col}: missing (NaN+empty) = {missing_total} / {total} ({pct:.1f}%)")
if __name__ == "__main__":
    main()