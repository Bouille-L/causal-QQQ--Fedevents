"""
Align EOD covariates from Options_features_L1_VF.parquet to a control minute-level dataset.

What this script actually does (accurate description):

 - Inputs
     - Control (minute-level) data: expected to contain columns `date` and `time` which
         are concatenated and parsed into timestamps. The code currently does NOT read a
         `timestamp` column directly; instead it builds timestamps from `date` + `time`.
     - Covariates: must contain a `Date` column. `Date` is normalized to a calendar date
         (Python date) and used as the key for EOD covariates.

 - Timezones and session classification
     - All parsed timestamps are localized/converted to America/New_York (Eastern time)
         for the purpose of determining `event_date` and session classification. NaT values
         are preserved when parsing fails.
     - Session buckets (Eastern time):
             * pre-market: time < 09:30  (label: 'pre')
             * regular:    09:30 <= time < 16:00  (label: 'regular')
             * after:      16:00 <= time < 20:00  (label: 'after')

 - Covariate selection logic
     - The covariates file is converted to an index of calendar dates (python date objects)
         and sorted.
     - For rows classified as 'after': the script prefers the same-day covariate if it
         exists in the covariates set; otherwise it selects the most recent previous
         available covariate date.
     - For rows classified as 'pre' or 'regular' (including timestamps on Mondays), the
         script always selects the most recent previous available covariate date (i.e. a
         covariate strictly before the event_date). This implicitly handles Monday
         timestamps mapping back to Friday's EOD covariates when appropriate.

 - Filtering and output
     - Before mapping, control rows are filtered to the inclusive range of available
         covariate dates (rows with event_date outside the covariate date span are removed).
     - After covariates are mapped into the control frame, the script prints per-column
         NaN diagnostics and then drops any row that contains any NaN. Although there is a
         CLI flag `--keep-na`, the current implementation does not honor it; rows with
         NaNs are always dropped.
     - The resulting cleaned dataframe is written to the output parquet path provided.

Usage:
        python align_covariates.py --control path/to/control.parquet \
                --covariates path/to/Options_features_L1_VF.parquet --out path/to/output.parquet

"""
from __future__ import annotations

import argparse
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import logging
import numpy as np
import time as time_mod


# module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# default session time constants for classifying minutes
PRE_MARKET_START = time(0, 0)
AFTER_HOURS_END = time(20, 0)
PRE_MARKET_END = time(9, 30)
REGULAR_SESSION_END = time(16, 0)

def session_type(ts: pd.Timestamp) -> str:
    """Return 'pre'|'regular'|'after' depending on time of day of ts."""
    t = ts.time()
    if t < PRE_MARKET_END:
        return "pre"
    if t <= REGULAR_SESSION_END:
        return "regular"
    return "after"


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def prepare_covariates(cov_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure Date is a date (no time component) and set as index
    cov_df = cov_df.copy()
    if 'Date' not in cov_df.columns:
        raise KeyError("covariates file must contain a 'Date' column")
    # normalize Date to timezone-naive date (calendar date without time)
    cov_df['Date'] = pd.to_datetime(cov_df['Date']).dt.tz_localize(None).dt.date
    cov_df = cov_df.set_index('Date')
    return cov_df


def normalize_to_eastern(ts_series: pd.Series, eastern_tz: ZoneInfo | None = None) -> pd.Series:
    """Return a tz-aware Series in America/New_York (NaT preserved).

    Accepts tz-naive or tz-aware Series. NaT entries are preserved.
    """
    if eastern_tz is None:
        eastern_tz = ZoneInfo('America/New_York')
    ts = pd.to_datetime(ts_series, errors='coerce')
    if pd.api.types.is_datetime64tz_dtype(ts):
        return ts.dt.tz_convert(eastern_tz)
    else:
        return ts.dt.tz_localize(eastern_tz)


def logret_series(a, b):
    # a and b are arrays or Series; return log(b)-log(a) safely
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(a.shape, np.nan, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b)) & (a > 0) & (b > 0)
    out[mask] = np.log(b[mask]) - np.log(a[mask])
    return out


def classify_session_vec(ts_ser: pd.Series) -> np.ndarray:
    # vectorized session classifier returns array of session strings
    tt = ts_ser.dt.time
    pre = (tt >= PRE_MARKET_START) & (tt < PRE_MARKET_END)
    reg = (tt >= PRE_MARKET_END) & (tt < REGULAR_SESSION_END)
    aft = (tt >= REGULAR_SESSION_END) & (tt < AFTER_HOURS_END)
    sess = np.empty(len(ts_ser), dtype=object)
    sess[:] = 'overnight'
    sess[pre] = 'pre'
    sess[reg] = 'regular'
    sess[aft] = 'after'
    return sess


def align(control_df: pd.DataFrame, cov_df: pd.DataFrame, ts_series: pd.Series | None = None) -> pd.DataFrame:
    """Align covariates to control_df.

    ts_series: optional Series of datetimes (naive or tz-aware) built from date+time
               if provided. If None, function expects control_df to contain a
               'timestamp' column.
    """
    df = control_df.copy()

    # Timestamp handling: use provided ts_series if given, else read df['timestamp']
    if ts_series is not None:
        # normalize to Eastern once using helper (preserves NaT)
        ts = normalize_to_eastern(ts_series)
    else:
        raise KeyError("control file must contain a 'timestamp' column or provide date+time")

    # event_date is the date in Eastern time
    df['event_date'] = ts.dt.date
    # vectorized session classification using the temporary ts
    df['session'] = classify_session_vec(ts)

    # Prepare a sorted array of available covariate dates (python date objects)
    cov_dates = sorted(set(pd.to_datetime(cov_df.index).date))
    logger.debug("Prepared covariate dates: %d unique dates (sample: %s)", len(cov_dates), cov_dates[:5])

    # Vectorized cov_date assignment using numpy.searchsorted for speed
    cov_dates_arr = np.array(cov_dates, dtype='object')

    # convert event_date series to numpy array of python dates
    event_dates = np.array(df['event_date'].tolist(), dtype='object')

    # For each event_date, find insertion position
    # searchsorted requires sorted array; cov_dates_arr is sorted
    # We'll use np.searchsorted on an object array by casting to datetime64 for comparison
    cov_dates_dt64 = np.array(pd.to_datetime(cov_dates_arr))
    event_dates_dt64 = np.array(pd.to_datetime(event_dates))
    pos = np.searchsorted(cov_dates_dt64, event_dates_dt64, side='left')

    # prepare output cov_date array
    cov_date_result = np.empty(len(event_dates), dtype=object)
    cov_date_result[:] = None

    # session 'after' -> prefer same-day if available, else fallback to previous available date
    is_after = df['session'] == 'after'
    # for after: if pos < len and cov_dates[pos] == event_date -> same-day; else previous (pos-1) if >=0
    for i in np.nonzero(is_after)[0]:
        p = pos[i]
        if p < len(cov_dates):
            if cov_dates_dt64[p] == event_dates_dt64[i]:
                cov_date_result[i] = cov_dates[p]
                continue
        prev = p - 1
        if prev >= 0:
            cov_date_result[i] = cov_dates[prev]
        else:
            cov_date_result[i] = None

    # for pre/regular: always use previous available trading day (strictly < event_date)
    is_pre_reg = ~is_after
    for i in np.nonzero(is_pre_reg)[0]:
        p = pos[i]
        prev = p - 1
        if prev >= 0:
            cov_date_result[i] = cov_dates[prev]
        else:
            cov_date_result[i] = None

    df['cov_date'] = cov_date_result

    missing_cov_dates = pd.isna(df['cov_date']).sum()
    if missing_cov_dates:
        logger.warning("%d rows have no cov_date (no previous covariate available); these will produce NaNs", missing_cov_dates)

    # Map covariates into df using cov_date; cov_df index are python date objects
    cov_df_indexed = cov_df.copy()
    cov_df_indexed.index = pd.to_datetime(cov_df_indexed.index).date
    cov_cols = [c for c in cov_df.columns]
    for col in cov_cols:
        df[col] = df['cov_date'].map(cov_df_indexed[col])

    logger.debug("Mapped %d covariate columns into control frame", len(cov_cols))

    return df


def main():
    parser = argparse.ArgumentParser()
 
    default_control = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\QQQ_1min_2011-01_2025-07.parquet"
    default_cov = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\Options_features_L1_VF.parquet"
    default_out = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\Control_dataset_days_without_events\control_sample_QQQ_aligned.parquet"

    parser.add_argument('--control', required=False, default=default_control, help=f'path to minute parquet (default: {default_control})')
    parser.add_argument('--covariates', required=False, default=default_cov, help=f'path to Options_features_L1_VF.parquet (default: {default_cov})')
    parser.add_argument('--out', required=False, default=default_out, help=f'output parquet path (default: {default_out})')
    parser.add_argument('--keep-na', action='store_true', help='If set, keep rows with NaNs in covariates; default is to drop any row with NaN')
    args = parser.parse_args()

    control_path = Path(args.control)
    cov_path = Path(args.covariates)
    out_path = Path(args.out)

    control_df = load_parquet(control_path)
    cov_df = load_parquet(cov_path)

    # --- Remove control rows that fall on macro event release dates ---
    # Source: Layer2/macro_events_2011-06-17_2025-06-16_allreports.parquet (column: release_date)
    try:
        events_path = Path(r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer2\macro_events_2011-06-17_2025-06-16_allreports.parquet")
        events_df = pd.read_parquet(events_path)
        if 'release_date' not in events_df.columns:
            raise KeyError("events file missing 'release_date' column")

        # Parse release_date to calendar dates
        ev_dates = pd.to_datetime(events_df['release_date'], errors='coerce')
        # If tz-aware, drop tz; then take calendar date
        if pd.api.types.is_datetime64tz_dtype(ev_dates):
            ev_dates = ev_dates.dt.tz_localize(None)
        ev_dates = ev_dates.dt.date.dropna().unique()
        ev_set = set(ev_dates)

        if 'date' not in control_df.columns:
            raise KeyError("control file missing 'date' column")

        before_event_filter = len(control_df)
        c_dates = pd.to_datetime(control_df['date'], errors='coerce').dt.date
        mask = ~c_dates.isin(ev_set)
        removed_count = int((~mask).sum())
        control_df = control_df.loc[mask].reset_index(drop=True)

        kept = len(control_df)
        pct = (removed_count / before_event_filter * 100) if before_event_filter else 0.0
        print(f"Dropped {removed_count} control rows ({pct:.2f}%) on {len(ev_set)} macro event release dates; kept {kept} of {before_event_filter}")
    except Exception as e:
        print(f"Warning: event-date filtering skipped ({e})")

    cov_df_prepared = prepare_covariates(cov_df)

    # --- Filter control rows to covariate date range BEFORE aligning ---
    cov_dates_all = sorted(set(pd.to_datetime(cov_df_prepared.index).date))
    if cov_dates_all:
        first_cov_date = cov_dates_all[0]
        last_cov_date = cov_dates_all[-1]
    else:
        first_cov_date = last_cov_date = None

     # Build a temporary timestamp Series for processing (not added as a column)
    if {'date', 'time'}.issubset(control_df.columns):
        ts_series = pd.to_datetime(control_df['date'].astype(str) + ' ' + control_df['time'].astype(str), errors='coerce')
        nat_count = ts_series.isna().sum()
        if nat_count:
            logger.warning("Parsed timestamps contain %d unparsable entries (NaT). These rows may be filtered or dropped later.", nat_count)
    else:
        raise SystemExit("control_df must contain 'date' and 'time' columns (or a 'timestamp' column).")
    

    initial_control_count = len(control_df)
    # If we have timestamps (either original or built), compute event_date and filter
    if ts_series is not None:
        # normalize timestamps once to Eastern and reuse for filtering and align
        ts_series = normalize_to_eastern(ts_series)
        event_dates = ts_series.dt.date

        if first_cov_date is not None and last_cov_date is not None:
            before_count = (event_dates < first_cov_date).sum()
            after_count = (event_dates > last_cov_date).sum()
            mask = event_dates.between(first_cov_date, last_cov_date)
            control_df = control_df.loc[mask].copy().reset_index(drop=True)
            ts_series = ts_series.loc[mask].reset_index(drop=True)
            filtered_count = len(control_df)
            print(f'Filtered control rows to covariate range {first_cov_date}..{last_cov_date}: kept {filtered_count} of {initial_control_count} (before: {before_count}, after: {after_count})')
        else:
            print('No covariate dates found; skipping control filtering')
    else:
        # No timestamps available: we'll attempt to align but align() will raise if missing
        print('No timestamp or date+time columns found in control file; align will require timestamp')

    # Now align using the filtered control_df; pass ts_series so align doesn't create a column
    merged = align(control_df, cov_df_prepared, ts_series=ts_series)

    # ------------------
    # Compute closes at 16:00 for available trading dates (from full control file)
    # ------------------
    # Reload full control file to build a reliable close-price map (use original control data)
    try:
        control_all = load_parquet(control_path)
    except Exception:
        # fallback to using the filtered control frame if reload fails
        control_all = pd.read_parquet(control_path)

    close_map: dict = {}
    if {'date', 'time', 'price'}.issubset(control_all.columns):
        # Vectorized approach: parse timestamps once, group by date, and pick the closest sample to 16:00 within +/-5 minutes
        ts_all = pd.to_datetime(control_all['date'].astype(str) + ' ' + control_all['time'].astype(str), errors='coerce')
        ts_all = normalize_to_eastern(ts_all)
        valid_mask = ~ts_all.isna()
        df_all = control_all.loc[valid_mask].copy()
        df_all['ts'] = ts_all.loc[valid_mask].values
        df_all['date_e'] = df_all['ts'].dt.date
        df_all['mins'] = df_all['ts'].dt.hour * 60 + df_all['ts'].dt.minute
        df_all['price_f'] = df_all['price'].astype(float)

        target_min = REGULAR_SESSION_END.hour * 60 + REGULAR_SESSION_END.minute
        tol = 5
        window_lo, window_hi = target_min - tol, target_min + tol

        def pick_close(g: pd.DataFrame) -> pd.Series:
            within = g.loc[(g['mins'] >= window_lo) & (g['mins'] <= window_hi)]
            if within.empty:
                return pd.Series({'close': None})
            # pick the row whose minutes is closest to target_min
            idx = (within['mins'] - target_min).abs().idxmin()
            return pd.Series({'close': float(g.loc[idx, 'price_f'])})

        close_df = df_all.groupby('date_e', sort=True).apply(pick_close)
        # convert to dict mapping python date -> float or None
        close_map = close_df['close'].to_dict()
    else:
        logger.warning("Cannot compute close map: control file lacks 'date'/'time'/'price' columns")

    # Prepare arrays for nearest-available lookup
    available_dates = sorted(close_map.keys())
    if available_dates:
        avail_dt64 = np.array(pd.to_datetime(available_dates))
    else:
        avail_dt64 = np.array([], dtype='datetime64[ns]')

    # Now compute pre_effect_log and post_effect_log per the session rules
    # price_t0: minute-level price at the row
    price_t0 = pd.to_numeric(merged.get('price'), errors='coerce')
    event_dates = np.array(pd.to_datetime(merged['event_date']).astype('datetime64[ns]'))

    pre_effect = np.full(len(merged), np.nan, dtype=float)
    post_effect = np.full(len(merged), np.nan, dtype=float)

    # Vectorized computation of pre_effect_log and post_effect_log
    # Build arrays of available dates and closes
    available_dates = sorted(close_map.keys())
    if available_dates:
        avail_dt64 = np.array(pd.to_datetime(available_dates))
        close_vals = np.array([np.nan if close_map[d] is None else float(close_map[d]) for d in available_dates], dtype=float)
    else:
        avail_dt64 = np.array([], dtype='datetime64[ns]')
        close_vals = np.array([], dtype=float)

    # event_dates as datetime64
    event_dt64 = np.array(pd.to_datetime(merged['event_date']).astype('datetime64[ns]'))
    if avail_dt64.size == 0:
        # no available closes -> all remain NaN
        merged['pre_effect_log'] = pre_effect
        merged['post_effect_log'] = post_effect
    else:
        pos = np.searchsorted(avail_dt64, event_dt64, side='left')

        # same-day mask
        same_day = (pos < len(avail_dt64)) & (avail_dt64[pos] == event_dt64)

        # previous index (strictly before)
        prev_idx = pos - 1
        prev_close = np.full(len(event_dt64), np.nan, dtype=float)
        valid_prev = prev_idx >= 0
        prev_clip = np.clip(prev_idx, 0, len(close_vals) - 1)
        prev_close[valid_prev] = close_vals[prev_clip[valid_prev]]

        # same-day close
        same_close = np.full(len(event_dt64), np.nan, dtype=float)
        same_pos = pos
        valid_same = same_day
        same_close[valid_same] = close_vals[same_pos[valid_same]]

        # next-day close: if same_day -> pos+1 else pos; must be strictly after
        next_idx = np.where(same_day, pos + 1, pos)
        next_close = np.full(len(event_dt64), np.nan, dtype=float)
        valid_next = (next_idx < len(close_vals)) & (avail_dt64[next_idx] > event_dt64)
        next_clip = np.clip(next_idx, 0, len(close_vals) - 1)
        next_close[valid_next] = close_vals[next_clip[valid_next]]

        p0 = pd.to_numeric(merged.get('price'), errors='coerce').to_numpy(dtype=float)

        # compute pre_effect_log: ln(price_t0 / prev_close)
        mask_pre = (~np.isnan(p0)) & (~np.isnan(prev_close)) & (p0 > 0) & (prev_close > 0)
        pre_effect[mask_pre] = np.log(p0[mask_pre] / prev_close[mask_pre])

        # compute post_effect_log depending on session
        sess_arr = merged['session'].values
        is_after = sess_arr == 'after'

        mask_post_after = is_after & (~np.isnan(p0)) & (~np.isnan(next_close)) & (p0 > 0) & (next_close > 0)
        post_effect[mask_post_after] = np.log(next_close[mask_post_after] / p0[mask_post_after])

        mask_post_other = (~is_after) & (~np.isnan(p0)) & (~np.isnan(same_close)) & (p0 > 0) & (same_close > 0)
        post_effect[mask_post_other] = np.log(same_close[mask_post_other] / p0[mask_post_other])

        merged['pre_effect_log'] = pre_effect
        merged['post_effect_log'] = post_effect

    merged['pre_effect_log'] = pre_effect
    merged['post_effect_log'] = post_effect

    # Print counts for NaNs in the new effect columns
    n_total = len(merged)
    n_pre_na = merged['pre_effect_log'].isna().sum()
    n_post_na = merged['post_effect_log'].isna().sum()
    n_both_na = merged[merged['pre_effect_log'].isna() & merged['post_effect_log'].isna()].shape[0]
    print(f"\nEffect columns NaN summary (rows: {n_total}):")
    print(f"  pre_effect_log NaNs: {n_pre_na} ({n_pre_na / n_total * 100 if n_total else 0:.2f}%)")
    print(f"  post_effect_log NaNs: {n_post_na} ({n_post_na / n_total * 100 if n_total else 0:.2f}%)")
    print(f"  both NaNs: {n_both_na} ({n_both_na / n_total * 100 if n_total else 0:.2f}%)")

    # --- Per-column NaN diagnostics (before dropna) ---
    na_counts = merged.isna().sum().sort_values(ascending=False)
    # Print top 40 columns with most NaNs
    top_na = na_counts.head(40)
    print('\nTop columns by NaN count (top 40):')
    for col, cnt in top_na.items():
        pct = cnt / len(merged) * 100 if len(merged) else 0
        print(f'  {col}: {cnt} NaNs ({pct:.2f}%)')


    # Diagnostics: identify rows that fall after the last available covariate date
    cov_dates = sorted(set(pd.to_datetime(cov_df_prepared.index).date))
    if cov_dates:
        last_cov_date = cov_dates[-1]
    else:
        last_cov_date = None
    # also compute first covariate date
    if cov_dates:
        first_cov_date = cov_dates[0]
    else:
        first_cov_date = None

    initial_count = len(merged)
    # how many rows have event_date > last_cov_date (i.e., no same-day or previous covariate possible)
    if last_cov_date is not None:
        rows_after_cov_end = (merged['event_date'] > last_cov_date).sum()
        print(f'Last covariate date: {last_cov_date}; control rows with event_date after that: {rows_after_cov_end}')
    else:
        rows_after_cov_end = 0
        print('No covariate dates found in covariates file')

    # How many rows have no cov_date assigned (these were logged earlier)
    no_cov_date = merged['cov_date'].isna().sum()
    print(f'Rows with no cov_date assigned: {no_cov_date}')

    # How many rows have event_date before the first covariate date
    if first_cov_date is not None:
        rows_before_cov_start = (merged['event_date'] < first_cov_date).sum()
        print(f'First covariate date: {first_cov_date}; control rows with event_date before that: {rows_before_cov_start}')
    else:
        rows_before_cov_start = 0
        print('No covariate dates found in covariates file')

    # Now perform the original drop: drop any row with any NaN and report counts
    cleaned = merged.dropna(how='any')
    dropped = initial_count - len(cleaned)

    # Of the dropped rows, how many had event_date after last_cov_date or before first_cov_date?
    dropped_rows = merged[~merged.index.isin(cleaned.index)]
    if last_cov_date is not None:
        dropped_after_cov_end = (dropped_rows['event_date'] > last_cov_date).sum()
    else:
        dropped_after_cov_end = 0

    if first_cov_date is not None:
        dropped_before_cov_start = (dropped_rows['event_date'] < first_cov_date).sum()
    else:
        dropped_before_cov_start = 0

    print(f'Rows before dropna: {initial_count}; dropped: {dropped}; remaining: {len(cleaned)}')
    print(f'Of dropped rows, {dropped_after_cov_end} have event_date after last covariate date ({last_cov_date})')
    print(f'Of dropped rows, {dropped_before_cov_start} have event_date before first covariate date ({first_cov_date})')

    # write parquet
    # reset index for clean output
    cleaned = cleaned.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_path, index=False)
    print(f'Wrote aligned data to {out_path} (rows: {len(cleaned)})')


if __name__ == '__main__':
    main()
