"""Alpha Vantage Intraday Price Provider for QQQ (or any symbol)
--------------------------------------------------------------------------
Minimal, cache-first accessor used by Layer 3 event reaction builder.

Design goals
  • Avoid re-downloading months (intraday endpoint pages by month=YYYY-MM)
  • Keep API key out of source control (read from env var ALPHAVANTAGE_API_KEY)
  • Provide a simple price_at(t0) lookup with tolerance for missing exact minute
  • Consistent adjustment: user chooses adjusted True/False once
    • Extended hours always enabled (pre + post market)

Caching layout (relative to cache_dir):
    intraday/1min/2024-07.parquet
    (daily caching removed)

Note: This module purposefully does NOT hardcode any API key.
Set in PowerShell before running:
  $env:ALPHAVANTAGE_API_KEY = "YOUR_KEY_HERE"

If you ever accidentally commit a key, revoke it immediately.
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List

# Auto-load .env (silent if python-dotenv not installed)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import pandas as pd
import requests

ALPHAVANTAGE_BASE = "https://www.alphavantage.co/query"


@dataclass
class ProviderConfig:
    symbol: str = "QQQ"
    interval: str = "1min"  
    adjusted: bool = False
    extended_hours: bool = True
    tolerance_minutes: int = 5
    cache_dir: Path = Path("C:/Users/larbi/Desktop/My Doc/AICAUSAL/Layer3")  # NOTE: Adjust the file path below to match your local setup.
    requests_per_second: float = 5.0 


class AlphaVantageError(RuntimeError):
    pass


class AlphaVantagePriceProvider:
    def __init__(self, cfg: ProviderConfig):
        """Initialize provider (intraday-only)."""
        self.cfg = cfg
        self.api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise AlphaVantageError("Environment variable ALPHAVANTAGE_API_KEY not set. Do NOT hardcode keys.")
        self._intraday_month_cache: Dict[str, pd.DataFrame] = {}
        self._last_call_time: float = 0.0
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._verbose = True

    def _log(self, msg: str):
        """Internal conditional logger (enabled via self._verbose)."""
        if self._verbose:
            print(f"[AlphaVantage] {msg}")

    # ---------------------- Networking helpers ----------------------
    def _respect_rate_limit(self):
        """Sleep just enough so we do not exceed configured requests_per_second."""
        min_interval = 1.0 / self.cfg.requests_per_second
        el = time.time() - self._last_call_time
        if el < min_interval:
            time.sleep(min_interval - el)

    def _get(self, params: Dict[str, str], expect_csv: bool = False) -> str:
        """Generic GET wrapper.
        Adds api key, respects rate limit, returns either raw CSV text or JSON string.
        Raises AlphaVantageError on HTTP errors or throttle notes.
        """
        self._respect_rate_limit()
        params["apikey"] = self.api_key
        r = requests.get(ALPHAVANTAGE_BASE, params=params, timeout=30)
        self._last_call_time = time.time()
        if r.status_code != 200:
            raise AlphaVantageError(f"HTTP {r.status_code}: {r.text[:200]}")
        # Alpha Vantage returns JSON with 'Note' when throttled, even if 200
        if expect_csv:
            txt = r.text
            if txt.startswith("{") and '"Note"' in txt:
                raise AlphaVantageError("Rate limited (Note in CSV mode). Slow down or retry later.")
            return txt
        else:
            js = r.json()
            if any(k in js for k in ("Note", "Information")):
                raise AlphaVantageError(js.get("Note") or js.get("Information"))
            return json.dumps(js)


    # ---------------------- Intraday monthly ----------------------
    def load_intraday_month(self, month: str, force: bool = False) -> pd.DataFrame:
        """Fetch a single month of intraday 1-min data.
        NOTE: Does NOT persist to disk (only in-memory) to save storage.
        Parameters
        ----------
        month : str  YYYY-MM
        force : bool if True re-fetch even if already cached in memory
        Returns
        -------
        DataFrame with tz-aware 'timestamp' (America/New_York) and OHLCV subset.
        """
        if month in self._intraday_month_cache and not force:
            return self._intraday_month_cache[month]

        self._log(f"Fetching INTRADAY month={month} interval={self.cfg.interval} adjusted={self.cfg.adjusted}")
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": self.cfg.symbol,
            "interval": self.cfg.interval,
            "month": month,
            "outputsize": "full",
            "datatype": "csv",
            "extended_hours": str(self.cfg.extended_hours).lower(),
            "adjusted": str(self.cfg.adjusted).lower(),
        }
        txt = self._get(params, expect_csv=True)
        from io import StringIO
        df = pd.read_csv(StringIO(txt))
        if "timestamp" not in df.columns:
            raise AlphaVantageError(f"Intraday CSV missing timestamp column. Columns={df.columns.tolist()}")
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("America/New_York")
        keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].sort_values("timestamp")
        self._log(f"Downloaded {len(df):,} rows for {month} (not saved to disk)")
        self._intraday_month_cache[month] = df
        return df

    # ---------------------- Convenience lookups ----------------------
    def get_price_at(self, ts_et: pd.Timestamp) -> Optional[float]:
        """Return minute close price at or before ts_et within tolerance.
        - Converts naive timestamps to America/New_York.
        - Loads relevant month intraday data into memory if not loaded.
        - Finds last bar <= ts_et and checks freshness (minutes difference <= tolerance).
        Returns float price or None if no acceptable bar.
        """
        if ts_et.tzinfo is None:
            ts_et = ts_et.tz_localize("America/New_York")
        ts_month = ts_et.strftime("%Y-%m")
        df = self.load_intraday_month(ts_month)
        # Filter up to ts_et
        sub = df[df["timestamp"] <= ts_et]
        if sub.empty:
            self._log(f"No intraday data before {ts_et} (month {ts_month})")
            return None
        last_row = sub.iloc[-1]
        delta_minutes = (ts_et - last_row["timestamp"]).total_seconds() / 60.0
        if delta_minutes > self.cfg.tolerance_minutes:
            self._log(f"Last bar {last_row['timestamp']} is {delta_minutes:.1f}m older than tolerance {self.cfg.tolerance_minutes}m")
            return None
        try:
            return float(last_row["close"])
        except Exception:
            return None


    def ensure_months(self, months: List[str]):
        """Preload a list of months (YYYY-MM) into memory, ignoring failures."""
        for m in sorted(set(months)):
            try:
                self.load_intraday_month(m)
            except AlphaVantageError as e:
                print(f"Warning: failed to load {m}: {e}")

__all__ = [
    "ProviderConfig",
    "AlphaVantagePriceProvider",
    "AlphaVantageError",
]

# ---------------------- Bulk Backfill Helper ----------------------
def month_range(start_month: str, end_month: str):
    """Inclusive list of YYYY-MM between start_month and end_month."""
    start = pd.to_datetime(start_month + "-01")
    end = pd.to_datetime(end_month + "-01")
    months = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        # add one month
        year = cur.year + (cur.month // 12)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = pd.Timestamp(year=year, month=month, day=1)
    return months


def backfill_range(provider: AlphaVantagePriceProvider, start_month: str, end_month: str, out_parquet: str):
    """Backfill all months into a single consolidated parquet.
    - Iterates month_range.
    - Fetches each month intraday (in-memory only).
    - Concatenates, deduplicates timestamps, extracts date/time/price columns.
    - Writes parquet to requested path; prints head/tail preview.
    Returns None.
    """
    months = month_range(start_month, end_month)
    frames = []
    total = len(months)
    for idx, m in enumerate(months, 1):
        print(f"[{idx}/{total}] Fetching month {m} ...")
        try:
            dfm = provider.load_intraday_month(m)
        except AlphaVantageError as e:
            print(f"  Skipped {m} (error: {e})")
            continue
        # Ensure expected columns
        if 'timestamp' not in dfm.columns or 'close' not in dfm.columns:
            print(f"  Skipped {m} (missing columns)")
            continue
        frames.append(dfm[['timestamp','close']].copy())
    if not frames:
        raise AlphaVantageError("No data downloaded; aborting write.")
    full = pd.concat(frames, ignore_index=True).sort_values('timestamp')
    # Deduplicate in case of overlaps (keep last occurrence)
    full = full.drop_duplicates('timestamp', keep='last')
    # Build required columns
    # timestamp is ET already; derive date/time parts
    full['date'] = full['timestamp'].dt.strftime('%Y-%m-%d')
    full['time'] = full['timestamp'].dt.strftime('%H:%M:%S')
    full['price'] = full['close'].astype(float)
    out_df = full[['date','time','price']]
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"✅ Wrote consolidated minute file → {out_parquet}  (rows={len(out_df):,})")
    # Preview first / last 5 rows
    try:
        print("First 5 rows:")
        print(out_df.head(5).to_string(index=False))
        print("Last 5 rows:")
        print(out_df.tail(5).to_string(index=False))
    except Exception as e:
        print(f"Preview failed: {e}")


if __name__ == "__main__":
    """Entry point to run a historical 1-min intraday backfill producing one consolidated parquet under Layer3."""
    START_MONTH = "2011-01"
    END_MONTH = "2025-07" 
    OUTPUT_PARQUET = "C:/Users/larbi/Desktop/My Doc/AICAUSAL/Layer3/QQQ_1min_2011-01_2025-07.parquet"

    cfg = ProviderConfig(symbol="QQQ", interval="1min", adjusted=False, extended_hours=True)
    provider = AlphaVantagePriceProvider(cfg)
    print(f"Starting intraday backfill {START_MONTH} → {END_MONTH} (symbol={cfg.symbol}, adjusted={cfg.adjusted})")
    backfill_range(provider, START_MONTH, END_MONTH, OUTPUT_PARQUET)
    print("Done intraday backfill.")
