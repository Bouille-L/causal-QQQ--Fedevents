
"""
Layer 1C — Final Option-Feature Cleaning (drops short-tenor buckets)

Pipeline:
  Input  : Layer 1A features  -> Options_features_L1_A.parquet
  Drop   : D01, D02, D03, D05, D07 (and derived short-tenor features like TS_7_30)
  Filter : per-tenor thresholds (REL_SPREAD max, OI_SUM min,min_Dollar_VOL) — mask violators to NaN
  Winsor : clamp extreme ratios (PCR_*, TURNOVER_*) at 1–99th pct caps (from Layer 1B)
  Output :
      Options_features_L1_C.parquet  (clean features for Layer 3)
      L1C_cleaning_summary.csv       (what was masked/clipped)
      L1C_cleaning_config_used.json  (exact thresholds/caps used)
      L1C_columns_kept.csv           (manifest of columns that survived; useful for thesis appendix)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

# -------- PATHS --------
IN_PARQUET  = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\Options_features_L1_A.parquet"
OUT_PARQUET = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\Options_features_L1_VF.parquet"
OUT_SUMMARY = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\L1C_cleaning_summary.csv"
OUT_CONFIG  = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\L1C_cleaning_config_used.json"
OUT_COLS    = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\L1C_columns_kept.csv"

# -------- DROPS (from Layer 1B decision) --------
# Remove all columns that end with these tenor suffixes — they had very high NaN%
DROP_TENORS = ["D01", "D02", "D03", "D05", "D07"]
# Keep only these tenors downstream
KEEP_TENORS = ["D10", "D30", "D60", "D>60"]
# Also drop short-tenor-derived features (e.g., TS_7_30 depends on D07)
EXTRA_DROP = ["TS_7_30"]

# -------- THRESHOLDS (per-tenor) --------
# Keep tenor on a given row only if:
#    REL_SPREAD_[tenor] <= max_rel_spread  AND  OI_SUM_[tenor] >= min_oi_sum
# Otherwise we mask ALL columns for that tenor on that row (set to NaN).
THRESHOLDS = {
    "D10":  {"max_rel_spread": 0.082438958, "min_oi_sum":   84_088, "min_Dollar_VOL":4340080.4},
    "D30":  {"max_rel_spread": 0.048004026, "min_oi_sum": 1_055_642, "min_Dollar_VOL":13692734},
    "D60":  {"max_rel_spread": 0.032515467, "min_oi_sum":  471_790.6,"min_Dollar_VOL":8443763.2},
    "D>60": {"max_rel_spread": 0.054741753, "min_oi_sum": 1_097_391.2,"min_Dollar_VOL":11998522},
}

# -------- WINSOR CAPS (1–99th pct from Layer 1B) --------
# These tame heavy-tailed ratios but preserve rank information.
CAPS = {
    # PCR (Dollar, OTM)
    "PCR_DOLLAR_OTM_D10":  (0.329999621, 11.44671522),
    "PCR_DOLLAR_OTM_D30":  (0.414203119,  7.864405411),
    "PCR_DOLLAR_OTM_D60":  (0.272802928, 15.79014214),
    "PCR_DOLLAR_OTM_D>60": (0.355047705, 14.07606868),
    # PCR (Vega, OTM)
    "PCR_VEGA_OTM_D10":    (0.376716308,  7.859420988),
    "PCR_VEGA_OTM_D30":    (0.389128312,  6.599157238),
    "PCR_VEGA_OTM_D60":    (0.232544033, 11.5378895),
    "PCR_VEGA_OTM_D>60":   (0.291292676, 12.33069865),
    # TURNOVER
    "TURNOVER_D10":        (0.061266597,  3.86015e12),
    "TURNOVER_D30":        (0.045481492,  0.413044831),
    "TURNOVER_D60":        (0.018179703,  0.352562624),
    "TURNOVER_D>60":       (0.007651972,  0.124931889),
}

# -------- Helpers --------
def drop_tenor_columns(df: pd.DataFrame, tenors_to_drop: list[str]) -> pd.DataFrame:
    """Drop all columns whose names end with any of the given tenor suffixes."""
    drop_cols = []
    for t in tenors_to_drop:
        suf = f"_{t}"
        drop_cols.extend([c for c in df.columns if c.endswith(suf)])
    return df.drop(columns=sorted(set(drop_cols)), errors="ignore")

def tenor_columns(df: pd.DataFrame, tenor: str) -> list[str]:
    """Return all columns for a given tenor suffix, e.g., *_D30."""
    suffix = f"_{tenor}"
    return [c for c in df.columns if c.endswith(suffix)]

def apply_thresholds_per_tenor(df: pd.DataFrame, thresholds: dict) -> tuple[pd.DataFrame, dict]:
    """
    Row-wise masking per tenor:
      Keep a row for a tenor only if all available checks pass:
        REL_SPREAD_[tenor] <= max_rel_spread
        OI_SUM_[tenor]      >= min_oi_sum
        Dollar_VOL_[tenor]  >= min_Dollar_VOL  (if provided in thresholds)
      Rows failing any available check get ALL columns for that tenor masked (NaN).
    Returns:
      df_out: masked dataframe
      stats : {tenor: {"masked_rows": int, "total_rows": int, "note": str}}
    """
    out = df.copy()
    stats = {}

    for tenor, cfg in thresholds.items():
        rel_col = f"REL_SPREAD_{tenor}"
        oi_col  = f"OI_SUM_{tenor}"
        dv_col  = f"DOLLAR_VOL_{tenor}"

        # Coerce present columns to numeric to avoid string compare issues
        for col in (rel_col, oi_col, dv_col):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        checks_applied = 0
        notes = []

        good = pd.Series(True, index=out.index)

        if rel_col in out.columns:
            checks_applied += 1
            good &= out[rel_col] <= cfg.get("max_rel_spread", np.inf)
        else:
            notes.append("REL_SPREAD missing")

        if oi_col in out.columns:
            checks_applied += 1
            good &= out[oi_col] >= cfg.get("min_oi_sum", -np.inf)
        else:
            notes.append("OI_SUM missing")

        # Apply Dollar_VOL only if a threshold is provided for this tenor
        if "min_Dollar_VOL" in cfg:
            if dv_col in out.columns:
                checks_applied += 1
                good &= out[dv_col] >= cfg["min_Dollar_VOL"]
            else:
                notes.append("Dollar_VOL missing")

        if checks_applied == 0:
            stats[tenor] = {"masked_rows": 0, "total_rows": int(len(out)), "note": "columns missing"}
            continue

        cols_to_mask = tenor_columns(out, tenor)
        masked_rows = int((~good).sum())
        out.loc[~good, cols_to_mask] = np.nan

        s = {"masked_rows": masked_rows, "total_rows": int(len(out))}
        if notes:
            s["note"] = "; ".join(notes)
        stats[tenor] = s

    return out, stats

def winsorize(df: pd.DataFrame, caps: dict) -> tuple[pd.DataFrame, dict]:
    """
    Clip listed columns to (low, high) caps.
    Returns:
      df_out: winsorized dataframe
      counts: {col: {"low_clipped": int, "high_clipped": int}}
    """
    out = df.copy()
    counts = {}

    # Coerce targets to numeric (defensive) before clipping
    for col in caps.keys():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col, (lo, hi) in caps.items():
        if col not in out.columns:
            continue
        x = out[col]
        low_hits  = int((x < lo).sum(skipna=True))
        high_hits = int((x > hi).sum(skipna=True))
        out[col] = x.clip(lower=lo, upper=hi)
        counts[col] = {"low_clipped": low_hits, "high_clipped": high_hits}
    return out, counts

def main():
    # ---- Load Layer 1A ----
    df = pd.read_parquet(IN_PARQUET)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

    # ---- Drop short-tenor columns decided in Layer 1B ----
    before_cols = len(df.columns)
    df = drop_tenor_columns(df, DROP_TENORS)
    # Also drop any derived short-tenor columns
    df = df.drop(columns=[c for c in EXTRA_DROP if c in df.columns], errors="ignore")
    after_cols  = len(df.columns)
    print(f"Dropped short-tenor columns: {before_cols - after_cols} removed "
          f"(kept tenors: {', '.join(KEEP_TENORS)})")

    # ---- Apply thresholds per kept tenor (mask bad rows per tenor) ----
    df_thr, thr_stats = apply_thresholds_per_tenor(df, THRESHOLDS)

    # ---- Winsorize extreme ratios ----
    df_cap, cap_stats = winsorize(df_thr, CAPS)

    # ---- Optional: drop columns that became entirely NaN after masking ----
    df_cap = df_cap.dropna(axis=1, how="all")

    # ---- Save outputs ----
    Path(OUT_PARQUET).parent.mkdir(parents=True, exist_ok=True)
    df_cap.reset_index().to_parquet(OUT_PARQUET, index=False)

    # Cleaning summary CSV
    rows = []
    for tenor, s in thr_stats.items():
        rows.append({
            "what": "threshold_mask", "target": tenor,
            "masked_rows": s.get("masked_rows", 0),
            "total_rows": s.get("total_rows", len(df_cap)),
            "masked_share": s.get("masked_rows", 0) / max(1, s.get("total_rows", len(df_cap))),
            "note": s.get("note", "")
        })
    for col, s in cap_stats.items():
        rows.append({
            "what": "winsor_clip", "target": col,
            "low_clipped": s["low_clipped"], "high_clipped": s["high_clipped"]
        })
    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False)

    # Config JSON (for exact reproducibility in thesis/paper)
    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump({
            "drop_tenors": DROP_TENORS,
            "keep_tenors": KEEP_TENORS,
            "extra_drop": EXTRA_DROP,
            "thresholds": THRESHOLDS,
            "caps": CAPS
        }, f, indent=2)

    # Columns kept manifest (handy for Layer 3 & appendix)
    (df_cap.columns.to_series()
          .to_frame("column")
          .assign(tenor=df_cap.columns.to_series().str.extract(r'_(D10|D30|D60|D>60)$', expand=False))
          .to_csv(OUT_COLS, index=False))

    # Info line: how many dates still have at least one usable kept-tenor signal
    kept_cols = [c for c in df_cap.columns if c.endswith(tuple(KEEP_TENORS))]
    rows_with_any_signal = int(df_cap[kept_cols].dropna(how="all").shape[0]) if kept_cols else 0
    print(f"✅ Saved cleaned features → {OUT_PARQUET}")
    print(f"ℹ️  Cleaning summary     → {OUT_SUMMARY}")
    print(f"🧾 Config used          → {OUT_CONFIG}")
    print(f"📄 Columns kept         → {OUT_COLS}")
    print(f"📊 Rows with any kept-tenor signal: {rows_with_any_signal:,}")

if __name__ == "__main__":
    main()
