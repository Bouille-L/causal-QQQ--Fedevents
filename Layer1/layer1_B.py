"""
Layer-1B: Descriptive Statistics
- Loads Layer-1A features parquet
- Summarizes distributions per tenor & feature family
- Exports CSV summaries and a Markdown report
"""

import pandas as pd
from pathlib import Path
import logging


# Configuration
# NOTE: Adjust the file path below to match your local setup.
IN_PARQUET  = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\Options_features_L1_A.parquet"
OUT_DIR     = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\reports_layer1b"
# Percentiles to compute
PCTS = [0.01, 0.05, 0.10, 0.20, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def detect_families(df: pd.DataFrame):
    """Return tenor list and feature families present in the columns."""
    # infer tenors from suffixes
    tenors = []
    for c in df.columns:
        if "_" in c:
            suf = c.split("_")[-1]
            if suf in ["D01","D02","D03","D05","D07","D10","D30","D60","D>60"]:
                tenors.append(suf)
    tenors = sorted(set(tenors), key=lambda x: ["D01","D02","D03","D05","D07","D10","D30","D60","D>60"].index(x)) or []

    # families of features
    families = {
        "ATM_IV":        [f"ATM_IV_{t}" for t in tenors],
        "RR25":          [f"RR25_{t}" for t in tenors],
        "REL_SPREAD":    [f"REL_SPREAD_{t}" for t in tenors],
        "TURNOVER":      [f"TURNOVER_{t}" for t in tenors],
        "PCR_DOLLAR_OTM":[f"PCR_DOLLAR_OTM_{t}" for t in tenors],
        "PCR_VEGA_OTM":  [f"PCR_VEGA_OTM_{t}" for t in tenors],
        "OI_SUM":        [f"OI_SUM_{t}" for t in tenors],
        "DOLLAR_VOL":    [f"DOLLAR_VOL_{t}" for t in tenors if f"DOLLAR_VOL_{t}" in df.columns],
        "EXTRAS":        [c for c in ["SKEW_DECAY_30D", "TS_7_30", "TS_30_60", "PCR_OI", "NET_OI"] if c in df.columns]
    }
    return tenors, families,families.get("EXTRAS", [])

def summarize_family(df: pd.DataFrame, cols: list, pcts=PCTS) -> pd.DataFrame:
    """Compute NaN share and percentiles for a set of columns."""
    out = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        pct_vals = s.quantile(pcts).to_dict()
        row = {
            "column": c,
            "n": s.size,
            "n_nan": s.isna().sum(),
            "nan_share": s.isna().mean()
        }
        for p, v in pct_vals.items():
            row[f"p{int(p*100):02d}"] = v
        # simple robust center / spread
        row["median"] = s.median()
        row["iqr"]    = (s.quantile(0.75) - s.quantile(0.25))
        out.append(row)
    return pd.DataFrame(out)

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PARQUET)
    logging.info(f"Loaded features: shape={df.shape}")

    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        df = df.sort_values("Date").reset_index(drop=True)

    tenors, families, extras = detect_families(df)
    logging.info(f"Detected tenors: {tenors}")
    logging.info(f"Detected families: {list(families.keys())}")
    logging.info(f"Extra columns: {extras}")

    # 2) Percentile tables per family
    family_tables = {}
    for fam, cols in families.items():
        stats = summarize_family(df, cols)
        stats.to_csv(out_dir / f"stats_{fam.lower()}.csv", index=False)
        family_tables[fam] = stats

    logging.info("CSV outputs:")
    for p in [f"stats_{k.lower()}.csv" for k in families.keys()]:
        logging.info(f" - {out_dir}\\{p}")

if __name__ == "__main__":
    main()
