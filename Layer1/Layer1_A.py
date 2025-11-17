import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 1) Read and concatenate option-data parquet files
option_files = [
    r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\qqq_optiondata_1.parquet",
    r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\qqq_optiondata_2.parquet",
    r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\qqq_optiondata_3.parquet",
    r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\qqq_optiondata_4.parquet",
    r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\qqq_optiondata_5.parquet",
    r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\qqq_optiondata_6.parquet",
    r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\qqq_optiondata_7.parquet",
]
logging.info(f"Reading option files: {option_files}")

option_dfs = [pd.read_parquet(fp) for fp in option_files]
options_all = pd.concat(option_dfs, ignore_index=True)
logging.info(f"Concatenated options shape: {options_all.shape}")

# Rename and normalize date
options_all.rename(columns={"tradeDate": "Date"}, inplace=True)
options_all["Date"] = pd.to_datetime(options_all["Date"]).dt.normalize()

# Flatten MultiIndex if necessary
if isinstance(options_all.columns, pd.MultiIndex):
    options_all.columns = options_all.columns.get_level_values(0)
    logging.info("Flattened price DF columns")

# Quick peek
if options_all.empty:
    logging.error("Merged DataFrame is empty! No overlapping dates found.")
else:
    logging.info("First 5 rows of merged DataFrame:")
    print(options_all.head(5).to_string(index=False))
    logging.info("Last 5 rows of merged DataFrame:")
    print(options_all.tail(5).to_string(index=False))

def compute_option_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Layer-1A: build daily option features (features ONLY; no returns/sentiment).

    Key ideas:
      • Tenorized features (D01..D>60) for ATM IV, RR25, PCR, TURNOVER, REL_SPREAD.
      • Vega/dollar-vol weighting; OTM-only for PCR.
      • No thresholds or winsorization here (do in Layer-1B/1C).

    Expected inputs:
      Date, expirDate, dte, strike, stockPrice/spotPrice, residualRate,
      callVolume, putVolume, callOpenInterest, putOpenInterest,
      callBidPrice, callAskPrice, putBidPrice, putAskPrice,
      callMidIv, putMidIv, delta (optional), vega
    """
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["expirDate"] = pd.to_datetime(df["expirDate"]).dt.normalize()
    df["dte"] = df["dte"].astype(float)

    # Exclude only D0 (today expiry). Keep D1+ for same-day/intraday reaction studies.
    df = df[df["dte"] >= 1].copy()

    # Choose a spot: prefer spotPrice if present; else stockPrice
    if "spotPrice" in df.columns and df["spotPrice"].notna().any():
        df["spot"] = df["spotPrice"]
    else:
        df["spot"] = df["stockPrice"]

    # Forward & moneyness
    r = df.get("residualRate", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["Fwd"] = df["spot"] * np.exp(r * df["dte"] / 365.0)
    df["logM"] = np.log(df["strike"] / df["Fwd"])

    # 🔹 Convert inf/-inf to NaN, then drop rows where Fwd or logM is invalid
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Fwd", "logM"])


    # Quote cleaning & relative spreads
    def _rel_spread(bid, ask):
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid

    c_bid = df["callBidPrice"].astype(float)
    c_ask = df["callAskPrice"].astype(float)
    df["c_mid"] = (c_bid + c_ask) / 2.0
    df["c_rel_spread"] = _rel_spread(c_bid, c_ask)

    p_bid = df["putBidPrice"].astype(float)
    p_ask = df["putAskPrice"].astype(float)
    df["p_mid"] = (p_bid + p_ask) / 2.0
    df["p_rel_spread"] = _rel_spread(p_bid, p_ask)

    valid = (
    (c_bid > 0) & (c_ask > 0) & (c_ask >= c_bid) &
    (p_bid > 0) & (p_ask > 0) & (p_ask >= p_bid) &
    (df["c_rel_spread"].between(0, 1)) &
    (df["p_rel_spread"].between(0, 1)) &
    (df["spot"] > 0) &
    (df["c_mid"] > 0) & (df["p_mid"] > 0)   # ✅ new guard
    )
    df = df[valid].copy()


    # Dollar flow & vega weights
    contract_mult = 100.0
    df["c_dollar_vol"] = df["c_mid"].clip(lower=1e-6) * df["callVolume"].clip(lower=0) * contract_mult
    df["p_dollar_vol"] = df["p_mid"].clip(lower=1e-6) * df["putVolume"].clip(lower=0) * contract_mult
    df["vega_w"] = df["vega"].astype(float).clip(lower=1e-8)

    # Tenor buckets
    df["tenor"] = pd.cut(
        df["dte"],
        bins=[0, 1, 2, 3, 5, 7, 10, 40, 80, np.inf],
        labels=["D01", "D02", "D03", "D05", "D07", "D10", "D30", "D60", "D>60"],
        right=True,
        include_lowest=True
    )

    # ATM & 25Δ bands
    if "delta" in df.columns:
        atm_mask = df["delta"].between(0.45, 0.55) | df["logM"].abs().le(0.02)
        df["put_abs_delta"] = (df["delta"] - 1.0).abs()
        call_25 = df["delta"].between(0.20, 0.30)
        put_25  = df["put_abs_delta"].between(0.20, 0.30)
    else:
        atm_mask = df["logM"].abs().le(0.02)
        call_25 = df["logM"].between(0.06, 0.12)
        put_25  = df["logM"].between(-0.12, -0.06)

    call_otm = df["strike"] > df["Fwd"]
    put_otm  = df["strike"] < df["Fwd"]

    def _safe_vw_mean(x, w):
        wsum = w.sum()
        return np.nan if wsum == 0 else (x * w).sum() / wsum

    g = df.groupby(["Date", "tenor"], observed=True)
    agg = g.apply(lambda s: pd.Series({
        "ATM_IV": _safe_vw_mean(
            pd.concat([s.loc[atm_mask.loc[s.index], "callMidIv"],
                       s.loc[atm_mask.loc[s.index], "putMidIv"]]).astype(float),
            pd.concat([s.loc[atm_mask.loc[s.index], "vega_w"],
                       s.loc[atm_mask.loc[s.index], "vega_w"]]).astype(float)
        ),
        "RR25": (
            _safe_vw_mean(s.loc[put_25.loc[s.index], "putMidIv"].astype(float),
                          s.loc[put_25.loc[s.index], "vega_w"].astype(float))
            -
            _safe_vw_mean(s.loc[call_25.loc[s.index], "callMidIv"].astype(float),
                          s.loc[call_25.loc[s.index], "vega_w"].astype(float))
        ),
        "PCR_DOLLAR_OTM": (
            s.loc[put_otm.loc[s.index], "p_dollar_vol"].sum()
            / max(s.loc[call_otm.loc[s.index], "c_dollar_vol"].sum(), 1e-8)
        ),
        "PCR_VEGA_OTM": (
            (s.loc[put_otm.loc[s.index], "vega_w"] * s.loc[put_otm.loc[s.index], "putVolume"].clip(lower=0)).sum()
            / max((s.loc[call_otm.loc[s.index], "vega_w"] * s.loc[call_otm.loc[s.index], "callVolume"].clip(lower=0)).sum(), 1e-8)
        ),
        "TURNOVER": (
            (s["callVolume"].clip(lower=0).sum() + s["putVolume"].clip(lower=0).sum())
            / max(s["callOpenInterest"].clip(lower=0).sum() + s["putOpenInterest"].clip(lower=0).sum(), 1e-8)
        ),
        "REL_SPREAD": _safe_vw_mean(
            pd.concat([s["c_rel_spread"], s["p_rel_spread"]]).astype(float),
            pd.concat([s["c_dollar_vol"], s["p_dollar_vol"]]).astype(float)
        ),
        "DOLLAR_VOL": s["c_dollar_vol"].sum() + s["p_dollar_vol"].sum(),
        "OI_SUM": s["callOpenInterest"].clip(lower=0).sum() + s["putOpenInterest"].clip(lower=0).sum(),
    })).reset_index()

    # Pivot wide (one row per Date)
    wide = agg.pivot(index="Date", columns="tenor")
    wide.columns = ["_".join([c for c in col if c]).strip("_") for col in wide.columns.to_flat_index()]
    wide = wide.sort_index()

    # Term structure from ATM IVs
    for a, b, name in [("D07", "D30", "TS_7_30"), ("D30", "D60", "TS_30_60")]:
        num = wide.get(f"ATM_IV_{a}")
        den = wide.get(f"ATM_IV_{b}")
        if num is not None and den is not None:
            wide[name] = (num / den) - 1.0

    # Skew change (DoD) on 30D
    if "RR25_D30" in wide.columns:
        wide["SKEW_DECAY_30D"] = wide["RR25_D30"].diff().fillna(0.0)

    # Day-level OI structure
    daily_oi = df.groupby("Date").agg(
        OI_call=("callOpenInterest", "sum"),
        OI_put=("putOpenInterest", "sum"),
        VOL_call=("callVolume", "sum"),
        VOL_put=("putVolume", "sum"),
    )
    daily_oi["PCR_OI"] = daily_oi["OI_put"] / daily_oi["OI_call"].clip(lower=1.0)
    daily_oi["NET_OI"] = (daily_oi["OI_put"] - daily_oi["OI_call"]) / (daily_oi["OI_put"] + daily_oi["OI_call"]).replace(0, np.nan)

    # Join and return (indexed by Date)
    features = wide.join(daily_oi[["PCR_OI", "NET_OI"]], how="left")
    features = features.sort_index()
    return features

if __name__ == "__main__":
    out_path = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer1\Options_features_L1_A.parquet"

    # 1) Compute features ONLY
    daily_features = compute_option_features(options_all)

    # 2) Ensure Date is unique & sorted
    daily_features = daily_features[~daily_features.index.duplicated(keep="first")].sort_index()

    # 3) Save features-only parquet
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    daily_features.reset_index().to_parquet(out_path, index=False)

    print(f"✅ Saved features-only dataset to: {out_path}")
    print(f"Rows: {len(daily_features):,} | Cols: {len(daily_features.columns):,}")
