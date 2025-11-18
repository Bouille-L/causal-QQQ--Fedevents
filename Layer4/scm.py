"""
SCM pipeline (final): EWM-only surprise for macro events -> QQQ reaction
- Single treatment: EWM z-score surprise within event type (past-only)
- FOMC handled as policy change (Δ bps, midpoint when both bounds present; dose only at block starts)
- Per-event-type DRFs (separate figures/files) via g-computation
- Minimal diagnostics: lead test (optional pre-window return), overlap
- NEW: rolling time-series CV metrics (RMSE, R²) per event family
- NEW: residual-vs-fitted sanity plot for headline families
- Robust to NaNs; no Z→X predictability check; no sensitivity grids


"""
from __future__ import annotations

import os
import json
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# -----------------
# Logging
# -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------
# Configuration
# -----------------

@dataclass
class Config:
    # Input
    # NOTE: Adjust the file path below to match your local setup.
    parquet_path: str = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\Treatment_dataset_Fedmacr-_event\events_aligned_L3_E1.parquet"
    time_col: str = "time_et"
    event_type_col: str = "event_type"
    value_col: str = "values"                 # realized event value container 
    
    # Outcome (single column used throughout the code)
    outcome_col: str = "post_effect_log"      

    # pre-window return for lead test (set None to skip)
    pre_return_col: Optional[str] = "pre_effect_log"

    # Pre-event covariates (options features); strictly T-1
    market_state: List[str] = field(default_factory=lambda: [
        "ATM_IV_D30","ATM_IV_D60",
        "RR25_D30","RR25_D60",
        "DOLLAR_VOL_D30","DOLLAR_VOL_D60",
        "TS_30_60"
    ])
    liquidity: List[str] = field(default_factory=lambda: [
        "TURNOVER_D30","TURNOVER_D60",
        "REL_SPREAD_D30","REL_SPREAD_D60",
        "OI_SUM_D30","OI_SUM_D60"
    ])
    positioning: List[str] = field(default_factory=lambda: [
        "PCR_DOLLAR_OTM_D30","PCR_DOLLAR_OTM_D60",
        "PCR_VEGA_OTM_D30","PCR_VEGA_OTM_D60",
        "PCR_OI","NET_OI"
    ])

    # Calendar features (helpful, but remain pre-event)
    use_calendar: bool = True

    # Event type subset to run (leave [] for all types)
    event_types_subset: List[str] = field(default_factory=list)

    # EWM half-life per event frequency (in number of releases of that frequency)
    half_life_monthly: int = 12   # months
    half_life_weekly: int = 26    # weeks
    half_life_quarterly: int = 6  # quarters

    # Optional: per-type overrides (event_type -> half-life integer)
    half_life_by_type: Dict[str, int] = field(default_factory=dict)

    # Event frequency map (event_type -> frequency)
    frequency_by_type: Dict[str, str] = field(default_factory=lambda: {
        "CPI Release": "monthly",
        "PPI Release": "monthly",
        "PCE Release": "monthly",
        "RetailSales Release": "monthly",
        "IndustrialProduction Release": "monthly",
        "HousingStarts Release": "monthly",
        "TradePrices Release": "monthly",
        "MichiganSentiment Release": "monthly",
        "JOLTS Release": "monthly",
        "NonfarmPayrolls Release": "monthly",
        "UnemploymentRate Release": "monthly",
        "DurableGoodsOrders Release": "monthly",
        "GDP Release": "quarterly",
        "InitialJoblessClaims Release": "weekly",
        "FOMC Decision": "policy",
    })

    # Map event_type -> key inside the 'values' dict to extract numeric X_t
    value_key_by_type: Dict[str, str] = field(default_factory=lambda: {
        "CPI Release": "cpi_headline_mom",
        "PPI Release": "ppi_headline_mom",
        "PCE Release": "pce_price_index_mom",
        "RetailSales Release": "retail_sales_mom",
        "IndustrialProduction Release": "industrial_production_mom",
        "HousingStarts Release": "housing_starts_mom",
        "TradePrices Release": "import_prices_mom",
        "MichiganSentiment Release": "michigan_sentiment_mom",
        "JOLTS Release": "jolts_job_openings_mom",
        "NonfarmPayrolls Release": "nfp_payroll_change",
        "UnemploymentRate Release": "unemployment_rate",
        "DurableGoodsOrders Release": "durable_goods_new_orders_mom",
        "GDP Release": "gdp_qoq",
        "InitialJoblessClaims Release": "initial_claims_wow_change",
    })

    # FOMC keys inside 'values'
    fomc_upper: str = "fed_funds_target_upper"
    fomc_lower: str = "fed_funds_target_lower"
    

    # Outcome model for E[Y | dose, Z]
    outcome_model: str = "gbrt"  

    # DRF grid resolution
    dose_grid_points: int = 21

    # Dose grid clipping (presentation stability)
    clip_dose: bool = True
    clip_low: float = -2.5
    clip_high: float = 2.5

    # Bootstrap settings for uncertainty
    bootstrap_enabled: bool = True
    n_boot: int = 50
    ci_level: float = 0.95
    random_state: int = 42

    # Optional: include EWM-standardized FOMC Δ as additional robustness output
    standardize_fomc_delta: bool = True

    # Output directory
    out_dir: str = "scm_outputs_ewm"

    # Headline families for residual sanity plots
    residual_plot_types: List[str] = field(default_factory=lambda: [
        "CPI Release", "NonfarmPayrolls Release", "FOMC Decision"
    ])

    # Rolling/expanding CV settings
    ts_cv_folds: int = 8  # will be clipped based on sample size
    # Minimum usable events required for DRF estimation (configurable)
    min_events_for_drf: int = 60
    # If futures data missing, forward/back fill expected_rate so dose is still computable
    propagate_fomc_expected_fallback: bool = True


# =====================================================
# Utility helpers
# =====================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    logger.info(f"Ensured output directory exists: {path}")


def add_calendar(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[time_col])
    return df.assign(
        cal_year=ts.dt.year,
        cal_month=ts.dt.month,
        cal_dayofweek=ts.dt.dayofweek,
        cal_hour=ts.dt.hour,
        cal_quarter=ts.dt.quarter,
        cal_week=ts.dt.isocalendar().week.astype(int)
    )


def build_model(kind: str) -> Pipeline:
    logger.info(f"Building outcome model: {kind}")
    if kind == "elasticnet":
        base = ElasticNet(alpha=0.001, l1_ratio=0.1, max_iter=4000)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", base),
        ])
    elif kind == "rf":
        base = RandomForestRegressor(n_estimators=600, min_samples_leaf=5, random_state=42, n_jobs=-1)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base),
        ])
    else:
        base = GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.8, random_state=42
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base),
        ])


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns and pd.api.types.is_object_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =====================================================
# EWM surprise & value extraction helpers
# =====================================================

def _extract_from_values(val, key: str) -> float:
    """Extract a numeric from a row's 'values' cell."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    if isinstance(val, dict):
        v = val.get(key, np.nan)
        try:
            return float(v) if v is not None else np.nan
        except Exception:
            return np.nan
    if isinstance(val, str):
        try:
            obj = json.loads(val)
            v = obj.get(key, np.nan)
            return float(v) if v is not None else np.nan
        except Exception:
            try:
                return float(val)
            except Exception:
                return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def half_life_for_type(e: str, cfg: Config) -> Optional[int]:
    if e in cfg.half_life_by_type:
        return cfg.half_life_by_type[e]
    freq = cfg.frequency_by_type.get(e, None)
    if freq == "monthly":
        return cfg.half_life_monthly
    if freq == "weekly":
        return cfg.half_life_weekly
    if freq == "quarterly":
        return cfg.half_life_quarterly
    if freq == "policy":
        return cfg.half_life_monthly
    return cfg.half_life_monthly


def compute_ewm_surprise_per_type(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    logger.info("Computing EWM surprise per event type...")
    df = df.sort_values(cfg.time_col).copy()
    df["dose_ewm"] = np.nan

    for etype, g in df.groupby(cfg.event_type_col, sort=False):
        logger.info(f"  Processing event type: {etype} ({len(g)} events)")
        g = g.sort_values(cfg.time_col).copy()
        hl = half_life_for_type(etype, cfg)
        if hl is None:
            continue

        if etype in cfg.value_key_by_type:
            key = cfg.value_key_by_type[etype]
            x = g[cfg.value_col].apply(lambda v: _extract_from_values(v, key)).astype(float)
        else:
            x = pd.to_numeric(g[cfg.value_col], errors="coerce").astype(float)

        # No seasonal demeaning
        x_demean = x

        x_shift = x_demean.shift(1)
        mu_ewm = x_shift.ewm(halflife=hl, adjust=False).mean()
        sd_ewm = x_shift.ewm(halflife=hl, adjust=False).std(bias=False)

        z = (x_demean - mu_ewm) / sd_ewm.replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)

        df.loc[g.index, "dose_ewm"] = z.values

    return df


# =====================================================
# FOMC helpers: futures expectations & actual rate
# =====================================================

def compute_fomc_expectations_from_csv(fomc_csv_path: str) -> pd.DataFrame:
    """
    Load FOMC meeting dates from CSV and compute expected rates using ZQ=F futures.
    The CSV must contain a column named 'date' (YYYY-MM-DD).
    Returns: DataFrame with columns: fomc_day, expected_rate
    """
    logger.info(f"Loading FOMC statement dates from: {fomc_csv_path}")
    fomc = pd.read_csv(fomc_csv_path)
    fomc["fomc_day"] = pd.to_datetime(fomc["date"]).dt.date
    fomc["expected_rate"] = np.nan
    tickers = ["ZQ=F", "FF=F"]
    for idx, d in fomc["fomc_day"].items():
        start = d - timedelta(days=6)
        end = d - timedelta(days=1)
        got = False
        for tk in tickers:
            logger.info(f"  Fetching {tk}: {start} -> {end} (meeting {d})")
            try:
                zq = yf.download(
                    tk,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1d",
                    progress=False,
                    auto_adjust=False,
                )
            except Exception as e:
                logger.warning(f"    Download error {tk} {d}: {e}")
                zq = None
            if zq is not None and len(zq) > 0 and "Close" in zq.columns:
                try:
                    # Use .item() for single-element Series, fallback to .iloc[-1] for multi-row
                    close_series = zq["Close"]
                    if len(close_series) == 1:
                        last_close = float(close_series.item())
                    else:
                        last_close = float(close_series.iloc[-1])
                    fomc.loc[idx, "expected_rate"] = 100.0 - last_close
                    got = True
                    break
                except Exception as e:
                    logger.warning(f"    Parse close error {tk} {d}: {e}")
        if not got:
            logger.warning(f"    No futures data for {d}; expected_rate remains NaN")
    n_non_nan = int(fomc['expected_rate'].notna().sum())
    logger.info(f"Computed expected rates for {n_non_nan} FOMC meetings out of {len(fomc)}")
    return fomc[["fomc_day", "expected_rate"]]



def fomc_actual_rate_from_values(val) -> float:
    """
    Extract midpoint fed funds target from 'values' cell.
    Accepts dict or JSON string.
    """
    if val is None:
        return np.nan

    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            return np.nan

    if not isinstance(val, dict):
        return np.nan

    up = val.get("fed_funds_target_upper")
    lo = val.get("fed_funds_target_lower")

    if up is None or lo is None:
        return np.nan

    try:
        return (float(up) + float(lo)) / 2.0
    except Exception:
        return np.nan



# =====================================================
# FOMC policy change (original Δ bps logic; kept but no longer used as main dose)
# =====================================================

def apply_fomc_policy_dose(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Legacy: FOMC dose = Δ(midpoint) in bps, computed only at *block starts* where the target range changes.
    Kept for reference; main dose now uses futures-based surprise (dose_fomc_surprise).
    """
    logger.info("Applying legacy FOMC policy dose (Δ midpoint in bps)...")
    df = df.copy()
    is_fomc = (df[cfg.event_type_col] == "FOMC Decision")
    if not is_fomc.any():
        logger.warning("  No 'FOMC Decision' events found")
        return df
    logger.info(f"  Found {is_fomc.sum()} FOMC event rows")

    g = df.loc[is_fomc].sort_values(cfg.time_col).copy()

    upper = g[cfg.value_col].apply(lambda v: _extract_from_values(v, cfg.fomc_upper)).astype(float)
    lower = g[cfg.value_col].apply(lambda v: _extract_from_values(v, cfg.fomc_lower)).astype(float)

    mid = (upper + lower) / 2.0

    # First row when midpoint changes
    block_start = mid.ne(mid.shift(1))

    # Δ midpoint vs previous block (percentage points), then to bps
    delta_mid = mid - mid.shift(1)
    dose_bps_block = (delta_mid * 100.0).where(block_start)

    # First block (no previous) -> set 0 at that first row
    first_idx = dose_bps_block.index[0]
    if pd.isna(dose_bps_block.loc[first_idx]):
        dose_bps_block.loc[first_idx] = 0.0

    # Write only at block starts; non-start rows remain NaN
    df.loc[dose_bps_block.index, "dose_fomc_bps"] = dose_bps_block.values

    # Optional: standardized Δ (not the headline dose)
    if cfg.standardize_fomc_delta:
        x_shift = delta_mid.shift(1)
        hl = cfg.half_life_monthly or 12
        mu_ewm = x_shift.ewm(halflife=hl, adjust=False).mean()
        sd_ewm = x_shift.ewm(halflife=hl, adjust=False).std(bias=False)
        z = (delta_mid - mu_ewm) / sd_ewm.replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan)
        df.loc[g.index, "dose_fomc_ewm"] = z.values

    return df


# =====================================================
# Rolling/expanding time-series CV (sanity metric)
# =====================================================

def ts_cv_scores(X: pd.DataFrame, y: np.ndarray, build: callable, k: int) -> Tuple[float, float]:
    """Rolling-origin CV: split chronologically into k folds.
    Returns (avg_RMSE, avg_R2) across folds. Uses fresh model each fold.
    """
    n = len(X)
    if n < 60:
        return (float("nan"), float("nan"))
    k = max(3, min(k, 10, n // 40))  # keep folds sensible
    idx = np.arange(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    boundaries = np.cumsum(fold_sizes)

    rmses, r2s = [], []
    start = 0
    for b in boundaries:
        # train = [0:start_train_end), test = [start_train_end:b)
        train_end = max(start + (b - start) // 2, 20)  # ensure some train points
        if train_end <= start or b <= train_end:
            start = b
            continue
        tr_idx = idx[:train_end]
        te_idx = idx[train_end:b]
        Xtr, ytr = X.iloc[tr_idx], y[tr_idx]
        Xte, yte = X.iloc[te_idx], y[te_idx]
        mdl = build()
        mdl.fit(Xtr, ytr)
        yhat = mdl.predict(Xte)
        rmses.append(math.sqrt(mean_squared_error(yte, yhat)))
        r2s.append(r2_score(yte, yhat))
        start = b

    if not rmses:
        return (float("nan"), float("nan"))
    return (float(np.mean(rmses)), float(np.mean(r2s)))


# =====================================================
# DRF via g-computation (per event type)
# =====================================================

def outcome_model_from_cfg(cfg: Config) -> Pipeline:
    return build_model(cfg.outcome_model)


def grid_from_series(s: pd.Series, n: int, cfg: Config) -> np.ndarray:
    s = s.dropna()
    if len(s) == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = (np.percentile(s, [2, 98]) if len(s) > 20 else (s.min(), s.max()))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = -1.0, 1.0
    if cfg.clip_dose:
        lo = max(lo, cfg.clip_low)
        hi = min(hi, cfg.clip_high)
        if lo >= hi:
            lo, hi = cfg.clip_low, cfg.clip_high
    return np.linspace(lo, hi, n)


def _slope_at_zero(dose_grid: np.ndarray, drf: np.ndarray) -> float:
    idx0 = np.argmin(np.abs(dose_grid - 0.0))
    if 1 <= idx0 < len(dose_grid)-1:
        return float((drf[idx0+1]-drf[idx0-1])/(dose_grid[idx0+1]-dose_grid[idx0-1]))
    return np.nan


def _bootstrap_drf_and_slope(
    g: pd.DataFrame,
    Z_cols: List[str],
    dose_col: str,
    cfg: Config,
    base_model: Pipeline,
    dose_grid: np.ndarray,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(g)
    slopes = np.empty(cfg.n_boot, dtype=float)
    drf_boot = np.zeros((cfg.n_boot, len(dose_grid)), dtype=float)

    for b in range(cfg.n_boot):
        idx = rng.randint(0, n, size=n)
        gb = g.iloc[idx]
        Xb = gb[[dose_col] + Z_cols].copy()
        yb = gb[cfg.outcome_col].values
        mdl = Pipeline(base_model.steps)
        for c in Xb.columns:
            if pd.api.types.is_object_dtype(Xb[c]):
                Xb[c] = pd.to_numeric(Xb[c], errors="coerce")
        mdl.fit(Xb, yb)

        Zmat_b = gb[Z_cols].copy()
        preds = []
        for d in dose_grid:
            row = pd.concat([pd.Series(d, index=Zmat_b.index, name=dose_col), Zmat_b], axis=1)
            preds.append(mdl.predict(row))
        preds = np.vstack(preds)
        drf_b = preds.mean(axis=1)
        drf_boot[b, :] = drf_b
        slopes[b] = _slope_at_zero(dose_grid, drf_b)

    drf_mean = drf_boot.mean(axis=0)
    return drf_mean, slopes, drf_boot


def drf_per_event_type(df: pd.DataFrame, cfg: Config) -> Dict[str, dict]:
    """Compute DRF per event type; saves plots/json per type. Returns summary dict.
    Uses dose_ewm for non-FOMC types; uses dose_fomc_bps for FOMC.
    Adds: time-series CV metrics and residual sanity plots for headline families.
    """
    logger.info("="*60)
    logger.info("Computing DRF per event type...")
    ensure_dir(cfg.out_dir)


    all_opts = cfg.market_state + cfg.liquidity + cfg.positioning
    calendar_cols = ["cal_year","cal_month","cal_dayofweek","cal_hour","cal_quarter","cal_week"] if cfg.use_calendar else []
    summary = {}


    for etype, g in df.groupby(cfg.event_type_col, sort=False):
        logger.info(f"\n--- Processing event type: {etype} ---")
        logger.info(f"  Initial group size for {etype}: {len(g)} rows")
        g = g.sort_values(cfg.time_col)

        dose_col = "dose_fomc_surprise" if etype == "FOMC Decision" else "dose_ewm"
        if dose_col not in g.columns:
            logger.warning(f"  Skipping {etype}: dose column '{dose_col}' missing")
            continue

        logger.info(f"  Events before any filtering: {len(g)}")

                # Add explicit NaN diagnostics for FOMC Decision dose/outcome columns
        if etype == "FOMC Decision":
            dose_nan_count = g[dose_col].isna().sum()
            outcome_nan_count = g[cfg.outcome_col].isna().sum()
            both_nan_count = g[[dose_col, cfg.outcome_col]].isna().all(axis=1).sum()
            logger.info(f"  FOMC Decision NaN counts: {dose_col}={dose_nan_count}, outcome={outcome_nan_count}, both NaN={both_nan_count}")
        # Step 1: dropna on dose and outcome
        g1 = g.dropna(subset=[dose_col, cfg.outcome_col])
        logger.info(f"  Events after dropna on dose/outcome: {len(g1)}")

        # Step 2: dropna on covariates
        if etype == "FOMC Decision":
            fomc_covariates = [c for c in ["TS_30_60", "PCR_OI", "NET_OI"] if c in g1.columns]
            Z_cols = fomc_covariates
            logger.info(f"  Using FOMC covariates (no calendar features): {Z_cols}")
            nan_counts_fomc = g1[Z_cols].isna().sum()
            logger.info(f"  FOMC covariate NaN counts before dropna: {nan_counts_fomc.to_dict()}")
        else:
            Z_cols = [c for c in (all_opts + calendar_cols) if c in g1.columns]
            logger.info(f"  Using {len(Z_cols)} covariates for confounding adjustment")

        g2 = g1.dropna(subset=Z_cols)
        logger.info(f"  Events after dropna on selected covariates: {len(g2)}")
        if len(g2) < cfg.min_events_for_drf:
            logger.warning(f"  Skipping {etype}: insufficient events (< {cfg.min_events_for_drf})")
            continue

        # Use g2 for modeling
        g = g2

        Xcols = [dose_col] + Z_cols
        X = g[Xcols].copy()
        y = g[cfg.outcome_col].values
        for c in X.columns:
            if pd.api.types.is_object_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")

        # Fit once on full sample
        base_model = outcome_model_from_cfg(cfg)
        logger.info(f"  Fitting outcome model on {len(g)} events...")
        base_model.fit(X, y)

        # Rolling-origin CV metrics
        logger.info("  Computing rolling-origin CV metrics (RMSE, R²)...")
        def _builder():
            return outcome_model_from_cfg(cfg)
        rmse_cv, r2_cv = ts_cv_scores(X, y, _builder, cfg.ts_cv_folds)

        # DRF grid and g-computation
        dose_grid = grid_from_series(g[dose_col], cfg.dose_grid_points, cfg)
        Zmat = g[Z_cols].copy()
        preds = []
        for d in dose_grid:
            row = pd.concat([pd.Series(d, index=Zmat.index, name=dose_col), Zmat], axis=1)
            preds.append(base_model.predict(row))
        preds = np.vstack(preds)
        drf = preds.mean(axis=1)
        slope0 = _slope_at_zero(dose_grid, drf)

        # Residual sanity plot for headline families
        if etype in cfg.residual_plot_types:
            fitted = base_model.predict(X)
            resid = y - fitted
            safe_name = etype.replace(" ", "_").replace("/", "-")
            res_path = os.path.join(cfg.out_dir, f"Residuals_{safe_name}.png")
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(fitted, resid, s=10, alpha=0.6)
            plt.axhline(0, linestyle='--', linewidth=1)
            plt.title(f"Residual vs Fitted: {etype} (n={len(g)})")
            plt.xlabel("Fitted")
            plt.ylabel("Residual")
            plt.tight_layout()
            plt.savefig(res_path, dpi=130)
            plt.close()
            logger.info(f"  Saved residual sanity plot: {res_path}")

        # Bootstrap uncertainty
        drf_lo = drf_hi = None
        slope_ci = (None, None)
        p_value = None
        if cfg.bootstrap_enabled and cfg.n_boot > 0:
            logger.info(f"  Running {cfg.n_boot} bootstrap iterations...")
            rng = np.random.RandomState(cfg.random_state)
            boot_base = outcome_model_from_cfg(cfg)
            drf_mean_b, slope_samples, drf_boot = _bootstrap_drf_and_slope(
                g, Z_cols, dose_col, cfg, boot_base, dose_grid, rng
            )
            alpha = 1.0 - cfg.ci_level
            lo_q, hi_q = 100*alpha/2.0, 100*(1.0-alpha/2.0)
            drf_lo = np.percentile(drf_boot, lo_q, axis=0)
            drf_hi = np.percentile(drf_boot, hi_q, axis=0)
            slope_ci = (float(np.percentile(slope_samples, lo_q)),
                        float(np.percentile(slope_samples, hi_q)))
            if np.isfinite(slope0):
                p_value = float(2 * min((np.mean(slope_samples >= slope0)),
                                        (np.mean(slope_samples <= slope0))))

        # Save per-type DRF figure
        safe_name = etype.replace(" ", "_").replace("/", "-")
        title = f"DRF ({'EWM z' if etype!='FOMC Decision' else 'Δ bps'}): {etype}"
        fig_path = os.path.join(cfg.out_dir, f"DRF_{safe_name}.png")
        plt.figure(figsize=(7.5, 5))
        if drf_lo is not None and drf_hi is not None:
            plt.fill_between(dose_grid, drf_lo, drf_hi, alpha=0.2, label=f"{etype} {int(cfg.ci_level*100)}% CI")
        plt.plot(dose_grid, drf, label=f"{etype} (n={len(g)})")
        plt.axvline(0, linestyle='--', linewidth=1)
        plt.title(title)
        plt.xlabel("Dose" + (" (EWM z)" if etype != "FOMC Decision" else " (bps)"))
        plt.ylabel("E[Y | do(dose)] (g-computed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info(f"  Saved DRF plot: {fig_path}")

        # Save JSON (includes CV metrics)
        out_json = {
            "event_type": etype,
            "dose_col": dose_col,
            "n": int(len(g)),
            "dose_grid": [float(v) for v in dose_grid],
            "drf": [float(v) for v in drf],
            "slope_at_zero": None if (isinstance(slope0, float) and (not np.isfinite(slope0))) else float(slope0),
            "drf_lo": None if drf_lo is None else [float(v) for v in drf_lo],
            "drf_hi": None if drf_hi is None else [float(v) for v in drf_hi],
            "slope_ci": None if slope_ci == (None, None) else [slope_ci[0], slope_ci[1]],
            "p_value": p_value,
            "cv_rmse": None if not np.isfinite(rmse_cv) else float(rmse_cv),
            "cv_r2": None if not np.isfinite(r2_cv) else float(r2_cv),
        }
        json_path = os.path.join(cfg.out_dir, f"DRF_{safe_name}.json")
        with open(json_path, "w") as f:
            json.dump(out_json, f, indent=2)
        logger.info(f"  Saved DRF results JSON: {json_path}")

        summary[etype] = {
            "dose_col": dose_col,
            "n": int(len(g)),
            "slope_at_zero": None if (isinstance(slope0, float) and (not np.isfinite(slope0))) else float(slope0),
            "slope_ci_low": None if slope_ci == (None, None) else slope_ci[0],
            "slope_ci_high": None if slope_ci == (None, None) else slope_ci[1],
            "p_value": p_value,
            "cv_rmse": None if not np.isfinite(rmse_cv) else float(rmse_cv),
            "cv_r2": None if not np.isfinite(r2_cv) else float(r2_cv),
        }

    # Save a small summary index + CSV table
    with open(os.path.join(cfg.out_dir, "DRF_summary_index.json"), "w") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for et, v in summary.items():
        rows.append({"event_type": et, **v})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(cfg.out_dir, "results_table.csv"), index=False)

    return summary


# =====================================================
# Diagnostics (minimal)
# =====================================================

def lead_test(df: pd.DataFrame, cfg: Config) -> Dict[str, Optional[float]]:
    logger.info("\nRunning lead test (dose vs pre-window returns)...")
    out: Dict[str, Optional[float]] = {}
    if not cfg.pre_return_col or (cfg.pre_return_col not in df.columns):
        logger.info("  Skipping: no pre_return_col configured")
        return out

    for etype, g in df.groupby(cfg.event_type_col, sort=False):
        dose_col = "dose_fomc_surprise" if etype == "FOMC Decision" else "dose_ewm"
        if dose_col not in g.columns:
            out[etype] = None
            continue
        sub = g[[dose_col, cfg.pre_return_col]].dropna()
        if len(sub) < 50:
            out[etype] = None
            continue
        corr = float(sub.corr().iloc[0, 1])
        out[etype] = corr
    return out


def overlap_stats(df: pd.DataFrame, cfg: Config) -> Dict[str, Dict[str, Tuple[float, float]]]:
    logger.info("\nComputing dose overlap statistics...")
    res: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for etype, g in df.groupby(cfg.event_type_col, sort=False):
        dose_col = "dose_fomc_surprise" if etype == "FOMC Decision" else "dose_ewm"
        if dose_col not in g.columns:
            continue
        s = g[dose_col].dropna()
        if len(s) < 20:
            continue
        lo, hi = np.percentile(s, [1, 99])
        res[etype] = {"p1_p99": (float(lo), float(hi)), "n": int(len(s))}
    return res

    covs = ["TS_30_60", "PCR_OI", "NET_OI"]

    print("NaN counts:")
    print(df_fomc[covs].isna().sum())

    print("\nTotal rows:", len(df_fomc))
    print("\nNon-NaN counts:")
    print(df_fomc[covs].notna().sum())

# =====================================================
# Runner
# =====================================================

def main():
   
   
    logger.info("="*60)
    logger.info("Starting SCM Pipeline")
    logger.info("="*60)

    cfg = Config()
    cfg.out_dir = os.path.abspath(cfg.out_dir)
    ensure_dir(cfg.out_dir)

    # Load
    logger.info(f"Loading data from: {cfg.parquet_path}")
    df = pd.read_parquet(cfg.parquet_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    df = df.sort_values(cfg.time_col).reset_index(drop=True)

    # =====================================================================
    # === INSERT FOMC COVARIATE NAN CHECK RIGHT HERE ======================
    # =====================================================================
    df_fomc = df[df["event_type"] == "FOMC Decision"].copy()

    covs = ["TS_30_60", "PCR_OI", "NET_OI"]
    covs = [c for c in covs if c in df_fomc.columns]

    print("\n==== FOMC COVARIATE NAN CHECK ====")
    print("NaN counts:")
    print(df_fomc[covs].isna().sum())

    print("\nTotal FOMC rows:", len(df_fomc))

    print("\nNon-NaN counts:")
    print(df_fomc[covs].notna().sum())

    print("\nRows with ANY NaN in the 3 covariates:",
        df_fomc[covs].isna().any(axis=1).sum())
    # =====================================================================



    # Subset types if requested
    if cfg.event_types_subset:
        logger.info(f"Filtering to event types: {cfg.event_types_subset}")
        df = df[df[cfg.event_type_col].isin(cfg.event_types_subset)].copy()
        logger.info(f"After filtering: {len(df)} rows")

    # NEW: FOMC futures-based surprises
    fomc_csv_path = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer2\fomc_statement_dates.csv"
    logger.info("Computing FOMC expected rates from ZQ=F using CSV of meeting dates...")
    fomc_expect = compute_fomc_expectations_from_csv(fomc_csv_path)

    df["dose_fomc_surprise"] = np.nan
    df["dose_fomc_bps"] = np.nan  # ensure exists

    is_fomc = df[cfg.event_type_col] == "FOMC Decision"

    # --- Diagnostics: NaN counts for each event type ---
    for etype, g in df.groupby(cfg.event_type_col, sort=False):
        logger.info(f"\n--- NaN diagnostics for event type: {etype} ---")
        if etype == "FOMC Decision":
            reduced_covariates = [c for c in ["TS_30_60", "PCR_OI", "NET_OI"] if c in g.columns]
            nan_counts = g[reduced_covariates].isna().sum().sort_values(ascending=False)
            logger.info(f"FOMC reduced covariate NaN counts: {nan_counts.to_dict()}")
            # Drop unused covariate columns for FOMC events
            drop_cols = [c for c in g.columns if c not in reduced_covariates + [cfg.time_col, cfg.event_type_col, cfg.value_col, cfg.outcome_col, "dose_fomc_bps", "dose_fomc_surprise"]]
            df.loc[g.index, drop_cols] = np.nan
        else:
            all_covariates = cfg.market_state + cfg.liquidity + cfg.positioning
            calendar_cols = ["cal_year","cal_month","cal_dayofweek","cal_hour","cal_quarter","cal_week"] if cfg.use_calendar else []
            used_covariates = [c for c in (all_covariates + calendar_cols) if c in g.columns]
            nan_counts = g[used_covariates].isna().sum().sort_values(ascending=False)
            logger.info(f"{etype} covariate NaN counts: {nan_counts.to_dict()}")

    # --- FOMC expected rate logic ---

    if is_fomc.any():
        # Ensure fomc_day column exists in df for FOMC Decision events
        df.loc[is_fomc, "fomc_day"] = pd.to_datetime(df.loc[is_fomc, cfg.time_col]).dt.date
        fomc_rows = df.loc[is_fomc].copy()
        fomc_rows["actual_rate"] = fomc_rows[cfg.value_col].apply(fomc_actual_rate_from_values)
        merged = fomc_rows.merge(fomc_expect, on="fomc_day", how="left", sort=False)
        # Diagnostics: show index alignment
        logger.info(f"FOMC merge: fomc_rows shape={fomc_rows.shape}, merged shape={merged.shape}")
        logger.info(f"FOMC merge: fomc_rows index head={fomc_rows.index[:5].tolist()}, merged index head={merged.index[:5].tolist()}")
        # Fallback for expected_rate
        if merged["expected_rate"].isna().any():
            logger.info(f"FOMC expected_rate NaNs pre-fallback: {int(merged['expected_rate'].isna().sum())}")
            merged = merged.sort_values("fomc_day")
            merged["expected_rate"] = merged["expected_rate"].ffill().bfill()
            logger.info(f"FOMC expected_rate NaNs post-fallback: {int(merged['expected_rate'].isna().sum())}")
        # Assign dose columns
        merged["dose_fomc_bps"] = (merged["actual_rate"] - merged["expected_rate"]) * 100.0
        merged["dose_fomc_surprise"] = merged["actual_rate"] - merged["expected_rate"]
        # Patch: assign by fomc_day, not index, to ensure all FOMC events get values
        for col in ["dose_fomc_bps", "dose_fomc_surprise", "actual_rate", "expected_rate"]:
            assign_map = dict(zip(merged["fomc_day"], merged[col]))
            df.loc[df["event_type"] == "FOMC Decision", col] = df.loc[df["event_type"] == "FOMC Decision", "fomc_day"].map(assign_map)
        # Diagnostics: show which rows have NaN actual_rate or expected_rate
        logger.info("FOMC events with NaN actual_rate:")
        logger.info(merged[merged["actual_rate"].isna()][["fomc_day", "actual_rate", "expected_rate"]].to_string())
        logger.info("FOMC events with NaN expected_rate:")
        logger.info(merged[merged["expected_rate"].isna()][["fomc_day", "actual_rate", "expected_rate"]].to_string())
        # Diagnostics: show which FOMC events in df have NaN dose_fomc_surprise after assignment
        nan_dose = df.loc[is_fomc & df["dose_fomc_surprise"].isna(), [cfg.time_col, "fomc_day", "actual_rate", "expected_rate"]]
        if not nan_dose.empty:
            logger.warning(f"FOMC events with NaN dose_fomc_surprise after assignment: {len(nan_dose)}")
            logger.warning(nan_dose.to_string())
        else:
            logger.info("All FOMC events have dose_fomc_surprise assigned.")
        usable = (df.loc[is_fomc, 'dose_fomc_bps'].notna() & df.loc[is_fomc, 'actual_rate'].notna() & df.loc[is_fomc, 'expected_rate'].notna()).sum()
        logger.info(
            f"Computed FOMC surprises: dose_nonNaN={df.loc[is_fomc, 'dose_fomc_bps'].notna().sum()} actual_nonNaN={df.loc[is_fomc, 'actual_rate'].notna().sum()} expected_nonNaN={df.loc[is_fomc, 'expected_rate'].notna().sum()} usable={usable}"
        )
        # Diagnostic: log NaN reasons for FOMC
        merged_full = df.loc[is_fomc].copy()
        n_total = len(merged_full)
        n_dose_nan = merged_full['dose_fomc_bps'].isna().sum()
        n_outcome_nan = merged_full['post_effect_log'].isna().sum()
        n_both_nan = ((merged_full['dose_fomc_bps'].isna()) & (merged_full['post_effect_log'].isna())).sum()
        n_both_not_nan = ((merged_full['dose_fomc_bps'].notna()) & (merged_full['post_effect_log'].notna())).sum()
        logger.info(f"FOMC diagnostic: total={n_total} dose_fomc_bps NaN={n_dose_nan} post_effect_log NaN={n_outcome_nan} both NaN={n_both_nan} both not NaN={n_both_not_nan}")
        logger.info("Sample FOMC rows with missing post_effect_log:")
        logger.info(merged_full[merged_full['post_effect_log'].isna()].head(5).to_string())
    else:
        logger.info("No FOMC Decision rows found.")

    # Calendar
    if cfg.use_calendar:
        logger.info("Adding calendar features...")
        df = add_calendar(df, cfg.time_col)

    # Compute EWM surprise for non-FOMC types
    df = compute_ewm_surprise_per_type(df, cfg)

    # Build DRFs per type (with CV metrics + residual plots)
    summary = drf_per_event_type(df, cfg)

    # Diagnostics
    lead = lead_test(df, cfg)
    lead_path = os.path.join(cfg.out_dir, "lead_test_by_type.json")
    logger.info(f"Saving lead test results: {lead_path}")
    with open(lead_path, "w") as f:
        json.dump(lead, f, indent=2)

    overlap = overlap_stats(df, cfg)
    overlap_path = os.path.join(cfg.out_dir, "overlap_by_type.json")
    logger.info(f"Saving overlap statistics: {overlap_path}")
    with open(overlap_path, "w") as f:
        json.dump(overlap, f, indent=2)

    logger.info("="*60)
    logger.info(f"✅ Pipeline complete! Outputs saved to: {cfg.out_dir}")
    logger.info("="*60)
    logger.info("DRF Summary:")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
