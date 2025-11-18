from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import argparse
from pathlib import Path

# show all columns and widen output for the terminal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)            # increase if the  terminal is wider
pd.set_option("display.max_colwidth", 200)

@dataclass
class MatchResult:
    matched_df: pd.DataFrame
    att_matched_only: float
    n_treated: int
    n_matched: int
    ate_ipw: float
    details: Dict[str, Any]


class Matching:
    """
    Implements propensity-score matching and inverse probability weighting (IPW) for causal inference.

    This class matches treated units to control units using 1:k nearest neighbor matching on the logit (propensity score) scale,
    with an optional caliper to restrict matches to similar controls. It handles missing values via mean-imputation and missingness indicators,
    fits a logistic regression to estimate propensity scores, and computes treatment effect estimates and balance diagnostics.

    Methods
    -------
    fit_match(df_treated, df_control, save_matched_pairs=None) -> MatchResult
        Performs matching and returns matched pairs, ATT for matched treated units, IPW ATE, and diagnostics.

    Parameters
    ----------
    covariates : List[str]
        List of covariate column names to use for matching.
    outcome_col : str
        Name of the outcome column.
    treat_col : str
        Name of the treatment indicator column.
    treat_time_col : Optional[str]
        Optional column name for treatment time.
    control_time_col : Optional[str]
        Optional column name for control time.
    k : int
        Number of nearest neighbors to match.
    caliper_multiplier : float
        Multiplier for caliper width (in SD of logit).
    replacement : bool
        Whether to allow controls to be matched multiple times.
    random_state : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        covariates: List[str],
        outcome_col: str = "post_effect_log",
        treat_col: str = "T",
        treat_time_col: Optional[str] = None,
        control_time_col: Optional[str] = None,
        k: int = 2,
        caliper_multiplier: float = 0.3,
        replacement: bool = True,
        random_state: Optional[int] = 0,
    ) -> None:
        self.covariates = covariates
        self.outcome_col = outcome_col
        self.treat_col = treat_col
        self.treat_time_col = treat_time_col
        self.control_time_col = control_time_col
        self.k = k
        self.caliper_multiplier = caliper_multiplier
        self.replacement = replacement
        self.random_state = random_state
        self.model: Optional[LogisticRegression] = None

    def _validate_columns(self, df: pd.DataFrame, required: List[str], df_name: str) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {df_name}: {missing}")

    def fit_match(
        self,
        df_treated: pd.DataFrame,
        df_control: pd.DataFrame,
        save_matched_pairs: Optional[str] = None,
    ) -> MatchResult:
        """Perform matching between treated and control groups.

        This method validates input DataFrames, estimates propensity scores using logistic regression,
        performs 1:k nearest neighbor matching on the logit scale with an optional caliper,
        computes treatment effect estimates (ATT, ATE), and evaluates covariate balance before and after matching.

        Parameters
        ----------
        df_treated : pd.DataFrame
            DataFrame containing the treated group data.
        df_control : pd.DataFrame
            DataFrame containing the control group data.
        save_matched_pairs : Optional[str], optional
            If provided, the matched pairs will be saved to this file path in parquet format.

        Returns
        -------
        MatchResult
            A dataclass containing the matched DataFrame, ATT for matched treated units,
            ATE estimate using IPW, and a dictionary of diagnostic details.

        Raises
        ------
        ValueError
            If required columns are missing in the input DataFrames.
        """

        # Validate input DataFrames and required columns
        required = list(self.covariates) + [self.outcome_col]
        self._validate_columns(df_treated, required, "treated")
        self._validate_columns(df_control, required, "control")

        # Add treatment indicator column to both treated and control DataFrames
        df_t = df_treated.copy()
        df_c = df_control.copy()
        df_t[self.treat_col] = 1
        df_c[self.treat_col] = 0

        # Combine treated and control samples for joint processing
        combined = pd.concat([df_t, df_c], ignore_index=True)

        # --- Handle missing values (mean-impute) and add missingness indicators ---
        # Drop rows with missing outcome (can't use these for ATT/IPW)
        if combined[self.outcome_col].isna().any():
            n_before = len(combined)
            combined = combined.dropna(subset=[self.outcome_col]).reset_index(drop=True)
            print(f"Warning: dropped {n_before - len(combined)} rows with missing outcome '{self.outcome_col}'")

        # Create missingness indicator columns (one per covariate)
        indicator_cols = []
        for c in self.covariates:
            mcol = f"{c}_missing"
            combined[mcol] = combined[c].isna().astype(int)
            indicator_cols.append(mcol)

        # Mean-impute covariates using combined means
        cov_means = combined[self.covariates].mean()
        if combined[self.covariates].isna().any().any():
            print("Info: imputing missing covariate values with combined column means")
        combined[self.covariates] = combined[self.covariates].fillna(cov_means)

        # Final covariates used in the model: original covariates + indicators
        model_covariates = list(self.covariates) + indicator_cols

        # Standardize model covariates using combined mean/std
        means = combined[model_covariates].mean()
        stds = combined[model_covariates].std(ddof=0)
        stds_replaced = stds.replace(0, 1.0)
        Z = (combined[model_covariates] - means) / stds_replaced

        # Fit logistic regression
        X = Z.values
        y = combined[self.treat_col].values

        # Try a reasonably regularized logistic regression; handle convergence issues
        try:
            model = LogisticRegression(
                solver="lbfgs",
                random_state=self.random_state,
                max_iter=500,
                C=1.0,
            )
            model.fit(X, y)
        except Exception:
            model = LogisticRegression(
                solver="lbfgs",
                random_state=self.random_state,
                max_iter=1000,
                C=0.1,
                class_weight="balanced",
            )
            model.fit(X, y)

        self.model = model

        # Propensity scores and logits
        p = model.predict_proba(X)[:, 1]
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        eta = np.log(p / (1 - p))

        combined = combined.reset_index(drop=True)
        combined = combined.copy()
        combined["ps"] = p
        combined["eta"] = eta

        # Compute caliper
        eta_sd = float(np.std(eta, ddof=0))
        caliper = float(self.caliper_multiplier * eta_sd)

        # Split back
        treated_mask = combined[self.treat_col] == 1
        treated_df = combined[treated_mask].reset_index()
        control_df = combined[~treated_mask].reset_index()

        # Use NearestNeighbors on 1D eta
        nbrs = NearestNeighbors(n_neighbors=min(self.k, len(control_df)), metric="euclidean")
        control_eta = control_df[["eta"]].values
        if len(control_eta) == 0:
            raise ValueError("No control observations available for matching")
        nbrs.fit(control_eta)

        treated_eta = treated_df[["eta"]].values
        distances, indices = nbrs.kneighbors(treated_eta, return_distance=True)

        # For each treated, filter neighbors by caliper; construct matched weights
        matched_records = []
        control_used_counts = np.zeros(len(control_df), dtype=int)

        for i, (dists_row, inds_row) in enumerate(zip(distances, indices)):
            within = [j for j, dist in zip(inds_row, dists_row) if abs(dist) <= caliper + 1e-12]
            if len(within) == 0:
                # no match within caliper
                matched_records.append({
                    "treated_index": int(treated_df.loc[i, "index"]),
                    "control_indices": [],
                    "control_weights": [],
                })
                continue

            # If replacement=False, pick controls not exhausted yet (greedy)
            chosen = []
            for idx in within:
                if self.replacement:
                    chosen.append(idx)
                else:
                    # only choose if not used yet
                    if control_used_counts[idx] == 0 and len(chosen) < self.k:
                        chosen.append(idx)
            if len(chosen) == 0:
                matched_records.append({
                    "treated_index": int(treated_df.loc[i, "index"]),
                    "control_indices": [],
                    "control_weights": [],
                })
                continue

            # weights uniform among chosen neighbors
            w = [1.0 / len(chosen)] * len(chosen)
            for idx in chosen:
                control_used_counts[idx] += 1

            matched_records.append({
                "treated_index": int(treated_df.loc[i, "index"]),
                "control_indices": [int(control_df.loc[j, "index"]) for j in chosen],
                "control_weights": w,
            })

        # Build matched DataFrame linking treated to controls and compute ATT
        matched_list = []
        sum_att = 0.0
        sum_att_matched = 0.0
        matched_count = 0
        for rec in matched_records:
            ti = rec["treated_index"]
            y_t = float(combined.loc[ti, self.outcome_col])
            if len(rec["control_indices"]) == 0:
                matched_list.append({"treated_index": ti, "control_indices": [], "treated_outcome": y_t, "control_weighted_outcome": None, "n_matched": 0, "delta": float('nan')})
                continue
            control_vals = [float(combined.loc[cidx, self.outcome_col]) for cidx in rec["control_indices"]]
            weights = rec["control_weights"]
            control_weighted_outcome = float(np.dot(control_vals, weights))
            diff = y_t - control_weighted_outcome
            sum_att += diff
            sum_att_matched += diff
            matched_count += 1
            matched_list.append({"treated_index": ti, "control_indices": rec["control_indices"], "treated_outcome": y_t, "control_weighted_outcome": control_weighted_outcome, "n_matched": len(rec["control_indices"]), "delta": diff})

        n_treated = len(treated_df)
        
        att_matched_only = float(sum_att_matched / matched_count) if matched_count > 0 else float('nan')

        matched_df = pd.DataFrame(matched_list)
        # ---- Balance diagnostics: compute SMD before and after matching on original covariates ----
        # We'll compute SMD_before using pre-match treated vs control means and pooled SD.
        # For SMD_after we compute, for each treated unit with matches, the weighted mean of its matched controls
        # on each covariate, then average those control-means across matched treated units and compare to treated mean.
        smd_before = {}
        smd_after = {}
        eps = 1e-8
        for cov in self.covariates:
            # ensure numeric
            try:
                xt = treated_df[cov].astype(float)
                xc = control_df[cov].astype(float)
            except Exception:
                # if conversion fails, coerce via to_numeric
                xt = pd.to_numeric(treated_df[cov], errors="coerce").astype(float)
                xc = pd.to_numeric(control_df[cov], errors="coerce").astype(float)

            mean_t = float(xt.mean())
            mean_c = float(xc.mean())
            var_t = float(xt.var(ddof=0))
            var_c = float(xc.var(ddof=0))
            pooled_sd = float(np.sqrt((var_t + var_c) / 2.0))
            if pooled_sd <= 0:
                pooled_sd = eps
            smd_b = (mean_t - mean_c) / pooled_sd
            smd_before[cov] = smd_b

            # matched-control means: for each matched treated, compute the (uniform) weighted mean of its matched controls
            matched_control_means = []
            for _, row in matched_df.iterrows():
                ctr_idxs = row.get("control_indices") or []
                if not ctr_idxs:
                    continue
                # control indices are stored as combined dataframe indices; use combined to fetch covariate values
                vals = [float(combined.loc[int(ci), cov]) for ci in ctr_idxs]
                # use uniform weights (matching used uniform weights)
                matched_control_means.append(float(np.mean(vals)))

            if len(matched_control_means) == 0:
                smd_a = float('nan')
            else:
                matched_control_mean = float(np.mean(matched_control_means))
                smd_a = (mean_t - matched_control_mean) / pooled_sd
            smd_after[cov] = smd_a

        # aggregate SMD summaries (use absolute values)
        abs_before = np.array([abs(v) for v in smd_before.values()], dtype=float) if smd_before else np.array([], dtype=float)
        Mean_SMD_Before = float(np.nanmean(abs_before)) if abs_before.size > 0 else float('nan')
        abs_after_list = [abs(v) for v in smd_after.values() if not pd.isna(v)]
        abs_after = np.array(abs_after_list, dtype=float) if len(abs_after_list) > 0 else np.array([], dtype=float)
        Mean_SMD_After = float(np.nanmean(abs_after)) if abs_after.size > 0 else float('nan')
        Max_SMD_After = float(np.nanmax(abs_after)) if abs_after.size > 0 else float('nan')

        # Compute IPW ATE using the formula from attachments: ATE = 1/N sum_i [ T_i*Y_i/p_i - (1-T_i)*Y_i/(1-p_i) ]
        N = len(combined)
        p_series = combined["ps"].values
        y_series = combined[self.outcome_col].values.astype(float)
        T_series = combined[self.treat_col].values
        eps = 1e-6
        p_series = np.clip(p_series, eps, 1 - eps)
        ate_ipw = float(np.mean(T_series * y_series / p_series - (1 - T_series) * y_series / (1 - p_series)))

        details = {
            "n_treated": n_treated,
            "n_controls": len(control_df),
            "n_matched_treated": matched_count,
            "caliper": caliper,
            "eta_sd": eta_sd,
            "Mean_SMD_Before": Mean_SMD_Before,
            "Mean_SMD_After": Mean_SMD_After,
            "Max_SMD_After": Max_SMD_After,
        }

        if save_matched_pairs:
            # Attach more readable fields and save
            out = matched_df.copy()
            out.to_parquet(save_matched_pairs, index=False)
            # Also save per-covariate SMDs so downstream diagnostics can plot covariate-level balance
            try:
                smd_records = []
                for cov in smd_before.keys():
                    smd_records.append({
                        "covariate": cov,
                        "SMD_Before": float(smd_before.get(cov, float('nan'))),
                        "SMD_After": float(smd_after.get(cov, float('nan'))),
                    })
                cov_df = pd.DataFrame(smd_records)
                # place per-family cov_smds under matched_pairs/cov_smds/
                save_path = Path(save_matched_pairs)
                cov_dir = save_path.parent / "cov_smds"
                cov_dir.mkdir(parents=True, exist_ok=True)
                cov_out = cov_dir / (save_path.stem + "_cov_smds.parquet")
                cov_df.to_parquet(cov_out, index=False)
            except Exception as e:
                print(f"Warning: failed to save covariate SMDs for {save_matched_pairs}: {e}")

        return MatchResult(
            matched_df=matched_df,
            att_matched_only=att_matched_only,
            n_treated=n_treated,
            n_matched=matched_count,
            ate_ipw=ate_ipw,
            details=details,
        )


def _default_covariates() -> List[str]:
    return [
        "ATM_IV_D10", "ATM_IV_D30", "ATM_IV_D60", "ATM_IV_D>60",
        "RR25_D10", "RR25_D30", "RR25_D60", "RR25_D>60",
        "PCR_DOLLAR_OTM_D10", "PCR_DOLLAR_OTM_D30", "PCR_DOLLAR_OTM_D60", "PCR_DOLLAR_OTM_D>60",
        "PCR_VEGA_OTM_D10", "PCR_VEGA_OTM_D30", "PCR_VEGA_OTM_D60", "PCR_VEGA_OTM_D>60",
        "TURNOVER_D10", "TURNOVER_D30", "TURNOVER_D60", "TURNOVER_D>60",
        "REL_SPREAD_D10", "REL_SPREAD_D30", "REL_SPREAD_D60", "REL_SPREAD_D>60",
        "OI_SUM_D10", "OI_SUM_D30", "OI_SUM_D60", "OI_SUM_D>60",
        "TS_30_60", "DOLLAR_VOL_D10", "DOLLAR_VOL_D30", "DOLLAR_VOL_D60", "DOLLAR_VOL_D>60",
        "PCR_OI", "NET_OI",
    ]


def main():
    # Default paths (hardcoded) - # NOTE: Adjust the file path below to match your local setup.
    DEFAULT_TREATED = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\Treatment_dataset_Fedmacr-_event\events_aligned_L3_E1.parquet"
    DEFAULT_CONTROLS = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer3\Control_dataset_days_without_events\control_sample_QQQ_aligned.parquet"
    DEFAULT_OUT = r"C:\Users\larbi\Desktop\My Doc\AICAUSAL\Layer4\matched_pairs.parquet"

    parser = argparse.ArgumentParser(description="Run propensity-score matching (Layer4.matching)")
    parser.add_argument("--treated", required=False, default=DEFAULT_TREATED, help=f"Path to treated parquet (events) [default: {DEFAULT_TREATED}]")
    parser.add_argument("--controls", required=False, default=DEFAULT_CONTROLS, help=f"Path to control parquet [default: {DEFAULT_CONTROLS}]")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--caliper", type=float, default=0.2)
    parser.add_argument("--no-replacement", dest="replacement", action="store_false")
    # removed --out: matched pairs will be saved automatically under Layer4/matched_pairs/
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and exit without reading files")
    args = parser.parse_args()

    treated_path = Path(args.treated)
    control_path = Path(args.controls)
    # output directory for matched pairs (auto)
    layer4_dir = Path(__file__).resolve().parent
    matched_dir = layer4_dir / "matched_pairs"
    matched_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run - using the following paths:")
        print(f"  treated: {treated_path}")
        print(f"  controls: {control_path}")
        print(f"  matched_pairs_dir: {matched_dir}")
        return

    df_t = pd.read_parquet(treated_path)
    df_c = pd.read_parquet(control_path)

    covs = _default_covariates()

    matcher = Matching(covariates=covs)

    # principal behavior: run per-event-family. Require an 'event_type' column in treated data.
    GROUP_COL = "event_type"
    if GROUP_COL not in df_t.columns:
        print(f"Error: required column '{GROUP_COL}' not found in treated data. Please include an '{GROUP_COL}' column to run per-family matching.")
        raise SystemExit(1)

    groups = df_t[GROUP_COL].dropna().unique()
    print(f"Running matching per group on column '{GROUP_COL}' with {len(groups)} groups")
    summary_rows = []
    for g in groups:
        safe = str(g).replace(" ", "_").replace("/", "_").replace("\\", "_")
        print(f"\n--- Group: {g} ---")
        df_t_g = df_t[df_t[GROUP_COL] == g].reset_index(drop=True)
        if df_t_g.shape[0] == 0:
            print("  (no treated rows for this group, skipping)")
            continue

        out_g = matched_dir / f"matched_pairs_{safe}.parquet"
        res_g = matcher.fit_match(df_t_g, df_c, save_matched_pairs=str(out_g))

        # compute highest and lowest individual treatment effect (delta) from matched_df
        md = res_g.matched_df
        if "delta" in md.columns:
            deltas = md[md["delta"].notna()]["delta"].astype(float)
            delta_max = float(deltas.max()) if not deltas.empty else float('nan')
            delta_min = float(deltas.min()) if not deltas.empty else float('nan')
        else:
            delta_max = float('nan')
            delta_min = float('nan')

        # compute matched fraction safely
        n_t = res_g.details.get("n_treated") or 0
        n_m = res_g.details.get("n_matched_treated") or 0
        matched_frac = float(n_m) / float(n_t) if n_t > 0 else float('nan')

        summary_rows.append({
            "family": str(g),
            "att_matched_only": res_g.att_matched_only,
            "ate_ipw": res_g.ate_ipw,
            "n_treated": res_g.details.get("n_treated"),
            "n_controls": res_g.details.get("n_controls"),
            "n_matched_treated": res_g.details.get("n_matched_treated"),
            "matched_fraction": matched_frac,
            "trim_fraction": float(1.0 - matched_frac) if (n_t > 0 and not pd.isna(matched_frac)) else float('nan'),
            "Mean_SMD_Before": res_g.details.get("Mean_SMD_Before"),
            "Mean_SMD_After": res_g.details.get("Mean_SMD_After"),
            "Max_SMD_After": res_g.details.get("Max_SMD_After"),
            "caliper": res_g.details.get("caliper"),
            "eta_sd": res_g.details.get("eta_sd"),
            "delta_max": delta_max,
            "delta_min": delta_min,
            
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = matched_dir / "summary_by_family.parquet"
    summary_df.to_parquet(summary_path, index=False)
    print("\nSummary by family:")
    print(summary_df)


if __name__ == "__main__":
    main()
