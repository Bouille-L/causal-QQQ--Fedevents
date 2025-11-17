# Layer3 — align treatment sample and control sample with its covariates and effects 

Collect U.S. macro event-style data from FRED over a fixed window and produce a strictly timed, unified event file for downstream use.




  align_covariates.py : the objective of this component is to align end-of-day (EOD) covariates from an options features file to a minute-level control dataset (QQQ).which be consedered as Control sample in our project. the alignement will applies rules for pre-market / regular / after-market timestamps when selecting which EOD row to attach following the Key rules bellow : 
        - Pre-market (< 09:30 ET) and regular session (09:30–16:00 ET): use the previous available EOD covariate (previous trading day).
        - After-market (> 16:00 ET): prefer same-day EOD covariate if it exists, otherwise fall back to the previous available EOD covariate.
        - All timestamp handling and session classification use America/New_York timezone.
    Inputs & Assumptions: 
        - control parquet (control_sample_QQQ.parquet or similar) contains:
            - a `timestamp` column (timezone-aware or naive) OR both `date` and `time` columns that can be combined.
        - covariates parquet (Options_features_L1_VF.parquet) contains:
            - a `Date` column (date or datetime) representing EOD covariate rows keyed by date.
        - Input datetimes are treated as Eastern Time (NY). Naive timestamps are localized to ET.

    Outputs : 
      - A parquet file with:
          - original minute-level columns,
      - added columns: event_date, session, cov_date, plus all covariate columns appended.
      - By default, rows with any NaNs are dropped before writing (use --keep-na to preserve them).
    

  layer3_align_events_E1.py : this component is responsible of aligning the end-of-day (EOD) covariates from an options features file to a minute-level control dataset (QQQ) and Fed Macro Event .which be consedered as Treatment sample in our project.
      - Core rule: each event is joined to option EOD features from the prior trading session so covariates represent information available before the event (no lookahead). by using :
        - Anchor_day is the event calendar day in ET;
        -  feat_date is the previous trading-session date relative to that anchor_day.
        - Session exceptions:
            - Pre-market (before 09:30 ET) and regular (09:30–16:00 ET): use the previous trading day’s EOD features.
            - After-market (after 16:00 ET): prefer same-day EOD features if they exist; otherwise fall back to the previous trading day.
            - Overnight: treated like pre/regular (use previous trading session).
            - Monday pre/regular events therefore pick Friday’s EOD features (previous trading day).
    Inputs
        Options_features_L1_VF.parquet — EOD covariates (joined on feat_date).
        events_unified_E1.parquet — event stream (event timestamps, event_id, etc.).
        QQQ_1min_2011-01_2025-07.parquet — minute prices for price_t0 and outcomes (requires date,time,price or timestamp,price).
        QQQ_daily.parquet (legacy daily prices) — fallback for some price lookups.
        event_reactions_E1.parquet (intermediate reactions) — optional depending on pipeline stage.
    Required columns (key)
        events: event timestamp (convertible to ET), event_id, any session/time fields.
        minute prices: timestamp or date+time and price.
        features: Date as EOD key.
    Outputs
        events_aligned_L3_E1.parquet — final one-row-per-event dataset with feat_date, covariates and computed outcomes (price_t0, pre/post returns, flags).
        events_aligned_L3_summary_E1.txt — coverage and diagnostic summary.
        layer3_stageC_policy_used_E1.json — recorded alignment / feature-selection policy used.
  
  
  layer3_align_events_E1.py
Purpose
- Align each event to the previous trading-session features (feat_date), compute price_t0 and pre/post outcome metrics using minute/daily prices, and produce a treatment dataset ready for causal analysis.

Key behavior
- Converts event times to US/Eastern and classifies session (pre_market/regular/after_hours/overnight).
- Computes anchor_day and feat_date = previous trading session for joining features.
- Looks up intraday prices with a tolerance (PRICE_T0_TOLERANCE_MIN) and computes pre/post log-returns and diagnostics.
- Drops events missing required timing or pricing data per policy.

Outputs
- events_aligned_L3_E1.parquet — one row per event with covariates and computed outcomes.
- events_aligned_L3_summary_E1.txt — coverage and diagnostic summary.
- layer3_stageC_policy_used_E1.json — recorded policy/config used for alignment.

## Requirements
- Python 3.8+
- pandas, numpy, pyarrow
- requests (if external data fetches are used)
- python-dotenv (optional)


## Requirements

At minimum, the following Python packages are likely required to run the scripts of this layer 2:

Required packages

pandas (>=1.3)
numpy (>=1.21)
pyarrow (>=5.0) or fastparquet — for reading/writing parquet
backports.zoneinfo (only if Python < 3.9) or use stdlib zoneinfo on 3.9+
python-dateutil (installed with pandas)

Optional / used in layer3_align_events_E1.py

requests (if any external data fetches)
python-dotenv (optional, for env config)
pytz (only if you prefer pytz timezone handling instead of zoneinfo)



