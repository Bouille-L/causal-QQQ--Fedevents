# Layer1 — Options data collection and cleaning

The objective of this folder is to build and enginner features using option data that can be used as cavariates in in our project, it devide to foor components :

  ## Data_collection.py
  this is component we colllection option data from 2011-06-17 till 2025-06-16 using Orats API and due the computation cost, it was required for us  to use parquet file as outcome file format  due to large size of data and   also, to download the data by 2 years tim frame till we reach 2025 due the limitation of our API imposed by our suppliers and high cost computation, then we merged all this file in Layer1_A.py. 

  ## Layer1_A.py
  this component is responsible of merging all parquet file we get from data_collection.py and form one parquet file tha host all option data from 2011 to 2025 then inside the same one file, we builds daily, tenorized      
  option features dataset using features bellow. as Output, we get one parquet file where we have one row per date, with engineered option data features across tenor buckets. :
      ** Tenor buckets: by grouping contracts by days-to-expiry:  D01, D02, D03, D05, D07, D10, D30, D60, D>60 

      ** Features : for each of the tenor, we engineer and cmputes features bellow :
          Forward & Moneyness
            Computes forward price (Fwd) using spot price and residual interest rate.
            Derives log-moneyness (logM).
          Liquidity & Market Quality
            Relative bid/ask spreads (c_rel_spread, p_rel_spread)   
            Dollar volumes (c_dollar_vol, p_dollar_vol).
            Vega weights (vega_w).
          Volatility Features
            ATM implied volatility (ATM_IV).
            Risk reversal 25Δ (RR25).
            Term-structure slopes (TS_7_30, TS_30_60).
            Skew decay (SKEW_DECAY_30D).
          Volume & Open Interest Features
            Put/Call Ratios (dollar, vega-weighted, open interest based).
                Dollar-volume based (PCR_DOLLAR_OTM)
                Vega-weighted (PCR_VEGA_OTM)
                Open-interest based (PCR_OI)
            Net open interest (NET_OI).
            Turnover (volume/open interest).
            Volume mix shares (VOLMIX_*).
          Liquidity Composite
            Relative spread weighted by dollar flow (REL_SPREAD).


  ## Layer1_B.py
  this component objective is to load the option engineered featutes stored in parquet file and produces per tenor descriptive statistic, data-driven threshold proposals (liquidity/participation quality), and winsorization 
  caps for volatile ratios. It exports CSV summaries plus a concise Markdown report. so we can assess feature quality and guide cleaning choices in Layer-1C..
  ### inputs
    Options_features_L1_A.parquet (from Layer-1A)
  ### outputs
        - NaN share for every column.
        - stats_<family>.csv — Percentiles & robust spread per feature family and tenor.
        - proposed_thresholds.csv — Tenor-specific keep thresholds for REL_SPREAD_*, OI_SUM_*, and DOLLAR_VOL_*.
        - proposed_winsor_caps.csv — Tenor-specific p1/p99 caps for volatile ratios (PCR_*, TURNOVER_*).
        


  ## Layer1_C.py
  thic component is final component fo the layer 1 and its objective is to produce a clean, modeling-ready option features by executing this three  steps concluded after reviewng the outcomes reports from layer1_B:
      - Dropping short-tenor buckets (D01, D02, D03, D05, D07) and any derived short-tenor features (TS_7_30).
      - Masking (to NaN) per-tenor values that fail quality thresholds (liquidity & participation).
      - Winsorizing heavy-tailed ratios (PCR_*, TURNOVER_*) using 1st/99th percentile caps from Layer-1B.
      - Emitting a summary, exact config used, and a column manifest.
  ### as outputs, we get : 
      Options_features_L1_VF.parquet — Cleaned, tenor-filtered features (Date-indexed, modeling-ready).
      L1C_cleaning_summary.csv — What was masked per tenor (thresholds) and clipped per column (winsorization).
      L1C_cleaning_config_used.json — Exact tenors dropped/kept, thresholds, and winsor caps (full reproducibility).
      L1C_columns_kept.csv — Manifest of columns that survived cleaning, with inferred tenor tags (useful for appendix).


## Requirements

At minimum, the following Python packages are likely required to run the scripts of this layer 1:

- Python 3.8+ (pyarrow parquet reading/writing prefers modern Python)
- pandas
- pyarrow
- numpy





