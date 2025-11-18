# Layer2 — FRED Macro Event Collection & Unification

Collect U.S. macro event-style data from FRED over a fixed window and produce a strictly timed, unified event file for downstream use.




  ## fred_datacollection.py 
  the objective of this component is to build a single event-style dataset of U.S. macro releases from FRED over a hard-wired date window (2011-06-17 to 2025-06-16) using official release calendar and pull first release 
  (as published) event value,  

    what it collects (as fed macro events and values)
      PPI: ppi_headline_yoy (PPIACO, pc1), ppi_headline_mom (PPIACO, pch)
      HousingStarts: housing_starts_mom (HOUST, pch), housing_starts_yoy (HOUST, pc1)
      JOLTS: jolts_job_openings_mom (JTSJOL, pch), …_yoy (pc1)
      MichiganSentiment: …_mom (UMCSENT, pch), …_yoy (pc1)
      TradePrices: import_prices_mom (IR, pch), …_yoy (pc1), export_prices_mom (IQ, pch), …_yoy (pc1)
      RetailSales: retail_sales_mom (RSXFS, pch), …_yoy (pc1)
      IndustrialProduction: …_mom (INDPRO, pch), …_yoy (pc1)
      CPI: cpi_headline_mom (CPIAUCSL, pch), …_yoy (pc1)
      UnemploymentRate: unemployment_rate (UNRATE, lin)
      GDP: gdp_qoq (GDP, pch), gdp_yoy (pc1)
      FOMC: fed_funds_target_upper (DFEDTARU, lin), …_lower (DFEDTARL), fed_funds_effective (FEDFUNDS)
      InitialJoblessClaims: initial_claims_level (ICSA, lin), …_wow_change (chg)
      DurableGoodsOrders: …_mom (DGORDER, pch), …_yoy (pc1)
      NonfarmPayrolls: nfp_payroll_change (PAYEMS, chg), unemployment_rate (UNRATE, lin)
      PCE/Core PCE: pce_price_index_mom (PCEPI, pch), …_yoy (pc1), core_pce_mom (PCEPILFE, pch), …_yoy (pc1)


    Ouputs :
      macro_events_2011-09-13_2025-06-12_allreports.parquet — Event table with release dates/times and value blobs (JSON strings). with columns include: event_id, event_type, series_id, values.

  ## Layer2_format_events.py 
  this component is responsible of unifeing fred macro evenst dataset we collected from component "fred_datacollection.py " and make sure is striclty timed  using policy bellow : 
        STRICT timing: rows without a release_time are dropped (no fallbacks).
        All timestamps are tz-aware and localized to America/New_York with DST-safe handling.
        JSON-like value columns are normalized to Python objects (dict/list) for reliability.
        
    Outputs:
        events_unified_E1.parquet


## Requirements

At minimum, the following Python packages are likely required to run the scripts of this layer 2:

- Python 3.8+ (pyarrow parquet reading/writing prefers modern Python)
- pandas, numpy, pyarrow, requests, python-dotenv





