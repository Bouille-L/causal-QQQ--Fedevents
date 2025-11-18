# Causal_QQQET_FedEevents

Causal QQQ FedEevents is  a Causal Inference Pipeline for Fed Macroeconomic Events Using Options-Derived Covariates on QQQ. This is a multi-layered Python project for causal analysis of financial and economic events, focusing on the QQQ ETF and Federal Reserve events. 
The project is organized into several layers, each responsible for different stages of data collection, cleaning, event alignment, and causal inference.

## Project Structure

- **Layer1/**: Data collection and cleaning for base datasets. Includes configuration files, cleaning scripts, and summary reports.
- **Layer2/**: Event formatting and integration of external datasets (e.g., FOMC statements, FRED data).
- **Layer3/**: High-frequency price data collection and alignment of covariates/events for treatment/control datasets.
- **Layer4/**: Causal matching and synthetic control methods, with outputs for matched pairs and event results.

## Key Files & Directories

- `align_events_version 3minute .py`: Main script for aligning events at a 3-minute frequency.
- `event_value_keys_selected.json`: Selected event value keys for analysis.
- `Layer1/`: Data cleaning and summary reports.
- `Layer2/`: Event formatting and external data integration.
- `Layer3/`: Price data collection and event alignment.
- `Layer4/`: Matching algorithms and causal inference outputs.

## Usage

1. **Data Preparation**: Start with Layer1 scripts to clean and prepare raw data.
2. **Event Formatting**: Use Layer2 scripts to format and integrate event datasets.
3. **Event Alignment**: Run Layer3 scripts to align price data and events for analysis.
4. **Causal Analysis**: Use Layer4 scripts for matching and synthetic control analysis.

## Requirements

- Python 3.x
-  The requirements for each layer are provided in the corresponding layer’s README file.

## Outputs

- Cleaned datasets, event-aligned summaries, matched pairs, and causal inference results.


## How to Run

Navigate to each layer and follow the instructions in the respective README files. Run scripts in the recommended order for end-to-end analysis.

## Important Notes

- Large output files (such as Parquet and Excel) are not included in this repository due to their size. You can generate these files by running the scripts in each layer.
- Many scripts use hardcoded file paths for loading and saving data. Please adjust these paths in the code to match your local directory structure and environment.
