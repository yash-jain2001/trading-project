# Quantitative Trading Strategy Development

## Overview

This project implements a complete quantitative trading system for NIFTY 50, featuring:

- **Data Pipeline**: Fetching, cleaning, and merging NIFTY Spot, Futures, and Options data
- **Feature Engineering**: EMAs, Options Greeks (Black-Scholes), IV metrics, PCR, and derived features
- **Regime Detection**: Hidden Markov Model (HMM) to classify market into Uptrend/Downtrend/Sideways
- **Trading Strategy**: 5/15 EMA crossover with regime filter
- **ML Enhancement**: XGBoost and LSTM models for trade profitability prediction
- **Outlier Analysis**: Statistical analysis of high-performance trades

## Project Structure

```
trading project/
├── data/                          # CSV data files
│   ├── nifty_spot_5min.csv
│   ├── nifty_futures_5min.csv
│   ├── nifty_options_5min.csv
│   ├── nifty_merged_5min.csv
│   ├── nifty_features_5min.csv
│   └── data_cleaning_report.txt
├── models/                        # Saved ML models
├── notebooks/                     # Jupyter notebooks (7 notebooks)
├── plots/                         # Visualizations
├── results/                       # Backtest results
│   ├── backtest_trades.csv
│   ├── analysis_summary.txt
│   └── outlier_feature_comparison.csv
├── src/                           # Python source code
│   ├── __init__.py
│   ├── data_utils.py              # Data fetching and cleaning
│   ├── features.py                # Feature engineering (EMAs, Greeks)
│   ├── regime.py                  # HMM regime detection
│   ├── strategy.py                # Trading strategy implementation
│   ├── backtest.py                # Performance metrics calculation
│   ├── ml_models.py               # XGBoost and LSTM models
│   └── analysis.py                # Outlier analysis
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Pipeline

```bash
python src/fetch_data.py      # Generate/fetch data
python src/process_data.py    # Clean and merge data
```

### 2. Feature Engineering

```bash
python src/engineer_features.py
```

### 3. Backtesting

```bash
python src/backtest_complete.py
```

### 4. ML Models

```bash
python src/ml_models.py
```

### 5. Outlier Analysis

```bash
python src/analysis.py
```

## Key Results

### Baseline Strategy Performance (Test Set: 30%)

| Metric | Value |
|--------|-------|
| Total Trades | 116 |
| Win Rate | 38.79% |
| Profitable Trades | 45 |

### Outlier Analysis

- Outlier trades (Z > 3): 2 (4.4% of profitable)
- Average Outlier PnL: 77.45
- Average Normal PnL: 11.66
- Key distinguishing features: Futures Returns, IV Spread, PCR

## Technical Details

### Strategy Logic

- **Long Entry**: 5 EMA crosses above 15 EMA AND Regime = +1 (Uptrend)
- **Short Entry**: 5 EMA crosses below 15 EMA AND Regime = -1 (Downtrend)
- **Exit**: Opposite crossover signal
- **No trades in Regime 0 (Sideways)**

### HMM Features

- Average IV, IV Spread
- PCR (OI-based)
- ATM Delta, Gamma, Vega
- Futures Basis
- Spot Returns

### ML Models

- **XGBoost**: Gradient boosting classifier for trade profitability
- **LSTM**: Sequence model using last 10 candles

## Author

Priyanshu Jain
