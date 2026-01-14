# Quantitative Trading Strategy Development

## Overview
Complete quantitative trading system for NIFTY 50 with data pipeline, feature engineering, HMM regime detection, EMA crossover strategy, ML enhancement, and outlier analysis.

## Structure

```
├── data/           # CSV data files
├── models/         # Saved ML models
├── notebooks/      # 7 Jupyter notebooks
├── plots/          # Visualizations
├── results/        # Backtest results
├── src/            # Python modules
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate data
python src/fetch_data.py
python src/process_data.py

# 2. Engineer features
python src/engineer_features.py

# 3. Run backtest
python src/backtest_complete.py

# 4. Analyze results
python src/analysis.py
```

## Results

| Metric | Value |
|--------|-------|
| Total Trades | 116 |
| Win Rate | 38.79% |
| Sharpe Ratio | -0.44 |
| Outlier Trades | 2 (4.4%) |

## Strategy

- **Entry**: 5/15 EMA crossover + Regime filter
- **Long**: EMA cross up + Regime=+1
- **Short**: EMA cross down + Regime=-1
- **No trades in Sideways (Regime=0)**

## Author
Priyanshu Jain
