# Momentum Trading Strategy using Machine Learning

A machine learning-based long-only momentum strategy that predicts weekly stock returns and constructs an equal-weight portfolio.

## Objective

Build and backtest a machine learning-based long-only momentum strategy in Python using daily stock data.

## Stock Universe

AAPL, MSFT, GOOGL, AMZN, META, TSLA, JPM, V, JNJ, BRK-B

Data sourced from Yahoo Finance (yfinance). Time period: 2017–2025.

## Strategy

1. Predict whether each stock will generate a positive return over the next week using ensemble classification (Logistic Regression, Random Forest, XGBoost).
2. Rank all 10 stocks each week by predicted probability of positive next-week return.
3. Select the top 2 stocks.
4. Construct an equal-weight long-only portfolio.
5. Hold for 1 week and rebalance weekly.
6. Transaction costs: 0.1% entry + 0.1% exit.

## Validation

- **Static split:** Train on 2017–2022, test on 2023–2025.
- **Walk-forward:** Retrain every 4 weeks, predict weekly from 2023 onward.

## Features

| Feature | Description |
|---|---|
| Ret_W1 | Prior 1-week return |
| Ret_W2 | Prior 2-week cumulative return (non-overlapping) |
| Ret_W4 | Prior 4-week cumulative return (non-overlapping) |
| Vol_4W | 4-week rolling volatility |

## Results

### Model Classification (Static Split 2017–2022 train / 2023–2025 test)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 51.46% | 56.34% | 66.03% | 0.6080 | 0.4753 |
| Random Forest | 53.18% | 57.84% | 65.92% | 0.6162 | 0.5031 |
| XGBoost | 52.36% | 57.39% | 63.80% | 0.6042 | 0.5124 |
| **Ensemble (Voting)** | **53.31%** | **57.76%** | **67.37%** | **0.6220** | **0.5063** |

### Walk-Forward ROC AUC

- Mean ROC AUC across 140 weekly predictions: **0.5192**
- Range: 0.0 – 1.0
- Outperforms random (0.5) on average

### Portfolio Performance (2023–2025)

| Metric | Before Costs | After Costs |
|---|---|---|
| Cumulative Return | **+126.86%** | **+77.06%** |
| Annualized Return | 31.40% | 20.98% |
| Annualized Volatility | 25.08% | 25.05% |
| Sharpe Ratio | **1.25** | **0.84** |
| Max Drawdown | –18.61% | –19.68% |
| Hit Rate | 56.41% | 55.13% |
| Average Weekly Return | 0.59% | 0.43% |

### Selected Stocks (first 5 weeks)

| Date | Stock 1 | Stock 2 |
|---|---|---|
| 2023-01-06 | META | V |
| 2023-01-13 | JNJ | AMZN |
| 2023-01-20 | AMZN | GOOGL |
| 2023-01-27 | AMZN | V |
| 2023-02-03 | TSLA | V |

## Deliverables

- [`main.py`](main.py) – Full implementation (data, features, modeling, backtest, plots)

### Output Files (13 total)

| File | Description |
|---|---|
| `output/backtest_plots.png` | Cumulative return, drawdown, ROC AUC, feature importance |
| `output/features.csv` | Engineered features (4630 rows, 7 cols: Date, Ticker, Ret_W1, Ret_W2, Ret_W4, Vol_4W, Target) |
| `output/metrics.csv` | Portfolio performance summary (7 metrics) |
| `output/model_classification_metrics.csv` | Static split model comparison (4 models) |
| `output/portfolio_returns.csv` | Portfolio returns before/after costs (156 weeks) |
| `output/predictions_with_probs.csv` | Weekly predictions with probabilities (1570 rows) |
| `output/selected_stocks.csv` | Selected stocks per week (157 weeks) |
| `output/test_data.csv` | Test set 2023–2025 (1570 rows) |
| `output/train_data.csv` | Training set 2017–2022 (3060 rows) |
| `output/walk_forward_roc_auc.csv` | Walk-forward ROC AUC per week (157 weeks, 140 non-null) |
| `output/weekly_prices.csv` | Weekly adjusted close prices (470 weeks) |
| `output/weekly_returns.csv` | Weekly returns (470 weeks) |
| `output/weekly_stock_predictions_ensemble.csv` | Predictions with selection & weights (1570 rows) |

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
xgboost>=2.0.0
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Output

All CSVs and plots are saved to the `output/` directory (gitignored).
