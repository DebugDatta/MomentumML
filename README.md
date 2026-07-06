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

Performance metrics are reported before and after transaction costs:

- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Max drawdown
- Hit rate
- Average weekly return

## Deliverables

- [`main.py`](main.py) – Full implementation (data, features, modeling, backtest, plots)
- `output/features.csv` – Engineered features
- `output/model_classification_metrics.csv` – Static split model comparison
- `output/walk_forward_roc_auc.csv` – Walk-forward ROC AUC per week
- `output/predictions_with_probs.csv` – Weekly predictions with probabilities
- `output/selected_stocks.csv` – Selected stocks per week
- `output/weekly_stock_predictions_ensemble.csv` – Predictions with selection & weights
- `output/portfolio_returns.csv` – Portfolio returns before/after costs
- `output/metrics.csv` – Performance metrics
- `output/backtest_plots.png` – Cumulative return, drawdown, ROC AUC, feature importance

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
