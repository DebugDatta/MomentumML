import warnings
warnings.filterwarnings('ignore')
import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'BRK-B']
start_date = '2017-01-01'
end_date = '2025-12-31'
COST_PER_SIDE = 0.001  # 0.1% entry, 0.1% exit
TOP_N = 2
OUTPUT_DIR = 'output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(name):
    return os.path.join(OUTPUT_DIR, name)

def log(msg):
    t = time.strftime('%H:%M:%S')
    print(f"[{t}] {msg}", flush=True)

log("Downloading data...")
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
log(f"Downloaded {len(data)} rows x {len(data.columns)} tickers")

weekly = data.resample('W-FRI').last()
returns = weekly.pct_change()
log(f"Weekly data: {len(weekly)} weeks")

log("Engineering features...")
feature_list = []
for t in tickers:
    s = returns[t]
    df_t = pd.DataFrame(index=weekly.index)
    df_t['Ticker'] = t
    df_t['Ret_W1'] = s
    df_t['Ret_W2'] = s.shift(1).rolling(2).apply(lambda x: (1 + x).prod() - 1)
    df_t['Ret_W4'] = s.shift(3).rolling(4).apply(lambda x: (1 + x).prod() - 1)
    df_t['Vol_4W'] = s.rolling(4).std()
    df_t['Target'] = (s.shift(-1) > 0).astype(int)
    feature_list.append(df_t)

features = pd.concat(feature_list).dropna()
feature_cols = ['Ret_W1', 'Ret_W2', 'Ret_W4', 'Vol_4W']
log(f"Features shape: {features.shape}, date range: {features.index.min()} to {features.index.max()}")

log("Fitting static split models...")
train_static = features[features.index.year <= 2022]
test_static = features[features.index.year >= 2023]

lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
xgb = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
model = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)], voting='soft'
)
model.fit(train_static[feature_cols], train_static['Target'])
log("Static models fitted")

X_ts = test_static[feature_cols]
y_ts = test_static['Target']
eval_rows = []
for name, m in zip(['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble'], [*model.estimators_, model]):
    yp = m.predict(X_ts)
    yprob = m.predict_proba(X_ts)[:, 1]
    eval_rows.append({
        'Model': name, 'Accuracy': accuracy_score(y_ts, yp),
        'Precision': precision_score(y_ts, yp), 'Recall': recall_score(y_ts, yp),
        'F1 Score': f1_score(y_ts, yp), 'ROC AUC': roc_auc_score(y_ts, yprob)
    })

pd.DataFrame(eval_rows).set_index('Model').round(4).to_csv(out('model_classification_metrics.csv'))

log("Starting walk-forward retraining (retrain every 4 weeks, predict weekly)...")
test_weeks = sorted(features[features.index.year >= 2023].index.unique())
wf_rows = []
roc_tracker = []

total = len(test_weeks)
t_start = time.time()
last_model = None

for idx, wd in enumerate(test_weeks):
    train_wf = features[features.index < wd].dropna()
    test_wf = features[features.index == wd]
    if len(train_wf) < 50 or len(test_wf) == 0:
        continue

    if idx % 4 == 0 or last_model is None:
        lr_wf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        rf_wf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_wf = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0)
        last_model = VotingClassifier(
            estimators=[('lr', lr_wf), ('rf', rf_wf), ('xgb', xgb_wf)], voting='soft'
        )
        last_model.fit(train_wf[feature_cols], train_wf['Target'])
        log(f"  Retrained week {idx+1}/{total}")

    probs = last_model.predict_proba(test_wf[feature_cols])[:, 1]

    for j, row_idx in enumerate(test_wf.index):
        wf_rows.append({'Date': row_idx, 'Ticker': test_wf['Ticker'].values[j], 'Prob': probs[j]})

    if len(test_wf) > 1:
        roc_tracker.append({'Date': wd, 'ROC_AUC': roc_auc_score(test_wf['Target'], probs)})

    if (idx + 1) % 20 == 0 or idx == total - 1:
        elapsed = time.time() - t_start
        log(f"  Progress: {idx+1}/{total} | {elapsed:.0f}s elapsed | last ROC AUC = {roc_tracker[-1]['ROC_AUC']:.3f}")

wf_preds = pd.DataFrame(wf_rows)
roc_df = pd.DataFrame(roc_tracker)
roc_df.to_csv(out('walk_forward_roc_auc.csv'), index=False)
log(f"Walk-forward done: {len(roc_df)} weeks, mean ROC AUC = {roc_df['ROC_AUC'].mean():.4f}")

log("Running portfolio backtest...")
def safe_top_n(group, n=TOP_N):
    picks = group.nlargest(n, 'Prob')['Ticker'].tolist()
    return picks if len(picks) == n else []

selected = wf_preds.groupby('Date', sort=True).apply(safe_top_n)
selected = selected[selected.apply(len) == TOP_N]
log(f"Selected stocks for {len(selected)} weeks")

ret_before, ret_after = [], []
prev = []
dates = selected.index

for i in range(len(dates) - 1):
    date, nxt = dates[i], dates[i + 1]
    cur = selected.loc[date]
    r1, r2 = returns.loc[nxt, cur[0]], returns.loc[nxt, cur[1]]
    port_r = 0.5 * r1 + 0.5 * r2
    ret_before.append(port_r)

    if not prev:
        to = 1.0
    else:
        to = sum(0.5 for s in cur if s not in prev) + sum(0.5 for s in prev if s not in cur)
    ret_after.append(port_r - to * COST_PER_SIDE)
    prev = cur

results = pd.DataFrame({'Ret_Before': ret_before, 'Ret_After': ret_after}, index=dates[:-1])

log("Calculating portfolio metrics...")
def calc_metrics(r):
    cum = (1 + r).prod() - 1
    ann_r = (1 + cum) ** (52 / len(r)) - 1
    ann_v = r.std() * np.sqrt(52)
    sharpe = ann_r / ann_v if ann_v else 0
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return {
        'Cumulative Return': cum, 'Annualized Return': ann_r,
        'Annualized Volatility': ann_v, 'Sharpe Ratio': sharpe,
        'Max Drawdown': dd.min(), 'Hit Rate': (r > 0).mean(),
        'Average Weekly Return': r.mean()
    }

metrics_df = pd.DataFrame({
    'Before Costs': calc_metrics(results['Ret_Before']),
    'After Costs': calc_metrics(results['Ret_After'])
})
log("Portfolio performance:")
print(metrics_df.round(4))
metrics_df.to_csv(out('metrics.csv'))

log("Exporting CSVs...")
features.to_csv(out('features.csv'))
train_static.to_csv(out('train_data.csv'))
test_static.to_csv(out('test_data.csv'))
weekly.to_csv(out('weekly_prices.csv'))
returns.to_csv(out('weekly_returns.csv'))
results.to_csv(out('portfolio_returns.csv'))

wf_preds.to_csv(out('predictions_with_probs.csv'), index=False)

selected_set = {d: v for d, v in selected.items()}
wf_preds['Selected'] = wf_preds.apply(lambda r: r['Ticker'] in selected_set.get(r['Date'], []), axis=1)
wf_preds['Weight'] = wf_preds['Selected'].astype(float) * (1.0 / TOP_N)
wf_preds.to_csv(out('weekly_stock_predictions_ensemble.csv'), index=False)

sel_df = pd.DataFrame(selected.tolist(), index=selected.index, columns=['Stock1', 'Stock2'])
sel_df.to_csv(out('selected_stocks.csv'))

log("Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot((1 + results['Ret_Before']).cumprod(), label='Before Costs')
axes[0, 0].plot((1 + results['Ret_After']).cumprod(), label='After Costs')
axes[0, 0].set_title('Cumulative Return')
axes[0, 0].legend()
axes[0, 0].grid(True)

wealth = (1 + results['Ret_After']).cumprod()
dd = wealth / wealth.cummax() - 1
axes[0, 1].fill_between(dd.index, dd * 100, 0, color='red', alpha=0.3)
axes[0, 1].set_title('Drawdown (%)')
axes[0, 1].grid(True)

if len(roc_df) > 0:
    axes[1, 0].plot(roc_df['Date'], roc_df['ROC_AUC'], marker='.', label='ROC AUC')
    axes[1, 0].axhline(0.5, color='gray', linestyle='--', label='Random')
    axes[1, 0].set_title('Walk-Forward ROC AUC per Week')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

imp = pd.Series(model.estimators_[1].feature_importances_, index=feature_cols).sort_values()
axes[1, 1].barh(imp.index, imp.values)
axes[1, 1].set_title('Random Forest Feature Importance')

plt.tight_layout()
plt.savefig(out('backtest_plots.png'), dpi=150, bbox_inches='tight')
log("Done!")
plt.show()
