# 🛍️ Retail Sales Forecasting with XGBoost

This project builds a machine learning pipeline to forecast **daily sales** using real-world Brazilian retail data. Using `XGBoost`, I predicted short-term sales trends by engineering lag-based and temporal features, achieving solid results with interpretable evaluation metrics.

---

## 📂 Dataset

- **Name:** mock_kaggle.csv
- **Source:** Simulated real-world retail dataset (Brazilian region)
- **Rows:** ~1000
- **Columns:**
  - `data` → date of transaction (YYYY-MM-DD)
  - `venda` → total sales for the day (target)
  - `estoque` → inventory level
  - `preco` → unit price of product

---

## 🧠 Problem Statement

> Can we accurately predict daily sales using previous patterns, calendar trends, and business variables like price and inventory?

This can help retail businesses optimize:
- Inventory restocking  
- Dynamic pricing  
- Warehouse planning

---

## 🛠 Feature Engineering

| Feature       | Description                         |
|---------------|-------------------------------------|
| `day`, `month`, `year` | Date components |
| `dayofweek`            | Weekday (0=Monday) |
| `is_weekend`           | Boolean for Sat/Sun |
| `lag1`, `lag3`, `lag7` | Sales 1, 3, 7 days ago |
| `rolling7`             | 7-day moving average of sales |
| `estoque`              | Stock level |
| `preco`                | Price (marketing signal) |

📌 Target `venda` was log-transformed to stabilize variance.

---

## 🤖 Model: XGBoost Regressor

```python
XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
---

## 📈 Model Evaluation

- ✅ Trained on **80%** of historical time-series data  
- ✅ Tested on the **most recent 20%**  
- ✅ Applied log transformation using `np.log1p()` for target stability  
- ✅ Used `np.expm1()` to **invert predictions** after training

---

## 📊 Metrics & Interpretation

| Metric | Value     |
|--------|-----------|
| RMSE   | **87.15** |
| SMAPE  | **48.27%** |

### 📉 What This Means

- **RMSE** shows that the model captures the **magnitude of daily sales** effectively  
- **SMAPE** is elevated due to:
  - Extremely low sales days inflating percent errors  
  - Irregular demand spikes with no promo/holiday signals

### 🔧 How to Improve

- Add **holiday/promotion flags**
- Include **product category or region segmentation**
- Experiment with **LSTM or hybrid models**



