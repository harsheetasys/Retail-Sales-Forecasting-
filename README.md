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
