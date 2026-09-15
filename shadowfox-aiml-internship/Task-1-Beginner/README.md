# Task 1 (Beginner) — Boston House Price Prediction ✅

## 📌 Problem Statement
Using the provided dataset containing features such as number of rooms, crime rates, and other
relevant factors, design and implement a regression model to accurately predict Boston house prices.
Solution involves data preprocessing, model selection, training, and evaluation.

## 📊 Dataset
Boston Housing dataset (506 suburbs, 13 features + target `medv`), provided in the internship
materials — included here as `boston_housing.csv`.

## 🔧 Approach
1. **Health check** — no missing values; dtypes and ranges verified
2. **EDA** — price distribution, correlation heatmap (`rm` ≈ +0.70, `lstat` ≈ −0.74), key scatter plots
3. **Outliers** — **1.5×IQR technique** (from the internship overview) + removed 16 survey-capped rows (`medv` = 50) → 466 clean rows
4. **Split** — 80/20 train/test, `random_state=42`
5. **Model selection** — Linear Regression vs Decision Tree vs Random Forest vs Gradient Boosting
6. **Fine-tuning** — `RandomizedSearchCV` (40 iterations × 5-fold CV) on the champion
7. **Validation** — feature importances, actual-vs-predicted plot
8. **Export** — `joblib` → `boston_price_model.pkl`

## 📈 Results

| Model | R² (↑) | RMSE ($1000s) (↓) |
|---|---|---|
| Linear Regression | 0.75 | 3.15 |
| Decision Tree | 0.73 | 3.27 |
| Random Forest | 0.85 | 2.43 |
| **Gradient Boosting (FINAL)** | **0.875** | **2.24** |

Top price drivers (feature importance): `lstat`, `rm`, `dis` — matching the EDA.

## ⚖️ Ethical note
scikit-learn removed this dataset over the racial bias embedded in the `b` feature; used here
as specified by the task, with a fairness caveat documented in the notebook.

## ▶️ How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn joblib jupyter
jupyter notebook Boston_House_Price_Prediction.ipynb
```

📖 New to this? Read **`EXPLAIN.md`** — the 60-second story, concept cheat-sheet and interview Q&A.
