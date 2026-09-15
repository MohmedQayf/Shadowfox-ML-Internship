# Task 2 (Intermediate) — Car Selling Price Prediction + Flask Deployment ✅

## 📌 Problem Statement
Develop an ML model for car selling price prediction and analysis. The deployed system provides an
approximate selling price based on fuel type, years of service, showroom price, previous owners,
kilometers driven, seller type, and transmission — deployed as a web application/API (Flask).

## 📊 Dataset
CarDekho listings (301 cars, 9 columns), provided in the internship materials — included as `car_data.csv`.
Prices are in **lakhs** (₹100,000).

## 🔧 Approach
1. **Health check** — 0 missing values; categorical levels inspected
2. **EDA** — price & age distributions, fuel/seller/transmission counts, showroom-vs-selling trend (+0.88)
3. **Feature engineering** — `Car_Age = current_year − Year` (the "years of service"); dropped free-text `Car_Name` (high cardinality)
4. **Encoding** — one-hot with `drop_first=True` (avoids dummy trap) → 8 final features
5. **Model** — `RandomForestRegressor` → tuned via `RandomizedSearchCV` (40 iters × 5-fold CV)
6. **Evaluation** — MSE/RMSE/MAE/R² on hidden 20% test set + actual-vs-predicted plot
7. **Deployment** — Flask app: HTML form (`/`), JSON API (`/api/predict`), health check (`/health`)

## 📈 Results
- **Test R² = 0.96**, RMSE = 0.96 lakh, MAE = 0.64 lakh
- Top drivers: `Present_Price` (showroom price), `Car_Age` — feature importance plot in the notebook
- Model artifacts: `app/car_price_model.pkl`, `app/feature_columns.pkl` (exact feature order), `app/model_info.json`

## 🌐 Run the deployed app
```bash
cd app
pip install -r requirements.txt
python app.py
# Browser: http://localhost:5000      (form UI)
# API:     POST /api/predict  with JSON:
# {"present_price": 8.5, "year": 2016, "kms_driven": 42000, "owner": 0,
#  "fuel_type": "Petrol", "seller_type": "Dealer", "transmission": "Manual"}
```

## 🔁 Monitoring (production note)
The notebook documents production monitoring: log requests/predictions, track residual drift
against real sale prices, retrain on fresh listings periodically.

📖 New to this? Read **`EXPLAIN.md`** — the 60-second story, the 8-feature mapping, deployment Q&A.
