# 🎓 Task 2 — How to Explain Your Project (Car Selling Price Prediction + Flask)

## The 60-second story (memorize this)

> "The task was to predict a used car's selling price from showroom price, age, kms driven,
> fuel type, seller type, transmission and owners — and deploy it. The CarDekho dataset has
> 301 listings with prices in lakhs. EDA showed showroom price correlates +0.88 with selling
> price, and age depreciates value. I engineered `Car_Age` from the year, dropped `Car_Name`
> (too high cardinality), and one-hot encoded the categories with `drop_first=True` to avoid
> the dummy trap. A Random Forest — chosen because it handles non-linear tabular data without
> scaling — reached **R² = 0.96, RMSE ≈ 0.96 lakh** on the hidden test set; I tuned it with
> RandomizedSearchCV and kept the honest winner. Feature importance confirmed showroom price
> and age dominate. Finally I deployed it as a **Flask app**: an HTML form plus a JSON API,
> loading the pickled model and the saved feature-order file."

## The pipeline in one picture

```
car_data.csv (301 rows)
 → health check (0 missing)
 → EDA (distributions, countplots, Present_Price vs Selling_Price)
 → engineer Car_Age = this_year − Year ; drop Car_Name
 → one-hot encode (Fuel_Type, Seller_Type, Transmission), drop_first
 → 8 final features → train/test 80/20
 → RandomForestRegressor baseline → RandomizedSearchCV (40 iters × 5-fold)
 → R² = 0.96 / RMSE = 0.96 lakh / MAE = 0.64 lakh
 → joblib: car_price_model.pkl + feature_columns.pkl + model_info.json
 → Flask:  /  (HTML form)   /api/predict  (JSON API)   /health
```

## The 8 final model inputs (know this table cold — it's how the web form maps to the model)

| Feature | Type | From form field |
|---|---|---|
| `Present_Price` | number | showroom price (lakhs) |
| `Kms_Driven` | number | kilometers |
| `Owner` | number | previous owners |
| `Car_Age` | derived | `current_year − year` |
| `Fuel_Type_Diesel` | 0/1 dummy | fuel = Diesel |
| `Fuel_Type_Petrol` | 0/1 dummy | fuel = Petrol (CNG = both 0 → **base class**) |
| `Seller_Type_Individual` | 0/1 dummy | seller = Individual (Dealer = base) |
| `Transmission_Manual` | 0/1 dummy | transmission = Manual (Automatic = base) |

## Concept cheat-sheet

| Concept | Say it like this |
|---|---|
| **Lakh** | "Indian unit: 1 lakh = ₹100,000. Prices like 5.59 = ₹5.59 lakh." |
| **Feature engineering** | "Creating better inputs from raw ones — here: `Car_Age` from `Year` (the 'years of service' the task asked for)." |
| **High cardinality** | "`Car_Name` has ~100 unique spellings for 301 rows — it memorizes instead of generalizing, so I dropped it." |
| **One-hot encoding** | "Turning each category value into a 0/1 column; models only understand numbers." |
| **Dummy trap** | "With all dummies kept, one column is perfectly redundant (sum of the others = 1) — `drop_first=True` drops it; the dropped value becomes the base case." |
| **Random Forest here** | "Mixed numeric + dummies, curved relationships (depreciation flattens with age), outliers (luxury tail) — RF handles all of it without scaling." |
| **RandomizedSearchCV** | "Randomly samples 40 hyperparameter combos × 5-fold CV — much cheaper than GridSearchCV's exhaustive grid." |
| **R² = 0.96** | "96% of resale-price variation explained on cars the model never saw." |
| **RMSE 0.96 lakh** | "Typical prediction miss ≈ ₹96k — expected given the small dataset's premium-car tail." |
| **feature_columns.pkl** | "The exact training column order. The app rebuilds each request in that exact order — wrong order = silently wrong predictions." |
| **Flask routes** | "`/` renders the form, `/predict` handles form POST, `/api/predict` returns JSON for programmatic users, `/health` for monitoring." |
| **Joblib pickle** | "Serializes the fitted forest to disk; training once, serving forever." |
| **Monitoring** | "Production note in the notebook: log predictions, watch for input drift (newer car years!), retrain quarterly." |

## API example (for the video, show this JSON answer)

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"present_price": 8.5, "year": 2016, "kms_driven": 42000, "owner": 0,
       "fuel_type": "Petrol", "seller_type": "Dealer", "transmission": "Manual"}'
# → {"predicted_selling_price_lakhs": 6.18, "r2_test": 0.96, ...}
```

## Likely mentor questions → your answers

1. **Why Random Forest and not Linear Regression?** — "Depreciation is non-linear and interactions matter (automatic+new+petrol ≠ sum of parts); trees capture that — and R² 0.96 proves it."
2. **What is the dummy variable trap?** — "Perfect multicollinearity among dummy columns; dropping the first category fixes it, leaving the base case implicit."
3. **How does RandomizedSearchCV differ from GridSearchCV?** — "Grid tries every combo (slow); Random samples n_iter combos — statistically almost as good, much faster."
4. **What happens if you send features in wrong order to predict?** — "Silently wrong outputs — no error. That's why I persisted `feature_columns.pkl` and reindex every request to it."
5. **How would the model degrade in production?** — "Data drift: new model-years and inflation shift price distributions over time; monitor residuals vs real sale prices, retrain periodically."
6. **What is MAE vs RMSE?** — "MAE = average absolute miss (robust); RMSE squares first, so big misses hurt more. I report both."
7. **Why drop `Car_Name` — could names matter (BMW vs Maruti)?** — "They do, but as spelled it's free text with ~100 unique values on 301 rows — it overfits. A real system would map to brand first."
8. **How secure/robust is the Flask app?** — "Inputs are type-coerced and unknown categories degrade to base classes; for real production add validation, a WSGI server (gunicorn), and rate limiting."
9. **What is `min_samples_leaf`?** — "Minimum rows allowed at a tree's final answer node — raising it smooths predictions and fights overfitting."
10. **Next improvements?** — "More data, brand extraction, log-transform target for the luxury tail, and cross-validated ensembles."

## What to show in the video

1. EDA charts (price vs showroom price trend; countplots)
2. The correlation heatmap after encoding — point at `Present_Price` = +0.88
3. Model table: baseline vs tuned vs FINAL line (R² = 0.96)
4. **Live demo** — the Flask form in a browser: fill a car → get ₹ price
5. One JSON API call in terminal/Postman
6. Show the three saved files (`car_price_model.pkl`, `feature_columns.pkl`, `model_info.json`)
