# 🎓 Task 1 — How to Explain Your Project (Boston House Price Prediction)

## The 60-second story (memorize this)

> "The task was to predict Boston house prices from features like rooms and crime rate.
> First I checked data health — no missing values. Then I did EDA: correlation showed
> `rm` (rooms, +0.70) and `lstat` (poverty %, −0.74) drive prices most. For preprocessing
> I applied the **IQR outlier technique from the ShadowFox material** and also dropped the
> survey-capped rows (prices were clipped at $50k in 1978). I compared four regressors —
> Linear, Decision Tree, Random Forest, Gradient Boosting — **Gradient Boosting won with
> R² = 0.875**, RMSE ≈ 2.24 ($1000s). I tuned it with RandomizedSearchCV, honestly kept the
> better of tuned-vs-baseline, checked feature importance — `lstat` and `rm` on top, matching
> the EDA — and saved the model with joblib for deployment."

## The pipeline in one picture

```
boston_housing.csv
   → health check (missing values? dtypes?)
   → EDA (histogram, heatmap, scatters)
   → IQR outliers + capped rows removed  (506 → 466 rows)
   → X/y split → train/test 80/20 (random_state=42)
   → 4 models fit → compare MSE/RMSE/R²
   → RandomizedSearchCV on the champion
   → feature importances + actual-vs-predicted plot
   → joblib.dump → boston_price_model.pkl
```

## Concept cheat-sheet (what every step *means*)

| Concept | Say it like this |
|---|---|
| **Regression** (vs classification) | "We predict a *number* (price), not a category — so regression, not classification." |
| **Features / target** | "X = the 13 describing columns, y = the price we want to predict (`medv`)." |
| **EDA** | "Plotting before modeling to see which features matter and whether data is sane." |
| **Correlation** | "How strongly two columns move together; −1 perfect opposite, +1 perfect together. `rm` ≈ +0.70, `lstat` ≈ −0.74 with price." |
| **IQR outlier rule** | "IQR = Q3−Q1. Points below Q1−1.5·IQR or above Q3+1.5·IQR are outliers and get excluded. Our limits for price were [$5.06k, $36.96k] → removed 24 outliers + 16 capped rows." |
| **Why drop medv=50 rows** | "The 1978 survey clipped prices at $50k — those are censored measurements, not real prices; keeping them teaches the model to under-predict at the top end." |
| **Train/test split** | "Learn on 80%, *prove* on hidden 20%. Testing on training data is cheating — the model memorized it." |
| **random_state=42** | "Fixes the shuffle seed so results are reproducible every run." |
| **Linear Regression** | "One best straight hyper-plane through the data — simple, fails on curved patterns (we saw `lstat` is curved)." |
| **Decision Tree** | "A flowchart of yes/no questions ('rooms > 6?'). Flexible but can overfit alone." |
| **Random Forest** | "Hundreds of trees trained on random subsets, voting — reduces overfitting (wisdom of crowds)." |
| **Gradient Boosting** | "Trees added one after another, each fixing the errors of the previous — usually the strongest on tabular data, and it won here." |
| **MSE / RMSE** | "Average squared error; RMSE takes the square root back so it's in the same unit as price. Ours: ≈ 2.24 → typical miss of $2,240." |
| **R²** | "Fraction of price variation explained. 0 = useless, 1 = perfect. 0.875 means ~88% of what makes prices differ is captured." |
| **RandomizedSearchCV** | "Instead of hand-picking hyperparameters, it randomly tries 40 combos with 5-fold cross-validation on the training set and keeps the best." |
| **5-fold CV** | "Split training data into 5 parts; rotate which part validates; average — hyperparameters are chosen fairly, without touching the test set." |
| **Feature importance** | "Tree models report which columns they split on most → lstat & rm dominated, which cross-validates our EDA." |
| **joblib / .pkl** | "Pickles the trained model object to disk so an app can load it without retraining." |

## The honest-tuning talking point ( 💡 reviewers love this)

If asked *"did tuning help?"* — the truth is: **no, slightly not** (baseline R² 0.8746 vs tuned 0.8680
on this small dataset). The notebook compares both on the same hidden test set and keeps the winner.
Saying *"tuning is not magic; I measured both and kept the honest winner"* signals real engineering maturity.

## Likely mentor questions → your answers

1. **Why Gradient Boosting over Linear Regression?** — "R² 0.88 vs 0.75; the relationships in the data are non-linear (see the curved `lstat` scatter), and boosting captures that."
2. **How did you handle missing values?** — "There were none — verified with `isnull().sum()`; if there were, I'd impute median (numeric) or mode (categorical)."
3. **What is overfitting and how do you fight it?** — "When a model memorizes training data and fails on new data. Fought via train/test split, cross-validation, and ensemble models instead of one deep tree."
4. **Why RMSE and not only MSE?** — "RMSE is in the same unit as the target ($1000s), so it's interpretable: typical error ≈ $2.2k."
5. **Can R² be negative?** — "Yes — worse than just predicting the mean for everything."
6. **Why is the dataset controversial?** — "The `b` column encodes a racial assumption from the 1970s; sklearn actually removed this dataset for fairness reasons. In production I'd audit sensitive features."
7. **How would you deploy this?** — "Exactly like my Task 2: save with joblib, wrap in Flask, accept JSON, return predictions."
8. **What's inside feature_importances_?** — "How much each split using that feature reduced impurity, averaged across trees, normalized to sum 1."
9. **Does scaling matter for these models?** — "Not for tree ensembles; Linear Regression coefficients absorb raw scale. I verified trees dominate, so scaling was unnecessary."
10. **What would you do next?** — "Cross-validate more thoroughly, add polynomial features for linear baseline, and test XGBoost/LightGBM."

## What to show in the video (screen plan)

1. Scroll: notebook sections 2→5 (health check, heatmap, IQR math cell output)
2. Section 7: the 4-model comparison table — point at the winning row
3. Section 10: feature importance bar chart
4. Section 11: actual-vs-predicted plot — "points hug the red line"
5. Section 12: the saved `boston_price_model.pkl` + reload check
