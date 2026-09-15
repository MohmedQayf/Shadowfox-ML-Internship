"""
Car Selling Price Predictor — Flask deployment
==============================================
Serves two interfaces:

  GET  /            -> HTML form (human users)
  POST /predict     -> form submission -> renders the estimate
  POST /api/predict -> JSON API (programmatic users)

The trained Random Forest and the exact training-time feature order
were saved by the notebook (Car_Price_Prediction.ipynb). The app
rebuilds a single-row DataFrame in that exact column order before
asking the model for a prediction.
"""

import datetime
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "car_price_model.pkl"))
FEATURE_COLUMNS = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
with open(os.path.join(BASE_DIR, "model_info.json")) as f:
    MODEL_INFO = json.load(f)

app = Flask(__name__)


def build_feature_row(present_price, kms_driven, owner, year,
                      fuel_type, seller_type, transmission):
    """Turn raw form/API inputs into a one-row DataFrame that matches
    the exact one-hot-encoded training schema (all missing -> 0)."""
    car_age = datetime.datetime.now().year - int(year)

    row = {col: 0 for col in FEATURE_COLUMNS}          # start every dummy at 0
    row["Present_Price"] = float(present_price)
    row["Kms_Driven"] = int(kms_driven)
    row["Owner"] = int(owner)
    row["Car_Age"] = car_age

    # switch on the matching one-hot columns (drop_first schema -> base class has no column)
    for col in (f"Fuel_Type_{fuel_type}", f"Seller_Type_{seller_type}", f"Transmission_{transmission}"):
        if col in row:
            row[col] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.route("/")
def home():
    return render_template("index.html", info=MODEL_INFO, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    f = request.form
    X = build_feature_row(
        present_price=f["present_price"],
        kms_driven=f["kms_driven"],
        owner=f["owner"],
        year=f["year"],
        fuel_type=f["fuel_type"],
        seller_type=f["seller_type"],
        transmission=f["transmission"],
    )
    price_lakhs = float(model.predict(X)[0])
    return render_template(
        "index.html",
        info=MODEL_INFO,
        prediction=round(price_lakhs, 2),
        inputs=f,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    X = build_feature_row(
        present_price=data["present_price"],
        kms_driven=data["kms_driven"],
        owner=data["owner"],
        year=data["year"],
        fuel_type=data["fuel_type"],
        seller_type=data["seller_type"],
        transmission=data["transmission"],
    )
    price_lakhs = float(model.predict(X)[0])
    return jsonify({
        "predicted_selling_price_lakhs": round(price_lakhs, 2),
        "note": "1 lakh = Rs.100,000",
        "model": MODEL_INFO["model"],
        "r2_test": MODEL_INFO["r2_test"],
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_INFO})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
