from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
import shap
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
CORS(app)

# Load model and encoders
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Load and prepare background for SHAP
background_df = pd.read_csv("german_credit_data.csv")
background_df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
background_df.rename(columns={
    "Age": "age",
    "Sex": "sex",
    "Job": "job",
    "Housing": "housing",
    "Saving accounts": "saving_accounts",
    "Checking account": "checking_account",
    "Credit amount": "credit_amount",
    "Duration": "duration",
    "Purpose": "purpose",
    "Risk": "risk"
}, inplace=True)

# Encode background exactly as training
for col, le in encoders.items():
    if col in background_df.columns:
        background_df[col] = le.transform(background_df[col].astype(str))

background_X = background_df.drop("risk", axis=1)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("Received payload:", data)

    input_vals = []
    for col in model.feature_names_in_:
        raw = data.get(col)
        if raw in (None, ""):
            return jsonify({"error": f"Missing value for {col}"}), 400

        # Normalize categorical
        if col in encoders:
            s = str(raw).strip().lower()
            if s in ("na", "none"):
                s = "nan"               # match training's missing
            try:
                code = encoders[col].transform([s])[0]
            except ValueError:
                return jsonify({"error": f"Invalid value for {col}: {raw}"}), 400
            val = code
        else:
            # numeric
            try:
                val = float(raw)
            except ValueError:
                return jsonify({"error": f"Expected numeric for {col}, got '{raw}'"}), 400

        input_vals.append(val)

    try:
        # Make DataFrame for both prediction & SHAP
        X_new = pd.DataFrame([input_vals], columns=model.feature_names_in_)

        # Predict
        pred  = model.predict(X_new)[0]
        prob  = model.predict_proba(X_new)[0][pred]
        label = encoders["risk"].inverse_transform([pred])[0]

        # SHAP
        masker    = shap.maskers.Independent(background_X)
        explainer = shap.LinearExplainer(model, masker=masker)
        sv        = explainer(X_new)

        # Build contributions dict
        contributions = {
            feature: round(val, 4)
            for feature, val in zip(model.feature_names_in_, sv.values[0])
        }

        # Draw bar chart
        plt.figure(figsize=(8, 4))
        shap.plots.bar(sv[0], show=False)
        os.makedirs("static", exist_ok=True)
        path = os.path.join("static", "risk_chart.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        return jsonify({
            "prediction": label,
            "risk_score": round(prob, 4),
            "explanation": contributions,
            "chart_url": "/static/risk_chart.png"
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
