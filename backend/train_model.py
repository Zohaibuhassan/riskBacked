import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Drop index column if present
df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)

# Rename columns to match frontend keys
df.rename(columns={
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

# Encode categorical columns
categorical_cols = ["sex", "housing", "saving_accounts", "checking_account", "purpose", "risk"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Split data
X = df.drop("risk", axis=1)
y = df["risk"]

# Train model on DataFrame to preserve feature names
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Save model and encoders
with open("loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Optional: Save feature names separately for SHAP alignment
with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Model and encoders saved successfully.")
