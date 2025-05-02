import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv("data/Train.csv")
data.dropna(subset=["Item_Outlet_Sales"], inplace=True)

# Features & Target
X = data.drop(columns=["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"])
y = data["Item_Outlet_Sales"]

# Feature types
numerical = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical = X.select_dtypes(include=["object"]).columns.tolist()

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical),
    ("cat", cat_pipeline, categorical)
])

# Final pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/bigmart_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model saved to model/bigmart_model.pkl")
