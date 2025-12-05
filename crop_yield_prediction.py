"""
Crop Yield Prediction using Machine Learning

This script trains a model to predict crop yield based on
features like rainfall, temperature, fertilizer, soil pH and area.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load dataset
# Put your dataset in the same folder with name: crop_yield.csv
data = pd.read_csv("crop_yield.csv")

# 2. Select features and target
# 👉 Change these column names to match your CSV file
FEATURE_COLS = ["Rainfall", "Temperature", "Fertilizer", "Soil_pH", "Area"]
TARGET_COL = "Yield"

X = data[FEATURE_COLS]
y = data[TARGET_COL]

# 3. Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model (Random Forest)
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("=== Crop Yield Prediction Results ===")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.3f}")

# 7. Save model + scaler
joblib.dump(model, "crop_yield_model.pkl")
joblib.dump(scaler, "crop_yield_scaler.pkl")
print("Saved model and scaler to disk.")
