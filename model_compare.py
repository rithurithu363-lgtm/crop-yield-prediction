from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd

df = pd.read_csv("crop_yield.csv")

X = df[["Rainfall", "Temperature", "Fertilizer", "Soil_pH", "Area"]]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "XGBoost": XGBRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = round(r2_score(y_test, y_pred), 3)

results
