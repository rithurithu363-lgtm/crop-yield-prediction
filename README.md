# Crop Yield Prediction

This project predicts crop yield using machine learning.  
Input features include rainfall, temperature, fertilizer usage, soil pH, and area.

## Dataset

Place your dataset file in the project folder with the name:

`crop_yield.csv`

Example columns (you can change in the code):

- `Rainfall`
- `Temperature`
- `Fertilizer`
- `Soil_pH`
- `Area`
- `Yield` (target)

Update the `FEATURE_COLS` and `TARGET_COL` in `crop_yield_prediction.py`
to match your actual column names.

## Model

- Train/test split (80/20)
- StandardScaler for feature scaling
- Random Forest Regressor for prediction
- Metrics: MAE, MSE, RMSE, R²

## How to Run

```bash
pip install -r requirements.txt
python crop_yield_prediction.py
