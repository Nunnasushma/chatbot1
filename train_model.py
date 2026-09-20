import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure UTF-8 output encoding for Windows terminal output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=" * 70)
    print("🚗 CAR PRICE PREDICTOR - MODEL TRAINING")
    print("=" * 70)

    # 1. Load CSV dataset
    dataset_path = os.path.join("dataset", "car_data.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # 2. Inspect dataset
    print("\n--- DATASET INSPECTION ---")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns in CSV: {list(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    # 3. Clean Data
    if df.duplicated().sum() > 0:
        dup_count = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {dup_count} duplicate rows. New shape: {df.shape}")

    # Define Feature and Target Columns based on actual CSV columns
    target_col = 'selling_price'
    num_features = ['year', 'km_driven']
    cat_features = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
    feature_cols = num_features + cat_features

    # Verify column presence
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in CSV! Available columns: {list(df.columns)}")

    # Clean rows missing feature or target values
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    print(f"Cleaned Dataset Shape: {df_clean.shape}")

    # 4. Separate Features (X) and Target (y)
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print("\n--- FEATURE SELECTION ---")
    print(f"Numerical Features: {num_features}")
    print(f"Categorical Features: {cat_features}")
    print(f"Target Column: {target_col}")

    # 5. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size:  {len(X_test)} samples")

    # 6. Build Preprocessing Pipeline
    # Using OneHotEncoder(handle_unknown='ignore') to safely handle unseen car models in test/app
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    # 7. Define Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=200, random_state=42)
    }

    print("\n" + "=" * 70)
    print("MODEL TRAINING & COMPARISON")
    print("=" * 70)

    best_model_name = None
    best_r2 = -float('inf')
    best_metrics = {}
    best_pipeline = None

    results = {}

    for name, model in models.items():
        # Create Scikit-learn Pipeline combining preprocessor and regressor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Fit on training data to prevent data leakage
        pipeline.fit(X_train, y_train)

        # Predict on testing set
        y_pred = pipeline.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        print(f"\n📊 {name}:")
        print(f"   - MAE:  ₹{mae:,.2f}")
        print(f"   - RMSE: ₹{rmse:,.2f}")
        print(f"   - R² Score: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            best_pipeline = pipeline
    print("\n" + "=" * 70)
    print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
    print(f"   - MAE:  ₹{best_metrics['MAE']:,.2f}")
    print(f"   - RMSE: ₹{best_metrics['RMSE']:,.2f}")
    print(f"   - R² Score: {best_metrics['R2']:.4f}")
    print("=" * 70)
    # 8. Save Best Pipeline
    os.makedirs("model", exist_ok=True)
    model_save_path = os.path.join("model", "car_price_model.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"\n✅ Complete pipeline successfully saved to: {model_save_path}")
if __name__ == "__main__":
    main()
