import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

from src.preprocess import load_data, clean_data
from src.feature_engineering import prepare_features

from src.config import (
    FEATURES,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    RF_MODEL_PATH,
    LR_MODEL_PATH
)
def evaluate_model(model_name, y_true, y_pred):
    """
    Evaluate regression model performance.
    """

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : £{mae:,.2f}")
    print(f"RMSE     : £{rmse:,.2f}")

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }
def main():

    print("=" * 60)
    print("USED CAR PRICE PREDICTION")
    print("=" * 60)

    print("\nLoading dataset...")

    df = load_data()

    print("Cleaning dataset...")

    df = clean_data(df)

    print("Creating features...")

    df = prepare_features(df)
    X = df[FEATURES]

    y = df[TARGET]

    print("\nFeature Matrix Shape :", X.shape)
    print("Target Shape         :", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE

    )

    print("\nTrain Size :", len(X_train))
    print("Test Size  :", len(X_test))

    print("\nTraining Linear Regression...")

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    lr_predictions = lr.predict(X_test)

    lr_results = evaluate_model(

        "Linear Regression",

        y_test,

        lr_predictions

    )

    print("\nTraining Random Forest...")

    rf = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

    rf.fit(X_train, y_train)

    rf_predictions = rf.predict(X_test)

    rf_results = evaluate_model(

        "Random Forest",

        y_test,

        rf_predictions

    )

    os.makedirs("models", exist_ok=True)

    joblib.dump(

        lr,

        LR_MODEL_PATH

    )

    joblib.dump(

        rf,

        RF_MODEL_PATH

    )

    print("\nModels saved successfully.")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"""
Linear Regression

R²   : {lr_results["R2"]:.4f}

MAE  : £{lr_results["MAE"]:,.2f}

RMSE : £{lr_results["RMSE"]:,.2f}

Random Forest

R²   : {rf_results["R2"]:.4f}

MAE  : £{rf_results["MAE"]:,.2f}

RMSE : £{rf_results["RMSE"]:,.2f}
""")

if __name__ == "__main__":
    main()