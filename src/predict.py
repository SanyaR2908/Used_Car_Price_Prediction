import os
import joblib
import pandas as pd

from src.config import (
    FEATURES,
    RF_MODEL_PATH,
    CURRENT_YEAR
)

# Check whether the trained model exists
if not os.path.exists(RF_MODEL_PATH):
    raise FileNotFoundError(
        "\nRandom Forest model not found!\n\n"
        "Please train the model first by running:\n\n"
        "python src/train.py\n"
    )

# Load trained Random Forest model
rf_model = joblib.load(RF_MODEL_PATH)


def predict_price(car):
    """
    Predict the resale price of a car.
    """

    # Feature Engineering
    car["car_age"] = CURRENT_YEAR - car["year"]
    car["mileage_per_year"] = car["mileage"] / max(car["car_age"], 1)

    # Transmission Encoding
    car["transmission_Manual"] = (
        1 if car["transmission"] == "Manual" else 0
    )

    car["transmission_Semi-Auto"] = (
        1 if car["transmission"] == "Semi-Auto" else 0
    )

    # Fuel Type Encoding
    car["fuelType_Hybrid"] = (
        1 if car["fuelType"] == "Hybrid" else 0
    )

    car["fuelType_Other"] = (
        1 if car["fuelType"] == "Other" else 0
    )

    car["fuelType_Petrol"] = (
        1 if car["fuelType"] == "Petrol" else 0
    )

    # Brand Encoding
    car["brand_VW"] = (
        1 if car["brand"] == "VW" else 0
    )

    # Create DataFrame
    input_df = pd.DataFrame([car])[FEATURES]

    prediction = rf_model.predict(input_df)

    return prediction[0]


if __name__ == "__main__":

    sample_car = {
        "brand": "VW",
        "year": 2018,
        "mileage": 30000,
        "tax": 145,
        "mpg": 54.3,
        "engineSize": 1.4,
        "transmission": "Manual",
        "fuelType": "Petrol"
    }

    predicted_price = predict_price(sample_car)

    print("=" * 50)
    print("USED CAR PRICE PREDICTION")
    print("=" * 50)
    print(f"Estimated Price: £{predicted_price:,.2f}")