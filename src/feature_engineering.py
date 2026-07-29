import pandas as pd

from src.config import CURRENT_YEAR
def create_features(df):
    """
    Create engineered features used for model training.
    """

   
    df["car_age"] = CURRENT_YEAR - df["year"]

    # Avoid division by zero
    df["mileage_per_year"] = df["mileage"] / df["car_age"].replace(0, 1)

    return df


def encode_features(df):
    """
    One-hot encode categorical variables.
    """

    categorical_columns = [
        "transmission",
        "fuelType",
        "brand"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    return df


def prepare_features(df):
    """
    Complete feature engineering pipeline.
    """

    df = create_features(df)

    df = encode_features(df)

    return df