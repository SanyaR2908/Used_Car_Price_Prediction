"""
Configuration file for the Used Car Price Prediction project.
All project constants are defined here.
"""

from datetime import datetime

# ==============================
# Project Settings
# ==============================

# Automatically uses the current year
CURRENT_YEAR = datetime.now().year
MIN_YEAR = 2000
MAX_YEAR = 2020

# Reproducibility
RANDOM_STATE = 42

# Train-Test Split
TEST_SIZE = 0.20

# Random Forest Parameters
N_ESTIMATORS = 100

# ==============================
# Dataset Columns
# ==============================

FEATURES = [
    "mileage",
    "tax",
    "mpg",
    "engineSize",
    "car_age",
    "mileage_per_year",
    "transmission_Manual",
    "transmission_Semi-Auto",
    "fuelType_Hybrid",
    "fuelType_Other",
    "fuelType_Petrol",
    "brand_VW"
]

TARGET = "price"

# ==============================
# Model Save Paths
# ==============================

RF_MODEL_PATH = "models/random_forest.pkl"
LR_MODEL_PATH = "models/linear_regression.pkl"

# ==============================
# Dataset Paths
# ==============================

VW_DATA_PATH = "data/vw.csv"
AUDI_DATA_PATH = "data/audi.csv"