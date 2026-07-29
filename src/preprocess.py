import pandas as pd


def load_data():
    """
    Load Volkswagen and Audi datasets.
    """

    # Load datasets
    vw = pd.read_csv("data/vw.csv")
    audi = pd.read_csv("data/audi.csv")

    # Add brand column
    vw["brand"] = "VW"
    audi["brand"] = "Audi"

    # Combine datasets
    df = pd.concat([vw, audi], ignore_index=True)

    return df


def clean_data(df):
    """
    Perform basic data cleaning.
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    return df