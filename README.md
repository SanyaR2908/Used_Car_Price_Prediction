# Used Car Price Prediction

A Machine Learning web application that predicts the resale price of Volkswagen and Audi used cars using a Random Forest Regressor.

---

## Project Overview

This project estimates the market value of used cars based on their specifications such as:

- Brand
- Manufacturing Year
- Mileage
- Engine Size
- Transmission
- Fuel Type
- Road Tax
- Miles Per Gallon (MPG)

The application is built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## Dataset

The dataset consists of **25,458** used car listings from:

- Volkswagen
- Audi

The data was cleaned, preprocessed, and combined before training the machine learning models.

---

## Machine Learning Models

Two regression models were trained:

- Linear Regression
- Random Forest Regressor

Random Forest produced the best performance and was selected for deployment.

---

## Model Performance

| Metric   | Value  |
| -------- | ------ |
| R² Score | 94.17% |
| MAE      | £1,495 |
| RMSE     | £2,387 |

---

## Project Structure

```
Used_Car_Price_Prediction/

│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── data/
│ ├── audi.csv
│ └── vw.csv
│
├── models/
│ ├── linear_regression.pkl
│ └── random_forest.pkl
│
├── outputs/
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── preprocess.py
    ├── feature_engineering.py
    ├── train.py
    ├── predict.py
    └── utils.py
```

---

## Features

- Data preprocessing
- Feature engineering
- Model training
- Model comparison
- Real-time price prediction
- Interactive Streamlit interface
- Clean modular project structure

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

---

## Installation

Clone the repository

```bash
git clone <repository-link>
```

Move into the project folder

```bash
cd Used_Car_Price_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Application Preview

### Home Screen

![Home Screen](screenshots/home.png)

---

### Prediction Result

![Prediction Result](screenshots/prediction.png)

---

### Prediction Result

![Prediction Result](screenshots/prediction2.png)

---

### Model Performance

![Model Information](screenshots/model_info.png)

## Future Improvements

- Support additional car brands
- Hyperparameter tuning
- Model explainability using SHAP
- Cloud deployment
- Improved UI/UX

---

## Author

Sanya Ray

B.Tech Electronics and Communication Engineering

Machine Learning | Data Science | Python
