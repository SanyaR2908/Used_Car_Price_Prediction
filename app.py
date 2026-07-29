import streamlit as st
from src.predict import predict_price

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Used Car Price Estimator",
    layout="wide"
)

# Custom CSS for spacing and polish
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
        stMetric {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Header Section
# -------------------------
st.title("Used Car Price Prediction")

st.markdown("""
Predict the resale value of Volkswagen and Audi used cars
using a Random Forest machine learning model trained on
25,458 historical vehicle listings.
"""
)
st.divider()

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Configuration Panel")
st.sidebar.markdown("Provide the vehicle details below:")

with st.sidebar.form("prediction_form"):
    brand = st.selectbox("Brand", ["VW", "Audi"])
    year = st.slider(
    "Manufacturing Year",
    min_value=2000,
    max_value=2020,
    value=2018,
    help="Note: Current model is trained on verified historical market data up to 2020."
)
    mileage = st.number_input("Mileage (miles)", min_value=0, value=30000, step=1000)
    engine_size = st.selectbox("Engine Size (L)", [1.0, 1.2, 1.4, 1.5, 1.6, 2.0, 2.5, 3.0])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Other"])
    tax = st.number_input("Road Tax (£)", min_value=0, value=145, step=5)
    mpg = st.number_input("Fuel Economy (MPG)", min_value=0.0, value=54.3, step=0.1)
    
    # Form submit button
    submitted = st.form_submit_button("Predict Resale Price", use_container_width=True)

# -------------------------
# Main Panel / Results
# -------------------------
if submitted:
    car = {
        "brand": brand,
        "year": year,
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size,
        "transmission": transmission,
        "fuelType": fuel_type
    }

    with st.spinner("Analyzing vehicle data and running inference..."):
        price = predict_price(car)

    # Layout for results
    col_res1, col_res2 = st.columns([1.2, 1.8], gap="large")

    with col_res1:
        st.success("Prediction Generated Successfully!")
        st.markdown("### Estimated Market Value")
        st.metric(
            label="Predicted Price",
            value=f"£{price:,.2f}",
            delta="Estimated Resale"
        )

    with col_res2:
        st.markdown("Vehicle Specifications")
        summary_data = {
            "Feature": ["Brand", "Year", "Mileage", "Engine Size", "Transmission", "Fuel Type", "Tax", "MPG"],
            "Value": [
                brand,
                str(year),
                f"{mileage:,} mi",
                f"{engine_size} L",
                transmission,
                fuel_type,
                f"£{tax}",
                f"{mpg} MPG"
            ]
        }
        st.dataframe(summary_data, use_container_width=True, hide_index=True)

else:
    st.info("Fill out the vehicle details in the sidebar and click **'Predict Resale Price'** to see results.")

st.divider()

# -------------------------
# Model Performance Section
# -------------------------
st.subheader("Model Performance & Overview")

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric(label="Accuracy ($R^2$ Score)", value="94.17%", help="Coefficient of determination on test data")
with m_col2:
    st.metric(label="Mean Absolute Error (MAE)", value="£1,495", help="Average absolute deviation from actual price")
with m_col3:
    st.metric(label="Root Mean Squared Error (RMSE)", value="£2,387", help="Penalizes larger prediction errors more heavily")

st.markdown("")
st.caption(
    " **Dataset Reference:** Trained on 25,458 verified Volkswagen and Audi listings. "
    "**Architecture:** Random Forest Regressor optimized via Scikit-Learn."
)