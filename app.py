import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgb_model.pkl")

st.title("📊 Customer Lifetime Value Predictor")
st.markdown("Enter basic metrics below. We'll calculate the rest and predict CLV.")

# User inputs
cost = st.number_input("Cost", min_value=0.0001, format="%.4f")
conversion_rate = st.number_input("Conversion Rate", min_value=0.0001, max_value=1.0, format="%.5f")
revenue = st.number_input("Revenue", min_value=0.0, format="%.2f")

# Channel selection
channel = st.selectbox(
    "Select Marketing Channel",
    ["email marketing", "paid advertising", "referral", "social media"]
)

# Predict button
if st.button("Predict CLV"):
    # Derived features
    conversion_efficiency = conversion_rate / cost
    revenue_per_conversion = revenue / (conversion_rate * cost)

    # Cost tier logic
    if cost < 10:
        cost_tier = 0
    elif cost < 30:
        cost_tier = 1
    else:
        cost_tier = 2

    # Revenue segment logic
    if revenue < 1000:
        revenue_segment = 1.0
    elif revenue < 3000:
        revenue_segment = 2.0
    else:
        revenue_segment = 3.0

    # One-hot encode channel
    channels = ["channel_email marketing", "channel_paid advertising", "channel_referral", "channel_social media"]
    channel_data = {ch: 0 for ch in channels}
    channel_data[f"channel_{channel}"] = 1  # Set selected channel to 1

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "customer_id": 0,  # default value
        "cost": cost,
        "conversion_rate": conversion_rate,
        "revenue": revenue,
        "conversion_efficiency": conversion_efficiency,
        "revenue_per_conversion": revenue_per_conversion,
        "cost_tier": cost_tier,
        "revenue_segment": revenue_segment,
        **channel_data
    }])

    # Reorder columns to match model
    column_order = [
        "customer_id",
        "cost",
        "conversion_rate",
        "revenue",
        "cost_tier",
        "conversion_efficiency",
        "revenue_per_conversion",
        "revenue_segment",
        "channel_email marketing",
        "channel_paid advertising",
        "channel_referral",
        "channel_social media"
    ]
    input_data = input_data[column_order]

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted CLV: {prediction:.2f}")
