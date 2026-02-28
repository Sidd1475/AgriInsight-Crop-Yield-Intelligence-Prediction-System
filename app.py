import streamlit as st 
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import numpy as np 

from src.insights import (
    get_region_summary , 
    get_top_crops_by_yield, 
    yearly_trend, 
    features_importance_insights
)

from src.predict import predict_yield

## Load the Data 
@st.cache_data
def load_data():
    return pd.read_csv("./data/preprocessed/cleaned_agridata.csv")

@st.cache_resource 
def load_model():
    return joblib.load("./models/final_model.pkl")

df = load_data()
model = load_model()

st.set_page_config(page_title="Crop Yield Intelligence Tool", layout="wide")

st.title("Crop Yield Prediction & Regional Intelligence Dashboard")

## Sidebar Navigation
menu = st.sidebar.selectbox(
    "Select Option", 
    ["Yield Prediction", "District Insights", "Model Insights"]
)

## Prediction 
if menu == "Yield Prediction":
    st.header("Predict Crop Yield")

    col1,col2 = st.columns(2)

    with col1:
        state = st.selectbox("State", sorted(df["State"].unique()))
        district = st.selectbox(
            "District",
            sorted(df[df["State"]==state]["District"].unique())
        )
        crop = st.selectbox("Crop",sorted(df["Crop"].unique()))

    with col2:
        season = st.selectbox("Season",sorted(df["Season"].unique()))
        area = st.number_input("Area(in hectares)",min_value = 5)
        year = st.number_input("Year", min_value=2010, max_value=2035)

    if st.button("Predict Yield"):

        input_data = {
            "District": district,
            "State": state,
            "Crop": crop,
            "Season": season,
            "Area": area,
            "Year": year
        }

        predicted_yield = predict_yield(input_data)

        st.success(f"Predicted Yield: {round(predicted_yield, 2)} tonnes per hectare")

## District insights 
elif menu=="District Insights":
    st.header("Regional Intelligence Dashboard")

    state = st.selectbox("Select State", sorted(df["State"].unique()))
    district = st.selectbox(
        "Select District", 
        sorted(df[df["State"]==state]["District"].unique())
    )

    summary = get_region_summary(df, district)
    crops_info = get_top_crops_by_yield(df, district)
    trend_info = yearly_trend(df, district)

    if summary:

        col1, col2, col3 = st.columns(3)

        col1.metric("Avg Yield", summary["avg_yield"])
        col2.metric("Total Production", summary["total_production"])
        col3.metric("Yield Growth %", summary["Yield_growth_percentage"])
        
        st.subheader("Yield Trend Over Years")
        st.line_chart(trend_info["yearly_yield"].loc[1998:2023])

        st.subheader("Top Performing Crops")
        st.dataframe(crops_info["top_5_crops"])

        st.subheader("Least Performing Crops")
        st.dataframe(crops_info["bottom_5_crops"])

        st.subheader(" Additional Info")
        st.write(f"Peak Year: {trend_info['peak_year']}")
        st.write(f"Lowest Year: {trend_info['lowest_year']}")
        st.write(f"Volatility: {trend_info['volatility']}")

    else:
        st.warning("No data available for selected district.")


elif menu == "Model Insights":
    st.header("Model Feature Importance")
    importance = features_importance_insights(model)
    st.subheader("Top 10 Most Important Features")
    st.dataframe(importance["top_features"])
    st.subheader("Model Insights Summary")

    top_features = importance["top_features"]

    # Get top 3 features
    top1 = top_features.iloc[0]["Feature"]
    top2 = top_features.iloc[1]["Feature"]
    top3 = top_features.iloc[2]["Feature"]

    # Clean feature names (remove pipeline prefixes)
    def clean_feature(name):
        return name.split("__")[-1]

    top1_clean = clean_feature(top1)
    top2_clean = clean_feature(top2)
    top3_clean = clean_feature(top3)

    st.write(f"🔎 The model is primarily influenced by **{top1_clean}**, indicating that this factor plays the most significant role in determining crop yield.")

    st.write(f"📊 The second most important driver is **{top2_clean}**, suggesting that regional or crop-level characteristics strongly impact productivity.")

    st.write(f"🌾 Additionally, **{top3_clean}** contributes meaningfully to predictions, reinforcing the importance of agricultural conditions and local variations.")

    st.write("📌 Overall, the model relies more heavily on crop type and regional attributes compared to seasonal or year-based factors, highlighting the dominance of structural agricultural patterns in yield prediction.")