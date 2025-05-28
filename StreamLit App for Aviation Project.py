import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from windrose import WindroseAxes

# -------------------------------
# 📌 Streamlit App Title & File Upload
# -------------------------------
st.title("🌤️ Weather Data Analysis & Forecasting")
st.sidebar.header("📂 Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ✅ Set Global Seaborn Style
    sns.set_theme(style="whitegrid", palette="pastel")

    # -------------------------------
    # 1️⃣ DATA PREPROCESSING
    # -------------------------------
    st.subheader("🔍 Data Preprocessing")

    # ✅ Remove duplicate columns
    # df = df.loc[:, ~df.columns.duplicated()]

    # ✅ Drop unnamed columns
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors="ignore")

    # ✅ Detect & rename timestamp column
    time_columns = ["time", "timestamp", "date_time", "datetime", "Time"]
    found_time_col = next((col for col in time_columns if col in df.columns), None)

    if found_time_col:
        df.rename(columns={found_time_col: "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        st.error("❌ No valid time column found! Please check the dataset.")

    df = df.dropna(subset=["time"])  # Drop rows with missing timestamps

    # ✅ Extract Date-Time Features
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour

    # ✅ Convert Temperature from Kelvin to Celsius
    if "Temperature" in df.columns:
        df["Temperature"] = df["Temperature"] - 273.15

    # ✅ Define Seasons Based on Month
    season_mapping = {12: "Winter", 1: "Winter", 2: "Winter",
                      3: "Spring", 4: "Spring", 5: "Spring",
                      6: "Summer", 7: "Summer", 8: "Summer",
                      9: "Autumn", 10: "Autumn", 11: "Autumn"}
    df["season"] = df["month"].map(season_mapping)

    # ✅ Select Numerical Columns & Convert to Numeric
    numerical_cols = ["Temperature", "Relative Humidity", "Wind Speed", "Wind Gust", "Sea Level Pressure", "Visibility"]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert text values to NaN

    # ✅ Fill Missing Values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    # ✅ Create Cleaned Data Copy for Outlier Removal
    df_cleaned = df.copy()

    st.write("✅ Processed Data Sample:")
    st.dataframe(df.head())

    # -------------------------------
    # 2️⃣ ALL PLOTS (MATCHING JUPYTER OUTPUT)
    # -------------------------------

    ## ✅ SCATTER PLOTS
    st.subheader("📊 Scatter Plots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(df["Temperature"], df["Relative Humidity"], alpha=0.5, color="blue")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Relative Humidity (%)")
    axes[0].set_title("Temperature vs. Relative Humidity")
    axes[1].scatter(df["Wind Speed"], df["Sea Level Pressure"], alpha=0.5, color="green")
    axes[1].set_xlabel("Wind Speed (m/s)")
    axes[1].set_ylabel("Sea Level Pressure (hPa)")
    axes[1].set_title("Wind Speed vs. Sea Level Pressure")
    st.pyplot(fig)

    ## ✅ HISTOGRAMS
    st.subheader("📊 Histograms")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i // 3, i % 3])
        axes[i // 3, i % 3].set_title(f"Distribution of {col}")
    plt.tight_layout()
    st.pyplot(fig)


    # -------------------------------
    # 2️⃣ OUTLIER DETECTION & REMOVAL
    # -------------------------------
    st.subheader("📊 Box Plot Before Outlier Removal")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="month", y="Wind Speed", data=df, palette="Set3", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_title("Wind Speed Distribution by Month (Before Outlier Removal)")
    st.pyplot(fig)


    # ✅ Remove Outliers Using IQR
    def remove_outliers_iqr(df, column):
        if pd.api.types.is_numeric_dtype(df[column]):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
        return df


    for col in ["Wind Speed", "Temperature"]:
        df_cleaned = remove_outliers_iqr(df_cleaned, col)

    # ✅ Box Plot After Outlier Removal
    st.subheader("📊 Box Plot After Outlier Removal")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="month", y="Wind Speed", data=df_cleaned, palette="Set3", ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_title("Wind Speed Distribution by Month (After Outlier Removal)")
    st.pyplot(fig)

    ## ✅ TIME SERIES PLOT
    st.subheader("📊 Time Series Plot of Weather Data")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, col in enumerate(numerical_cols):
        sns.lineplot(x=df_cleaned.index, y=df_cleaned[col], ax=axes[i // 2, i % 2], color="blue", alpha=0.6)
        axes[i // 2, i % 2].set_title(f"Time Series of {col}")
    plt.tight_layout()
    st.pyplot(fig)

    # -------------------------------
    # 📊 SEASON-WISE CORRELATION: TEMPERATURE & HUMIDITY
    # -------------------------------
    st.subheader("📊 Season-Wise Correlation: Temperature & Humidity")

    # ✅ Ensure 'season' is a categorical variable in the correct order
    df_cleaned["season"] = pd.Categorical(df_cleaned["season"], categories=["Winter", "Spring", "Summer", "Autumn"],
                                          ordered=True)

    # ✅ Calculate correlation between Temperature & Relative Humidity for each season
    seasonal_corr = df_cleaned.groupby("season")[["Temperature", "Relative Humidity"]].corr().iloc[0::2, -1].droplevel(
        1)

    # ✅ Plot the correlation values
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=seasonal_corr.index, y=seasonal_corr.values, palette="coolwarm", ax=ax)

    # ✅ Labels & Title
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Correlation Coefficient", fontsize=12)
    ax.set_title("Seasonal Correlation: Temperature vs. Humidity", fontsize=14, fontweight='bold')
    st.pyplot(fig)

    # -------------------------------
    # 3️⃣ ARIMA FORECASTING (SEPARATE PLOT)
    # -------------------------------
    st.subheader("🔮 ARIMA Forecasting")

    df_cleaned.index = pd.to_datetime(df_cleaned["time"])
    temp_clean = df_cleaned["Temperature"]

    # ✅ ARIMA Model
    arima_model = ARIMA(temp_clean, order=(5, 1, 0)).fit()

    # ✅ Extended Forecast for Future Prediction
    forecast_steps = 48  # Increased from 24 to 48 for longer predictions
    forecast_arima = arima_model.forecast(steps=forecast_steps)

    # ✅ Create Forecast Index
    forecast_index = pd.date_range(start=df_cleaned.index[-1], periods=forecast_steps + 1, freq="H")[1:]

    # ✅ Display ARIMA Forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_cleaned.index, temp_clean, label="Actual Temperature", color="blue")
    ax.plot(forecast_index, forecast_arima, label="ARIMA Forecast", linestyle="dashed", color="orange", marker="o")
    ax.set_title("Temperature Forecast Using ARIMA")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # 4️⃣ EXPONENTIAL SMOOTHING FORECASTING (SEPARATE PLOT)
    # -------------------------------
    st.subheader("🔮 Exponential Smoothing Forecasting")

    # ✅ Exponential Smoothing Model
    exp_smooth = ExponentialSmoothing(temp_clean, trend="add", seasonal="add", seasonal_periods=24).fit()
    forecast_exp = exp_smooth.forecast(steps=forecast_steps)

    # ✅ Display Exponential Smoothing Forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_cleaned.index, temp_clean, label="Actual Temperature", color="blue")
    ax.plot(forecast_index, forecast_exp, label="Exponential Smoothing Forecast", linestyle="dashed", color="red",
            marker="o")
    ax.set_title("Temperature Forecast Using Exponential Smoothing")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

st.sidebar.write("🚀 Click 'Run' to Start the Application!")
