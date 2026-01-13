
"""
Forecasting & Reporting Integration Script
Author: Troy Alexander
Purpose: Combine historical data, calculate Contact Rate, forecast Membership and Call Volume,
validate models, and build a Streamlit UI for sensitivity analysis.
"""

# -----------------------------
# 1. Import Required Libraries
# -----------------------------
import pandas as pd
import numpy as np
import os
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import streamlit as st

# -----------------------------
# 2. Define File Paths
# -----------------------------
input_folder = "project/data"  # Source files stored in project/data
output_file = "project/output/forecast_results_2026.csv"  # Suggested output file
folder_path = os.path.dirname(output_file)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# -----------------------------
# 3. Load and Combine CSV Files
# -----------------------------

def load_and_combine_csv(folder_path):
    """Reads all CSV files from the folder and combines them into a single DataFrame."""
    try:
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not all_files:
            raise FileNotFoundError("No CSV files found in the specified folder.")
        df_list = [pd.read_csv(file) for file in all_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        return combined_df
    except Exception as e:
        print(f"Error loading files: {e}")
        return pd.DataFrame()

data = load_and_combine_csv(input_folder)

# -----------------------------
# 4. Data Cleaning & Validation
# -----------------------------
if data.empty:
    raise ValueError("DataFrame is empty. Check input folder path or file format.")

print("Data Info:")
print(data.info())
print("Missing Values:")
print(data.isnull().sum())

# Fill missing values if any
data.fillna(method='ffill', inplace=True)

# -----------------------------
# 5. Calculate Contact Rate
# -----------------------------
def Contact_Rate(data):
    """Calculate Contact Rate as Calls Received / Membership Count."""
    data['Contact_Rate'] = data['Calls_Received'] / data['Membership_Count']
    return data

# -----------------------------
# 6. Forecast Membership for 2027
# -----------------------------
def forecast_membership(data, method="prophet"):
    """Forecast membership using Prophet or ARIMA."""
    if method == "prophet":
        membership_df = data[['Date', 'Membership_Count']].rename(columns={'Date': 'ds', 'Membership_Count': 'y'})
        model = Prophet()
        model.fit(membership_df)
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    elif method == "arima":
        ts = data.set_index('Date')['Membership_Count']
        model = ARIMA(ts, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = fitted_model.get_forecast(steps=12)
        forecast_df = pd.DataFrame({
            'ds': pd.date_range(start=ts.index[-1], periods=12, freq='M'),
            'yhat': forecast.predicted_mean,
            'yhat_lower': forecast.conf_int().iloc[:, 0],
            'yhat_upper': forecast.conf_int().iloc[:, 1]
        })
        return forecast_df
    else:
        raise ValueError("Invalid method. Choose 'prophet' or 'arima'.")

forecasted_membership = forecast_membership(data, method="prophet")

# -----------------------------
# 7. Forecast Call Volume for 2026
# -----------------------------
# # Calculate the mean Contact Rate
def calculate_contact_rate(data):
    data['Contact_Rate'] = data['Historical_Call_Volume'] / data['Membership_Count']
    return data

data = calculate_contact_rate(data)
avg_contact_rate = data['Contact_Rate'].mean()
forecasted_membership['Forecasted_Call_Volume'] = forecasted_membership['yhat'] * avg_contact_rate

# -----------------------------
# 8. Validate Forecasts
# -----------------------------
actual = data['Membership_Count'][-12:]
predicted = forecasted_membership['yhat'][:12]
mape = mean_absolute_percentage_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f"Validation Metrics -> MAPE: {mape:.2f}, RMSE: {rmse:.2f}")

# -----------------------------
# 9. Save Output
# -----------------------------
forecasted_membership.to_csv(output_file, index=False)
print(f"Forecast saved to {output_file}")

# -----------------------------
# 10. Streamlit UI for Sensitivity Analysis
# -----------------------------
def run_ui():
    st.title("Forecast Sensitivity Analysis")

    # Sliders for key parameters
    membership = st.slider("Membership Count", 50000, 200000, 100000)
    aht = st.slider("Average Handle Time (AHT) in seconds", 300, 900, 600)
    shrinkage = st.slider("Shrinkage (%)", 10, 40, 25)
    occupancy = st.slider("Occupancy (%)", 70, 95, 85)

    # Calculate adjusted volume
    adjusted_volume = membership * avg_contact_rate
    st.write(f"Adjusted Call Volume: {adjusted_volume:,.0f}")

    # Staffing calculation
    agents_needed = adjusted_volume * aht / (occupancy / 100) / (1 - shrinkage / 100) / (8 * 3600)
    st.write(f"Agents Needed: {round(agents_needed)}")

    # Visualization
    st.subheader("Forecasted Membership Trend")
    st.line_chart(forecasted_membership[['ds', 'yhat']].set_index('ds'))

    st.subheader("Forecasted Call Volume Trend")
    st.line_chart(forecasted_membership[['ds', 'Forecasted_Call_Volume']].set_index('ds'))

    # Download button
    st.download_button("Download Forecast CSV", data=forecasted_membership.to_csv(index=False), file_name="forecast_results_2026.csv")

if __name__ == "__main__":
    # Uncomment below to run Streamlit UI
    # run_ui()
    pass