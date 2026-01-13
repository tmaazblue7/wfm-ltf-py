
"""
forecasting.py
Purpose: Forecast Membership and Call Volume using ARIMA and Annual_Contact_Rate.
"""

import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Paths
input_file = "project/data/"
output_file = "project/output/forecast_results_2026.csv"

# Load data
data = pd.read_csv(input_file)
data['Date'] = pd.to_datetime(data['Date'])

# Fill missing values using obj.ffill() and obj.bfill()
data['Membership_Count'] = data['Membership_Count'].ffill().bfill()
data['Annual_Contact_Rate'] = data['Annual_Contact_Rate'].ffill().bfill()

# Sort and set index
data = data.sort_values('Date').set_index('Date')

# Time series for Membership
ts_membership = data['Membership_Count']

# ARIMA model
model = ARIMA(ts_membership, order=(1, 1, 1))
fitted_model = model.fit()

# Forecast next 12 months (Month-End)
forecast = fitted_model.get_forecast(steps=12)
forecast_df = pd.DataFrame({
    'Date': pd.date_range(start=ts_membership.index[-1] + pd.offsets.MonthEnd(), periods=12, freq='ME'),
    'Forecasted_Membership': forecast.predicted_mean,
    'Lower_CI': forecast.conf_int().iloc[:, 0],
    'Upper_CI': forecast.conf_int().iloc[:, 1]
})

# Use Annual_Contact_Rate for Call Volume
avg_contact_rate = data['Annual_Contact_Rate'].mean()
forecast_df['Forecasted_Call_Volume'] = forecast_df['Forecasted_Membership'] * avg_contact_rate

# Validate
actual = ts_membership[-12:]
predicted = fitted_model.predict(start=len(ts_membership)-12, end=len(ts_membership)-1)
mape = mean_absolute_percentage_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f"Validation Metrics -> MAPE: {mape:.2f}, RMSE: {rmse:.2f}")

# Save output
forecast_df.to_csv(output_file, index=False)
print(f"Forecast saved to {output_file}")
