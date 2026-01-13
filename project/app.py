
"""
app.py
Purpose: Streamlit UI for sensitivity analysis and visualization.
"""

import streamlit as st
import pandas as pd

# Load forecast results
forecast_df = pd.read_csv("project/output/forecast_results_2026.csv")

# Average contact rate (from historical data)
avg_contact_rate = forecast_df['Forecasted_Call_Volume'].sum() / forecast_df['Forecasted_Membership'].sum()

# UI
st.title("Forecast Sensitivity Analysis")

membership = st.slider("Membership Count", 50000, 200000, 100000)
aht = st.slider("Average Handle Time (AHT) in seconds", 300, 900, 600)
shrinkage = st.slider("Shrinkage (%)", 10, 40, 25)
occupancy = st.slider("Occupancy (%)", 70, 95, 85)

adjusted_volume = membership * avg_contact_rate
st.write(f"Adjusted Call Volume: {adjusted_volume:,.0f}")

agents_needed = adjusted_volume * aht / (occupancy / 100) / (1 - shrinkage / 100) / (8 * 3600)
st.write(f"Agents Needed: {round(agents_needed)}")

st.subheader("Forecasted Membership Trend")
st.line_chart(forecast_df[['Date', 'Forecasted_Membership']].set_index('Date'))

st.subheader("Forecasted Call Volume Trend")
st.line_chart(forecast_df[['Date', 'Forecasted_Call_Volume']].set_index('Date'))

st.download_button("Download Forecast CSV", data=forecast_df.to_csv(index=False), file_name="forecast_results_2026.csv")