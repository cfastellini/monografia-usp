import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

file_path = os.path.join('../files', 'data.csv')
df = pd.read_csv(file_path, delimiter=';')

# Assuming your DataFrame is named df
# Convert the "Data e hora" column to datetime format
df['Data e hora'] = pd.to_datetime(df['Data e hora'], format='%d/%m/%Y %H.%M')

# Extract the date and set it as the index
df['Date'] = df['Data e hora'].dt.date
df.set_index('Date', inplace=True)

# Aggregate the quantity data by date (summing the quantities for each date)
ts_quantity = df.groupby('Date')['Quantidade'].sum()

# Sort the DataFrame by date
ts_quantity.sort_index(inplace=True)

# Fit the Holt-Winters model
model = ExponentialSmoothing(ts_quantity, trend='add', seasonal='add', seasonal_periods=7)
fit_model = model.fit()

# Forecast for the upcoming weeks (adjust as needed)
forecast_periods = 7 * 4  # Forecast for 4 weeks (7 days per week)
forecast = fit_model.forecast(steps=forecast_periods)

# Generate dates for the forecast period
forecast_dates = pd.date_range(ts_quantity.index[-1], periods=forecast_periods + 1, freq='D')[1:]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(ts_quantity.index, ts_quantity, label='Actual')
plt.plot(forecast_dates, forecast, label='Forecast', linestyle='dashed', color='red')
plt.title('Holt-Winters Forecasting for Quantity (Upcoming Weeks)')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.show()
