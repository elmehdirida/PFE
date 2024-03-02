import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

# Assuming your data is stored in a pandas DataFrame called 'data'
# and the frequency is 24 for hourly data
import matplotlib.pyplot as plt

data = pd.read_csv('opsd_germany_daily.csv', parse_dates=['Date'])
result = seasonal_decompose(data['Consumption'], model='additive', period=24)

# Access the seasonal, trend, and residuals components
seasonal = result.seasonal
trend = result.trend
residuals = result.resid
 
# Plot the decomposed time series
result.plot()
plt.show()