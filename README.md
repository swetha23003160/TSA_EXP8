# Ex.No: 08     MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

file_name = 'tsla_2014_2023.csv'
data = pd.read_csv(file_name)
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)
time_series_data = data['close']

print("Shape:", time_series_data.shape)
print(time_series_data.head())

scaler = MinMaxScaler(feature_range=(0.001, 1))
scaled_data_array = scaler.fit_transform(time_series_data.values.reshape(-1, 1))
scaled_data = pd.Series(scaled_data_array.flatten(), index=time_series_data.index)

rolling_mean_5 = time_series_data.rolling(window=5).mean()
rolling_mean_10 = time_series_data.rolling(window=10).mean()

plt.figure(figsize=(12,6))
plt.plot(time_series_data, label='Original Data')
plt.plot(rolling_mean_5, label='MA window=5')
plt.plot(rolling_mean_10, label='MA window=10')
plt.legend()
plt.title('Moving Average - Close Price')
plt.grid()
plt.show()

x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=252).fit()
test_predictions = model.forecast(len(test_data))

plt.figure(figsize=(12,6))
plt.plot(train_data, label="Train Data (Scaled)")
plt.plot(test_data, label="Test Data (Scaled)")
plt.plot(test_predictions, label="Predictions (Scaled)")
plt.legend()
plt.title('Forecasting on Close Price (Scaled)')
plt.show()

print("RMSE (on Scaled Data):", np.sqrt(mean_squared_error(test_data, test_predictions)))
```

### OUTPUT:

Moving Average

<img width="1962" height="1055" alt="Screenshot 2025-11-14 204439" src="https://github.com/user-attachments/assets/da3c514a-bdc1-4a4a-8585-c286dfb23ffa" />

Exponential Smoothing
<img width="1939" height="1060" alt="Screenshot 2025-11-14 204452" src="https://github.com/user-attachments/assets/fbdedf0c-6707-4b4a-987d-448bb89a58e9" />



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
