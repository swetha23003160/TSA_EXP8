# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date:14/10/2025
### Name:Dhivya Dharshini B
### Reg:212223240031


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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data = pd.read_csv('gold_price_data.csv', parse_dates=['Date'], index_col='Date')

# Use the 'Value' column which contains the gold prices
value_data = data[['Value']]

print("Shape of the dataset:", value_data.shape)
print("First 10 rows of the dataset:")
print(value_data.head(10))

plt.figure(figsize=(12, 6))
plt.plot(value_data['Value'], label='Original Gold Price Data')
plt.title('Original Gold Price Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = value_data['Value'].rolling(window=5).mean()
rolling_mean_10 = value_data['Value'].rolling(window=10).mean()

# Display rolling means (optional - can remove if not needed in final output)
print("\nRolling Mean (window=5):")
print(rolling_mean_5.head(10))
print("\nRolling Mean (window=10):")
print(rolling_mean_10.head(20))


plt.figure(figsize=(12, 6))
plt.plot(value_data['Value'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Average of Gold Price Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Resample to monthly data for Exponential Smoothing
data_monthly = data['Value'].resample('MS').mean()   #Month start, using mean for aggregation

# Handle missing values before scaling
data_monthly_filled = data_monthly.ffill().bfill()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly_filled.values.reshape(-1, 1)).flatten(),index=data_monthly_filled.index)

# Add a small constant to make all values strictly positive
scaled_data = scaled_data + 1e-9


x=int(len(scaled_data)*0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# Using additive trend and multiplicative seasonality
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

plt.figure(figsize=(12, 6))
ax=train_data.plot(label="train_data")
test_predictions_add.plot(ax=ax, label="test_predictions_add")
test_data.plot(ax=ax, label="test_data")
ax.legend()
ax.set_title('Visual evaluation')
plt.show()


print("\nRMSE on Test Data:", np.sqrt(mean_squared_error(test_data, test_predictions_add)))

# You might want to analyze the variance and mean of the scaled data if needed
print("\nScaled Data Variance and Mean:", np.sqrt(scaled_data.var()), scaled_data.mean())


# Build and train the final model on the entire monthly dataset
model = ExponentialSmoothing(data_monthly_filled, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model.forecast(steps=int(len(data_monthly_filled)/4)) # Forecast for the next 1/4 of the data length

plt.figure(figsize=(12, 6))
ax=data_monthly_filled.plot(label="data_monthly_filled")
predictions.plot(ax=ax, label="predictions")
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Average Gold Price')
ax.set_title('Gold Price Prediction using Exponential Smoothing')
plt.show()
```
### OUTPUT:
### Orginal Data
<img width="349" height="305" alt="image" src="https://github.com/user-attachments/assets/61314cf0-74f5-4ced-bb2a-d1bcaafbb6cf" />

<img width="1014" height="547" alt="download" src="https://github.com/user-attachments/assets/e6fed0f6-fd69-4b95-8c2e-eaa52f97740b" />


<img width="280" height="286" alt="image" src="https://github.com/user-attachments/assets/96f5aaa5-2d2a-4152-98cb-3cfcbac96c99" />

<img width="327" height="510" alt="image" src="https://github.com/user-attachments/assets/8c808a45-d1ff-4e59-8281-e9f4ca47a03f" />

### Moving Average
<img width="1014" height="547" alt="download" src="https://github.com/user-attachments/assets/12fd15c7-fedf-4e1d-b131-6c8c453345bc" />

### Plot Transform Dataset

<img width="981" height="547" alt="download" src="https://github.com/user-attachments/assets/1a27c8b4-47cd-43b3-9dcc-6de235399d07" />

### Exponential Smoothing
<img width="366" height="24" alt="image" src="https://github.com/user-attachments/assets/7894e65b-a77e-41b9-96e4-d5c187820c09" />

<img width="1014" height="547" alt="download" src="https://github.com/user-attachments/assets/732619f1-7dc7-45a1-9347-d0a716b53ff5" />


### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
