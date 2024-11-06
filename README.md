## DEVELOPED BY: MANOJ G
## REGISTER NO: 212222240060
## DATE:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/astrobiological_activity_monitoring.csv')

# Update these column names based on your file structure
date_column = 'Date'  # Replace with the actual date column name
target_variable = 'Atmospheric_Composition_O2'  # Replace with the actual target variable column name

# Convert the date column to datetime and set it as the index
data[date_column] = pd.to_datetime(data[date_column])
data.set_index(date_column, inplace=True)

# Plot the time series data
plt.plot(data.index, data[target_variable])
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.title(f'{target_variable} Time Series')
plt.show()

# Function to check stationarity using the ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data[target_variable])

# Plot ACF and PACF to help determine SARIMA parameters
plot_acf(data[target_variable])
plt.show()
plot_pacf(data[target_variable])
plt.show()

# Train-test split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data[target_variable][:train_size], data[target_variable][train_size:]

# Define and fit the SARIMA model on the training data
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions on the test set
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot the actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel(target_variable)
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()


```
### OUTPUT:
![image](https://github.com/user-attachments/assets/5af5a7da-0487-4b56-a1ee-ff5ccca7a679)

![image](https://github.com/user-attachments/assets/a00c2b33-894d-4647-b860-38682d7107d2)

![image](https://github.com/user-attachments/assets/7963af5b-f017-4610-9946-6d5f73e4933a)

![image](https://github.com/user-attachments/assets/d0f7c9f2-7d45-4c9e-a80b-e246d395fc66)

![image](https://github.com/user-attachments/assets/69db6fd9-ffd8-40f3-8031-a2b214ed8842)


### RESULT:
Thus, the pyhton program based on the SARIMA model is executed successfully.
