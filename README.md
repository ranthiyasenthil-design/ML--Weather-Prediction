# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required Python libraries.

• Load the weather station dataset.

• Convert the time column into datetime format.

• Extract month, day, and hour features from the time column.

• Handle missing values in the dataset.

• Select humidity, pressure, wind speed, month, day, and hour as input features.

• Select temperature, PM2.5, and energy values as target variables.

• Split the dataset into training and testing datasets.

• Create the Random Forest Regression model.

• Train the model using training data.

• Predict the output using testing data.

• Evaluate the model using R² Score, MAE, and RMSE.

• Perform 5-fold cross-validation to verify model stability.

• Display the prediction accuracy and final results. 

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: RANTHIYA S
RegisterNumber:212225230225
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv('weather-station-eee-block_2024_07_13.csv')

# Display first five rows
print("First Five Rows of Dataset")
print(df.head())

# Convert time column into datetime format
df['time'] = pd.to_datetime(df['time'])

# Extract date and time features
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.hour

# Handle missing values
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Select input features
X = df[['hum', 'pressure', 'wind_speed', 'month', 'day', 'hour']]

# Select target variables
Y = df[['tem', 'pm2_5', 'tsr']]

# Split dataset into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)
```

## Output:
<img width="1653" height="521" alt="image" src="https://github.com/user-attachments/assets/bc9330b7-f787-4c34-ae3d-e757a5528098" />
<img width="1686" height="327" alt="image" src="https://github.com/user-attachments/assets/062efdda-a8ba-4ec3-a033-35def6241d1c" />


## Result:
