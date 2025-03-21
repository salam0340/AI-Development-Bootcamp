# Import necessary libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Create a DataFrame from the dataset
df = housing.frame

# Display the first few rows of the dataset
print("California Housing Data:")
print(df.head())

# Features (independent variables) and target (dependent variable)
X = df.drop('MedHouseVal', axis=1)  # Drop the target column 'MedHouseVal' (Median House Value)
y = df['MedHouseVal']  # Target column: Median House Value

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error and R^2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Display the coefficients of the model
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Create a DataFrame for the coefficients with their corresponding feature names
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nCoefficients for each feature:")
print(coef_df)

# Test the model with new data
new_data = pd.DataFrame({
    'MedInc': [5],  # Median income in block group
    'HouseAge': [30],  # Median house age
    'AveRooms': [6],  # Average rooms per household
    'AveBedrms': [1],  # Average bedrooms per household
    'Population': [500],  # Population in block group
    'AveOccup': [3],  # Average house occupancy
    'Latitude': [34.05],  # Latitude of block group
    'Longitude': [-118.25]  # Longitude of block group
})

predicted_price = model.predict(new_data)
print(f"\nPredicted house price for the new data: ${predicted_price[0]:,.2f}")
