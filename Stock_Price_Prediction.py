# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample historical stock price data (replace this with actual stock data)
data = {
    'Date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
    'Close': [150, 152, 153, 155, 154, 156, 157, 158, 159, 160]  # Closing stock prices
}

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

# Convert the 'Date' column to numeric format (ordinal) for regression analysis
df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

# Display the first few rows of the dataset
print("Stock Price Data:")
print(df.head())

# Features (independent variable) and target (dependent variable)
X = df[['Date']]  # Date as feature (ordinal date)
y = df['Close']  # Target: Closing price

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

# Plot the actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_test, label="Actual Stock Prices", marker='o')
plt.plot(X_test, y_pred, label="Predicted Stock Prices", marker='x')
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Date (Ordinal)")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.show()

# Test the model with new data (future date)
future_date = pd.Timestamp('2022-01-11').toordinal()

# Create a DataFrame with the feature name 'Date'
future_data = pd.DataFrame({'Date': [future_date]})

# Predict the stock price for the future date
predicted_price = model.predict(future_data)
print(f"\nPredicted stock price for {pd.Timestamp.fromordinal(future_date).date()}: ${predicted_price[0]:.2f}")
