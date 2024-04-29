from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("buildingdata10.csv")

# Selecting features and target variable
X = df.drop('Total_electricity_consumption', axis=1)
y = df['Total_electricity_consumption']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)

# Define the Decision Tree model
model = DecisionTreeRegressor(max_depth=3, random_state=42)

# Train the Decision Tree model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE, MSE, R-squared, Explained Variance, and Mean Absolute Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) * 100
explained_var = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Accuracy: {r2:.4f}%')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Explained Variance: {explained_var}')
print(f'Mean Absolute Error (MAE): {mae}')

# Plotting actual vs. predicted valuesi
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs. Predicted Values', fontweight='bold')
plt.xlabel('Observation', fontweight='bold')
plt.ylabel('Total Electricity Consumption', fontweight='bold')
plt.legend()
plt.show()
