import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("E:/project/buildingdata2.csv")

# Selecting features and target variable
X = df.drop('Total_electricity_consumption', axis=1)
y = df['Total_electricity_consumption']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Combine actual and predicted values for sorting
train_data = sorted(zip(y_train, y_train_pred))
test_data = sorted(zip(y_test, y_test_pred))

# Separate sorted values
y_train_sorted, y_train_pred_sorted = zip(*train_data)
y_test_sorted, y_test_pred_sorted = zip(*test_data)

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_train_sorted, label='Actual Values (Training)', color='blue')
plt.plot(y_train_pred_sorted, label='Predicted Values (Training)', linestyle='--', color='green')
plt.plot(y_test_sorted, label='Actual Values (Testing)', color='red')
plt.plot(y_test_pred_sorted, label='Predicted Values (Testing)', linestyle='--', color='orange')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values (Sorted)')
plt.legend()
plt.show()
