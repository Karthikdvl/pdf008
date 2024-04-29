import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv("E:/project/buildingdata1.csv")

# Selecting features and target variable
X = df.drop('Total_electricity_consumption', axis=1)
y = df['Total_electricity_consumption']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store training and testing RMSE values
train_rmse = []
test_rmse = []

# Train Decision Tree models with different depths
depths = range(1, 28)
for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    # Predict on training set
    y_train_pred = model.predict(X_train)
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))

    # Predict on testing set
    y_test_pred = model.predict(X_test)
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Plotting RMSE values
plt.plot(depths, train_rmse, label='Train RMSE')
plt.plot(depths, test_rmse, label='Test RMSE')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('RMSE')
plt.title('Decision Tree Training and Testing RMSE vs. Max Depth')
plt.legend()
plt.show()
