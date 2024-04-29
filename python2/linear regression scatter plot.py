from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv("E:/project/buildingdata0.csv")

# Selecting features and target variable
X = df.drop('Total_electricity_consumption', axis=1)
y = df['Total_electricity_consumption']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the Linear Regression model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Create a scatter plot with group indications
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test.index, y=y_test, label='Actual', marker='o', color='blue')
sns.scatterplot(x=y_test.index, y=y_pred, label='Predicted', marker='x', color='black')
plt.title('Scatter Plot of Actual vs. Predicted Values (Linear Regression)')
plt.xlabel('Index (or Row Numbers)')
plt.ylabel('Total Electricity Consumption')
plt.legend()
plt.show()
