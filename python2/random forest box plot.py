import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("E:/project/buildingdata1.csv")

# Selecting features and target variable
X = df.drop('Total_electricity_consumption', axis=1)
y = df['Total_electricity_consumption']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)

# Train the Random Forest model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Create a boxplot of actual and predicted values
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))
plt.title('Boxplot of Actual vs. Predicted Values (Random Forest)')
plt.show()
