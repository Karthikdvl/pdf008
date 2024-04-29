import xgboost as xgb
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

# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # for regression tasks
    'eval_metric': ['rmse'],           # evaluation metric (use 'rmse' for MSE)
    'max_depth': 3,                    # maximum depth of a tree
    'learning_rate': 0.1,              # step size shrinkage to prevent overfitting
    'n_estimators': 21,                # number of boosting rounds
    'seed': 42                          # random seed for reproducibility
}

# Train the XGBoost model
model = xgb.train(params, dtrain, params['n_estimators'])

# Predict on the test set
y_pred = model.predict(dtest)

# Create a scatter plot with group indications
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test.index, y=y_test, label='Actual', marker='o', color='blue')
sns.scatterplot(x=y_test.index, y=y_pred, label='Predicted', marker='x', color='black')
plt.title('Scatter Plot of Actual vs. Predicted Values (KNN)')
plt.xlabel('Index (or Row Numbers)')
plt.ylabel('Total Electricity Consumption')
plt.legend()
plt.show()
