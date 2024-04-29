import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("E:/project/buildingdata1.csv")

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
    'eval_metric': 'rmse',             # evaluation metric (use 'rmse' for MSE)
    'max_depth': 3,                     # maximum depth of a tree
    'learning_rate': 0.1,               # step size shrinkage to prevent overfitting
    'num_boost_round': 100,             # number of boosting rounds
    'seed': 42                          # random seed for reproducibility
}

# Train the XGBoost model
model = xgb.train(params, dtrain, num_boost_round=params['num_boost_round'])

# Plot feature importance
xgb.plot_importance(model, importance_type='weight', xlabel='Weight')
plt.show()
