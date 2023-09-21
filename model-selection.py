import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('updated-syntehtic-data.csv')  # Replace 'your_dataset.csv' with your dataset's file path
# data.drop(columns=['Location'])
# Select your features and target variable
X = data[['ClientID','NumAccounts','SavingsBalance','CheckingBalance','InvestmentBalance','TransactionFrequency','TransactionVolume','CreditScore','Tenure','ChurnHistory','Complaints','MarketingResponse']]  # Replace with the actual feature names

y = data['CreditScore']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'SVR': SVR(),
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor()
}

trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[model_name] = model

for model_name, model in trained_models.items():
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'{model_name} - MSE: {mse}, R2: {r2}')

    # Save the trained model to a pickle file
    model_filename = f'{model_name}_model.pkl'
    joblib.dump(model, model_filename)
    print(f'Saved {model_name} model to {model_filename}')


