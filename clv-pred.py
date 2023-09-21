import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

# Define a function to perform model selection and evaluation
def perform_model_selection(file_path, target_variable):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Define features (X) and the target variable (y)
    X = data.drop(columns=[target_variable,'Location','ChurnHistory','MarketingResponse'])
    y = data[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train multiple regression models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Machine': SVR(kernel='linear')
    }

    best_model = None
    best_mse = np.inf
    model_value = {}

    for model_name, model in models.items():
        # Perform cross-validation to evaluate each model
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mse_scores = -cv_scores
        avg_mse = np.mean(mse_scores)
        # Check if the current model has the lowest MSE
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_model = model_name

    # Train the best model on the entire training set
    selected_model = models[best_model]
    selected_model.fit(X_train, y_train)

    # Make predictions on the test set using the selected model
    y_pred = selected_model.predict(X_test)

    # Evaluate the selected model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return best_model, mse, r2

# Streamlit app
st.title("Client Value Prediction")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # User selects target variable
    target_variable = st.text_input("Enter Target Variable Name:")

    if target_variable:
        # Perform model selection and evaluation
        best_model, mse, r2 = perform_model_selection(uploaded_file, target_variable)

        # Display results
        st.header("Results:")
        st.write(f"Selected Model: {best_model}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R2) Score: {r2:.2f}")
