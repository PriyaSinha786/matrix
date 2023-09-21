import streamlit as st
import joblib
import numpy as np
import os

# Load the trained models
model_names = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR', 'Random Forest Regressor', 'Gradient Boosting Regressor']
models = {}

"""for model_name in model_names:
    # model_filename = f'{model_name.replace(" ", "_")}_model.pkl'
    model_filename = 
    model = joblib.load(model_filename)
    models[model_name] = model"""

folder_path = 'D:\Projects\CLV-prediction\pkl_files'  # Replace 'your_folder_path' with the actual folder path

# List all .pkl files in the folder
pkl_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

# Process each .pkl file
for pkl_file in pkl_files:
    # Construct the full file path
    file_path = os.path.join(folder_path, pkl_file)
    
    # Load the .pkl file
    model = joblib.load(file_path)
    models[pkl_file] = model
    print("SELECTED PKL FILES")



# Create a Streamlit web app
st.title('Model Prediction App')

# User input for features
st.sidebar.header('Input Features')
client_id = st.sidebar.number_input('Client ID', min_value=0, step=1)
num_accounts = st.sidebar.number_input('Number of Accounts', min_value=0, step=1)
savings_balance = st.sidebar.number_input('Savings Balance', min_value=0, step=1)
checking_balance = st.sidebar.number_input('Checking Balance', min_value=0, step=1)
Investment_balance = st.sidebar.number_input('Investment Balance', min_value=0, step=1)
Transaction_frequency = st.sidebar.number_input('Transaction Frequency', min_value=0, step=1)
Transaction_volume = st.sidebar.number_input('Transaction Volume', min_value=0, step=1)
credit_score = st.sidebar.number_input('Credit Score', min_value=0, step=1)
tenure = st.sidebar.number_input('Tenure', min_value=0, step=1)
churn_history = st.sidebar.number_input('Churn History', min_value=0, step=1)
complaints = st.sidebar.number_input('Complaints', min_value=0, step=1)
market_response = st.sidebar.number_input('Market_Response', min_value=0, step=1)

# Add more input fields for other features here

# Make predictions and display results
if st.sidebar.button('Run Predictions'):
    input_data = np.array([[client_id, num_accounts,savings_balance,checking_balance,Investment_balance,Transaction_frequency,Transaction_volume,credit_score,tenure,churn_history,complaints,market_response]])  # Add other feature values here
    st.write('### Predicted Credit Scores:')
    
    for model_name, model in models.items():
        prediction = model.predict(input_data)
        st.write(f'{model_name}: {prediction[0]:.2f}')

# Display model information
st.sidebar.header('Model Information')
st.sidebar.write('Trained models:')
st.sidebar.write(model_names)

