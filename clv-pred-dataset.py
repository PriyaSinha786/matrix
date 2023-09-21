import pandas as pd
import random


# Create an empty DataFrame
data = pd.DataFrame(columns=[
    'ClientID', 'Location', 'NumAccounts', 'SavingsBalance', 'CheckingBalance',
    'InvestmentBalance', 'TransactionFrequency', 'TransactionVolume',
    'CreditScore', 'Tenure', 'ChurnHistory', 'Complaints', 'MarketingResponse'
])

# Define possible values for categorical variables
locations = ['New York', 'Chicago', 'Houston', 'Dallas']
churn_history_values = ['Yes', 'No']
marketing_response_values = ['Yes', 'No']

# Generate 100+ rows of synthetic data
for i in range(500):
    client_id = 3004 + i + 1  # Increment ClientID
    location = random.choice(locations)
    num_accounts = random.randint(1, 6)
    savings_balance = random.randint(1000, 50000)
    checking_balance = random.randint(1000, 30000)
    investment_balance = random.randint(10000, 200000)
    transaction_frequency = random.randint(5, 20)
    transaction_volume = random.randint(20, 60)
    credit_score = random.randint(600, 850)
    tenure = random.randint(1, 10)
    churn_history = random.choice(churn_history_values)
    complaints = random.randint(0, 3)
    marketing_response = random.choice(marketing_response_values)

    # Append the generated data to the DataFrame
    data = data.append({
        'ClientID': client_id,
        'Location': location,
        'NumAccounts': num_accounts,
        'SavingsBalance': savings_balance,
        'CheckingBalance': checking_balance,
        'InvestmentBalance': investment_balance,
        'TransactionFrequency': transaction_frequency,
        'TransactionVolume': transaction_volume,
        'CreditScore': credit_score,
        'Tenure': tenure,
        'ChurnHistory': churn_history,
        'Complaints': complaints,
        'MarketingResponse': marketing_response
    }, ignore_index=True)

# Save the synthetic dataset to a CSV file
data_0 = data.to_csv('synthetic-clv-data.csv', index=False)

