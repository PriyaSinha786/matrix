import pandas as pd

# Load your CSV file
df = pd.read_csv('synthetic-clv-data.csv')  # Replace 'your_file.csv' with your CSV file's path

# Define the columns with 'Yes'/'No' values that you want to convert
columns_to_convert = ['ChurnHistory', 'MarketingResponse']  # Replace with your actual column names

# Define the mapping of values
value_mapping = {'Yes': 1, 'No': 0}

# Use the map() function to replace values in the specified columns
for col in columns_to_convert:
    df[col] = df[col].map(value_mapping)

# Save the updated DataFrame to a new CSV file
df.to_csv('updated-syntehtic-data.csv', index=False)  # Replace 'updated_file.csv' with your desired output file path
