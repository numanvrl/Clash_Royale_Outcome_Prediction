import pandas as pd

# Load your data into a DataFrame
data = pd.read_excel('player_battle_logs-output.xlsx')

# Check for NaN values in each column
nan_columns = data.isna().any()

# Check for NaN values in each row
nan_rows = data.isna().any(axis=1)

# Print columns with NaN values
print("Columns with NaN values:")
print(nan_columns)

# Print rows with NaN values
print("\nRows with NaN values:")
print(nan_rows)