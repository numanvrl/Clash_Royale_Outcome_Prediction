import requests
import json
import pandas as pd

# Make the API request
url = "https://proxy.royaleapi.dev/v1/cards"
headers = {
    'Authorization': "Bearer [Your API Key without the brackets]",
    'Accept' : "application/json"
}
response = requests.request("GET", url, headers=headers, timeout=30)

name_df = pd.DataFrame({'name': [item['name'] for item in response.json()['items']]})

name_df.to_excel('cards.xlsx', index=False)

print("Excel file saved successfully.")
