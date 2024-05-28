import requests
import json
import pandas as pd

# Make the API request.
# This code takes only the player tags from the clans with members between 11 and 15 you can change it upto min:0 and max:20. 
# But there is a cap and you need to extract with little in between.

url = "https://proxy.royaleapi.dev/v1/clans?minMembers=11&maxMembers=15&limit=1000&minScore=1"
headers = {
    'Authorization': "Bearer [Your API Key without the brackets]",
    'Accept' : "application/json"
}
response = requests.request("GET", url, headers=headers)

# Extract 'tag' column from the response and create a DataFrame
tag_df = pd.DataFrame({'tag': [item['tag'][1:] for item in response.json()['items']]})

# Save DataFrame to Excel file
tag_df.to_excel('clan_members_tags.xlsx', index=False)

print("Excel file with clan tags saved successfully.")
