import requests
import json
import pandas as pd

# Make the API request.
# This code takes only the player tags from the clans with members between 11 and 15 you can change it upto min:0 and max:20. 
# But there is a cap and you need to extract with little in between.

url = "https://proxy.royaleapi.dev/v1/clans?minMembers=11&maxMembers=15&limit=1000&minScore=1"
headers = {
    'Authorization': "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6Ijk0NTdjOTk2LWNjNWEtNDY1Ny1iM2QxLWIwMjYyYTgyMjRlMCIsImlhdCI6MTcxMDI2NDQ0OCwic3ViIjoiZGV2ZWxvcGVyL2NlMGYwMWY2LTcxNTgtMTIxMS1mNzgwLWYwZTA3MjFkMGIxNCIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyI0NS43OS4yMTguNzkiXSwidHlwZSI6ImNsaWVudCJ9XX0.qxJlx8GqxruPCMHhMGo9NrLCEO1Ia3VufLCTDGHN9XqFf9ZTyHArgMKw6G5H9VE32FzH-EEmxVI6jih7UvrAPA",
    'Accept' : "application/json"
}
response = requests.request("GET", url, headers=headers)

# Extract 'tag' column from the response and create a DataFrame
tag_df = pd.DataFrame({'tag': [item['tag'][1:] for item in response.json()['items']]})

# Save DataFrame to Excel file
tag_df.to_excel('clan_tags-11-15.xlsx', index=False)

print("Excel file with clan tags saved successfully.")