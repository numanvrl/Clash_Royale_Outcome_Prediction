import requests
import json
import pandas as pd

# Make the API request
url = "https://proxy.royaleapi.dev/v1/cards"
headers = {
    'Authorization': "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6Ijk0NTdjOTk2LWNjNWEtNDY1Ny1iM2QxLWIwMjYyYTgyMjRlMCIsImlhdCI6MTcxMDI2NDQ0OCwic3ViIjoiZGV2ZWxvcGVyL2NlMGYwMWY2LTcxNTgtMTIxMS1mNzgwLWYwZTA3MjFkMGIxNCIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyI0NS43OS4yMTguNzkiXSwidHlwZSI6ImNsaWVudCJ9XX0.qxJlx8GqxruPCMHhMGo9NrLCEO1Ia3VufLCTDGHN9XqFf9ZTyHArgMKw6G5H9VE32FzH-EEmxVI6jih7UvrAPA",
    'Accept' : "application/json"
}
response = requests.request("GET", url, headers=headers, timeout=30)

name_df = pd.DataFrame({'name': [item['name'] for item in response.json()['items']]})

name_df.to_excel('cards.xlsx', index=False)

print("Excel file saved successfully.")
