
# Google Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account
import google.auth
from google.cloud import secretmanager
# Python API for Google Sheets
import gspread
from gspread_dataframe import set_with_dataframe

# Standard Python packages
import pandas as pd
import numpy as np
# to create deep copy
import copy
# for JSON encoding or decoding
import json

SPREADSHEET_ID = "1_jYx0O8yC0BiH6mn8ci-axA8vah9DHSfx1oRyyhnv8g"

GET_RANGE_NAME = "Sheet1!A1:C"

project_number = 635971252563

secret_gsheet = "moftah"

secret_version_gsheet = 1


# Authenticate Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
client = secretmanager.SecretManagerServiceClient()

secret_gsheet_name = f"projects/{project_number}/secrets/{secret_gsheet}/versions/{secret_version_gsheet}"
secret_gsheet_value = client.access_secret_version(request={"name": secret_gsheet_name}).payload.data.decode("UTF-8")

# Assign secret
SERVICE_ACCOUNT_FILE = json.loads(secret_gsheet_value)



# always specify creds as "None" first before setting it afterwards
CREDS = None
CREDS = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_FILE, scopes=SCOPES) 


# Now call the Sheets API    
gsheet_service = build('sheets', 'v4', credentials = CREDS)
sheet = gsheet_service.spreadsheets()
result = sheet.values().get(spreadsheetId = SPREADSHEET_ID, range = GET_RANGE_NAME).execute()

# get the data, otherwise return an empty list
feedback = result.get('values', []) 

# assign headers which is always going to be row index 0
feedback_headers = feedback.pop(0)

# create a new dataframe with the headers
feedback_df = pd.DataFrame(feedback, columns = feedback_headers)

# view it
print(feedback_df)