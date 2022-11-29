
# Google Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account
import google.auth
from google.cloud import secretmanager
# Python API for Google Sheets
import gspread
import pandas as pd

#from gspread_dataframe import set_with_dataframe
from gspread import set_with_dataframe

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







import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

############################  
# Run Sentiment Analysis
############################

# create a deep copy of the df so we dont mess up the original df
sentiment_df = copy.deepcopy(feedback_df)

# set your stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
print(stop_words)

# remove stop words from feedback column. Assign it to a new column called "feedback_without stopwords"
sentiment_df['feedback_without_stopwords'] = sentiment_df['Feedback'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# view df
print(sentiment_df)

# now load vader
nltk.download('vader_lexicon')

# get the vader sentiment intensity analyser
the_force = SentimentIntensityAnalyzer()

# get polarity scores from column in df where stopwords have been removed. Assign to a new column called "polarity scores".
sentiment_df['polarity_scores'] = sentiment_df['feedback_without_stopwords'].apply(the_force.polarity_scores)

# view df
print(sentiment_df)


# get the compound scores only, round to 2 decimals
compound_scores = [round(the_force.polarity_scores(i)['compound'], 2) for i in sentiment_df['feedback_without_stopwords']]

# now create new column in the dataframe and save compound scores in it
sentiment_df['compound_scores'] = compound_scores

# view df
print(sentiment_df)

# create simple logic
sent_logic = [
    (sentiment_df['compound_scores'] < 0),
    (sentiment_df['compound_scores'] >= 0) & (sentiment_df['compound_scores'] < 0.5),
    (sentiment_df['compound_scores']  >= 0.5)
    ]

sent_summary = ['negative', 'neutral', 'positive']

# assign to a new column
sentiment_df['sentiment'] = np.select(sent_logic, sent_summary)

# view df
print(sentiment_df)