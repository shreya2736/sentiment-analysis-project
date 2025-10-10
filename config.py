import os
from dotenv import load_dotenv
import streamlit as st

# Try to load from .env (local development)
load_dotenv()

# For production (Streamlit Cloud) - get from secrets
def get_secret(key):
    try:
        # First try Streamlit secrets
        return st.secrets[key]
    except:
        # Fallback to environment variables
        return os.getenv(key)

# Use the function to get keys
NEWSAPI_KEY = get_secret("NEWSAPI_KEY")
SERPAPI_KEY = get_secret("SERPAPI_KEY")
TWITTER_BEARERTOKEN = get_secret("TWITTER_BEARERTOKEN")
REDDIT_CLIENT_ID = get_secret("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_secret("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = get_secret("REDDIT_USER_AGENT")
SLACK_BOT_TOKEN = get_secret("SLACK_BOT_TOKEN")
SLACK_CHANNEL = get_secret("SLACK_CHANNEL") or "#alerts"

QUERY = "industry trends OR competitor analysis OR market insights OR Artificial Intelligence"
MAX_RESULTS = 100