import os
import streamlit as st
from dotenv import load_dotenv


# Load environment variables for local development
load_dotenv()

def get_secret(key, default=None):
    """Get secrets from Streamlit Cloud or environment variables with better error handling"""
    try:
        # First try Streamlit secrets (for cloud deployment)
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    
    # Fallback to environment variables
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    # Final fallback to default
    return default

# Use the function to get keys with safe fallbacks
NEWSAPI_KEY = get_secret("NEWS_API_KEY")
SERPAPI_KEY = get_secret("SERP_API_KEY")
TWITTER_BEARER_TOKEN = get_secret("TWITTER_BEARER_TOKEN")
REDDIT_CLIENT_ID = get_secret("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = get_secret("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = get_secret("REDDIT_USER_AGENT", "strategic_intelligence_dashboard_v1.0")
SLACK_BOT_TOKEN = get_secret("SLACK_BOT_TOKEN")
SLACK_CHANNEL = get_secret("SLACK_CHANNEL", "#alerts")

# Reduce limits for cloud deployment to avoid timeouts
QUERY = "Business OR market OR Artificial Intelligence OR finance OR technology"
MAX_RESULTS = 50  # Reduced from 100 for cloud compatibility

# Cloud deployment check
IS_CLOUD = os.getenv('STREAMLIT_CLOUD', False) or 'STREAMLIT_SHARING' in os.environ