import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from your original code
QUERY = "industry trends OR competitor analysis OR market insights OR Artificial Intelligence"
MAX_RESULTS = 100

# API Keys
NEWSAPI_KEY = os.getenv("NEWS_API_KEY")
SERPAPI_KEY = os.getenv("SERP_API_KEY")
TWITTER_BEARERTOKEN = os.getenv("TWITTER_BEARERTOKEN")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#alerts")