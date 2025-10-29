from config import *
import requests
import pandas as pd
from serpapi import GoogleSearch
import tweepy
import praw
import time
import feedparser
from datetime import datetime, timedelta
import json

def fetch_newsapi(query):
    """Fetch news from NewsAPI"""
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&pageSize={MAX_RESULTS}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"NewsAPI Error: {response.status_code} - {response.text}")
            return []

        articles = response.json().get("articles", [])
        news_data = []

        for art in articles:
            sector = determine_sector_from_text(f"{art.get('title', '')} {art.get('description', '')}")
            news_data.append({
                "title": art.get("title", ""),
                "description": art.get("description", ""),
                "url": art.get("url", ""),
                "publishedAt": art.get("publishedAt", ""),
                "source": art.get("source", {}).get("name", "Unknown"),
                "type": "news",
                "content": art.get("content", ""),
                "sector": sector
            })
        print(f"✓ NewsAPI: {len(news_data)} articles")
        return news_data
    except Exception as e:
        print(f"✗ NewsAPI Error: {e}")
        return []

def fetch_serpapi(query):
    """Fetch news from SerpAPI"""
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "tbm": "nws",
            "num": MAX_RESULTS,
            "hl": "en",
            "gl": "us"
        })

        results = search.get_dict().get("news_results", [])
        serp_data = []

        for item in results:
            sector = determine_sector_from_text(f"{item.get('title', '')} {item.get('snippet', '')}")
            serp_data.append({
                "title": item.get("title", ""),
                "description": item.get("snippet", ""),
                "url": item.get("link", ""),
                "publishedAt": item.get("date", ""),
                "source": item.get("source", ""),
                "type": "news",
                "content": item.get("snippet", ""),
                "sector": sector
            })
        print(f"✓ SerpAPI: {len(serp_data)} articles")
        return serp_data
    except Exception as e:
        print(f"✗ SerpAPI Error: {e}")
        return []

def fetch_reddit(query=QUERY, max_words=200):
    """Fetch posts from Reddit"""
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )

        submissions = reddit.subreddit("all").search(query, limit=100, time_filter='month')

        reddit_data = []
        for submission in submissions:
            raw_content = submission.selftext if submission.selftext else submission.title

            # Shorten content to max_words
            words = raw_content.split()
            if len(words) > max_words:
                raw_content = " ".join(words[:max_words]) + "..."

            sector = determine_sector_from_text(f"{submission.title} {raw_content}")

            reddit_data.append({
                "title": submission.title,
                "description": submission.selftext[:200] + "..." if submission.selftext else submission.title,
                "url": f"https://www.reddit.com{submission.permalink}",
                "publishedAt": pd.to_datetime(submission.created_utc, unit="s"),
                "source": f"Reddit/r/{submission.subreddit.display_name}",
                "type": "reddit_post",
                "content": raw_content,
                "sector": sector
            })
        print(f"✓ Reddit: {len(reddit_data)} posts")
        return reddit_data
    except Exception as e:
        print(f"✗ Reddit API Error: {e}")
        return []

def fetch_google_news_rss(query):
    """Fetch news from Google News RSS"""
    try:
        base_url = "https://news.google.com/rss/search?q="
        rss_url = f"{base_url}{query.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(rss_url)
        rss_data = []
        
        for entry in feed.entries[:80]:
            sector = determine_sector_from_text(f"{entry.title} {entry.get('summary', '')}")
            rss_data.append({
                "title": entry.title,
                "description": entry.get('summary', ''),
                "url": entry.link,
                "publishedAt": entry.published,
                "source": entry.get('source', {}).get('title', 'Google News'),
                "type": "news",
                "content": entry.get('summary', ''),
                "sector": sector
            })
        print(f"✓ Google News RSS: {len(rss_data)} articles")
        return rss_data
    except Exception as e:
        print(f"✗ Google News RSS Error: {e}")
        return []

def fetch_bing_news_search(query):
    """Fetch news using Bing Web Search API (fallback)"""
    try:
        # This is a simplified version - you'd need Bing Search API key
        search_url = "https://api.bing.microsoft.com/v7.0/news/search"
        headers = {"Ocp-Apim-Subscription-Key": ""}  # Would need actual key
        
        params = {
            "q": query,
            "count": 50,
            "mkt": "en-US"
        }
        
        # For now, return empty list - implement if you have Bing API key
        return []
    except Exception as e:
        print(f"✗ Bing News Error: {e}")
        return []

def determine_sector_from_text(text):
    """Determine sector based on text content"""
    if not text or pd.isna(text):
        return "general"
        
    text_lower = str(text).lower()
    
    sector_keywords = {
        "finance": ["bank", "finance", "investment", "stock", "market", "economy", "revenue", 
                   "profit", "loss", "financial", "banking", "investor", "trading", "currency",
                   "bitcoin", "crypto", "dollar", "euro", "investment", "portfolio", "wealth"],
        "technology": ["tech", "software", "ai", "artificial intelligence", "machine learning", 
                      "computer", "digital", "app", "platform", "startup", "innovation", "coding",
                      "programming", "cloud", "data", "algorithm", "robot", "automation"],
        "healthcare": ["health", "medical", "hospital", "doctor", "patient", "medicine", 
                      "healthcare", "pharma", "treatment", "clinical", "drug", "vaccine",
                      "therapy", "surgery", "diagnosis", "wellness", "fitness"],
        "energy": ["energy", "oil", "gas", "renewable", "solar", "wind", "power", "electricity",
                  "petroleum", "energy sector", "nuclear", "coal", "battery", "ev", "electric vehicle"],
        "retail": ["retail", "store", "shop", "consumer", "ecommerce", "amazon", "walmart", 
                  "sales", "shopping", "merchandise", "customer", "product", "price", "discount"],
        "manufacturing": ["manufacturing", "factory", "production", "industrial", "supply chain", 
                         "logistics", "assembly", "manufacturer", "production", "factory", "plant"]
    }
    
    sector_scores = {}
    for sector, keywords in sector_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            sector_scores[sector] = score
    
    if sector_scores:
        return max(sector_scores.items(), key=lambda x: x[1])[0]
    
    return "general"

def collect_all_data(query):
    """Collect data from all available sources"""
    print("🚀 Starting data collection from multiple sources...")
    print(f"📝 Query: {query}")
    
    # Collect from multiple sources
    all_data = []
    
    # NewsAPI
    newsapi_data = fetch_newsapi(query)
    all_data.extend(newsapi_data)
    
    # SerpAPI
    serpapi_data = fetch_serpapi(query)
    all_data.extend(serpapi_data)
    
    # Reddit
    reddit_data = fetch_reddit(query)
    all_data.extend(reddit_data)
    
    # Google News RSS
    google_news_data = fetch_google_news_rss(query)
    all_data.extend(google_news_data)
    
    # Add small delay to avoid rate limiting
    time.sleep(1)

    if not all_data:
        print("❌ No data collected from any source!")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Data cleaning and processing
    df = clean_collected_data(df)
    
    print(f"✅ Data collection completed! Total records: {len(df)}")
    print(f"📊 Sector distribution: {df['sector'].value_counts().to_dict()}")
    print(f"📰 Source distribution: {df['source'].value_counts().head().to_dict()}")
    
    return df

def clean_collected_data(df):
    """Clean and process the collected data"""
    # Remove duplicates based on title and URL
    df = df.drop_duplicates(subset=['title', 'url'])
    
    # Handle missing values
    df['description'] = df['description'].fillna(df['title'])
    df['content'] = df['content'].fillna(df['description'])
    
    # Ensure sector column exists and fill missing values
    if 'sector' not in df.columns:
        df['sector'] = 'general'
    
    # Improve sector detection for rows without proper sector
    for idx, row in df.iterrows():
        if pd.isna(row['sector']) or row['sector'] == 'general':
            text_content = f"{row['title']} {row['description']} {row['content']}"
            detected_sector = determine_sector_from_text(text_content)
            df.at[idx, 'sector'] = detected_sector
    
    # Clean date format
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['publishedAt'])
    
    # Ensure type column is clean
    df['type'] = df['type'].str.lower().fillna("news")
    df.loc[df['url'].str.contains("twitter.com", case=False, na=False), 'type'] = "tweet"
    df.loc[df['url'].str.contains("reddit.com", case=False, na=False), 'type'] = "reddit_post"
    
    return df

if __name__ == "__main__":
    df = collect_all_data(QUERY)
    if not df.empty:
        output_path = "industry_insights_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Data saved to: {output_path}")
        print(f"📈 Final dataset shape: {df.shape}")
    else:
        print("❌ No data to save.")