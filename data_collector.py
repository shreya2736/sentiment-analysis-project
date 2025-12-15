from config import *
import requests
import pandas as pd

import tweepy
import praw
import time
import feedparser
from datetime import datetime, timedelta
import json
import streamlit as st
from config import *
import re


def is_english_text(text):
    """ULTRA-SIMPLE English detection - only block obvious non-English scripts"""
    if not text or pd.isna(text):
        return True
    
    text_str = str(text)
    
    # Only block texts that contain obvious non-English character blocks
    non_english_scripts = [
        r'[\u4e00-\u9fff]',  # Chinese characters
        r'[\u0400-\u04ff]',  # Cyrillic (Russian, etc.)
        r'[\u0600-\u06ff]',  # Arabic
        r'[\u0900-\u097f]',  # Devanagari (Hindi)
        r'[\uac00-\ud7af]',  # Korean Hangul
        r'[\u3040-\u309f]',  # Japanese Hiragana
        r'[\u30a0-\u30ff]',  # Japanese Katakana
    ]
    
    for pattern in non_english_scripts:
        if re.search(pattern, text_str):
            print(f"🔍 Filtered non-English content: {text_str[:50]}...")
            return False
    
    # If no obvious non-English scripts found, assume it's English
    return True

# Add this function to handle API limits in cloud
def safe_data_collection(query, max_results=MAX_RESULTS):
    """Safe data collection with error handling for cloud deployment"""
    try:
        return collect_all_data(query)
    except Exception as e:
        # Log error for debugging
        if 'st' in globals():
            st.error(f"Data collection error: {e}")
        print(f"Data collection error: {e}")
        return pd.DataFrame()


def fetch_newsapi(query):
    """Fetch news from NewsAPI with English language filter"""
    try:

        if not NEWSAPI_KEY or NEWSAPI_KEY == "your_newsapi_key_here":
            print("⚠️ NewsAPI key not configured - skipping NewsAPI")
            return []
        
        # Add language parameter to only get English content
        url = f"https://newsapi.org/v2/everything?q={query}&pageSize={MAX_RESULTS}&sortBy=publishedAt&language=en&apiKey={NEWSAPI_KEY}"
        response = requests.get(url,timeout=10)

        if response.status_code != 200:
            print(f"NewsAPI Error: {response.status_code} - {response.text}")
            return []

        articles = response.json().get("articles", [])
        news_data = []

        for art in articles:
            # Additional English check
            title_desc = f"{art.get('title', '')} {art.get('description', '')}"
            if not is_english_text(title_desc):
                continue
                
            sector = determine_sector_from_text(title_desc)
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
        print(f"✓ NewsAPI: {len(news_data)} English articles")
        return news_data
    except Exception as e:
        print(f"✗ NewsAPI Error: {e}")
        return []

def fetch_serpapi(query):
    """Fetch news from SerpAPI with English language preference"""
    try:
        # ✅ CHECK 1: Is API key configured?
        if not SERPAPI_KEY or SERPAPI_KEY == "your_serpapi_key_here":
            print("⚠️  SerpAPI key not configured - skipping SerpAPI")
            return []
        try:
            from serpapi.google_search import GoogleSearch
        except ImportError:
            print("⚠️  serpapi module not installed - skipping SerpAPI")
            print("💡 To use SerpAPI, add 'google-search-results' to requirements.txt")
            return []
    
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "tbm": "nws",
            "num": MAX_RESULTS,
            "hl": "en",  # Language: English
            "gl": "us",   # Country: USA
            "lr": "lang_en"  # Language restriction: English
        })

        results = search.get_dict().get("news_results", [])
        serp_data = []

        for item in results:
            title_desc = f"{item.get('title', '')} {item.get('snippet', '')}"
            if not is_english_text(title_desc):
                continue
                
            sector = determine_sector_from_text(title_desc)
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
        print(f"✓ SerpAPI: {len(serp_data)} English articles")
        return serp_data
    except Exception as e:
        print(f"✗ SerpAPI Error: {e}")
        return []
# Replace your fetch_reddit function with this debug version:

def fetch_reddit(query=QUERY, max_words=200):
    """Fetch posts from Reddit with enhanced debugging"""
    try:
        # ✅ CHECK: Are Reddit credentials configured?
        if not REDDIT_CLIENT_ID or REDDIT_CLIENT_ID == "your_reddit_client_id_here":
            print("⚠️ Reddit credentials not configured - skipping Reddit")
            return []
        
        if not REDDIT_CLIENT_SECRET or REDDIT_CLIENT_SECRET == "your_reddit_client_secret_here":
            print("⚠️ Reddit credentials not configured - skipping Reddit")
            return []
        
        try:
            import praw
        except ImportError:
            print("⚠️ praw module not installed - skipping Reddit")
            return []
        
        print(f"🔍 Starting Reddit search for: '{query}'")
        
        # Test Reddit credentials
        print(f"📋 Reddit Config - Client ID: {REDDIT_CLIENT_ID[:10]}..., Client Secret: {REDDIT_CLIENT_SECRET[:10]}..., User Agent: {REDDIT_USER_AGENT}")
        
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        
        # Test authentication
        try:
            user = reddit.user.me()
            print(f"✅ Reddit authenticated as: {user}")
        except Exception as auth_error:
            print(f"❌ Reddit authentication failed: {auth_error}")
            print("💡 Check your Reddit API credentials in config.py")
            return []

        print(f"📢 Searching Reddit with query: '{query}'")
        
        # Try different search approaches
        reddit_data = []
        
        # Approach 1: Search in all subreddits
        try:
            submissions = reddit.subreddit("all").search(query, limit=50, time_filter='month')
            submission_list = list(submissions)
            print(f"🔍 Found {len(submission_list)} submissions via search")
            
            for submission in submission_list:
                try:
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
                    
                except Exception as post_error:
                    print(f"⚠️ Error processing post: {post_error}")
                    continue
                    
        except Exception as search_error:
            print(f"❌ Reddit search failed: {search_error}")
            
            # Fallback: Get hot posts from popular subreddits
            print("🔄 Trying fallback: Getting hot posts from popular subreddits...")
            try:
                subreddits = ["technology", "business", "news", "investing", "stocks", "finance"]
                for subreddit_name in subreddits:
                    try:
                        subreddit = reddit.subreddit(subreddit_name)
                        for submission in subreddit.hot(limit=10):
                            # Check if query matches
                            text_content = f"{submission.title} {submission.selftext}".lower()
                            if any(keyword in text_content for keyword in query.lower().split()):
                                raw_content = submission.selftext if submission.selftext else submission.title
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
                    except Exception as sub_error:
                        print(f"⚠️ Error in subreddit {subreddit_name}: {sub_error}")
                        continue
                        
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
        
        print(f"✓ Reddit: Processed {len(reddit_data)} posts")
        return reddit_data
        
    except Exception as e:
        print(f"✗ Reddit API Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return []

def fetch_google_news_rss(query):
    """Fetch news from Google News RSS with English preference"""
    try:
        # Add language and region parameters
        base_url = "https://news.google.com/rss/search?q="
        rss_url = f"{base_url}{query.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(rss_url)
        rss_data = []
        
        for entry in feed.entries[:80]:
            title_desc = f"{entry.title} {entry.get('summary', '')}"
            if not is_english_text(title_desc):
                continue
                
            sector = determine_sector_from_text(title_desc)
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
        print(f"✓ Google News RSS: {len(rss_data)} English articles")
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
    """Clean and process the collected data with selective English language filtering"""
    if df.empty:
        return df
    
    # Remove duplicates based on title and URL
    df = df.drop_duplicates(subset=['title', 'url'])
    
    # Handle missing values
    df['description'] = df['description'].fillna(df['title'])
    df['content'] = df['content'].fillna(df['description'])
    
    # ✅ SELECTIVE ENGLISH FILTERING - Only for news sources, not for Reddit
    print("🔍 Filtering obvious non-English content from news sources...")
    initial_count = len(df)
    
    # Apply English filter only to news sources (not Reddit/tweets)
    news_mask = ~df['source'].str.contains('reddit', case=False, na=False)
    
    if news_mask.any():
        df_news = df[news_mask]
        df_reddit = df[~news_mask]
        
        # Only check news articles for obvious non-English content
        df_news['is_english'] = df_news.apply(
            lambda row: is_english_text(f"{row['title']}"),  # Only check title for simplicity
            axis=1
        )
        
        df_news = df_news[df_news['is_english'] == True]
        df_news = df_news.drop('is_english', axis=1)
        
        # Combine back (Reddit + filtered news)
        df = pd.concat([df_news, df_reddit], ignore_index=True)
        
        removed_count = initial_count - len(df)
        print(f"✅ Removed {removed_count} obvious non-English news articles")
    else:
        print("✅ No English filtering needed - only Reddit data")
    
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