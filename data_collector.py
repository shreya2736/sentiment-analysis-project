from config import *
import requests
import pandas as pd
from serpapi import GoogleSearch
import tweepy
import praw

# Your original data collection functions
def fetch_newsapi(query):
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={MAX_RESULTS}&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print("NewsAPI Error:", response.text)
        return []

    articles = response.json().get("articles", [])
    news_data = []

    for art in articles:
        news_data.append({
            "title": art.get("title"),
            "description": art.get("description"),
            "url": art.get("url"),
            "publishedAt": art.get("publishedAt"),
            "source": art.get("source", {}).get("name"),
            "type": "news",
            "content": art.get("content")
        })
    return news_data

def fetch_serpapi(query):
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_KEY,
        "tbm": "nws",
        "num": MAX_RESULTS
    })

    results = search.get_dict().get("news_results", [])
    serp_data = []

    for item in results:
        serp_data.append({
            "title": item.get("title"),
            "description": item.get("snippet"),
            "url": item.get("link"),
            "publishedAt": item.get("date"),
            "source": item.get("source"),
            "type": "news",
            "content": item.get("snippet")
        })
    return serp_data

def fetch_reddit(query=QUERY, max_words=200):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    submissions = reddit.subreddit("all").search(query, limit=60)

    reddit_data = []
    for submission in submissions:
        raw_content = submission.selftext if submission.selftext else submission.title

        # Shorten content to max_words
        words = raw_content.split()
        if len(words) > max_words:
            raw_content = " ".join(words[:max_words]) + "..."

        reddit_data.append({
            "title": submission.title,
            "description": submission.selftext[:200] + "..." if submission.selftext else submission.title,
            "url": f"https://www.reddit.com{submission.permalink}",
            "publishedAt": pd.to_datetime(submission.created_utc, unit="s"),
            "source": "Reddit",
            "type": "reddit_post",
            "content": raw_content
        })
    return reddit_data

def collect_all_data(query):
    newsapi_data = fetch_newsapi(query)
    serpapi_data = fetch_serpapi(query)
    reddit_data = fetch_reddit(query=QUERY)

    # Combine only non-empty lists
    combined_data = []
    if newsapi_data:
        combined_data.extend(newsapi_data)
    if serpapi_data:
        combined_data.extend(serpapi_data)
    if reddit_data:
        combined_data.extend(reddit_data)

    if not combined_data:
        print("No data collected!")
        return pd.DataFrame()

    df = pd.DataFrame(combined_data)

    # Ensure that the type column is clean
    df['type'] = df['type'].str.lower().fillna("news")
    df.loc[df['url'].str.contains("twitter.com", case=False, na=False), 'type'] = "tweet"
    df.loc[df['url'].str.contains("reddit.com", case=False, na=False), 'type'] = "reddit_post"

    return df

if __name__ == "__main__":
    df = collect_all_data(QUERY)
    if not df.empty:
        output_path = "industry_insights_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"Data collected and saved to: {output_path}")
    else:
        print("No data to save.")