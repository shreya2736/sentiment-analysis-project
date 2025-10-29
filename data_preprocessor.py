import re
import pandas as pd
import numpy as np
from datetime import datetime

def clean_date(date_val):
    """Clean and standardize date format"""
    try:
        return pd.to_datetime(date_val).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None

def clean_html(text):
    """Remove HTML tags from text"""
    if pd.isna(text):
        return ""
    return re.sub(r"<.*?>", "", str(text))

def preprocess_text(text):
    """Preprocess text for analysis"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # remove hashtag symbol, keep word
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)  # remove special chars except basic punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text

def extract_keywords(text, max_keywords=10):
    """Extract important keywords from text"""
    if pd.isna(text) or not text:
        return []
    
    # Common stop words to exclude
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'])
    
    words = text.lower().split()
    # Filter words
    keywords = [word for word in words if len(word) > 2 and word not in stop_words and word.isalpha()]
    
    # Count frequency
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:max_keywords]]

def clean_and_preprocess_data():
    """Main preprocessing function"""
    print("🔄 Starting data preprocessing...")
    
    try:
        # Load the collected CSV
        df = pd.read_csv("industry_insights_clean.csv")
        print(f"📥 Loaded data with {len(df)} records")
        
        # Remove duplicates (based on title and URL)
        initial_count = len(df)
        df.drop_duplicates(subset=["title", "url"], inplace=True)
        print(f"🧹 Removed {initial_count - len(df)} duplicates")

        # Handle missing values
        df.dropna(subset=["title", "description"], how='all', inplace=True)
        df["description"] = df["description"].fillna(df["title"])
        df["content"] = df["content"].fillna(df["description"])

        # Standardize date format
        df["publishedAt"] = df["publishedAt"].apply(clean_date)
        df.dropna(subset=["publishedAt"], inplace=True)

        # Trim whitespace in all string columns
        str_cols = df.select_dtypes(include=['object']).columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

        # Remove rows with empty or very short content
        df["content"] = df.apply(
            lambda row: row["content"] if isinstance(row["content"], str) and len(row["content"]) > 20
            else (str(row["title"]) + " " + str(row["description"])),
            axis=1
        )

        # Remove HTML tags
        df["content"] = df["content"].apply(clean_html)
        df["title"] = df["title"].apply(clean_html)
        df["description"] = df["description"].apply(clean_html)

        # Text preprocessing for sentiment analysis
        df["clean_content"] = df["content"].apply(preprocess_text)
        df["clean_title"] = df["title"].apply(preprocess_text)
        df["clean_description"] = df["description"].apply(preprocess_text)

        # Extract keywords
        print("🔍 Extracting keywords...")
        df["keywords"] = df["clean_content"].apply(lambda x: extract_keywords(x))

        # Ensure sector column exists
        if 'sector' not in df.columns:
            df['sector'] = 'general'

        # Reset index
        df.reset_index(drop=True, inplace=True)

        # Save cleaned and preprocessed data
        cleaned_path = "preprocessed.csv"
        df.to_csv(cleaned_path, index=False)
        print(f"💾 Cleaned & preprocessed data saved to: {cleaned_path}")
        print(f"📊 Final preprocessed shape: {df.shape}")
        
        return df

    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = clean_and_preprocess_data()
    if not df.empty:
        print(f"✅ Preprocessing completed. Shape: {df.shape}")
    else:
        print("❌ Preprocessing failed.")