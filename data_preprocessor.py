import re
import pandas as pd

# Your original preprocessing functions
def clean_date(date_val):
    try:
        return pd.to_datetime(date_val).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return None

def clean_html(text):
    if pd.isna(text):
        return text
    return re.sub(r"<.*?>", "", str(text))

def preprocess_text(text):
    if pd.isna(text):
        return text
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # remove hashtag symbol, keep word
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)  # remove special chars except basic punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text

def clean_and_preprocess_data():
    # Load the collected CSV
    df = pd.read_csv("industry_insights_clean.csv")

    # Remove duplicates (based on title and URL)
    df.drop_duplicates(subset=["title", "url"], inplace=True)

    # Handle missing values
    df.dropna(subset=["title", "description"], how='all', inplace=True)
    df["description"] = df["description"].fillna(df["title"])
    df["content"] = df["content"].fillna(df["description"])

    # Standardize date format
    df["publishedAt"] = df["publishedAt"].apply(clean_date)
    df.dropna(subset=["publishedAt"], inplace=True)

    # Trim whitespace in all string columns
    str_cols = df.select_dtypes(include=['object']).columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

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
    df["content"] = df["content"].apply(preprocess_text)
    df["title"] = df["title"].apply(preprocess_text)
    df["description"] = df["description"].apply(preprocess_text)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Save cleaned and preprocessed data
    cleaned_path = "preprocessed.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned & preprocessed data saved to: {cleaned_path}")
    
    return df

if __name__ == "__main__":
    df = clean_and_preprocess_data()
    print(f"Preprocessing completed. Shape: {df.shape}")