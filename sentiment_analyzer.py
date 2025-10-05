from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

def analyze_sentiment_with_finbert():
    df = pd.read_csv("preprocessed.csv")

    # Load FinBERT for financial sentiment
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    def analyze_sentiment(row):
        #title + description + content
        text = f"{row['title']} {row['description']} {row['content']}"
        if pd.isna(text) or len(text.strip()) == 0:
            return {"label": "neutral", "score": 0.0}

        #truncate to model limit(512 tokens)
        result = finbert(text[:512])[0]
        label = result['label'].lower()
        score = result['score']

        #map to numeric sentiment score
        if label == "positive":
            numeric_score = +score
        elif label == "negative":
            numeric_score = -score
        else:  # neutral
            numeric_score = 0.0

        return {"label": label, "score": numeric_score}

    # Apply analysis and split into two new columns
    sentiment_results = df.apply(analyze_sentiment, axis=1)
    df["sentiment"] = sentiment_results.apply(lambda x: x["label"])
    df["sentiment_score"] = sentiment_results.apply(lambda x: x["score"])

    # Save with sentiment label + score
    output_path = "industry_insights_with_financial_sentiment.csv"
    df.to_csv(output_path, index=False)
    print(f"Sentiment analysis completed and saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    df = analyze_sentiment_with_finbert()
    print(f"Sentiment analysis completed. Shape: {df.shape}")