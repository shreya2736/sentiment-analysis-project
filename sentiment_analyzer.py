from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import numpy as np
from typing import Dict, List
import re

class SentimentAnalyzer:
    def __init__(self):
        self.finbert = None
        self.vader = None
        self.load_models()
    
    def load_models(self):
        """Load sentiment analysis models"""
        try:
            print("🔄 Loading FinBERT model...")
            model_name = "ProsusAI/finbert"
            self.finbert = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
            print("✅ FinBERT model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading FinBERT: {e}")
            self.finbert = None
    
    def analyze_sentiment_finbert(self, text):
        """Analyze sentiment using FinBERT"""
        if not text or pd.isna(text) or len(str(text).strip()) == 0:
            return {"label": "neutral", "score": 0.0, "sentiment_score": 0.0}
        
        try:
            # Truncate to model limit (512 tokens)
            text_str = str(text)[:2000]  # Conservative truncation
            
            result = self.finbert(text_str)[0]
            label = result['label'].lower()
            score = result['score']
            
            # Map to numeric sentiment score
            if label == "positive":
                numeric_score = score
            elif label == "negative":
                numeric_score = -score
            else:  # neutral
                numeric_score = 0.0
                
            return {
                "label": label,
                "score": score,
                "sentiment_score": round(numeric_score, 4)
            }
        except Exception as e:
            print(f"❌ FinBERT analysis error: {e}")
            return {"label": "neutral", "score": 0.0, "sentiment_score": 0.0}
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment for a single text"""
        return self.analyze_sentiment_finbert(text)

def analyze_sentiment_with_finbert():
    """Main sentiment analysis function"""
    print("🎯 Starting sentiment analysis...")
    
    try:
        # Load preprocessed data
        df = pd.read_csv("preprocessed.csv")
        print(f"📥 Loaded {len(df)} records for sentiment analysis")
        
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        if analyzer.finbert is None:
            print("❌ No sentiment model available")
            return pd.DataFrame()
        
        # Analyze sentiment for each article
        print("🔍 Analyzing sentiment for each article...")
        
        sentiment_results = []
        for idx, row in df.iterrows():
            if idx % 50 == 0:
                print(f"📊 Processed {idx}/{len(df)} articles...")
            
            # Combine title, description, and content for analysis
            text = f"{row['title']} {row['description']} {row.get('content', '')}"
            sentiment = analyzer.analyze_text_sentiment(text)
            sentiment_results.append(sentiment)
        
        # Add sentiment results to dataframe
        sentiment_df = pd.DataFrame(sentiment_results)
        df["sentiment"] = sentiment_df["label"]
        df["sentiment_score"] = sentiment_df["sentiment_score"]
        df["sentiment_confidence"] = sentiment_df["score"]
        
        # Calculate sentiment statistics
        positive_count = (df['sentiment'] == 'positive').sum()
        negative_count = (df['sentiment'] == 'negative').sum()
        neutral_count = (df['sentiment'] == 'neutral').sum()
        
        print(f"📈 Sentiment Distribution:")
        print(f"   Positive: {positive_count} ({positive_count/len(df)*100:.1f}%)")
        print(f"   Negative: {negative_count} ({negative_count/len(df)*100:.1f}%)")
        print(f"   Neutral:  {neutral_count} ({neutral_count/len(df)*100:.1f}%)")
        print(f"   Average Score: {df['sentiment_score'].mean():.3f}")
        
        # Save results
        output_path = "industry_insights_with_financial_sentiment.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Sentiment analysis completed and saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    df = analyze_sentiment_with_finbert()
    if not df.empty:
        print(f"✅ Sentiment analysis completed. Shape: {df.shape}")
    else:
        print("❌ Sentiment analysis failed.")