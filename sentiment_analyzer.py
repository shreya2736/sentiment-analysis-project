from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
import numpy as np
from typing import Dict, List
import re
import os
import streamlit as st

class SentimentAnalyzer:
    def __init__(self):
        self.finbert = None
        self.tokenizer = None
        self.load_models()
    
    def load_models(self):
        """Load sentiment analysis models"""
        try:
            print("🔄 Loading FinBERT model and tokenizer...")
            model_name = "ProsusAI/finbert"
            # Check if running on cloud with limited resources
            is_cloud = os.getenv('STREAMLIT_CLOUD', False) or 'STREAMLIT_SHARING' in os.environ
            
            if is_cloud:
                print("☁️ Cloud environment detected - using optimized settings")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert = pipeline(
                "sentiment-analysis", 
                model=model_name, 
                tokenizer=model_name,
                truncation=True,  # Enable automatic truncation
                max_length=512,   # Set max length
                padding=True      # Enable padding
            )
            print("✅ FinBERT model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading FinBERT: {e}")
            self.finbert = None
            self.tokenizer = None
    
    def analyze_sentiment_optimized():
        """Optimized sentiment analysis for cloud deployment"""
        try:
            # Try to load preprocessed data
            df = pd.read_csv("preprocessed.csv")
            
            # For cloud: process in smaller batches to avoid memory issues
            batch_size = min(50, len(df))  # Smaller batches for cloud
            
            if len(df) > 100:  # If large dataset, use sampling for cloud
                st.info("📊 Large dataset detected. Using optimized processing for cloud...")
                # Take sample for cloud deployment to avoid timeouts
                df_sample = df.sample(n=min(100, len(df)), random_state=42)
                result = analyze_sentiment_with_finbert()
                return result
            else:
                return analyze_sentiment_with_finbert()
                
        except Exception as e:
            print(f"Optimized analysis failed: {e}")
            return analyze_sentiment_with_finbert()  # Fallback to original
    
    def truncate_text_by_tokens(self, text, max_tokens=500):
        """Properly truncate text by token count, not character count"""
        if not text or pd.isna(text):
            return ""
        
        text_str = str(text)

        if not self.tokenizer:
            # Fallback: truncate by characters
            return text_str[:1500]
        try:
            # Tokenize the text
            tokens = self.tokenizer.encode(text_str, truncation=False)
            
            # If text is within limit, return as is
            if len(tokens) <= max_tokens:
                return text_str
            
            # Truncate tokens and convert back to text
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            return truncated_text
        except Exception as e:
           print(f"⚠️ Token truncation error: {e}")
           return text_str[:1500] 

    
    def analyze_sentiment_finbert(self, text):
        """Analyze sentiment using FinBERT with proper text truncation"""
        if not text or pd.isna(text) or len(str(text).strip()) == 0:
            return {"label": "neutral", "score": 0.0, "sentiment_score": 0.0}
        
        if not self.finbert:
            return self.analyze_sentiment_fallback(text)
        
        try:
            # Truncate text properly by tokens
            if self.tokenizer:
                truncated_text = self.truncate_text_by_tokens(text, max_tokens=500)
                if not truncated_text or len(truncated_text.strip()) < 10:
                    return {"label": "neutral", "score": 0.0, "sentiment_score": 0.0}
            else:
                # Fallback: truncate by characters (less accurate)
                truncated_text = str(text)[:1500]
            
            # Analyze sentiment
            result = self.finbert(truncated_text)[0]
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
    
    def analyze_text_sentiment_smart(self, row):
        """Smart sentiment analysis that prioritizes important text parts"""
        try:
            # Strategy 1: Try title + description first (usually most important)
            primary_text = f"{row['title']}. {row['description']}"
            primary_text = primary_text.strip()
            
            if len(primary_text) > 50:  # Only if we have meaningful text
                result = self.analyze_sentiment_finbert(primary_text)
                if result['score'] > 0.7:  # High confidence from title+description
                    return result
            
            # Strategy 2: If low confidence or short text, add beginning of content
            if 'content' in row and pd.notna(row['content']):
                # Take first 2-3 sentences from content
                content_preview = str(row['content'])[:800]  # Conservative character limit
                full_text = f"{primary_text}. {content_preview}"
                return self.analyze_sentiment_finbert(full_text)
            
            # Strategy 3: Fallback to just title + description
            return self.analyze_sentiment_finbert(primary_text)
            
        except Exception as e:
            print(f"❌ Smart analysis error: {e}")
            return {"label": "neutral", "score": 0.0, "sentiment_score": 0.0}
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment for a single text (backward compatibility)"""
        return self.analyze_sentiment_finbert(text)

def analyze_sentiment_with_finbert():
    """Main sentiment analysis function with enhanced error handling"""
    print("🎯 Starting sentiment analysis...")
    
    try:
        # Check if preprocessed file exists
        if not os.path.exists("preprocessed.csv"):
            print("❌ preprocessed.csv not found. Looking for industry_insights_clean.csv...")
            
            if not os.path.exists("industry_insights_clean.csv"):
                print("❌ No data files found. Please run data collection first.")
                return pd.DataFrame()
            
            # If preprocessed doesn't exist, use clean file
            df = pd.read_csv("industry_insights_clean.csv")
            print("📥 Using industry_insights_clean.csv")
        else:
            df = pd.read_csv("preprocessed.csv")
            print(f"📥 Loaded {len(df)} records from preprocessed.csv")
        
        if df.empty:
            print("❌ Data file is empty")
            return pd.DataFrame()
        
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
        successful_analyses = 0
        failed_analyses = 0
        
        for idx, row in df.iterrows():
            if idx % 20 == 0:  # More frequent updates
                print(f"📊 Processed {idx}/{len(df)} articles... (Success: {successful_analyses}, Failed: {failed_analyses})")
            
            try:
                # Use smart analysis that handles text length properly
                sentiment = analyzer.analyze_text_sentiment_smart(row)
                sentiment_results.append(sentiment)
                successful_analyses += 1
                
            except Exception as e:
                print(f"❌ Error analyzing article {idx}: {e}")
                sentiment_results.append({"label": "neutral", "score": 0.0, "sentiment_score": 0.0})
                failed_analyses += 1
        
        # Add sentiment results to dataframe
        sentiment_df = pd.DataFrame(sentiment_results)
        df["sentiment"] = sentiment_df["label"]
        df["sentiment_score"] = sentiment_df["sentiment_score"]
        df["sentiment_confidence"] = sentiment_df["score"]
        
        # Calculate sentiment statistics
        total_articles = len(df)
        positive_count = (df['sentiment'] == 'positive').sum()
        negative_count = (df['sentiment'] == 'negative').sum()
        neutral_count = (df['sentiment'] == 'neutral').sum()
        
        print(f"\n📈 Sentiment Analysis Results:")
        print(f"   Total articles analyzed: {total_articles}")
        print(f"   Successful analyses: {successful_analyses}")
        print(f"   Failed analyses: {failed_analyses}")
        print(f"   Positive: {positive_count} ({positive_count/total_articles*100:.1f}%)")
        print(f"   Negative: {negative_count} ({negative_count/total_articles*100:.1f}%)")
        print(f"   Neutral:  {neutral_count} ({neutral_count/total_articles*100:.1f}%)")
        print(f"   Average Score: {df['sentiment_score'].mean():.3f}")
        print(f"   Average Confidence: {df['sentiment_confidence'].mean():.3f}")
        
        # Show some examples
        print(f"\n🎯 Sample Results:")
        sample_positive = df[df['sentiment'] == 'positive'].head(2)
        sample_negative = df[df['sentiment'] == 'negative'].head(2)
        
        for i, (idx, row) in enumerate(sample_positive.iterrows()):
            print(f"   Positive #{i+1}: Score={row['sentiment_score']:.3f}, '{row['title'][:60]}...'")
        
        for i, (idx, row) in enumerate(sample_negative.iterrows()):
            print(f"   Negative #{i+1}: Score={row['sentiment_score']:.3f}, '{row['title'][:60]}...'")
        
        # Save results
        output_path = "industry_insights_with_financial_sentiment.csv"
        df.to_csv(output_path, index=False)
        print(f"\n💾 Sentiment analysis completed and saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Sentiment analysis error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Alternative function for batch processing (more efficient)
def analyze_sentiment_batch():
    """Batch process sentiment analysis for better performance"""
    print("🎯 Starting batch sentiment analysis...")
    
    try:
        df = pd.read_csv("preprocessed.csv")
        print(f"📥 Loaded {len(df)} records")
        
        analyzer = SentimentAnalyzer()
        
        if analyzer.finbert is None:
            print("❌ No sentiment model available")
            return pd.DataFrame()
        
        # Prepare texts for batch processing
        texts = []
        for idx, row in df.iterrows():
            # Use smart text combination
            primary_text = f"{row['title']}. {row['description']}".strip()
            texts.append(primary_text)
        
        # Process in batches
        batch_size = 16
        sentiment_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"🔍 Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            
            try:
                batch_results = analyzer.finbert(batch_texts)
                
                for result in batch_results:
                    label = result['label'].lower()
                    score = result['score']
                    
                    if label == "positive":
                        numeric_score = score
                    elif label == "negative":
                        numeric_score = -score
                    else:
                        numeric_score = 0.0
                    
                    sentiment_results.append({
                        "label": label,
                        "score": score,
                        "sentiment_score": round(numeric_score, 4)
                    })
                    
            except Exception as batch_error:
                print(f"❌ Batch error: {batch_error}")
                # Add neutral results for failed batch
                for _ in range(len(batch_texts)):
                    sentiment_results.append({"label": "neutral", "score": 0.0, "sentiment_score": 0.0})
        
        # Add results to dataframe
        sentiment_df = pd.DataFrame(sentiment_results)
        df["sentiment"] = sentiment_df["label"]
        df["sentiment_score"] = sentiment_df["sentiment_score"]
        df["sentiment_confidence"] = sentiment_df["score"]
        
        # Save results
        output_path = "industry_insights_with_financial_sentiment.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Batch analysis completed and saved to: {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Batch analysis error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Try batch processing first, fallback to individual processing
    df = analyze_sentiment_batch()
    
    if df.empty or len(df) == 0:
        print("🔄 Batch processing failed, trying individual processing...")
        df = analyze_sentiment_with_finbert()
    
    if not df.empty:
        print(f"✅ Sentiment analysis completed. Shape: {df.shape}")
    else:
        print("❌ Sentiment analysis failed.")