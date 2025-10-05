"""
Main execution script for the sentiment analysis pipeline
"""
import pandas as pd
import sys
from data_collector import collect_all_data
from data_preprocessor import clean_and_preprocess_data
from sentiment_analyzer import analyze_sentiment_with_finbert
from forecasting import forecast_sentiment
from alert_system import check_alerts
from dashboard import generate_full_dashboard
from visualization_utils import create_competitor_analysis, create_trend_evolution_analysis, create_alert_history_dashboard
from config import QUERY

def run_full_pipeline():
    """Run the complete sentiment analysis pipeline"""
    print("="*60)
    print("STARTING SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    # Step 1: Data Collection
    print("\n1. COLLECTING DATA...")
    df = collect_all_data(QUERY)
    if df.empty:
        print("No data collected. Exiting pipeline.")
        return
    
    print(f"Collected {len(df)} records from various sources")
    
    # Save collected data
    df.to_csv("industry_insights_clean.csv", index=False)
    print("ZSaved raw data to: industry_insights_clean.csv")
    
    # Step 2: Data Preprocessing
    print("\n2. PREPROCESSING DATA...")
    df_clean = clean_and_preprocess_data()
    if df_clean is None or df_clean.empty:
        print("Preprocessing failed. Exiting pipeline.")
        return
    
    # Save preprocessed data (clean_and_preprocess_data already saves it, but we'll confirm)
    print("Saved preprocessed data to: preprocessed.csv")
    
    # Step 3: Sentiment Analysis
    print("\n3. PERFORMING SENTIMENT ANALYSIS...")
    df_sentiment = analyze_sentiment_with_finbert()
    if df_sentiment is None or df_sentiment.empty:
        print("Sentiment analysis failed. Exiting pipeline.")
        return
    
    # Save sentiment analysis results (analyze_sentiment_with_finbert already saves it)
    print("Saved sentiment analysis to: industry_insights_with_financial_sentiment.csv")
    
    # Step 4: Generate Daily Sentiment Data for Alerts
    print("\n4. GENERATING DAILY SENTIMENT DATA...")
    df_sentiment['date'] = pd.to_datetime(df_sentiment['publishedAt'], errors='coerce')
    daily_sentiment = df_sentiment.groupby(df_sentiment['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'avg_sentiment']
    
    # Save daily sentiment data
    daily_sentiment.to_csv("daily_sentiment.csv", index=False)
    print("Saved daily sentiment data to: daily_sentiment.csv")
    
    # Step 5: Forecasting
    print("\n5. GENERATING SENTIMENT FORECAST...")
    try:
        forecasts, forecast_df, daily_data = forecast_sentiment()
        if forecasts:
            print("Forecasting completed successfully!")
            # Save forecast results
            if forecast_df is not None:
                forecast_df.to_csv("sentiment_forecast.csv", index=False)
                print("Saved forecast data to: sentiment_forecast.csv")
        else:
            print("Forecasting failed, but continuing with alerts...")
            forecast_df = None
    except Exception as e:
        print(f"Forecasting error: {e}")
        forecast_df = None
    
    # Step 6: Check Alerts
    print("\n6. CHECKING FOR ALERTS...")
    check_alerts(daily_sentiment, forecast_df=forecast_df)
    
    # Step 7: Generate Dashboard
    print("\n7. GENERATING STRATEGIC INTELLIGENCE DASHBOARD...")
    try:
        # Use non-interactive backend for matplotlib
        import matplotlib
        matplotlib.use('Agg')  # This prevents the dashboard from hanging
        import matplotlib.pyplot as plt
        
        from dashboard import generate_full_dashboard
        print("  - Starting dashboard generation...")
        generate_full_dashboard()
        print("  - Dashboard figures created, closing plots...")
        
        # Force close all figures after dashboard generation
        plt.close('all')
        print("Dashboard generation completed successfully!")
    except Exception as e:
        print(f"Dashboard generation error: {e}")
        print("Continuing without dashboard...")
    
    # Step 8: Generate Additional Visualizations
    print("\n8. GENERATING ADDITIONAL VISUALIZATIONS...")
    try:
        create_competitor_analysis(df_sentiment)
        print("Competitor analysis saved: competitor_analysis.png")
        
        create_trend_evolution_analysis(df_sentiment)
        print("Trend analysis saved: trend_evolution.png")
        
        create_alert_history_dashboard(df_sentiment)
        print("Alert history saved: alert_history.png")
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Continuing without additional visualizations...")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final dataset: {len(df_sentiment)} records")
    print("\nGenerated files:")
    print("industry_insights_clean.csv (raw collected data)")
    print("preprocessed.csv (cleaned data)")
    print("industry_insights_with_financial_sentiment.csv (sentiment analysis)")
    print("daily_sentiment.csv (aggregated daily sentiment)")
    if forecasts:
        print("sentiment_forecast.csv (forecast data)")
        print("prophet_forecast.png (forecast visualization)")
        print("prophet_components.png (forecast components)")
    print("dashboard_overview.png (main dashboard)")
    print("dashboard_forecast.png (forecast dashboard)")
    print("competitor_analysis.png (source comparison)")
    print("trend_evolution.png (trend analysis)")
    print("alert_history.png (alert timeline)")

def run_data_collection_only():
    """Run only data collection step"""
    print("Running data collection only...")
    df = collect_all_data(QUERY)
    if not df.empty:
        df.to_csv("industry_insights_clean.csv", index=False)
        print(f"Data collection completed. Collected {len(df)} records.")
        print("Saved to: industry_insights_clean.csv")
    else:
        print("No data collected.")

def run_preprocessing_only():
    """Run only preprocessing step"""
    print("Running preprocessing only...")
    df = clean_and_preprocess_data()
    if df is not None and not df.empty:
        print(f"Preprocessing completed. Shape: {df.shape}")
        print("Saved to: preprocessed.csv")
    else:
        print("Preprocessing failed.")

def run_sentiment_analysis_only():
    """Run only sentiment analysis step"""
    print("Running sentiment analysis only...")
    df = analyze_sentiment_with_finbert()
    if df is not None and not df.empty:
        print(f"Sentiment analysis completed. Shape: {df.shape}")
        print("Saved to: industry_insights_with_financial_sentiment.csv")
    else:
        print("Sentiment analysis failed.")

def run_forecasting_only():
    """Run only forecasting step"""
    print("Running forecasting only...")
    forecasts, forecast_df, daily_data = forecast_sentiment()
    if forecasts:
        print("✓ Forecasting completed successfully!")
        if forecast_df is not None:
            forecast_df.to_csv("sentiment_forecast.csv", index=False)
            print("Saved forecast data to: sentiment_forecast.csv")
    else:
        print("Forecasting failed.")

def run_alerts_only():
    """Run only alerts step"""
    print("Running alerts only...")
    try:
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment']
        
        # Save daily sentiment data
        daily_sentiment.to_csv("daily_sentiment.csv", index=False)
        print("Generated daily sentiment data")
        
        check_alerts(daily_sentiment)
    except FileNotFoundError:
        print("Sentiment analysis file not found. Please run the pipeline first.")

def run_dashboard_only():
    """Run only dashboard generation"""
    print("Running dashboard generation only...")
    try:
        # Use non-interactive backend for matplotlib
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        from dashboard import generate_full_dashboard
        print("  - Starting dashboard generation...")
        generate_full_dashboard()
        print("  - Dashboard function completed")
        
        # Force close all figures
        plt.close('all')
        print("Dashboard generation completed!")
    except Exception as e:
        print(f"Dashboard generation failed: {e}")
        import traceback
        traceback.print_exc()

def run_competitor_analysis():
    """Run competitor/source analysis"""
    print("Running competitor analysis...")
    try:
        from visualization_utils import create_competitor_analysis
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        create_competitor_analysis(df)
        print("Competitor analysis completed!")
        print("Saved to: competitor_analysis.png")
    except FileNotFoundError:
        print("Sentiment analysis file not found. Please run the pipeline first.")
    except Exception as e:
        print(f"Competitor analysis failed: {e}")

def run_trend_analysis():
    """Run trend evolution analysis"""
    print("Running trend evolution analysis...")
    try:
        from visualization_utils import create_trend_evolution_analysis
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        create_trend_evolution_analysis(df)
        print("Trend analysis completed!")
        print("Saved to: trend_evolution.png")
    except FileNotFoundError:
        print("Sentiment analysis file not found. Please run the pipeline first.")
    except Exception as e:
        print(f"Trend analysis failed: {e}")

def run_alert_history():
    """Run alert history analysis"""
    print("Running alert history analysis...")
    try:
        from visualization_utils import create_alert_history_dashboard
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        create_alert_history_dashboard(df)
        print("Alert history analysis completed!")
        print("Saved to: alert_history.png")
    except FileNotFoundError:
        print("Sentiment analysis file not found. Please run the pipeline first.")
    except Exception as e:
        print(f"Alert history analysis failed: {e}")

if __name__ == "__main__":
    # Command line argument handling
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "collect":
            run_data_collection_only()
        elif command == "preprocess":
            run_preprocessing_only()
        elif command == "sentiment":
            run_sentiment_analysis_only()
        elif command == "forecast":
            run_forecasting_only()
        elif command == "alerts":
            run_alerts_only()
        elif command == "dashboard":
            run_dashboard_only()
        elif command == "competitor":
            run_competitor_analysis()
        elif command == "trends":
            run_trend_analysis()
        elif command == "history":
            run_alert_history()
        elif command == "full":
            run_full_pipeline()
        else:
            print("Usage: python main.py [command]")
            print("\nAvailable commands:")
            print("  collect    - Run data collection only")
            print("  preprocess - Run preprocessing only")
            print("  sentiment  - Run sentiment analysis only")
            print("  forecast   - Run forecasting only")
            print("  alerts     - Run alerts only")
            print("  dashboard  - Run dashboard generation only")
            print("  competitor - Run competitor/source analysis")
            print("  trends     - Run trend evolution analysis")
            print("  history    - Run alert history analysis")
            print("  full       - Run complete pipeline (default)")
    else:
        # Default: run full pipeline
        run_full_pipeline()

    # Force cleanup and exit
    import matplotlib.pyplot as plt
    plt.close('all')
    
    print("Pipeline finished - exiting now")
    sys.exit(0)