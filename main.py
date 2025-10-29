"""
Main execution script for the sentiment analysis pipeline
"""
import pandas as pd
import sys
import os
from data_collector import collect_all_data
from data_preprocessor import clean_and_preprocess_data
from sentiment_analyzer import analyze_sentiment_with_finbert
from forecasting import forecast_sentiment
from alert_system import check_alerts
from visualization_utils import create_competitor_analysis, create_trend_evolution_analysis, create_alert_history_dashboard
from config import QUERY

def run_full_pipeline():
    """Run the complete sentiment analysis pipeline"""
    print("="*60)
    print("🚀 STARTING SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    # Step 1: Data Collection
    print("\n1. 📥 COLLECTING DATA...")
    df = collect_all_data(QUERY)
    if df.empty:
        print("❌ No data collected. Exiting pipeline.")
        return
    
    print(f"✅ Collected {len(df)} records from various sources")
    
    # Save collected data
    df.to_csv("industry_insights_clean.csv", index=False)
    print("💾 Saved raw data to: industry_insights_clean.csv")
    
    # Step 2: Data Preprocessing
    print("\n2. 🧹 PREPROCESSING DATA...")
    df_clean = clean_and_preprocess_data()
    if df_clean is None or df_clean.empty:
        print("❌ Preprocessing failed. Exiting pipeline.")
        return
    
    print("✅ Preprocessing completed")
    
    # Step 3: Sentiment Analysis
    print("\n3. 🎯 PERFORMING SENTIMENT ANALYSIS...")
    df_sentiment = analyze_sentiment_with_finbert()
    if df_sentiment is None or df_sentiment.empty:
        print("❌ Sentiment analysis failed. Exiting pipeline.")
        return
    
    print("✅ Sentiment analysis completed")
    
    # Step 4: Generate Daily Sentiment Data for Alerts
    print("\n4. 📊 GENERATING DAILY SENTIMENT DATA...")
    df_sentiment['date'] = pd.to_datetime(df_sentiment['publishedAt'], errors='coerce')
    daily_sentiment = df_sentiment.groupby(df_sentiment['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'avg_sentiment']
    
    # Save daily sentiment data
    daily_sentiment.to_csv("daily_sentiment.csv", index=False)
    print("💾 Saved daily sentiment data to: daily_sentiment.csv")
    
    # Step 5: Forecasting
    print("\n5. 🔮 GENERATING SENTIMENT FORECAST...")
    try:
        forecasts, forecast_df, daily_data = forecast_sentiment()
        if forecasts:
            print("✅ Forecasting completed successfully!")
            # Save forecast results
            if forecast_df is not None:
                forecast_df.to_csv("sentiment_forecast.csv", index=False)
                print("💾 Saved forecast data to: sentiment_forecast.csv")
        else:
            print("⚠️ Forecasting failed, but continuing with alerts...")
            forecast_df = None
    except Exception as e:
        print(f"❌ Forecasting error: {e}")
        forecast_df = None
    
    # Step 6: Check Alerts
    print("\n6. 🚨 CHECKING FOR ALERTS...")
    try:
        check_alerts(daily_sentiment, forecast_df=forecast_df)
        print("✅ Alert check completed")
    except Exception as e:
        print(f"⚠️ Alert system error: {e}")
    
    # Step 7: Generate Visualizations
    print("\n7. 🎨 GENERATING VISUALIZATIONS...")
    try:
        # Use non-interactive backend for matplotlib
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print("  - Generating competitor analysis...")
        create_competitor_analysis(df_sentiment)
        print("✅ Competitor analysis saved: competitor_analysis.html")
        
        print("  - Generating trend evolution analysis...")
        create_trend_evolution_analysis(df_sentiment)
        print("✅ Trend analysis saved: trend_evolution.html")
        
        print("  - Generating alert history dashboard...")
        create_alert_history_dashboard(df_sentiment)
        print("✅ Alert history saved: alert_history.html")
        
        # Generate interactive dashboards
        try:
            from dashboard import StrategicDashboard
            dashboard = StrategicDashboard()
            print("  - Generating interactive overview dashboard...")
            dashboard.create_interactive_overview_dashboard()
            print("✅ Interactive overview dashboard saved: interactive_dashboard_overview.html")
            
            print("  - Generating interactive forecast dashboard...")
            dashboard.create_interactive_forecast_dashboard(forecast_df)
            print("✅ Interactive forecast dashboard saved: interactive_dashboard_forecast.html")
            
        except Exception as e:
            print(f"⚠️ Interactive dashboard generation error: {e}")
        
        # Force close all figures after dashboard generation
        plt.close('all')
        print("✅ Visualization generation completed successfully!")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        print("Continuing without visualizations...")
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 Final dataset: {len(df_sentiment)} records")
    print("\n📁 Generated files:")
    generated_files = [
        "industry_insights_clean.csv (raw collected data)",
        "preprocessed.csv (cleaned data)", 
        "industry_insights_with_financial_sentiment.csv (sentiment analysis)",
        "daily_sentiment.csv (aggregated daily sentiment)",
        "competitor_analysis.html (interactive source comparison)",
        "trend_evolution.html (interactive trend analysis)", 
        "alert_history.html (interactive alert timeline)"
    ]
    
    if forecasts:
        generated_files.extend([
            "sentiment_forecast.csv (forecast data)",
            "prophet_forecast.png (forecast visualization)",
            "prophet_components.png (forecast components)",
            "interactive_forecast.html (interactive forecast)",
            "interactive_dashboard_overview.html (interactive overview)",
            "interactive_dashboard_forecast.html (interactive forecast dashboard)"
        ])
    
    for file_info in generated_files:
        print(f"  • {file_info}")

def run_data_collection_only():
    """Run only data collection step"""
    print("🔄 Running data collection only...")
    df = collect_all_data(QUERY)
    if not df.empty:
        df.to_csv("industry_insights_clean.csv", index=False)
        print(f"✅ Data collection completed. Collected {len(df)} records.")
        print("💾 Saved to: industry_insights_clean.csv")
    else:
        print("❌ No data collected.")

def run_preprocessing_only():
    """Run only preprocessing step"""
    print("🧹 Running preprocessing only...")
    df = clean_and_preprocess_data()
    if df is not None and not df.empty:
        print(f"✅ Preprocessing completed. Shape: {df.shape}")
        print("💾 Saved to: preprocessed.csv")
    else:
        print("❌ Preprocessing failed.")

def run_sentiment_analysis_only():
    """Run only sentiment analysis step"""
    print("🎯 Running sentiment analysis only...")
    df = analyze_sentiment_with_finbert()
    if df is not None and not df.empty:
        print(f"✅ Sentiment analysis completed. Shape: {df.shape}")
        print("💾 Saved to: industry_insights_with_financial_sentiment.csv")
    else:
        print("❌ Sentiment analysis failed.")

def run_forecasting_only():
    """Run only forecasting step"""
    print("🔮 Running forecasting only...")
    forecasts, forecast_df, daily_data = forecast_sentiment()
    if forecasts:
        print("✅ Forecasting completed successfully!")
        if forecast_df is not None:
            forecast_df.to_csv("sentiment_forecast.csv", index=False)
            print("💾 Saved forecast data to: sentiment_forecast.csv")
    else:
        print("❌ Forecasting failed.")

def run_alerts_only():
    """Run only alerts step"""
    print("🚨 Running alerts only...")
    try:
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment']
        
        # Save daily sentiment data
        daily_sentiment.to_csv("daily_sentiment.csv", index=False)
        print("✅ Generated daily sentiment data")
        
        check_alerts(daily_sentiment)
    except FileNotFoundError:
        print("❌ Sentiment analysis file not found. Please run the pipeline first.")

def run_visualizations_only():
    """Run only visualizations generation"""
    print("🎨 Running visualizations only...")
    try:
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        
        # Generate interactive visualizations
        create_competitor_analysis(df)
        print("✅ Competitor analysis saved: competitor_analysis.html")
        
        create_trend_evolution_analysis(df)
        print("✅ Trend analysis saved: trend_evolution.html")
        
        create_alert_history_dashboard(df)
        print("✅ Alert history saved: alert_history.html")
        
        # Generate interactive dashboards
        from dashboard import StrategicDashboard
        dashboard = StrategicDashboard()
        dashboard.create_interactive_overview_dashboard()
        print("✅ Interactive overview dashboard saved: interactive_dashboard_overview.html")
        
        # Try to load forecast data for forecast dashboard
        forecast_df = None
        if os.path.exists("sentiment_forecast.csv"):
            forecast_df = pd.read_csv("sentiment_forecast.csv")
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        dashboard.create_interactive_forecast_dashboard(forecast_df)
        print("✅ Interactive forecast dashboard saved: interactive_dashboard_forecast.html")
        
        print("🎉 All visualizations generated successfully!")
        
    except FileNotFoundError:
        print("❌ Sentiment analysis file not found. Please run the pipeline first.")
    except Exception as e:
        print(f"❌ Visualization generation failed: {e}")

def run_competitor_analysis():
    """Run competitor/source analysis"""
    print("🏢 Running competitor analysis...")
    try:
        from visualization_utils import create_competitor_analysis
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        create_competitor_analysis(df)
        print("✅ Competitor analysis completed!")
        print("💾 Saved to: competitor_analysis.html")
    except FileNotFoundError:
        print("❌ Sentiment analysis file not found. Please run the pipeline first.")
    except Exception as e:
        print(f"❌ Competitor analysis failed: {e}")

def run_trend_analysis():
    """Run trend evolution analysis"""
    print("📈 Running trend evolution analysis...")
    try:
        from visualization_utils import create_trend_evolution_analysis
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        create_trend_evolution_analysis(df)
        print("✅ Trend analysis completed!")
        print("💾 Saved to: trend_evolution.html")
    except FileNotFoundError:
        print("❌ Sentiment analysis file not found. Please run the pipeline first.")
    except Exception as e:
        print(f"❌ Trend analysis failed: {e}")

def run_alert_history():
    """Run alert history analysis"""
    print("🚨 Running alert history analysis...")
    try:
        from visualization_utils import create_alert_history_dashboard
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        create_alert_history_dashboard(df)
        print("✅ Alert history analysis completed!")
        print("💾 Saved to: alert_history.html")
    except FileNotFoundError:
        print("❌ Sentiment analysis file not found. Please run the pipeline first.")
    except Exception as e:
        print(f"❌ Alert history analysis failed: {e}")

def show_help():
    """Show help information"""
    print("""
🤖 Strategic Intelligence Dashboard - Help

Available commands:
  collect      - Run data collection only
  preprocess   - Run preprocessing only  
  sentiment    - Run sentiment analysis only
  forecast     - Run forecasting only
  alerts       - Run alerts only
  visualizations - Run visualizations only
  competitor   - Run competitor/source analysis
  trends       - Run trend evolution analysis
  history      - Run alert history analysis
  full         - Run complete pipeline (default)

Examples:
  python main.py full          # Run complete pipeline
  python main.py collect       # Collect new data only
  python main.py sentiment     # Analyze sentiment on existing data
  python main.py visualizations # Generate all visualizations

For the web dashboard:
  streamlit run app.py         # Launch interactive web dashboard
    """)

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
        elif command == "visualizations":
            run_visualizations_only()
        elif command == "competitor":
            run_competitor_analysis()
        elif command == "trends":
            run_trend_analysis()
        elif command == "history":
            run_alert_history()
        elif command == "full":
            run_full_pipeline()
        elif command in ["help", "-h", "--help"]:
            show_help()
        else:
            print(f"❌ Unknown command: {command}")
            show_help()
    else:
        # Default: run full pipeline
        run_full_pipeline()

    # Force cleanup and exit
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except:
        pass
    
    print("\n🎯 Pipeline finished - exiting now")
    sys.exit(0)