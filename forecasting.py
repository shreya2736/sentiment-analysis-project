import numpy as np
from datetime import datetime, timedelta
import warnings
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


warnings.filterwarnings('ignore')

def forecast_sentiment():
    """Generate sentiment forecasts using Prophet with better error handling"""
    print("🔮 Starting sentiment forecasting...")
    
    try:
        # Load and process sentiment data
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        print(f"📥 Loaded {len(df)} records for forecasting")

        # Convert to datetime
        df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')

        # Aggregate daily sentiment
        daily_sentiment = df.groupby(df['date'].dt.date).agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'article_count']
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        daily_sentiment = daily_sentiment.sort_values('date')

        print(f"📊 Prepared {len(daily_sentiment)} days of sentiment data")
        
        # Check if we have enough data for forecasting
        if len(daily_sentiment) < 7:
            print(f"⚠️ Insufficient data for forecasting. Need at least 7 days, have {len(daily_sentiment)}")
            print("💡 Generating simple trend analysis instead...")
            return generate_simple_trend_analysis(daily_sentiment), None, daily_sentiment
        
        # Generate forecast
        forecasts, forecast_df = forecast_with_prophet(daily_sentiment)
        
        if forecasts:
            # Create interactive forecast visualization
            create_interactive_forecast_plot(daily_sentiment, forecast_df)
            
            # Display results
            display_forecast_results(forecasts, daily_sentiment)
            
            return forecasts, forecast_df, daily_sentiment
        else:
            print("❌ Prophet forecasting failed, using fallback...")
            return generate_simple_trend_analysis(daily_sentiment), None, daily_sentiment

    except Exception as e:
        print(f"❌ Forecasting error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def generate_simple_trend_analysis(daily_sentiment):
    """Generate simple trend analysis when Prophet fails"""
    print("📈 Generating simple trend analysis...")
    
    forecasts = []
    
    if len(daily_sentiment) >= 3:
        # Calculate simple moving average
        recent_avg = daily_sentiment['avg_sentiment'].tail(3).mean()
        trend = "stable"
        
        if len(daily_sentiment) >= 5:
            older_avg = daily_sentiment['avg_sentiment'].iloc[-5:-2].mean()
            if recent_avg > older_avg + 0.1:
                trend = "improving"
            elif recent_avg < older_avg - 0.1:
                trend = "declining"
        
        # Create simple forecast
        for i in range(1, 8):
            forecast_date = (daily_sentiment['date'].max() + timedelta(days=i)).strftime('%Y-%m-%d')
            forecasts.append({
                'date': forecast_date,
                'sentiment': recent_avg,
                'lower_bound': recent_avg - 0.2,
                'upper_bound': recent_avg + 0.2,
                'method': 'simple_trend',
                'trend': trend
            })
    
    return forecasts

def forecast_with_prophet(daily_sentiment):
    """Generate forecast using Facebook Prophet with enhanced error handling"""
    try:
        print("🤖 Generating forecast with Prophet...")
        
        # Prepare data for Prophet
        prophet_df = daily_sentiment[['date', 'avg_sentiment']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()

        # Check data quality
        if len(prophet_df) < 7:
            raise ValueError(f"Need at least 7 days of data, got {len(prophet_df)}")
        
        if prophet_df['y'].nunique() <= 1:
            raise ValueError("Insufficient variation in sentiment data")

        print(f"📈 Training on {len(prophet_df)} days of data")

        # Initialize and configure Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.8
        )

        # Fit model
        model.fit(prophet_df)

        # Create future dataframe for 7 days forecast
        future = model.make_future_dataframe(periods=7, freq='D')
        forecast = model.predict(future)

        # Extract forecasted values
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
        forecast_df.columns = ['date', 'sentiment', 'lower_bound', 'upper_bound']
        
        # Convert to list of forecast dictionaries
        forecasts = []
        for _, row in forecast_df.iterrows():
            forecasts.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'sentiment': round(row['sentiment'], 3),
                'lower_bound': round(row['lower_bound'], 3),
                'upper_bound': round(row['upper_bound'], 3),
                'method': 'prophet'
            })

        # Create forecast visualization
        create_prophet_forecast_plot(model, forecast, prophet_df)

        return forecasts, forecast_df

    except Exception as e:
        print(f"❌ Prophet forecasting error: {e}")
        return None, None

# Rest of the functions remain the same...
def create_interactive_forecast_plot(historical_data, forecast_df):
    """Create interactive forecast plot using Plotly"""
    try:
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['avg_sentiment'],
            mode='lines+markers',
            name='Historical Sentiment',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add forecast if available
        if forecast_df is not None and not forecast_df.empty:
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['sentiment'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=6)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='80% Confidence Interval'
            ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Sentiment Forecast',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        # Add neutral line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Save interactive plot
        fig.write_html("interactive_forecast.html")
        print("💾 Interactive forecast plot saved as interactive_forecast.html")
        
    except Exception as e:
        print(f"❌ Interactive plot error: {e}")

def create_prophet_forecast_plot(model, forecast, historical_data):
    """Create Prophet forecast visualization"""
    try:
        # Prophet's built-in plot
        fig1 = model.plot(forecast)
        plt.title('Sentiment Forecast with Prophet')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Neutral (0.0)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('prophet_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot components
        fig2 = model.plot_components(forecast)
        plt.tight_layout()
        plt.savefig('prophet_components.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("💾 Forecast plots saved as prophet_forecast.png and prophet_components.png")

    except Exception as e:
        print(f"❌ Prophet plotting error: {e}")

def display_forecast_results(forecasts, historical_data):
    """Display forecast results"""
    print("\n" + "="*60)
    print("🎯 SENTIMENT FORECAST RESULTS")
    print("="*60)

    if forecasts and len(forecasts) > 0:
        # Show forecasts with confidence intervals
        print("\n📅 Forecast Results:")
        print("-" * 60)
        for forecast in forecasts[:5]:  # Show first 5 days
            sentiment_str = f"{forecast['sentiment']:+.3f}"
            if 'lower_bound' in forecast and 'upper_bound' in forecast:
                ci_str = f"[{forecast['lower_bound']:+.3f}, {forecast['upper_bound']:+.3f}]"
            else:
                ci_str = "[no confidence interval]"
            print(f"{forecast['date']}: {sentiment_str} {ci_str}")

        # Calculate statistics
        if historical_data is not None and len(historical_data) > 0:
            recent_avg = historical_data['avg_sentiment'].tail(min(7, len(historical_data))).mean()
            forecast_avg = np.mean([f['sentiment'] for f in forecasts])
            
            print(f"\n📊 Forecast Analysis:")
            print("-" * 20)
            print(f"Recent average: {recent_avg:+.3f}")
            print(f"Forecasted average: {forecast_avg:+.3f}")

            # Determine trend
            diff = forecast_avg - recent_avg
            if abs(diff) > 0.05:
                trend_direction = "📈 Improving" if diff > 0 else "📉 Declining"
            else:
                trend_direction = "➡️ Stable"
            
            print(f"Predicted trend: {trend_direction}")
            print(f"Method: {forecasts[0].get('method', 'unknown')}")
    else:
        print("❌ No forecast generated - insufficient data")
    
    print(f"\n✅ Forecasting completed!")

if __name__ == "__main__":
    forecasts, forecast_df, daily_data = forecast_sentiment()
    if forecasts:
        print(f"\n🎉 Forecasting completed successfully!")