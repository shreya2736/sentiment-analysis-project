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
    """Generate sentiment forecasts using Prophet"""
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
        
        # Generate forecast
        forecasts, forecast_df = forecast_with_prophet(daily_sentiment)
        
        if forecasts:
            # Create interactive forecast visualization
            create_interactive_forecast_plot(daily_sentiment, forecast_df)
            
            # Display results
            display_forecast_results(forecasts, daily_sentiment)
            
            return forecasts, forecast_df, daily_sentiment
        else:
            print("❌ Forecasting failed")
            return None, None, None

    except Exception as e:
        print(f"❌ Forecasting error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def forecast_with_prophet(daily_sentiment):
    """Generate forecast using Facebook Prophet"""
    try:
        print("🤖 Generating forecast with Prophet...")
        
        # Prepare data for Prophet
        prophet_df = daily_sentiment[['date', 'avg_sentiment']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()

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
        
        # Add forecast
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
    print("🎯 PROPHET SENTIMENT FORECAST RESULTS")
    print("="*60)

    # Show forecasts with confidence intervals
    print("\n📅 Forecast Results (with 80% confidence intervals):")
    print("-" * 60)
    for forecast in forecasts:
        sentiment_str = f"{forecast['sentiment']:+.3f}"
        ci_str = f"[{forecast['lower_bound']:+.3f}, {forecast['upper_bound']:+.3f}]"
        print(f"{forecast['date']}: {sentiment_str} {ci_str}")

    # Calculate statistics
    recent_avg = historical_data['avg_sentiment'].tail(7).mean()
    forecast_avg = np.mean([f['sentiment'] for f in forecasts])

    print(f"\n📊 Forecast Analysis:")
    print("-" * 20)
    print(f"Recent 7-day average: {recent_avg:+.3f}")
    print(f"Forecasted 7-day average: {forecast_avg:+.3f}")

    # Determine trend
    diff = forecast_avg - recent_avg
    if diff > 0.1:
        trend_direction = "🚀 Very strong positive trend"
    elif diff > 0.05:
        trend_direction = "📈 Strong positive trend"
    elif diff > 0.02:
        trend_direction = "↗️ Moderate positive trend"
    elif diff < -0.1:
        trend_direction = "📉 Very strong negative trend"
    elif diff < -0.05:
        trend_direction = "🔻 Strong negative trend"
    elif diff < -0.02:
        trend_direction = "↘️ Moderate negative trend"
    else:
        trend_direction = "➡️ Neutral trend"

    print(f"Predicted direction: {trend_direction}")
    print(f"Trend change: {diff:+.3f}")

    # Show forecast statistics
    print(f"\n📈 Forecast Statistics:")
    print("-" * 20)
    confidence_widths = [f['upper_bound'] - f['lower_bound'] for f in forecasts]
    print(f"Average confidence interval width: {np.mean(confidence_widths):.3f}")
    print(f"Forecast volatility: {np.std([f['sentiment'] for f in forecasts]):.3f}")

    print(f"\n✅ Forecasting completed successfully!")
    print(f"Generated 7-day sentiment forecast with confidence intervals")

if __name__ == "__main__":
    forecasts, forecast_df, daily_data = forecast_sentiment()
    if forecasts:
        print(f"\n🎉 Next steps:")
        print("- Review the generated plots and interactive forecast")
        print("- Monitor actual sentiment values to validate forecasts")
        print("- Check the interactive_forecast.html for detailed analysis")