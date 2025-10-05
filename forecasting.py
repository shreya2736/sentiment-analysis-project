import numpy as np
from datetime import datetime, timedelta
import warnings
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def forecast_sentiment():
    # Load and process sentiment data
    print("Loading sentiment data...")
    df = pd.read_csv("industry_insights_with_financial_sentiment.csv")

    # Convert to datetime
    if 'publishedAt' in df.columns:
        df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    # Aggregate daily sentiment
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'avg_sentiment']

    print(f"Loaded {len(daily_sentiment)} days of sentiment data")
    print("\nRecent sentiment data:")
    print(daily_sentiment.tail(10))

    def forecast_with_prophet():
        """Generate sentiment forecast using Facebook Prophet"""
        try:
            print("\n===== Prophet Sentiment Forecasting =====")
            print("Preparing data for Prophet...")

            # Prepare data for Prophet
            prophet_df = daily_sentiment.copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

            print(f"Data shape: {prophet_df.shape}")
            print(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")

            # Initialize and configure Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,  # Regularization for flexibility
                seasonality_prior_scale=10.0,  # Seasonality strength
                interval_width=0.8  # Confidence interval width
            )

            print("Fitting Prophet model...")
            model.fit(prophet_df)

            # Create future dataframe for 5 days forecast
            future = model.make_future_dataframe(periods=5, freq='D')
            print(f"Future dataframe shape: {future.shape}")

            print("Generating forecast...")
            forecast = model.predict(future)

            # Extract the forecasted values
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
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

            # Plot the forecast
            plot_prophet_forecast(model, forecast, prophet_df)

            return forecasts, forecast_df

        except Exception as e:
            print(f"Prophet Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def plot_prophet_forecast(model, forecast, historical_data):
        """Plot Prophet forecast results"""
        try:
            fig = model.plot(forecast)
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

            print("Forecast plots saved as 'prophet_forecast.png' and 'prophet_components.png'")

        except Exception as e:
            print(f"Plotting error: {e}")

    def calculate_forecast_metrics(historical_data, forecast_periods=5):
        """Calculate forecast accuracy metrics on recent data"""
        try:
            # Use last 5 days as validation set
            if len(historical_data) > 10:
                train_data = historical_data[:-5]
                test_data = historical_data[-5:]

                # Retrain on training data
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    changepoint_prior_scale=0.05
                )
                model.fit(train_data)

                # Forecast for test period
                future = model.make_future_dataframe(periods=5, freq='D')
                forecast_val = model.predict(future).tail(5)

                # Calculate metrics
                actual = test_data['y'].values
                predicted = forecast_val['yhat'].values

                mae = np.mean(np.abs(actual - predicted))
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100  # Avoid division by zero

                return {
                    'mae': round(mae, 4),
                    'rmse': round(rmse, 4),
                    'mape': round(mape, 2),
                    'correlation': round(np.corrcoef(actual, predicted)[0, 1], 3)
                }
            return None
        except:
            return None

    def display_forecast_results(forecasts, historical_data, metrics=None):
        """Display comprehensive forecast results"""
        print("\n" + "="*60)
        print("PROPHET SENTIMENT FORECAST RESULTS")
        print("="*60)

        # Show forecasts with confidence intervals
        print("\nForecast Results (with 80% confidence intervals):")
        print("-" * 60)
        for forecast in forecasts:
            sentiment_str = f"{forecast['sentiment']:+.3f}"
            ci_str = f"[{forecast['lower_bound']:+.3f}, {forecast['upper_bound']:+.3f}]"
            print(f"{forecast['date']}: {sentiment_str} {ci_str}")

        # Calculate statistics
        recent_avg = historical_data['avg_sentiment'].tail(7).mean()
        forecast_avg = np.mean([f['sentiment'] for f in forecasts])

        print(f"\nForecast Analysis:")
        print("-" * 20)
        print(f"Recent 7-day average: {recent_avg:+.3f}")
        print(f"Forecasted 5-day average: {forecast_avg:+.3f}")

        # Determine trend
        diff = forecast_avg - recent_avg
        if diff > 0.1:
            trend_direction = "Very strong positive trend"
        elif diff > 0.05:
            trend_direction = "Strong positive trend"
        elif diff > 0.02:
            trend_direction = "Moderate positive trend"
        elif diff < -0.1:
            trend_direction = "Very strong negative trend"
        elif diff < -0.05:
            trend_direction = "Strong negative trend"
        elif diff < -0.02:
            trend_direction = "Moderate negative trend"
        else:
            trend_direction = "Neutral trend"

        print(f"Predicted direction: {trend_direction}")

        # Show metrics if available
        if metrics:
            print(f"\nModel Performance Metrics:")
            print("-" * 20)
            print(f"MAE (Mean Absolute Error): {metrics['mae']}")
            print(f"RMSE (Root Mean Square Error): {metrics['rmse']}")
            print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']}%")
            print(f"Correlation: {metrics['correlation']}")

    # Prepare data for Prophet
    prophet_data = daily_sentiment.copy()
    prophet_data.columns = ['ds', 'y']
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])

    # Show data summary
    print(f"\nData Summary:")
    print(f"- Total days: {len(prophet_data)}")
    print(f"- Date range: {prophet_data['ds'].min()} to {prophet_data['ds'].max()}")
    print(f"- Recent 7-day average: {prophet_data['y'].tail(7).mean():.3f}")
    print(f"- Overall average: {prophet_data['y'].mean():.3f}")
    print(f"- Volatility (std): {prophet_data['y'].std():.3f}")

    # Calculate validation metrics
    print("\nCalculating model validation metrics...")
    metrics = calculate_forecast_metrics(prophet_data)

    # Generate forecast
    forecasts, forecast_df = forecast_with_prophet()

    if forecasts:
        # Display results
        display_forecast_results(forecasts, daily_sentiment, metrics)

        # Show forecast statistics
        print(f"\nForecast Statistics:")
        print("-" * 20)
        print(f"Average confidence interval width: {np.mean([f['upper_bound'] - f['lower_bound'] for f in forecasts]):.3f}")
        print(f"Forecast volatility: {np.std([f['sentiment'] for f in forecasts]):.3f}")

        # Check if forecasts are within valid range
        out_of_bounds = sum(1 for f in forecasts if f['sentiment'] < -1 or f['sentiment'] > 1)
        if out_of_bounds > 0:
            print(f"Warning: {out_of_bounds} forecasts outside [-1, 1] range")

        print(f"\nForecasting completed successfully!")
        print(f"Generated 5-day sentiment forecast with confidence intervals")

        return forecasts, forecast_df, daily_sentiment
    else:
        print("Forecasting failed.")
        return None, None, None

if __name__ == "__main__":
    forecasts, forecast_df, daily_data = forecast_sentiment()
    if forecasts:
        print(f"\nNext steps:")
        print("- Review the generated plots: prophet_forecast.png and prophet_components.png")
        print("- Monitor actual sentiment values to validate forecasts")
        print("- Consider adding external regressors (news volume, market data)")
        print("- Adjust model hyperparameters based on validation metrics")