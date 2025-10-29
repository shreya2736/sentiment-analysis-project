import os
import pandas as pd
import ast
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from config import SLACK_BOT_TOKEN, SLACK_CHANNEL
import plotly.graph_objects as go
from datetime import datetime, timedelta

slack_client = WebClient(token=SLACK_BOT_TOKEN)
CHANNEL = SLACK_CHANNEL

def send_slack_alert(message: str, attachment=None):
    """Send a single alert message to Slack with optional attachment"""
    try:
        if attachment:
            slack_client.chat_postMessage(
                channel=CHANNEL, 
                text=message,
                attachments=[attachment]
            )
        else:
            slack_client.chat_postMessage(channel=CHANNEL, text=message)
        print("✅ Alert sent:", message)
    except SlackApiError as e:
        print("❌ Slack error:", e.response['error'])

def create_alert_visualization(df, alert_type, alert_data):
    """Create interactive visualization for alerts"""
    fig = go.Figure()
    
    if alert_type == "negative_sentiment":
        # Create negative sentiment alert visualization
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['avg_sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='blue', width=2)
        ))
        
        # Highlight negative threshold
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", 
                     annotation_text="Negative Threshold")
        
        # Mark alert points
        alert_points = df[df['avg_sentiment'] <= -0.5]
        if not alert_points.empty:
            fig.add_trace(go.Scatter(
                x=alert_points['date'],
                y=alert_points['avg_sentiment'],
                mode='markers',
                name='Negative Alerts',
                marker=dict(color='red', size=10, symbol='diamond')
            ))
    
    elif alert_type == "positive_surge":
        # Create positive surge alert visualization
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['avg_sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='blue', width=2)
        ))
        
        # Highlight positive threshold
        fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                     annotation_text="Positive Threshold")
        
        # Mark alert points
        alert_points = df[df['avg_sentiment'] >= 0.7]
        if not alert_points.empty:
            fig.add_trace(go.Scatter(
                x=alert_points['date'],
                y=alert_points['avg_sentiment'],
                mode='markers',
                name='Positive Surges',
                marker=dict(color='green', size=10, symbol='star')
            ))
    
    elif alert_type == "trend_reversal":
        # Create trend reversal visualization
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['avg_sentiment'],
            mode='lines+markers',
            name='Sentiment',
            line=dict(color='blue', width=2)
        ))
        
        # Add trend lines or other relevant markers
    
    fig.update_layout(
        title=f"Alert: {alert_type.replace('_', ' ').title()}",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=400
    )
    
    # Save visualization
    filename = f"alert_{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    fig.write_html(filename)
    return filename

def check_alerts(df: pd.DataFrame,
                 neg_threshold=-0.5,
                 pos_threshold=0.7,
                 keyword_surge_thresh=5,
                 trend_window=5,
                 forecast_df: pd.DataFrame = None):
    
    alerts = []
    alert_details = []

    # Last-day threshold check
    last_sent = df['avg_sentiment'].iloc[-1]
    last_date = df['date'].iloc[-1]
    
    if last_sent <= neg_threshold:
        alert_msg = f"🚨 NEGATIVE SENTIMENT ALERT\nDate: {last_date}\nSentiment: {last_sent:.3f}\nThreshold: {neg_threshold}"
        alerts.append(alert_msg)
        alert_details.append({
            'type': 'negative_sentiment',
            'date': last_date,
            'value': last_sent,
            'threshold': neg_threshold
        })
        
    elif last_sent >= pos_threshold:
        alert_msg = f"🚀 POSITIVE SENTIMENT SURGE\nDate: {last_date}\nSentiment: {last_sent:.3f}\nThreshold: {pos_threshold}"
        alerts.append(alert_msg)
        alert_details.append({
            'type': 'positive_surge', 
            'date': last_date,
            'value': last_sent,
            'threshold': pos_threshold
        })

    # Trend detection - historical trend change
    if len(df) >= trend_window * 2:
        recent_avg = df['avg_sentiment'].iloc[-trend_window:].mean()
        prev_avg = df['avg_sentiment'].iloc[-2*trend_window:-trend_window].mean()

        if prev_avg < 0 and recent_avg > 0:
            alert_msg = f"🔄 TREND REVERSAL DETECTED: Negative → Positive\nPrevious {trend_window}-day avg: {prev_avg:.3f}\nRecent {trend_window}-day avg: {recent_avg:.3f}"
            alerts.append(alert_msg)
            alert_details.append({
                'type': 'trend_reversal',
                'from': 'negative',
                'to': 'positive',
                'previous_avg': prev_avg,
                'recent_avg': recent_avg
            })
            
        elif prev_avg > 0 and recent_avg < 0:
            alert_msg = f"🔄 TREND REVERSAL DETECTED: Positive → Negative\nPrevious {trend_window}-day avg: {prev_avg:.3f}\nRecent {trend_window}-day avg: {recent_avg:.3f}"
            alerts.append(alert_msg)
            alert_details.append({
                'type': 'trend_reversal',
                'from': 'positive', 
                'to': 'negative',
                'previous_avg': prev_avg,
                'recent_avg': recent_avg
            })

    # Keyword surge detection (if keywords column exists)
    if 'keywords' in df.columns:
        try:
            last_keywords = df['keywords'].iloc[-1]
            if isinstance(last_keywords, str):
                last_keywords = ast.literal_eval(last_keywords)
            
            if isinstance(last_keywords, dict):
                for keyword, count in last_keywords.items():
                    if count > keyword_surge_thresh:
                        alert_msg = f"🔥 KEYWORD SURGE DETECTED\nKeyword: '{keyword}'\nMentions: {count}\nThreshold: {keyword_surge_thresh}"
                        alerts.append(alert_msg)
                        alert_details.append({
                            'type': 'keyword_surge',
                            'keyword': keyword,
                            'count': count,
                            'threshold': keyword_surge_thresh
                        })
        except:
            pass  # Skip keyword analysis if there's an error

    # Forecast-based alerts
    if forecast_df is not None and not forecast_df.empty:
        forecast_avg = forecast_df['sentiment'].mean()
        hist_avg = df['avg_sentiment'].tail(7).mean()

        diff = forecast_avg - hist_avg
        if diff > 0.05:
            alert_msg = f"📈 FORECAST: Positive Trend Expected\nChange: +{diff:.3f}\nHistorical Avg: {hist_avg:.3f}\nForecast Avg: {forecast_avg:.3f}"
            alerts.append(alert_msg)
            alert_details.append({
                'type': 'positive_forecast',
                'change': diff,
                'historical_avg': hist_avg,
                'forecast_avg': forecast_avg
            })
            
        elif diff < -0.05:
            alert_msg = f"📉 FORECAST: Negative Trend Expected\nChange: {diff:.3f}\nHistorical Avg: {hist_avg:.3f}\nForecast Avg: {forecast_avg:.3f}"
            alerts.append(alert_msg)
            alert_details.append({
                'type': 'negative_forecast',
                'change': diff,
                'historical_avg': hist_avg, 
                'forecast_avg': forecast_avg
            })

    # Volatility alert
    recent_volatility = df['avg_sentiment'].tail(7).std()
    if recent_volatility > 0.5:
        alert_msg = f"⚡ HIGH VOLATILITY DETECTED\nRecent 7-day volatility: {recent_volatility:.3f}\nThreshold: 0.5"
        alerts.append(alert_msg)
        alert_details.append({
            'type': 'high_volatility',
            'volatility': recent_volatility,
            'threshold': 0.5
        })

    # Send or log alerts
    if alerts:
        print(f"🚨 Found {len(alerts)} alerts to send")
        
        for i, alert in enumerate(alerts):
            print(f"Sending alert {i+1}/{len(alerts)}")
            
            # Create visualization for major alerts
            if i < len(alert_details):
                alert_detail = alert_details[i]
                if alert_detail['type'] in ['negative_sentiment', 'positive_surge', 'trend_reversal']:
                    try:
                        viz_file = create_alert_visualization(df, alert_detail['type'], alert_detail)
                        # In a real implementation, you might upload this file or include it differently
                        print(f"Created visualization: {viz_file}")
                    except Exception as e:
                        print(f"Could not create visualization: {e}")
            
            # Send Slack alert
            send_slack_alert(alert)
            
        # Create summary alert
        summary_msg = f"📊 ALERT SUMMARY\nTotal Alerts: {len(alerts)}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_slack_alert(summary_msg)
        
    else:
        print("✅ No alerts triggered.")
        # Send all-clear message (optional)
        # send_slack_alert("✅ No significant alerts detected. System operating normally.")

def save_alert_history(alerts, filename="alert_history.csv"):
    """Save alert history to CSV file"""
    try:
        alert_data = []
        for alert in alerts:
            alert_data.append({
                'timestamp': datetime.now(),
                'alert_message': alert,
                'alert_type': 'auto_detected'
            })
        
        alert_df = pd.DataFrame(alert_data)
        
        # Append to existing file or create new
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, alert_df], ignore_index=True)
            combined_df.to_csv(filename, index=False)
        else:
            alert_df.to_csv(filename, index=False)
            
        print(f"✅ Alert history saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving alert history: {e}")

if __name__ == "__main__":
    try:
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        # Convert to daily aggregated format
        df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment']
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        # Try to load forecast data
        forecast_df = None
        if os.path.exists("sentiment_forecast.csv"):
            forecast_df = pd.read_csv("sentiment_forecast.csv")
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        print("🔍 Checking for alerts...")
        check_alerts(daily_sentiment, forecast_df=forecast_df)
        
    except FileNotFoundError:
        print("❌ Please run the full pipeline first to generate the sentiment data file.")
    except Exception as e:
        print(f"❌ Error in alert system: {e}")