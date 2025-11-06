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

import sys
import io
import pandas as pd
from datetime import datetime

def check_alerts(df: pd.DataFrame,
                 neg_threshold=-0.2,
                 pos_threshold=0.3,
                 keyword_surge_thresh=5,
                 trend_window=5,
                 forecast_df: pd.DataFrame = None):

    # Capture all printed output (for Streamlit later)
    log_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = log_buffer

    alerts = []
    alert_details = []

    print("🚨 CHECKING FOR ALERTS...")

    # --- EXISTING ALERT LOGIC STARTS HERE ---

    last_sent = df['avg_sentiment'].iloc[-1]
    last_date = df['date'].iloc[-1]
    
    # Negative sentiment alert
    if last_sent <= neg_threshold:
        severity = "HIGH" if last_sent <= -0.3 else "MEDIUM" if last_sent <= -0.2 else "LOW"
        alert_msg = f"🚨 NEGATIVE SENTIMENT ALERT ({severity})\nDate: {last_date}\nSentiment: {last_sent:.3f}\nThreshold: {neg_threshold}"
        alerts.append(alert_msg)
        alert_details.append({
            'type': 'negative_sentiment',
            'date': last_date,
            'value': last_sent,
            'threshold': neg_threshold,
            'severity': severity
        })
        
    # Positive sentiment alert
    elif last_sent >= pos_threshold:
        severity = "HIGH" if last_sent >= 0.5 else "MEDIUM" if last_sent >= 0.3 else "LOW"
        alert_msg = f"🚀 POSITIVE SENTIMENT SURGE ({severity})\nDate: {last_date}\nSentiment: {last_sent:.3f}\nThreshold: {pos_threshold}"
        alerts.append(alert_msg)
        alert_details.append({
            'type': 'positive_surge', 
            'date': last_date,
            'value': last_sent,
            'threshold': pos_threshold,
            'severity': severity
        })

    # Mild sentiment alerts
    if -0.15 <= last_sent <= -0.05:
        alert_msg = f"⚠️ MILD NEGATIVE SENTIMENT\nDate: {last_date}\nSentiment: {last_sent:.3f}\nNote: Slightly negative trend detected"
        alerts.append(alert_msg)
        alert_details.append({'type': 'mild_negative', 'date': last_date, 'value': last_sent})
    
    elif 0.05 <= last_sent <= 0.15:
        alert_msg = f"📈 MILD POSITIVE SENTIMENT\nDate: {last_date}\nSentiment: {last_sent:.3f}\nNote: Slightly positive trend detected"
        alerts.append(alert_msg)
        alert_details.append({'type': 'mild_positive', 'date': last_date, 'value': last_sent})

    # Volatility-based alerts
    recent_volatility = df['avg_sentiment'].tail(7).std(ddof=0)
    if recent_volatility > 0.15:
        severity = "HIGH" if recent_volatility > 0.3 else "MEDIUM" if recent_volatility > 0.2 else "LOW"
        alert_msg = f"⚡ VOLATILITY DETECTED ({severity})\nRecent 7-day volatility: {recent_volatility:.3f}\nThreshold: 0.15"
        alerts.append(alert_msg)
        alert_details.append({
            'type': 'volatility',
            'volatility': recent_volatility,
            'threshold': 0.15,
            'severity': severity
        })

    # Trend-based alerts
    if len(df) >= 3:
        recent_trend = df['avg_sentiment'].tail(3).mean() - df['avg_sentiment'].iloc[-4:-1].mean()
        if abs(recent_trend) > 0.1:
            direction = "improving" if recent_trend > 0 else "deteriorating"
            alert_msg = f"📊 TREND CHANGE DETECTED\nDirection: {direction}\nChange: {recent_trend:+.3f}\nCurrent: {last_sent:.3f}"
            alerts.append(alert_msg)
            alert_details.append({
                'type': 'trend_change',
                'direction': direction,
                'change': recent_trend,
                'current': last_sent
            })

    # --- ALERT LOGIC ENDS HERE ---

    if alerts:
        print(f"🚨 Found {len(alerts)} alerts to send")
        for i, alert in enumerate(alerts):
            print(f"Sending alert {i+1}/{len(alerts)}")
            send_slack_alert(alert)
        summary_msg = f"📊 ALERT SUMMARY\nTotal Alerts: {len(alerts)}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_slack_alert(summary_msg)
    else:
        print("✅ No alerts triggered.")
        status_msg = f"✅ SYSTEM STATUS: All Clear\nCurrent Sentiment: {last_sent:.3f}\nRecent Volatility: {recent_volatility:.3f}\nLast Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_slack_alert(status_msg)

    print("✅ Alert check completed")

    # --- SAVE LOG OUTPUT FOR STREAMLIT ---
    sys.stdout = old_stdout
    log_text = log_buffer.getvalue()
    with open("alert_logs.txt", "w", encoding="utf-8") as f:
        f.write(log_text)

    print("✅ Log file updated: alert_logs.txt")

    return alert_details

        
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