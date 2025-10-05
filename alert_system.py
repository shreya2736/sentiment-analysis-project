import os
import pandas as pd
import ast
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from config import SLACK_BOT_TOKEN, SLACK_CHANNEL

slack_client = WebClient(token=SLACK_BOT_TOKEN)
CHANNEL = SLACK_CHANNEL

def send_slack_alert(message: str):
    """Send a single alert message to Slack"""
    try:
        slack_client.chat_postMessage(channel=CHANNEL, text=message)
        print("Alert sent:", message)
    except SlackApiError as e:
        print("Slack error:", e.response['error'])

def check_alerts(df: pd.DataFrame,
                 neg_threshold=-0.5,
                 pos_threshold=0.7,
                 keyword_surge_thresh=5,
                 trend_window=5,
                 forecast_df: pd.DataFrame = None):
    
    alerts = []

    #Last-day threshold check
    last_sent = df['avg_sentiment'].iloc[-1]
    if last_sent <= neg_threshold:
        alerts.append(f"Negative sentiment detected: {last_sent:.2f}")
    elif last_sent >= pos_threshold:
        alerts.append(f"Positive sentiment surge: {last_sent:.2f}")

    #Trend detection - historical trend change
    if len(df) >= trend_window * 2:
        recent_avg = df['avg_sentiment'].iloc[-trend_window:].mean()
        prev_avg = df['avg_sentiment'].iloc[-2*trend_window:-trend_window].mean()

        if prev_avg < 0 and recent_avg > 0:
            alerts.append(
                f"Trend reversal detected: Negative → Positive "
                f"(Prev {trend_window}-day avg={prev_avg:.3f}, Recent={recent_avg:.3f})"
            )
        elif prev_avg > 0 and recent_avg < 0:
            alerts.append(
                f"Trend reversal detected: Positive → Negative "
                f"(Prev {trend_window}-day avg={prev_avg:.3f}, Recent={recent_avg:.3f})"
            )

    #Keyword surge detection
    if 'keywords' in df.columns:
        kw = df['keywords'].iloc[-1]
        if isinstance(kw, str):
            try:
                kw = ast.literal_eval(kw)
            except Exception:
                kw = {}
        if isinstance(kw, dict):
            for k, count in kw.items():
                if count > keyword_surge_thresh:
                    alerts.append(f"Keyword surge: '{k}' mentioned {count} times")

    #Prophet forecast trend detection
    if forecast_df is not None and not forecast_df.empty:
        forecast_avg = forecast_df['sentiment'].mean()
        hist_avg = df['avg_sentiment'].tail(7).mean()

        diff = forecast_avg - hist_avg
        if diff > 0.05:
            alerts.append(f"Prophet forecast: Positive trend (Δ={diff:.3f})")
        elif diff < -0.05:
            alerts.append(f"Prophet forecast: Negative trend (Δ={diff:.3f})")

    # Send or log alerts
    if alerts:
        for a in alerts:
            send_slack_alert(a)
    else:
        print("No alerts triggered.")

if __name__ == "__main__":
    
    try:
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        # Convert to daily aggregated format
        df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'avg_sentiment']
        
        check_alerts(daily_sentiment)
    except FileNotFoundError:
        print("Please run the full pipeline first to generate the sentiment data file.")