"""
Additional visualization utilities for the Strategic Intelligence Dashboard
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

def create_competitor_analysis(df, save_path="competitor_analysis.png"):
    """Create competitor/source comparison analysis"""
    if df.empty:
        print("No data available for competitor analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Competitor/Source Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Sentiment by source over time
    ax1 = axes[0, 0]
    plot_sentiment_by_source_timeline(df, ax1)
    
    # 2. Source market share (article volume)
    ax2 = axes[0, 1]
    plot_source_market_share(df, ax2)
    
    # 3. Source sentiment heatmap
    ax3 = axes[1, 0]
    plot_source_sentiment_heatmap(df, ax3)
    
    # 4. Source performance ranking
    ax4 = axes[1, 1]
    plot_source_performance_ranking(df, ax4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Competitor analysis saved to {save_path}")

def plot_sentiment_by_source_timeline(df, ax):
    """Plot sentiment timeline for different sources"""
    # Prepare data
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Get top 5 sources by volume
    top_sources = df['source'].value_counts().head(5).index
    
    for source in top_sources:
        source_data = df[df['source'] == source]
        daily_sentiment = source_data.groupby(source_data['date'].dt.date)['sentiment_score'].mean()
        
        if len(daily_sentiment) >= 3:  # Only plot if enough data points
            ax.plot(daily_sentiment.index, daily_sentiment.values, 
                   marker='o', label=source, linewidth=2, alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Sentiment Timeline by Source (Top 5)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Sentiment Score')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

def plot_source_market_share(df, ax):
    """Plot market share by source (article volume)"""
    source_counts = df['source'].value_counts().head(8)
    
    # Create pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
    wedges, texts, autotexts = ax.pie(source_counts.values, labels=source_counts.index, 
                                     autopct='%1.1f%%', colors=colors, startangle=90)
    
    # Improve text formatting
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title('Market Share by Source (Article Volume)')

def plot_source_sentiment_heatmap(df, ax):
    """Create heatmap of sentiment by source and time period"""
    # Prepare data - weekly aggregation
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['week'] = df['date'].dt.to_period('W')
    
    # Get sources with sufficient data
    source_counts = df['source'].value_counts()
    valid_sources = source_counts[source_counts >= 10].index[:6]  # Top 6 sources with 10+ articles
    
    if len(valid_sources) == 0:
        ax.text(0.5, 0.5, 'Insufficient data\nfor heatmap', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Source Sentiment Heatmap')
        return
    
    # Create pivot table
    heatmap_data = df[df['source'].isin(valid_sources)].groupby(['source', 'week'])['sentiment_score'].mean().unstack(fill_value=0)
    
    if not heatmap_data.empty:
        # Create heatmap
        sns.heatmap(heatmap_data, cmap='RdYlGn', center=0, ax=ax, 
                   cbar_kws={'label': 'Sentiment Score'}, annot=True, fmt='.2f')
        ax.set_title('Source Sentiment Heatmap (Weekly)')
        ax.set_xlabel('Week')
        ax.set_ylabel('Source')
    else:
        ax.text(0.5, 0.5, 'No data available\nfor heatmap', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)

def plot_source_performance_ranking(df, ax):
    """Rank sources by performance metrics"""
    # Calculate performance metrics for each source
    source_metrics = df.groupby('source').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()
    
    source_metrics.columns = ['source', 'avg_sentiment', 'sentiment_volatility', 'article_count']
    
    # Filter sources with minimum articles
    source_metrics = source_metrics[source_metrics['article_count'] >= 5]
    
    if source_metrics.empty:
        ax.text(0.5, 0.5, 'Insufficient data\nfor ranking', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Source Performance Ranking')
        return
    
    # Calculate composite performance score (higher sentiment, lower volatility = better)
    source_metrics['performance_score'] = source_metrics['avg_sentiment'] - (source_metrics['sentiment_volatility'] * 0.5)
    source_metrics = source_metrics.sort_values('performance_score', ascending=True)
    
    # Create horizontal bar chart
    colors = ['red' if score < -0.1 else 'green' if score > 0.1 else 'gray' 
              for score in source_metrics['performance_score']]
    
    bars = ax.barh(source_metrics['source'], source_metrics['performance_score'], 
                   color=colors, alpha=0.7)
    
    # Add performance score labels
    for bar, score in zip(bars, source_metrics['performance_score']):
        width = bar.get_width()
        ax.text(width + 0.01 if width >= 0 else width - 0.01, 
               bar.get_y() + bar.get_height()/2,
               f'{score:.3f}', ha='left' if width >= 0 else 'right', 
               va='center', fontweight='bold')
    
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.set_title('Source Performance Ranking\n(Sentiment - 0.5×Volatility)')
    ax.set_xlabel('Performance Score')
    ax.grid(True, alpha=0.3, axis='x')

def create_trend_evolution_analysis(df, save_path="trend_evolution.png"):
    """Create trend evolution analysis over time"""
    if df.empty:
        print("No data available for trend analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trend Evolution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rolling sentiment trends
    ax1 = axes[0, 0]
    plot_rolling_trends(df, ax1)
    
    # 2. Sentiment momentum analysis
    ax2 = axes[0, 1]
    plot_sentiment_momentum(df, ax2)
    
    # 3. Volatility trends
    ax3 = axes[1, 0]
    plot_volatility_trends(df, ax3)
    
    # 4. Trend strength indicators
    ax4 = axes[1, 1]
    plot_trend_strength(df, ax4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Trend evolution analysis saved to {save_path}")

def plot_rolling_trends(df, ax):
    """Plot rolling average trends with multiple time windows"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.sort_values('date')
    
    if len(daily_sentiment) < 7:
        ax.text(0.5, 0.5, 'Insufficient data\nfor trend analysis', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Rolling Sentiment Trends')
        return
    
    # Plot daily data
    ax.plot(daily_sentiment['date'], daily_sentiment['sentiment'], 
           color='lightgray', alpha=0.5, label='Daily', linewidth=1)
    
    # Plot rolling averages
    if len(daily_sentiment) >= 3:
        ma_3 = daily_sentiment['sentiment'].rolling(window=3, center=True).mean()
        ax.plot(daily_sentiment['date'], ma_3, color='blue', linewidth=2, label='3-day MA')
    
    if len(daily_sentiment) >= 7:
        ma_7 = daily_sentiment['sentiment'].rolling(window=7, center=True).mean()
        ax.plot(daily_sentiment['date'], ma_7, color='red', linewidth=2, label='7-day MA')
    
    if len(daily_sentiment) >= 14:
        ma_14 = daily_sentiment['sentiment'].rolling(window=14, center=True).mean()
        ax.plot(daily_sentiment['date'], ma_14, color='green', linewidth=2, label='14-day MA')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Rolling Sentiment Trends')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

def plot_sentiment_momentum(df, ax):
    """Plot sentiment momentum (rate of change)"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.sort_values('date')
    
    if len(daily_sentiment) < 3:
        ax.text(0.5, 0.5, 'Insufficient data\nfor momentum analysis', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Sentiment Momentum')
        return
    
    # Calculate momentum (daily change)
    daily_sentiment['momentum'] = daily_sentiment['sentiment'].diff()
    
    # Create momentum chart
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in daily_sentiment['momentum']]
    ax.bar(daily_sentiment['date'], daily_sentiment['momentum'], color=colors, alpha=0.7)
    
    # Add trend line for momentum
    valid_momentum = daily_sentiment['momentum'].dropna()
    if len(valid_momentum) >= 3:
        z = np.polyfit(range(len(valid_momentum)), valid_momentum.values, 1)
        trend_line = np.poly1d(z)(range(len(valid_momentum)))
        ax.plot(daily_sentiment['date'].iloc[1:], trend_line, 
               color='black', linewidth=2, linestyle='--', label='Momentum Trend')
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_title('Sentiment Momentum (Daily Change)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Momentum (Δ Sentiment)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

def plot_volatility_trends(df, ax):
    """Plot sentiment volatility over time"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    
    # Calculate rolling volatility (7-day windows)
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.sort_values('date')
    
    if len(daily_sentiment) < 7:
        ax.text(0.5, 0.5, 'Insufficient data\nfor volatility analysis', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Sentiment Volatility Trends')
        return
    
    # Calculate rolling standard deviation
    daily_sentiment['volatility'] = daily_sentiment['sentiment'].rolling(window=7).std()
    
    ax.fill_between(daily_sentiment['date'], daily_sentiment['volatility'], 
                   alpha=0.6, color='orange', label='7-day Volatility')
    
    # Add average volatility line
    avg_volatility = daily_sentiment['volatility'].mean()
    ax.axhline(y=avg_volatility, color='red', linestyle='--', 
              label=f'Average: {avg_volatility:.3f}')
    
    # Highlight high volatility periods
    high_vol_threshold = daily_sentiment['volatility'].quantile(0.8)
    high_vol_periods = daily_sentiment[daily_sentiment['volatility'] > high_vol_threshold]
    
    if not high_vol_periods.empty:
        ax.scatter(high_vol_periods['date'], high_vol_periods['volatility'], 
                  color='red', s=50, alpha=0.8, label='High Volatility', zorder=5)
    
    ax.set_title('Sentiment Volatility Trends')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (7-day Std Dev)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

def plot_trend_strength(df, ax):
    """Plot trend strength indicators"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.sort_values('date')
    
    if len(daily_sentiment) < 10:
        ax.text(0.5, 0.5, 'Insufficient data\nfor trend strength', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Trend Strength Indicators')
        return
    
    # Calculate different trend strength metrics
    windows = [3, 7, 14] if len(daily_sentiment) >= 14 else [3, 7] if len(daily_sentiment) >= 7 else [3]
    
    trend_data = []
    for window in windows:
        if len(daily_sentiment) >= window:
            # Calculate trend slope for each window
            slopes = []
            for i in range(window, len(daily_sentiment)):
                y_vals = daily_sentiment['sentiment'].iloc[i-window:i].values
                x_vals = np.arange(len(y_vals))
                if len(y_vals) > 1:
                    slope = np.polyfit(x_vals, y_vals, 1)[0]
                    slopes.append(abs(slope))  # Use absolute value for strength
            
            if slopes:
                avg_slope = np.mean(slopes)
                trend_data.append({'window': f'{window}-day', 'strength': avg_slope})
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        colors = ['green' if strength > 0.05 else 'orange' if strength > 0.02 else 'red' 
                 for strength in trend_df['strength']]
        
        bars = ax.bar(trend_df['window'], trend_df['strength'], color=colors, alpha=0.7)
        
        # Add strength labels
        for bar, strength in zip(bars, trend_df['strength']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{strength:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Trend Strength Indicators\n(Average Absolute Slope)')
        ax.set_xlabel('Time Window')
        ax.set_ylabel('Trend Strength')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add strength interpretation
        max_strength = max(trend_df['strength'])
        if max_strength > 0.1:
            strength_text = "Very Strong Trends"
        elif max_strength > 0.05:
            strength_text = "Strong Trends"
        elif max_strength > 0.02:
            strength_text = "Moderate Trends"
        else:
            strength_text = "Weak Trends"
        
        ax.text(0.5, 0.9, strength_text, transform=ax.transAxes, ha='center',
               fontsize=12, fontweight='bold', 
               color='green' if max_strength > 0.05 else 'orange' if max_strength > 0.02 else 'red')

def create_alert_history_dashboard(df, save_path="alert_history.png"):
    """Create alert history and key events timeline"""
    if df.empty:
        print("No data available for alert history")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Alert History & Key Events Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Alert events timeline
    ax1 = axes[0, 0]
    plot_alert_timeline(df, ax1)
    
    # 2. Alert frequency analysis
    ax2 = axes[0, 1]
    plot_alert_frequency(df, ax2)
    
    # 3. Event impact analysis
    ax3 = axes[1, 0]
    plot_event_impact(df, ax3)
    
    # 4. Alert effectiveness tracking
    ax4 = axes[1, 1]
    plot_alert_effectiveness(df, ax4)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Alert history dashboard saved to {save_path}")

def plot_alert_timeline(df, ax):
    """Plot timeline of alert-worthy events"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # Plot sentiment timeline
    ax.plot(daily_sentiment['date'], daily_sentiment['sentiment'], 
           color='blue', linewidth=1, alpha=0.7)
    
    # Mark alert events
    alert_events = []
    
    # Negative sentiment alerts
    negative_alerts = daily_sentiment[daily_sentiment['sentiment'] <= -0.5]
    for _, row in negative_alerts.iterrows():
        ax.axvline(x=row['date'], color='red', linestyle='--', alpha=0.7)
        ax.scatter(row['date'], row['sentiment'], color='red', s=100, zorder=5)
        alert_events.append(('Negative Alert', row['date'], row['sentiment']))
    
    # Positive surge alerts
    positive_alerts = daily_sentiment[daily_sentiment['sentiment'] >= 0.7]
    for _, row in positive_alerts.iterrows():
        ax.axvline(x=row['date'], color='green', linestyle='--', alpha=0.7)
        ax.scatter(row['date'], row['sentiment'], color='green', s=100, zorder=5)
        alert_events.append(('Positive Surge', row['date'], row['sentiment']))
    
    # Add threshold lines
    ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, label='Negative Threshold')
    ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='Positive Threshold')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Neutral')
    
    ax.set_title(f'Alert Events Timeline ({len(alert_events)} events)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sentiment Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

def plot_alert_frequency(df, ax):
    """Plot alert frequency over time periods"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    
    # Count alerts by type
    negative_alerts = len(daily_sentiment[daily_sentiment['sentiment'] <= -0.5])
    positive_alerts = len(daily_sentiment[daily_sentiment['sentiment'] >= 0.7])
    
    # Calculate weekly alert frequency
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment['week'] = daily_sentiment['date'].dt.to_period('W')
    
    weekly_alerts = daily_sentiment.groupby('week').apply(
        lambda x: len(x[(x['sentiment'] <= -0.5) | (x['sentiment'] >= 0.7)])
    ).reset_index()
    weekly_alerts.columns = ['week', 'alert_count']
    
    if not weekly_alerts.empty:
        ax.bar(range(len(weekly_alerts)), weekly_alerts['alert_count'], 
              alpha=0.7, color='orange')
        ax.set_xlabel('Week')
        ax.set_ylabel('Alert Count')
        ax.set_title(f'Weekly Alert Frequency\n(Total: {negative_alerts + positive_alerts} alerts)')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, f'Alert Summary:\nNegative: {negative_alerts}\nPositive: {positive_alerts}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Alert Frequency Summary')

def plot_event_impact(df, ax):
    """Analyze impact of significant events"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_data = df.groupby(df['date'].dt.date).agg({
        'sentiment_score': 'mean',
        'source': 'count'  # Article volume
    }).reset_index()
    daily_data.columns = ['date', 'sentiment', 'volume']
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    # Find significant events (large sentiment changes)
    daily_data['sentiment_change'] = daily_data['sentiment'].diff().abs()
    significant_events = daily_data[daily_data['sentiment_change'] > daily_data['sentiment_change'].quantile(0.8)]
    
    if not significant_events.empty:
        # Plot sentiment with event markers
        ax.plot(daily_data['date'], daily_data['sentiment'], color='blue', alpha=0.7)
        
        for _, event in significant_events.iterrows():
            ax.scatter(event['date'], event['sentiment'], color='red', s=event['volume']*2, 
                      alpha=0.8, zorder=5)
            ax.annotate(f'Δ{event["sentiment_change"]:.3f}', 
                       (event['date'], event['sentiment']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_title('Significant Events Impact\n(Point size = article volume)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No significant\nevents detected', ha='center', va='center', 
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Event Impact Analysis')

def plot_alert_effectiveness(df, ax):
    """Track alert effectiveness and false positive rates"""
    # This is a conceptual visualization - in a real system you'd track prediction accuracy
    
    alert_types = ['Negative\nSentiment', 'Positive\nSurge', 'High\nVolatility', 'Trend\nReversal']
    
    # Simulated effectiveness data (in real system, this would be calculated from historical performance)
    effectiveness = [0.75, 0.68, 0.82, 0.71]  # Accuracy rates
    false_positives = [0.15, 0.22, 0.12, 0.18]  # False positive rates
    
    x = np.arange(len(alert_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, effectiveness, width, label='Effectiveness', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, false_positives, width, label='False Positive Rate', color='red', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Alert Type')
    ax.set_ylabel('Rate')
    ax.set_title('Alert System Performance\n(Simulated Metrics)')
    ax.set_xticks(x)
    ax.set_xticklabels(alert_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)