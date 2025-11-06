"""
Additional visualization utilities for the Strategic Intelligence Dashboard
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


warnings.filterwarnings('ignore')

def create_competitor_analysis(df, save_path="competitor_analysis.html"):
    """Create interactive competitor/source comparison analysis"""
    if df.empty:
        print("No data available for competitor analysis")
        return
    
    # Create subplots - UPDATED SPECS for bar chart compatibility
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sentiment Timeline by Source (Top 5)',
            'Market Share by Source',
            'Source Sentiment Heatmap',
            'Source Performance Ranking'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],  # Changed from "pie" to regular
            [{"type": "heatmap"}, {"secondary_y": False}]      # Changed from "bar" to regular
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Sentiment by source over time
    plot_sentiment_by_source_timeline_interactive(df, fig, 1, 1)
    
    # 2. Source market share (article volume)
    plot_source_market_share_interactive(df, fig, 1, 2)
    
    # 3. Source sentiment heatmap
    plot_source_sentiment_heatmap_interactive(df, fig, 2, 1)
    
    # 4. Source performance ranking
    plot_source_performance_ranking_interactive(df, fig, 2, 2)
    
    fig.update_layout(
        title_text='Interactive Competitor/Source Analysis Dashboard',
        height=1000,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.write_html(save_path)
    print(f"Interactive competitor analysis saved to {save_path}")
    return fig

def plot_sentiment_by_source_timeline_interactive(df, fig, row, col):
    """Plot interactive sentiment timeline for different sources"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Get top 5 sources by volume
    top_sources = df['source'].value_counts().head(5).index
    
    colors = px.colors.qualitative.Set1
    
    for i, source in enumerate(top_sources):
        source_data = df[df['source'] == source]
        daily_sentiment = source_data.groupby(source_data['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'sentiment']
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        if len(daily_sentiment) >= 3:
            fig.add_trace(
                go.Scatter(
                    x=daily_sentiment['date'],
                    y=daily_sentiment['sentiment'],
                    mode='lines+markers',
                    name=source,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{source}</b><br>Date: %{{x}}<br>Sentiment: %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Average Sentiment Score", row=row, col=col)

def plot_source_market_share_interactive(df, fig, row, col):
    """Plot interactive market share by source - USING BAR CHART INSTEAD"""
    source_counts = df['source'].value_counts().head(8)
    
    # Use horizontal bar chart instead of pie chart for better subplot compatibility
    fig.add_trace(
        go.Bar(
            x=source_counts.values,
            y=source_counts.index,
            orientation='h',
            marker_color=px.colors.qualitative.Pastel,
            hovertemplate='<b>%{y}</b><br>Articles: %{x}<br>Market Share: %{customdata:.1f}%<extra></extra>',
            customdata=(source_counts.values / source_counts.values.sum() * 100),
            name="Market Share"
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Article Count", row=row, col=col)
    fig.update_yaxes(title_text="Source", row=row, col=col)
def plot_source_sentiment_heatmap_interactive(df, fig, row, col):
    """Create interactive heatmap of sentiment by source and time period"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['week'] = df['date'].dt.to_period('W').astype(str)
    
    # Get sources with sufficient data
    source_counts = df['source'].value_counts()
    valid_sources = source_counts[source_counts >= 5].index[:6]
    
    if len(valid_sources) == 0:
        fig.add_annotation(
            text="Insufficient data for heatmap",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False,
            row=row, col=col
        )
        return
    
    # Create pivot table
    heatmap_data = df[df['source'].isin(valid_sources)].groupby(['source', 'week'])['sentiment_score'].mean().unstack(fill_value=0)
    
    if not heatmap_data.empty:
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                zmid=0,
                hovertemplate='<b>%{y}</b><br>Week: %{x}<br>Sentiment: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Sentiment Score")
            ),
            row=row, col=col
        )
        fig.update_xaxes(title_text="Week", row=row, col=col)
        fig.update_yaxes(title_text="Source", row=row, col=col)

def plot_source_performance_ranking_interactive(df, fig, row, col):
    """Interactive ranking of sources by performance metrics"""
    source_metrics = df.groupby('source').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()
    
    source_metrics.columns = ['source', 'avg_sentiment', 'sentiment_volatility', 'article_count']
    source_metrics = source_metrics[source_metrics['article_count'] >= 5]
    
    if source_metrics.empty:
        fig.add_annotation(
            text="Insufficient data for ranking",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False,
            row=row, col=col
        )
        return
    
    # Calculate composite performance score
    source_metrics['performance_score'] = source_metrics['avg_sentiment'] - (source_metrics['sentiment_volatility'] * 0.5)
    source_metrics = source_metrics.sort_values('performance_score')
    
    # Create color scale based on performance
    colors = []
    for score in source_metrics['performance_score']:
        if score > 0.1:
            colors.append('#28a745')  # Green
        elif score < -0.1:
            colors.append('#dc3545')  # Red
        else:
            colors.append('#ffc107')  # Yellow
    
    fig.add_trace(
        go.Bar(
            x=source_metrics['performance_score'],
            y=source_metrics['source'],
            orientation='h',
            marker_color=colors,
            hovertemplate='<b>%{y}</b><br>Performance: %{x:.3f}<br>Avg Sentiment: %{customdata[0]:.3f}<br>Volatility: %{customdata[1]:.3f}<br>Articles: %{customdata[2]}<extra></extra>',
            customdata=source_metrics[['avg_sentiment', 'sentiment_volatility', 'article_count']].values
        ),
        row=row, col=col
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
    fig.update_xaxes(title_text="Performance Score", row=row, col=col)
    fig.update_yaxes(title_text="Source", row=row, col=col)

def create_trend_evolution_analysis(df, save_path="trend_evolution.html"):
    """Create interactive trend evolution analysis"""
    if df.empty:
        print("No data available for trend analysis")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Rolling Sentiment Trends',
            'Sentiment Momentum Analysis',
            'Volatility Trends',
            'Trend Strength Indicators'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Rolling sentiment trends
    plot_rolling_trends_interactive(df, fig, 1, 1)
    
    # 2. Sentiment momentum analysis
    plot_sentiment_momentum_interactive(df, fig, 1, 2)
    
    # 3. Volatility trends
    plot_volatility_trends_interactive(df, fig, 2, 1)
    
    # 4. Trend strength indicators
    plot_trend_strength_interactive(df, fig, 2, 2)
    
    fig.update_layout(
        title_text='Interactive Trend Evolution Analysis',
        height=1000,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.write_html(save_path)
    print(f"Interactive trend evolution analysis saved to {save_path}")
    return fig

def plot_rolling_trends_interactive(df, fig, row, col):
    """Plot interactive rolling average trends"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.sort_values('date')
    
    if len(daily_sentiment) < 7:
        fig.add_annotation(
            text="Insufficient data for trend analysis",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False,
            row=row, col=col
        )
        return
    
    # Add daily sentiment
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['sentiment'],
            mode='lines',
            name='Daily',
            line=dict(color='lightgray', width=1),
            opacity=0.7,
            hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
        ),
        row=row, col=col
    )
    
    # Add moving averages
    if len(daily_sentiment) >= 3:
        daily_sentiment['ma_3'] = daily_sentiment['sentiment'].rolling(window=3, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['ma_3'],
                mode='lines',
                name='3-day MA',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>3-day MA: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    if len(daily_sentiment) >= 7:
        daily_sentiment['ma_7'] = daily_sentiment['sentiment'].rolling(window=7, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['ma_7'],
                mode='lines',
                name='7-day MA',
                line=dict(color='red', width=3),
                hovertemplate='Date: %{x}<br>7-day MA: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    if len(daily_sentiment) >= 14:
        daily_sentiment['ma_14'] = daily_sentiment['sentiment'].rolling(window=14, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['ma_14'],
                mode='lines',
                name='14-day MA',
                line=dict(color='green', width=3),
                hovertemplate='Date: %{x}<br>14-day MA: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Sentiment Score", row=row, col=col)

def plot_sentiment_momentum_interactive(df, fig, row, col):
    """Plot interactive sentiment momentum"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.sort_values('date')
    
    if len(daily_sentiment) < 3:
        fig.add_annotation(
            text="Insufficient data for momentum analysis",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False,
            row=row, col=col
        )
        return
    
    # Calculate momentum
    daily_sentiment['momentum'] = daily_sentiment['sentiment'].diff()
    daily_sentiment = daily_sentiment.iloc[1:]  # Remove first row with NaN
    
    # Create color array for positive/negative momentum
    colors = ['#28a745' if x > 0 else '#dc3545' for x in daily_sentiment['momentum']]
    
    fig.add_trace(
        go.Bar(
            x=daily_sentiment['date'],
            y=daily_sentiment['momentum'],
            marker_color=colors,
            name='Momentum',
            hovertemplate='Date: %{x}<br>Momentum: %{y:.3f}<extra></extra>'
        ),
        row=row, col=col
    )
    
    # Add trend line
    if len(daily_sentiment) >= 3:
        z = np.polyfit(range(len(daily_sentiment)), daily_sentiment['momentum'].values, 1)
        trend_line = np.poly1d(z)(range(len(daily_sentiment)))
        
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=trend_line,
                mode='lines',
                name='Momentum Trend',
                line=dict(color='black', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Trend: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Momentum (Δ Sentiment)", row=row, col=col)

def plot_volatility_trends_interactive(df, fig, row, col):
    """Plot interactive sentiment volatility trends"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.sort_values('date')
    
    if len(daily_sentiment) < 7:
        fig.add_annotation(
            text="Insufficient data for volatility analysis",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False,
            row=row, col=col
        )
        return
    
    # Calculate rolling volatility
    daily_sentiment['volatility'] = daily_sentiment['sentiment'].rolling(window=7).std()
    
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['volatility'],
            mode='lines',
            name='7-day Volatility',
            line=dict(color='orange', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,165,0,0.3)',
            hovertemplate='Date: %{x}<br>Volatility: %{y:.3f}<extra></extra>'
        ),
        row=row, col=col
    )
    
    # Add average line
    avg_volatility = daily_sentiment['volatility'].mean()
    fig.add_hline(
        y=avg_volatility, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f'Average: {avg_volatility:.3f}',
        row=row, col=col
    )
    
    # Highlight high volatility periods
    high_vol_threshold = daily_sentiment['volatility'].quantile(0.8)
    high_vol_periods = daily_sentiment[daily_sentiment['volatility'] > high_vol_threshold]
    
    if not high_vol_periods.empty:
        fig.add_trace(
            go.Scatter(
                x=high_vol_periods['date'],
                y=high_vol_periods['volatility'],
                mode='markers',
                name='High Volatility',
                marker=dict(color='red', size=8, symbol='diamond'),
                hovertemplate='<b>High Volatility</b><br>Date: %{x}<br>Volatility: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Volatility (7-day Std Dev)", row=row, col=col)

def plot_trend_strength_interactive(df, fig, row, col):
    """Plot interactive trend strength indicators"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment = daily_sentiment.sort_values('date')
    
    if len(daily_sentiment) < 10:
        fig.add_annotation(
            text="Insufficient data for trend strength",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False,
            row=row, col=col
        )
        return
    
    # Calculate trend strength for different windows
    windows = [3, 7, 14] if len(daily_sentiment) >= 14 else [3, 7] if len(daily_sentiment) >= 7 else [3]
    
    trend_data = []
    for window in windows:
        if len(daily_sentiment) >= window:
            slopes = []
            for i in range(window, len(daily_sentiment)):
                y_vals = daily_sentiment['sentiment'].iloc[i-window:i].values
                x_vals = np.arange(len(y_vals))
                if len(y_vals) > 1:
                    slope = np.polyfit(x_vals, y_vals, 1)[0]
                    slopes.append(abs(slope))
            
            if slopes:
                avg_slope = np.mean(slopes)
                trend_data.append({'window': f'{window}-day', 'strength': avg_slope})
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        
        # Color based on strength
        colors = []
        for strength in trend_df['strength']:
            if strength > 0.05:
                colors.append('#28a745')  # Strong
            elif strength > 0.02:
                colors.append('#ffc107')  # Moderate
            else:
                colors.append('#dc3545')  # Weak
        
        fig.add_trace(
            go.Bar(
                x=trend_df['window'],
                y=trend_df['strength'],
                marker_color=colors,
                hovertemplate='Window: %{x}<br>Strength: %{y:.4f}<extra></extra>',
                name='Trend Strength'
            ),
            row=row, col=col
        )
        
        # Add strength interpretation
        max_strength = max(trend_df['strength'])
        if max_strength > 0.1:
            strength_text = "Very Strong Trends"
            color = "#28a745"
        elif max_strength > 0.05:
            strength_text = "Strong Trends"
            color = "#28a745"
        elif max_strength > 0.02:
            strength_text = "Moderate Trends"
            color = "#ffc107"
        else:
            strength_text = "Weak Trends"
            color = "#dc3545"
        
        fig.add_annotation(
            text=strength_text,
            xref=f"x{col} domain", yref=f"y{col} domain",
            x=0.5, y=0.9, showarrow=False,
            font=dict(size=12, color=color),
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Time Window", row=row, col=col)
    fig.update_yaxes(title_text="Trend Strength", row=row, col=col)

def create_alert_history_dashboard(df, save_path="alert_history.html"):
    """Create interactive alert history and key events timeline"""
    if df.empty:
        print("No data available for alert history")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Alert Events Timeline',
            'Alert Frequency Analysis',
            'Event Impact Analysis',
            'Alert System Performance'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Alert events timeline
    plot_alert_timeline_interactive(df, fig, 1, 1)
    
    # 2. Alert frequency analysis
    plot_alert_frequency_interactive(df, fig, 1, 2)
    
    # 3. Event impact analysis
    plot_event_impact_interactive(df, fig, 2, 1)
    
    # 4. Alert effectiveness tracking
    plot_alert_effectiveness_interactive(df, fig, 2, 2)
    
    fig.update_layout(
        title_text='Interactive Alert History & Key Events Dashboard',
        height=1000,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.write_html(save_path)
    print(f"Interactive alert history dashboard saved to {save_path}")
    return fig

def plot_alert_timeline_interactive(df, fig, row, col):
    """Plot interactive timeline of alert events"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    
    # Add sentiment timeline
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['sentiment'],
            mode='lines',
            name='Daily Sentiment',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
        ),
        row=row, col=col
    )
    
    # Mark alert events
    negative_alerts = daily_sentiment[daily_sentiment['sentiment'] <= -0.5]
    positive_alerts = daily_sentiment[daily_sentiment['sentiment'] >= 0.7]
    
    total_alerts = len(negative_alerts) + len(positive_alerts)
    
    if not negative_alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=negative_alerts['date'],
                y=negative_alerts['sentiment'],
                mode='markers',
                name='Negative Alert',
                marker=dict(color='red', size=10, symbol='diamond'),
                hovertemplate='<b>Negative Alert</b><br>Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    if not positive_alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=positive_alerts['date'],
                y=positive_alerts['sentiment'],
                mode='markers',
                name='Positive Surge',
                marker=dict(color='green', size=10, symbol='star'),
                hovertemplate='<b>Positive Surge</b><br>Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    # Add threshold lines
    fig.add_hline(
        y=-0.5, line_dash="dot", line_color="red", opacity=0.7,
        annotation_text="Negative Threshold", 
        row=row, col=col
    )
    fig.add_hline(
        y=0.7, line_dash="dot", line_color="green", opacity=0.7,
        annotation_text="Positive Threshold",
        row=row, col=col
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Sentiment Score", row=row, col=col)

def plot_alert_frequency_interactive(df, fig, row, col):
    """Plot interactive alert frequency analysis"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['date', 'sentiment']
    
    # Count alerts by type
    negative_alerts = len(daily_sentiment[daily_sentiment['sentiment'] <= -0.5])
    positive_alerts = len(daily_sentiment[daily_sentiment['sentiment'] >= 0.7])
    
    # Weekly alert frequency
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment['week'] = daily_sentiment['date'].dt.to_period('W').astype(str)
    
    weekly_alerts = daily_sentiment.groupby('week').apply(
        lambda x: len(x[(x['sentiment'] <= -0.5) | (x['sentiment'] >= 0.7)])
    ).reset_index()
    weekly_alerts.columns = ['week', 'alert_count']
    
    if not weekly_alerts.empty:
        fig.add_trace(
            go.Bar(
                x=weekly_alerts['week'],
                y=weekly_alerts['alert_count'],
                marker_color='orange',
                hovertemplate='Week: %{x}<br>Alerts: %{y}<extra></extra>',
                name='Weekly Alerts'
            ),
            row=row, col=col
        )
    else:
        # Show summary if no weekly data
        fig.add_trace(
            go.Bar(
                x=['Negative', 'Positive'],
                y=[negative_alerts, positive_alerts],
                marker_color=['red', 'green'],
                hovertemplate='Type: %{x}<br>Count: %{y}<extra></extra>',
                name='Alert Summary'
            ),
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Week", row=row, col=col)
    fig.update_yaxes(title_text="Alert Count", row=row, col=col)

def plot_event_impact_interactive(df, fig, row, col):
    """Plot interactive event impact analysis"""
    df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    daily_data = df.groupby(df['date'].dt.date).agg({
        'sentiment_score': 'mean',
        'source': 'count'
    }).reset_index()
    daily_data.columns = ['date', 'sentiment', 'volume']
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    # Find significant events
    daily_data['sentiment_change'] = daily_data['sentiment'].diff().abs()
    significant_events = daily_data[daily_data['sentiment_change'] > daily_data['sentiment_change'].quantile(0.8)]
    
    if not significant_events.empty:
        # Plot sentiment with event markers
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['sentiment'],
                mode='lines',
                name='Sentiment',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add significant events
        fig.add_trace(
            go.Scatter(
                x=significant_events['date'],
                y=significant_events['sentiment'],
                mode='markers',
                name='Significant Event',
                marker=dict(
                    color='red',
                    size=significant_events['volume'] * 0.5 + 10,  # Scale by volume
                    sizemode='diameter',
                    sizeref=2.0,
                    opacity=0.8
                ),
                hovertemplate='<b>Significant Event</b><br>Date: %{x}<br>Sentiment: %{y:.3f}<br>Volume: %{marker.size:.0f}<br>Change: %{customdata:.3f}<extra></extra>',
                customdata=significant_events['sentiment_change']
            ),
            row=row, col=col
        )
    else:
        fig.add_annotation(
            text="No significant events detected",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, showarrow=False,
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Sentiment Score", row=row, col=col)

def plot_alert_effectiveness_interactive(df, fig, row, col):
    """Plot interactive alert effectiveness tracking"""
    # Simulated performance data (in real system, use historical data)
    alert_types = ['Negative\nSentiment', 'Positive\nSurge', 'High\nVolatility', 'Trend\nReversal']
    effectiveness = [0.75, 0.68, 0.82, 0.71]  # Accuracy rates
    false_positives = [0.15, 0.22, 0.12, 0.18]  # False positive rates
    
    # Effectiveness bars
    fig.add_trace(
        go.Bar(
            x=alert_types,
            y=effectiveness,
            name='Effectiveness',
            marker_color='#28a745',
            hovertemplate='Type: %{x}<br>Effectiveness: %{y:.2f}<extra></extra>'
        ),
        row=row, col=col
    )
    
    # False positive bars
    fig.add_trace(
        go.Bar(
            x=alert_types,
            y=false_positives,
            name='False Positive Rate',
            marker_color='#dc3545',
            hovertemplate='Type: %{x}<br>False Positive Rate: %{y:.2f}<extra></extra>'
        ),
        row=row, col=col
    )
    
    fig.update_xaxes(title_text="Alert Type", row=row, col=col)
    fig.update_yaxes(title_text="Rate", range=[0, 1], row=row, col=col)

if __name__ == "__main__":
    # Test the visualizations
    try:
        df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
        create_competitor_analysis(df)
        create_trend_evolution_analysis(df)
        create_alert_history_dashboard(df)
        print("All interactive visualizations generated successfully!")
    except FileNotFoundError:
        print("Please run the sentiment analysis pipeline first.")