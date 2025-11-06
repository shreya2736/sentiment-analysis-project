import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from config import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


warnings.filterwarnings('ignore')

class StrategicDashboard:
    def __init__(self, data_file="industry_insights_with_financial_sentiment.csv"):
        """Initialize dashboard with sentiment data"""
        self.data_file = data_file
        self.df = None
        self.daily_sentiment = None
        self.load_data()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_data(self):
        """Load and prepare data for dashboard"""
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.df)} records from {self.data_file}")
            
            # Convert dates and prepare daily aggregation
            self.df['date'] = pd.to_datetime(self.df['publishedAt'], errors='coerce')
            self.df = self.df.dropna(subset=['date'])
            self.df = self.df.sort_values('date')
            
            # Create daily sentiment aggregation
            self.daily_sentiment = self.df.groupby(self.df['date'].dt.date).agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'sentiment': lambda x: x.value_counts().to_dict()
            }).reset_index()
            
            # Flatten column names
            self.daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'article_count', 'sentiment_dist']
            self.daily_sentiment['date'] = pd.to_datetime(self.daily_sentiment['date'])
            
            print(f"Prepared daily sentiment data: {len(self.daily_sentiment)} days")
            
        except FileNotFoundError:
            print(f"Data file {self.data_file} not found. Please run the pipeline first.")
            self.df = pd.DataFrame()
            self.daily_sentiment = pd.DataFrame()
    
    def create_interactive_overview_dashboard(self, save_path="interactive_dashboard_overview.html"):
        """Create comprehensive interactive overview dashboard"""
        if self.daily_sentiment.empty:
            print("No data available for dashboard")
            return
        
        # Create subplot figure - UPDATED SPECS
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Sentiment Timeline with Trends',
                'Sentiment Score Distribution',
                'Volume vs Sentiment Correlation',
                'Sentiment by Source',
                'Trend Analysis',
                'Alert Summary & Key Metrics'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]  # Changed from "domain" to regular
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Sentiment Timeline
        self.plot_interactive_sentiment_timeline(fig, 1, 1)
        
        # 2. Sentiment Distribution
        self.plot_interactive_sentiment_distribution(fig, 1, 2)
        
        # 3. Volume vs Sentiment
        self.plot_interactive_volume_sentiment_correlation(fig, 2, 1)
        
        # 4. Source Analysis
        self.plot_interactive_source_analysis(fig, 2, 2)
        
        # 5. Trend Analysis
        self.plot_interactive_trend_analysis(fig, 3, 1)
        
        # 6. Alert Summary - UPDATED to use a different visualization
        self.plot_interactive_alert_summary_fixed(fig, 3, 2)
        
        fig.update_layout(
            title_text='Strategic Intelligence Dashboard - Interactive Overview',
            height=1400,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.write_html(save_path)
        print(f"Interactive overview dashboard saved to {save_path}")
        return fig
    
    def plot_interactive_sentiment_timeline(self, fig, row, col):
        """Plot interactive sentiment timeline"""
        if self.daily_sentiment.empty:
            return
            
        # Main sentiment line
        fig.add_trace(
            go.Scatter(
                x=self.daily_sentiment['date'],
                y=self.daily_sentiment['avg_sentiment'],
                mode='lines+markers',
                name='Daily Sentiment',
                line=dict(color='blue', width=3),
                marker=dict(size=4),
                hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # 7-day moving average
        if len(self.daily_sentiment) >= 7:
            ma_7 = self.daily_sentiment['avg_sentiment'].rolling(window=7, center=True).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.daily_sentiment['date'],
                    y=ma_7,
                    mode='lines',
                    name='7-day Moving Average',
                    line=dict(color='red', width=3, dash='dash'),
                    hovertemplate='Date: %{x}<br>7-day MA: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Add sentiment zones
        fig.add_hrect(
            y0=-1, y1=-0.3,
            fillcolor="red", opacity=0.2,
            line_width=0, row=row, col=col,
            annotation_text="Negative Zone", annotation_position="left"
        )
        
        fig.add_hrect(
            y0=0.3, y1=1,
            fillcolor="green", opacity=0.2,
            line_width=0, row=row, col=col,
            annotation_text="Positive Zone", annotation_position="left"
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Sentiment Score", row=row, col=col)
    
    def plot_interactive_sentiment_distribution(self, fig, row, col):
        """Plot interactive sentiment distribution"""
        sentiment_scores = self.df['sentiment_score'].dropna()
        
        if sentiment_scores.empty:
            return
            
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=sentiment_scores,
                nbinsx=30,
                name='Sentiment Distribution',
                marker_color='skyblue',
                opacity=0.7,
                hovertemplate='Sentiment: %{x:.3f}<br>Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add statistical lines
        mean_sentiment = sentiment_scores.mean()
        median_sentiment = sentiment_scores.median()
        
        fig.add_vline(
            x=mean_sentiment, line_dash="dash", line_color="red",
            annotation_text=f"Mean: {mean_sentiment:.3f}", 
            row=row, col=col
        )
        
        fig.add_vline(
            x=median_sentiment, line_dash="dash", line_color="green",
            annotation_text=f"Median: {median_sentiment:.3f}",
            row=row, col=col
        )
        
        fig.add_vline(x=0, line_color="black", opacity=0.5, row=row, col=col)
        
        fig.update_xaxes(title_text="Sentiment Score", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    def plot_interactive_volume_sentiment_correlation(self, fig, row, col):
        """Plot interactive volume vs sentiment correlation"""
        if self.daily_sentiment.empty:
            return
            
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=self.daily_sentiment['article_count'],
                y=self.daily_sentiment['avg_sentiment'],
                mode='markers',
                name='Daily Data',
                marker=dict(
                    size=8,
                    color=self.daily_sentiment['avg_sentiment'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Sentiment")
                ),
                hovertemplate='Articles: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Trend line
        if len(self.daily_sentiment) > 1:
            z = np.polyfit(self.daily_sentiment['article_count'], self.daily_sentiment['avg_sentiment'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(self.daily_sentiment['article_count'].min(), self.daily_sentiment['article_count'].max(), 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='Articles: %{x}<br>Trend: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Calculate correlation
        correlation = np.corrcoef(self.daily_sentiment['article_count'], self.daily_sentiment['avg_sentiment'])[0, 1]
        
        fig.update_xaxes(title_text="Daily Article Count", row=row, col=col)
        fig.update_yaxes(title_text="Average Sentiment", row=row, col=col)
        
        # Add correlation annotation
        fig.add_annotation(
            xref=f"x{col}", yref=f"y{col}",
            x=0.05, y=0.95, xanchor='left', yanchor='top',
            text=f"Correlation: r = {correlation:.3f}",
            showarrow=False,
            bgcolor="white",
            row=row, col=col
        )
    
    def plot_interactive_source_analysis(self, fig, row, col):
        """Plot interactive source sentiment analysis"""
        if 'source' not in self.df.columns:
            return
            
        source_sentiment = self.df.groupby('source').agg({
            'sentiment_score': ['mean', 'count']
        }).reset_index()
        
        source_sentiment.columns = ['source', 'avg_sentiment', 'count']
        source_sentiment = source_sentiment[source_sentiment['count'] >= 5]
        source_sentiment = source_sentiment.sort_values('avg_sentiment')
        
        if source_sentiment.empty:
            return
            
        # Color based on sentiment
        colors = []
        for sentiment in source_sentiment['avg_sentiment']:
            if sentiment < -0.1:
                colors.append('red')
            elif sentiment > 0.1:
                colors.append('green')
            else:
                colors.append('gray')
        
        fig.add_trace(
            go.Bar(
                x=source_sentiment['avg_sentiment'],
                y=source_sentiment['source'],
                orientation='h',
                marker_color=colors,
                hovertemplate='<b>%{y}</b><br>Sentiment: %{x:.3f}<br>Articles: %{customdata}<extra></extra>',
                customdata=source_sentiment['count'],
                name='Source Sentiment'
            ),
            row=row, col=col
        )
        
        fig.add_vline(x=0, line_color="black", opacity=0.5, row=row, col=col)
        fig.update_xaxes(title_text="Average Sentiment Score", row=row, col=col)
        fig.update_yaxes(title_text="Source", row=row, col=col)
    
    def plot_interactive_trend_analysis(self, fig, row, col):
        """Plot interactive trend analysis"""
        if len(self.daily_sentiment) >= 14:
            recent_data = self.daily_sentiment.tail(7)['avg_sentiment'].mean()
            historical_data = self.daily_sentiment.iloc[-14:-7]['avg_sentiment'].mean()
            
            categories = ['Historical (7-14 days ago)', 'Recent (Last 7 days)']
            values = [historical_data, recent_data]
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=['lightblue', 'orange'],
                    hovertemplate='%{x}<br>Sentiment: %{y:.3f}<extra></extra>',
                    name='Trend Comparison'
                ),
                row=row, col=col
            )
            
            # Add trend arrow and text
            trend_change = recent_data - historical_data
            if trend_change > 0.02:
                trend_symbol = "↗"
                trend_color = "green"
            elif trend_change < -0.02:
                trend_symbol = "↘"
                trend_color = "red"
            else:
                trend_symbol = "→"
                trend_color = "black"
            
            fig.add_annotation(
                xref=f"x{col}", yref=f"y{col}",
                x=0.5, y=max(values) * 0.8,
                text=f"Trend: {trend_symbol} ({trend_change:+.3f})",
                showarrow=False,
                font=dict(color=trend_color, size=12),
                row=row, col=col
            )
        else:
            fig.add_annotation(
                xref=f"x{col} domain", yref=f"y{col} domain",
                x=0.5, y=0.5,
                text="Insufficient data for trend analysis (need 14+ days)",
                showarrow=False,
                row=row, col=col
            )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
        fig.update_xaxes(title_text="Time Period", row=row, col=col)
        fig.update_yaxes(title_text="Average Sentiment Score", row=row, col=col)
    
    def plot_interactive_alert_summary_fixed(self, fig, row, col):
        """Plot interactive alert summary - FIXED VERSION without table"""
        # Calculate key metrics
        total_articles = len(self.df)
        avg_sentiment = self.df['sentiment_score'].mean()
        sentiment_volatility = self.df['sentiment_score'].std()
        
        # Count sentiment categories
        sentiment_counts = self.df['sentiment'].value_counts()
        positive_pct = (sentiment_counts.get('positive', 0) / total_articles) * 100
        negative_pct = (sentiment_counts.get('negative', 0) / total_articles) * 100
        neutral_pct = (sentiment_counts.get('neutral', 0) / total_articles) * 100
        
        # Create alert conditions
        alerts = []
        if avg_sentiment < -0.3:
            alerts.append("🔻 Negative sentiment dominance")
        elif avg_sentiment > 0.3:
            alerts.append("🚀 Positive sentiment dominance")
        
        if sentiment_volatility > 0.5:
            alerts.append("⚡ High volatility detected")
        
        # Recent trend check
        if len(self.daily_sentiment) >= 3:
            recent_trend = self.daily_sentiment.tail(3)['avg_sentiment'].diff().mean()
            if recent_trend > 0.1:
                alerts.append("📈 Strong positive trend")
            elif recent_trend < -0.1:
                alerts.append("📉 Strong negative trend")
        
        # Create a bar chart showing key metrics instead of a table
        metrics_names = ['Total Articles', 'Avg Sentiment', 'Volatility', 'Positive %', 'Negative %', 'Neutral %']
        metrics_values = [total_articles, avg_sentiment, sentiment_volatility, positive_pct, negative_pct, neutral_pct]
        
        # Create colors for the bars
        colors = ['blue', 'green', 'orange', 'lightgreen', 'lightcoral', 'lightgray']
        
        fig.add_trace(
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>',
                name='Key Metrics'
            ),
            row=row, col=col
        )
        
        # Add alerts as annotation (now safe since it's a regular subplot)
        if alerts:
            alert_text = "<b>ACTIVE ALERTS:</b><br>" + "<br>".join([f"• {alert}" for alert in alerts[:3]])
        else:
            alert_text = "<b>ACTIVE ALERTS:</b><br>• No alerts triggered"
        
        fig.add_annotation(
            xref=f"x{col}", yref=f"y{col}",
            x=0.5, y=0.95 * max(metrics_values),
            text=alert_text,
            showarrow=False,
            align='left',
            bgcolor="lightyellow",
            bordercolor="black",
            borderwidth=1,
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Metrics", row=row, col=col)
        fig.update_yaxes(title_text="Values", row=row, col=col)
    
    def create_interactive_forecast_dashboard(self, forecast_df=None, save_path="interactive_dashboard_forecast.html"):
        """Create interactive forecast-focused dashboard"""
        try:
            # Try to load forecast data if not provided
            if forecast_df is None and os.path.exists("sentiment_forecast.csv"):
                forecast_df = pd.read_csv("sentiment_forecast.csv")
                forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            if forecast_df is None or forecast_df.empty:
                print("No forecast data available")
                return
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Historical + Forecast Timeline',
                    'Forecast Confidence Analysis',
                    'Forecast vs Recent Trend',
                    'Risk Assessment Matrix'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # 1. Historical + Forecast Timeline
            self.plot_interactive_forecast_timeline(fig, forecast_df, 1, 1)
            
            # 2. Forecast Confidence Intervals
            self.plot_interactive_forecast_confidence(fig, forecast_df, 1, 2)
            
            # 3. Forecast vs Recent Trend
            self.plot_interactive_forecast_vs_trend(fig, forecast_df, 2, 1)
            
            # 4. Risk Assessment
            self.plot_interactive_risk_assessment(fig, forecast_df, 2, 2)
            
            fig.update_layout(
                title_text='Strategic Intelligence Dashboard - Interactive Forecast Analysis',
                height=1000,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.write_html(save_path)
            print(f"Interactive forecast dashboard saved to {save_path}")
            return fig
            
        except Exception as e:
            print(f"Error creating interactive forecast dashboard: {e}")
    
    def plot_interactive_forecast_timeline(self, fig, forecast_df, row, col):
        """Plot interactive historical data with forecast"""
        # Plot historical data (last 30 days)
        recent_data = self.daily_sentiment.tail(30)
        
        fig.add_trace(
            go.Scatter(
                x=recent_data['date'],
                y=recent_data['avg_sentiment'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue', width=3),
                marker=dict(size=4),
                hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Plot forecast
        if forecast_df is not None and not forecast_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['sentiment'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=6, symbol='diamond'),
                    hovertemplate='Date: %{x}<br>Forecast: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                    y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='80% Confidence Interval',
                    hovertemplate='Date: %{x}<br>Upper: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Sentiment Score", row=row, col=col)
    
    def plot_interactive_forecast_confidence(self, fig, forecast_df, row, col):
        """Plot interactive forecast confidence analysis - FIXED VERSION"""
        if forecast_df is not None and not forecast_df.empty:
            days = list(range(1, len(forecast_df) + 1))  # Convert range to list
            confidence_width = forecast_df['upper_bound'] - forecast_df['lower_bound']
            
            fig.add_trace(
                go.Bar(
                    x=days,  # Now using list instead of range
                    y=confidence_width,
                    marker_color='orange',
                    hovertemplate='Day %{x}<br>CI Width: %{y:.3f}<extra></extra>',
                    name='Confidence Width'
                ),
                row=row, col=col
            )
            
            # Add average line
            avg_width = confidence_width.mean()
            fig.add_hline(
                y=avg_width, line_dash="dash", line_color="red",
                annotation_text=f"Avg Width: {avg_width:.3f}",
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Forecast Day", row=row, col=col)
            fig.update_yaxes(title_text="Confidence Interval Width", row=row, col=col)
        else:
            fig.add_annotation(
                xref=f"x{col} domain", yref=f"y{col} domain",
                x=0.5, y=0.5,
                text="No forecast data available",
                showarrow=False,
                row=row, col=col
            )
    def plot_interactive_forecast_vs_trend(self, fig, forecast_df, row, col):
        """Plot interactive forecast vs recent trend comparison"""
        if forecast_df is not None and not forecast_df.empty:
            recent_avg = self.daily_sentiment.tail(7)['avg_sentiment'].mean()
            forecast_avg = forecast_df['sentiment'].mean()
            
            categories = ['Recent Average (Last 7 days)', 'Forecast Average (Next 7 days)']
            values = [recent_avg, forecast_avg]
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=['lightblue', 'orange'],
                    hovertemplate='%{x}<br>Value: %{y:.3f}<extra></extra>',
                    name='Comparison'
                ),
                row=row, col=col
            )
            
            # Add comparison text
            diff = forecast_avg - recent_avg
            if diff > 0.05:
                comparison_text = "Improvement Expected"
                color = "green"
            elif diff < -0.05:
                comparison_text = "Decline Expected" 
                color = "red"
            else:
                comparison_text = "Stable Trend"
                color = "black"
            
            fig.add_annotation(
                xref=f"x{col}", yref=f"y{col}",
                x=0.5, y=max(values) * 0.8,
                text=f"Forecast vs Recent: {diff:+.3f}<br>{comparison_text}",
                showarrow=False,
                font=dict(color=color, size=11),
                row=row, col=col
            )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=row, col=col)
        fig.update_xaxes(title_text="Comparison", row=row, col=col)
        fig.update_yaxes(title_text="Sentiment Score", row=row, col=col)
    
    def plot_interactive_risk_assessment(self, fig, forecast_df, row, col):
        """Plot interactive risk assessment"""
        risk_factors = []
        risk_scores = []
        
        # Calculate various risk metrics
        if forecast_df is not None and not forecast_df.empty:
            # Volatility risk
            forecast_volatility = forecast_df['sentiment'].std()
            volatility_risk = min(forecast_volatility * 2, 1.0)
            risk_factors.append('Forecast Volatility')
            risk_scores.append(volatility_risk)
            
            # Negative trend risk
            forecast_trend = forecast_df['sentiment'].diff().mean()
            trend_risk = max(-forecast_trend, 0) if forecast_trend < 0 else 0
            risk_factors.append('Negative Trend')
            risk_scores.append(min(trend_risk * 2, 1.0))
            
            # Low confidence risk (wide intervals)
            avg_ci_width = (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
            confidence_risk = min(avg_ci_width / 2, 1.0)
            risk_factors.append('Low Confidence')
            risk_scores.append(confidence_risk)
            
            # Overall sentiment risk
            min_forecast = forecast_df['sentiment'].min()
            sentiment_risk = max(-min_forecast, 0) if min_forecast < 0 else 0
            risk_factors.append('Negative Sentiment')
            risk_scores.append(min(sentiment_risk, 1.0))
        else:
            # Default risk assessment
            risk_factors = ['Data Quality', 'Model Reliability']
            risk_scores = [0.8, 0.8]
        
        # Create risk assessment chart
        colors = []
        for score in risk_scores:
            if score > 0.7:
                colors.append('red')
            elif score > 0.4:
                colors.append('orange')
            else:
                colors.append('green')
        
        fig.add_trace(
            go.Bar(
                x=risk_scores,
                y=risk_factors,
                orientation='h',
                marker_color=colors,
                hovertemplate='%{y}<br>Risk Score: %{x:.2f}<extra></extra>',
                name='Risk Score'
            ),
            row=row, col=col
        )
        
        # Add risk zones
        fig.add_vrect(
            x0=0, x1=0.3,
            fillcolor="green", opacity=0.2,
            line_width=0, row=row, col=col,
            annotation_text="Low Risk", annotation_position="top"
        )
        
        fig.add_vrect(
            x0=0.3, x1=0.7,
            fillcolor="orange", opacity=0.2,
            line_width=0, row=row, col=col,
            annotation_text="Medium Risk", annotation_position="top"
        )
        
        fig.add_vrect(
            x0=0.7, x1=1.0,
            fillcolor="red", opacity=0.2,
            line_width=0, row=row, col=col,
            annotation_text="High Risk", annotation_position="top"
        )
        
        fig.update_xaxes(title_text="Risk Score (0 = Low, 1 = High)", range=[0, 1], row=row, col=col)
        fig.update_yaxes(title_text="Risk Factor", row=row, col=col)

def generate_interactive_dashboard():
    """Generate complete interactive dashboard suite"""
    print("="*60)
    print("🎨 GENERATING INTERACTIVE STRATEGIC INTELLIGENCE DASHBOARD")
    print("="*60)
    
    # Initialize dashboard
    dashboard = StrategicDashboard()
    
    if dashboard.df.empty:
        print("No data available. Please run the sentiment analysis pipeline first.")
        return
    
    # Generate interactive overview dashboard
    print("\n1. Creating Interactive Overview Dashboard...")
    overview_fig = dashboard.create_interactive_overview_dashboard()
    
    # Generate interactive forecast dashboard
    print("\n2. Creating Interactive Forecast Dashboard...")
    forecast_fig = dashboard.create_interactive_forecast_dashboard()
    
    print("\n" + "="*60)
    print("✅ INTERACTIVE DASHBOARD GENERATION COMPLETED!")
    print("="*60)
    print("📁 Generated interactive files:")
    print("  • interactive_dashboard_overview.html")
    print("  • interactive_dashboard_forecast.html")
    print("\n🎯 Features:")
    print("  • Interactive charts with hover effects")
    print("  • Zoom and pan capabilities")
    print("  • Real-time data exploration")
    print("  • Export functionality")
    print("  • Responsive design")

if __name__ == "__main__":
    generate_interactive_dashboard()