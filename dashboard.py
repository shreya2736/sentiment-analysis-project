
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from config import *

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
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Configure plot styling"""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
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
    
    def create_overview_dashboard(self, save_path="dashboard_overview.png"):
        """Create comprehensive overview dashboard"""
        if self.daily_sentiment.empty:
            print("No data available for dashboard")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Strategic Intelligence Dashboard - Sentiment Overview', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Timeline
        ax1 = axes[0, 0]
        self.plot_sentiment_timeline(ax1)
        
        # 2. Sentiment Distribution
        ax2 = axes[0, 1]
        self.plot_sentiment_distribution(ax2)
        
        # 3. Volume vs Sentiment
        ax3 = axes[0, 2]
        self.plot_volume_sentiment_correlation(ax3)
        
        # 4. Source Analysis
        ax4 = axes[1, 0]
        self.plot_source_analysis(ax4)
        
        # 5. Trend Analysis
        ax5 = axes[1, 1]
        self.plot_trend_analysis(ax5)
        
        # 6. Alert Summary
        ax6 = axes[1, 2]
        self.plot_alert_summary(ax6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Overview dashboard saved to {save_path}")
    
    def plot_sentiment_timeline(self, ax):
        """Plot sentiment over time with trend line"""
        ax.plot(self.daily_sentiment['date'], self.daily_sentiment['avg_sentiment'], 
                color='blue', alpha=0.7, linewidth=2, label='Daily Sentiment')
        
        # Add 7-day moving average
        if len(self.daily_sentiment) >= 7:
            ma_7 = self.daily_sentiment['avg_sentiment'].rolling(window=7, center=True).mean()
            ax.plot(self.daily_sentiment['date'], ma_7, 
                   color='red', linewidth=3, label='7-day Moving Average')
        
        # Add neutral line and sentiment zones
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Neutral')
        ax.fill_between(self.daily_sentiment['date'], -1, -0.3, alpha=0.2, color='red', label='Negative Zone')
        ax.fill_between(self.daily_sentiment['date'], 0.3, 1, alpha=0.2, color='green', label='Positive Zone')
        
        ax.set_title('Sentiment Timeline')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    def plot_sentiment_distribution(self, ax):
        """Plot sentiment score distribution"""
        sentiment_scores = self.df['sentiment_score'].dropna()
        
        # Histogram
        ax.hist(sentiment_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical lines for mean and median
        mean_sentiment = sentiment_scores.mean()
        median_sentiment = sentiment_scores.median()
        
        ax.axvline(mean_sentiment, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_sentiment:.3f}')
        ax.axvline(median_sentiment, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_sentiment:.3f}')
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, label='Neutral')
        
        ax.set_title('Sentiment Score Distribution')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_volume_sentiment_correlation(self, ax):
        """Plot correlation between article volume and sentiment"""
        ax.scatter(self.daily_sentiment['article_count'], self.daily_sentiment['avg_sentiment'], 
                  alpha=0.6, s=60)
        
        # Add trend line
        z = np.polyfit(self.daily_sentiment['article_count'], self.daily_sentiment['avg_sentiment'], 1)
        p = np.poly1d(z)
        ax.plot(self.daily_sentiment['article_count'], p(self.daily_sentiment['article_count']), 
               "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Calculate correlation
        correlation = np.corrcoef(self.daily_sentiment['article_count'], 
                                self.daily_sentiment['avg_sentiment'])[0, 1]
        
        ax.set_title(f'Volume vs Sentiment (r={correlation:.3f})')
        ax.set_xlabel('Daily Article Count')
        ax.set_ylabel('Average Sentiment')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_source_analysis(self, ax):
        """Analyze sentiment by source"""
        source_sentiment = self.df.groupby('source').agg({
            'sentiment_score': ['mean', 'count']
        }).reset_index()
        
        source_sentiment.columns = ['source', 'avg_sentiment', 'count']
        source_sentiment = source_sentiment[source_sentiment['count'] >= 5]  # Filter sources with <5 articles
        source_sentiment = source_sentiment.sort_values('avg_sentiment')
        
        if not source_sentiment.empty:
            colors = ['red' if x < -0.1 else 'green' if x > 0.1 else 'gray' 
                     for x in source_sentiment['avg_sentiment']]
            
            bars = ax.barh(source_sentiment['source'], source_sentiment['avg_sentiment'], color=colors, alpha=0.7)
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, source_sentiment['count'])):
                width = bar.get_width()
                ax.text(width + 0.01 if width >= 0 else width - 0.01, bar.get_y() + bar.get_height()/2, 
                       f'n={count}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('Sentiment by Source (min 5 articles)')
        ax.set_xlabel('Average Sentiment Score')
        ax.grid(True, alpha=0.3, axis='x')
    
    def plot_trend_analysis(self, ax):
        """Show recent vs historical trend analysis"""
        if len(self.daily_sentiment) >= 14:
            recent_data = self.daily_sentiment.tail(7)['avg_sentiment'].mean()
            historical_data = self.daily_sentiment.iloc[-14:-7]['avg_sentiment'].mean() if len(self.daily_sentiment) >= 14 else 0
            
            categories = ['Historical\n(7-14 days ago)', 'Recent\n(Last 7 days)']
            values = [historical_data, recent_data]
            colors = ['lightblue', 'orange']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                       f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
            
            # Add trend arrow and text
            trend_change = recent_data - historical_data
            trend_text = f"Trend: {'↗' if trend_change > 0.02 else '↘' if trend_change < -0.02 else '→'} ({trend_change:+.3f})"
            ax.text(0.5, max(values) * 0.8, trend_text, ha='center', transform=ax.transData, 
                   fontsize=12, fontweight='bold', 
                   color='green' if trend_change > 0.02 else 'red' if trend_change < -0.02 else 'black')
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor trend analysis\n(need 14+ days)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Trend Analysis')
        ax.set_ylabel('Average Sentiment Score')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_alert_summary(self, ax):
        """Show alert summary and key metrics"""
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
            alerts.append("Negative sentiment dominance")
        elif avg_sentiment > 0.3:
            alerts.append("Positive sentiment dominance")
        
        if sentiment_volatility > 0.5:
            alerts.append("High volatility detected")
        
        # Recent trend check
        if len(self.daily_sentiment) >= 3:
            recent_trend = self.daily_sentiment.tail(3)['avg_sentiment'].diff().mean()
            if recent_trend > 0.1:
                alerts.append("Strong positive trend")
            elif recent_trend < -0.1:
                alerts.append("Strong negative trend")
        
        # Create summary text
        summary_text = f"""KEY METRICS:
        
Total Articles: {total_articles}
Avg Sentiment: {avg_sentiment:.3f}
Volatility: {sentiment_volatility:.3f}

SENTIMENT BREAKDOWN:
Positive: {positive_pct:.1f}%
Negative: {negative_pct:.1f}%
Neutral: {neutral_pct:.1f}%

ACTIVE ALERTS:"""
        
        if alerts:
            for alert in alerts[:4]:  # Show max 4 alerts
                summary_text += f"\n• {alert}"
        else:
            summary_text += "\n• No alerts triggered"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        ax.set_title('Alert Summary & Key Metrics')
        ax.axis('off')
    
    def create_forecast_dashboard(self, forecast_df=None, save_path="dashboard_forecast.png"):
        """Create forecast-focused dashboard"""
        try:
            # Try to load forecast data if not provided
            if forecast_df is None:
                from forecasting import forecast_sentiment
                forecasts, forecast_df, daily_data = forecast_sentiment()
                if forecast_df is None:
                    print("No forecast data available")
                    return
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Strategic Intelligence Dashboard - Forecast Analysis', fontsize=16, fontweight='bold')
            
            # 1. Historical + Forecast Timeline
            ax1 = axes[0, 0]
            self.plot_forecast_timeline(ax1, forecast_df)
            
            # 2. Forecast Confidence Intervals
            ax2 = axes[0, 1]
            self.plot_forecast_confidence(ax2, forecast_df)
            
            # 3. Forecast vs Recent Trend
            ax3 = axes[1, 0]
            self.plot_forecast_vs_trend(ax3, forecast_df)
            
            # 4. Risk Assessment
            ax4 = axes[1, 1]
            self.plot_risk_assessment(ax4, forecast_df)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Forecast dashboard saved to {save_path}")
            
        except Exception as e:
            print(f"Error creating forecast dashboard: {e}")
    
    def plot_forecast_timeline(self, ax, forecast_df):
        """Plot historical data with forecast"""
        # Plot historical data (last 30 days)
        recent_data = self.daily_sentiment.tail(30)
        ax.plot(recent_data['date'], recent_data['avg_sentiment'], 
               color='blue', linewidth=2, label='Historical', marker='o', markersize=4)
        
        # Plot forecast
        if forecast_df is not None and not forecast_df.empty:
            ax.plot(forecast_df['date'], forecast_df['sentiment'], 
                   color='red', linewidth=2, label='Forecast', marker='s', markersize=5, linestyle='--')
            
            # Add confidence intervals
            ax.fill_between(forecast_df['date'], forecast_df['lower_bound'], forecast_df['upper_bound'],
                           alpha=0.3, color='red', label='80% Confidence Interval')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Sentiment Forecast Timeline')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def plot_forecast_confidence(self, ax, forecast_df):
        """Plot forecast confidence analysis"""
        if forecast_df is not None and not forecast_df.empty:
            days = range(1, len(forecast_df) + 1)
            confidence_width = forecast_df['upper_bound'] - forecast_df['lower_bound']
            
            ax.bar(days, confidence_width, alpha=0.7, color='orange')
            ax.set_title('Forecast Uncertainty by Day')
            ax.set_xlabel('Forecast Day')
            ax.set_ylabel('Confidence Interval Width')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add average line
            avg_width = confidence_width.mean()
            ax.axhline(y=avg_width, color='red', linestyle='--', 
                      label=f'Avg Width: {avg_width:.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No forecast data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
    
    def plot_forecast_vs_trend(self, ax, forecast_df):
        """Compare forecast with recent historical trend"""
        if forecast_df is not None and not forecast_df.empty:
            recent_avg = self.daily_sentiment.tail(7)['avg_sentiment'].mean()
            forecast_avg = forecast_df['sentiment'].mean()
            
            categories = ['Recent Average\n(Last 7 days)', 'Forecast Average\n(Next 5 days)']
            values = [recent_avg, forecast_avg]
            colors = ['lightblue', 'orange']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                       f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
            
            # Add comparison text
            diff = forecast_avg - recent_avg
            comparison_text = f"Forecast vs Recent: {diff:+.3f}\n{'Improvement Expected' if diff > 0.05 else 'Decline Expected' if diff < -0.05 else 'Stable Trend'}"
            ax.text(0.5, max(values) * 0.8, comparison_text, ha='center', transform=ax.transData,
                   fontsize=11, fontweight='bold',
                   color='green' if diff > 0.05 else 'red' if diff < -0.05 else 'black')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Forecast vs Recent Trend')
        ax.set_ylabel('Sentiment Score')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_risk_assessment(self, ax, forecast_df):
        """Show risk assessment based on forecast"""
        risk_factors = []
        risk_scores = []
        
        # Calculate various risk metrics
        if forecast_df is not None and not forecast_df.empty:
            # Volatility risk
            forecast_volatility = forecast_df['sentiment'].std()
            volatility_risk = min(forecast_volatility * 2, 1.0)  # Scale to 0-1
            risk_factors.append('Forecast\nVolatility')
            risk_scores.append(volatility_risk)
            
            # Negative trend risk
            forecast_trend = forecast_df['sentiment'].diff().mean()
            trend_risk = max(-forecast_trend, 0) if forecast_trend < 0 else 0
            risk_factors.append('Negative\nTrend')
            risk_scores.append(min(trend_risk * 2, 1.0))
            
            # Low confidence risk (wide intervals)
            avg_ci_width = (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
            confidence_risk = min(avg_ci_width / 2, 1.0)  # Scale to 0-1
            risk_factors.append('Low\nConfidence')
            risk_scores.append(confidence_risk)
            
            # Overall sentiment risk
            min_forecast = forecast_df['sentiment'].min()
            sentiment_risk = max(-min_forecast, 0) if min_forecast < 0 else 0
            risk_factors.append('Negative\nSentiment')
            risk_scores.append(min(sentiment_risk, 1.0))
        else:
            # Default risk assessment
            risk_factors = ['Data\nQuality', 'Model\nReliability']
            risk_scores = [0.8, 0.8]  # High risk when no forecast available
        
        # Create risk assessment chart
        colors = ['red' if score > 0.7 else 'orange' if score > 0.4 else 'green' for score in risk_scores]
        bars = ax.barh(risk_factors, risk_scores, color=colors, alpha=0.7)
        
        # Add risk score labels
        for bar, score in zip(bars, risk_scores):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Risk Score (0 = Low, 1 = High)')
        ax.set_title('Risk Assessment Matrix')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add risk zones
        ax.axvspan(0, 0.3, alpha=0.2, color='green', label='Low Risk')
        ax.axvspan(0.3, 0.7, alpha=0.2, color='orange', label='Medium Risk')
        ax.axvspan(0.7, 1.0, alpha=0.2, color='red', label='High Risk')

def generate_full_dashboard():
    """Generate complete dashboard suite"""
    print("="*60)
    print("GENERATING STRATEGIC INTELLIGENCE DASHBOARD")
    print("="*60)
    
    # Initialize dashboard
    dashboard = StrategicDashboard()
    
    if dashboard.df.empty:
        print("No data available. Please run the sentiment analysis pipeline first.")
        return
    
    # Generate overview dashboard
    print("\n1. Creating Overview Dashboard...")
    dashboard.create_overview_dashboard()
    
    # Generate forecast dashboard
    print("\n2. Creating Forecast Dashboard...")
    dashboard.create_forecast_dashboard()
    
    print("\n" + "="*60)
    print("DASHBOARD GENERATION COMPLETED!")
    print("="*60)
    print("Generated files:")
    print("- dashboard_overview.png")
    print("- dashboard_forecast.png")
    print("\nDashboard provides:")
    print("- Sentiment timeline with trends")
    print("- Distribution analysis")
    print("- Source comparison")
    print("- Volume correlation")
    print("- Forecast analysis")
    print("- Risk assessment")
    print("- Key metrics and alerts")

if __name__ == "__main__":
    generate_full_dashboard()