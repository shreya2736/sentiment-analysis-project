import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import numpy as np

# Configure the page
st.set_page_config(
    page_title="Strategic Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .alert-positive {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-negative {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
    def __init__(self):
        self.df = None
        self.daily_sentiment = None
        self.load_data()
    
    def load_data(self):
        """Load sentiment data from CSV files"""
        try:
            self.df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
            self.df['date'] = pd.to_datetime(self.df['publishedAt'], errors='coerce')
            self.df = self.df.dropna(subset=['date'])
            
            # Create daily sentiment aggregation
            self.daily_sentiment = self.df.groupby(self.df['date'].dt.date).agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).reset_index()
            self.daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'article_count']
            self.daily_sentiment['date'] = pd.to_datetime(self.daily_sentiment['date'])
            
        except FileNotFoundError:
            st.error("Data files not found. Please run the sentiment analysis pipeline first.")
            self.df = pd.DataFrame()
            self.daily_sentiment = pd.DataFrame()
    
    def run(self):
        """Main dashboard interface"""
        # Header
        st.markdown('<h1 class="main-header">Strategic Intelligence Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar filters
        st.sidebar.title("Dashboard Controls")
        
        # Data status
        if self.df.empty:
            st.sidebar.error("No data available")
            return
        
        # Filters
        date_range = self.get_date_filters()
        sector_filter = self.get_sector_filters()
        competitor_filter = self.get_competitor_filters()
        
        # Apply filters
        filtered_df = self.apply_filters(date_range, sector_filter, competitor_filter)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview", 
            "Competitor Analysis", 
            "Trend Evolution", 
            "Alert History", 
            "Forecast"
        ])
        
        with tab1:
            self.render_overview_tab(filtered_df)
        
        with tab2:
            self.render_competitor_tab(filtered_df)
        
        with tab3:
            self.render_trend_tab(filtered_df)
        
        with tab4:
            self.render_alerts_tab(filtered_df)
        
        with tab5:
            self.render_forecast_tab(filtered_df)
    
    def get_date_filters(self):
        """Date range filter"""
        st.sidebar.subheader("Time Range")
        min_date = self.df['date'].min().date()
        max_date = self.df['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            return date_range
        return (min_date, max_date)
    
    def get_sector_filters(self):
        """Sector/focus area filter"""
        st.sidebar.subheader("Sector Focus")
        
        # Extract sectors from content (simplified)
        sectors = ["All", "Technology", "Finance", "Healthcare", "Energy", "Retail", "Manufacturing"]
        selected_sector = st.sidebar.selectbox("Select sector:", sectors)
        
        return selected_sector
    
    def get_competitor_filters(self):
        """Competitor/source filter"""
        st.sidebar.subheader("Competitor Tracking")
        
        if 'source' in self.df.columns:
            sources = self.df['source'].value_counts().head(15).index.tolist()
            selected_sources = st.sidebar.multiselect(
                "Select competitors/sources to track:",
                options=sources,
                default=sources[:5] if sources else []
            )
            return selected_sources
        return []
    
    def apply_filters(self, date_range, sector, competitors):
        """Apply all filters to data"""
        filtered_df = self.df.copy()
        
        # Date filter
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
        
        # Competitor/source filter
        if competitors and 'source' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['source'].isin(competitors)]
        
        # Sector filter (simplified text-based filtering)
        if sector != "All":
            filtered_df = filtered_df[
                filtered_df['content'].str.contains(sector, case=False, na=False) |
                filtered_df['title'].str.contains(sector, case=False, na=False)
            ]
        
        return filtered_df
    
    def render_overview_tab(self, df):
        """Render overview dashboard"""
        st.header("Sentiment Overview")
        
        if df.empty:
            st.warning("No data available for selected filters.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_sentiment = df['sentiment_score'].mean()
            sentiment_color = "green" if avg_sentiment > 0.1 else "red" if avg_sentiment < -0.1 else "gray"
            st.metric(
                "Average Sentiment", 
                f"{avg_sentiment:.3f}",
                delta=f"{avg_sentiment:.3f}" if abs(avg_sentiment) > 0.01 else "Neutral"
            )
        
        with col2:
            total_articles = len(df)
            st.metric("Total Articles", f"{total_articles:,}")
        
        with col3:
            positive_pct = (df['sentiment'] == 'positive').sum() / len(df) * 100
            st.metric("Positive Articles", f"{positive_pct:.1f}%")
        
        with col4:
            volatility = df['sentiment_score'].std()
            st.metric("Sentiment Volatility", f"{volatility:.3f}")
        
        # Sentiment timeline
        st.subheader("Sentiment Timeline")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        daily_filtered = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_filtered.columns = ['date', 'sentiment']
        daily_filtered['date'] = pd.to_datetime(daily_filtered['date'])
        
        ax.plot(daily_filtered['date'], daily_filtered['sentiment'], 
                color='blue', alpha=0.7, linewidth=2, label='Daily Sentiment')
        
        if len(daily_filtered) >= 7:
            ma_7 = daily_filtered['sentiment'].rolling(window=7, center=True).mean()
            ax.plot(daily_filtered['date'], ma_7, 
                   color='red', linewidth=3, label='7-day Moving Average')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Neutral')
        ax.fill_between(daily_filtered['date'], -1, -0.3, alpha=0.2, color='red', label='Negative Zone')
        ax.fill_between(daily_filtered['date'], 0.3, 1, alpha=0.2, color='green', label='Positive Zone')
        
        ax.set_title('Sentiment Timeline')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
        plt.close()
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sentiment_counts = df['sentiment'].value_counts()
            colors = ['green' if x == 'positive' else 'red' if x == 'negative' else 'gray' 
                     for x in sentiment_counts.index]
            
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
            ax.set_title('Sentiment Distribution')
            ax.set_ylabel('Number of Articles')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Source Performance")
            if 'source' in df.columns:
                source_sentiment = df.groupby('source').agg({
                    'sentiment_score': ['mean', 'count']
                }).reset_index()
                source_sentiment.columns = ['source', 'avg_sentiment', 'count']
                source_sentiment = source_sentiment[source_sentiment['count'] >= 3]
                source_sentiment = source_sentiment.sort_values('avg_sentiment')
                
                if not source_sentiment.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    colors = ['red' if x < -0.1 else 'green' if x > 0.1 else 'gray' 
                             for x in source_sentiment['avg_sentiment']]
                    
                    bars = ax.barh(source_sentiment['source'], source_sentiment['avg_sentiment'], 
                                 color=colors, alpha=0.7)
                    
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                    ax.set_title('Average Sentiment by Source')
                    ax.set_xlabel('Sentiment Score')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    st.pyplot(fig)
                    plt.close()
    
    def render_competitor_tab(self, df):
        """Render competitor analysis"""
        st.header("Competitor & Source Analysis")
        
        if df.empty:
            st.warning("No data available for selected filters.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Share by Source")
            if 'source' in df.columns:
                source_counts = df['source'].value_counts().head(8)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.Set3(np.linspace(0, 1, len(source_counts)))
                
                wedges, texts, autotexts = ax.pie(source_counts.values, 
                                                labels=source_counts.index, 
                                                autopct='%1.1f%%', 
                                                colors=colors, 
                                                startangle=90)
                
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
                
                ax.set_title('Market Share by Source (Article Volume)')
                st.pyplot(fig)
                plt.close()
        
        with col2:
            st.subheader("📈 Source Sentiment Timeline")
            if 'source' in df.columns:
                # Get top 5 sources by volume
                top_sources = df['source'].value_counts().head(5).index
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                for source in top_sources:
                    source_data = df[df['source'] == source]
                    daily_sentiment = source_data.groupby(source_data['date'].dt.date)['sentiment_score'].mean()
                    
                    if len(daily_sentiment) >= 3:
                        ax.plot(daily_sentiment.index, daily_sentiment.values, 
                               marker='o', label=source, linewidth=2, alpha=0.8)
                
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_title('Sentiment Timeline by Source (Top 5)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Average Sentiment Score')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                
                st.pyplot(fig)
                plt.close()
        
        # Source performance ranking
        st.subheader("Source Performance Ranking")
        if 'source' in df.columns:
            source_metrics = df.groupby('source').agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).reset_index()
            source_metrics.columns = ['source', 'avg_sentiment', 'sentiment_volatility', 'article_count']
            source_metrics = source_metrics[source_metrics['article_count'] >= 5]
            
            if not source_metrics.empty:
                # Calculate composite performance score
                source_metrics['performance_score'] = (
                    source_metrics['avg_sentiment'] - 
                    (source_metrics['sentiment_volatility'] * 0.5)
                )
                source_metrics = source_metrics.sort_values('performance_score', ascending=True)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
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
                
                st.pyplot(fig)
                plt.close()
    
    def render_trend_tab(self, df):
        """Render trend evolution analysis"""
        st.header("Trend Evolution Analysis")
        
        if df.empty:
            st.warning("No data available for selected filters.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Rolling Sentiment Trends")
            
            daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
            daily_sentiment.columns = ['date', 'sentiment']
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            daily_sentiment = daily_sentiment.sort_values('date')
            
            if len(daily_sentiment) >= 7:
                fig, ax = plt.subplots(figsize=(12, 8))
                
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
                plt.xticks(rotation=45)
                
                st.pyplot(fig)
                plt.close()
        
        with col2:
            st.subheader("Sentiment Momentum")
            
            daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
            daily_sentiment.columns = ['date', 'sentiment']
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            daily_sentiment = daily_sentiment.sort_values('date')
            
            if len(daily_sentiment) >= 3:
                # Calculate momentum (daily change)
                daily_sentiment['momentum'] = daily_sentiment['sentiment'].diff()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create momentum chart
                colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' 
                         for x in daily_sentiment['momentum']]
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
                plt.xticks(rotation=45)
                
                st.pyplot(fig)
                plt.close()
    
    def render_alerts_tab(self, df):
        """Render alert history and analysis"""
        st.header("Alert History & Key Events")
        
        if df.empty:
            st.warning("No data available for selected filters.")
            return
        
        # Calculate alerts
        daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['date', 'sentiment']
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        # Alert summary
        negative_alerts = len(daily_sentiment[daily_sentiment['sentiment'] <= -0.5])
        positive_alerts = len(daily_sentiment[daily_sentiment['sentiment'] >= 0.7])
        total_alerts = negative_alerts + positive_alerts
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Alerts", total_alerts)
        with col2:
            st.metric("Negative Alerts", negative_alerts)
        with col3:
            st.metric("Positive Surges", positive_alerts)
        
        # Alert timeline
        st.subheader("Alert Events Timeline")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot sentiment timeline
        ax.plot(daily_sentiment['date'], daily_sentiment['sentiment'], 
               color='blue', linewidth=1, alpha=0.7)
        
        # Mark alert events
        negative_alerts_data = daily_sentiment[daily_sentiment['sentiment'] <= -0.5]
        positive_alerts_data = daily_sentiment[daily_sentiment['sentiment'] >= 0.7]
        
        for _, row in negative_alerts_data.iterrows():
            ax.axvline(x=row['date'], color='red', linestyle='--', alpha=0.7)
            ax.scatter(row['date'], row['sentiment'], color='red', s=100, zorder=5)
        
        for _, row in positive_alerts_data.iterrows():
            ax.axvline(x=row['date'], color='green', linestyle='--', alpha=0.7)
            ax.scatter(row['date'], row['sentiment'], color='green', s=100, zorder=5)
        
        # Add threshold lines
        ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, label='Negative Threshold')
        ax.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='Positive Threshold')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Neutral')
        
        ax.set_title(f'Alert Events Timeline ({total_alerts} events)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        st.pyplot(fig)
        plt.close()
        
        # Recent alerts list
        st.subheader("Recent Alert Details")
        
        alerts_list = []
        for _, row in negative_alerts_data.iterrows():
            alerts_list.append({
                'date': row['date'],
                'type': 'Negative Sentiment',
                'sentiment': row['sentiment'],
                'severity': 'High' if row['sentiment'] < -0.7 else 'Medium'
            })
        
        for _, row in positive_alerts_data.iterrows():
            alerts_list.append({
                'date': row['date'],
                'type': 'Positive Surge',
                'sentiment': row['sentiment'],
                'severity': 'High' if row['sentiment'] > 0.8 else 'Medium'
            })
        
        if alerts_list:
            alerts_df = pd.DataFrame(alerts_list)
            alerts_df = alerts_df.sort_values('date', ascending=False)
            st.dataframe(alerts_df.head(10), width='stretch')  # FIXED: use_container_width -> width='stretch'
        else:
            st.info("No alerts triggered in the selected period.")
    
    def render_forecast_tab(self, df):
        """Render forecast analysis"""
        st.header("Sentiment Forecast")
        
        # Check if forecast files exist
        forecast_files = [
            "sentiment_forecast.csv", 
            "prophet_forecast.png", 
            "dashboard_forecast.png"
        ]
        
        existing_files = [f for f in forecast_files if os.path.exists(f)]
        
        if not existing_files:
            st.warning("""
            **Forecast data not available**
            
            To generate forecasts:
            1. Run the full pipeline: `python main.py full`
            2. Or run forecasting only: `python main.py forecast`
            """)
            return
        
        # Display forecast images if available
        col1, col2 = st.columns(2)
        
        if os.path.exists("dashboard_forecast.png"):
            with col1:
                st.subheader("Forecast Overview")
                st.image("dashboard_forecast.png", use_container_width=True)  # FIXED: use_column_width -> use_container_width
        
        if os.path.exists("prophet_forecast.png"):
            with col2:
                st.subheader("Forecast Components")
                st.image("prophet_forecast.png", use_container_width=True)  # FIXED: use_column_width -> use_container_width
        
        # Display forecast data if available
        if os.path.exists("sentiment_forecast.csv"):
            st.subheader("Forecast Data")
            forecast_df = pd.read_csv("sentiment_forecast.csv")
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            # Display forecast metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_forecast = forecast_df['sentiment'].mean()
                st.metric("Average Forecast", f"{avg_forecast:.3f}")
            
            with col2:
                forecast_volatility = forecast_df['sentiment'].std()
                st.metric("Forecast Volatility", f"{forecast_volatility:.3f}")
            
            with col3:
                confidence_width = (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
                st.metric("Avg Confidence Width", f"{confidence_width:.3f}")
            
            # Display forecast table
            st.dataframe(forecast_df, width='stretch')  # FIXED: use_container_width -> width='stretch'

def main():
    """Main Streamlit application"""
    dashboard = StreamlitDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()