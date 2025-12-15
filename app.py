import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import numpy as np
from config import QUERY
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time  # Added import
import io
import contextlib
from alert_system import check_alerts
from slack_sdk import WebClient
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()  # ensures Slack token and other env vars are loaded
# Optional temporary diagnostic to confirm:
st.write("SLACK_BOT_TOKEN found?", bool(os.getenv("SLACK_BOT_TOKEN")))
st.write("SLACK_CHANNEL found?", bool(os.getenv("SLACK_CHANNEL")))

IS_CLOUD = os.getenv('STREAMLIT_CLOUD', False) or 'STREAMLIT_SHARING' in os.environ

if IS_CLOUD:
    st.info("🌤️ Running in cloud mode with optimized settings")
    
    # Disable heavy operations in cloud
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_data_cloud():
        try:
            return pd.read_csv("industry_insights_with_financial_sentiment.csv")
        except:
            return pd.DataFrame()

# Configure the page
st.set_page_config(
    page_title="Strategic Intelligence Dashboard",
    page_icon="📊",
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
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
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
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
    def __init__(self):
        self.df = None
        self.daily_sentiment = None
    
        # Initialize session state for data persistence
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
            
        self.load_data()

    def _create_daily_sentiment(self, df):
        """Helper to create daily sentiment data from dataframe with robust error handling"""
        try:
            if df.empty:
                return pd.DataFrame()
                
            df_temp = df.copy()
            
            # Ensure date column exists
            if 'publishedAt' in df_temp.columns:
                df_temp['date'] = pd.to_datetime(df_temp['publishedAt'], errors='coerce')
            elif 'date' not in df_temp.columns:
                # If no date column, use current date
                df_temp['date'] = pd.Timestamp.now()
            
            df_temp = df_temp.dropna(subset=['date'])
            
            if df_temp.empty:
                return pd.DataFrame()
            
            # Ensure sentiment_score exists
            if 'sentiment_score' not in df_temp.columns:
                st.error("❌ sentiment_score column missing in data")
                return pd.DataFrame()
            
            # Create daily sentiment aggregation
            daily_sentiment = df_temp.groupby(df_temp['date'].dt.date).agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).reset_index()
            
            if not daily_sentiment.empty:
                daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'article_count']
                daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            
            return daily_sentiment
            
        except Exception as e:
            st.error(f"❌ Error creating daily sentiment: {e}")
            return pd.DataFrame()

    
    def load_data(self):
        """Load data with cloud compatibility and better error handling"""
        try:
            # ✅ ALWAYS load from file first after data collection to ensure fresh data
            @st.cache_data(show_spinner=False)  # removed ttl=60
            def load_sentiment_data():
                try:
                    df = pd.read_csv("industry_insights_with_financial_sentiment.csv")
                    
                    # Ensure date column exists and is properly formatted
                    if 'publishedAt' in df.columns:
                        df['date'] = pd.to_datetime(df['publishedAt'], errors='coerce')
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    else:
                        # If no date column, create one with current date
                        df['date'] = pd.Timestamp.now()
                    
                    df = df.dropna(subset=['date'])
                    
                    # Ensure sentiment_score exists
                    if 'sentiment_score' not in df.columns:
                        st.error("❌ sentiment_score column missing")
                        return pd.DataFrame(), pd.DataFrame()
                    
                    # Create daily sentiment aggregation
                    daily_sentiment = df.groupby(df['date'].dt.date).agg({
                        'sentiment_score': ['mean', 'std', 'count']
                    }).reset_index()
                    daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'article_count']
                    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
                    
                    return df, daily_sentiment
                except FileNotFoundError:
                    st.info("No data file found")
                    return pd.DataFrame(), pd.DataFrame()
                except Exception as e:
                    st.error(f"Error loading data: {e}")
                    return pd.DataFrame(), pd.DataFrame()
            
            # Load from file
            self.df, self.daily_sentiment = load_sentiment_data()
            
            # Update session state with loaded data
            if not self.df.empty and not self.daily_sentiment.empty:
                st.session_state.current_data = (self.df.copy(), self.daily_sentiment.copy())
                st.session_state.data_loaded = True
                st.session_state.last_update = datetime.now()
                st.success(f"✅ Loaded {len(self.df)} records with sentiment analysis")
            else:
                st.info("🔍 No data found. Click 'Collect New Data' to get started!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            # Initialize with empty dataframes
            self.df = pd.DataFrame()
            self.daily_sentiment = pd.DataFrame()

    def run_data_collection_pipeline(self):
        """Run complete data collection pipeline with progress updates"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔥 Step 1/4: Collecting data from APIs...")
            from data_collector import collect_all_data
            df_raw = collect_all_data(QUERY)
            progress_bar.progress(25)
            
            if not df_raw.empty:
                status_text.text("🧹 Step 2/4: Preprocessing data...")
                from data_preprocessor import clean_and_preprocess_data
                df_clean = clean_and_preprocess_data()
                progress_bar.progress(50)
                
                status_text.text("🎯 Step 3/4: Analyzing sentiment...")
                from sentiment_analyzer import analyze_sentiment_with_finbert
                df_sentiment = analyze_sentiment_with_finbert()
                progress_bar.progress(75)
                
                status_text.text("📊 Step 4/4: Updating dashboard data...")
                
                if not df_sentiment.empty:
                    # Verify the file was saved successfully
                    import os
                    if os.path.exists("industry_insights_with_financial_sentiment.csv"):
                        progress_bar.progress(100)
                        status_text.text("✅ Data collection complete! Refreshing dashboard...")
                        
                        # Clear ALL caches to force fresh load
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        
                        # Clear session state to force reload from file
                        if 'current_data' in st.session_state:
                            del st.session_state.current_data
                        if 'data_loaded' in st.session_state:
                            del st.session_state.data_loaded
                        
                        st.balloons()
                        
                        # Small delay to show completion message
                        import time
                        time.sleep(2)

                        # ✅ Show success message before rerun
                        st.success(f"✅ Successfully collected {len(df_sentiment)} articles with sentiment analysis!")
                        st.info("🔄 Dashboard will refresh in 2 seconds...")
                        time.sleep(2)
                        
                        # Force complete reload
                        st.rerun()
                        return True
                    else:
                        status_text.text("❌ Data file not saved properly")
                        progress_bar.progress(0)
                        return False
                else:
                    status_text.text("❌ Sentiment analysis returned no data")
                    progress_bar.progress(0)
                    return False
            else:
                status_text.text("❌ No data collected")
                progress_bar.progress(0)
                return False
                
        except Exception as e:
            status_text.text(f"❌ Error: {str(e)}")
            progress_bar.progress(0)
            import traceback
            st.error(traceback.format_exc())
            return False

    def run(self):
        """Main dashboard interface"""
        # Header
        st.markdown('<h1 class="main-header">📊 Strategic Intelligence Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar filters
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # === ENHANCED DATA COLLECTION SECTION ===
        st.sidebar.subheader("🔄 Data Collection")
        
        # Data status
        if not self.df.empty:
            st.sidebar.metric("📊 Current Articles", f"{len(self.df):,}")
            
            try:
                latest_date = self.df['date'].max()
                data_age = (pd.Timestamp.now() - latest_date).days
                
                st.sidebar.metric(
                    "Latest Data", 
                    latest_date.strftime('%Y-%m-%d'),
                    delta=f"{data_age} days ago" if data_age > 0 else "Today"
                )
                
                # Data freshness indicator
                if data_age <= 1:
                    st.sidebar.success("🟢 Data is fresh")
                elif data_age <= 3:
                    st.sidebar.warning("🟡 Data is getting stale")
                else:
                    st.sidebar.error("🔴 Data needs refresh")
                    
            except Exception as e:
                st.sidebar.info("📊 Data loaded")
        else:
            st.sidebar.metric("📊 Current Articles", "0")
            st.sidebar.info("📭 No data available")

        if st.sidebar.button("🔄 Collect New Data", type="primary", use_container_width=True):
            success = self.run_data_collection_pipeline()
            if not success:
                st.sidebar.error("❌ Data collection failed")

        # Manual refresh
        if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            st.sidebar.success("✅ Dashboard refreshed!")

        # Auto refresh option
        st.sidebar.subheader("⚡ Auto Refresh")
        auto_refresh = st.sidebar.checkbox("Auto-refresh after data collection", value=True)

        # Data status
        if self.df.empty:
            st.sidebar.error("📭 No data available")
            st.sidebar.info("💡 Click the button above to collect data")
            return
        
        # Filters
        st.sidebar.subheader("🔍 Filters")
        date_range = self.get_date_filters()
        sector_filter = self.get_sector_filters()
        competitor_filter = self.get_competitor_filters()
        sentiment_filter = self.get_sentiment_filters()
        
        # Apply filters
        filtered_df = self.apply_filters(date_range, sector_filter, competitor_filter, sentiment_filter)
        
        # Display filter summary
        self.display_filter_summary(filtered_df, date_range, sector_filter, competitor_filter, sentiment_filter)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Overview", 
            "🏢 Competitor Analysis", 
            "📊 Trend Evolution", 
            "🚨 Alert History", 
            "🔮 Forecast",
            "📋 Data Explorer"
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
            
        with tab6:
            self.render_data_explorer_tab(filtered_df)
    
    def get_date_filters(self):
        """Date range filter"""
        if self.df.empty:
            return (datetime.now().date() - timedelta(days=30), datetime.now().date())
            
        min_date = self.df['date'].min().date()
        max_date = self.df['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "📅 Select date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            return date_range
        return (min_date, max_date)
    
    def get_sector_filters(self):
        """Sector/focus area filter"""
        if self.df.empty:
            return "All"
            
        # Get unique sectors from data
        if 'sector' in self.df.columns:
            sectors = ["All"] + sorted(self.df['sector'].unique().tolist())
        else:
            sectors = ["All", "Technology", "Finance", "Healthcare", "Energy", "Retail", "Manufacturing", "General"]
        
        selected_sector = st.sidebar.selectbox("🏭 Select sector:", sectors)
        
        return selected_sector
    
    def get_competitor_filters(self):
        """Competitor/source filter"""
        if self.df.empty or 'source' not in self.df.columns:
            return []
            
        sources = self.df['source'].value_counts().head(20).index.tolist()
        selected_sources = st.sidebar.multiselect(
            "🎯 Select competitors/sources to track:",
            options=sources,
            default=sources[:5] if sources else [],
            help="Select specific sources or competitors to analyze"
        )
        return selected_sources
    
    def get_sentiment_filters(self):
        """Sentiment filter"""
        sentiments = ["All", "Positive", "Negative", "Neutral"]
        selected_sentiment = st.sidebar.selectbox("😊 Filter by sentiment:", sentiments)
        
        sentiment_map = {
            "All": None,
            "Positive": "positive",
            "Negative": "negative", 
            "Neutral": "neutral"
        }
        
        return sentiment_map[selected_sentiment]
    
    def apply_filters(self, date_range, sector, competitors, sentiment):
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
        
        # Sector filter
        if sector != "All" and 'sector' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sector'] == sector]
        
        # Sentiment filter
        if sentiment and 'sentiment' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sentiment'] == sentiment]
        
        return filtered_df
    
    def display_filter_summary(self, filtered_df, date_range, sector, competitors, sentiment):
        """Display summary of applied filters"""
        if filtered_df.empty:
            st.warning("⚠️ No data available for selected filters.")
            return
            
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📄 Filtered Articles", len(filtered_df))
        
        with col2:
            start_date, end_date = date_range
            st.metric("📅 Date Range", f"{start_date}\nto {end_date}")
        
        with col3:
            st.metric("🏭 Selected Sector", sector)
        
        with col4:
            comp_count = len(competitors) if competitors else "All"
            st.metric("🎯 Competitors Tracked", comp_count)
            
        with col5:
            sent_display = sentiment if sentiment else "All"
            st.metric("😊 Sentiment Filter", sent_display)
        
        st.markdown("---")
    
    def render_overview_tab(self, df):
        """Render overview dashboard with interactive charts"""
        st.header("📈 Sentiment Overview")
        
        if df.empty:
            st.warning("⚠️ No data available for selected filters.")
            return
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_sentiment = df['sentiment_score'].mean()
            delta_color = "normal"
            if avg_sentiment > 0.1:
                delta_color = "normal"
            elif avg_sentiment < -0.1:
                delta_color = "inverse"
            st.metric(
                "📊 Average Sentiment", 
                f"{avg_sentiment:.3f}",
                delta=f"{avg_sentiment:.3f}" if abs(avg_sentiment) > 0.01 else "Neutral",
                delta_color=delta_color
            )
        
        with col2:
            total_articles = len(df)
            st.metric("📄 Total Articles", f"{total_articles:,}")
        
        with col3:
            positive_pct = (df['sentiment'] == 'positive').sum() / len(df) * 100
            st.metric("😊 Positive Articles", f"{positive_pct:.1f}%")
        
        with col4:
            negative_pct = (df['sentiment'] == 'negative').sum() / len(df) * 100
            st.metric("😔 Negative Articles", f"{negative_pct:.1f}%")
            
        with col5:
            volatility = df['sentiment_score'].std()
            st.metric("📉 Sentiment Volatility", f"{volatility:.3f}")
        
        # Interactive Sentiment Timeline
        st.subheader("📅 Interactive Sentiment Timeline")
        
        # Prepare daily data
        daily_df = df.groupby(df['date'].dt.date).agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()
        daily_df.columns = ['date', 'sentiment', 'std', 'count']
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add sentiment line
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['sentiment'],
            mode='lines+markers',
            name='Daily Sentiment',
            line=dict(color='blue', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br><b>Sentiment:</b> %{y:.3f}<extra></extra>'
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['sentiment'] + daily_df['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['sentiment'] - daily_df['std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,255,0.2)',
            name='Std Dev',
            hovertemplate='<b>Date:</b> %{x}<br><b>Std Dev:</b> ±%{customdata:.3f}<extra></extra>',
            customdata=daily_df['std']
        ))
        
        # Add moving average if enough data
        if len(daily_df) >= 7:
            daily_df['ma_7'] = daily_df['sentiment'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['ma_7'],
                mode='lines',
                name='7-day Moving Average',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>7-day MA:</b> %{y:.3f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Sentiment Timeline',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        # Add neutral line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution and source performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Distribution")
            
            # Interactive pie chart
            sentiment_counts = df['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545', 
                    'neutral': '#6c757d'
                },
                hole=0.4
            )
            fig_pie.update_traces(
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            fig_pie.update_layout(
                title='Sentiment Distribution',
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("🏢 Source Performance")
            if 'source' in df.columns and len(df) > 0:
                source_metrics = df.groupby('source').agg({
                    'sentiment_score': ['mean', 'count']
                }).reset_index()
                source_metrics.columns = ['source', 'avg_sentiment', 'count']
                source_metrics = source_metrics[source_metrics['count'] >= 3]
                
                if not source_metrics.empty:
                    # Interactive bar chart
                    fig_bar = px.bar(
                        source_metrics.sort_values('avg_sentiment'),
                        x='avg_sentiment',
                        y='source',
                        orientation='h',
                        color='avg_sentiment',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=0,
                        hover_data=['count']
                    )
                    fig_bar.update_layout(
                        title='Average Sentiment by Source',
                        xaxis_title='Average Sentiment Score',
                        yaxis_title='Source',
                        height=400,
                        showlegend=False
                    )
                    fig_bar.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("ℹ️ Not enough data for source analysis")
            else:
                st.info("ℹ️ No source data available")

    def render_competitor_tab(self, df):
        """Render competitor analysis with interactive charts"""
        st.header("🏢 Competitor & Source Analysis")
        
        if df.empty:
            st.warning("⚠️ No data available for selected filters.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Market Share by Source")
            if 'source' in df.columns:
                source_counts = df['source'].value_counts().head(10)
                
                # Interactive treemap
                fig_treemap = px.treemap(
                    names=source_counts.index,
                    parents=[''] * len(source_counts),
                    values=source_counts.values,
                    title='Market Share by Source (Article Volume)'
                )
                fig_treemap.update_traces(
                    textinfo='label+value+percent parent',
                    hovertemplate='<b>%{label}</b><br>Articles: %{value}<br>Market Share: %{percentParent:.1%}<extra></extra>'
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
        
        with col2:
            st.subheader("📊 Source Sentiment Timeline")
            if 'source' in df.columns:
                # Get top sources
                top_sources = df['source'].value_counts().head(5).index
                
                # Prepare data for timeline
                source_timeline_data = []
                for source in top_sources:
                    source_data = df[df['source'] == source]
                    daily_sentiment = source_data.groupby(source_data['date'].dt.date)['sentiment_score'].mean().reset_index()
                    daily_sentiment['source'] = source
                    source_timeline_data.append(daily_sentiment)
                
                if source_timeline_data:
                    timeline_df = pd.concat(source_timeline_data, ignore_index=True)
                    
                    # Interactive line chart
                    fig_timeline = px.line(
                        timeline_df,
                        x='date',
                        y='sentiment_score',
                        color='source',
                        title='Sentiment Timeline by Source (Top 5)',
                        markers=True
                    )
                    fig_timeline.update_layout(
                        xaxis_title='Date',
                        yaxis_title='Average Sentiment Score',
                        height=500
                    )
                    fig_timeline.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                    st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Source performance ranking
        st.subheader("🥇 Source Performance Ranking")
        if 'source' in df.columns:
            source_metrics = df.groupby('source').agg({
                'sentiment_score': ['mean', 'std', 'count']
            }).reset_index()
            source_metrics.columns = ['source', 'avg_sentiment', 'sentiment_volatility', 'article_count']
            source_metrics = source_metrics[source_metrics['article_count'] >= 5]
            
            if not source_metrics.empty:
                # Calculate performance score
                source_metrics['performance_score'] = (
                    source_metrics['avg_sentiment'] - 
                    (source_metrics['sentiment_volatility'] * 0.3)
                )
                source_metrics = source_metrics.sort_values('performance_score')
                
                # Interactive bar chart
                fig_performance = px.bar(
                    source_metrics,
                    x='performance_score',
                    y='source',
                    orientation='h',
                    color='performance_score',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    hover_data=['avg_sentiment', 'sentiment_volatility', 'article_count'],
                    title='Source Performance Ranking (Sentiment - 0.3×Volatility)'
                )
                fig_performance.update_layout(
                    xaxis_title='Performance Score',
                    yaxis_title='Source',
                    height=400
                )
                fig_performance.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
                st.plotly_chart(fig_performance, use_container_width=True)

    def render_trend_tab(self, df):
        """Render trend evolution analysis with interactive charts"""
        st.header("📊 Trend Evolution Analysis")
        
        if df.empty:
            st.warning("⚠️ No data available for selected filters.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Rolling Sentiment Trends")
            
            # Prepare daily data
            daily_df = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
            daily_df.columns = ['date', 'sentiment']
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_df = daily_df.sort_values('date')
            
            if len(daily_df) >= 3:
                # Calculate rolling averages
                daily_df['ma_3'] = daily_df['sentiment'].rolling(window=3, center=True).mean()
                if len(daily_df) >= 7:
                    daily_df['ma_7'] = daily_df['sentiment'].rolling(window=7, center=True).mean()
                if len(daily_df) >= 14:
                    daily_df['ma_14'] = daily_df['sentiment'].rolling(window=14, center=True).mean()
                
                # Create interactive plot
                fig = go.Figure()
                
                # Add daily sentiment
                fig.add_trace(go.Scatter(
                    x=daily_df['date'],
                    y=daily_df['sentiment'],
                    mode='lines',
                    name='Daily',
                    line=dict(color='lightgray', width=1),
                    opacity=0.7
                ))
                
                # Add moving averages
                if 'ma_3' in daily_df.columns:
                    fig.add_trace(go.Scatter(
                        x=daily_df['date'],
                        y=daily_df['ma_3'],
                        mode='lines',
                        name='3-day MA',
                        line=dict(color='blue', width=2)
                    ))
                
                if 'ma_7' in daily_df.columns:
                    fig.add_trace(go.Scatter(
                        x=daily_df['date'],
                        y=daily_df['ma_7'],
                        mode='lines',
                        name='7-day MA',
                        line=dict(color='red', width=3)
                    ))
                
                if 'ma_14' in daily_df.columns:
                    fig.add_trace(go.Scatter(
                        x=daily_df['date'],
                        y=daily_df['ma_14'],
                        mode='lines',
                        name='14-day MA',
                        line=dict(color='green', width=3)
                    ))
                
                fig.update_layout(
                    title='Rolling Sentiment Trends',
                    xaxis_title='Date',
                    yaxis_title='Sentiment Score',
                    height=500,
                    hovermode='x unified'
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🚀 Sentiment Momentum")
            
            if len(daily_df) >= 3:
                # Calculate momentum
                daily_df['momentum'] = daily_df['sentiment'].diff()
                
                # Create momentum chart
                fig_momentum = px.bar(
                    daily_df.iloc[1:],  # Skip first row with NaN
                    x='date',
                    y='momentum',
                    color='momentum',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0,
                    title='Sentiment Momentum (Daily Change)'
                )
                fig_momentum.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Momentum (Δ Sentiment)',
                    height=500,
                    showlegend=False
                )
                fig_momentum.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                st.plotly_chart(fig_momentum, use_container_width=True)



    def render_alerts_tab(self, df):
        """Render alert history, timeline, and auto-display alert_system logs on load."""
        st.header("🚨 Alert History & Key Events")
        
        if df.empty:
            st.warning("⚠️ No data available for selected filters.")
            return
        
        # Calculate daily average sentiment (same as before)
        daily_df = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_df.columns = ['date', 'avg_sentiment']
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.sort_values('date')
        daily_df['volatility'] = daily_df['avg_sentiment'].rolling(window=7, min_periods=1).std(ddof=1)

        # -------------------------------------------------------------------------
        # 🆕 AUTO DISPLAY ALERT_SYSTEM LOGS WHEN TAB LOADS
        # -------------------------------------------------------------------------
        st.subheader("🖥️ System Alert Logs (Auto-Generated)")

        log_stream = io.StringIO()
        with contextlib.redirect_stdout(log_stream):
            try:
                check_alerts(daily_df)  # run real alert logic from alert_system.py
            except Exception as e:
                print(f"❌ Error running alert_system: {e}")
        log_output = log_stream.getvalue()

        if log_output.strip():
            with st.expander("📋 View Alert System Logs", expanded=True):
                st.text(log_output)
        else:
            st.info("✅ No system alerts generated on initial load.")

        # -------------------------------------------------------------------------
        # METRICS SECTION
        # -------------------------------------------------------------------------
        negative_alerts = len(daily_df[daily_df['avg_sentiment'] <= -0.2])
        positive_alerts = len(daily_df[daily_df['avg_sentiment'] >= 0.3])
        volatility_alerts = len(daily_df[daily_df['volatility'] > 0.15])
        mild_negative = len(daily_df[(daily_df['avg_sentiment'] <= -0.05) & (daily_df['avg_sentiment'] >= -0.15)])
        mild_positive = len(daily_df[(daily_df['avg_sentiment'] >= 0.05) & (daily_df['avg_sentiment'] <= 0.15)])
        total_alerts = negative_alerts + positive_alerts + volatility_alerts + mild_negative + mild_positive

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("🚨 Total Alerts", total_alerts)
        with col2: st.metric("🔻 Negative Alerts", negative_alerts)
        with col3: st.metric("🚀 Positive Surges", positive_alerts)
        with col4: st.metric("⚡ Volatility Alerts", volatility_alerts)
        with col5: st.metric("⚠️ Mild Alerts", mild_negative + mild_positive)

        # -------------------------------------------------------------------------
        # CURRENT ALERT STATUS
        # -------------------------------------------------------------------------
        st.subheader("🔍 Current Alert Status")
        if not daily_df.empty:
            latest_data = daily_df.iloc[-1]
            current_sentiment = latest_data['avg_sentiment']
            current_volatility = latest_data['volatility']

            # Determine sentiment status
            if current_sentiment <= -0.2:
                sentiment_status, sentiment_color = "🔻 NEGATIVE ALERT", "red"
            elif current_sentiment >= 0.3:
                sentiment_status, sentiment_color = "🚀 POSITIVE SURGE", "green"
            elif -0.15 <= current_sentiment <= -0.05:
                sentiment_status, sentiment_color = "⚠️ MILD NEGATIVE", "orange"
            elif 0.05 <= current_sentiment <= 0.15:
                sentiment_status, sentiment_color = "📈 MILD POSITIVE", "lightgreen"
            else:
                sentiment_status, sentiment_color = "✅ NEUTRAL", "blue"

            # Volatility status
            if current_volatility > 0.3:
                vol_status, vol_color = "⚡ HIGH VOLATILITY", "red"
            elif current_volatility > 0.2:
                vol_status, vol_color = "⚠️ MEDIUM VOLATILITY", "orange"
            elif current_volatility > 0.15:
                vol_status, vol_color = "📈 ELEVATED VOLATILITY", "yellow"
            else:
                vol_status, vol_color = "✅ NORMAL VOLATILITY", "green"

            # Trend status
            if len(daily_df) >= 3:
                recent_trend = daily_df['avg_sentiment'].tail(3).mean() - daily_df['avg_sentiment'].iloc[-4:-1].mean()
                if abs(recent_trend) > 0.1:
                    trend_status, trend_color = "📊 TREND CHANGE", "orange"
                else:
                    trend_status, trend_color = "📊 STABLE TREND", "blue"
            else:
                trend_status, trend_color, recent_trend = "📊 INSUFFICIENT DATA", "gray", 0

            # Display current status cards
            current_col1, current_col2, current_col3, current_col4 = st.columns(4)
            with current_col1:
                st.markdown(f"<div style='background-color:{sentiment_color}20;padding:15px;border-radius:10px;border-left:5px solid {sentiment_color};'><h4 style='margin:0;color:{sentiment_color};'>{sentiment_status}</h4><p style='margin:5px 0 0 0;font-size:18px;font-weight:bold;'>Sentiment: {current_sentiment:.3f}</p></div>", unsafe_allow_html=True)
            with current_col2:
                st.markdown(f"<div style='background-color:{vol_color}20;padding:15px;border-radius:10px;border-left:5px solid {vol_color};'><h4 style='margin:0;color:{vol_color};'>{vol_status}</h4><p style='margin:5px 0 0 0;font-size:18px;font-weight:bold;'>Volatility: {current_volatility:.3f}</p></div>", unsafe_allow_html=True)
            with current_col3:
                st.markdown(f"<div style='background-color:{trend_color}20;padding:15px;border-radius:10px;border-left:5px solid {trend_color};'><h4 style='margin:0;color:{trend_color};'>{trend_status}</h4><p style='margin:5px 0 0 0;font-size:18px;font-weight:bold;'>Trend: {recent_trend:+.3f}</p></div>", unsafe_allow_html=True)
            with current_col4:
                st.markdown(f"<div style='background-color:#6c757d20;padding:15px;border-radius:10px;border-left:5px solid #6c757d;'><h4 style='margin:0;color:#6c757d;'>📅 LAST UPDATE</h4><p style='margin:5px 0 0 0;font-size:18px;font-weight:bold;'>{latest_data['date'].strftime('%Y-%m-%d')}</p></div>", unsafe_allow_html=True)

        st.subheader("🧾 Alert System Logs")
        log_file = "alert_logs.txt"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                log_content = f.read().strip()
            if log_content:
                log_lines = log_content.splitlines()
                st.text_area("📜 All Alert Logs", "\n".join(log_lines), height=300)
            else:
                st.info("ℹ️ Log file is empty. No alerts recorded yet.")
        else:
            st.info("ℹ️ Log file not found. Run the alert system to generate logs.")
        # -------------------------------------------------------------------------
        # ALERT TIMELINE (your existing Plotly code)
        # -------------------------------------------------------------------------
        st.subheader("📅 Alert Events Timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['avg_sentiment'],
                                mode='lines+markers', name='Daily Sentiment',
                                line=dict(color='blue', width=3),
                                marker=dict(size=4),
                                hovertemplate='Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['volatility'],
                                mode='lines', name='7-day Volatility',
                                line=dict(color='orange', width=2, dash='dash'),
                                yaxis='y2',
                                hovertemplate='Date: %{x}<br>Volatility: %{y:.3f}<extra></extra>'))

        alert_data_sources = [
            (daily_df[daily_df['avg_sentiment'] <= -0.2], 'red', 'diamond', '🔻 Negative Alert'),
            (daily_df[daily_df['avg_sentiment'] >= 0.3], 'green', 'star', '🚀 Positive Surge'),
            (daily_df[daily_df['volatility'] > 0.15], 'purple', 'circle', '⚡ Volatility Alert'),
            (daily_df[(daily_df['avg_sentiment'] <= -0.05) & (daily_df['avg_sentiment'] >= -0.15)], 'orange', 'square', '⚠️ Mild Negative'),
            (daily_df[(daily_df['avg_sentiment'] >= 0.05) & (daily_df['avg_sentiment'] <= 0.15)], 'lightgreen', 'triangle-up', '📈 Mild Positive')
        ]
        for alert_data, color, symbol, name in alert_data_sources:
            if not alert_data.empty:
                fig.add_trace(go.Scatter(x=alert_data['date'], y=alert_data['avg_sentiment'],
                                        mode='markers', name=name,
                                        marker=dict(color=color, size=12, symbol=symbol, line=dict(width=2, color='white')),
                                        hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Sentiment: %{{y:.3f}}<extra></extra>'))
        fig.add_hline(y=-0.2, line_dash="dot", line_color="red", opacity=0.7, annotation_text="Negative Threshold")
        fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.7, annotation_text="Positive Threshold")
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        fig.add_hline(y=0.15, line_dash="dot", line_color="purple", opacity=0.7, annotation_text="Volatility Threshold", yref="y2")
        fig.update_layout(title=f'Alert Events Timeline ({total_alerts} events)',
                        xaxis_title='Date',
                        yaxis_title='Sentiment Score',
                        yaxis2=dict(title='Volatility', overlaying='y', side='right', showgrid=False),
                        height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------------------
        # RECENT ALERTS TABLE (unchanged)
        # -------------------------------------------------------------------------
        st.subheader("📋 Recent Alert Details")
        alerts_list = []
        for _, row in daily_df[daily_df['avg_sentiment'] <= -0.2].iterrows():
            severity = "HIGH" if row['avg_sentiment'] <= -0.3 else "MEDIUM"
            alerts_list.append({'date': row['date'], 'type': '🔻 Negative Sentiment',
                                'sentiment': row['avg_sentiment'], 'volatility': row.get('volatility', 0),
                                'severity': severity})
        for _, row in daily_df[daily_df['avg_sentiment'] >= 0.3].iterrows():
            severity = "HIGH" if row['avg_sentiment'] >= 0.5 else "MEDIUM"
            alerts_list.append({'date': row['date'], 'type': '🚀 Positive Surge',
                                'sentiment': row['avg_sentiment'], 'volatility': row.get('volatility', 0),
                                'severity': severity})
        for _, row in daily_df[daily_df['volatility'] > 0.15].iterrows():
            severity = "HIGH" if row['volatility'] > 0.3 else "MEDIUM"
            alerts_list.append({'date': row['date'], 'type': '⚡ Volatility Alert',
                                'sentiment': row['avg_sentiment'], 'volatility': row['volatility'],
                                'severity': severity})
        for _, row in daily_df[(daily_df['avg_sentiment'] <= -0.05) & (daily_df['avg_sentiment'] >= -0.15)].iterrows():
            alerts_list.append({'date': row['date'], 'type': '⚠️ Mild Negative',
                                'sentiment': row['avg_sentiment'], 'volatility': row.get('volatility', 0),
                                'severity': 'LOW'})
        for _, row in daily_df[(daily_df['avg_sentiment'] >= 0.05) & (daily_df['avg_sentiment'] <= 0.15)].iterrows():
            alerts_list.append({'date': row['date'], 'type': '📈 Mild Positive',
                                'sentiment': row['avg_sentiment'], 'volatility': row.get('volatility', 0),
                                'severity': 'LOW'})

        if alerts_list:
            alerts_df = pd.DataFrame(alerts_list).sort_values('date', ascending=False)
            st.dataframe(alerts_df.head(20), use_container_width=True)
            csv = alerts_df.to_csv(index=False)
            st.download_button("📥 Export Alerts to CSV", data=csv,
                            file_name="alert_history.csv", mime="text/csv")
        else:
            st.info("ℹ️ No alerts triggered in the selected period.")

        # -------------------------------------------------------------------------
        # MANUAL ALERT CHECK BUTTON (unchanged)
        # -------------------------------------------------------------------------
        st.subheader("🔄 Manual Alert Check")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚨 Run Alert Check Now", type="primary"):
                with st.spinner("Checking for alerts..."):
                    try:
                        token = os.getenv("SLACK_BOT_TOKEN")
                        channel = os.getenv("SLACK_CHANNEL")
                        if not token or not channel:
                            st.error("❌ Slack credentials missing. Please check your .env or Streamlit secrets.")
                        else:
                            client = WebClient(token=token)
                            try:
                                auth = client.auth_test()
                                st.success(f"✅ Slack connected to {auth.get('team')} as {auth.get('user')}")
                            except Exception as e:
                                st.warning(f"⚠️ Slack auth failed: {e}")
                        df_alert = df.copy()
                        df_alert['date'] = pd.to_datetime(df_alert['publishedAt'], errors='coerce')
                        daily_sentiment_alert = df_alert.groupby(df_alert['date'].dt.date)['sentiment_score'].mean().reset_index()
                        daily_sentiment_alert.columns = ['date', 'avg_sentiment']
                        daily_sentiment_alert['date'] = pd.to_datetime(daily_sentiment_alert['date'])
                        check_alerts(daily_sentiment_alert)
                        st.success("✅ Alert check completed! Check Slack for messages.")
                        st.cache_data.clear()
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error running alert check: {e}")
        with col2:
            st.info("This will run the same alert system that sends notifications to Slack and refresh this page.")


    def render_forecast_tab(self, df):
        """Render forecast analysis"""
        st.header("🔮 Sentiment Forecast")
        
        # Check if forecast files exist
        forecast_files = [
            "sentiment_forecast.csv", 
            "interactive_forecast.html",
            "prophet_forecast.png"
        ]
        
        existing_files = [f for f in forecast_files if os.path.exists(f)]
        
        if not existing_files:
            st.warning("""
            **📭 Forecast data not available**
            
            To generate forecasts:
            1. Run the full pipeline: `python main.py full`
            2. Or run forecasting only: `python main.py forecast`
            """)
            
            if st.button("🔮 Generate Forecast Now"):
                with st.spinner("🔄 Generating forecast... This may take a few minutes."):
                    try:
                        from forecasting import forecast_sentiment
                        forecasts, forecast_df, daily_data = forecast_sentiment()
                        if forecasts:
                            st.success("✅ Forecast generated successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Forecast generation failed.")
                    except Exception as e:
                        st.error(f"❌ Forecast error: {e}")
            return
        
        # Display interactive forecast if available
        if os.path.exists("interactive_forecast.html"):
            st.subheader("📈 Interactive Forecast")
            with open("interactive_forecast.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        
        # Display forecast data if available
        if os.path.exists("sentiment_forecast.csv"):
            st.subheader("📊 Forecast Data")
            forecast_df = pd.read_csv("sentiment_forecast.csv")
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            # Display forecast metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_forecast = forecast_df['sentiment'].mean()
                st.metric("📈 Average Forecast", f"{avg_forecast:.3f}")
            
            with col2:
                forecast_volatility = forecast_df['sentiment'].std()
                st.metric("📉 Forecast Volatility", f"{forecast_volatility:.3f}")
            
            with col3:
                confidence_width = (forecast_df['upper_bound'] - forecast_df['lower_bound']).mean()
                st.metric("🎯 Avg Confidence Width", f"{confidence_width:.3f}")
            
            # Display forecast table
            st.dataframe(
                forecast_df,
                use_container_width=True,
                column_config={
                    "date": st.column_config.DatetimeColumn("📅 Date"),
                    "sentiment": st.column_config.NumberColumn("📊 Forecast", format="%.3f"),
                    "lower_bound": st.column_config.NumberColumn("🔽 Lower Bound", format="%.3f"),
                    "upper_bound": st.column_config.NumberColumn("🔼 Upper Bound", format="%.3f")
                }
            )
            
        # Display forecast image if available
        if os.path.exists("prophet_forecast.png"):
            st.subheader("📈 Forecast Visualization")
            st.image("prophet_forecast.png", use_container_width=True)
    
    def render_data_explorer_tab(self, df):
        """Render interactive data explorer - FIXED to show latest data"""
        st.header("📋 Data Explorer")
            
        # Use self.df (latest data) instead of filtered_df for data explorer
        if self.df.empty:
            st.warning("⚠️ No data available.")
            return
            
        # Data summary - Show both total and filtered counts
        col1, col2, col3, col4 = st.columns(4)
            
        with col1:
            st.metric("📄 Total Articles", len(self.df))
        with col2:
            st.metric("🔍 Filtered Articles", len(df) if not df.empty else 0)
        with col3:
            st.metric("🏭 Unique Sources", self.df['source'].nunique() if 'source' in self.df.columns else 0)
        with col4:
            st.metric("📅 Date Range", f"{self.df['date'].min().date()} to {self.df['date'].max().date()}")
            
        # Show data freshness
        latest_date = self.df['date'].max()
        st.info(f"📅 **Latest data**: {latest_date.strftime('%Y-%m-%d %H:%M')}")
            
        # Interactive data table
        st.subheader("🔍 Article Data")
            
        # Let user choose between filtered view and all data
        view_option = st.radio(
                "Select data view:",
                ["All Data", "Filtered View (applies your sidebar filters)"],
                horizontal=True
            )
            
        # Use either all data or filtered data based on selection
        display_data = self.df if view_option == "All Data" else df
            
        if display_data.empty:
            st.warning("No data available for the selected view.")
            return
            
        # Column selector
        available_columns = ['title', 'source', 'sector', 'sentiment', 'sentiment_score', 'date', 'description']
        selected_columns = st.multiselect(
                "Select columns to display:",
                options=available_columns,
                default=['title', 'source', 'sector', 'sentiment', 'sentiment_score', 'date']
            )
            
        if selected_columns:
            display_df = display_data[selected_columns].copy()
                
            # Configure columns for better display
            column_config = {}
            if 'date' in selected_columns:
                column_config['date'] = st.column_config.DatetimeColumn("📅 Date")
            if 'sentiment_score' in selected_columns:
                column_config['sentiment_score'] = st.column_config.NumberColumn("📊 Sentiment", format="%.3f")
            if 'title' in selected_columns:
                column_config['title'] = st.column_config.TextColumn("📰 Title", width="large")
            if 'source' in selected_columns:
                column_config['source'] = st.column_config.TextColumn("🏢 Source")
            if 'sector' in selected_columns:
                column_config['sector'] = st.column_config.TextColumn("🏭 Sector")
            if 'sentiment' in selected_columns:
                column_config['sentiment'] = st.column_config.TextColumn("😊 Sentiment")
                
            st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config=column_config,
                    hide_index=True
                )
                
            # Show data stats
            st.metric("Displaying Articles", len(display_df))
            
            # Data export
            st.subheader("💾 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export Current View to CSV"):
                    csv = display_data.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name="current_view_data.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Export Summary Statistics"):
                    summary = display_data.describe(include='all').round(3)
                    st.dataframe(summary, use_container_width=True)

def main():
    """Main Streamlit application"""
    try:
        dashboard = StreamlitDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        st.info("💡 Please check if all data files are available and try collecting data first.")

if __name__ == "__main__":
    main()