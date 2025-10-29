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
            
            st.success(f"✅ Loaded {len(self.df)} records with sentiment analysis")
            
        except FileNotFoundError:
            st.error("❌ Data files not found. Please run the sentiment analysis pipeline first.")
            self.df = pd.DataFrame()
            self.daily_sentiment = pd.DataFrame()
    
    def run(self):
        """Main dashboard interface"""
        # Header
        st.markdown('<h1 class="main-header">📊 Strategic Intelligence Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar filters
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # === DATA COLLECTION BUTTON ===
        st.sidebar.subheader("🔄 Data Collection")
        if st.sidebar.button("🔄 Collect New Data", type="primary", use_container_width=True):
            with st.spinner("🔄 Collecting data from APIs... This may take a few minutes."):
                try:
                    from data_collector import collect_all_data
                    from data_preprocessor import clean_and_preprocess_data
                    from sentiment_analyzer import analyze_sentiment_with_finbert
                    
                    # Run data collection pipeline
                    st.sidebar.info("📥 Step 1/3: Collecting data...")
                    df_raw = collect_all_data(QUERY)
                    
                    if not df_raw.empty:
                        st.sidebar.info("🧹 Step 2/3: Preprocessing data...")
                        df_clean = clean_and_preprocess_data()
                        
                        st.sidebar.info("🎯 Step 3/3: Analyzing sentiment...")
                        df_sentiment = analyze_sentiment_with_finbert()
                        
                        st.sidebar.success("✅ Data collection complete!")
                        st.rerun()  # Refresh the dashboard with new data
                    else:
                        st.sidebar.error("❌ No data collected. Check API keys.")
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Error during data collection: {str(e)}")
        
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
        """Render alert history and analysis"""
        st.header("🚨 Alert History & Key Events")
        
        if df.empty:
            st.warning("⚠️ No data available for selected filters.")
            return
        
        # Calculate alerts
        daily_df = df.groupby(df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_df.columns = ['date', 'sentiment']
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # Alert summary
        negative_alerts = len(daily_df[daily_df['sentiment'] <= -0.5])
        positive_alerts = len(daily_df[daily_df['sentiment'] >= 0.7])
        total_alerts = negative_alerts + positive_alerts
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚨 Total Alerts", total_alerts)
        with col2:
            st.metric("🔻 Negative Alerts", negative_alerts)
        with col3:
            st.metric("🚀 Positive Surges", positive_alerts)
        
        # Interactive alert timeline
        st.subheader("📅 Alert Events Timeline")
        
        fig = go.Figure()
        
        # Add sentiment timeline
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['sentiment'],
            mode='lines',
            name='Daily Sentiment',
            line=dict(color='blue', width=2)
        ))
        
        # Mark negative alerts
        negative_data = daily_df[daily_df['sentiment'] <= -0.5]
        if not negative_data.empty:
            fig.add_trace(go.Scatter(
                x=negative_data['date'],
                y=negative_data['sentiment'],
                mode='markers',
                name='Negative Alert',
                marker=dict(color='red', size=10, symbol='diamond'),
                hovertemplate='<b>Negative Alert</b><br>Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ))
        
        # Mark positive alerts
        positive_data = daily_df[daily_df['sentiment'] >= 0.7]
        if not positive_data.empty:
            fig.add_trace(go.Scatter(
                x=positive_data['date'],
                y=positive_data['sentiment'],
                mode='markers',
                name='Positive Surge',
                marker=dict(color='green', size=10, symbol='star'),
                hovertemplate='<b>Positive Surge</b><br>Date: %{x}<br>Sentiment: %{y:.3f}<extra></extra>'
            ))
        
        # Add threshold lines
        fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.7, annotation_text="Negative Threshold")
        fig.add_hline(y=0.7, line_dash="dot", line_color="green", opacity=0.7, annotation_text="Positive Threshold")
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title=f'Alert Events Timeline ({total_alerts} events)',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent alerts table
        st.subheader("📋 Recent Alert Details")
        
        alerts_list = []
        for _, row in negative_data.iterrows():
            alerts_list.append({
                'date': row['date'],
                'type': '🔻 Negative Sentiment',
                'sentiment': row['sentiment'],
                'severity': 'High' if row['sentiment'] < -0.7 else 'Medium'
            })
        
        for _, row in positive_data.iterrows():
            alerts_list.append({
                'date': row['date'],
                'type': '🚀 Positive Surge',
                'sentiment': row['sentiment'],
                'severity': 'High' if row['sentiment'] > 0.8 else 'Medium'
            })
        
        if alerts_list:
            alerts_df = pd.DataFrame(alerts_list)
            alerts_df = alerts_df.sort_values('date', ascending=False)
            
            # Display interactive table
            st.dataframe(
                alerts_df.head(20),
                use_container_width=True,
                column_config={
                    "date": st.column_config.DatetimeColumn("📅 Date"),
                    "type": st.column_config.TextColumn("🚨 Alert Type"),
                    "sentiment": st.column_config.NumberColumn("📊 Sentiment", format="%.3f"),
                    "severity": st.column_config.TextColumn("⚠️ Severity")
                }
            )
        else:
            st.info("ℹ️ No alerts triggered in the selected period.")

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

    def render_data_explorer_tab(self, df):
        """Render interactive data explorer"""
        st.header("📋 Data Explorer")
        
        if df.empty:
            st.warning("⚠️ No data available for selected filters.")
            return
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Total Articles", len(df))
        with col2:
            st.metric("🏭 Unique Sources", df['source'].nunique() if 'source' in df.columns else 0)
        with col3:
            st.metric("📅 Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        
        # Interactive data table
        st.subheader("🔍 Article Data")
        
        # Column selector
        available_columns = ['title', 'source', 'sector', 'sentiment', 'sentiment_score', 'date', 'description']
        selected_columns = st.multiselect(
            "Select columns to display:",
            options=available_columns,
            default=['title', 'source', 'sector', 'sentiment', 'sentiment_score', 'date']
        )
        
        if selected_columns:
            display_df = df[selected_columns].copy()
            
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
        
        # Data export
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Filtered Data to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name="filtered_sentiment_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export Summary Statistics"):
                summary = df.describe(include='all').round(3)
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