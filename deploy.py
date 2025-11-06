"""
Deployment script for the Strategic Intelligence Dashboard
"""

import os
import sys
import subprocess
import webbrowser
import threading
import time

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
        'requests', 'python-dotenv', 'transformers', 'torch',
        'prophet', 'scikit-learn', 'praw', 'feedparser'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(missing_packages):
    """Install missing packages"""
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
        print("🎉 Installation completed!")
    else:
        print("✅ All required packages are already installed!")

def setup_environment():
    """Set up environment variables"""
    if not os.path.exists('.env'):
        print("📝 Creating .env template file...")
        with open('.env', 'w') as f:
            f.write("""# API Keys for Strategic Intelligence Dashboard
# Get your API keys from the respective services

# NewsAPI (https://newsapi.org)
NEWS_API_KEY=your_newsapi_key_here

# SerpAPI (https://serpapi.com) 
SERP_API_KEY=your_serpapi_key_here

# Twitter API (https://developer.twitter.com)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=strategic_intelligence_dashboard_v1.0

# Slack API (https://api.slack.com)
SLACK_BOT_TOKEN=your_slack_bot_token_here
SLACK_CHANNEL=#alerts

# Query parameters
QUERY=industry trends OR competitor analysis OR market insights OR Artificial Intelligence OR finance OR technology OR healthcare OR energy OR retail OR manufacturing
MAX_RESULTS=100
""")
        print("✅ Created .env template file")
        print("⚠️  Please edit .env file with your actual API keys")
    else:
        print("✅ .env file already exists")

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        "industry_insights_with_financial_sentiment.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Strategic Intelligence Dashboard...")
    print("🌐 The dashboard will open in your browser automatically.")
    print("📱 If it doesn't open, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main deployment function"""
    print("=" * 60)
    print("🤖 STRATEGIC INTELLIGENCE DASHBOARD - DEPLOYMENT")
    print("=" * 60)
    
    # Step 1: Check dependencies
    print("\n1. 🔍 Checking dependencies...")
    missing_packages = check_dependencies()
    if missing_packages:
        install_missing_packages(missing_packages)
    else:
        print("✅ All dependencies are satisfied!")
    
    # Step 2: Setup environment
    print("\n2. ⚙️ Setting up environment...")
    setup_environment()
    
    # Step 3: Check data files
    print("\n3. 📊 Checking data files...")
    missing_files = check_data_files()
    if missing_files:
        print(f"⚠️  Missing data files: {', '.join(missing_files)}")
        print("💡 You can collect data using: python main.py collect")
    else:
        print("✅ All data files are available!")
    
    # Step 4: Run dashboard
    print("\n4. 🚀 Starting dashboard...")
    run_dashboard()

if __name__ == "__main__":
    main()