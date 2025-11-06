"""
Script to run the Streamlit dashboard with enhanced features
"""

import subprocess
import sys
import os
import webbrowser
import threading
import time
from datetime import datetime

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check required files
    required_files = [
        "app.py",
        "config.py", 
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Check data files (optional but recommended)
    data_files = [
        "industry_insights_with_financial_sentiment.csv"
    ]
    
    missing_data = []
    for file in data_files:
        if not os.path.exists(file):
            missing_data.append(file)
    
    if missing_data:
        print(f"⚠️  Missing data files: {', '.join(missing_data)}")
        print("💡 You can collect data using: python main.py collect")
    else:
        print("✅ All data files available")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Checking dependencies...")
    
    try:
        # Check if streamlit is installed
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True

def check_environment():
    """Check if the environment is properly set up for cloud deployment"""
    print("🔍 Checking environment...")
    
    # Check required files (but don't fail if missing in cloud)
    required_files = [
        "app.py",
        "config.py", 
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        # Don't return False immediately for cloud - some files might be created dynamically
        if "app.py" in missing_files:
            return False  # app.py is critical
    
    # For cloud deployment, data files are optional (will be created)
    print("✅ Environment check passed (cloud compatible)")
    return True

def open_browser_delayed():
    """Open browser after a delay"""
    time.sleep(5)  # Wait for Streamlit to start
    webbrowser.open("http://localhost:8501")
    print("🌐 Browser opened automatically")

def run_streamlit_dashboard():
    """Run the Streamlit dashboard with enhanced features"""
    print("="*60)
    print("🚀 STRATEGIC INTELLIGENCE DASHBOARD")
    print("="*60)
    
    # Environment check
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        return
    
    # Dependency check
    if not install_dependencies():
        print("❌ Dependency installation failed.")
        return
    
    print("\n🎯 Starting Enhanced Dashboard Features:")
    print("   • Interactive Plotly charts")
    print("   • Real-time data filtering") 
    print("   • Sector-based analysis")
    print("   • Competitor tracking")
    print("   • Alert system integration")
    print("   • Forecast visualization")
    print("   • Export capabilities")
    
    print(f"\n📅 Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌐 Dashboard will open at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("\n" + "="*60)
    
    try:
        # Start browser in background thread
        browser_thread = threading.Thread(target=open_browser_delayed)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Streamlit with enhanced configuration
        env = os.environ.copy()
        env['STREAMLIT_SERVER_HEADLESS'] = 'false'
        env['STREAMLIT_SERVER_PORT'] = '8501'
        env['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--theme.primaryColor=#1f77b4",
            "--theme.backgroundColor=#ffffff",
            "--theme.secondaryBackgroundColor=#f0f2f6",
            "--theme.textColor=#262730"
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check if port 8501 is available")
        print("2. Verify all required files are present")
        print("3. Ensure Python dependencies are installed")
        print("4. Check your internet connection for external APIs")

def main():
    """Main function to run the dashboard"""
    try:
        run_streamlit_dashboard()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔧 For manual troubleshooting, run:")
        print("   streamlit run app.py")
        print("\n📚 Or check the documentation for setup instructions")

if __name__ == "__main__":
    main()