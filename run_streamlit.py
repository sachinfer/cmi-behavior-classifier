#!/usr/bin/env python3
"""
Simple runner script for Behavior Analytics Dashboard
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function to run the Streamlit application"""
    print("📊 Behavior Analytics Dashboard")
    print("=" * 50)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🎯 Starting Streamlit application...")
    print("The application will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run the Streamlit application
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_streamlit.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
