#!/usr/bin/env python3
"""
Simple runner script for SenseBehavior application
Run this script to start the Flask application
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'torch', 'scikit-learn', 
        'matplotlib', 'seaborn'
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
    """Main function to run the application"""
    print("🚀 SenseBehavior - Human Behavior Classification Dashboard")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists("lstm_model.pth"):
        print("\n⚠️  Warning: lstm_model.pth not found")
        print("The application will run in demo mode without PyTorch")
    
    print("\n🎯 Starting Flask application...")
    print("The application will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Run the Flask application
    try:
        from app import app
        app.run(host='0.0.0.0', port=7860, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 