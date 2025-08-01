#!/usr/bin/env python3
"""
Script to run SenseBehavior Flask app without any caching
"""

import os
import sys

# Set environment variables to disable caching
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PIP_NO_CACHE_DIR'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

# Disable Python bytecode generation
sys.dont_write_bytecode = True

if __name__ == '__main__':
    # Import and run the app
    from app import app
    
    port = int(os.environ.get('PORT', 7860))
    print(f"🚀 Starting SenseBehavior Flask Application (No Cache Mode)...")
    print(f"📡 Server will run on port {port}")
    print(f"🌐 Access at: http://localhost:{port}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting Flask server: {e}")
        sys.exit(1) 