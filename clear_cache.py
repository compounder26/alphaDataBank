#!/usr/bin/env python
"""
Utility script to clear the analysis cache.
This forces all alphas to be re-analyzed on next dashboard run.

Usage:
    python clear_cache.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.clear_analysis_cache import clear_analysis_cache

def main():
    """Clear the analysis cache."""
    print("=" * 60)
    print("🔄 Clearing Analysis Cache")
    print("=" * 60)
    
    try:
        if clear_analysis_cache():
            print("✅ Analysis cache cleared successfully!")
            print("   All alphas will be re-analyzed on next dashboard run.")
            print("\n✨ Ready! You can now start the production server:")
            print("   Windows: waitress-serve --port=8050 wsgi:server")
            print("   Linux/Mac: gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server")
        else:
            print("⚠️ No cache to clear or cache clearing failed")
            
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()