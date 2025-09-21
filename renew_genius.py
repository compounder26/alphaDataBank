#!/usr/bin/env python
"""
Utility script to refresh operators and datafields from the WorldQuant Brain API.
This should be run before starting the production server when you need fresh data.

Usage:
    python renew_operators.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.platform_data_utils import fetch_dynamic_platform_data
from clear_cache import clear_analysis_cache

def main():
    """Fetch fresh operators/datafields and clear cache."""
    print("=" * 60)
    print("🔄 Refreshing Operators and Datafields")
    print("=" * 60)
    
    try:
        # Force refresh of operators and datafields
        dynamic_operators_file, datafields_saved, using_dynamic_data = fetch_dynamic_platform_data(force_refresh=True)
        
        if using_dynamic_data:
            print(f"✅ Successfully fetched dynamic operators from: {dynamic_operators_file}")
            print(f"✅ Datafields saved to database: {datafields_saved}")
            
            # Clear analysis cache when renewing operators/datafields
            print("\n🔄 Clearing analysis cache...")
            if clear_analysis_cache():
                print("✅ Analysis cache cleared - all alphas will be re-analyzed")
                print("\n✨ Ready! You can now start the production server:")
                print("   Windows: waitress-serve --host=127.0.0.1 --port=8050 wsgi:server")
                print("   Linux/Mac: gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server")
            else:
                print("⚠️ Could not clear analysis cache, but operators/datafields were updated")
        else:
            print("⚠️ Could not fetch dynamic data, will use existing cached data")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()