#!/usr/bin/env python
"""
Utility script to refresh operators and datafields from the WorldQuant Brain API.
This should be run before starting the production server when you need fresh data.

Usage:
    python renew_genius.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.platform_data_utils import fetch_dynamic_platform_data
from clear_cache import clear_analysis_cache
from utils.progress import (
    print_header, print_success, print_error, print_info,
    configure_minimal_logging, suppress_retry_logs
)

def main():
    """Fetch fresh operators/datafields and clear cache."""
    # Configure minimal logging for clean output
    configure_minimal_logging()
    suppress_retry_logs()

    print_header("Refreshing Operators and Datafields")

    try:
        # Force refresh of operators and datafields
        print_info("Fetching operators and datafields from API...")
        dynamic_operators_file, datafields_saved, using_dynamic_data = fetch_dynamic_platform_data(force_refresh=True)

        if using_dynamic_data:
            print_success(f"Fetched dynamic operators from: {dynamic_operators_file}")
            print_success(f"Datafields saved to database: {datafields_saved}")

            # Clear analysis cache when renewing operators/datafields
            print_info("Clearing analysis cache...")
            if clear_analysis_cache():
                print_success("Analysis cache cleared - all alphas will be re-analyzed")
                print_info("\nReady! You can now start the production server:")
                print_info("  Windows: waitress-serve --host=127.0.0.1 --port=8050 wsgi:server")
                print_info("  Linux/Mac: gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server")
            else:
                print_error("Could not clear analysis cache, but operators/datafields were updated")
        else:
            print_error("Could not fetch dynamic data, will use existing cached data")

    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()