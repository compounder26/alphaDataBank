"""
Platform data utilities for WorldQuant Brain API integration.

This module provides utilities for fetching and managing operators and datafields
from the WorldQuant Brain API.
"""
import os
import logging
from typing import Tuple, Optional

# Suppress verbose logging from platform data fetcher
logging.getLogger('api.platform_data_fetcher').setLevel(logging.ERROR)


def fetch_dynamic_platform_data(force_refresh: bool = False) -> Tuple[Optional[str], Optional[bool], bool]:
    """
    Fetch operators and datafields from API if requested.

    Args:
        force_refresh: If True, fetch fresh data. If False, use cached data if available.

    Returns:
        Tuple of (operators_file_path, datafields_saved_to_db, using_dynamic_data)
    """
    try:
        from api.platform_data_fetcher import PlatformDataFetcher

        print("[REFRESH] Fetching platform data (operators and datafields)...")
        fetcher = PlatformDataFetcher()

        # Fetch and cache data
        operators_cache, datafields_cache = fetcher.fetch_and_cache_all(force_refresh=force_refresh)

        if operators_cache and datafields_cache:
            print("[SUCCESS] Successfully fetched and cached platform data!")
            print(f"   Operators cache: {operators_cache}")
            print(f"   Datafields cache: {datafields_cache}")
            return operators_cache, datafields_cache, True
        else:
            print("[WARNING] Failed to fetch platform data, will use static files")
            return None, None, False

    except ImportError as e:
        print(f"[WARNING] Could not import platform data fetcher: {e}")
        print("   Will use static files instead")
        return None, None, False
    except Exception as e:
        print(f"[WARNING] Error fetching platform data: {e}")
        print("   Will use static files instead")
        return None, None, False


def check_and_auto_fetch_platform_data() -> Tuple[Optional[str], Optional[bool], bool]:
    """
    Check if operators/datafields cache exist. If not, auto-fetch them.

    Returns:
        Tuple of (operators_file_path, datafields_saved_to_db, using_dynamic_data)
    """
    try:
        from api.platform_data_fetcher import PlatformDataFetcher

        fetcher = PlatformDataFetcher()

        # Check if operators cache exists (datafields now in database)
        operators_exist = os.path.exists(fetcher.operators_cache_file)

        if operators_exist:
            print("Found existing operators cache")
            return fetcher.operators_cache_file, True, True

        # Auto-fetch missing data
        missing_items = []
        if not operators_exist:
            missing_items.append("operators")
        # Always check/refresh datafields in database
        missing_items.append("datafields")

        print(f"Missing cache files: {', '.join(missing_items)}")
        print("Auto-fetching missing platform data...")

        return fetch_dynamic_platform_data(force_refresh=False)

    except ImportError as e:
        print(f"Warning: Could not import platform data fetcher: {e}")
        return None, None, False
    except Exception as e:
        print(f"Warning: Error checking platform data: {e}")
        return None, None, False