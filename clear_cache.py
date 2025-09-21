#!/usr/bin/env python
"""
Utility script to clear the analysis cache.
This forces all alphas to be re-analyzed on next dashboard run.

Usage:
    python clear_cache.py [--quiet]

Options:
    --quiet    Less verbose output (for use by other scripts)
"""

import sys
import os
import argparse

# Setup project path
from utils.bootstrap import setup_project_path
setup_project_path()

try:
    from database.schema import get_connection
    from sqlalchemy import text

    def clear_analysis_cache(quiet=False):
        """Clear the alpha_analysis_cache table."""
        global args  # Access args from main if needed
        try:
            db_engine = get_connection()
            with db_engine.connect() as connection:
                with connection.begin():
                    # Count current entries
                    count_result = connection.execute(text("SELECT COUNT(*) FROM alpha_analysis_cache"))
                    current_count = count_result.scalar()

                    if not quiet:
                        print(f"Found {current_count} cached analysis entries")

                    if current_count > 0:
                        # Clear the cache
                        connection.execute(text("DELETE FROM alpha_analysis_cache"))
                        if not quiet:
                            print(f"✅ Cleared {current_count} entries from alpha_analysis_cache")
                            print("Next dashboard run will re-parse all alphas with corrected logic")
                        else:
                            print(f"✅ Cleared {current_count} cache entries")
                    else:
                        if not quiet:
                            print("Cache is already empty")

        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False

        return True

except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("This script requires database dependencies to run")

    def clear_analysis_cache(quiet=False):
        """Fallback function when dependencies are missing."""
        print(f"❌ Error: Missing dependencies - {e}")
        return False

def main():
    """Clear the analysis cache."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Clear analysis cache")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if not args.quiet:
        print("=" * 60)
        print("🔄 Clearing Analysis Cache")
        print("=" * 60)

    try:
        if clear_analysis_cache(quiet=args.quiet):
            if not args.quiet:
                print("✅ Analysis cache cleared successfully!")
                print("   All alphas will be re-analyzed on next dashboard run.")
                print("\n✨ Ready! You can now start the production server:")
                print("   Windows: waitress-serve --host=127.0.0.1 --port=8050 wsgi:server")
                print("   Linux/Mac: gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server")
            sys.exit(0)  # Success
        else:
            if not args.quiet:
                print("⚠️ No cache to clear or cache clearing failed")
            sys.exit(1)  # Failure

    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()