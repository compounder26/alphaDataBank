#!/usr/bin/env python
"""
Utility script to regenerate clustering data for all regions.
This ensures fresh clustering analysis before starting the production server.

Usage:
    python refresh_clustering.py
"""

import sys
import os
import argparse

# Setup project path
from utils.bootstrap import setup_project_path
setup_project_path()

from utils.clustering_utils import generate_all_regions_if_requested, delete_all_clustering_files
from config.database_config import REGIONS
from utils.progress import (
    print_header, print_success, print_error, print_info,
    configure_minimal_logging, suppress_retry_logs,
    create_progress_bar, update_progress_bar, close_progress_bar
)
import logging

# Suppress verbose clustering logs
logging.getLogger('utils.clustering_utils').setLevel(logging.ERROR)

def main():
    """Regenerate clustering data for specified or all regions."""
    # Configure minimal logging for clean output
    configure_minimal_logging()
    suppress_retry_logs()

    parser = argparse.ArgumentParser(description="Regenerate clustering data")
    parser.add_argument("--regions", nargs='*', help="Specific regions to regenerate (default: all)")
    args = parser.parse_args()

    print_header("Regenerating Clustering Data")

    try:
        # Determine which regions to process
        if args.regions:
            regions_to_process = args.regions
            print_info(f"Processing specific regions: {', '.join(regions_to_process)}")
        else:
            regions_to_process = list(REGIONS)
            print_info(f"Processing all regions: {', '.join(regions_to_process)}")

        # Delete all existing clustering JSON files first
        print_info("Cleaning up existing clustering files...")
        # Suppress verbose delete output
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        deleted_count = delete_all_clustering_files()
        sys.stdout = old_stdout
        if deleted_count > 0:
            print_success(f"  Deleted {deleted_count} old clustering files")

        # Generate clustering for regions with progress bar
        success_count = 0
        failed_regions = []

        # Create progress bar for regions
        region_bar = create_progress_bar(
            len(regions_to_process),
            "Generating clustering",
            unit="regions"
        )

        for region in regions_to_process:
            try:
                # Suppress verbose output from clustering function
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                capture_output = io.StringIO()
                sys.stdout = capture_output
                sys.stderr = capture_output

                result = generate_all_regions_if_requested([region])

                sys.stdout = old_stdout
                sys.stderr = old_stderr

                # Check captured output for specific error messages
                output = capture_output.getvalue()

                if result and len(result) > 0:
                    success_count += 1
                else:
                    # Parse output to understand why it failed
                    reason = ""
                    if "No alphas found" in output:
                        reason = "no alphas in database"
                    elif "Found 0 alphas" in output:
                        reason = "no alphas in database"
                    elif "Found 1 alphas" in output:
                        reason = "only 1 alpha (need at least 2 for clustering)"
                    elif "Found 2 alphas" in output:
                        reason = "only 2 alphas (insufficient for meaningful clustering)"
                    elif "Not enough clusters" in output or "insufficient_data" in output:
                        # Try to extract the number of alphas from output
                        import re
                        match = re.search(r"Found (\d+) alphas", output)
                        if match:
                            num_alphas = match.group(1)
                            reason = f"only {num_alphas} alphas"
                        else:
                            reason = "insufficient data for clustering"
                    elif "No PnL data" in output:
                        reason = "no PnL data available"
                    else:
                        reason = "unknown reason"

                    print_info(f"\n  {region}: Skipped ({reason})")
                    failed_regions.append(region)
            except Exception as e:
                sys.stdout = old_stdout  # Restore stdout in case of error
                sys.stderr = old_stderr
                print_error(f"\n  {region}: Error - {str(e)}")
                failed_regions.append(region)

            update_progress_bar(region_bar)

        close_progress_bar(region_bar)

        print_success(f"Clustering Generation Complete!")
        print_info(f"  Successful: {success_count}/{len(regions_to_process)} regions")

        if failed_regions:
            print_info(f"  Failed/Skipped: {', '.join(failed_regions)}")

        print_info("\nReady to start the production server:")
        print_info("  Windows: waitress-serve --host=127.0.0.1 --port=8050 wsgi:server")
        print_info("  Linux/Mac: gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server")

        # Return success code based on results
        if failed_regions:
            sys.exit(1)  # Some failures
        else:
            sys.exit(0)  # All successful

    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()