#!/usr/bin/env python3
"""
Alpha Analysis Dashboard Runner

This script runs the enhanced alpha analysis dashboard that provides:
- Operator usage analysis with unique/nominal counting
- Datafield usage analysis grouped by categories  
- Interactive filtering by region, universe, delay, and alpha type
- Beautiful visualizations with charts and graphs
- Auto-launch browser functionality

Usage:
    python run_analysis_dashboard.py [OPTIONS]
    
Examples:
    # Run with default settings (analysis-only mode)
    python run_analysis_dashboard.py
    
    # Initialize database schema first
    python run_analysis_dashboard.py --init-db
    
    # Run on different port
    python run_analysis_dashboard.py --port 8051
    
    # Run without opening browser
    python run_analysis_dashboard.py --no-browser
    
    # Include clustering data if available
    python run_analysis_dashboard.py --data-file clustering_results.json
    
    # Fetch fresh operators/datafields and clear cache for re-analysis
    python run_analysis_dashboard.py --renew
    
    # Clear analysis cache to force re-analysis (without fetching new data)
    python run_analysis_dashboard.py --clear-cache
    
    # Import existing CSV datafields to database for better performance
    python run_analysis_dashboard.py --import-csv
"""

import sys
import os
import argparse
import subprocess

# Setup project path
from utils.bootstrap import setup_project_path
setup_project_path()

# Import only clustering utilities (others will be handled by subprocess)
from utils.clustering_utils import (
    check_or_generate_clustering_data, auto_generate_missing_regions
)



def main():
    """Run the analysis dashboard with auto-generated clustering data."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Alpha Analysis & Clustering Dashboard")
    parser.add_argument("--data-file", type=str, help="Path to clustering data JSON file")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--no-clustering", action="store_true", help="Skip clustering data generation")
    parser.add_argument("--region", type=str, default='USA', help="Region for clustering analysis")
    parser.add_argument("--all-regions", action="store_true", help="Generate clustering data for all regions")
    parser.add_argument("--regions", nargs='*', help="List of specific regions to generate clustering for")
    parser.add_argument("--renew", action="store_true", help="Fetch fresh operators and datafields from API (user-specific)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear analysis cache to force re-analysis of all alphas")
    parser.add_argument("--keep-clustering", action="store_true", help="Keep existing clustering files instead of regenerating")
    parser.add_argument("--import-csv", action="store_true", help="Import existing datafields CSV to database for better performance")
    
    args = parser.parse_args()
    
    # Handle dynamic platform data fetching via subprocess
    dynamic_operators_file = None
    using_dynamic_data = False

    if args.renew:
        # Execute renew_genius.py script
        print("🔄 Executing renew_genius.py to refresh operators and datafields...")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, "renew_genius.py"],
            capture_output=True,
            text=True
        )

        # Display output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}", file=sys.stderr)

        if result.returncode != 0:
            print("❌ Failed to refresh operators/datafields")
            sys.exit(1)
        else:
            # Set flag for successful data refresh
            using_dynamic_data = True
            # Check if operators file exists for dashboard use
            operators_file_path = "data/operators_dynamic.json"
            if os.path.exists(operators_file_path):
                dynamic_operators_file = operators_file_path
    else:
        # Check if operators/datafields exist (for first-time use auto-fetch)
        operators_file_path = "data/operators_dynamic.json"
        if os.path.exists(operators_file_path):
            dynamic_operators_file = operators_file_path
            using_dynamic_data = True
        else:
            print("ℹ️ No operators/datafields found. Run with --renew to fetch them.")
            print("   Dashboard will continue with limited functionality.")

    # Handle explicit cache clearing via subprocess
    if args.clear_cache:
        print("🔄 Executing clear_cache.py...")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, "clear_cache.py"],
            capture_output=True,
            text=True
        )

        # Display output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}", file=sys.stderr)

        if result.returncode != 0:
            print("⚠️ Cache clearing had issues")
    
    # Handle explicit CSV import (legacy support)
    if args.import_csv:
        print("⚠️ CSV import is no longer supported - datafields are now stored directly in database")
        print("   Use --renew to fetch fresh datafields data")
    
    # Handle clustering generation
    if args.no_clustering:
        # No clustering mode
        clustering_file = None
        print("📊 Running dashboard without clustering (--no-clustering specified)")
    else:
        # Use subprocess for clustering regeneration if needed
        if args.all_regions or args.regions:
            cmd = [sys.executable, "refresh_clustering.py"]
            if args.regions:
                cmd.extend(["--regions"] + args.regions)

            print("🔄 Executing refresh_clustering.py...")
            print("=" * 60)
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Display output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"Errors: {result.stderr}", file=sys.stderr)

            if result.returncode != 0:
                print("⚠️ Clustering generation had issues")

            # Set clustering file for dashboard
            region_to_use = args.regions[0] if args.regions else args.region
            clustering_file = check_or_generate_clustering_data(region_to_use)
        else:
            # Check for existing clustering or generate if missing
            force_regenerate = not args.keep_clustering
            if force_regenerate:
                print("Auto-generating clustering data...")
                auto_generate_missing_regions(force_regenerate=True)
            clustering_file = args.data_file or check_or_generate_clustering_data(args.region)
    
    # Import here to avoid import errors if dependencies are missing
    try:
        # Auto-initialize analysis database schema if needed
        print("Checking analysis database schema...")
        from database.schema import get_connection, initialize_analysis_database
        from sqlalchemy import text
        import psycopg2.errors
        
        try:
            db_engine = get_connection()
            with db_engine.connect() as connection:
                # Test if alpha_analysis_cache table exists
                result = connection.execute(text("SELECT 1 FROM alpha_analysis_cache LIMIT 1"))
                print("Analysis database schema is ready.")
                
                # Check if datafields table has data
                datafields_count = connection.execute(text("SELECT COUNT(*) FROM datafields")).scalar()
                if datafields_count == 0:
                    print("⚠️ Datafields table is empty. Please run with --renew to populate datafields.")
                    print("   Dashboard will continue but may have limited datafield analysis functionality.")
                else:
                    print(f"Datafields table contains {datafields_count} entries")
        except psycopg2.errors.UndefinedTable as e:
            if 'alpha_analysis_cache' in str(e):
                print("Analysis cache table missing. Initializing analysis database schema...")
                initialize_analysis_database()
                print("Analysis database schema initialized successfully.")
            else:
                raise
        except Exception as e:
            # If it's a different error, still try to initialize in case tables don't exist
            print(f"Database check failed ({e}). Attempting to initialize analysis schema...")
            try:
                initialize_analysis_database()
                print("Analysis database schema initialized successfully.")
            except Exception as init_error:
                print(f"Warning: Failed to initialize analysis schema: {init_error}")
                print("Dashboard may not work properly without the analysis tables.")
        
        # Temporarily modify sys.argv to pass arguments to the visualization server
        original_argv = sys.argv.copy()
        sys.argv = ['visualization_server.py']
        
        if clustering_file:
            sys.argv.extend(['--data-file', clustering_file])
        
        sys.argv.extend(['--port', str(args.port)])
        
        if args.debug:
            sys.argv.append('--debug')
        if args.no_browser:
            sys.argv.append('--no-browser')
        
        # Pass dynamic data files if available (regardless of whether they were just fetched or already existed)
        if dynamic_operators_file:
            sys.argv.extend(['--operators-file', dynamic_operators_file])
        
        # Use refactored dashboard server
        from analysis.dashboard.server import main as server_main
        server_main()
        
    except ImportError as e:
        print(f"Error: Missing required dependencies: {e}")
        print("\nPlease install required packages:")
        print("pip install dash dash-bootstrap-components plotly pandas numpy sqlalchemy psycopg2")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()