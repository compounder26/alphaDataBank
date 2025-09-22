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
    # Run with default settings
    python run_analysis_dashboard.py

    # Run on different port
    python run_analysis_dashboard.py --port 8051

    # Fetch fresh operators/datafields and clear cache for re-analysis
    python run_analysis_dashboard.py --renew

    # Clear analysis cache to force re-analysis (without fetching new data)
    python run_analysis_dashboard.py --clear-cache
"""

import sys
import os
import argparse
import subprocess

# Setup project path
from utils.bootstrap import setup_project_path
setup_project_path()

# Clustering utilities removed - no longer needed



def main():
    """Run the analysis dashboard."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Alpha Analysis Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--renew", action="store_true", help="Fetch fresh operators and datafields from API (user-specific)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear analysis cache to force re-analysis of all alphas")
    
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
    
    # Clustering functionality has been removed

    # Import here to avoid import errors if dependencies are missing
    try:
        # Temporarily modify sys.argv to pass arguments to the visualization server
        original_argv = sys.argv.copy()
        sys.argv = ['visualization_server.py']
        
        
        sys.argv.extend(['--port', str(args.port)])
        
        if args.debug:
            sys.argv.append('--debug')
        
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