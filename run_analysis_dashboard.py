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
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_or_generate_clustering_data(region='USA'):
    """Check if clustering data exists for any region, generate if not."""
    from config.database_config import REGIONS
    
    # First check if any clustering data exists for the requested region
    region_files = [
        f"clustering_results_{region}.json",
        f"analysis/clustering/alpha_clustering_{region}*.json"
    ]
    
    # Check for existing files
    import glob
    for pattern in region_files:
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getctime)
            print(f"Using existing clustering data: {latest_file}")
            return latest_file
    
    # No clustering data found, generate for requested region
    print(f"No clustering data found for {region}. Generating clustering data...")
    print("Available regions:", ", ".join(REGIONS))
    
    if region not in REGIONS:
        print(f"Warning: {region} not in available regions. Using USA as default.")
        region = 'USA'
    
    try:
        from analysis.clustering.clustering_analysis import generate_clustering_data, save_clustering_results
        
        print(f"This may take a few minutes for region {region}...")
        results = generate_clustering_data(region)
        if results:
            output_path = save_clustering_results(results)
            print(f"Generated clustering data: {output_path}")
            return output_path
        else:
            print("Warning: Could not generate clustering data. Dashboard will run without clustering.")
            return None
    except Exception as e:
        print(f"Warning: Error generating clustering data: {e}")
        print("Dashboard will run without clustering visualization.")
        return None

def generate_all_regions_if_requested(regions_list):
    """Generate clustering data for multiple regions."""
    from analysis.clustering.clustering_analysis import generate_clustering_data, save_clustering_results
    
    generated_files = []
    for region in regions_list:
        print(f"\nGenerating clustering data for {region}...")
        try:
            results = generate_clustering_data(region)
            if results:
                output_path = save_clustering_results(results)
                generated_files.append(output_path)
                print(f"Generated: {output_path}")
            else:
                print(f"Failed to generate data for {region}")
        except Exception as e:
            print(f"Error generating {region}: {e}")
    
    return generated_files

def auto_generate_missing_regions():
    """Auto-generate clustering data for all regions that don't have existing files."""
    from config.database_config import REGIONS
    import glob
    
    print("Checking for existing clustering files across all regions...")
    
    missing_regions = []
    existing_regions = []
    
    for region in REGIONS:
        # Check for existing clustering files for this region
        pattern = f"analysis/clustering/alpha_clustering_{region}_*.json"
        files = glob.glob(pattern)
        
        if files:
            latest_file = max(files, key=os.path.getctime)
            existing_regions.append((region, latest_file))
            print(f"[OK] {region}: {latest_file}")
        else:
            missing_regions.append(region)
            print(f"[MISSING] {region}: No clustering data found")
    
    if missing_regions:
        print(f"\nGenerating clustering data for {len(missing_regions)} missing regions...")
        generated_files = generate_all_regions_if_requested(missing_regions)
        print(f"\nGenerated {len(generated_files)} new clustering files")
    else:
        print("\nAll regions have existing clustering data!")
    
    return existing_regions + [(region, None) for region in missing_regions]

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
    
    args = parser.parse_args()
    
    # Handle multi-region generation first
    if args.all_regions and not args.no_clustering:
        from config.database_config import REGIONS
        print("Generating clustering data for all regions...")
        generate_all_regions_if_requested(REGIONS)
        clustering_file = check_or_generate_clustering_data(args.region)  # Use default region for dashboard
    elif args.regions and not args.no_clustering:
        print(f"Generating clustering data for specified regions: {args.regions}")
        generate_all_regions_if_requested(args.regions)
        clustering_file = check_or_generate_clustering_data(args.regions[0])  # Use first region for dashboard
    elif not args.no_clustering:
        # Auto-generate missing regions by default, then use specified region for dashboard
        print("Auto-generating missing region clustering data...")
        auto_generate_missing_regions()
        clustering_file = args.data_file or check_or_generate_clustering_data(args.region)
    else:
        # No clustering mode
        clustering_file = None
    
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
        
        from analysis.clustering.visualization_server import main as server_main
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