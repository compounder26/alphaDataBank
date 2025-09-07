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

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fetch_dynamic_platform_data(force_refresh: bool = False):
    """
    Fetch operators and datafields from API if requested.
    
    Args:
        force_refresh: If True, fetch fresh data. If False, use cached data if available.
        
    Returns:
        Tuple of (operators_file_path, datafields_saved_to_db, using_dynamic_data)
    """
    try:
        from api.platform_data_fetcher import PlatformDataFetcher
        
        print("🔄 Fetching platform data (operators and datafields)...")
        fetcher = PlatformDataFetcher()
        
        # Fetch and cache data
        operators_cache, datafields_cache = fetcher.fetch_and_cache_all(force_refresh=force_refresh)
        
        if operators_cache and datafields_cache:
            print("✅ Successfully fetched and cached platform data!")
            print(f"   Operators cache: {operators_cache}")
            print(f"   Datafields cache: {datafields_cache}")
            return operators_cache, datafields_cache, True
        else:
            print("⚠️ Failed to fetch platform data, will use static files")
            return None, None, False
            
    except ImportError as e:
        print(f"⚠️ Could not import platform data fetcher: {e}")
        print("   Will use static files instead")
        return None, None, False
    except Exception as e:
        print(f"⚠️ Error fetching platform data: {e}")
        print("   Will use static files instead")
        return None, None, False

def check_and_auto_fetch_platform_data():
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

def delete_all_clustering_files():
    """Delete all existing clustering JSON files to force fresh recalculation."""
    import glob
    from datetime import datetime
    
    print(f"\n🗑️ Cleaning up old clustering files at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Patterns for clustering files
    patterns = [
        "analysis/clustering/alpha_clustering_*.json",
        "clustering_results_*.json",
        "alpha_clustering_*.json"
    ]
    
    deleted_count = 0
    deleted_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                # Get file age before deletion
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getctime(file))
                os.remove(file)
                deleted_files.append((file, file_age))
                deleted_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not delete {file}: {e}")
    
    if deleted_count > 0:
        print(f"✅ Deleted {deleted_count} old clustering files:")
        for file, age in deleted_files[:5]:  # Show first 5 files
            age_str = f"{age.days}d {age.seconds//3600}h" if age.days > 0 else f"{age.seconds//3600}h {(age.seconds%3600)//60}m"
            print(f"   - {os.path.basename(file)} (was {age_str} old)")
        if deleted_count > 5:
            print(f"   ... and {deleted_count - 5} more files")
    else:
        print("   No clustering files found to delete")
    
    return deleted_count

def auto_generate_missing_regions(force_regenerate=True):
    """Auto-generate clustering data for all regions that don't have existing files."""
    from config.database_config import REGIONS
    import glob
    
    if force_regenerate:
        # Delete all existing clustering files first
        delete_all_clustering_files()
        print("\n🔄 Force regenerating clustering data for ALL regions...")
        
        # Generate for all regions since we deleted everything
        generated_files = generate_all_regions_if_requested(REGIONS)
        print(f"\n✅ Generated {len(generated_files)} new clustering files")
        return [(region, None) for region in REGIONS]
    
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
    parser.add_argument("--renew", action="store_true", help="Fetch fresh operators and datafields from API (user-specific)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear analysis cache to force re-analysis of all alphas")
    parser.add_argument("--keep-clustering", action="store_true", help="Keep existing clustering files instead of regenerating")
    parser.add_argument("--import-csv", action="store_true", help="Import existing datafields CSV to database for better performance")
    
    args = parser.parse_args()
    
    # Handle dynamic platform data fetching
    dynamic_operators_file = None
    using_dynamic_data = False
    
    if args.renew:
        # Force refresh of operators and datafields
        dynamic_operators_file, datafields_saved, using_dynamic_data = fetch_dynamic_platform_data(force_refresh=True)
        
        # Clear analysis cache when renewing operators/datafields since exclusion logic may change
        if using_dynamic_data:
            print("🔄 Clearing analysis cache due to updated operators/datafields...")
            try:
                from scripts.clear_analysis_cache import clear_analysis_cache
                if clear_analysis_cache():
                    print("✅ Analysis cache cleared - all alphas will be re-analyzed with new operators/datafields")
                else:
                    print("⚠️ Failed to clear analysis cache - some exclusions may be outdated")
            except Exception as e:
                print(f"⚠️ Could not clear analysis cache: {e}")
                print("   Some exclusions may be outdated until manual cache clear")
    else:
        # Auto-fetch operators/datafields if they don't exist (first-time use)
        dynamic_operators_file, datafields_saved, using_dynamic_data = check_and_auto_fetch_platform_data()
        
        # Clear analysis cache if we fetched new data
        if using_dynamic_data and dynamic_operators_file and datafields_saved:
            # Check if operators file is newly created (less than 5 minutes old)
            import os
            from datetime import datetime, timedelta
            
            operators_age = datetime.now() - datetime.fromtimestamp(os.path.getctime(dynamic_operators_file))
            
            if operators_age < timedelta(minutes=5):
                print("🔄 Clearing analysis cache for newly fetched platform data...")
                try:
                    from scripts.clear_analysis_cache import clear_analysis_cache
                    if clear_analysis_cache():
                        print("✅ Analysis cache cleared - all alphas will be re-analyzed with fresh platform data")
                    else:
                        print("⚠️ Failed to clear analysis cache - some exclusions may be outdated")
                except Exception as e:
                    print(f"⚠️ Could not clear analysis cache: {e}")
                    print("   Some exclusions may be outdated until manual cache clear")
    
    # Handle explicit cache clearing
    if args.clear_cache:
        print("🔄 Clearing analysis cache as requested...")
        try:
            from scripts.clear_analysis_cache import clear_analysis_cache
            if clear_analysis_cache():
                print("✅ Analysis cache cleared - all alphas will be re-analyzed")
            else:
                print("⚠️ Failed to clear analysis cache")
        except Exception as e:
            print(f"⚠️ Could not clear analysis cache: {e}")
    
    # Handle explicit CSV import (legacy support)
    if args.import_csv:
        print("⚠️ CSV import is no longer supported - datafields are now stored directly in database")
        print("   Use --renew to fetch fresh datafields data")
    
    # Handle multi-region generation first
    if args.no_clustering:
        # No clustering mode
        clustering_file = None
        print("📊 Running dashboard without clustering (--no-clustering specified)")
    else:
        # Determine if we should force regeneration
        force_regenerate = not args.keep_clustering
        
        if force_regenerate:
            print("\n🔄 FORCE REGENERATING ALL CLUSTERING DATA")
            print("   (Use --keep-clustering to reuse existing files)")
            print("=" * 60)
        
        if args.all_regions:
            from config.database_config import REGIONS
            if force_regenerate:
                # Delete all existing files and regenerate everything
                delete_all_clustering_files()
                print("Generating fresh clustering data for all regions...")
                generate_all_regions_if_requested(REGIONS)
            else:
                print("Generating clustering data for all regions (keeping existing)...")
                auto_generate_missing_regions(force_regenerate=False)
            clustering_file = check_or_generate_clustering_data(args.region)  # Use default region for dashboard
        elif args.regions:
            if force_regenerate:
                # Delete files for specified regions and regenerate
                import glob
                for region in args.regions:
                    pattern = f"analysis/clustering/alpha_clustering_{region}_*.json"
                    for file in glob.glob(pattern):
                        try:
                            os.remove(file)
                            print(f"Deleted: {file}")
                        except:
                            pass
            print(f"Generating clustering data for specified regions: {args.regions}")
            generate_all_regions_if_requested(args.regions)
            clustering_file = check_or_generate_clustering_data(args.regions[0])  # Use first region for dashboard
        else:
            # Default behavior: regenerate all regions unless --keep-clustering is specified
            print("Auto-generating clustering data...")
            auto_generate_missing_regions(force_regenerate=force_regenerate)
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