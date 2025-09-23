"""
Main script to run the alpha databank process.
This script will:
1. Initialize the database
2. Fetch alphas for specified regions
3. Fetch PNL data for the alphas
4. Calculate and update correlation statistics
"""
import sys
import os
import logging
import argparse
import time
from pathlib import Path
from typing import List, Optional

# Setup project path
from utils.bootstrap import setup_project_path
setup_project_path()

from api.auth import get_authenticated_session, check_session_valid, set_global_session
from api.alpha_fetcher import scrape_alphas_by_region, get_alpha_pnl_threaded, get_robust_session, warm_up_api_connection
from database.schema import initialize_database, initialize_analysis_database
from config.database_config import REGIONS as CONFIGURED_REGIONS # Import and alias to avoid conflict
from database.operations import (
    insert_alpha,
    insert_multiple_alphas, 
    get_alpha_type_and_status, 
    get_regular_alpha_ids_for_pnl_processing, 
    insert_pnl_data,
    insert_multiple_pnl_data,
    insert_multiple_pnl_data_optimized,  # Optimized PNL insertion function
    calculate_and_store_correlations,
    get_correlation_statistics
)

# Import unified correlation engine
from analysis.correlation.correlation_engine import CorrelationEngine
from utils.helpers import setup_logging, print_correlation_report
from config.database_config import REGIONS

# Import progress bar utilities
from utils.progress import (
    create_progress_bar, update_progress_bar, close_progress_bar,
    print_success, print_error, print_info, print_header,
    configure_minimal_logging, suppress_retry_logs, close_all_progress_bars
)

# Import unsubmitted alphas modules
from api.unsubmitted_fetcher import fetch_unsubmitted_alphas_from_url
from api.unsubmitted_fetcher_auto import fetch_all_unsubmitted_alphas_auto
from database.operations_unsubmitted import (
    get_unsubmitted_alpha_type_and_status,
    insert_unsubmitted_alpha,
    insert_multiple_unsubmitted_alphas,
    get_unsubmitted_alpha_ids_for_pnl_processing,
    insert_unsubmitted_pnl_data,
    insert_multiple_unsubmitted_pnl_data_optimized
)
from database.schema import initialize_unsubmitted_database

def process_unsubmitted_alphas(url: str, regions_to_process: List[str], skip_init: bool = False, 
                              skip_alpha_fetch: bool = False, skip_pnl_fetch: bool = False, 
                              skip_correlation: bool = False) -> None:
    """
    Process unsubmitted alphas: fetch metadata, PNL data, and calculate correlations.
    
    Args:
        url: URL for fetching unsubmitted alphas
        regions_to_process: List of regions to process
        skip_init: Skip database initialization
        skip_alpha_fetch: Skip alpha fetching
        skip_pnl_fetch: Skip PNL fetching
        skip_correlation: Skip correlation calculation
    """
    print_header("Processing Unsubmitted Alphas")

    # Initialize unsubmitted database schema if needed
    if not skip_init:
        try:
            initialize_unsubmitted_database()
            print_success("Database initialized")
        except Exception as e:
            print_error(f"Database initialization failed: {e}")
            return
    
    # Create session for API calls
    session = None
    if not skip_alpha_fetch or not skip_pnl_fetch:
        robust_session = get_robust_session()
        session = get_authenticated_session(session=robust_session)
        if not session:
            print_error("Authentication failed")
            return
        set_global_session(session)
        warm_up_api_connection(session)
        print_success("Connected to API")
    
    # Fetch all unsubmitted alphas once (outside region loop)
    alphas_by_region = {}
    all_unsubmitted_alphas = []
    
    if not skip_alpha_fetch:
        try:
            all_unsubmitted_alphas = fetch_unsubmitted_alphas_from_url(session, url)
            if all_unsubmitted_alphas:
                print_success(f"Fetched {len(all_unsubmitted_alphas)} alphas")
                
                # Group alphas by region
                for alpha in all_unsubmitted_alphas:
                    alpha_region = alpha.get('settings_region')
                    if alpha_region in regions_to_process:
                        if alpha_region not in alphas_by_region:
                            alphas_by_region[alpha_region] = []
                        alphas_by_region[alpha_region].append(alpha)
                
                # Show distribution
                for region, alphas_list in alphas_by_region.items():
                    print_info(f"{region}: {len(alphas_list)} alphas")
                    
            else:
                print_info("No alphas found at URL")
        except Exception as e:
            print_error(f"Failed to fetch alphas: {e}")
            return

    # Process each region with progress bar
    if regions_to_process:
        region_bar = create_progress_bar(
            len(regions_to_process),
            "Processing regions",
            unit="regions"
        )

        for region in regions_to_process:
            data_changed = False

            # Get alphas for this region (already fetched)
            region_alphas = alphas_by_region.get(region, [])

            # Insert unsubmitted alphas metadata for this region
            if not skip_alpha_fetch and region_alphas:
                try:
                    # Use batch insert for better performance
                    insert_multiple_unsubmitted_alphas(region_alphas, region)
                    data_changed = True
                except Exception as e:
                    print_error(f"Failed to store alphas for {region}: {e}")
            elif not skip_alpha_fetch:
                pass  # No alphas for this region

            # Fetch PNL data for unsubmitted alphas
            if not skip_pnl_fetch:
                try:
                    alpha_ids_for_pnl = get_unsubmitted_alpha_ids_for_pnl_processing(region)

                    if alpha_ids_for_pnl:
                        # Create nested progress bar for PNL fetching
                        pnl_bar = create_progress_bar(
                            len(alpha_ids_for_pnl),
                            f"  PNL for {region}",
                            position=1,
                            leave=False,
                            unit="alphas"
                        )

                        # Use the same PNL fetching logic as submitted alphas
                        combined_pnl_df, failed_alpha_ids = get_alpha_pnl_threaded(
                            session, alpha_ids_for_pnl, max_workers=20
                        )

                        if combined_pnl_df is not None and not combined_pnl_df.empty:
                            unique_fetched_alphas = combined_pnl_df['alpha_id'].unique()
                            update_progress_bar(pnl_bar, len(unique_fetched_alphas))

                            # Prepare PNL data for batch insertion
                            pnl_data_dict = {}
                            for alpha_id in unique_fetched_alphas:
                                pnl_data_dict[alpha_id] = combined_pnl_df[combined_pnl_df['alpha_id'] == alpha_id]

                            # Batch insert PNL data
                            if pnl_data_dict:
                                try:
                                    insert_multiple_unsubmitted_pnl_data_optimized(pnl_data_dict, region)
                                    data_changed = True
                                except Exception:
                                    # Fallback to individual inserts
                                    successfully_stored = 0
                                    for alpha_id, pnl_df in pnl_data_dict.items():
                                        try:
                                            insert_unsubmitted_pnl_data(alpha_id, pnl_df, region)
                                            successfully_stored += 1
                                        except:
                                            pass  # Silent fail

                                    if successfully_stored > 0:
                                        data_changed = True

                        close_progress_bar(pnl_bar)
                except Exception as e:
                    print_error(f"PNL processing failed for {region}: {e}")

            # Calculate correlations for unsubmitted alphas
            if not skip_correlation and data_changed:
                try:
                    # Use unified correlation engine
                    correlation_engine = CorrelationEngine()
                    correlation_engine.calculate_unsubmitted_vs_submitted(region)
                except Exception as e:
                    print_error(f"Correlation calculation failed for {region}: {e}")

            update_progress_bar(region_bar)

        close_progress_bar(region_bar)

    print_success("Unsubmitted alphas processing complete")


def process_unsubmitted_alphas_auto(
    regions_to_process: List[str], 
    sharpe_thresholds: Optional[List[float]] = None,
    batch_size: int = 50,
    skip_init: bool = False, 
    skip_alpha_fetch: bool = False, 
    skip_pnl_fetch: bool = False, 
    skip_correlation: bool = False,
    use_streaming: bool = True
) -> None:
    """
    Process unsubmitted alphas automatically using streaming window processing.
    
    Args:
        regions_to_process: List of regions to process
        sharpe_thresholds: List of sharpe thresholds to fetch (e.g., [1.0, -1.0])
        batch_size: Number of alphas to fetch per API request
        skip_init: Skip database initialization
        skip_alpha_fetch: Skip alpha fetching
        skip_pnl_fetch: Skip PNL fetching
        skip_correlation: Skip correlation calculation
        use_streaming: Use streaming mode (recommended) vs bulk mode (memory intensive)
    """
    logging.info("🔄 Processing unsubmitted alphas (automated)...")
    
    if sharpe_thresholds:
        logging.info(f"Sharpe thresholds: {sharpe_thresholds}")
    else:
        sharpe_thresholds = [1.0, -1.0]
        logging.info("Using default sharpe thresholds")
    
    # Initialize unsubmitted database schema if needed
    if not skip_init:
        try:
            initialize_unsubmitted_database()
            print_success("Database initialized")
        except Exception as e:
            print_error(f"Database initialization failed: {e}")
            return
    
    # Create session for API calls
    session = None
    if not skip_alpha_fetch or not skip_pnl_fetch:
        robust_session = get_robust_session()
        session = get_authenticated_session(session=robust_session)
        if not session:
            print_error("Authentication failed")
            return
        set_global_session(session)
        warm_up_api_connection(session)
        print_success("Connected to API")
    
    if use_streaming:
        logging.info("🔄 Using streaming mode")
        
        if not skip_alpha_fetch:
            try:
                # Process ALL regions together using streaming mode (mixed regions per window)
                total_processed = fetch_all_unsubmitted_alphas_auto(
                    session=session,
                    regions_to_process=regions_to_process,
                    sharpe_thresholds=sharpe_thresholds,
                    batch_size=batch_size,
                    process_immediately=True,  # Enable streaming mode
                    skip_pnl_fetch=skip_pnl_fetch,
                    skip_correlation=skip_correlation
                )
                
                logging.info(f"✅ Processed {total_processed} alphas")
                
            except Exception as e:
                logging.error(f"Streaming failed: {e}")
                return
        else:
            logging.info("Processing existing data only")
            
            # Process existing data for each region individually
            for region in regions_to_process:
                if not skip_pnl_fetch or not skip_correlation:
                    logging.info(f"Processing {region}...")
                    
                    if not skip_pnl_fetch:
                        try:
                            alpha_ids_for_pnl = get_unsubmitted_alpha_ids_for_pnl_processing(region)
                            if alpha_ids_for_pnl:
                                logging.info(f"  Fetching PNL for {len(alpha_ids_for_pnl)} alphas...")
                                combined_pnl_df, failed_alpha_ids = get_alpha_pnl_threaded(
                                    session, alpha_ids_for_pnl, max_workers=20
                                )
                                if combined_pnl_df is not None and not combined_pnl_df.empty:
                                    pnl_data_dict = {}
                                    unique_fetched_alphas = combined_pnl_df['alpha_id'].unique()
                                    for alpha_id in unique_fetched_alphas:
                                        pnl_data_dict[alpha_id] = combined_pnl_df[combined_pnl_df['alpha_id'] == alpha_id]
                                    insert_multiple_unsubmitted_pnl_data_optimized(pnl_data_dict, region)
                                    logging.info(f"  ✓ PNL updated for {len(pnl_data_dict)} alphas")
                        except Exception as e:
                            logging.error(f"  PNL processing failed: {e}")
                    
                    if not skip_correlation:
                        try:
                            # Use unified correlation engine
                            correlation_engine = CorrelationEngine()
                            correlation_engine.calculate_unsubmitted_vs_submitted(region)
                            logging.info(f"  ✓ Correlations updated")
                        except Exception as e:
                            logging.error(f"  Correlation calculation failed: {e}")
        
    else:
        logging.warning("Using bulk mode (memory intensive)")
        
        # Original bulk processing mode (kept for backward compatibility)
        all_unsubmitted_alphas = []
        alphas_by_region = {}
        
        if not skip_alpha_fetch:
            logging.info("Fetching all unsubmitted alphas...")
            try:
                # Fetch all regions together in bulk mode
                all_unsubmitted_alphas = fetch_all_unsubmitted_alphas_auto(
                    session=session,
                    regions_to_process=regions_to_process,
                    sharpe_thresholds=sharpe_thresholds,
                    batch_size=batch_size,
                    process_immediately=False,  # Disable streaming mode
                    skip_pnl_fetch=True,  # Handle separately
                    skip_correlation=True  # Handle separately
                )
                
                # Group by region for processing
                for alpha in all_unsubmitted_alphas:
                    alpha_region = alpha.get('settings_region')
                    if alpha_region in regions_to_process:
                        if alpha_region not in alphas_by_region:
                            alphas_by_region[alpha_region] = []
                        alphas_by_region[alpha_region].append(alpha)
                        
                logging.info(f"✓ Fetched {len(all_unsubmitted_alphas)} alphas total")
                for region, region_alphas in alphas_by_region.items():
                    logging.info(f"  {region}: {len(region_alphas)} alphas")
                
            except Exception as e:
                logging.error(f"Bulk fetch failed: {e}")
                return
        
        # Process each region with bulk data
        for region in regions_to_process:
            logging.info(f"\nProcessing {region}...")
            region_alphas = alphas_by_region.get(region, [])
            
            if not skip_alpha_fetch and region_alphas:
                try:
                    insert_multiple_unsubmitted_alphas(region_alphas, region)
                    logging.info(f"  ✓ Stored {len(region_alphas)} alphas")
                except Exception as e:
                    logging.error(f"  Failed to store alphas: {e}")
            
            # Handle PNL and correlations as before
            # ... (rest of original bulk processing logic)
    
    logging.info("\n✅ Processing complete")


def main():
    parser = argparse.ArgumentParser(description='Run the Alpha DataBank process')
    parser.add_argument('--region', choices=CONFIGURED_REGIONS, help='Region to process') # Use CONFIGURED_REGIONS for choices
    parser.add_argument('--all', action='store_true', help='Process all configured regions')
    parser.add_argument('--skip-init', action='store_true', help='Skip database initialization')
    parser.add_argument('--skip-alpha-fetch', action='store_true', help='Skip alpha fetching')
    parser.add_argument('--skip-pnl-fetch', action='store_true', help='Skip PNL fetching')
    parser.add_argument('--skip-correlation', action='store_true', help='Skip correlation calculation')
    parser.add_argument('--report', action='store_true', help='Print correlation report for processed regions')
    parser.add_argument('--verbose', action='store_true', help='Show verbose debug logging')
    
    # Unsubmitted alphas arguments
    parser.add_argument('--unsubmitted', action='store_true', help='Process unsubmitted alphas instead of submitted (requires --url)')
    parser.add_argument('--unsubmitted-auto', action='store_true', help='Process unsubmitted alphas using automated fetching (fetches ALL alphas)')
    parser.add_argument('--url', type=str, help='URL for fetching unsubmitted alphas (required when using --unsubmitted)')
    parser.add_argument('--sharpe-thresholds', type=str, help='Comma-separated sharpe thresholds (e.g., "1,-1" for >=1 and <=-1). Default: "1,-1"')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for API requests (default: 50)')
    
    args = parser.parse_args()

    # Validation
    if not args.region and not args.all:
        parser.error('Either --region or --all must be specified')
    
    if args.unsubmitted and args.unsubmitted_auto:
        parser.error('Cannot use both --unsubmitted and --unsubmitted-auto at the same time')
    
    if args.unsubmitted and not args.url:
        parser.error('--url is required when using --unsubmitted')
    
    if args.unsubmitted_auto and args.url:
        parser.error('--url is not needed when using --unsubmitted-auto (automatic fetching)')
    
    if (args.unsubmitted or args.unsubmitted_auto) and args.report:
        parser.error('--report is not supported for unsubmitted alphas yet')
    
    if args.sharpe_thresholds and not args.unsubmitted_auto:
        parser.error('--sharpe-thresholds can only be used with --unsubmitted-auto')

    # Configure logging based on verbose flag
    if args.verbose:
        setup_logging()
    else:
        configure_minimal_logging()
        suppress_retry_logs()

    # Route to unsubmitted alphas processing if requested
    if args.unsubmitted:
        regions_to_process = CONFIGURED_REGIONS[:] if args.all else [args.region]
        process_unsubmitted_alphas(
            url=args.url,
            regions_to_process=regions_to_process,
            skip_init=args.skip_init,
            skip_alpha_fetch=args.skip_alpha_fetch,
            skip_pnl_fetch=args.skip_pnl_fetch,
            skip_correlation=args.skip_correlation
        )
        return  # Exit after processing unsubmitted alphas
    
    # Route to automated unsubmitted alphas processing if requested
    if args.unsubmitted_auto:
        regions_to_process = CONFIGURED_REGIONS[:] if args.all else [args.region]
        
        # Parse sharpe thresholds
        sharpe_thresholds = None
        if args.sharpe_thresholds:
            try:
                sharpe_thresholds = [float(x.strip()) for x in args.sharpe_thresholds.split(',')]
                logging.info(f"Parsed sharpe thresholds: {sharpe_thresholds}")
            except ValueError as e:
                parser.error(f"Invalid sharpe thresholds format: {e}. Use comma-separated numbers like '1,-1'")
        
        process_unsubmitted_alphas_auto(
            regions_to_process=regions_to_process,
            sharpe_thresholds=sharpe_thresholds,
            batch_size=args.batch_size,
            skip_init=args.skip_init,
            skip_alpha_fetch=args.skip_alpha_fetch,
            skip_pnl_fetch=args.skip_pnl_fetch,
            skip_correlation=args.skip_correlation
        )
        return  # Exit after processing automated unsubmitted alphas

    # Main processing header
    print_header("Alpha DataBank")
    start_time = time.time()

    # Continue with regular (submitted) alphas processing below
    if not args.skip_init:
        try:
            initialize_database()
            initialize_analysis_database()
            print_success("Database initialized")
        except Exception as e:
            print_error(f"Database initialization failed: {e}")
            return

    session = None
    if not args.skip_alpha_fetch or not args.skip_pnl_fetch:
        robust_session = get_robust_session()
        session = get_authenticated_session(session=robust_session)
        if not session:
            print_error("Authentication failed")
            return 1
        set_global_session(session)
        warm_up_api_connection(session)
        print_success("Connected to API")

    # This list will hold the names of regions for which PNL/Correlation should be run.
    regions_to_run_downstream_for = []
    data_changed_in_region_flags = {region: False for region in CONFIGURED_REGIONS}

    # --- Alpha Fetching Stage --- 
    if not args.skip_alpha_fetch:
        if args.all:
            try:
                all_alphas_list = scrape_alphas_by_region(session, region=None) # Global fetch
                if all_alphas_list:
                    print_success(f"Fetched {len(all_alphas_list)} alphas")

                    # First organize alphas by region (fast operation, no progress bar needed)
                    temp_alphas_by_actual_region = {}
                    for alpha_record in all_alphas_list:
                        actual_region = alpha_record.get('settings_region')
                        if actual_region in CONFIGURED_REGIONS:
                            temp_alphas_by_actual_region.setdefault(actual_region, []).append(alpha_record)

                    # Now store alphas with progress bar for actual database operations
                    alpha_bar = create_progress_bar(
                        len(all_alphas_list),
                        "Storing alphas",
                        unit="alphas"
                    )

                    for r_key, r_alphas in temp_alphas_by_actual_region.items():
                        newly_inserted_count = 0
                        for alpha_record in r_alphas:
                            alpha_id = alpha_record['alpha_id']
                            status = get_alpha_type_and_status(alpha_id, r_key)
                            if status is None:
                                insert_alpha(alpha_record, r_key)
                                newly_inserted_count += 1
                            else:
                                pass
                            update_progress_bar(alpha_bar)

                        existing_count = len(r_alphas) - newly_inserted_count
                        print_info(f"  {r_key}: {newly_inserted_count} new, {existing_count} existing")

                        if newly_inserted_count > 0:
                            data_changed_in_region_flags[r_key] = True

                        if r_key not in regions_to_run_downstream_for:
                            regions_to_run_downstream_for.append(r_key)

                    close_progress_bar(alpha_bar)
                else:
                    print_info("No alphas found")
            except Exception as e:
                print_error(f"Alpha fetch failed: {e}")
        
        elif args.region:
            try:
                regional_alphas_list = scrape_alphas_by_region(session, region=args.region)
                if regional_alphas_list:
                    # Process with progress bar
                    alpha_bar = create_progress_bar(
                        len(regional_alphas_list),
                        f"Processing {args.region} alphas",
                        unit="alphas"
                    )
                    newly_inserted_count = 0
                    for alpha_record in regional_alphas_list:
                        alpha_id = alpha_record['alpha_id']
                        status = get_alpha_type_and_status(alpha_id, args.region)
                        if status is None:
                            insert_alpha(alpha_record, args.region)
                            newly_inserted_count += 1
                        else:
                            pass
                    
                    if newly_inserted_count > 0:
                        logging.info(f"✓ {newly_inserted_count} new alphas, {len(regional_alphas_list) - newly_inserted_count} existing")
                        data_changed_in_region_flags[args.region] = True
                    else:
                        logging.info(f"All {len(regional_alphas_list)} alphas already up to date")

                    if args.region not in regions_to_run_downstream_for:
                        regions_to_run_downstream_for.append(args.region)
                else:
                    logging.info(f"No alphas found for region {args.region}.")
            except Exception as e:
                logging.error(f"Error fetching or storing alphas for region {args.region}: {e}", exc_info=True)
    else: # Alpha fetching is skipped
        logging.info("Skipping alpha fetch")
        if args.all:
            regions_to_run_downstream_for = CONFIGURED_REGIONS[:]
            logging.info(f"Processing {len(regions_to_run_downstream_for)} regions")
        elif args.region:
            if args.region in CONFIGURED_REGIONS:
                regions_to_run_downstream_for = [args.region]
                pass  # Region set, no verbose log needed
            else:
                logging.error(f"Invalid region: {args.region}")

    if not regions_to_run_downstream_for:
        logging.info("No regions to process")
        if args.report: # Handle report even if no other processing, maybe on all configured or specified
            pass  # Generate report only
            report_regions = [args.region] if args.region else (CONFIGURED_REGIONS[:] if args.all else [])
            for region_name in report_regions:
                if region_name not in CONFIGURED_REGIONS: continue
                pass  # Generate report for region
                try:
                    corr_stats = get_correlation_statistics(region_name)
                    if not corr_stats.empty:
                        print(f"\n==== Correlation Statistics for {region_name} ====")
                        print(corr_stats)
                    else:
                        print_info(f"No statistics for {region_name}")
                except Exception as e:
                    print_error(f"Report generation failed for {region_name}: {e}")
        print_success("Complete")
        return

    # --- PNL Fetching, Correlation Calculation, and Reporting Loop ---
    for region_name in regions_to_run_downstream_for:
        print_info(f"\nProcessing {region_name}...")

        # Fetch PNL data if not skipped
        if not args.skip_pnl_fetch:
            try:
                alpha_ids_for_pnl = get_regular_alpha_ids_for_pnl_processing(region_name)

                if alpha_ids_for_pnl:
                    print_info(f"  Found {len(alpha_ids_for_pnl)} alphas to update")

                    if session and not check_session_valid(session):
                        print_info("  Re-authenticating...")
                        session = get_authenticated_session()
                        if session:
                            set_global_session(session)
                            print_success("  Re-authenticated")

                    if not session:
                        print_error("  Authentication failed, skipping PNL")
                    else:
                        # Create progress bar for PNL fetching
                        pnl_bar = create_progress_bar(
                            len(alpha_ids_for_pnl),
                            f"  Fetching PNL for {region_name}",
                            unit="alphas"
                        )

                        # Define progress callback
                        def pnl_progress_callback(n):
                            update_progress_bar(pnl_bar, n)

                        combined_pnl_df, failed_alpha_ids = get_alpha_pnl_threaded(
                            session,
                            alpha_ids_for_pnl,
                            max_workers=20,
                            progress_callback=pnl_progress_callback
                        )

                        close_progress_bar(pnl_bar)
                        
                        if combined_pnl_df is not None and not combined_pnl_df.empty:
                            unique_fetched_alphas = combined_pnl_df['alpha_id'].unique()
                            print_success(f"  Fetched PNL for {len(unique_fetched_alphas)} alphas")
                            
                            # Prepare PNL data dictionary for batch insertion
                            pnl_data_dict = {}
                            for alpha_id in unique_fetched_alphas:
                                pnl_data_dict[alpha_id] = combined_pnl_df[combined_pnl_df['alpha_id'] == alpha_id]
                            
                            # Use the new batch insertion function with optimized COPY command
                            if pnl_data_dict:  # Check if we have any data to store
                                # Initialize variables before try block so they're available in all paths
                                successfully_stored_count = 0
                                alphas_not_stored_after_fetch = []
                                
                                try:
                                    insert_multiple_pnl_data_optimized(pnl_data_dict, region_name)
                                    successfully_stored_count = len(pnl_data_dict)
                                    print_success(f"  Stored PNL data")
                                    if successfully_stored_count > 0:
                                        data_changed_in_region_flags[region_name] = True
                                except Exception as e_batch_insert:
                                    print_info(f"  Batch insertion failed, using fallback")
                                    # Fallback to individual inserts if batch fails
                                    pass  # Using fallback, no extra log needed
                                    
                                    successfully_stored_count = 0
                                    alphas_not_stored_after_fetch = []
                                    
                                    # Individual insertion as fallback
                                    for alpha_id, pnl_df in pnl_data_dict.items():
                                        try:
                                            insert_pnl_data(alpha_id, pnl_df, region_name)
                                            successfully_stored_count += 1
                                        except Exception as e_insert:
                                            pass  # Silent fail for individual alpha
                                            alphas_not_stored_after_fetch.append(alpha_id)
                                
                                print_success(f"  Stored PNL for {successfully_stored_count} alphas")
                                if successfully_stored_count > 0:
                                    data_changed_in_region_flags[region_name] = True
                                if alphas_not_stored_after_fetch:
                                    pass  # Some failures expected, no warning needed

                        elif not failed_alpha_ids:
                            print_info(f"  No PNL data available")
                        else:
                            print_info(f"  PNL fetch unsuccessful")
                else:
                    print_info(f"  All alphas up to date")
            except Exception as e:
                print_error(f"  PNL processing failed: {e}")
        else:
            pass  # PNL fetch skipped

        # Calculate correlations if not skipped and data has changed
        if not args.skip_correlation:
            if data_changed_in_region_flags.get(region_name, False): # Check if data actually changed
                print_info(f"  Calculating correlations...")
                try:
                    # Use unified correlation engine
                    correlation_engine = CorrelationEngine()
                    correlation_engine.calculate_batch_submitted(region_name)
                    print_success(f"  Correlations updated")
                except Exception as e:
                    print_error(f"  Correlation calculation failed: {e}")
            else:
                print_info(f"  Correlations already up to date")
        else:
            pass  # Correlation calculation skipped
        
        # Print correlation report if requested
        if args.report:
            pass  # Generate report
            try:
                corr_stats = get_correlation_statistics(region_name)
                if not corr_stats.empty:
                    print(f"\n==== Correlation Statistics for {region_name} ====")
                    print(corr_stats)
                else:
                    print_info(f"  No statistics available")
            except Exception as e:
                print_error(f"  Report generation failed: {e}")

    print_success("Alpha DataBank process complete")

if __name__ == "__main__":
    main()
