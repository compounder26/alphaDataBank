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

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from api.auth import get_authenticated_session, check_session_valid, set_global_session
from api.alpha_fetcher import scrape_alphas_by_region, get_alpha_pnl_threaded, get_robust_session, warm_up_api_connection
from database.schema import initialize_database
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

# Import optimized correlation calculation
from scripts.update_correlations_optimized import calculate_and_store_correlations_optimized
from utils.helpers import setup_logging, print_correlation_report
from config.database_config import REGIONS

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
from scripts.calculate_unsubmitted_correlations import calculate_unsubmitted_vs_submitted_correlations

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
    logging.info("Starting unsubmitted alphas processing...")
    
    # Initialize unsubmitted database schema if needed
    if not skip_init:
        logging.info("Initializing unsubmitted alphas database schema...")
        try:
            initialize_unsubmitted_database()
            logging.info("Unsubmitted database schema initialization complete")
        except Exception as e:
            logging.error(f"Unsubmitted database schema initialization failed: {e}")
            return
    
    # Create session for API calls
    session = None
    if not skip_alpha_fetch or not skip_pnl_fetch:
        robust_session = get_robust_session()
        session = get_authenticated_session(session=robust_session)
        if not session:
            logging.error("Failed to authenticate session for unsubmitted alphas")
            return
        set_global_session(session)  # Enable automatic reauthentication
        warm_up_api_connection(session)
    
    # Fetch all unsubmitted alphas once (outside region loop)
    alphas_by_region = {}
    all_unsubmitted_alphas = []
    
    if not skip_alpha_fetch:
        logging.info("Fetching all unsubmitted alphas from URL...")
        try:
            all_unsubmitted_alphas = fetch_unsubmitted_alphas_from_url(session, url)
            if all_unsubmitted_alphas:
                logging.info(f"Successfully fetched {len(all_unsubmitted_alphas)} unsubmitted alphas total")
                
                # Group alphas by region
                for alpha in all_unsubmitted_alphas:
                    alpha_region = alpha.get('settings_region')
                    if alpha_region in regions_to_process:
                        if alpha_region not in alphas_by_region:
                            alphas_by_region[alpha_region] = []
                        alphas_by_region[alpha_region].append(alpha)
                
                # Log distribution
                for region, alphas_list in alphas_by_region.items():
                    logging.info(f"Found {len(alphas_list)} unsubmitted alphas for region {region}")
                    
            else:
                logging.warning("No unsubmitted alphas fetched from URL")
        except Exception as e:
            logging.error(f"Error fetching unsubmitted alphas from URL: {e}")
            return
    
    # Process each region with its pre-fetched alphas
    for region in regions_to_process:
        logging.info(f"--- Processing unsubmitted alphas for region: {region} ---")
        data_changed = False
        
        # Get alphas for this region (already fetched)
        region_alphas = alphas_by_region.get(region, [])
        
        # Insert unsubmitted alphas metadata for this region
        if not skip_alpha_fetch and region_alphas:
            try:
                # Use batch insert for better performance
                insert_multiple_unsubmitted_alphas(region_alphas, region)
                logging.info(f"Processed {len(region_alphas)} unsubmitted alphas for region {region}")
                data_changed = True
            except Exception as e:
                logging.error(f"Error inserting unsubmitted alphas for region {region}: {e}")
        elif not skip_alpha_fetch:
            logging.info(f"No unsubmitted alphas found for region {region} in the fetched data")
        
        # Fetch PNL data for unsubmitted alphas
        if not skip_pnl_fetch:
            logging.info(f"Fetching PNL data for unsubmitted alphas in region {region}...")
            try:
                alpha_ids_for_pnl = get_unsubmitted_alpha_ids_for_pnl_processing(region)
                
                if alpha_ids_for_pnl:
                    logging.info(f"Found {len(alpha_ids_for_pnl)} unsubmitted alphas needing PNL processing in region {region}")
                    
                    # Use the same PNL fetching logic as submitted alphas
                    combined_pnl_df, failed_alpha_ids = get_alpha_pnl_threaded(
                        session, alpha_ids_for_pnl, max_workers=10
                    )
                    
                    if combined_pnl_df is not None and not combined_pnl_df.empty:
                        unique_fetched_alphas = combined_pnl_df['alpha_id'].unique()
                        logging.info(f"Successfully fetched PNL data for {len(unique_fetched_alphas)} unsubmitted alphas in region {region}")
                        
                        # Prepare PNL data for batch insertion
                        pnl_data_dict = {}
                        for alpha_id in unique_fetched_alphas:
                            pnl_data_dict[alpha_id] = combined_pnl_df[combined_pnl_df['alpha_id'] == alpha_id]
                        
                        # Batch insert PNL data
                        if pnl_data_dict:
                            try:
                                insert_multiple_unsubmitted_pnl_data_optimized(pnl_data_dict, region)
                                logging.info(f"Batch stored PNL data for {len(pnl_data_dict)} unsubmitted alphas in region {region}")
                                data_changed = True
                            except Exception as e:
                                logging.error(f"Error during batch PNL insertion for unsubmitted alphas in region {region}: {e}")
                                # Fallback to individual inserts
                                successfully_stored = 0
                                for alpha_id, pnl_df in pnl_data_dict.items():
                                    try:
                                        insert_unsubmitted_pnl_data(alpha_id, pnl_df, region)
                                        successfully_stored += 1
                                    except Exception as e_insert:
                                        logging.error(f"Error storing PNL data for unsubmitted alpha {alpha_id}: {e_insert}")
                                
                                if successfully_stored > 0:
                                    logging.info(f"Stored PNL data for {successfully_stored}/{len(unique_fetched_alphas)} unsubmitted alphas in region {region}")
                                    data_changed = True
                    else:
                        logging.info(f"No PNL data returned for unsubmitted alphas in region {region}")
                else:
                    logging.info(f"No unsubmitted alphas found needing PNL processing in region {region}")
            except Exception as e:
                logging.error(f"Error during PNL processing for unsubmitted alphas in region {region}: {e}")
        
        # Calculate correlations for unsubmitted alphas
        if not skip_correlation and data_changed:
            logging.info(f"Calculating correlations for unsubmitted alphas in region {region}...")
            try:
                calculate_unsubmitted_vs_submitted_correlations(region)
                logging.info(f"Successfully calculated correlations for unsubmitted alphas in region {region}")
            except Exception as e:
                logging.error(f"Error calculating correlations for unsubmitted alphas in region {region}: {e}")
        elif not skip_correlation:
            logging.info(f"Skipping correlation calculation for unsubmitted alphas in region {region} - no data changes")
    
    logging.info("Unsubmitted alphas processing complete.")


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
    logging.info("Starting automated unsubmitted alphas processing...")
    
    if sharpe_thresholds:
        logging.info(f"Using custom sharpe thresholds: {sharpe_thresholds}")
    else:
        sharpe_thresholds = [1.0, -1.0]
        logging.info("Using default sharpe thresholds: >= 1.0 and <= -1.0")
    
    # Initialize unsubmitted database schema if needed
    if not skip_init:
        logging.info("Initializing unsubmitted alphas database schema...")
        try:
            initialize_unsubmitted_database()
            logging.info("Unsubmitted database schema initialization complete")
        except Exception as e:
            logging.error(f"Unsubmitted database schema initialization failed: {e}")
            return
    
    # Create session for API calls
    session = None
    if not skip_alpha_fetch or not skip_pnl_fetch:
        robust_session = get_robust_session()
        session = get_authenticated_session(session=robust_session)
        if not session:
            logging.error("Failed to authenticate session for automated unsubmitted alphas")
            return
        set_global_session(session)  # Enable automatic reauthentication
        warm_up_api_connection(session)
    
    if use_streaming:
        logging.info("🔄 STREAMING MODE: Processing windows globally (mixed regions per window)")
        
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
                
                logging.info(f"✅ Streaming complete: {total_processed} total alphas processed across all regions")
                
            except Exception as e:
                logging.error(f"Error during streaming processing: {e}")
                return
        else:
            logging.info(f"⏭️  Skipping alpha fetch - processing existing data only")
            
            # Process existing data for each region individually
            for region in regions_to_process:
                if not skip_pnl_fetch or not skip_correlation:
                    logging.info(f"Processing existing data for region {region}...")
                    
                    if not skip_pnl_fetch:
                        try:
                            alpha_ids_for_pnl = get_unsubmitted_alpha_ids_for_pnl_processing(region)
                            if alpha_ids_for_pnl:
                                logging.info(f"📈 Fetching PNL for {len(alpha_ids_for_pnl)} existing alphas in {region}...")
                                combined_pnl_df, failed_alpha_ids = get_alpha_pnl_threaded(
                                    session, alpha_ids_for_pnl, max_workers=10
                                )
                                if combined_pnl_df is not None and not combined_pnl_df.empty:
                                    pnl_data_dict = {}
                                    unique_fetched_alphas = combined_pnl_df['alpha_id'].unique()
                                    for alpha_id in unique_fetched_alphas:
                                        pnl_data_dict[alpha_id] = combined_pnl_df[combined_pnl_df['alpha_id'] == alpha_id]
                                    insert_multiple_unsubmitted_pnl_data_optimized(pnl_data_dict, region)
                                    logging.info(f"✅ Updated PNL for {len(pnl_data_dict)} alphas in {region}")
                        except Exception as e:
                            logging.error(f"Error processing PNL for {region}: {e}")
                    
                    if not skip_correlation:
                        try:
                            calculate_unsubmitted_vs_submitted_correlations(region)
                            logging.info(f"✅ Updated correlations for {region}")
                        except Exception as e:
                            logging.error(f"Error calculating correlations for {region}: {e}")
        
    else:
        logging.warning("⚠️  BULK MODE: Using memory-intensive processing (not recommended for large datasets)")
        
        # Original bulk processing mode (kept for backward compatibility)
        all_unsubmitted_alphas = []
        alphas_by_region = {}
        
        if not skip_alpha_fetch:
            logging.info("Starting automated fetch of ALL unsubmitted alphas...")
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
                        
                logging.info(f"Total fetched in bulk mode: {len(all_unsubmitted_alphas)} alphas")
                for region, region_alphas in alphas_by_region.items():
                    logging.info(f"  - {region}: {len(region_alphas)} alphas")
                
            except Exception as e:
                logging.error(f"Error during bulk fetch: {e}")
                return
        
        # Process each region with bulk data
        for region in regions_to_process:
            logging.info(f"--- Processing region {region} in bulk mode ---")
            region_alphas = alphas_by_region.get(region, [])
            
            if not skip_alpha_fetch and region_alphas:
                try:
                    insert_multiple_unsubmitted_alphas(region_alphas, region)
                    logging.info(f"Inserted {len(region_alphas)} alphas for region {region}")
                except Exception as e:
                    logging.error(f"Error inserting alphas for region {region}: {e}")
            
            # Handle PNL and correlations as before
            # ... (rest of original bulk processing logic)
    
    logging.info("Automated unsubmitted alphas processing complete.")


def main():
    parser = argparse.ArgumentParser(description='Run the Alpha DataBank process')
    parser.add_argument('--region', choices=CONFIGURED_REGIONS, help='Region to process') # Use CONFIGURED_REGIONS for choices
    parser.add_argument('--all', action='store_true', help='Process all configured regions')
    parser.add_argument('--skip-init', action='store_true', help='Skip database initialization')
    parser.add_argument('--skip-alpha-fetch', action='store_true', help='Skip alpha fetching')
    parser.add_argument('--skip-pnl-fetch', action='store_true', help='Skip PNL fetching')
    parser.add_argument('--skip-correlation', action='store_true', help='Skip correlation calculation')
    parser.add_argument('--report', action='store_true', help='Print correlation report for processed regions')
    
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

    setup_logging()

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

    # Continue with regular (submitted) alphas processing below
    if not args.skip_init:
        logging.info("Initializing database...")
        try:
            initialize_database()
            logging.info("Database initialization complete")
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            return

    session = None
    if not args.skip_alpha_fetch or not args.skip_pnl_fetch:
        # Create a robust session with optimized connection pooling
        robust_session = get_robust_session()
        # Authenticate the robust session
        session = get_authenticated_session(session=robust_session)
        if not session:
            logging.error("Failed to authenticate session")
            return 1
        
        set_global_session(session)  # Enable automatic reauthentication
        # Warm up the API connection before intensive operations
        warm_up_api_connection(session)

    # This list will hold the names of regions for which PNL/Correlation should be run.
    regions_to_run_downstream_for = []
    data_changed_in_region_flags = {region: False for region in CONFIGURED_REGIONS}

    # --- Alpha Fetching Stage --- 
    if not args.skip_alpha_fetch:
        if args.all:
            logging.info("Global alpha fetch initiated (--all)...")
            try:
                all_alphas_list = scrape_alphas_by_region(session, region=None) # Global fetch
                if all_alphas_list:
                    logging.info(f"Globally fetched {len(all_alphas_list)} alphas. Storing by actual region...")
                    temp_alphas_by_actual_region = {}
                    for alpha_record in all_alphas_list:
                        actual_region = alpha_record.get('settings_region')
                        if actual_region in CONFIGURED_REGIONS:
                            temp_alphas_by_actual_region.setdefault(actual_region, []).append(alpha_record)
                        else:
                            logging.warning(f"Alpha {alpha_record.get('alpha_id')} has region '{actual_region}' which is not in CONFIGURED_REGIONS. Skipping.")
                    
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
                        
                        if newly_inserted_count > 0:
                            logging.info(f"Stored {newly_inserted_count} new alphas for region {r_key} from global fetch. {len(r_alphas) - newly_inserted_count} already existed.")
                            data_changed_in_region_flags[r_key] = True
                        else:
                            logging.info(f"No new alphas to store for region {r_key} from global fetch. All {len(r_alphas)} already existed.")

                        if r_key not in regions_to_run_downstream_for:
                            regions_to_run_downstream_for.append(r_key)
                else:
                    logging.warning("Global alpha fetch (--all) returned no alphas.")
            except Exception as e:
                logging.error(f"Error during global alpha fetch: {e}", exc_info=True)
        
        elif args.region:
            logging.info(f"Regional alpha fetch initiated for {args.region}...")
            try:
                regional_alphas_list = scrape_alphas_by_region(session, region=args.region)
                if regional_alphas_list:
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
                        logging.info(f"Stored {newly_inserted_count} new alphas for region {args.region}. {len(regional_alphas_list) - newly_inserted_count} already existed.")
                        data_changed_in_region_flags[args.region] = True
                    else:
                        logging.info(f"No new alphas to store for region {args.region}. All {len(regional_alphas_list)} already existed.")

                    if args.region not in regions_to_run_downstream_for:
                        regions_to_run_downstream_for.append(args.region)
                else:
                    logging.info(f"No alphas found for region {args.region}.")
            except Exception as e:
                logging.error(f"Error fetching or storing alphas for region {args.region}: {e}", exc_info=True)
    else: # Alpha fetching is skipped
        logging.info("Alpha fetching skipped (--skip-alpha-fetch).")
        if args.all:
            regions_to_run_downstream_for = CONFIGURED_REGIONS[:]
            logging.info(f"Downstream processing will target all configured regions: {regions_to_run_downstream_for}")
        elif args.region:
            if args.region in CONFIGURED_REGIONS:
                regions_to_run_downstream_for = [args.region]
                logging.info(f"Downstream processing will target region: {args.region}")
            else:
                logging.error(f"Specified region {args.region} is not in CONFIGURED_REGIONS. Cannot add to downstream processing list.")

    if not regions_to_run_downstream_for:
        logging.info("No regions identified for PNL/Correlation processing. Exiting.")
        if args.report: # Handle report even if no other processing, maybe on all configured or specified
            logging.info("Attempting to generate report based on --report flag, but no regions were actively processed for PNL/Corr.")
            report_regions = [args.region] if args.region else (CONFIGURED_REGIONS[:] if args.all else [])
            for region_name in report_regions:
                if region_name not in CONFIGURED_REGIONS: continue
                logging.info(f"Generating correlation report for {region_name} (as per --report flag only)...")
                try:
                    corr_stats = get_correlation_statistics(region_name)
                    if not corr_stats.empty:
                        print(f"\n==== Correlation Statistics for {region_name} ====")
                        print(corr_stats)
                    else: logging.warning(f"No correlation statistics found for region {region_name}")
                except Exception as e: logging.error(f"Error generating report for {region_name}: {e}", exc_info=True)
        logging.info("Alpha DataBank process complete (or no regions to process).")
        return

    # --- PNL Fetching, Correlation Calculation, and Reporting Loop --- 
    for region_name in regions_to_run_downstream_for:
        logging.info(f"--- Processing region: {region_name} ---")

        # Fetch PNL data if not skipped
        if not args.skip_pnl_fetch:
            logging.info(f"Attempting PNL fetch for region: {region_name}...")
            try:
                alpha_ids_for_pnl = get_regular_alpha_ids_for_pnl_processing(region_name)

                if alpha_ids_for_pnl:
                    logging.info(f"Found {len(alpha_ids_for_pnl)} REGULAR/SUPER alphas needing PNL processing in region {region_name}.")
                    
                    if session and not check_session_valid(session):
                        logging.info(f"Session became invalid before batch PNL fetch for region {region_name}. Attempting to re-authenticate...")
                        session = get_authenticated_session()
                        if session:
                            set_global_session(session)  # Update global session
                    
                    if not session:
                        logging.error(f"No valid session to fetch PNL for region {region_name}. Skipping PNL fetch for this region.")
                    else:
                        combined_pnl_df, failed_alpha_ids = get_alpha_pnl_threaded(
                            session, 
                            alpha_ids_for_pnl, 
                            max_workers=10 
                        )
                        
                        if combined_pnl_df is not None and not combined_pnl_df.empty:
                            unique_fetched_alphas = combined_pnl_df['alpha_id'].unique()
                            logging.info(f"Successfully fetched PNL data for {len(unique_fetched_alphas)} alphas in region {region_name}.")
                            
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
                                    logging.info(f"Batch stored PNL data for {successfully_stored_count} alphas in region {region_name} using optimized COPY command.")
                                    if successfully_stored_count > 0:
                                        data_changed_in_region_flags[region_name] = True
                                except Exception as e_batch_insert:
                                    logging.error(f"Error during optimized batch PNL insertion for region {region_name}: {e_batch_insert}", exc_info=True)
                                    # Fallback to individual inserts if batch fails
                                    logging.warning(f"Falling back to individual PNL insertions for {len(unique_fetched_alphas)} alphas in region {region_name}")
                                    
                                    successfully_stored_count = 0
                                    alphas_not_stored_after_fetch = []
                                    
                                    # Individual insertion as fallback
                                    for alpha_id, pnl_df in pnl_data_dict.items():
                                        try:
                                            insert_pnl_data(alpha_id, pnl_df, region_name)
                                            successfully_stored_count += 1
                                        except Exception as e_insert:
                                            logging.error(f"Error storing PNL data for alpha {alpha_id} (region {region_name}): {e_insert}", exc_info=True)
                                            alphas_not_stored_after_fetch.append(alpha_id)
                                
                                logging.info(f"Stored PNL data for {successfully_stored_count}/{len(unique_fetched_alphas)} fetched alphas in region {region_name} (fallback mode).")
                                if successfully_stored_count > 0:
                                    data_changed_in_region_flags[region_name] = True
                                if alphas_not_stored_after_fetch:
                                    logging.warning(f"Failed to store PNL for {len(alphas_not_stored_after_fetch)} alphas")

                        elif not failed_alpha_ids:
                            logging.info(f"No PNL data returned for {len(alpha_ids_for_pnl)} REGULAR/SUPER alphas in region {region_name}.")
                        else:
                            logging.warning(f"No PNL data successfully fetched for region {region_name}. Failures: {len(failed_alpha_ids)}.")
                else:
                    logging.info(f"No REGULAR/SUPER alphas found in DB for region {region_name} needing PNL processing. Skipping PNL fetch.")
            except Exception as e:
                logging.error(f"Error during PNL processing for region {region_name}: {e}")
        else:
            logging.info(f"Skipping PNL fetch for region {region_name} as per --skip-pnl-fetch flag.")

        # Calculate correlations if not skipped and data has changed
        if not args.skip_correlation:
            if data_changed_in_region_flags.get(region_name, False): # Check if data actually changed
                logging.info(f"Attempting optimized correlation calculation for region: {region_name} (data changed)...")
                try:
                    # Use the Cython-accelerated correlation calculation
                    calculate_and_store_correlations_optimized(region_name)
                    logging.info(f"Successfully updated correlation statistics for region {region_name} with optimized calculation.")
                except Exception as e:
                    logging.error(f"Error calculating correlations for {region_name}: {e}")
            else:
                logging.info(f"Skipping correlation calculation for region {region_name} as no new alpha metadata or PNL data was processed in this run.")
        else:
            logging.info(f"Skipping correlation calculation for region {region_name} as per --skip-correlation flag.")
        
        # Print correlation report if requested
        if args.report:
            logging.info(f"Attempting to generate correlation report for region: {region_name}...")
            try:
                corr_stats = get_correlation_statistics(region_name)
                if not corr_stats.empty:
                    print(f"\n==== Correlation Statistics for {region_name} ====")
                    print(corr_stats)
                else:
                    logging.info(f"No correlation statistics found to report for region {region_name}.")
            except Exception as e:
                logging.error(f"Error generating correlation report for {region_name}: {e}")
    
    logging.info("Alpha DataBank process complete.")

if __name__ == "__main__":
    main()
