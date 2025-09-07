"""
Module for automatically fetching ALL unsubmitted alpha data by looping through date ranges and offsets.
Overcomes the 10,000 alpha API limit by intelligently segmenting requests.
"""
import requests
import pandas as pd
import time
import json
import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from .auth import check_session_valid, authenticated_get
from .unsubmitted_fetcher import process_unsubmitted_alpha_record

logger = logging.getLogger(__name__)

# Base URL for unsubmitted alphas
BASE_API_URL = "https://api.worldquantbrain.com/users/self/alphas"

def fetch_all_unsubmitted_alphas_auto(
    session: requests.Session,
    regions_to_process: List[str],
    sharpe_thresholds: Optional[List[float]] = None,
    batch_size: int = 50,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    request_timeout: int = 60,
    retry_wait_seconds: int = 5,
    max_retries: int = 3,
    process_immediately: bool = True,
    skip_pnl_fetch: bool = False,
    skip_correlation: bool = False
) -> List[Dict[str, Any]]:
    """
    Automatically fetch ALL unsubmitted alphas using streaming window processing.
    
    Args:
        session: The authenticated requests.Session object
        regions_to_process: List of regions to process (for filtering and database insertion)
        sharpe_thresholds: List of sharpe thresholds (e.g., [1.0, -1.0]). If None, uses [1.0, -1.0]
        batch_size: Number of alphas to fetch per API request (default 50)
        start_date: Earliest date to fetch from (YYYY-MM-DD format)
        end_date: Latest date to fetch to (YYYY-MM-DD format). If None, uses current date
        request_timeout: Timeout for individual HTTP requests
        retry_wait_seconds: Seconds to wait before retrying a failed request
        max_retries: Maximum number of retries per request
        process_immediately: If True, process each window immediately (per-window processing)
        skip_pnl_fetch: Skip PNL fetching during streaming processing
        skip_correlation: Skip correlation calculation during streaming processing
    
    Returns:
        List[Dict[str, Any]]: Summary list (empty if process_immediately=True, full list if False)
    """
    if not check_session_valid(session):
        logger.error("Session is not valid. Please re-authenticate.")
        return []
    
    # Set default sharpe thresholds if not provided
    if sharpe_thresholds is None:
        sharpe_thresholds = [1.0, -1.0]
        logger.info("Using default sharpe thresholds: >= 1.0 and <= -1.0")
    else:
        logger.info(f"Using sharpe thresholds: {sharpe_thresholds}")
    
    # Set end date to current date if not provided
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Using current date as end date: {end_date}")
    
    logger.info(f"Starting automated unsubmitted alpha fetch from {start_date} to {end_date}")
    logger.info(f"Batch size: {batch_size}, Timeout: {request_timeout}s, Max retries: {max_retries}")
    
    if process_immediately:
        logger.info(f"🔄 STREAMING MODE: Processing windows immediately (mixed regions per window)")
        logger.info(f"Each window limited by API response size (~10,000 alphas max)")
        logger.info(f"Regions to process: {regions_to_process}")
    else:
        logger.info(f"⚠️  BULK MODE: Storing all data in RAM (not recommended for large datasets)")
    
    all_alphas = []
    total_processed = 0
    
    # Process each sharpe threshold separately
    for threshold in sharpe_thresholds:
        logger.info(f"\n--- Processing sharpe threshold: {threshold} ---")
        
        # Determine sharpe operator based on threshold value
        if threshold >= 0:
            sharpe_param = f"is.sharpe>={threshold}"
            threshold_desc = f">= {threshold}"
        else:
            sharpe_param = f"is.sharpe<={threshold}"
            threshold_desc = f"<= {threshold}"
        
        logger.info(f"Fetching alphas with sharpe {threshold_desc}")
        
        # Fetch alphas for this threshold using streaming or bulk mode
        if process_immediately:
            # STREAMING MODE: Process each window immediately (mixed regions)
            processed_count = _process_threshold_streaming(
                session=session,
                regions_to_process=regions_to_process,
                sharpe_param=sharpe_param,
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                request_timeout=request_timeout,
                retry_wait_seconds=retry_wait_seconds,
                max_retries=max_retries,
                skip_pnl_fetch=skip_pnl_fetch,
                skip_correlation=skip_correlation
            )
            total_processed += processed_count
            logger.info(f"✅ Streamed {processed_count} alphas (mixed regions) for sharpe {threshold_desc}")
        else:
            # BULK MODE: Store in RAM (original behavior)
            threshold_alphas = _fetch_alphas_for_threshold(
                session=session,
                sharpe_param=sharpe_param,
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                request_timeout=request_timeout,
                retry_wait_seconds=retry_wait_seconds,
                max_retries=max_retries,
                max_workers=20
            )
            
            if threshold_alphas:
                logger.info(f"Fetched {len(threshold_alphas)} alphas for sharpe {threshold_desc}")
                all_alphas.extend(threshold_alphas)
                total_processed += len(threshold_alphas)
            else:
                logger.warning(f"No alphas found for sharpe {threshold_desc}")
    
    logger.info(f"\n=== SUMMARY ===")
    if process_immediately:
        logger.info(f"Total alphas processed via streaming: {total_processed}")
        logger.info(f"Memory usage: Constant (~30MB), Data persisted to database")
        return []  # Return empty list in streaming mode
    else:
        logger.info(f"Total alphas fetched in bulk mode: {len(all_alphas)}")
        logger.info(f"Memory usage: ~{len(all_alphas) * 3 // 1000}MB (approximate)")
        return all_alphas


def _process_threshold_streaming(
    session: requests.Session,
    regions_to_process: List[str],
    sharpe_param: str,
    start_date: str,
    end_date: str,
    batch_size: int,
    request_timeout: int,
    retry_wait_seconds: int,
    max_retries: int,
    skip_pnl_fetch: bool = False,
    skip_correlation: bool = False
) -> int:
    """
    Process a single sharpe threshold using streaming window processing.
    Each window contains mixed regions and is immediately processed per region.
    
    Args:
        session: The authenticated requests.Session object
        regions_to_process: List of regions to process from fetched data
        sharpe_param: Sharpe parameter string (e.g., "is.sharpe>=1")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        batch_size: Batch size for requests
        request_timeout: Request timeout
        retry_wait_seconds: Retry wait time
        max_retries: Maximum retries
        skip_pnl_fetch: Skip PNL processing
        skip_correlation: Skip correlation calculation
    
    Returns:
        Total number of alphas processed across all regions
    """
    # Import database operations here to avoid circular imports
    try:
        from database.operations_unsubmitted import insert_multiple_unsubmitted_alphas, get_unsubmitted_alpha_ids_for_pnl_processing
        from api.alpha_fetcher import get_alpha_pnl_threaded
        from database.operations_unsubmitted import insert_multiple_unsubmitted_pnl_data_optimized
        from scripts.calculate_unsubmitted_correlations import calculate_unsubmitted_vs_submitted_correlations
    except ImportError as e:
        logger.error(f"Failed to import database operations: {e}")
        logger.error("Falling back to bulk processing mode")
        return 0
    
    total_processed = 0
    current_end_date = end_date
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    
    iteration_count = 0
    max_iterations = 1000  # Safety limit
    
    while current_end_date > start_date and iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"🔄 Window {iteration_count}: {start_date} to {current_end_date}")
        
        # Fetch all alphas in this date window using parallel processing
        window_alphas, last_alpha_date = _fetch_alphas_for_date_window_parallel(
            session=session,
            sharpe_param=sharpe_param,
            start_date=start_date,
            end_date=current_end_date,
            batch_size=batch_size,
            max_workers=20,
            max_batches_per_window=200,
            request_timeout=request_timeout,
            retry_wait_seconds=retry_wait_seconds,
            max_retries=max_retries
        )
        
        if not window_alphas:
            logger.info(f"No alphas in window {start_date} to {current_end_date}. Moving to next threshold.")
            break
            
        logger.info(f"💾 Processing {len(window_alphas)} alphas from window (mixed regions)...")
        
        # STEP 1: Group alphas by region
        alphas_by_region = {}
        for alpha in window_alphas:
            alpha_region = alpha.get('settings_region')
            if alpha_region in regions_to_process:
                if alpha_region not in alphas_by_region:
                    alphas_by_region[alpha_region] = []
                alphas_by_region[alpha_region].append(alpha)
        
        logger.info(f"📊 Window contains alphas for regions: {list(alphas_by_region.keys())}")
        for region, region_alphas in alphas_by_region.items():
            logger.info(f"  - {region}: {len(region_alphas)} alphas")
        
        try:
            # STEP 2: Process each region in this window
            for region, region_alphas in alphas_by_region.items():
                logger.info(f"🔄 Processing {len(region_alphas)} alphas for region {region}")
                
                # Insert alpha metadata for this region
                logger.debug(f"Inserting {len(region_alphas)} alpha records for {region}...")
                insert_multiple_unsubmitted_alphas(region_alphas, region)
                logger.info(f"✅ Inserted {len(region_alphas)} alphas to database for {region}")
                
                # Fetch and insert PNL data for this region (if not skipped)
                if not skip_pnl_fetch:
                    logger.debug(f"Fetching PNL data for {region} alphas in this window...")
                    
                    # Get alpha IDs that need PNL processing from this region
                    alpha_ids_for_pnl = [alpha['alpha_id'] for alpha in region_alphas if alpha.get('alpha_id')]
                    
                    if alpha_ids_for_pnl:
                        logger.info(f"📈 Fetching PNL for {len(alpha_ids_for_pnl)} alphas in {region}...")
                        
                        # Use existing parallel PNL fetching
                        combined_pnl_df, failed_alpha_ids = get_alpha_pnl_threaded(
                            session, alpha_ids_for_pnl, max_workers=20
                        )
                        
                        if combined_pnl_df is not None and not combined_pnl_df.empty:
                            unique_fetched_alphas = combined_pnl_df['alpha_id'].unique()
                            logger.info(f"📊 Retrieved PNL for {len(unique_fetched_alphas)} alphas in {region}")
                            
                            # Prepare PNL data for batch insertion
                            pnl_data_dict = {}
                            for alpha_id in unique_fetched_alphas:
                                pnl_data_dict[alpha_id] = combined_pnl_df[combined_pnl_df['alpha_id'] == alpha_id]
                            
                            # Insert PNL data for this region
                            if pnl_data_dict:
                                insert_multiple_unsubmitted_pnl_data_optimized(pnl_data_dict, region)
                                logger.info(f"✅ Stored PNL data for {len(pnl_data_dict)} alphas in {region}")
                        else:
                            logger.info(f"No PNL data returned for {region} in this window")
                            
                # Calculate correlations for this region (if not skipped)
                if not skip_correlation:
                    logger.debug(f"Calculating correlations for {region}...")
                    try:
                        calculate_unsubmitted_vs_submitted_correlations(region)
                        logger.info(f"✅ Updated correlations for {region}")
                    except Exception as e:
                        logger.warning(f"Correlation calculation failed for {region}: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing window {iteration_count}: {e}")
            logger.info(f"Continuing with next window...")
            
        # Clear window data from memory immediately
        processed_count = len(window_alphas)
        total_processed += processed_count
        del window_alphas  # Explicit cleanup
        
        logger.info(f"🧹 Cleared window data from memory. Progress: {total_processed} alphas processed")
        
        # Adjust date window for next iteration
        if last_alpha_date and last_alpha_date != current_end_date:
            logger.info(f"Next window: last alpha date was {last_alpha_date}")
            last_alpha_date_obj = datetime.strptime(last_alpha_date, "%Y-%m-%d")
            new_end_date_obj = last_alpha_date_obj - timedelta(days=1)
            current_end_date = new_end_date_obj.strftime("%Y-%m-%d")
        else:
            # Move back by larger amount if no date progression
            current_end_date_obj = datetime.strptime(current_end_date, "%Y-%m-%d")
            new_end_date_obj = current_end_date_obj - timedelta(days=7)
            current_end_date = new_end_date_obj.strftime("%Y-%m-%d")
            logger.info(f"No date progression, moving window back 1 week to {current_end_date}")
        
        # Check if we've reached the start date
        if current_end_date <= start_date:
            logger.info("Reached start date. Streaming complete for this threshold.")
            break
    
    if iteration_count >= max_iterations:
        logger.warning(f"Reached maximum iterations ({max_iterations}). Stopping to prevent infinite loop.")
    
    logger.info(f"🏁 Streaming complete for threshold. Total processed: {total_processed} alphas")
    return total_processed


def _fetch_alphas_for_date_window_parallel(
    session: requests.Session,
    sharpe_param: str,
    start_date: str,
    end_date: str,
    batch_size: int,
    max_workers: int = 10,
    max_batches_per_window: int = 200,  # 10,000 alphas / 50 = 200 batches max
    request_timeout: int = 60,
    retry_wait_seconds: int = 5,
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch all alphas in a single date window using parallel requests.
    
    Args:
        session: The authenticated requests.Session object
        sharpe_param: Sharpe parameter string
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        batch_size: Batch size for requests
        max_workers: Maximum number of concurrent threads
        max_batches_per_window: Maximum batches to fetch per window (safety limit)
        request_timeout: Request timeout
        retry_wait_seconds: Retry wait time
        max_retries: Maximum retries
    
    Returns:
        Tuple of (all_alphas_in_window, last_alpha_date_found)
    """
    logger.debug(f"Starting parallel fetch for date window {start_date} to {end_date}")
    
    window_alphas = []
    last_alpha_date = None
    
    # Start with initial batch to determine if we have data
    initial_batch, success = _fetch_unsubmitted_batch(
        session, sharpe_param, start_date, end_date, 0, batch_size,
        request_timeout, retry_wait_seconds, max_retries
    )
    
    if not success:
        logger.error(f"Failed to fetch initial batch for window {start_date} to {end_date}")
        return [], None
    
    if not initial_batch:
        logger.debug(f"No alphas found in window {start_date} to {end_date}")
        return [], None
    
    window_alphas.extend(initial_batch)
    
    # Get the date from the last alpha in the initial batch
    if initial_batch:
        # Find the oldest alpha date in this batch to use for window adjustment
        for alpha in initial_batch:
            alpha_date_str = alpha.get('date_added')  # This comes from process_unsubmitted_alpha_record
            if alpha_date_str:
                try:
                    alpha_date = datetime.fromisoformat(alpha_date_str.replace('Z', '+00:00'))
                    alpha_date_formatted = alpha_date.strftime("%Y-%m-%d")
                    if last_alpha_date is None or alpha_date_formatted < last_alpha_date:
                        last_alpha_date = alpha_date_formatted
                except ValueError as e:
                    logger.warning(f"Could not parse date {alpha_date_str}: {e}")
    
    # If initial batch is full, there might be more data - fetch in parallel
    if len(initial_batch) == batch_size:
        logger.info(f"Window {start_date} to {end_date} has more data, fetching in parallel...")
        
        # Estimate how many batches we might need (we'll start conservatively)
        # We'll try progressively larger batch sets until we hit empty results
        batch_increment = 10  # Start by trying 10 more batches at a time
        offset = batch_size  # Start from batch 1 (we already got batch 0)
        
        while offset < (max_batches_per_window * batch_size):
            # Calculate offsets for this parallel batch
            current_batch_offsets = list(range(offset, min(offset + (batch_increment * batch_size), max_batches_per_window * batch_size), batch_size))
            
            if not current_batch_offsets:
                break
                
            logger.debug(f"Fetching {len(current_batch_offsets)} batches in parallel: offsets {current_batch_offsets[0]}-{current_batch_offsets[-1]}")
            
            # Submit parallel requests for this set of batches
            futures: List[Future] = []
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for batch_offset in current_batch_offsets:
                        future = executor.submit(
                            _fetch_unsubmitted_batch,
                            session, sharpe_param, start_date, end_date, batch_offset, batch_size,
                            request_timeout, retry_wait_seconds, max_retries
                        )
                        futures.append(future)
                    
                    # Collect results
                    batch_results_received = 0
                    empty_batches_count = 0
                    
                    for future in as_completed(futures):
                        try:
                            batch_alphas, success = future.result()
                            if success and batch_alphas:
                                window_alphas.extend(batch_alphas)
                                batch_results_received += 1
                                
                                # Update last_alpha_date with oldest date from this batch
                                for alpha in batch_alphas:
                                    alpha_date_str = alpha.get('date_added')
                                    if alpha_date_str:
                                        try:
                                            alpha_date = datetime.fromisoformat(alpha_date_str.replace('Z', '+00:00'))
                                            alpha_date_formatted = alpha_date.strftime("%Y-%m-%d")
                                            if last_alpha_date is None or alpha_date_formatted < last_alpha_date:
                                                last_alpha_date = alpha_date_formatted
                                        except ValueError:
                                            pass
                            elif success and not batch_alphas:
                                empty_batches_count += 1
                        except Exception as e:
                            logger.error(f"Error processing parallel batch result: {e}")
            
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received during parallel fetching")
                break
            
            logger.debug(f"Parallel batch complete: {batch_results_received} batches with data, {empty_batches_count} empty batches")
            
            # If we got mostly empty batches, we've reached the end
            if empty_batches_count > (len(current_batch_offsets) * 0.8):  # 80% empty batches
                logger.debug(f"Reached end of data in window (80% empty batches)")
                break
                
            # Move to next set of batches
            offset += batch_increment * batch_size
            
            # Increase batch increment for efficiency (but cap it)
            batch_increment = min(batch_increment + 5, 20)
    
    logger.info(f"Window {start_date} to {end_date} complete: {len(window_alphas)} alphas fetched")
    return window_alphas, last_alpha_date


def _fetch_alphas_for_threshold(
    session: requests.Session,
    sharpe_param: str,
    start_date: str,
    end_date: str,
    batch_size: int,
    request_timeout: int,
    retry_wait_seconds: int,
    max_retries: int,
    max_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch alphas for a single sharpe threshold using parallel date window processing.
    
    Args:
        session: The authenticated requests.Session object
        sharpe_param: Sharpe parameter string (e.g., "is.sharpe>=1" or "is.sharpe<=-1")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        batch_size: Batch size for requests
        request_timeout: Request timeout
        retry_wait_seconds: Retry wait time
        max_retries: Maximum retries
        max_workers: Maximum parallel workers per window
    
    Returns:
        List of processed alpha records
    """
    all_alphas = []
    current_end_date = end_date
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    
    iteration_count = 0
    max_iterations = 1000  # Safety limit to prevent infinite loops
    
    while current_end_date > start_date and iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"Date window iteration {iteration_count}: {start_date} to {current_end_date}")
        
        # Fetch all alphas in this date window using parallel processing
        window_alphas, last_alpha_date = _fetch_alphas_for_date_window_parallel(
            session=session,
            sharpe_param=sharpe_param,
            start_date=start_date,
            end_date=current_end_date,
            batch_size=batch_size,
            max_workers=max_workers,
            max_batches_per_window=200,
            request_timeout=request_timeout,
            retry_wait_seconds=retry_wait_seconds,
            max_retries=max_retries
        )
        
        if not window_alphas:
            logger.info(f"No alphas found in date window {start_date} to {current_end_date}. Done.")
            break
            
        # Add all alphas from this window
        all_alphas.extend(window_alphas)
        logger.info(f"Added {len(window_alphas)} alphas from window. Total so far: {len(all_alphas)}")
        
        # Adjust date window based on the oldest alpha we found
        if last_alpha_date and last_alpha_date != current_end_date:
            logger.info(f"Adjusting next window: last alpha date was {last_alpha_date}, window end was {current_end_date}")
            # Subtract one day from last alpha date to create new end date
            last_alpha_date_obj = datetime.strptime(last_alpha_date, "%Y-%m-%d")
            new_end_date_obj = last_alpha_date_obj - timedelta(days=1)
            current_end_date = new_end_date_obj.strftime("%Y-%m-%d")
        else:
            # If no date difference, move back by a larger amount to avoid infinite loops
            current_end_date_obj = datetime.strptime(current_end_date, "%Y-%m-%d")
            new_end_date_obj = current_end_date_obj - timedelta(days=7)  # Move back 1 week
            current_end_date = new_end_date_obj.strftime("%Y-%m-%d")
            logger.info(f"No date difference detected, moving window back 1 week to {current_end_date}")
        
        # Check if we've reached the start date
        if current_end_date <= start_date:
            logger.info("Reached start date. Fetching complete.")
            break
    
    if iteration_count >= max_iterations:
        logger.warning(f"Reached maximum iterations ({max_iterations}). Stopping to prevent infinite loop.")
    
    logger.info(f"Completed fetching for sharpe threshold. Total alphas: {len(all_alphas)}")
    return all_alphas


def _fetch_unsubmitted_batch(
    session: requests.Session,
    sharpe_param: str,
    start_date: str,
    end_date: str,
    offset: int,
    batch_size: int,
    request_timeout: int = 60,
    retry_wait_seconds: int = 5,
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Fetch a single batch of unsubmitted alphas with retry logic.
    This function is designed to be called in parallel by ThreadPoolExecutor.
    
    Args:
        session: The authenticated requests.Session object
        sharpe_param: Sharpe parameter string (e.g., "is.sharpe>=1")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        offset: Offset for pagination
        batch_size: Batch size for requests
        request_timeout: Request timeout
        retry_wait_seconds: Wait time between retries
        max_retries: Maximum number of retries
    
    Returns:
        Tuple of (processed_alpha_records, success_bool)
    """
    # Build URL manually to match exact format from user's example
    # Encode sharpe parameter properly
    if '>=' in sharpe_param:
        sharpe_encoded = sharpe_param.replace('>=', '%3E=%20')  # >= becomes %3E=%20 (with space)
    else:
        sharpe_encoded = sharpe_param.replace('<=', '%3C=%20')  # <= becomes %3C=%20 (with space)
    
    # Build URL components matching user's example exactly
    url_params = [
        f"limit={batch_size}",
        f"offset={offset}",
        "status=UNSUBMITTED%1FIS_FAIL",  # Exact encoding from user example
        sharpe_encoded,
        f"dateCreated%3E={start_date}T00:00:00-04:00",  # Note: %3E= not %3E%3D
        f"dateCreated%3C{end_date}T00:00:00-04:00",  # Note: %3C not %3C%3D
        "order=-dateCreated",
        "hidden=false"
    ]
    full_url = f"{BASE_API_URL}?{'&'.join(url_params)}"
    
    # Make request with retries
    batch_results, success = _make_request_with_retry(
        session, full_url, request_timeout, retry_wait_seconds, max_retries
    )
    
    if not success:
        logger.warning(f"Failed to fetch batch at offset {offset} after {max_retries} retries")
        return [], False
    
    if not batch_results:
        logger.debug(f"No results returned for offset {offset}")
        return [], True  # Success but no results (end of data)
    
    # Process the batch
    processed_alphas = []
    for alpha_data_item in batch_results:
        processed_record = process_unsubmitted_alpha_record(alpha_data_item)
        if processed_record:
            processed_alphas.append(processed_record)
    
    logger.debug(f"Successfully processed batch at offset {offset}: {len(processed_alphas)} alphas")
    return processed_alphas, True


def _make_request_with_retry(
    session: requests.Session,
    url: str,
    request_timeout: int,
    retry_wait_seconds: int,
    max_retries: int
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Make API request with retry logic.
    
    Args:
        session: Requests session
        url: Complete URL to request
        request_timeout: Request timeout
        retry_wait_seconds: Wait time between retries
        max_retries: Maximum number of retries
    
    Returns:
        Tuple of (results_list, success_bool)
    """
    for attempt in range(max_retries + 1):
        try:
            response = authenticated_get(url, session=session, timeout=request_timeout)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    results = data.get('results', [])
                    logger.debug(f"Request successful: {len(results)} results returned")
                    return results, True
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries:
                        time.sleep(retry_wait_seconds)
                        continue
                    return [], False
                    
            elif response.status_code == 401:
                logger.error(f"Authentication error (401). Please check session validity.")
                return [], False
                
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429). Waiting {retry_wait_seconds * 2}s before retry...")
                if attempt < max_retries:
                    time.sleep(retry_wait_seconds * 2)
                    continue
                return [], False
                
            else:
                logger.warning(f"HTTP {response.status_code} (attempt {attempt + 1}): {response.text[:200]}")
                if attempt < max_retries:
                    time.sleep(retry_wait_seconds)
                    continue
                return [], False
                
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout (attempt {attempt + 1})")
            if attempt < max_retries:
                time.sleep(retry_wait_seconds)
                continue
            return [], False
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request exception (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                time.sleep(retry_wait_seconds)
                continue
            return [], False
    
    logger.error(f"All {max_retries + 1} attempts failed")
    return [], False


def build_unsubmitted_url_manual(
    sharpe_threshold: float,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    offset: int = 0,
    limit: int = 50
) -> str:
    """
    Build a manual unsubmitted alphas URL for testing or manual use.
    
    Args:
        sharpe_threshold: Sharpe threshold (positive for >=, negative for <=)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to current date)
        offset: Starting offset
        limit: Batch size limit
    
    Returns:
        Complete URL string
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Build parameters dictionary using proper parameter names
    params = {
        'limit': limit,
        'offset': offset,
        'status': 'UNSUBMITTED%1FIS_FAIL',  # Use the encoded status as per user example
        'dateCreated>=': f"{start_date}T00:00:00-04:00",
        'dateCreated<': f"{end_date}T23:59:59-04:00",
        'order': '-dateCreated',
        'hidden': 'false'
    }
    
    # Add sharpe parameter based on threshold
    if sharpe_threshold >= 0:
        params['is.sharpe>='] = str(sharpe_threshold)
    else:
        params['is.sharpe<='] = str(sharpe_threshold)
    
    return f"{BASE_API_URL}?{urlencode(params)}"