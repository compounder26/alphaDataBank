"""
Module for fetching alpha data from the WorldQuant Brain API.
"""
import requests
import pandas as pd
import time
import json
import logging
import socket
import threading
import os
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .auth import check_session_valid, authenticated_get, set_global_session
from config.api_config import ALPHAS_ENDPOINT, DEFAULT_ALPHA_FETCH_PARAMS, DEFAULT_REQUEST_LIMIT # Import new config

logger = logging.getLogger(__name__)

# Custom headers for API requests
DEFAULT_API_HEADERS = {
    "User-Agent": "AlphaDataBank/1.0",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache"
}

# Global variables for request rate limiting
_alpha_request_times = {}  # Track last request time per alpha
MIN_REQUEST_INTERVAL = 1.0  # Seconds between requests for the same alpha

# Brain API URL - define it once at module level
brain_api_url = "https://api.worldquantbrain.com"

def get_robust_session():
    """
    Create a session with optimized connection pooling for API interactions.
    This improves connection management when making many parallel requests.
    """
    # Configure retry strategy at the connection level
    retry_strategy = Retry(
        total=3,                   # Number of retries at the connection level
        backoff_factor=1,          # Each retry will wait 1, 2, 4... seconds
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
        allowed_methods=["GET"]    # Only retry GET requests
    )
    
    # Create adapter with connection pool settings optimized for high concurrency
    adapter = HTTPAdapter(
        pool_connections=5,        # Number of connection pools to maintain
        pool_maxsize=250,          # Support up to 250 connections per host
        max_retries=retry_strategy # Apply retry strategy
    )
    
    # Create session and mount adapter for https requests
    session = requests.Session()
    session.mount("https://", adapter)
    
    # Set TCP keepalive to prevent connection resets
    # Note: This might require additional system-level settings on Windows
    for protocol in session.adapters:
        if hasattr(session.adapters[protocol], 'poolmanager'):
            session.adapters[protocol].poolmanager.connection_pool_kw.update({
                'socket_options': [
                    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                ]
            })
    
    return session


def warm_up_api_connection(session):
    """
    Send a simple request to warm up the API connection before intensive operations.
    This helps establish the connection pool and can reduce initial failures.
    
    Args:
        session: The authenticated requests.Session object.
    """
    try:
        # Make a simple request to establish connection
        result = authenticated_get(
            f"{brain_api_url}/authentication", 
            session=session,
            headers=DEFAULT_API_HEADERS,
            timeout=5
        )
        # Small delay to let connections stabilize
        time.sleep(1)
    except Exception:
        pass

# Helper function for fetching a single batch of alphas with indefinite retries
def _fetch_alpha_batch(session: requests.Session, 
                       offset: int, 
                       limit: int, 
                       region: Optional[str], # For logging and determining request type
                       is_global_fetch: bool,
                       retry_wait_seconds: int = 15,
                       request_timeout: int = 20) -> List[Dict[str, Any]]:
    """
    Fetches a single batch of alpha data with indefinite retries for transient errors.

    Args:
        session: The authenticated requests.Session object.
        offset: The offset for the API request.
        limit: The limit (batch size) for the API request.
        region: The region string, used if not a global fetch.
        is_global_fetch: True if fetching globally, False for regional.
        retry_wait_seconds: Seconds to wait before retrying a failed request.
        request_timeout: Timeout for the HTTP request.

    Returns:
        List[Dict[str, Any]]: A list of alpha records for the batch, or empty list on unrecoverable error.
    """
    actual_request_url = ALPHAS_ENDPOINT
    current_params_for_get_call = None
    log_params_display_for_batch = None

    if is_global_fetch:
        query_components = [
            f"limit={limit}",
            f"offset={offset}",
            "status!=UNSUBMITTED%1FIS-FAIL",
            "order=-os.sharpe",
            "hidden=false"
        ]
        actual_request_url = f"{ALPHAS_ENDPOINT}?{'&'.join(query_components)}"
        log_params_display_for_batch = "[URL Manually Constructed for batch]"
    else: # Regional fetch
        regional_params_list = [
            ('limit', str(limit)),
            ('offset', str(offset)),
            ('status!', 'UNSUBMITTED|IS-FAIL'),
            ('order', '-dateCreated'), # Regional default order
            ('hidden', 'false')
        ]
        if region: # Should always be true if not global_fetch
            regional_params_list.append(('settings.region', region))
        current_params_for_get_call = regional_params_list
        log_params_display_for_batch = regional_params_list

    while True: # Indefinite retry loop
        try:
            logger.debug(f"Requesting batch: URL: {actual_request_url}, Params: {log_params_display_for_batch}, Offset: {offset}, Limit: {limit}")
            response = authenticated_get(actual_request_url, session=session, params=current_params_for_get_call, timeout=request_timeout)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.debug(f"Successfully fetched batch for offset {offset}. Got {len(data.get('results',[]))} raw items.")
                    raw_results = data.get('results', [])
                    processed_batch_results = []
                    for alpha_data_item in raw_results:
                        settings = alpha_data_item.get('settings') or {}
                        regular_info = alpha_data_item.get('regular') or {}
                        is_metrics = alpha_data_item.get('is') or {}
                        os_metrics = alpha_data_item.get('os') or {}
                        is_risk_neutralized_metrics = is_metrics.get('riskNeutralized', {})

                        processed_record = {
                            'alpha_id': alpha_data_item.get('id'),
                            'alpha_type': alpha_data_item.get('type'),
                            'code': regular_info.get('code'),
                            'description': regular_info.get('description'),
                            'date_added': alpha_data_item.get('dateCreated'),
                            'last_updated': alpha_data_item.get('dateModified'),
                            
                            'settings_region': settings.get('region'), # Retained prefix based on memory/consistency
                            'universe': settings.get('universe'),      # Key changed as per user's implied schema
                            'delay': settings.get('delay'),          # Key changed
                            'decay': settings.get('decay'),          # Key changed
                            'neutralization': settings.get('neutralization'), # Added field
                            'settings_truncation': settings.get('truncation'),
                            'settings_pasteurization': settings.get('pasteurization'),
                            'settings_unit_handling': settings.get('unitHandling'),
                            'settings_nan_handling': settings.get('nanHandling'),
                            'settings_language': settings.get('language'),
                            'settings_max_stock_weight': settings.get('maxStockWeight'),
                            'settings_max_group_weight': settings.get('maxGroupWeight'),
                            'settings_max_turnover': settings.get('maxTurnover'),

                            'is_sharpe': is_metrics.get('sharpe'),
                            'is_fitness': is_metrics.get('fitness'),
                            'is_returns': is_metrics.get('returns'),
                            'is_drawdown': is_metrics.get('drawdown'),
                            'self_correlation': is_metrics.get('selfCorrelation'), # Added selfCorrelation
                            'is_longcount': is_metrics.get('longCount'),    # Key changed
                            'is_shortcount': is_metrics.get('shortCount'),  # Key changed
                            'is_turnover': is_metrics.get('turnover'),
                            'is_margin': is_metrics.get('margin'),
                            'prod_correlation': is_metrics.get('prodCorrelation'), # ADDED
                            'rn_sharpe': is_risk_neutralized_metrics.get('sharpe'), # Key changed
                            'rn_fitness': is_risk_neutralized_metrics.get('fitness'),# Key changed

                            'os_sharpe': os_metrics.get('sharpe'),
                            'os_fitness': os_metrics.get('fitness'),
                            # Include other relevant fields from API if needed by downstream processes
                        }
                        processed_batch_results.append(processed_record)
                    return processed_batch_results
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for offset {offset}: {e}. Response text: {response.text[:200]}. Retrying in {retry_wait_seconds}s...")
            elif response.status_code in [401, 403]:
                logger.error(f"Authentication error ({response.status_code}) fetching batch at offset {offset}. This is critical. Stopping retries for this batch and returning empty. Please check credentials/session.")
                return [] # Critical error, stop retrying for this batch
            elif response.status_code == 429: # Rate limit
                logger.warning(f"Rate limit (429) hit for offset {offset}. Retrying after 60 seconds (override default wait)...")
                time.sleep(60) # Specific longer wait for rate limits
                continue # Skip default wait, go to next iteration
            else:
                logger.warning(f"API request for offset {offset} failed with status {response.status_code}: {response.text[:200]}. Retrying in {retry_wait_seconds}s...")

        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out for offset {offset}. Retrying in {retry_wait_seconds}s...")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error for offset {offset}: {e}. Retrying in {retry_wait_seconds}s...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Generic request exception for offset {offset}: {e}. Retrying in {retry_wait_seconds}s...")
        
        # If any exception occurred or status was not 200 (and not auth error/rate limit with specific handling)
        try:
            time.sleep(retry_wait_seconds)
        except KeyboardInterrupt:
            logger.info(f"Keyboard interrupt received during retry wait for offset {offset}. Propagating interrupt.")
            raise # Re-raise KeyboardInterrupt


def scrape_alphas_by_region(session: requests.Session, 
                            region: Optional[str] = None, 
                            max_workers: int = 10, 
                            batch_size: int = DEFAULT_REQUEST_LIMIT,
                            retry_wait_seconds: int = 15,
                            request_timeout: int = 20) -> List[Dict[str, Any]]:
    """
    Scrapes alpha data from the WorldQuant Brain API using multithreading.
    If region is specified, fetches for that region.
    If region is None, fetches globally.

    Args:
        session: The authenticated requests.Session object.
        region: The optional region to scrape alphas for (e.g., 'USA', 'EUR').
        max_workers: Maximum number of concurrent threads for fetching batches.
        batch_size: Number of alphas to fetch per API request.
        retry_wait_seconds: Seconds to wait before retrying a failed batch request (in _fetch_alpha_batch).
        request_timeout: Timeout for individual HTTP requests (in _fetch_alpha_batch).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing an alpha record.
    """
    if not check_session_valid(session):
        logger.error("Session is not valid. Please re-authenticate.")
        return []

    all_alphas: List[Dict[str, Any]] = []
    is_global_fetch = region is None

    # --- 1. Initial request to get total count ---    
    initial_request_target_url_for_count = ALPHAS_ENDPOINT
    initial_params_for_count_call = None
    log_initial_params_display = None

    if is_global_fetch:
        logger.info("Starting global alpha scraping (all regions)...")
        initial_query_components = [
            "limit=1", "offset=0", "status!=UNSUBMITTED%1FIS-FAIL", 
            "order=-os.sharpe", "hidden=false"
        ]
        initial_request_target_url_for_count = f"{ALPHAS_ENDPOINT}?{'&'.join(initial_query_components)}"
        log_initial_params_display = "[URL Manually Constructed for count]"
    else:
        logger.info(f"Starting alpha scraping for region {region}...")
        initial_regional_params_list_for_count = [
            ('limit', '1'), ('offset', '0'), ('status!', 'UNSUBMITTED|IS-FAIL'),
            ('order', '-dateCreated'), ('hidden', 'false')
        ]
        if region: # Should be true here
             initial_regional_params_list_for_count.append(('settings.region', region))
        initial_params_for_count_call = initial_regional_params_list_for_count
        log_initial_params_display = initial_regional_params_list_for_count

    total_alphas_count = 0
    try:
        logger.info(f"Fetching total alpha count. URL: {initial_request_target_url_for_count}, Params: {log_initial_params_display}")
        response = authenticated_get(initial_request_target_url_for_count, session=session, params=initial_params_for_count_call, timeout=request_timeout)
        response.raise_for_status() 
        data = response.json()
        total_alphas_count = data.get('count', 0)
        if total_alphas_count == 0:
            logger.info(f"No alphas found for '{region if region else 'Global'}'.")
            return []
        logger.info(f"Total alphas to fetch for '{region if region else 'Global'}': {total_alphas_count}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch initial alpha count for '{region if region else 'Global'}': {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse initial alpha count response: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}")
        return []
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received during initial count fetch. Stopping.")
        return []

    # --- 2. Prepare and execute threaded batch requests ---    
    offsets_to_fetch = list(range(0, total_alphas_count, batch_size))
    if not offsets_to_fetch and total_alphas_count > 0:
        # This case handles if total_alphas_count < batch_size, range would be empty
        # but we still need to fetch one batch from offset 0.
        offsets_to_fetch = [0]
    elif not offsets_to_fetch:
        logger.info("No offsets to fetch based on count and batch size.")
        return []
        
    logger.info(f"Preparing {len(offsets_to_fetch)} batch requests with up to {max_workers} workers for {total_alphas_count} alphas (batch size: {batch_size}).")

    futures: List[Future] = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for current_offset in offsets_to_fetch:
                future = executor.submit(_fetch_alpha_batch, 
                                         session, 
                                         current_offset, 
                                         batch_size, 
                                         region, 
                                         is_global_fetch,
                                         retry_wait_seconds,
                                         request_timeout)
                futures.append(future)
            
            for future_item in as_completed(futures):
                try:
                    batch_results = future_item.result() # This is List[Dict[str, Any]]
                    if batch_results:
                        all_alphas.extend(batch_results)
                        logger.debug(f"Processed batch. Total alphas collected so far: {len(all_alphas)}")
                    # If batch_results is empty, it means _fetch_alpha_batch handled its errors or got an empty list.
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt caught while processing a batch. Attempting to shutdown executor...")
                    executor.shutdown(wait=False, cancel_futures=True) # Python 3.9+ for cancel_futures
                    # On older Pythons, shutdown(wait=False) and threads might complete if not interruptible.
                    raise # Re-raise to stop further processing
                except Exception as e:
                    # This catches exceptions from _fetch_alpha_batch if they weren't handled internally (should not happen for request errors)
                    # or exceptions during future.result() itself.
                    logger.error(f"Exception processing a batch future: {e}", exc_info=True)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received during threaded alpha scraping. Finalizing... Note: Some threads may still be running if not interruptible.")
        # Executor is shut down by exiting the 'with' block. 
        # Forcibly trying to cancel futures if possible (Python 3.9+)
        # This part is tricky as futures might be in uninterruptible C code (like socket ops)
        for f in futures:
            if not f.done():
                f.cancel() # Attempt to cancel
    except Exception as e:
        logger.error(f"An unexpected error occurred during threaded execution setup or teardown: {e}", exc_info=True)
    
    logger.info(f"Finished scraping. Total alphas fetched: {len(all_alphas)} out of {total_alphas_count} expected.")
    if len(all_alphas) != total_alphas_count and total_alphas_count > 0:
        # This warning is more complex now as _fetch_alpha_batch might return [] for auth errors on a specific batch.
        # The count of successfully fetched items might naturally be less than total_alphas_count if some batches had such unrecoverable errors.
        logger.warning(f"Mismatch in fetched alphas ({len(all_alphas)}) vs expected ({total_alphas_count}). Some batches might have failed critically or returned empty.")
        
    return all_alphas


def get_alpha_pnl(session: requests.Session, alpha_id: str) -> pd.DataFrame:
    """
    Get alpha PNL data from the WorldQuant Brain API with robust retries.

    Args:
        session: The authenticated requests.Session object.
        alpha_id: The alpha ID to fetch PNL for.

    Returns:
        DataFrame containing PNL data with Date as index, or empty DataFrame on failure.
    """
    global _alpha_request_times
    brain_api_url = "https://api.worldquantbrain.com"
    max_retries = 5
    default_retry_seconds = 15  # Base retry time if not specified by API

    # Check if we need to wait before making this request (pre-emptive rate limiting)
    now = time.time()
    if alpha_id in _alpha_request_times:
        elapsed = now - _alpha_request_times[alpha_id]
        if elapsed < MIN_REQUEST_INTERVAL:
            # Wait to respect minimum interval
            wait_time = MIN_REQUEST_INTERVAL - elapsed
            logger.debug(f"Rate limiting: Waiting {wait_time:.2f}s before requesting alpha {alpha_id}")
            time.sleep(wait_time)
    
    # Update last request time
    _alpha_request_times[alpha_id] = time.time()

    for attempt in range(max_retries):
        try:
            logger.debug(f"Fetching PNL for alpha {alpha_id}, Attempt {attempt + 1}/{max_retries}")
            
            # Use proper headers for API request with increased timeout for potentially slow responses
            result = authenticated_get(
                f"{brain_api_url}/alphas/{alpha_id}/recordsets/pnl", 
                session=session,
                headers=DEFAULT_API_HEADERS,
                timeout=(10, 30)  # (connect timeout, read timeout)
            )

            if "retry-after" in result.headers:
                try:
                    retry_after_seconds = float(result.headers["retry-after"])
                    # Actually use the API's suggested retry time with a small buffer
                    logger.info(f"API advised retry after {retry_after_seconds}s for alpha {alpha_id}. Waiting for that duration (Attempt {attempt + 1})")
                    time.sleep(retry_after_seconds + 0.1)  # Add a small buffer
                    continue
                except (ValueError, TypeError):
                    retry_after_seconds = default_retry_seconds
                    logger.warning(f"Invalid retry-after header for alpha {alpha_id}: {result.headers['retry-after']}. Using default retry time.")
                    time.sleep(default_retry_seconds)
                    continue
                except ValueError:
                    logger.warning(f"Could not parse Retry-After header value: {result.headers['Retry-After']}. Using default backoff.")
                    # Fall through to status code based handling with default delay
            
            if result.status_code == 200:
                # Check for empty content, which might be a subtle rate limit or no data
                if not result.content:
                    logger.warning(f"Received 200 OK with empty content for alpha {alpha_id} (Attempt {attempt + 1}). Retrying in {default_retry_seconds}s...")
                    if attempt < max_retries - 1:
                        time.sleep(default_retry_seconds)
                        continue
                    else:
                        logger.error(f"Received 200 OK with empty content for alpha {alpha_id} after {max_retries} attempts. Assuming no PNL data.")
                        return pd.DataFrame()

                try:
                    pnl_data = result.json()
                    pnl_records = pnl_data.get("records")

                    if pnl_records is None:
                        logger.warning(f"No 'records' key in PNL JSON response for alpha {alpha_id}. Response: {pnl_data}")
                        return pd.DataFrame() # No data to process
                    if not pnl_records: # Empty list of records
                        logger.info(f"No PNL records found for alpha {alpha_id} (API returned empty list).")
                        return pd.DataFrame() # No data to process

                    # Robust dynamic PNL column handling
                    schema = pnl_data.get("schema", {})
                    schema_properties = schema.get("properties", [])
                    
                    # Validate data structure
                    if not isinstance(pnl_records[0], list) or len(pnl_records[0]) < 2:
                        logger.error(f"Unexpected PNL record structure for alpha {alpha_id}: {pnl_records[0]}. Expected list with at least 2 items.")
                        return pd.DataFrame()
                    
                    num_data_columns = len(pnl_records[0])
                    
                    # Build column names dynamically from schema if available
                    if schema_properties and len(schema_properties) >= num_data_columns:
                        # Use schema-based column names
                        column_names = [prop.get("name", f"col_{i}") for i, prop in enumerate(schema_properties[:num_data_columns])]
                        logger.debug(f"Using schema-based column names for alpha {alpha_id}: {column_names}")
                    else:
                        # Fallback to positional column names based on data structure
                        if num_data_columns == 2:
                            column_names = ["date", "pnl"]
                        elif num_data_columns == 3:
                            column_names = ["date", "pnl", "extra_col"]
                        elif num_data_columns == 4:
                            column_names = ["date", "pnl", "extra_col1", "extra_col2"]
                        else:
                            # Handle any number of columns
                            column_names = ["date", "pnl"] + [f"extra_col{i-1}" for i in range(2, num_data_columns)]
                        logger.debug(f"Using fallback column names for alpha {alpha_id}: {column_names} (data has {num_data_columns} columns)")
                    
                    # Create DataFrame with all columns
                    pnl_df = pd.DataFrame(pnl_records, columns=column_names)
                    
                    # Always extract date and the PNL column (2nd column, index 1)
                    # This works for all formats since PNL is always in position 1
                    pnl_column_name = column_names[1]  # This is our PNL column regardless of its exact name
                    pnl_df = pnl_df[["date", pnl_column_name]]
                    
                    # Rename the PNL column to standard name for consistency
                    if pnl_column_name != "pnl":
                        pnl_df = pnl_df.rename(columns={pnl_column_name: "pnl"})
                    
                    logger.debug(f"Processing alpha {alpha_id}: {num_data_columns} columns, using '{pnl_column_name}' as PNL")
                    
                    
                    pnl_df = pnl_df.assign(
                        alpha_id=alpha_id,
                        date=lambda x: pd.to_datetime(x.date, format="%Y-%m-%d")
                    ).set_index("date")
                    
                    logger.info(f"Retrieved PNL data for alpha {alpha_id}: {len(pnl_df)} records")
                    return pnl_df

                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error for alpha {alpha_id} (Attempt {attempt + 1}): {je}. Response text: {result.text[:200]}")
                    # Fall through to retry logic if not last attempt

            elif result.status_code == 429: # Too Many Requests
                logger.warning(f"Explicit rate limit (429) for alpha {alpha_id} (Attempt {attempt + 1}).")
                # Fall through to retry logic with backoff
            elif 500 <= result.status_code < 600: # Server-side errors
                logger.warning(f"Server error ({result.status_code}) for alpha {alpha_id} (Attempt {attempt + 1}). Response: {result.text[:200]}")
                # Fall through to retry logic with backoff
            else: # Other client errors or unexpected status codes
                logger.error(f"Failed to fetch PNL for alpha {alpha_id}. Status: {result.status_code}, Response: {result.text[:200]} (Attempt {attempt + 1})")
                if 400 <= result.status_code < 500 and result.status_code != 429: # Non-retryable client errors
                    logger.error(f"Client error {result.status_code} for alpha {alpha_id}. Not retrying.")
                    return pd.DataFrame()
                # For other errors, fall through to retry logic if not last attempt

            # Retry logic for errors not handled by 'Retry-After' or specific status checks above
            if attempt < max_retries - 1:
                wait_duration = default_retry_seconds * (attempt + 1) # Exponential backoff
                logger.info(f"Retrying PNL fetch for alpha {alpha_id} in {wait_duration}s...")
                time.sleep(wait_duration)
            else:
                logger.error(f"All {max_retries} retries failed for alpha {alpha_id}. Final status: {result.status_code if 'result' in locals() else 'N/A'}")
                return pd.DataFrame()

        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out fetching PNL for alpha {alpha_id} (Attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(default_retry_seconds * (attempt + 1))
            else:
                logger.error(f"All {max_retries} retries failed due to timeout for alpha {alpha_id}.")
                return pd.DataFrame()
        except requests.exceptions.RequestException as re:
            logger.error(f"Request exception fetching PNL for alpha {alpha_id} (Attempt {attempt + 1}): {re}")
            if attempt < max_retries - 1:
                # For general request exceptions, use a moderate backoff
                time.sleep(default_retry_seconds * (attempt + 1))
            else:
                logger.error(f"All {max_retries} retries failed due to request exceptions for alpha {alpha_id}.")
                return pd.DataFrame()

            # Should not be reached if loop logic is correct, but as a fallback:
            logger.error(f"Failed to fetch PNL for alpha {alpha_id} after all attempts and error handling.")
            return pd.DataFrame()


def get_alpha_pnl_threaded(session: requests.Session, alpha_ids: List[str], max_workers: int = 200) -> Tuple[pd.DataFrame, List[str]]:
    """
    Get PNL data for multiple alphas in parallel using a thread pool executor with gradual ramp-up.

    Args:
        session: The authenticated requests.Session object.
        alpha_ids: List of alpha IDs to fetch PNL for.
        max_workers: Maximum number of worker threads to use.

    Returns:
        Tuple of (combined_pnl_df, failed_alpha_ids) where:
        - combined_pnl_df is a DataFrame containing all PNL data
        - failed_alpha_ids is a list of alpha IDs that failed to fetch
    """
    results = {}
    alpha_ids_list = list(alpha_ids)  # Copy to list if it's not already

    if not alpha_ids_list:
        logger.info("No alpha IDs provided to get_alpha_pnl_threaded.")
        return results
    
    # Warm up the API connection first to establish connections
    warm_up_api_connection(session)
    
    # Start with a smaller batch to warm up concurrency
    initial_batch_size = min(1, len(alpha_ids_list))
    initial_batch = alpha_ids_list[:initial_batch_size]
    remaining = alpha_ids_list[initial_batch_size:]
    
    # Start with initial batch before full concurrency
    
    # Process initial batch with a delay between requests
    for alpha_id in initial_batch:
        try:
            results[alpha_id] = get_alpha_pnl(session, alpha_id)
            # Ensure records were fetched successfully
            if not results[alpha_id].empty:
                record_count = len(results[alpha_id])
                pass
            else:
                logger.warning(f"No PNL data returned for alpha {alpha_id} in initial batch")
            # Small delay between initial requests
            time.sleep(0.5)
        except Exception as exc:
            logger.error(f"Error fetching PNL for alpha {alpha_id} in initial batch: {exc}")
            results[alpha_id] = pd.DataFrame()
    
    if not remaining:
        # Process results the same way as the main path to ensure consistent return type
        successful = sum(1 for df in results.values() if df is not None and not df.empty)
        failed_alpha_ids = [alpha_id for alpha_id, df in results.items() if df is None or df.empty]
        
        # Convert the dictionary of DataFrames to a single combined DataFrame
        all_pnl_dfs = []
        for alpha_id, pnl_df in results.items():
            if not pnl_df.empty:
                # Ensure alpha_id column exists
                if 'alpha_id' not in pnl_df.columns:
                    pnl_df = pnl_df.assign(alpha_id=alpha_id)
                all_pnl_dfs.append(pnl_df)
        
        # Combine all DataFrames or return empty DataFrame if none were fetched
        if all_pnl_dfs:
            try:
                combined_pnl_df = pd.concat(all_pnl_dfs)
                return combined_pnl_df, failed_alpha_ids
            except Exception as e:
                logger.error(f"Error combining initial batch PNL DataFrames: {e}")
                return pd.DataFrame(), alpha_ids_list  # Return all as failed if concat fails
        else:
            # No PNL data fetched in initial batch
            return pd.DataFrame(), alpha_ids_list
    
    # Use a semaphore to control the rate of concurrent requests
    # Process remaining alphas with parallelism
    semaphore = threading.Semaphore(50)  # Allow 50 concurrent requests even with 200 workers
    
    def fetch_with_throttle(alpha_id):
        with semaphore:
            try:
                return alpha_id, get_alpha_pnl(session, alpha_id)
            except Exception as exc:
                logger.error(f"Error fetching PNL for alpha {alpha_id}: {exc}")
                return alpha_id, pd.DataFrame()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_with_throttle, alpha_id) for alpha_id in remaining]
        try:
            for future in as_completed(futures):
                try:
                    # Add timeout to make it interruptible and handle KeyboardInterrupt
                    alpha_id, pnl_df = future.result(timeout=30)
                    results[alpha_id] = pnl_df
                    # Log success with record count
                    if not pnl_df.empty:
                        record_count = len(pnl_df)
                        pass
                    else:
                        logger.warning(f"No PNL data returned for alpha {alpha_id}")
                except Exception as exc:
                    logger.error(f"Unexpected error in thread pool executor: {exc}")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Cancelling remaining PNL fetch operations...")
            # Cancel remaining futures
            for future in futures:
                future.cancel()
            # Shutdown executor quickly without waiting for remaining tasks
            executor.shutdown(wait=False)
            # Mark remaining alphas that weren't processed as failed
            processed_alphas = set(results.keys())
            remaining_alphas = [alpha_id for alpha_id in remaining if alpha_id not in processed_alphas]
            logger.info(f"Cancelled PNL fetching. Processed {len(processed_alphas)} out of {len(remaining)} remaining alphas.")
            # Continue with partial results - don't re-raise the KeyboardInterrupt here
    
    # Summarize results
    successful = sum(1 for df in results.values() if df is not None and not df.empty)
    failed_alpha_ids = [alpha_id for alpha_id, df in results.items() if df is None or df.empty]
    # Record success/failure stats
    
    # Convert the dictionary of DataFrames to a single combined DataFrame
    all_pnl_dfs = []
    for alpha_id, pnl_df in results.items():
        if not pnl_df.empty:
            # Ensure alpha_id column exists
            if 'alpha_id' not in pnl_df.columns:
                pnl_df = pnl_df.assign(alpha_id=alpha_id)
            all_pnl_dfs.append(pnl_df)
    
    # Combine all DataFrames or return empty DataFrame if none were fetched
    if all_pnl_dfs:
        try:
            combined_pnl_df = pd.concat(all_pnl_dfs)
            # All PNL data combined successfully
            return combined_pnl_df, failed_alpha_ids
        except Exception as e:
            logger.error(f"Error combining PNL DataFrames: {e}")
            return pd.DataFrame(), alpha_ids_list  # Return all as failed if concat fails
    else:
        # No PNL data fetched
        return pd.DataFrame(), alpha_ids_list
