"""
Script to calculate cross-correlations between all alphas and find the highest correlation for each alpha.
"""
import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
from scipy import stats
from typing import List, Dict, Tuple, Optional
import requests
from itertools import combinations
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from api.auth import get_authenticated_session
from api.alpha_fetcher import get_robust_session, warm_up_api_connection
from config.database_config import REGIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_alpha_pnl_flexible(session: requests.Session, alpha_id: str) -> pd.DataFrame:
    """
    Fetch PNL data for a single alpha with flexible column handling.
    """
    brain_api_url = "https://api.worldquantbrain.com"
    max_retries = 5
    default_retry_seconds = 15
    
    for attempt in range(max_retries):
        try:
            result = session.get(
                f"{brain_api_url}/alphas/{alpha_id}/recordsets/pnl",
                timeout=(10, 60)
            )
            
            if "retry-after" in result.headers:
                retry_after_seconds = float(result.headers.get("retry-after", default_retry_seconds))
                logger.debug(f"API rate limit for {alpha_id}, waiting {retry_after_seconds}s")
                time.sleep(retry_after_seconds + 0.1)
                continue
                
            if result.status_code == 200:
                if not result.content:
                    logger.warning(f"Empty response for alpha {alpha_id}")
                    return pd.DataFrame()
                
                try:
                    pnl_data = result.json()
                    pnl_records = pnl_data.get("records")
                    
                    if not pnl_records:
                        logger.info(f"No PNL records for alpha {alpha_id}")
                        return pd.DataFrame()
                    
                    # Determine number of columns from first record
                    first_record = pnl_records[0]
                    if not isinstance(first_record, list):
                        logger.error(f"Unexpected record format for {alpha_id}: {first_record}")
                        return pd.DataFrame()
                    
                    num_cols = len(first_record)
                    
                    # Create appropriate column names based on number of columns
                    if num_cols == 2:
                        columns = ["date", "pnl"]
                    elif num_cols == 3:
                        columns = ["date", "pnl1", "pnl2"]
                    else:
                        logger.error(f"Unexpected column count for {alpha_id}: {num_cols}")
                        return pd.DataFrame()
                    
                    # Create DataFrame with correct columns
                    pnl_df = pd.DataFrame(pnl_records, columns=columns)
                    
                    # Convert date and set as index
                    pnl_df['date'] = pd.to_datetime(pnl_df['date'], format="%Y-%m-%d")
                    pnl_df = pnl_df.set_index('date').sort_index()
                    
                    # Add alpha_id for identification
                    pnl_df['alpha_id'] = alpha_id
                    
                    # For multiple PNL columns, select the best performing one
                    pnl_columns = [col for col in pnl_df.columns if col.startswith('pnl') and col != 'alpha_id']
                    if len(pnl_columns) > 1:
                        # Calculate total return for each PNL column and keep the best
                        best_col = None
                        best_total_return = -float('inf')
                        
                        for col in pnl_columns:
                            total_return = pnl_df[col].iloc[-1] - pnl_df[col].iloc[0]
                            if total_return > best_total_return:
                                best_total_return = total_return
                                best_col = col
                        
                        # Keep only the best PNL column and rename to 'pnl'
                        pnl_df = pnl_df[[best_col, 'alpha_id']].copy()
                        pnl_df = pnl_df.rename(columns={best_col: 'pnl'})
                        logger.debug(f"Alpha {alpha_id}: selected {best_col} (return: {best_total_return:.2f})")
                    
                    logger.info(f"✓ Alpha {alpha_id}: {len(pnl_df)} records")
                    return pnl_df
                    
                except Exception as e:
                    logger.error(f"Error parsing PNL data for {alpha_id}: {e}")
                    return pd.DataFrame()
            
            else:
                logger.warning(f"HTTP {result.status_code} for alpha {alpha_id}")
                if attempt < max_retries - 1:
                    wait_time = default_retry_seconds * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                else:
                    return pd.DataFrame()
                    
        except Exception as e:
            logger.error(f"Request error for alpha {alpha_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(default_retry_seconds * (attempt + 1))
                continue
            else:
                return pd.DataFrame()
    
    return pd.DataFrame()


def get_alpha_pnl_threaded_flexible(session: requests.Session, alpha_ids: List[str], max_workers: int = 200) -> Dict[str, pd.Series]:
    """
    Get PNL data for multiple alphas in parallel using flexible column handling.
    Combines the threading approach from get_alpha_pnl_threaded with flexible PNL parsing.

    Args:
        session: The authenticated requests.Session object
        alpha_ids: List of alpha IDs to fetch PNL for
        max_workers: Maximum number of worker threads to use

    Returns:
        Dictionary mapping alpha_id to PNL Series (only successful fetches)
    """
    results = {}
    alpha_ids_list = list(alpha_ids)

    if not alpha_ids_list:
        logger.info("No alpha IDs provided to get_alpha_pnl_threaded_flexible.")
        return results
    
    # Use robust session and warm up API connection
    robust_session = get_robust_session()
    if hasattr(session, 'cookies') and session.cookies:
        robust_session.cookies = session.cookies
    if hasattr(session, 'headers') and session.headers:
        robust_session.headers.update(session.headers)
    
    warm_up_api_connection(robust_session)
    
    # Start with a smaller batch to warm up concurrency
    initial_batch_size = min(1, len(alpha_ids_list))
    initial_batch = alpha_ids_list[:initial_batch_size]
    remaining = alpha_ids_list[initial_batch_size:]
    
    # Process initial batch with a delay between requests
    for alpha_id in initial_batch:
        try:
            pnl_df = fetch_alpha_pnl_flexible(robust_session, alpha_id)
            if not pnl_df.empty and 'pnl' in pnl_df.columns:
                results[alpha_id] = pnl_df['pnl']
            else:
                logger.warning(f"No PNL data returned for alpha {alpha_id} in initial batch")
            time.sleep(0.5)
        except Exception as exc:
            logger.error(f"Error fetching PNL for alpha {alpha_id} in initial batch: {exc}")
    
    if not remaining:
        return results
    
    # Use a semaphore to control the rate of concurrent requests
    semaphore = threading.Semaphore(50)  # Allow 50 concurrent requests even with 200 workers
    
    def fetch_with_throttle(alpha_id):
        with semaphore:
            try:
                pnl_df = fetch_alpha_pnl_flexible(robust_session, alpha_id)
                if not pnl_df.empty and 'pnl' in pnl_df.columns:
                    return alpha_id, pnl_df['pnl']
                else:
                    logger.warning(f"No PNL data returned for alpha {alpha_id}")
                    return alpha_id, None
            except Exception as exc:
                logger.error(f"Error fetching PNL for alpha {alpha_id}: {exc}")
                return alpha_id, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_with_throttle, alpha_id) for alpha_id in remaining]
        for future in as_completed(futures):
            try:
                alpha_id, pnl_series = future.result()
                if pnl_series is not None:
                    results[alpha_id] = pnl_series
            except Exception as exc:
                logger.error(f"Unexpected error in thread pool executor: {exc}")
    
    # Summarize results
    successful = len(results)
    failed = len(alpha_ids_list) - successful
    logger.info(f"Threaded PNL fetch complete: {successful} successful, {failed} failed")
    
    return results


def calculate_returns_correlation(pnl_data1: pd.Series, pnl_data2: pd.Series, initial_value: float = 10_000_000) -> Optional[float]:
    """
    Calculate correlation between returns of two alphas.
    
    Args:
        pnl_data1: Cumulative PNL series for first alpha
        pnl_data2: Cumulative PNL series for second alpha
        initial_value: Initial portfolio value
        
    Returns:
        Correlation coefficient or None
    """
    try:
        # Find common dates
        common_dates = pnl_data1.index.intersection(pnl_data2.index)
        if len(common_dates) < 20:
            return None
        
        # Get PNL values for common dates
        pnl1_common = pnl_data1.loc[common_dates].values
        pnl2_common = pnl_data2.loc[common_dates].values
        
        # Calculate daily PNL (first differences)
        daily_pnl1 = np.zeros(len(pnl1_common))
        daily_pnl2 = np.zeros(len(pnl2_common))
        
        daily_pnl1[0] = pnl1_common[0]
        daily_pnl2[0] = pnl2_common[0]
        
        for i in range(1, len(pnl1_common)):
            daily_pnl1[i] = pnl1_common[i] - pnl1_common[i-1]
            daily_pnl2[i] = pnl2_common[i] - pnl2_common[i-1]
        
        # Calculate portfolio values
        port_val1 = np.zeros(len(daily_pnl1))
        port_val2 = np.zeros(len(daily_pnl2))
        
        port_val1[0] = initial_value + daily_pnl1[0]
        port_val2[0] = initial_value + daily_pnl2[0]
        
        for i in range(1, len(daily_pnl1)):
            port_val1[i] = port_val1[i-1] + daily_pnl1[i]
            port_val2[i] = port_val2[i-1] + daily_pnl2[i]
        
        # Calculate returns
        returns1 = np.zeros(len(daily_pnl1))
        returns2 = np.zeros(len(daily_pnl2))
        
        returns1[0] = np.nan
        returns2[0] = np.nan
        
        for i in range(1, len(daily_pnl1)):
            returns1[i] = daily_pnl1[i] / port_val1[i-1] if port_val1[i-1] != 0 else np.nan
            returns2[i] = daily_pnl2[i] / port_val2[i-1] if port_val2[i-1] != 0 else np.nan
        
        # Filter valid returns
        valid_mask = ~(np.isnan(returns1) | np.isnan(returns2) | 
                       np.isinf(returns1) | np.isinf(returns2))
        
        returns1_clean = returns1[valid_mask]
        returns2_clean = returns2[valid_mask]
        
        if len(returns1_clean) < 20:
            return None
        
        # Calculate correlation
        corr, _ = stats.pearsonr(returns1_clean, returns2_clean)
        
        if np.isnan(corr):
            return None
            
        return corr
        
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return None


def process_csv_file(csv_file_path: str, region: str, reference_alpha_id: Optional[str] = None) -> None:
    """
    Process CSV file and add correlations for each alpha.
    If reference_alpha_id is provided, calculates correlation with that alpha.
    Otherwise, calculates highest cross-correlation with other alphas in the CSV.
    """
    try:
        # Read CSV
        logger.info(f"Reading CSV file: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        
        if 'alpha_id' not in df.columns:
            logger.error("CSV file must contain 'alpha_id' column")
            return
        
        alpha_ids = df['alpha_id'].unique().tolist()
        logger.info(f"Found {len(alpha_ids)} unique alpha IDs in CSV")
        
        # Add reference alpha if provided
        alphas_to_fetch = alpha_ids.copy()
        if reference_alpha_id:
            if reference_alpha_id not in alpha_ids:
                alphas_to_fetch.append(reference_alpha_id)
                logger.info(f"Added reference alpha {reference_alpha_id} to fetch list")
            else:
                logger.info(f"Reference alpha {reference_alpha_id} is already in CSV")
        
        # Authenticate
        logger.info("Authenticating with API...")
        session = get_authenticated_session()
        if not session:
            logger.error("Failed to authenticate")
            return
        
        # STEP 1: Fetch all PNL data using concurrent approach
        if reference_alpha_id:
            logger.info(f"Fetching PNL data concurrently for {len(alpha_ids)} CSV alphas + reference alpha {reference_alpha_id}...")
        else:
            logger.info(f"Fetching PNL data concurrently for all {len(alpha_ids)} alphas...")
        
        all_pnl_data = get_alpha_pnl_threaded_flexible(session, alphas_to_fetch)
        successful_alphas = list(all_pnl_data.keys())
        
        logger.info(f"Successfully fetched PNL for {len(successful_alphas)}/{len(alphas_to_fetch)} alphas")
        
        # Validate reference alpha was fetched if provided
        if reference_alpha_id and reference_alpha_id not in all_pnl_data:
            logger.error(f"Failed to fetch PNL data for reference alpha {reference_alpha_id}")
            return
        
        if len(successful_alphas) < 2:
            logger.error("Need at least 2 alphas with PNL data to calculate correlations")
            return
        
        # STEP 2: Calculate correlations
        if reference_alpha_id:
            # Reference alpha mode: calculate correlations with reference alpha only
            logger.info(f"Calculating correlations with reference alpha {reference_alpha_id}...")
            
            # Store correlation with reference alpha for each CSV alpha
            correlations = {alpha_id: 0.0 for alpha_id in alpha_ids}
            
            csv_alphas_with_data = [alpha_id for alpha_id in alpha_ids if alpha_id in all_pnl_data]
            total_pairs = len(csv_alphas_with_data)
            processed_pairs = 0
            
            for alpha_id in csv_alphas_with_data:
                processed_pairs += 1
                
                if processed_pairs % 10 == 0:
                    logger.info(f"Correlation progress: {processed_pairs}/{total_pairs} ({processed_pairs/total_pairs*100:.1f}%)")
                
                try:
                    corr = calculate_returns_correlation(
                        all_pnl_data[alpha_id], 
                        all_pnl_data[reference_alpha_id]
                    )
                    
                    if corr is not None:
                        correlations[alpha_id] = corr
                        logger.debug(f"Alpha {alpha_id} vs {reference_alpha_id}: {corr:.4f}")
                    else:
                        correlations[alpha_id] = 0.0
                
                except Exception as e:
                    logger.error(f"Error calculating correlation between {alpha_id} and {reference_alpha_id}: {e}")
                    correlations[alpha_id] = 0.0
            
            max_correlations = correlations
            
        else:
            # Cross-correlation mode: calculate all pairwise correlations
            logger.info("Calculating all pairwise correlations...")
            
            # Store the highest correlation for each alpha
            max_correlations = {alpha_id: 0.0 for alpha_id in alpha_ids}
            
            total_pairs = len(successful_alphas) * (len(successful_alphas) - 1) // 2
            processed_pairs = 0
            
            for i, alpha1 in enumerate(successful_alphas):
                for j, alpha2 in enumerate(successful_alphas[i+1:], i+1):
                    processed_pairs += 1
                    
                    if processed_pairs % 100 == 0:
                        logger.info(f"Correlation progress: {processed_pairs}/{total_pairs} ({processed_pairs/total_pairs*100:.1f}%)")
                    
                    try:
                        corr = calculate_returns_correlation(
                            all_pnl_data[alpha1], 
                            all_pnl_data[alpha2]
                        )
                        
                        if corr is not None and corr > 0:
                            # Update max correlation for both alphas if this is higher
                            if corr > max_correlations[alpha1]:
                                max_correlations[alpha1] = corr
                                logger.debug(f"New max for {alpha1}: {corr:.4f} (vs {alpha2})")
                            
                            if corr > max_correlations[alpha2]:
                                max_correlations[alpha2] = corr
                                logger.debug(f"New max for {alpha2}: {corr:.4f} (vs {alpha1})")
                    
                    except Exception as e:
                        logger.error(f"Error calculating correlation between {alpha1} and {alpha2}: {e}")
        
        # STEP 3: Update DataFrame with results
        df['self_corr'] = df['alpha_id'].map(max_correlations)
        
        # Replace None values with NaN
        df['self_corr'] = df['self_corr'].fillna(np.nan)
        
        # Save results
        output_path = csv_file_path.replace('.csv', '_with_self_corr.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to: {output_path}")
        
        # Summary statistics
        valid_corrs = [v for v in max_correlations.values() if v != 0.0]
        logger.info(f"\nSummary:")
        logger.info(f"  - Alphas with valid correlations: {len(valid_corrs)}/{len(alpha_ids)}")
        
        if reference_alpha_id:
            logger.info(f"  - Correlations calculated with reference alpha: {reference_alpha_id}")
            logger.info(f"  - Total correlations calculated: {processed_pairs}")
        else:
            logger.info(f"  - Total pairs calculated: {processed_pairs}")
        
        if valid_corrs:
            logger.info(f"  - Correlation stats:")
            logger.info(f"    Min: {min(valid_corrs):.4f}")
            logger.info(f"    Max: {max(valid_corrs):.4f}")
            logger.info(f"    Mean: {np.mean(valid_corrs):.4f}")
            logger.info(f"    Median: {np.median(valid_corrs):.4f}")
            
            if reference_alpha_id:
                pos_corrs = [v for v in valid_corrs if v > 0]
                neg_corrs = [v for v in valid_corrs if v < 0]
                logger.info(f"    Positive correlations: {len(pos_corrs)}")
                logger.info(f"    Negative correlations: {len(neg_corrs)}")
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Calculate cross-correlations between all alphas or correlations with a reference alpha'
    )
    parser.add_argument('--csv_file', required=True, 
                        help='Path to CSV file with alpha IDs')
    parser.add_argument('--region', required=True, choices=REGIONS,
                        help='Region for the alphas')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of alphas (for testing)')
    parser.add_argument('--reference_alpha_id', type=str, default=None,
                        help='Calculate correlations with this specific alpha instead of cross-correlations')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    logger.info("="*60)
    if args.reference_alpha_id:
        logger.info("Reference Alpha Correlation Calculator")
        logger.info("="*60)
        logger.info(f"CSV: {args.csv_file}")
        logger.info(f"Region: {args.region}")
        logger.info(f"Reference Alpha: {args.reference_alpha_id}")
        if args.limit:
            logger.info(f"Limit: {args.limit} alphas")
    else:
        logger.info("Cross-Correlation Calculator")
        logger.info("="*60)
        logger.info(f"CSV: {args.csv_file}")
        logger.info(f"Region: {args.region}")
        if args.limit:
            logger.info(f"Limit: {args.limit} alphas")
    
    start_time = time.time()
    
    try:
        # Handle limit if specified
        csv_path = args.csv_file
        if args.limit:
            df_full = pd.read_csv(args.csv_file)
            df_limited = df_full.head(args.limit)
            csv_path = args.csv_file.replace('.csv', f'_temp_limit_{args.limit}.csv')
            df_limited.to_csv(csv_path, index=False)
        
        process_csv_file(csv_path, args.region, args.reference_alpha_id)
        
        # Clean up temp file
        if args.limit and csv_path != args.csv_file:
            os.remove(csv_path)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Completed successfully in {elapsed:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()