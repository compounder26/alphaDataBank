"""
Script to calculate correlations for unsubmitted alphas against submitted alphas only.
For each unsubmitted alpha, calculates correlation with all submitted alphas and stores the maximum.
"""
import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.helpers import setup_logging
from database.operations import get_all_alpha_ids_by_region_basic, get_pnl_data_for_alphas
from database.operations_unsubmitted import (
    get_all_unsubmitted_alpha_ids_by_region, 
    get_unsubmitted_pnl_data_for_alphas,
    update_multiple_unsubmitted_alpha_self_correlations
)
from database.schema import get_connection, get_region_id
from config.database_config import REGIONS
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Import the correlation calculation function
try:
    import correlation_utils
    calculate_alpha_correlation_fast = correlation_utils.calculate_alpha_correlation_fast
except ImportError:
    logging.warning("Cython extension 'correlation_utils' not found. Using Python fallback.")
    from scripts.update_correlations_optimized import calculate_alpha_correlation_fast

def calculate_unsubmitted_vs_submitted_correlations(region: str) -> None:
    """
    Calculate correlations between unsubmitted alphas and submitted alphas in a region.
    For each unsubmitted alpha, finds the maximum correlation with any submitted alpha.
    
    Args:
        region: Region name
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting unsubmitted vs submitted correlation calculation for region {region}...")
        
        # Step 1: Get all submitted alpha IDs (REGULAR + SUPER) for the region
        submitted_alpha_ids = get_all_alpha_ids_by_region_basic(region)
        if not submitted_alpha_ids:
            logger.warning(f"No submitted alphas found for region {region}")
            return
        logger.info(f"Found {len(submitted_alpha_ids)} submitted alphas for region {region}")
        
        # Step 2: Get all unsubmitted alpha IDs for the region
        unsubmitted_alpha_ids = get_all_unsubmitted_alpha_ids_by_region(region)
        if not unsubmitted_alpha_ids:
            logger.warning(f"No unsubmitted alphas found for region {region}")
            return
        logger.info(f"Found {len(unsubmitted_alpha_ids)} unsubmitted alphas for region {region}")
        
        # Step 3: Load PNL data for all submitted alphas (keep in memory)
        logger.info(f"Loading PNL data for {len(submitted_alpha_ids)} submitted alphas...")
        submitted_pnl_data = get_pnl_data_for_alphas(submitted_alpha_ids, region)
        submitted_with_pnl = {aid: data for aid, data in submitted_pnl_data.items() 
                             if data.get('df') is not None and not data['df'].empty}
        
        if not submitted_with_pnl:
            logger.warning(f"No submitted alphas with PNL data found for region {region}")
            return
        logger.info(f"Loaded PNL data for {len(submitted_with_pnl)} submitted alphas with PNL data")
        
        # Step 4: Load PNL data for unsubmitted alphas
        logger.info(f"Loading PNL data for {len(unsubmitted_alpha_ids)} unsubmitted alphas...")
        unsubmitted_pnl_data = get_unsubmitted_pnl_data_for_alphas(unsubmitted_alpha_ids, region)
        unsubmitted_with_pnl = {aid: data for aid, data in unsubmitted_pnl_data.items() 
                               if data.get('df') is not None and not data['df'].empty}
        
        if not unsubmitted_with_pnl:
            logger.warning(f"No unsubmitted alphas with PNL data found for region {region}")
            return
        logger.info(f"Loaded PNL data for {len(unsubmitted_with_pnl)} unsubmitted alphas with PNL data")
        
        # Step 5: Calculate correlations using parallel processing
        logger.info("Calculating correlations between unsubmitted and submitted alphas...")
        correlation_results = calculate_correlations_parallel(
            unsubmitted_with_pnl, submitted_with_pnl, region
        )
        
        # Step 6: Store results in database
        if correlation_results:
            store_unsubmitted_correlation_results(correlation_results, region)
            logger.info(f"Successfully calculated and stored correlations for {len(correlation_results)} unsubmitted alphas")
            
            # Step 7: Update self_correlation in alphas_unsubmitted table
            update_multiple_unsubmitted_alpha_self_correlations(correlation_results, region)
            logger.info(f"Updated self_correlation for {len(correlation_results)} unsubmitted alphas in alphas_unsubmitted table")
        else:
            logger.warning(f"No correlation results to store for region {region}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed unsubmitted vs submitted correlation calculation for region {region} in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error calculating unsubmitted vs submitted correlations for region {region}: {e}")
        raise

def calculate_correlations_for_unsubmitted_alpha(unsubmitted_alpha_id: str, 
                                                unsubmitted_pnl_data: Dict[str, Any],
                                                submitted_pnl_data_dict: Dict[str, Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Calculate correlations for one unsubmitted alpha against all submitted alphas.
    
    Args:
        unsubmitted_alpha_id: ID of the unsubmitted alpha
        unsubmitted_pnl_data: PNL data for the unsubmitted alpha
        submitted_pnl_data_dict: Dictionary of all submitted alphas' PNL data
    
    Returns:
        Tuple of (alpha_id, correlation_result_dict or None)
    """
    try:
        # Validate unsubmitted alpha data
        if not isinstance(unsubmitted_pnl_data, dict) or 'df' not in unsubmitted_pnl_data:
            logger.warning(f"No dataframe key found for unsubmitted alpha {unsubmitted_alpha_id}")
            return unsubmitted_alpha_id, None
        
        unsubmitted_df = unsubmitted_pnl_data['df']
        if unsubmitted_df is None or not isinstance(unsubmitted_df, pd.DataFrame) or unsubmitted_df.empty:
            logger.warning(f"Invalid or empty dataframe for unsubmitted alpha {unsubmitted_alpha_id}")
            return unsubmitted_alpha_id, None
        
        # Apply 4-year time window filter
        try:
            end_date = unsubmitted_df.index.max()
            start_date = end_date - pd.DateOffset(years=4)
            
            windowed_df = unsubmitted_df
            if len(unsubmitted_df.loc[unsubmitted_df.index >= start_date]) >= 60:
                windowed_df = unsubmitted_df.loc[unsubmitted_df.index >= start_date]
        except Exception as e:
            logger.error(f"Error determining time window for unsubmitted alpha {unsubmitted_alpha_id}: {e}")
            return unsubmitted_alpha_id, None
        
        # Calculate correlations with all submitted alphas
        correlations = []
        best_correlated_alpha = None
        max_correlation = float('-inf')
        
        processed_count = 0
        for submitted_alpha_id, submitted_alpha_data in submitted_pnl_data_dict.items():
            # Validate submitted alpha data
            if not isinstance(submitted_alpha_data, dict) or 'df' not in submitted_alpha_data:
                continue
            
            submitted_df = submitted_alpha_data['df']
            if submitted_df is None or not isinstance(submitted_df, pd.DataFrame) or submitted_df.empty:
                continue
            
            processed_count += 1
            
            try:
                # Find common dates
                common_dates = windowed_df.index.intersection(submitted_df.index)
                if len(common_dates) < 20:  # Require at least 20 common dates
                    continue
                
                # Extract PNL values for common dates
                unsubmitted_pnl_values = windowed_df.loc[common_dates, 'pnl'].values
                submitted_pnl_values = submitted_df.loc[common_dates, 'pnl'].values
                
                # Calculate correlation
                corr = calculate_alpha_correlation_fast(unsubmitted_pnl_values, submitted_pnl_values)
                if corr is not None:
                    correlations.append(abs(corr))  # Use absolute correlation
                    if abs(corr) > max_correlation:
                        max_correlation = abs(corr)
                        best_correlated_alpha = submitted_alpha_id
                        
            except Exception as e:
                logger.debug(f"Error calculating correlation between {unsubmitted_alpha_id} and {submitted_alpha_id}: {e}")
                continue
        
        if not correlations:
            logger.warning(f"No valid correlations calculated for unsubmitted alpha {unsubmitted_alpha_id} (processed {processed_count} submitted alphas)")
            return unsubmitted_alpha_id, None
        
        result = {
            'max_correlation': max_correlation,
            'best_correlated_alpha': best_correlated_alpha,
            'total_correlations': len(correlations),
            'processed_submitted_count': processed_count
        }
        
        logger.debug(f"Calculated correlations for unsubmitted alpha {unsubmitted_alpha_id}: max={max_correlation:.4f} with {best_correlated_alpha}")
        return unsubmitted_alpha_id, result
        
    except Exception as e:
        logger.error(f"Unexpected error processing unsubmitted alpha {unsubmitted_alpha_id}: {e}")
        return unsubmitted_alpha_id, None

def calculate_correlations_parallel(unsubmitted_pnl_data: Dict[str, Dict[str, Any]], 
                                   submitted_pnl_data: Dict[str, Dict[str, Any]], 
                                   region: str, 
                                   max_workers: int = 8) -> Dict[str, Dict[str, Any]]:
    """
    Calculate correlations for all unsubmitted alphas in parallel.
    
    Args:
        unsubmitted_pnl_data: Dictionary of unsubmitted alphas' PNL data
        submitted_pnl_data: Dictionary of submitted alphas' PNL data
        region: Region name
        max_workers: Maximum number of worker threads
    
    Returns:
        Dictionary mapping unsubmitted alpha IDs to correlation results
    """
    results = {}
    
    logger.info(f"Starting parallel correlation calculation for {len(unsubmitted_pnl_data)} unsubmitted alphas against {len(submitted_pnl_data)} submitted alphas")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all correlation calculation tasks
            future_to_alpha = {
                executor.submit(
                    calculate_correlations_for_unsubmitted_alpha, 
                    alpha_id, 
                    pnl_data, 
                    submitted_pnl_data
                ): alpha_id for alpha_id, pnl_data in unsubmitted_pnl_data.items()
            }
            
            # Process completed futures
            for future in concurrent.futures.as_completed(future_to_alpha):
                alpha_id = future_to_alpha[future]
                try:
                    returned_alpha_id, result = future.result()
                    if result is not None:
                        results[returned_alpha_id] = result
                    
                    # Log progress periodically
                    if len(results) % 100 == 0:
                        logger.info(f"Processed {len(results)} unsubmitted alphas so far...")
                        
                except Exception as e:
                    logger.error(f"Error processing unsubmitted alpha {alpha_id}: {e}")
    
    except Exception as e:
        logger.error(f"Error in parallel correlation calculation: {e}")
        raise
    
    logger.info(f"Completed parallel correlation calculation: {len(results)} successful results out of {len(unsubmitted_pnl_data)} unsubmitted alphas")
    return results

def store_unsubmitted_correlation_results(correlation_results: Dict[str, Dict[str, Any]], region: str) -> None:
    """
    Store unsubmitted correlation results in the database.
    
    Args:
        correlation_results: Dictionary mapping alpha IDs to correlation results
        region: Region name
    """
    try:
        db_engine = get_connection()
        table_name = f"correlation_unsubmitted_{region.lower()}"
        
        with db_engine.connect() as connection:
            with connection.begin():
                for alpha_id, result in correlation_results.items():
                    upsert_query = f"""
                    INSERT INTO {table_name} (alpha_id, max_correlation_with_submitted, best_correlated_submitted_alpha)
                    VALUES (:alpha_id, :max_corr, :best_alpha)
                    ON CONFLICT (alpha_id) DO UPDATE SET
                        max_correlation_with_submitted = :max_corr,
                        best_correlated_submitted_alpha = :best_alpha,
                        last_updated = CURRENT_TIMESTAMP
                    """
                    stmt = text(upsert_query)
                    connection.execute(stmt, {
                        'alpha_id': alpha_id,
                        'max_corr': result['max_correlation'],
                        'best_alpha': result['best_correlated_alpha']
                    })
                
        logger.info(f"Stored correlation results for {len(correlation_results)} unsubmitted alphas in region {region}")
        
    except Exception as e:
        logger.error(f"Error storing unsubmitted correlation results for region {region}: {e}")
        raise

def main():
    """Main function to run unsubmitted correlation calculation."""
    parser = argparse.ArgumentParser(description='Calculate correlations for unsubmitted alphas vs submitted alphas')
    parser.add_argument('--region', choices=REGIONS, help='Region to process')
    parser.add_argument('--all', action='store_true', help='Process all configured regions')
    args = parser.parse_args()
    
    if not args.region and not args.all:
        parser.error('Either --region or --all must be specified')
    
    setup_logging()
    
    regions_to_process = REGIONS[:] if args.all else [args.region]
    
    total_start_time = time.time()
    
    for region in regions_to_process:
        logger.info(f"Processing unsubmitted correlations for region {region}...")
        
        try:
            calculate_unsubmitted_vs_submitted_correlations(region)
            logger.info(f"Successfully completed unsubmitted correlation calculation for region {region}")
        except Exception as e:
            logger.error(f"Error processing unsubmitted correlations for region {region}: {e}")
    
    total_elapsed_time = time.time() - total_start_time
    logger.info(f"Unsubmitted correlation calculation complete in {total_elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()