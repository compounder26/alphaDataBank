"""
Optimized script to calculate and update correlation statistics for alphas.
Uses Cython acceleration and parallel processing for significant performance improvements.
"""
import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import concurrent.futures
import time

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import setup_logging

logger = logging.getLogger(__name__)

# Import the correlation_utils module (Cython extension)
try:
    import correlation_utils
except ImportError:
    logging.warning("Cython extension 'correlation_utils' not found. Will use Python implementation.")
    # Define a fallback function if Cython module isn't available
    def calculate_alpha_correlation_fast(alpha_pnl, other_pnl, initial_value=10000000.0):
        """
        Python fallback implementation of the Cython-accelerated correlation calculation.
        Uses the same formula but in pure Python.
        
        Args:
            alpha_pnl: Numpy array of cumulative PNL values for the first alpha
            other_pnl: Numpy array of cumulative PNL values for the second alpha
            initial_value: Initial portfolio value for return calculation (default: 10,000,000)
            
        Returns:
            Correlation coefficient or None if calculation fails or isn't meaningful
        """
        # Input validation
        if len(alpha_pnl) < 20 or len(other_pnl) < 20:
            return None
        
        try:
            # Calculate daily PNL (first differences of cumulative PNL)
            alpha_daily = np.zeros_like(alpha_pnl)
            other_daily = np.zeros_like(other_pnl)
            
            alpha_daily[0] = alpha_pnl[0]  # First day's PNL is just the cumulative value
            other_daily[0] = other_pnl[0]
            
            # Calculate differences for remaining days
            for i in range(1, len(alpha_pnl)):
                alpha_daily[i] = alpha_pnl[i] - alpha_pnl[i-1]
                
            for i in range(1, len(other_pnl)):
                other_daily[i] = other_pnl[i] - other_pnl[i-1]
            
            # Calculate portfolio values (initial value + cumulative daily PNL)
            alpha_port_val = initial_value + np.cumsum(alpha_daily)
            other_port_val = initial_value + np.cumsum(other_daily)
            
            # Calculate daily returns as daily PNL / previous day's portfolio value
            alpha_returns = np.zeros_like(alpha_daily)
            other_returns = np.zeros_like(other_daily)
            
            for i in range(1, len(alpha_daily)):
                alpha_returns[i] = alpha_daily[i] / alpha_port_val[i-1]
                
            for i in range(1, len(other_daily)):
                other_returns[i] = other_daily[i] / other_port_val[i-1]
            
            # Filter out invalid values (NaN, infinity)
            valid = ~(np.isnan(alpha_returns) | np.isnan(other_returns) | 
                       np.isinf(alpha_returns) | np.isinf(other_returns))
            
            alpha_returns_clean = alpha_returns[valid]
            other_returns_clean = other_returns[valid]
            
            # Check if we have enough valid data points for a meaningful correlation
            if len(alpha_returns_clean) < 20:
                return None
            
            # Calculate Pearson correlation coefficient
            from scipy import stats
            corr = stats.pearsonr(alpha_returns_clean, other_returns_clean)[0]
            
            if np.isnan(corr):
                return None
            
            return corr
        except Exception as e:
            logger.error(f"Error in correlation calculation: {e}")
            return None
else:
    # Use the Cython implementation if available
    calculate_alpha_correlation_fast = correlation_utils.calculate_alpha_correlation_fast

from sqlalchemy import text, create_engine, Connection  
from sqlalchemy.orm import sessionmaker
from correlation_utils import calculate_alpha_correlation_fast
from typing import List, Dict, Tuple

# Assuming get_connection is defined elsewhere and accessible, e.g., from database.operations
# If not, it needs to be defined or imported here as well.
# For this example, let's assume it's available via: 
from database.operations import get_connection

# Import data retrieval functions from database.operations instead of duplicating them
from database.operations import (
    get_region_id,
    get_all_alpha_ids_by_region_basic,
    get_pnl_data_for_alphas
)

from config.database_config import REGIONS

# Configure logging
logger = logging.getLogger(__name__)

# Helper function for parallel processing - must be at module level for pickling
def process_single_alpha_correlations(args):
    """
    Calculates all correlations for a single primary alpha against all other alphas.
    This function is designed to be mapped in parallel.

    Args:
        args: A tuple (primary_alpha_id, all_pnl_data_dict)
              - primary_alpha_id: The ID of the alpha for which to calculate correlations.
              - all_pnl_data_dict: Dict of {alpha_id: {'df': full_pnl_dataframe}} for all alphas.

    Returns:
        A tuple (primary_alpha_id, stats_dict) where stats_dict contains
        min, max, avg, median correlations for the primary_alpha_id, or None if no valid correlations.
    """
    primary_alpha_id, all_pnl_data_dict = args

    try:
        # Create an empty list to store correlations for this primary alpha
        correlations_for_primary = []
        
        # ===== VALIDATE PRIMARY ALPHA DATA =====
        # Check if primary alpha exists in the data dict
        if primary_alpha_id not in all_pnl_data_dict:
            logger.warning(f"Primary alpha {primary_alpha_id} not found in PNL data dictionary")
            return primary_alpha_id, None
            
        # Get the primary alpha data dictionary
        primary_alpha_data = all_pnl_data_dict[primary_alpha_id]
        
        # Check if the primary alpha data dictionary has a 'df' key
        if not isinstance(primary_alpha_data, dict) or 'df' not in primary_alpha_data:
            logger.warning(f"No dataframe key found for alpha {primary_alpha_id}")
            return primary_alpha_id, None
            
        # Get the primary alpha PNL dataframe
        primary_alpha_pnl_df = primary_alpha_data['df']
        
        # Check if the dataframe is None
        if primary_alpha_pnl_df is None:
            logger.warning(f"None dataframe for alpha {primary_alpha_id}")
            return primary_alpha_id, None
            
        # Now check if the dataframe is empty or not a dataframe
        if not isinstance(primary_alpha_pnl_df, pd.DataFrame):
            logger.warning(f"Invalid dataframe type for alpha {primary_alpha_id}: {type(primary_alpha_pnl_df)}")
            return primary_alpha_id, None
            
        if primary_alpha_pnl_df.empty:
            logger.warning(f"Empty PNL dataframe for alpha {primary_alpha_id}")
            return primary_alpha_id, None
    
        # ===== DETERMINE TIME WINDOW FOR PRIMARY ALPHA =====
        try:
            primary_end_date = primary_alpha_pnl_df.index.max()
            primary_start_date = primary_end_date - pd.DateOffset(years=4)
            
            # Keep reference to original dataframe
            primary_windowed_df = primary_alpha_pnl_df
            
            # Apply time window filter if we have enough data
            filtered_data = primary_alpha_pnl_df.loc[primary_alpha_pnl_df.index >= primary_start_date]
            if len(filtered_data) >= 60:  # Only use windowed data if we have at least 60 data points
                primary_windowed_df = filtered_data
        except Exception as e:
            logger.error(f"Error determining time window for alpha {primary_alpha_id}: {e}")
            return primary_alpha_id, None
    
        # ===== CALCULATE CORRELATIONS WITH ALL OTHER ALPHAS =====
        # Track alphas processed and correlations found
        processed_alpha_count = 0
        correlations_found = 0
        
        for other_alpha_id, other_alpha_data in all_pnl_data_dict.items():
            # Skip self correlation
            if primary_alpha_id == other_alpha_id:
                continue
            
            processed_alpha_count += 1
            
            # ===== VALIDATE OTHER ALPHA DATA =====
            # Check if other alpha data is a dictionary with a 'df' key
            if not isinstance(other_alpha_data, dict) or 'df' not in other_alpha_data:
                logger.debug(f"No dataframe key found for alpha {other_alpha_id}")
                continue
                
            # Get the dataframe from the nested structure
            other_pnl_df = other_alpha_data['df']
            
            # Check if the dataframe is None
            if other_pnl_df is None:
                logger.debug(f"None dataframe for alpha {other_alpha_id}")
                continue
                
            # Check if the dataframe is a valid pandas DataFrame
            if not isinstance(other_pnl_df, pd.DataFrame):
                logger.debug(f"Invalid dataframe type for alpha {other_alpha_id}: {type(other_pnl_df)}")
                continue
                
            # Check if the dataframe is empty
            if other_pnl_df.empty:
                logger.debug(f"Empty PNL dataframe for alpha {other_alpha_id}")
                continue
    
            # ===== FIND COMMON DATES AND CALCULATE CORRELATION =====
            try:
                # Find common dates between primary_windowed_df and other_pnl_df
                common_dates = primary_windowed_df.index.intersection(other_pnl_df.index)
                
                if len(common_dates) < 20:  # Require at least 20 common dates
                    logger.debug(f"Insufficient common dates ({len(common_dates)}) between {primary_alpha_id} and {other_alpha_id}")
                    continue
                
                # Extract PNL values for common dates - these are cumulative PNLs
                primary_pnl_values = primary_windowed_df.loc[common_dates, 'pnl'].values
                other_pnl_values = other_pnl_df.loc[common_dates, 'pnl'].values
                
                # Calculate correlation using the optimized function
                corr = calculate_alpha_correlation_fast(primary_pnl_values, other_pnl_values)
                if corr is not None:
                    correlations_for_primary.append(corr)
                    correlations_found += 1
            except Exception as e:
                logger.debug(f"Error calculating correlation between {primary_alpha_id} and {other_alpha_id}: {e}")
                continue
    
        # ===== LOG SUMMARY AND RETURN RESULTS =====
        if processed_alpha_count > 0:
            logger.debug(f"Alpha {primary_alpha_id}: processed {processed_alpha_count} pairs, found {correlations_found} valid correlations")
                
        if not correlations_for_primary:
            logger.info(f"No valid correlations found for alpha {primary_alpha_id}")
            return primary_alpha_id, None
    
        # Calculate correlation statistics for the primary_alpha_id
        stats = {
            'min_corr': float(min(correlations_for_primary)),
            'max_corr': float(max(correlations_for_primary)),
            'avg_corr': float(sum(correlations_for_primary) / len(correlations_for_primary)),
            'median_corr': float(np.median(correlations_for_primary)),
            'count': len(correlations_for_primary)  # Track number of correlations calculated
        }
        return primary_alpha_id, stats
        
    except Exception as e:
        # Catch all exceptions to prevent worker failure
        logger.error(f"Unexpected error processing alpha {primary_alpha_id}: {e}")
        return primary_alpha_id, None

def calculate_and_store_correlations_optimized(region: str) -> None:
    """
    Optimized version of correlation calculation with parallelism and Cython acceleration.
    Uses the exact same formula as the original but with significant performance improvements.
    
    Args:
        region: Region name
    """
    start_time = time.time()
    try:
        logger.info(f"Correlation calculation for region {region} starting (using ORIGINAL data fetching methods for debugging)...")
        # Fetch all alpha IDs for the region using the ORIGINAL script's function
        alpha_ids = get_all_alpha_ids_by_region_basic(region)
        if not alpha_ids:
            logger.warning(f"No alpha IDs found for region {region} using get_all_alpha_ids_by_region_basic.")
            return
        logger.info(f"Retrieved {len(alpha_ids)} alpha IDs for region {region} using get_all_alpha_ids_by_region_basic.")

        # Fetch PNL data for all relevant alphas using the ORIGINAL script's function
        pnl_data_dict = get_pnl_data_for_alphas(alpha_ids, region)
        if not pnl_data_dict:
            logger.warning(f"No PNL data found for region {region} using get_pnl_data_for_alphas.")
            return
        logger.info(f"Retrieved PNL data for {len(pnl_data_dict)} alphas in region {region} using get_pnl_data_for_alphas.")
        
        # Since get_pnl_data_for_alphas now returns a properly nested structure
        # with {alpha_id: {'df': dataframe}}, we can use it directly without additional preprocessing
        preprocessed_data = pnl_data_dict
        
        # Validate that we have data in the expected format
        sample_keys = list(preprocessed_data.keys())[:5] if preprocessed_data else []
        for alpha_id in sample_keys:
            if 'df' not in preprocessed_data.get(alpha_id, {}):
                logger.warning(f"Data format validation: Missing 'df' key for alpha {alpha_id}")
                
        logger.info(f"Data format validation: Checked {len(sample_keys)} sample alphas")

        
        alpha_ids = list(preprocessed_data.keys())
        total_alphas = len(alpha_ids)

        if total_alphas == 0:
            logger.warning("No preprocessed PNL data available for correlation calculation.")
            return

        logger.info(f"Starting correlation calculation for {total_alphas} alphas.")

        # Use multiple CPU cores with a reasonable limit
        max_workers = min(os.cpu_count() or 4, 8)  # Default to 4 if cpu_count returns None
        logger.info(f"Using {max_workers} parallel workers for correlation calculation")

        # Create task arguments: (primary_alpha_id, all_preprocessed_pnl_data)
        # The all_preprocessed_pnl_data dict is passed to each worker; it's read-only.
        task_args = [(alpha_id, preprocessed_data) for alpha_id in alpha_ids]

        results = {}
        processed_count = 0
        start_calc_time = time.time()

        if total_alphas < 5 or max_workers == 1: # Small dataset or no parallelism benefit
            logger.info("Processing correlations sequentially due to small dataset or max_workers=1.")
            for single_task_args in task_args:
                primary_id, stats = process_single_alpha_correlations(single_task_args)
                if stats is not None:
                    results[primary_id] = stats
                processed_count += 1
                if processed_count % 20 == 0 or processed_count == total_alphas:
                    logger.info(f"Sequentially processed {processed_count}/{total_alphas} alphas...")
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # executor.map will pass each item from task_args to process_single_alpha_correlations
                for primary_id, stats in executor.map(process_single_alpha_correlations, task_args):
                    if stats is not None:
                        results[primary_id] = stats
                    processed_count += 1
                    if processed_count % 50 == 0 or processed_count == total_alphas:
                        logger.info(f"Processed {processed_count}/{total_alphas} alphas ({processed_count/total_alphas:.1%})...")
        
        calc_time = time.time() - start_calc_time
        logger.info(f"Correlation calculation for {len(results)} alphas completed in {calc_time:.2f} seconds")
        
        # Store results in database
        if not results:
            logger.warning(f"No correlation results to store for region {region}.")
            return

        logger.info(f"Storing correlation results for {len(results)} alphas in region {region}...")
        db_engine = get_connection() # Ensure we have the engine for this block
        table_name = f"correlation_{region.lower()}"
        upsert_count = 0
        
        with db_engine.connect() as connection:
            with connection.begin(): # Handles transaction commit/rollback
                for alpha_id_key, stats_data in results.items():
                    if stats_data is None:
                        continue
                    
                    upsert_query_sql = f"""
                    INSERT INTO {table_name} (alpha_id, min_correlation, max_correlation, avg_correlation, median_correlation)
                    VALUES (:alpha_id, :min_corr, :max_corr, :avg_corr, :median_corr)
                    ON CONFLICT (alpha_id) DO UPDATE SET
                        min_correlation = EXCLUDED.min_correlation,
                        max_correlation = EXCLUDED.max_correlation,
                        avg_correlation = EXCLUDED.avg_correlation,
                        median_correlation = EXCLUDED.median_correlation,
                        last_updated = CURRENT_TIMESTAMP
                    """
                    upsert_stmt = text(upsert_query_sql)
                    try:
                        connection.execute(upsert_stmt, {
                            "alpha_id": alpha_id_key, 
                            "min_corr": stats_data['min_corr'], 
                            "max_corr": stats_data['max_corr'], 
                            "avg_corr": stats_data['avg_corr'],
                            "median_corr": stats_data['median_corr']
                        })
                        upsert_count += 1
                    except Exception as e_upsert:
                        logger.error(f"Error upserting correlation for alpha_id {alpha_id_key} in region {region}: {e_upsert}")
            # Transaction commits here if no exception, or rolls back on exception
        
        if upsert_count > 0:
            logger.info(f"Successfully stored/updated correlation statistics for {upsert_count} alphas in region {region}.")
        else:
            pass  # No statistics to update
            
        total_time = time.time() - start_time
        logger.info(f"Optimized correlation calculation and storage for region {region} completed in {total_time:.2f} seconds.")
        
    except Exception as e:
        logger.error(f"Error calculating correlations for region {region}: {e}")
        raise

def main():
    """Main function to handle command-line arguments and execute the correlation calculation."""
    parser = argparse.ArgumentParser(description='Calculate and update correlation statistics for alphas (optimized version)')
    parser.add_argument('--region', choices=REGIONS, help='Region to update correlations for')
    parser.add_argument('--all', action='store_true', help='Update correlations for all regions')
    parser.add_argument('--workers', type=int, default=0, help='Number of worker processes (0 = auto)')
    args = parser.parse_args()
    
    if not args.region and not args.all:
        parser.error('Either --region or --all must be specified')
    
    # Set up logging
    setup_logging()
    
    # Set the worker count for concurrency if specified
    if args.workers > 0:
        os.environ["CORRELATIONS_MAX_WORKERS"] = str(args.workers)
    
    # Determine which regions to update
    regions_to_update = REGIONS if args.all else [args.region]
    
    total_start_time = time.time()
    
    for region in regions_to_update:
        logger.info(f"Calculating optimized correlations for region {region}...")
        
        try:
            calculate_and_store_correlations_optimized(region)
            logger.info(f"Successfully updated correlation statistics for region {region}")
        except Exception as e:
            logger.error(f"Error calculating optimized correlations for region {region}: {e}")
    
    total_elapsed_time = time.time() - total_start_time
    logger.info(f"Optimized correlation calculation complete in {total_elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
