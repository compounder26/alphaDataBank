"""
Unified correlation engine for all alpha correlation calculations.

Consolidates logic from:
- update_correlations_optimized.py (submitted alphas)
- calculate_unsubmitted_correlations.py (unsubmitted vs submitted)
- calculate_cross_correlation.py (cross-correlation analysis)
"""
import sys
import os
import logging
import numpy as np
import pandas as pd
import time
import concurrent.futures
import multiprocessing
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Setup project path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.bootstrap import setup_project_path
setup_project_path()

from utils.helpers import setup_logging
from database.operations import (
    get_all_alpha_ids_by_region_basic,
    get_pnl_data_for_alphas,
    get_submitted_pnl_data_optimized  # Optimized bulk loading
)
from database.operations_unsubmitted import (
    get_all_unsubmitted_alpha_ids_by_region,
    get_unsubmitted_pnl_data_for_alphas,
    update_multiple_unsubmitted_alpha_self_correlations,
    get_unsubmitted_pnl_data_optimized  # Optimized bulk loading
)
from database.schema import get_connection, get_region_id
from config.database_config import REGIONS
from sqlalchemy import text
import requests
import json
from api.auth import get_authenticated_session
from api.alpha_fetcher import get_robust_session

logger = logging.getLogger(__name__)

# Module-level functions for ProcessPoolExecutor (must be picklable)
def calculate_submitted_correlation_worker(args: Tuple) -> Tuple[str, List[float]]:
    """
    Module-level worker function for calculating all correlations for one submitted alpha.
    This function can be pickled and used with ProcessPoolExecutor.

    Args:
        args: Tuple of (primary_alpha_id, primary_df, other_alpha_data_list)
              where other_alpha_data_list contains tuples of (alpha_id, alpha_df)

    Returns:
        Tuple of (primary_alpha_id, list_of_correlations)
    """
    primary_alpha_id, primary_df, other_alpha_data_list = args
    correlations = []

    if primary_df is None or primary_df.empty:
        return primary_alpha_id, correlations

    for other_alpha_id, other_df in other_alpha_data_list:
        if other_df is None or other_df.empty:
            continue

        try:
            # Find common dates using DataFrame intersection
            common_dates = primary_df.index.intersection(other_df.index)

            if len(common_dates) < 20:
                continue

            # Extract PNL values
            primary_pnl = primary_df.loc[common_dates, 'pnl'].values.astype(np.float64)
            other_pnl = other_df.loc[common_dates, 'pnl'].values.astype(np.float64)

            # Calculate correlation using optimized method
            try:
                from utils.cython_helper import get_correlation_utils
                correlation_utils = get_correlation_utils()
                if correlation_utils is not None:
                    corr = correlation_utils.calculate_alpha_correlation_fast(primary_pnl, other_pnl)
                else:
                    corr = _calculate_correlation_numpy(primary_pnl, other_pnl)
            except:
                corr = _calculate_correlation_numpy(primary_pnl, other_pnl)

            if corr is not None:
                correlations.append(corr)

        except Exception as e:
            continue

    return primary_alpha_id, correlations

def calculate_single_correlation_worker(args: Tuple) -> Tuple[str, str, Optional[float]]:
    """
    Module-level worker function for calculating correlation between two alphas.
    This function can be pickled and used with ProcessPoolExecutor.

    Args:
        args: Tuple of (unsubmitted_id, submitted_id, unsubmitted_df, submitted_df)

    Returns:
        Tuple of (unsubmitted_id, submitted_id, correlation)
    """
    unsub_id, sub_id, unsub_df, sub_df = args

    try:
        if unsub_df is None or unsub_df.empty or sub_df is None or sub_df.empty:
            return unsub_id, sub_id, None

        # Use DataFrame's built-in intersection (faster than set conversion)
        common_dates = unsub_df.index.intersection(sub_df.index)
        if len(common_dates) < 20:
            return unsub_id, sub_id, None

        # Extract PNL values
        unsub_pnl = unsub_df.loc[common_dates, 'pnl'].values.astype(np.float64)
        sub_pnl = sub_df.loc[common_dates, 'pnl'].values.astype(np.float64)

        # Calculate correlation using the optimized method
        # Try to use Cython if available
        try:
            from utils.cython_helper import get_correlation_utils
            correlation_utils = get_correlation_utils()
            if correlation_utils is not None:
                corr = correlation_utils.calculate_alpha_correlation_fast(unsub_pnl, sub_pnl)
            else:
                # Fallback to numpy
                corr = _calculate_correlation_numpy(unsub_pnl, sub_pnl)
        except:
            # Fallback to numpy
            corr = _calculate_correlation_numpy(unsub_pnl, sub_pnl)

        return unsub_id, sub_id, corr

    except Exception as e:
        logger.debug(f"Error in correlation worker for {unsub_id}-{sub_id}: {e}")
        return unsub_id, sub_id, None

def _calculate_correlation_numpy(pnl1: np.ndarray, pnl2: np.ndarray) -> Optional[float]:
    """
    Fallback numpy correlation calculation when Cython is not available.
    """
    try:
        if len(pnl1) < 20:
            return None

        # Calculate returns
        returns1 = np.diff(pnl1) / pnl1[:-1]
        returns2 = np.diff(pnl2) / pnl2[:-1]

        # Remove NaN and inf
        mask = np.isfinite(returns1) & np.isfinite(returns2)
        if np.sum(mask) < 20:
            return None

        clean_returns1 = returns1[mask]
        clean_returns2 = returns2[mask]

        # Check for constant arrays
        if np.std(clean_returns1) == 0 or np.std(clean_returns2) == 0:
            return None

        # Calculate correlation
        corr = np.corrcoef(clean_returns1, clean_returns2)[0, 1]
        return corr if np.isfinite(corr) else None

    except:
        return None

class CorrelationEngine:
    """
    Unified correlation calculation engine supporting all alpha correlation scenarios.
    Optimized for better CPU utilization and memory efficiency.
    """

    def __init__(self, use_cython: bool = True):
        """
        Initialize the correlation engine.

        Args:
            use_cython: Whether to use Cython acceleration (default: True)
        """
        self.use_cython = use_cython
        self._load_correlation_function()
        self.optimal_workers = self._get_optimal_worker_count()
    
    def _load_correlation_function(self):
        """Load the correlation calculation function (Cython or Python fallback)."""
        if self.use_cython:
            # Use cython_helper to automatically compile if needed
            from utils.cython_helper import get_correlation_utils

            correlation_utils = get_correlation_utils()
            if correlation_utils is not None:
                self.calculate_alpha_correlation_fast = correlation_utils.calculate_alpha_correlation_fast
                logger.info("Using Cython-accelerated correlation calculations")
            else:
                logger.warning("Could not load Cython extension. Using Python fallback (100x slower).")
                self.calculate_alpha_correlation_fast = self._python_correlation_fallback
        else:
            self.calculate_alpha_correlation_fast = self._python_correlation_fallback
            logger.info("Using Python correlation calculations")

    def _get_optimal_worker_count(self) -> int:
        """
        Determine optimal number of workers based on system resources.

        Returns:
            Optimal number of workers
        """
        cpu_cores = multiprocessing.cpu_count()

        if PSUTIL_AVAILABLE:
            # Get available memory in GB
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            # Each worker might use ~500MB-1GB of memory for large datasets
            memory_based_workers = int(available_memory_gb / 0.5)

            # Use 2x CPU cores for I/O bound tasks, but cap based on memory
            cpu_based_workers = cpu_cores * 2

            optimal = min(cpu_based_workers, memory_based_workers, 32)

            logger.info(f"System has {cpu_cores} CPU cores and {available_memory_gb:.1f}GB available memory")
            logger.info(f"Optimal worker count: {optimal}")
        else:
            # Fallback if psutil not available
            optimal = min(cpu_cores * 2, 32)
            logger.info(f"System has {cpu_cores} CPU cores. Optimal worker count: {optimal}")

        return max(optimal, 4)  # Minimum 4 workers
    
    def _python_correlation_fallback(self, alpha_pnl: np.ndarray, other_pnl: np.ndarray, 
                                   initial_value: float = 10000000.0) -> Optional[float]:
        """
        Python fallback implementation of correlation calculation.
        Uses the same formula as Cython version but in pure Python.
        """
        if len(alpha_pnl) < 20 or len(other_pnl) < 20:
            return None
        
        try:
            # Calculate daily PNL (first differences of cumulative PNL)
            alpha_daily = np.zeros_like(alpha_pnl)
            other_daily = np.zeros_like(other_pnl)
            
            alpha_daily[0] = alpha_pnl[0]
            other_daily[0] = other_pnl[0]
            
            for i in range(1, len(alpha_pnl)):
                alpha_daily[i] = alpha_pnl[i] - alpha_pnl[i-1]
                other_daily[i] = other_pnl[i] - other_pnl[i-1]
            
            # Calculate portfolio values (initial value + cumulative daily PNL)
            alpha_port_val = np.zeros_like(alpha_pnl)
            other_port_val = np.zeros_like(other_pnl)
            
            alpha_port_val[0] = initial_value + alpha_daily[0]
            other_port_val[0] = initial_value + other_daily[0]
            
            for i in range(1, len(alpha_pnl)):
                alpha_port_val[i] = alpha_port_val[i-1] + alpha_daily[i]
                other_port_val[i] = other_port_val[i-1] + other_daily[i]
            
            # Calculate returns
            alpha_returns = np.full_like(alpha_pnl, np.nan)
            other_returns = np.full_like(other_pnl, np.nan)
            
            for i in range(1, len(alpha_pnl)):
                if alpha_port_val[i-1] != 0:
                    alpha_returns[i] = alpha_daily[i] / alpha_port_val[i-1]
                if other_port_val[i-1] != 0:
                    other_returns[i] = other_daily[i] / other_port_val[i-1]
            
            # Filter valid values
            valid_mask = ~(np.isnan(alpha_returns) | np.isnan(other_returns) | 
                          np.isinf(alpha_returns) | np.isinf(other_returns))
            
            alpha_returns_clean = alpha_returns[valid_mask]
            other_returns_clean = other_returns[valid_mask]
            
            if len(alpha_returns_clean) < 20:
                return None
            
            # Calculate correlation
            from scipy import stats
            import warnings

            # Suppress the constant input warning since we handle NaN results properly
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                corr, _ = stats.pearsonr(alpha_returns_clean, other_returns_clean)

            return corr if not np.isnan(corr) else None
            
        except Exception as e:
            logger.debug(f"Error in correlation calculation: {e}")
            return None
    
    def calculate_pairwise(self, alpha1_pnl: np.ndarray, alpha2_pnl: np.ndarray) -> Optional[float]:
        """
        Calculate correlation between two alpha PNL series.
        
        Args:
            alpha1_pnl: PNL array for first alpha
            alpha2_pnl: PNL array for second alpha
            
        Returns:
            Correlation coefficient or None if calculation fails
        """
        return self.calculate_alpha_correlation_fast(alpha1_pnl, alpha2_pnl)
    
    def _get_pnl_data_windowed(self, alpha_pnl_dict: Dict, primary_alpha_id: str, 
                             other_alpha_ids: List[str], window_days: int = 400) -> pd.DataFrame:
        """
        Get windowed PNL data for primary alpha based on availability.
        """
        primary_alpha_data = alpha_pnl_dict.get(primary_alpha_id)
        if not isinstance(primary_alpha_data, dict) or 'df' not in primary_alpha_data:
            logger.debug(f"Alpha {primary_alpha_id} filtered: missing data structure")
            return pd.DataFrame()
            
        primary_pnl_df = primary_alpha_data['df']
        if primary_pnl_df is None or primary_pnl_df.empty:
            logger.debug(f"Alpha {primary_alpha_id} filtered: None or empty dataframe")
            return pd.DataFrame()
        
        # Get the most recent dates, up to window_days
        if len(primary_pnl_df) <= window_days:
            return primary_pnl_df.copy()
        else:
            return primary_pnl_df.tail(window_days).copy()
    
    def _calculate_correlations_for_alpha(self, primary_alpha_id: str, 
                                        alpha_pnl_dict: Dict, other_alpha_ids: List[str], 
                                        window_days: int = 400) -> List[float]:
        """
        Calculate all correlations for a single primary alpha against other alphas.
        """
        correlations = []
        
        # Get windowed data for primary alpha
        primary_windowed_df = self._get_pnl_data_windowed(
            alpha_pnl_dict, primary_alpha_id, other_alpha_ids, window_days
        )
        
        if primary_windowed_df.empty:
            logger.warning(f"Alpha {primary_alpha_id} skipped from correlation calculation: insufficient windowed PNL data")
            return correlations
        
        # Calculate correlations with all other alphas
        for other_alpha_id in other_alpha_ids:
            if primary_alpha_id == other_alpha_id:
                continue
                
            other_alpha_data = alpha_pnl_dict.get(other_alpha_id)
            if not isinstance(other_alpha_data, dict) or 'df' not in other_alpha_data:
                continue
                
            other_pnl_df = other_alpha_data['df']
            if other_pnl_df is None or other_pnl_df.empty:
                continue
            
            try:
                # Find common dates
                common_dates = primary_windowed_df.index.intersection(other_pnl_df.index)
                
                if len(common_dates) < 20:
                    continue
                
                # Extract PNL values for common dates
                primary_pnl_values = primary_windowed_df.loc[common_dates, 'pnl'].values
                other_pnl_values = other_pnl_df.loc[common_dates, 'pnl'].values
                
                # Calculate correlation
                corr = self.calculate_alpha_correlation_fast(primary_pnl_values, other_pnl_values)
                if corr is not None:
                    correlations.append(corr)
                    
            except Exception as e:
                logger.debug(f"Error calculating correlation between {primary_alpha_id} and {other_alpha_id}: {e}")
                continue
        
        return correlations
    
    def calculate_batch_submitted(self, region: str, max_workers: Optional[int] = None,
                                window_days: int = 400) -> None:
        """
        Calculate correlations for submitted alphas in a region using ProcessPoolExecutor for true parallelism.
        Optimized version with bulk loading and better CPU utilization.
        """
        start_time = time.time()
        logger.info(f"Starting optimized batch correlation calculation for submitted alphas in region {region}")

        try:
            # Get all alpha IDs for the region
            alpha_ids = get_all_alpha_ids_by_region_basic(region)
            if not alpha_ids:
                logger.warning(f"No alphas found for region {region}")
                return

            logger.info(f"Processing correlations for {len(alpha_ids)} alphas in region {region}")

            # Use bulk loading for much faster data retrieval
            logger.info("Using bulk loading for PNL data...")
            from database.operations import get_pnl_data_bulk
            pnl_data_dict = get_pnl_data_bulk(alpha_ids, region)

            if not pnl_data_dict:
                logger.warning("No PNL data found for any alphas")
                return

            # Log which alphas were skipped
            skipped_alphas = [alpha_id for alpha_id in alpha_ids if alpha_id not in pnl_data_dict]
            if skipped_alphas:
                logger.warning(f"Skipped {len(skipped_alphas)} alphas due to missing PNL data")

            logger.info(f"Successfully loaded PNL data for {len(pnl_data_dict)} alphas using bulk query")

            # Apply windowing to all alphas
            windowed_data = {}
            for alpha_id, df in pnl_data_dict.items():
                if window_days and len(df) > window_days:
                    windowed_data[alpha_id] = df.iloc[-window_days:]
                else:
                    windowed_data[alpha_id] = df

            # Prepare work items for parallel processing
            work_items = []
            alphas_with_pnl = list(windowed_data.keys())

            for i, primary_alpha_id in enumerate(alphas_with_pnl):
                # Create list of other alphas (excluding self)
                other_alpha_data = [
                    (other_id, windowed_data[other_id])
                    for other_id in alphas_with_pnl
                    if other_id != primary_alpha_id
                ]

                work_items.append((
                    primary_alpha_id,
                    windowed_data[primary_alpha_id],
                    other_alpha_data
                ))

            # Calculate correlations in parallel using ProcessPoolExecutor
            correlation_results = {}
            completed_count = 0
            total_alphas = len(work_items)

            logger.info(f"Starting correlation calculation for {total_alphas} alphas with PNL data")

            # Use optimal worker count if not specified
            if max_workers is None:
                max_workers = self.optimal_workers

            # Use ProcessPoolExecutor for true parallelism (same approach as unsubmitted alphas)
            logger.info(f"Using ProcessPoolExecutor with {max_workers} workers for true parallelism")

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all work items
                futures = [executor.submit(calculate_submitted_correlation_worker, item) for item in work_items]

                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        alpha_id, correlations = future.result()

                        if correlations:
                            correlation_results[alpha_id] = {
                                'min': float(min(correlations)),
                                'max': float(max(correlations)),
                                'avg': float(np.mean(correlations)),
                                'median': float(np.median(correlations))
                            }

                        completed_count += 1
                        # Progress updates
                        if completed_count % 25 == 0 or completed_count == total_alphas:
                            logger.info(f"Completed correlations for {completed_count}/{total_alphas} alphas")

                    except Exception as e:
                        logger.error(f"Error processing correlation result: {e}")
            
            # Store results in database
            if correlation_results:
                self._store_correlation_results(correlation_results, region)
                logger.info(f"Successfully stored correlation results for {len(correlation_results)} alphas")
            else:
                logger.warning("No correlation results to store")
            
            # Final summary with detailed breakdown
            elapsed_time = time.time() - start_time
            logger.info(f"Batch correlation calculation completed in {elapsed_time:.2f} seconds")
            logger.info(f"Summary: {len(alpha_ids)} total alphas -> {len(alphas_with_pnl)} with PNL data -> {len(correlation_results)} with computed correlations")
            
        except Exception as e:
            logger.error(f"Error in batch correlation calculation: {e}")
            raise
    
    def _store_correlation_results(self, correlation_results: Dict, region: str):
        """Store correlation results in the database."""
        try:
            db_engine = get_connection()
            with db_engine.connect() as connection:
                # Clear existing correlations for this region
                table_name = f'correlation_{region.lower()}'
                connection.execute(text(f"DELETE FROM {table_name}"))
                
                # Insert new correlations
                for alpha_id, stats in correlation_results.items():
                    insert_query = text(f"""
                        INSERT INTO {table_name} 
                        (alpha_id, min_correlation, max_correlation, avg_correlation, median_correlation, last_updated)
                        VALUES (:alpha_id, :min_corr, :max_corr, :avg_corr, :median_corr, NOW())
                        ON CONFLICT (alpha_id) DO UPDATE SET
                            min_correlation = EXCLUDED.min_correlation,
                            max_correlation = EXCLUDED.max_correlation,
                            avg_correlation = EXCLUDED.avg_correlation,
                            median_correlation = EXCLUDED.median_correlation,
                            last_updated = EXCLUDED.last_updated
                    """)
                    
                    connection.execute(insert_query, {
                        'alpha_id': alpha_id,
                        'min_corr': float(stats['min']) if stats['min'] is not None else None,
                        'max_corr': float(stats['max']) if stats['max'] is not None else None,
                        'avg_corr': float(stats['avg']) if stats['avg'] is not None else None,
                        'median_corr': float(stats['median']) if stats['median'] is not None else None
                    })
                
                connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing correlation results: {e}")
            raise
    
    def calculate_unsubmitted_vs_submitted(self, region: str, max_workers: Optional[int] = None,
                                          use_batching: bool = True, batch_size: int = 100,
                                          use_multiprocessing: bool = True) -> None:
        """
        Calculate correlations between unsubmitted alphas and submitted alphas.
        For each unsubmitted alpha, finds the maximum correlation with any submitted alpha.
        This replaces the logic from calculate_unsubmitted_correlations.py

        Args:
            region: Region to process
            max_workers: Number of parallel workers. If None, uses CPU count * 2
            use_batching: Whether to use the optimized batched version (default: True)
            batch_size: Number of unsubmitted alphas per batch if batching (default: 100)
            use_multiprocessing: Use ProcessPoolExecutor for true parallelism (default: True)
        """
        # Use the optimized batched version by default
        if use_batching:
            logger.info(f"Using optimized batched correlation calculation for region {region}")
            return self.calculate_unsubmitted_vs_submitted_batched(
                region, batch_size, max_workers, use_multiprocessing
            )

        # Original implementation (kept for backward compatibility)
        start_time = time.time()
        logger.info(f"Starting unsubmitted vs submitted correlation calculation for region {region} (legacy mode)")
        
        try:
            # Get unsubmitted alpha IDs
            unsubmitted_alpha_ids = get_all_unsubmitted_alpha_ids_by_region(region)
            if not unsubmitted_alpha_ids:
                logger.warning(f"No unsubmitted alphas found for region {region}")
                return
            
            # Get submitted alpha IDs
            submitted_alpha_ids = get_all_alpha_ids_by_region_basic(region)
            if not submitted_alpha_ids:
                logger.warning(f"No submitted alphas found for region {region}")
                return
            
            logger.info(f"Processing correlations between {len(unsubmitted_alpha_ids)} unsubmitted "
                       f"and {len(submitted_alpha_ids)} submitted alphas in region {region}")
            
            # Get PNL data for both sets
            logger.info("Fetching PNL data...")
            unsubmitted_pnl_dict = get_unsubmitted_pnl_data_for_alphas(unsubmitted_alpha_ids, region)
            submitted_pnl_dict = get_pnl_data_for_alphas(submitted_alpha_ids, region)
            
            if not unsubmitted_pnl_dict:
                logger.warning("No PNL data found for unsubmitted alphas")
                return
            
            if not submitted_pnl_dict:
                logger.warning("No PNL data found for submitted alphas")
                return
            
            logger.info(f"Loaded PNL data for {len(unsubmitted_pnl_dict)} unsubmitted "
                       f"and {len(submitted_pnl_dict)} submitted alphas")
            
            # Calculate maximum correlations for each unsubmitted alpha
            correlation_results = {}
            
            def calculate_max_correlation_for_unsubmitted(unsubmitted_alpha_id: str) -> Tuple[str, Optional[float]]:
                """Calculate maximum correlation for a single unsubmitted alpha."""
                unsubmitted_data = unsubmitted_pnl_dict.get(unsubmitted_alpha_id)
                if not isinstance(unsubmitted_data, dict) or 'df' not in unsubmitted_data:
                    return unsubmitted_alpha_id, None
                
                unsubmitted_df = unsubmitted_data['df']
                if unsubmitted_df is None or unsubmitted_df.empty:
                    return unsubmitted_alpha_id, None
                
                max_correlation = 0.0
                correlations_found = 0
                
                for submitted_alpha_id in submitted_alpha_ids:
                    submitted_data = submitted_pnl_dict.get(submitted_alpha_id)
                    if not isinstance(submitted_data, dict) or 'df' not in submitted_data:
                        continue
                    
                    submitted_df = submitted_data['df']
                    if submitted_df is None or submitted_df.empty:
                        continue
                    
                    try:
                        # Find common dates
                        common_dates = unsubmitted_df.index.intersection(submitted_df.index)
                        if len(common_dates) < 20:
                            continue
                        
                        # Extract PNL values
                        unsubmitted_pnl = unsubmitted_df.loc[common_dates, 'pnl'].values
                        submitted_pnl = submitted_df.loc[common_dates, 'pnl'].values
                        
                        # Calculate correlation
                        corr = self.calculate_alpha_correlation_fast(unsubmitted_pnl, submitted_pnl)
                        if corr is not None:
                            abs_corr = abs(corr)
                            if abs_corr > max_correlation:
                                max_correlation = abs_corr
                            correlations_found += 1
                    
                    except Exception as e:
                        logger.debug(f"Error calculating correlation between {unsubmitted_alpha_id} and {submitted_alpha_id}: {e}")
                        continue
                
                result_corr = max_correlation if correlations_found > 0 else None
                logger.debug(f"Unsubmitted alpha {unsubmitted_alpha_id}: max correlation = {result_corr} "
                           f"(from {correlations_found} valid correlations)")
                
                return unsubmitted_alpha_id, result_corr
            
            # Use optimal worker count if not specified
            if max_workers is None:
                max_workers = self.optimal_workers
                logger.info(f"Using {max_workers} parallel workers for unsubmitted vs submitted correlation")

            # Process in parallel
            # Note: ThreadPoolExecutor is used due to pickling issues with nested functions
            # For better CPU utilization with CPU-bound tasks, consider refactoring to use ProcessPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(calculate_max_correlation_for_unsubmitted, alpha_id)
                    for alpha_id in unsubmitted_alpha_ids if alpha_id in unsubmitted_pnl_dict
                ]
                
                completed_count = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        alpha_id, max_corr = future.result()
                        if max_corr is not None:
                            correlation_results[alpha_id] = max_corr
                        
                        completed_count += 1
                        # More frequent progress updates for better monitoring
                        if completed_count % 25 == 0 or completed_count == len(unsubmitted_alpha_ids):
                            progress_pct = (completed_count / len(unsubmitted_alpha_ids)) * 100
                            logger.info(f"Completed {completed_count}/{len(unsubmitted_alpha_ids)} "
                                      f"unsubmitted alphas ({progress_pct:.1f}%)")
                    
                    except Exception as e:
                        logger.error(f"Error in parallel correlation calculation: {e}")
            
            # Store results
            if correlation_results:
                update_multiple_unsubmitted_alpha_self_correlations(correlation_results)
                logger.info(f"Successfully updated correlations for {len(correlation_results)} unsubmitted alphas")
            else:
                logger.warning("No correlations calculated for unsubmitted alphas")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Unsubmitted vs submitted correlation calculation completed in {elapsed_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error in unsubmitted vs submitted correlation calculation: {e}")
            raise

    def _calculate_single_pair_correlation(self, args: Tuple) -> Tuple[str, str, Optional[float]]:
        """
        Calculate correlation for a single pair of alphas.
        This is a standalone method that can be pickled for multiprocessing.

        Args:
            args: Tuple of (unsubmitted_id, submitted_id, unsubmitted_df, submitted_df)

        Returns:
            Tuple of (unsubmitted_id, submitted_id, correlation)
        """
        unsubmitted_id, submitted_id, unsubmitted_df, submitted_df = args

        try:
            if unsubmitted_df is None or unsubmitted_df.empty or submitted_df is None or submitted_df.empty:
                return unsubmitted_id, submitted_id, None

            # Find common dates
            common_dates = unsubmitted_df.index.intersection(submitted_df.index)
            if len(common_dates) < 20:
                return unsubmitted_id, submitted_id, None

            # Extract PNL values
            unsubmitted_pnl = unsubmitted_df.loc[common_dates, 'pnl'].values
            submitted_pnl = submitted_df.loc[common_dates, 'pnl'].values

            # Calculate correlation (suppress constant input warning)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='An input array is constant')
                corr = self.calculate_alpha_correlation_fast(unsubmitted_pnl, submitted_pnl)
            return unsubmitted_id, submitted_id, corr

        except Exception as e:
            logger.debug(f"Error calculating correlation for pair {unsubmitted_id}-{submitted_id}: {e}")
            return unsubmitted_id, submitted_id, None

    def calculate_unsubmitted_vs_submitted_batched(self, region: str, batch_size: int = 100,
                                                   max_workers: Optional[int] = None,
                                                   use_multiprocessing: bool = False) -> None:
        """
        Calculate correlations between unsubmitted and submitted alphas using memory-efficient batching.
        Processes unsubmitted alphas in batches to reduce memory usage and improve CPU utilization.

        Args:
            region: Region to process
            batch_size: Number of unsubmitted alphas to process at once (default: 100)
            max_workers: Number of parallel workers. If None, uses optimal count
            use_multiprocessing: Use multiprocessing.Pool instead of ThreadPoolExecutor for true parallelism
                                (experimental, may not work on Windows)
        """
        start_time = time.time()
        logger.info(f"Starting BATCHED unsubmitted vs submitted correlation calculation for region {region}")
        logger.info(f"Using batch size of {batch_size} unsubmitted alphas per batch")

        try:
            # Get all alpha IDs
            unsubmitted_alpha_ids = get_all_unsubmitted_alpha_ids_by_region(region)
            if not unsubmitted_alpha_ids:
                logger.warning(f"No unsubmitted alphas found for region {region}")
                return

            submitted_alpha_ids = get_all_alpha_ids_by_region_basic(region)
            if not submitted_alpha_ids:
                logger.warning(f"No submitted alphas found for region {region}")
                return

            total_unsubmitted = len(unsubmitted_alpha_ids)
            total_submitted = len(submitted_alpha_ids)
            total_pairs = total_unsubmitted * total_submitted

            logger.info(f"Processing {total_unsubmitted} unsubmitted × {total_submitted} submitted "
                       f"= {total_pairs:,} total correlations")

            # Load submitted PNL data ONCE (kept in memory for all batches)
            logger.info(f"Loading PNL data for {total_submitted} submitted alphas (reused across batches)...")
            logger.info("Using optimized BULK database loading (single query)")
            submitted_pnl_dict = get_submitted_pnl_data_optimized(submitted_alpha_ids, region)

            if not submitted_pnl_dict:
                logger.warning("No PNL data found for submitted alphas")
                return
            logger.info(f"Loaded PNL for {len(submitted_pnl_dict)} submitted alphas")

            # Prepare submitted DataFrames for efficient access
            submitted_dfs = {}
            for sub_id, sub_data in submitted_pnl_dict.items():
                if isinstance(sub_data, dict) and 'df' in sub_data:
                    submitted_dfs[sub_id] = sub_data['df']

            # Process unsubmitted alphas in batches
            all_correlation_results = {}
            total_processed = 0
            num_batches = (total_unsubmitted + batch_size - 1) // batch_size

            # Use optimal worker count
            if max_workers is None:
                max_workers = self.optimal_workers
                logger.info(f"Using {max_workers} parallel workers")

            for batch_num in range(num_batches):
                batch_start = batch_num * batch_size
                batch_end = min(batch_start + batch_size, total_unsubmitted)
                batch_ids = unsubmitted_alpha_ids[batch_start:batch_end]
                batch_size_actual = len(batch_ids)

                logger.info(f"Processing batch {batch_num + 1}/{num_batches}: "
                          f"{batch_size_actual} unsubmitted alphas "
                          f"({batch_start + 1}-{batch_end} of {total_unsubmitted})")

                # Load PNL data for this batch only
                logger.info(f"Loading batch PNL using optimized BULK query...")
                batch_pnl_dict = get_unsubmitted_pnl_data_optimized(batch_ids, region)

                if not batch_pnl_dict:
                    logger.warning(f"No PNL data for batch {batch_num + 1}")
                    continue

                logger.info(f"Loaded PNL for {len(batch_pnl_dict)} unsubmitted alphas in batch")

                # Prepare unsubmitted DataFrames for this batch
                unsubmitted_dfs = {}
                for unsub_id, unsub_data in batch_pnl_dict.items():
                    if isinstance(unsub_data, dict) and 'df' in unsub_data:
                        unsubmitted_dfs[unsub_id] = unsub_data['df']

                # Create pairs WITHOUT pre-computation (simpler and faster!)
                pairs = []
                for unsub_id in batch_ids:
                    if unsub_id not in unsubmitted_dfs:
                        continue
                    unsub_df = unsubmitted_dfs[unsub_id]
                    if unsub_df is None or unsub_df.empty:
                        continue

                    for sub_id in submitted_alpha_ids:
                        if sub_id not in submitted_dfs:
                            continue
                        sub_df = submitted_dfs[sub_id]
                        if sub_df is None or sub_df.empty:
                            continue

                        # Pass DataFrames directly - no pre-computation!
                        pairs.append((unsub_id, sub_id, unsub_df, sub_df))

                batch_pairs = len(pairs)
                logger.info(f"Processing {batch_pairs:,} correlation pairs in batch {batch_num + 1}")

                # Choose executor based on configuration
                batch_results = {}
                completed_in_batch = 0

                if use_multiprocessing:
                    # Use ProcessPoolExecutor for TRUE parallelism (bypasses GIL completely)
                    logger.info(f"Using ProcessPoolExecutor with {max_workers} workers for true parallelism")

                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all pairs using the module-level function
                        futures = [executor.submit(calculate_single_correlation_worker, pair) for pair in pairs]

                        # Process results as they complete
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                unsub_id, sub_id, corr = future.result()

                                # Track maximum correlation for each unsubmitted alpha
                                if corr is not None:
                                    abs_corr = abs(corr)
                                    if unsub_id not in batch_results or abs_corr > batch_results[unsub_id].get('max_correlation', 0):
                                        batch_results[unsub_id] = {'max_correlation': abs_corr}

                                completed_in_batch += 1
                                total_processed += 1

                                # Progress updates
                                if completed_in_batch % 1000 == 0 or completed_in_batch == batch_pairs:
                                    progress_pct = (total_processed / total_pairs) * 100
                                    logger.info(f"Batch {batch_num + 1} progress: {completed_in_batch}/{batch_pairs} pairs, "
                                              f"Overall: {total_processed:,}/{total_pairs:,} ({progress_pct:.1f}%)")

                            except Exception as e:
                                logger.error(f"Error processing correlation pair: {e}")

                else:
                    # Fallback to ThreadPoolExecutor (for Windows compatibility)
                    logger.info(f"Using ThreadPoolExecutor with {max_workers} workers (limited by GIL)")

                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all pairs using the module-level function
                        futures = [executor.submit(calculate_single_correlation_worker, pair) for pair in pairs]

                        # Process results as they complete
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                unsub_id, sub_id, corr = future.result()

                                # Track maximum correlation for each unsubmitted alpha
                                if corr is not None:
                                    abs_corr = abs(corr)
                                    if unsub_id not in batch_results or abs_corr > batch_results[unsub_id].get('max_correlation', 0):
                                        batch_results[unsub_id] = {'max_correlation': abs_corr}

                                completed_in_batch += 1
                                total_processed += 1

                                # Progress updates
                                if completed_in_batch % 1000 == 0 or completed_in_batch == batch_pairs:
                                    progress_pct = (total_processed / total_pairs) * 100
                                    logger.info(f"Batch {batch_num + 1} progress: {completed_in_batch}/{batch_pairs} pairs, "
                                              f"Overall: {total_processed:,}/{total_pairs:,} ({progress_pct:.1f}%)")

                            except Exception as e:
                                logger.error(f"Error processing correlation pair: {e}")

                # Add batch results to overall results
                all_correlation_results.update(batch_results)
                logger.info(f"Batch {batch_num + 1} complete. Found correlations for {len(batch_results)} alphas")

                # Clear batch memory before next iteration
                del batch_pnl_dict
                del unsubmitted_dfs
                del pairs
                logger.debug(f"Cleared memory for batch {batch_num + 1}")

            # Store all results
            if all_correlation_results:
                update_multiple_unsubmitted_alpha_self_correlations(all_correlation_results, region)
                logger.info(f"Successfully updated correlations for {len(all_correlation_results)} unsubmitted alphas")
            else:
                logger.warning("No correlations calculated for unsubmitted alphas")

            elapsed_time = time.time() - start_time
            logger.info(f"BATCHED correlation calculation completed in {elapsed_time:.2f} seconds")
            logger.info(f"Processed {total_pairs:,} pairs at {total_pairs/elapsed_time:.0f} pairs/second")

        except Exception as e:
            logger.error(f"Error in batched correlation calculation: {e}")
            raise

    def calculate_cross_correlation_analysis(self, alpha_ids: List[str], 
                                           reference_alpha_id: Optional[str] = None,
                                           csv_export_path: Optional[str] = None,
                                           session: Optional[requests.Session] = None) -> Dict[str, float]:
        """
        Perform cross-correlation analysis with optional CSV export.
        This replaces the logic from calculate_cross_correlation.py
        
        Args:
            alpha_ids: List of alpha IDs to analyze
            reference_alpha_id: Optional reference alpha for 1-to-many comparison
            csv_export_path: Optional path to save results as CSV
            session: Optional authenticated session for API calls
            
        Returns:
            Dictionary of alpha_id -> max_correlation
        """
        start_time = time.time()
        logger.info(f"Starting cross-correlation analysis for {len(alpha_ids)} alphas")
        
        try:
            # Fetch PNL data from API
            if session is None:
                session = get_authenticated_session()
                if session is None:
                    raise RuntimeError("Could not get authenticated session")
            
            all_pnl_data = {}
            successful_alphas = []
            
            logger.info("Fetching PNL data from API...")
            
            for i, alpha_id in enumerate(alpha_ids):
                if i % 10 == 0:
                    logger.info(f"Fetching PNL progress: {i}/{len(alpha_ids)}")
                
                try:
                    pnl_df = self._fetch_alpha_pnl_flexible(session, alpha_id)
                    if pnl_df is not None and not pnl_df.empty:
                        all_pnl_data[alpha_id] = pnl_df
                        successful_alphas.append(alpha_id)
                        logger.debug(f"Successfully fetched {len(pnl_df)} PNL records for {alpha_id}")
                    else:
                        logger.warning(f"No PNL data for alpha {alpha_id}")
                
                except Exception as e:
                    logger.error(f"Failed to fetch PNL for alpha {alpha_id}: {e}")
                    continue
            
            logger.info(f"Successfully fetched PNL data for {len(successful_alphas)} alphas")
            
            if len(successful_alphas) < 2:
                logger.warning("Need at least 2 alphas with valid PNL data for correlation analysis")
                return {}
            
            # Calculate correlations
            max_correlations = {alpha_id: 0.0 for alpha_id in successful_alphas}
            
            if reference_alpha_id and reference_alpha_id in successful_alphas:
                # 1-to-many comparison with reference alpha
                logger.info(f"Calculating correlations with reference alpha: {reference_alpha_id}")
                reference_pnl = all_pnl_data[reference_alpha_id]
                
                for alpha_id in successful_alphas:
                    if alpha_id == reference_alpha_id:
                        continue
                    
                    try:
                        corr = self._calculate_returns_correlation(
                            reference_pnl, all_pnl_data[alpha_id]
                        )
                        
                        if corr is not None and corr > 0:
                            max_correlations[alpha_id] = corr
                            max_correlations[reference_alpha_id] = max(
                                max_correlations[reference_alpha_id], corr
                            )
                    
                    except Exception as e:
                        logger.error(f"Error calculating correlation between {reference_alpha_id} and {alpha_id}: {e}")
            else:
                # All-to-all pairwise correlations
                logger.info("Calculating pairwise correlations for all alphas")
                total_pairs = len(successful_alphas) * (len(successful_alphas) - 1) // 2
                processed_pairs = 0
                
                for i, alpha1 in enumerate(successful_alphas):
                    for j, alpha2 in enumerate(successful_alphas[i+1:], i+1):
                        processed_pairs += 1
                        
                        if processed_pairs % 100 == 0:
                            logger.info(f"Correlation progress: {processed_pairs}/{total_pairs} ({processed_pairs/total_pairs*100:.1f}%)")
                        
                        try:
                            corr = self._calculate_returns_correlation(
                                all_pnl_data[alpha1], all_pnl_data[alpha2]
                            )
                            
                            if corr is not None and corr > 0:
                                if corr > max_correlations[alpha1]:
                                    max_correlations[alpha1] = corr
                                if corr > max_correlations[alpha2]:
                                    max_correlations[alpha2] = corr
                        
                        except Exception as e:
                            logger.error(f"Error calculating correlation between {alpha1} and {alpha2}: {e}")
            
            # Export to CSV if requested
            if csv_export_path:
                self._export_correlations_to_csv(max_correlations, csv_export_path)
            
            # Log summary
            valid_corrs = [v for v in max_correlations.values() if v > 0.0]
            logger.info(f"\nCross-correlation analysis summary:")
            logger.info(f"  - Alphas with valid correlations: {len(valid_corrs)}/{len(alpha_ids)}")
            
            if valid_corrs:
                logger.info(f"  - Max correlation: {max(valid_corrs):.4f}")
                logger.info(f"  - Average correlation: {np.mean(valid_corrs):.4f}")
                logger.info(f"  - Median correlation: {np.median(valid_corrs):.4f}")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Cross-correlation analysis completed in {elapsed_time:.2f} seconds")
            
            return max_correlations
        
        except Exception as e:
            logger.error(f"Error in cross-correlation analysis: {e}")
            raise
    
    def _fetch_alpha_pnl_flexible(self, session: requests.Session, alpha_id: str) -> Optional[pd.DataFrame]:
        """
        Fetch PNL data for a single alpha with flexible column handling.
        Based on fetch_alpha_pnl_flexible from calculate_cross_correlation.py
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
                        return None
                    
                    try:
                        pnl_data = result.json()
                        pnl_records = pnl_data.get("records")
                        
                        if not pnl_records:
                            logger.info(f"No PNL records for alpha {alpha_id}")
                            return None
                        
                        # Determine column count and create appropriate DataFrame
                        first_record = pnl_records[0]
                        if not isinstance(first_record, list):
                            logger.error(f"Unexpected record format for {alpha_id}")
                            return None
                        
                        num_cols = len(first_record)
                        
                        if num_cols == 2:
                            columns = ["date", "pnl"]
                        elif num_cols == 3:
                            columns = ["date", "pnl1", "pnl2"]
                        else:
                            logger.error(f"Unexpected column count for {alpha_id}: {num_cols}")
                            return None
                        
                        # Create DataFrame
                        pnl_df = pd.DataFrame(pnl_records, columns=columns)
                        pnl_df['date'] = pd.to_datetime(pnl_df['date'], format="%Y-%m-%d")
                        pnl_df = pnl_df.set_index('date').sort_index()
                        
                        # For multiple PNL columns, select the best performing one
                        pnl_columns = [col for col in pnl_df.columns if col.startswith('pnl')]
                        if len(pnl_columns) > 1:
                            best_col = None
                            best_total_return = -float('inf')
                            
                            for col in pnl_columns:
                                total_return = pnl_df[col].iloc[-1] - pnl_df[col].iloc[0]
                                if total_return > best_total_return:
                                    best_total_return = total_return
                                    best_col = col
                            
                            # Keep only the best column and rename to 'pnl'
                            pnl_df = pnl_df[[best_col]].rename(columns={best_col: 'pnl'})
                            logger.debug(f"Selected {best_col} for {alpha_id} (total return: {best_total_return:.2f})")
                        
                        return pnl_df
                    
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.error(f"Error parsing PNL data for {alpha_id}: {e}")
                        return None
                
                else:
                    logger.error(f"API error for {alpha_id}: {result.status_code} - {result.text[:200]}")
                    return None
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Final attempt failed for {alpha_id}: {e}")
                    return None
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for {alpha_id}: {e}, retrying...")
                    time.sleep(2)
                    continue
        
        return None
    
    def _calculate_returns_correlation(self, alpha1_df: pd.DataFrame, 
                                     alpha2_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate correlation between two alpha PNL DataFrames using returns.
        Based on calculate_returns_correlation from calculate_cross_correlation.py
        """
        try:
            # Find common dates
            common_dates = alpha1_df.index.intersection(alpha2_df.index)
            
            if len(common_dates) < 20:
                return None
            
            # Extract PNL values for common dates
            alpha1_pnl = alpha1_df.loc[common_dates, 'pnl'].values
            alpha2_pnl = alpha2_df.loc[common_dates, 'pnl'].values
            
            # Use the core correlation function
            return self.calculate_alpha_correlation_fast(alpha1_pnl, alpha2_pnl)
        
        except Exception as e:
            logger.debug(f"Error in returns correlation calculation: {e}")
            return None
    
    def _export_correlations_to_csv(self, correlations: Dict[str, float], 
                                   output_path: str) -> None:
        """Export correlation results to CSV file."""
        try:
            # Create DataFrame with results
            df = pd.DataFrame({
                'alpha_id': list(correlations.keys()),
                'max_correlation': list(correlations.values())
            })
            
            # Sort by correlation descending
            df = df.sort_values('max_correlation', ascending=False)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Exported correlation results to: {output_path}")
        
        except Exception as e:
            logger.error(f"Error exporting correlations to CSV: {e}")
            raise