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
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

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
import requests
import json
from api.auth import get_authenticated_session
from api.alpha_fetcher import get_robust_session

logger = logging.getLogger(__name__)

class CorrelationEngine:
    """
    Unified correlation calculation engine supporting all alpha correlation scenarios.
    """
    
    def __init__(self, use_cython: bool = True):
        """
        Initialize the correlation engine.
        
        Args:
            use_cython: Whether to use Cython acceleration (default: True)
        """
        self.use_cython = use_cython
        self._load_correlation_function()
    
    def _load_correlation_function(self):
        """Load the correlation calculation function (Cython or Python fallback)."""
        if self.use_cython:
            try:
                # Try to import from the project root directory
                import sys
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                import correlation_utils
                self.calculate_alpha_correlation_fast = correlation_utils.calculate_alpha_correlation_fast
                logger.info("Using Cython-accelerated correlation calculations")
            except ImportError:
                logger.warning("Cython extension 'correlation_utils' not found. Using Python fallback.")
                self.calculate_alpha_correlation_fast = self._python_correlation_fallback
        else:
            self.calculate_alpha_correlation_fast = self._python_correlation_fallback
            logger.info("Using Python correlation calculations")
    
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
    
    def calculate_batch_submitted(self, region: str, max_workers: int = 20, 
                                window_days: int = 400) -> None:
        """
        Calculate correlations for submitted alphas in a region.
        This replaces the logic from update_correlations_optimized.py
        """
        start_time = time.time()
        logger.info(f"Starting batch correlation calculation for submitted alphas in region {region}")
        
        try:
            # Get all alpha IDs for the region
            alpha_ids = get_all_alpha_ids_by_region_basic(region)
            if not alpha_ids:
                logger.warning(f"No alphas found for region {region}")
                return
            
            logger.info(f"Processing correlations for {len(alpha_ids)} alphas in region {region}")
            
            # Get PNL data for all alphas
            logger.info("Fetching PNL data for all alphas...")
            alpha_pnl_dict = get_pnl_data_for_alphas(alpha_ids, region)
            
            if not alpha_pnl_dict:
                logger.warning("No PNL data found for any alphas")
                return
            
            # Log which alphas were skipped and why
            skipped_alphas = [alpha_id for alpha_id in alpha_ids if alpha_id not in alpha_pnl_dict]
            if skipped_alphas:
                logger.warning(f"Skipped {len(skipped_alphas)} alphas due to missing/insufficient PNL data: {skipped_alphas}")
            
            logger.info(f"Successfully loaded PNL data for {len(alpha_pnl_dict)} alphas")
            
            # Calculate correlations in parallel
            correlation_results = {}
            alphas_with_pnl = [alpha_id for alpha_id in alpha_ids if alpha_id in alpha_pnl_dict]
            
            logger.info(f"Starting correlation calculation for {len(alphas_with_pnl)} alphas with PNL data")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_alpha = {
                    executor.submit(
                        self._calculate_correlations_for_alpha, 
                        alpha_id, alpha_pnl_dict, alpha_ids, window_days
                    ): alpha_id for alpha_id in alphas_with_pnl
                }
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_alpha):
                    alpha_id = future_to_alpha[future]
                    try:
                        correlations = future.result()
                        if correlations:
                            correlation_results[alpha_id] = {
                                'min': min(correlations),
                                'max': max(correlations),
                                'avg': np.mean(correlations),
                                'median': np.median(correlations)
                            }
                        
                        completed_count += 1
                        # More frequent progress updates and clearer messaging
                        if completed_count % 25 == 0 or completed_count == len(alphas_with_pnl):
                            logger.info(f"Completed correlations for {completed_count}/{len(alphas_with_pnl)} alphas")
                            
                    except Exception as e:
                        logger.error(f"Error processing correlations for alpha {alpha_id}: {e}")
            
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
                        'min_corr': stats['min'],
                        'max_corr': stats['max'],
                        'avg_corr': stats['avg'],
                        'median_corr': stats['median']
                    })
                
                connection.commit()
                
        except Exception as e:
            logger.error(f"Error storing correlation results: {e}")
            raise
    
    def calculate_unsubmitted_vs_submitted(self, region: str, max_workers: int = 20) -> None:
        """
        Calculate correlations between unsubmitted alphas and submitted alphas.
        For each unsubmitted alpha, finds the maximum correlation with any submitted alpha.
        This replaces the logic from calculate_unsubmitted_correlations.py
        """
        start_time = time.time()
        logger.info(f"Starting unsubmitted vs submitted correlation calculation for region {region}")
        
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
            
            # Process in parallel
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
                        if completed_count % 50 == 0:
                            logger.info(f"Completed {completed_count}/{len(unsubmitted_alpha_ids)} unsubmitted alphas")
                    
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