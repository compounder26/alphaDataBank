"""
Analysis operations for alpha expression analysis.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import text
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.schema import get_connection, get_region_id, initialize_analysis_database
from analysis.alpha_expression_parser import AlphaExpressionParser

logger = logging.getLogger(__name__)

class AnalysisOperations:
    """Handles database operations for alpha expression analysis."""
    
    def __init__(self, operators_file: str, datafields_file: str):
        """
        Initialize with parser.
        
        Args:
            operators_file: Path to operators.txt
            datafields_file: Path to datafields CSV
        """
        self.parser = AlphaExpressionParser(operators_file, datafields_file)
    
    def get_alphas_for_analysis(self, region: Optional[str] = None, 
                               universe: Optional[str] = None,
                               delay: Optional[int] = None,
                               date_from: Optional[str] = None,
                               date_to: Optional[str] = None) -> pd.DataFrame:
        """
        Get alphas from database for analysis.
        
        Args:
            region: Filter by region
            universe: Filter by universe
            delay: Filter by delay
            date_from: Filter by date added (from this date)
            date_to: Filter by date added (to this date)
            
        Returns:
            DataFrame with alpha data
        """
        try:
            db_engine = get_connection()
            with db_engine.connect() as connection:
                # Build dynamic query with filters
                query = """
                SELECT 
                    a.alpha_id, 
                    a.code, 
                    a.universe, 
                    a.delay, 
                    a.alpha_type,
                    a.neutralization,
                    a.date_added,
                    r.region_name
                FROM alphas a
                JOIN regions r ON a.region_id = r.region_id
                WHERE a.code IS NOT NULL AND a.code != ''
                """
                
                params = {}
                
                if region:
                    query += " AND r.region_name = :region"
                    params['region'] = region
                
                if universe:
                    query += " AND a.universe = :universe"
                    params['universe'] = universe
                
                if delay is not None:
                    query += " AND a.delay = :delay"
                    params['delay'] = delay
                
                if date_from:
                    query += " AND a.date_added >= :date_from"
                    params['date_from'] = date_from
                
                if date_to:
                    query += " AND a.date_added <= :date_to"
                    params['date_to'] = date_to
                
                
                query += " ORDER BY a.alpha_id"
                
                df = pd.read_sql(text(query), connection, params=params)
                logger.info(f"Retrieved {len(df)} alphas for analysis")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving alphas for analysis: {e}")
            raise
    
    def update_analysis_cache(self, alphas_df: pd.DataFrame) -> None:
        """
        Update analysis cache for given alphas.
        
        Args:
            alphas_df: DataFrame with alpha data
        """
        try:
            # Parse expressions
            analysis_results = self.parser.analyze_alpha_batch(alphas_df)
            
            db_engine = get_connection()
            with db_engine.connect() as connection:
                with connection.begin():
                    # Update cache for each alpha
                    for alpha_id, metadata in analysis_results['alpha_metadata'].items():
                        # Upsert into cache table
                        upsert_stmt = text("""
                        INSERT INTO alpha_analysis_cache (
                            alpha_id, operators_unique, operators_nominal, 
                            datafields_unique, datafields_nominal, last_updated
                        ) VALUES (
                            :alpha_id, :operators_unique, :operators_nominal,
                            :datafields_unique, :datafields_nominal, :last_updated
                        )
                        ON CONFLICT (alpha_id) DO UPDATE SET
                            operators_unique = EXCLUDED.operators_unique,
                            operators_nominal = EXCLUDED.operators_nominal,
                            datafields_unique = EXCLUDED.datafields_unique,
                            datafields_nominal = EXCLUDED.datafields_nominal,
                            last_updated = EXCLUDED.last_updated
                        """)
                        
                        connection.execute(upsert_stmt, {
                            'alpha_id': alpha_id,
                            'operators_unique': json.dumps(metadata['operators_unique']),
                            'operators_nominal': json.dumps(metadata['operators_nominal']),
                            'datafields_unique': json.dumps(metadata['datafields_unique']),
                            'datafields_nominal': json.dumps(metadata['datafields_nominal']),
                            'last_updated': datetime.now()
                        })
            
            logger.info(f"Updated analysis cache for {len(analysis_results['alpha_metadata'])} alphas")
            
        except Exception as e:
            logger.error(f"Error updating analysis cache: {e}")
            raise
    
    def get_analysis_summary(self, region: Optional[str] = None,
                            universe: Optional[str] = None,
                            delay: Optional[int] = None,
                            date_from: Optional[str] = None,
                            date_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Get analysis summary with filters.
        
        Args:
            region: Filter by region
            universe: Filter by universe  
            delay: Filter by delay
            date_from: Filter by date added (from this date)
            date_to: Filter by date added (to this date)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Get filtered alphas
            alphas_df = self.get_alphas_for_analysis(region, universe, delay, date_from, date_to)
            
            if alphas_df.empty:
                return {'operators': {}, 'datafields': {}, 'metadata': {'total_alphas': 0}}
            
            # Check cache status
            db_engine = get_connection()
            with db_engine.connect() as connection:
                # Get cached analysis data
                cache_query = text("""
                SELECT alpha_id, operators_unique, operators_nominal,
                       datafields_unique, datafields_nominal, last_updated
                FROM alpha_analysis_cache
                WHERE alpha_id = ANY(:alpha_ids)
                """)
                
                alpha_ids = alphas_df['alpha_id'].tolist()
                try:
                    cache_df = pd.read_sql(cache_query, connection, params={'alpha_ids': alpha_ids})
                except Exception as sql_error:
                    # Check if it's the missing table error
                    if 'alpha_analysis_cache' in str(sql_error) and 'does not exist' in str(sql_error):
                        logger.info("Analysis cache table doesn't exist. Initializing analysis database schema...")
                        initialize_analysis_database()
                        logger.info("Analysis database schema initialized successfully. Retrying operation...")
                        # Retry the operation now that the table exists
                        return self.get_analysis_summary(region, universe, delay)
                    else:
                        raise sql_error
                
                # Find alphas not in cache or with stale cache
                cached_alpha_ids = set(cache_df['alpha_id'].tolist())
                missing_alphas = alphas_df[~alphas_df['alpha_id'].isin(cached_alpha_ids)]
                
                # Update cache for missing alphas
                if not missing_alphas.empty:
                    logger.info(f"Updating cache for {len(missing_alphas)} missing alphas")
                    self.update_analysis_cache(missing_alphas)
                    
                    # Re-fetch cache data
                    cache_df = pd.read_sql(cache_query, connection, params={'alpha_ids': alpha_ids})
            
            # Process cached data into analysis results format
            results = self._process_cached_results(cache_df, alphas_df)
            
            logger.info(f"Generated analysis summary for {len(alphas_df)} alphas")
            return results
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            raise
    
    def _process_cached_results(self, cache_df: pd.DataFrame, alphas_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process cached results into analysis format.
        
        Args:
            cache_df: DataFrame with cached analysis data
            alphas_df: DataFrame with alpha metadata
            
        Returns:
            Analysis results dictionary
        """
        results = {
            'operators': {
                'unique_usage': {},
                'nominal_usage': {},
                'top_operators': [],
                'alpha_breakdown': {}
            },
            'datafields': {
                'unique_usage': {},
                'nominal_usage': {},
                'by_category': {},
                'top_datafields': [],
                'alpha_breakdown': {}
            },
            'metadata': {
                'total_alphas': len(alphas_df),
                'regions': alphas_df['region_name'].value_counts().to_dict(),
                'universes': alphas_df['universe'].value_counts().to_dict(),
                'delays': alphas_df['delay'].value_counts().to_dict(),
                'neutralizations': alphas_df['neutralization'].value_counts().to_dict() if 'neutralization' in alphas_df.columns else {},
                'min_date_added': alphas_df['date_added'].min().strftime('%Y-%m-%d') if 'date_added' in alphas_df.columns and not alphas_df['date_added'].isna().all() else None,
                'max_date_added': alphas_df['date_added'].max().strftime('%Y-%m-%d') if 'date_added' in alphas_df.columns and not alphas_df['date_added'].isna().all() else None
            }
        }
        
        # Aggregate from cached data
        operator_unique_usage = {}
        operator_nominal_usage = {}
        datafield_unique_usage = {}
        datafield_nominal_usage = {}
        datafield_categories = {}
        
        for _, row in cache_df.iterrows():
            alpha_id = row['alpha_id']
            
            # Parse JSON data - handle both string and list cases
            ops_unique = row['operators_unique'] if isinstance(row['operators_unique'], list) else (json.loads(row['operators_unique']) if row['operators_unique'] else [])
            ops_nominal = row['operators_nominal'] if isinstance(row['operators_nominal'], dict) else (json.loads(row['operators_nominal']) if row['operators_nominal'] else {})
            dfs_unique = row['datafields_unique'] if isinstance(row['datafields_unique'], list) else (json.loads(row['datafields_unique']) if row['datafields_unique'] else [])
            dfs_nominal = row['datafields_nominal'] if isinstance(row['datafields_nominal'], dict) else (json.loads(row['datafields_nominal']) if row['datafields_nominal'] else {})
            
            # Store alpha breakdown
            results['operators']['alpha_breakdown'][alpha_id] = ops_nominal
            results['datafields']['alpha_breakdown'][alpha_id] = dfs_nominal
            
            # Aggregate operators
            for op in ops_unique:
                if op not in operator_unique_usage:
                    operator_unique_usage[op] = []
                operator_unique_usage[op].append(alpha_id)
            
            for op, count in ops_nominal.items():
                operator_nominal_usage[op] = operator_nominal_usage.get(op, 0) + count
            
            # Aggregate datafields
            for df in dfs_unique:
                if df not in datafield_unique_usage:
                    datafield_unique_usage[df] = []
                datafield_unique_usage[df].append(alpha_id)
                
                # Get category
                if df in self.parser.datafields:
                    category = self.parser.datafields[df].get('data_category', 'unknown')
                    if category not in datafield_categories:
                        datafield_categories[category] = {}
                    if df not in datafield_categories[category]:
                        datafield_categories[category][df] = []
                    datafield_categories[category][df].append(alpha_id)
            
            for df, count in dfs_nominal.items():
                datafield_nominal_usage[df] = datafield_nominal_usage.get(df, 0) + count
        
        # Convert to final format
        results['operators']['unique_usage'] = operator_unique_usage
        results['operators']['nominal_usage'] = operator_nominal_usage
        results['operators']['top_operators'] = sorted(
            [(op, len(alphas)) for op, alphas in operator_unique_usage.items()],
            key=lambda x: x[1], reverse=True
        )  # Return ALL operators, let frontend handle display limits
        
        results['datafields']['unique_usage'] = datafield_unique_usage
        results['datafields']['nominal_usage'] = datafield_nominal_usage
        results['datafields']['by_category'] = datafield_categories
        results['datafields']['top_datafields'] = sorted(
            [(df, len(alphas)) for df, alphas in datafield_unique_usage.items()],
            key=lambda x: x[1], reverse=True
        )  # Return ALL datafields, let frontend handle display limits
        
        # Pre-process dataset mappings for fast lookup
        dataset_to_datafields = {}
        datafield_to_dataset = {}
        for df, info in self.parser.datafields.items():
            dataset_id = info.get('dataset_id')
            if dataset_id and df in datafield_unique_usage:  # Only include used datafields
                if dataset_id not in dataset_to_datafields:
                    dataset_to_datafields[dataset_id] = []
                dataset_to_datafields[dataset_id].append(df)
                datafield_to_dataset[df] = dataset_id
        
        results['datafields']['dataset_mappings'] = {
            'dataset_to_datafields': dataset_to_datafields,
            'datafield_to_dataset': datafield_to_dataset
        }
        
        return results
    
    def clear_analysis_cache(self, alpha_ids: Optional[List[str]] = None) -> None:
        """
        Clear analysis cache.
        
        Args:
            alpha_ids: Specific alphas to clear (None for all)
        """
        try:
            db_engine = get_connection()
            with db_engine.connect() as connection:
                with connection.begin():
                    if alpha_ids:
                        delete_stmt = text("DELETE FROM alpha_analysis_cache WHERE alpha_id = ANY(:alpha_ids)")
                        connection.execute(delete_stmt, {'alpha_ids': alpha_ids})
                        logger.info(f"Cleared cache for {len(alpha_ids)} specific alphas")
                    else:
                        delete_stmt = text("DELETE FROM alpha_analysis_cache")
                        connection.execute(delete_stmt)
                        logger.info("Cleared entire analysis cache")
        except Exception as e:
            logger.error(f"Error clearing analysis cache: {e}")
            raise