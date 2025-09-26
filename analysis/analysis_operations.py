"""
Analysis operations for alpha expression analysis.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import text
import logging
import os
import sys
from functools import lru_cache

# Setup project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bootstrap import setup_project_path
setup_project_path()

from database.schema import get_connection, get_region_id, initialize_analysis_database
from analysis.alpha_expression_parser import AlphaExpressionParser

logger = logging.getLogger(__name__)

class AnalysisOperations:
    """Handles database operations for alpha expression analysis."""
    
    def __init__(self, operators_file: str, operators_list: Optional[List[str]] = None,
                 available_datafields_list: Optional[List[str]] = None):
        """
        Initialize with parser.

        Args:
            operators_file: Path to operators.txt (or JSON file for dynamic data)
            operators_list: Optional list of operators to use directly (overrides file)
            available_datafields_list: Optional list of datafields available to user's tier
        """
        # Build region-datafield map if we can
        region_datafields_map = self._build_region_datafields_map()

        self.parser = AlphaExpressionParser(operators_file, operators_list, available_datafields_list,
                                           region_datafields_map)
        self._db_engine = None  # Cache database engine
        self._datafield_cache = {}  # Cache datafield availability data
    
    def _get_db_engine(self):
        """Get cached database engine or create new one if needed."""
        if self._db_engine is None:
            self._db_engine = get_connection()
        return self._db_engine

    def _build_region_datafields_map(self) -> Dict[Tuple[str, str], bool]:
        """
        Build a map of (region, datafield_id) -> available from database.
        This enables region-aware datafield filtering.
        """
        region_datafields_map = {}
        try:
            db_engine = get_connection()
            with db_engine.connect() as connection:
                # Get all available datafields with their regions
                query = text("""
                    SELECT DISTINCT datafield_id, region
                    FROM datafields
                    WHERE datafield_id IS NOT NULL
                """)
                result = connection.execute(query)

                for row in result:
                    datafield_id = row.datafield_id
                    region = row.region
                    if datafield_id and region:
                        # Mark this (region, datafield) combo as available
                        region_datafields_map[(region, datafield_id.lower())] = True

                logger.info(f"Built region-datafield map with {len(region_datafields_map)} entries")

                # Log available regions
                regions = set(region for region, _ in region_datafields_map.keys())
                logger.info(f"Available regions with datafields: {sorted(regions)}")

        except Exception as e:
            logger.warning(f"Could not build region-datafield map: {e}")
            # Return empty map on error
            return {}

        return region_datafields_map
    
    def _get_cached_datafield_availability(self, selected_data_type: Optional[str] = None):
        """Get cached datafield availability data, with optional type filtering."""
        cache_key = f"datafield_availability_{selected_data_type or 'all'}"
        
        if cache_key not in self._datafield_cache:
            logger.info(f"Loading datafield availability into cache for type: {selected_data_type or 'all'}")
            
            # Load from database
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                data_type_filter = ""
                params = {}
                
                if selected_data_type and selected_data_type != 'all':
                    data_type_filter = "AND data_type = :data_type"
                    params['data_type'] = selected_data_type
                
                datafields_query = text(f"""
                    SELECT datafield_id, data_description, region, data_type, delay
                    FROM datafields
                    WHERE datafield_id IS NOT NULL AND datafield_id != ''
                    {data_type_filter}
                    ORDER BY datafield_id
                """)
                
                datafields_result = connection.execute(datafields_query, params)
                
                datafield_availability = {}
                datafield_descriptions = {}
                
                for row in datafields_result:
                    df_id = row.datafield_id
                    region = row.region or ''
                    description = row.data_description or ''
                    data_type = row.data_type or ''
                    delay = row.delay if row.delay is not None else 0
                    
                    if df_id and region:
                        if df_id not in datafield_availability:
                            datafield_availability[df_id] = {}
                        datafield_availability[df_id][region] = {
                            'data_type': data_type,
                            'delay': delay,
                            'description': description
                        }
                        datafield_descriptions[df_id] = description
                
                # Cache the results
                self._datafield_cache[cache_key] = {
                    'availability': datafield_availability,
                    'descriptions': datafield_descriptions
                }
                
                logger.info(f"Cached {len(datafield_availability)} datafields for type: {selected_data_type or 'all'}")
        
        return self._datafield_cache[cache_key]
    
    def clear_datafield_cache(self):
        """Clear the datafield availability cache."""
        self._datafield_cache.clear()
        logger.info("Cleared datafield availability cache")
    
    def _get_alphas_with_connection(self, connection, region: Optional[str] = None, 
                                   universe: Optional[str] = None,
                                   delay: Optional[int] = None,
                                   date_from: Optional[str] = None,
                                   date_to: Optional[str] = None) -> pd.DataFrame:
        """
        Get alphas from database for analysis using an existing connection.
        
        Args:
            connection: SQLAlchemy connection object
            region: Filter by region
            universe: Filter by universe
            delay: Filter by delay
            date_from: Filter by date added (from this date)
            date_to: Filter by date added (to this date)
            
        Returns:
            DataFrame with alpha data
        """
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
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                return self._get_alphas_with_connection(connection, region, universe, delay, date_from, date_to)
                
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
            
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                with connection.begin():
                    # Update cache for each alpha
                    for alpha_id, metadata in analysis_results['alpha_metadata'].items():
                        # Upsert into cache table
                        upsert_stmt = text("""
                        INSERT INTO alpha_analysis_cache (
                            alpha_id, operators_unique, operators_nominal, 
                            datafields_unique, datafields_nominal, last_updated,
                            excluded, exclusion_reason
                        ) VALUES (
                            :alpha_id, :operators_unique, :operators_nominal,
                            :datafields_unique, :datafields_nominal, :last_updated,
                            :excluded, :exclusion_reason
                        )
                        ON CONFLICT (alpha_id) DO UPDATE SET
                            operators_unique = EXCLUDED.operators_unique,
                            operators_nominal = EXCLUDED.operators_nominal,
                            datafields_unique = EXCLUDED.datafields_unique,
                            datafields_nominal = EXCLUDED.datafields_nominal,
                            last_updated = EXCLUDED.last_updated,
                            excluded = EXCLUDED.excluded,
                            exclusion_reason = EXCLUDED.exclusion_reason
                        """)
                        
                        connection.execute(upsert_stmt, {
                            'alpha_id': alpha_id,
                            'operators_unique': json.dumps(metadata['operators_unique']),
                            'operators_nominal': json.dumps(metadata['operators_nominal']),
                            'datafields_unique': json.dumps(metadata['datafields_unique']),
                            'datafields_nominal': json.dumps(metadata['datafields_nominal']),
                            'last_updated': datetime.now(),
                            'excluded': metadata.get('excluded', False),
                            'exclusion_reason': metadata.get('exclusion_reason', None)
                        })
            
            logger.info(f"Updated analysis cache for {len(analysis_results['alpha_metadata'])} alphas")
            
        except Exception as e:
            logger.error(f"Error updating analysis cache: {e}")
            raise
    
    def get_analysis_summary(self, region: Optional[str] = None,
                            universe: Optional[str] = None,
                            delay: Optional[int] = None,
                            date_from: Optional[str] = None,
                            date_to: Optional[str] = None,
                            _retry_count: int = 0) -> Dict[str, Any]:
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
        # Prevent infinite recursion 
        if _retry_count > 1:
            logger.error("Maximum retries exceeded in get_analysis_summary")
            return {'operators': {}, 'datafields': {}, 'metadata': {'total_alphas': 0, 'error': 'Max retries exceeded'}}
            
        try:
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                # Get filtered alphas using the same connection
                alphas_df = self._get_alphas_with_connection(connection, region, universe, delay, date_from, date_to)
                
                if alphas_df.empty:
                    return {'operators': {}, 'datafields': {}, 'metadata': {'total_alphas': 0}}
                
                # Get cached analysis data
                cache_query = text("""
                SELECT alpha_id, operators_unique, operators_nominal,
                       datafields_unique, datafields_nominal, last_updated,
                       excluded, exclusion_reason
                FROM alpha_analysis_cache
                WHERE alpha_id = ANY(:alpha_ids)
                """)
                
                alpha_ids = alphas_df['alpha_id'].tolist()
                try:
                    cache_df = pd.read_sql(cache_query, connection, params={'alpha_ids': alpha_ids})
                except Exception as sql_error:
                    # Check if it's the missing table error and we haven't retried yet
                    if ('alpha_analysis_cache' in str(sql_error) and 'does not exist' in str(sql_error) 
                        and _retry_count == 0):
                        logger.info("Analysis cache table doesn't exist. Initializing analysis database schema...")
                        # Close current connection before initializing
                        connection.close()
                        initialize_analysis_database()
                        logger.info("Analysis database schema initialized successfully. Retrying operation...")
                        # Retry the operation now that the table exists (only once)
                        return self.get_analysis_summary(region, universe, delay, date_from, date_to, _retry_count=1)
                    else:
                        logger.error(f"Database error in get_analysis_summary: {sql_error}")
                        # Return empty result instead of crashing on repeated failures
                        return {'operators': {}, 'datafields': {}, 'metadata': {'total_alphas': 0, 'error': str(sql_error)}}
                
                # Find alphas not in cache or with stale cache
                cached_alpha_ids = set(cache_df['alpha_id'].tolist())
                missing_alphas = alphas_df[~alphas_df['alpha_id'].isin(cached_alpha_ids)]
                
                # Update cache for missing alphas
                if not missing_alphas.empty:
                    logger.info(f"Updating cache for {len(missing_alphas)} missing alphas")
                    self.update_analysis_cache(missing_alphas)
                    
                    # Re-fetch cache data with same connection
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
        # Include ALL alphas regardless of tier restrictions
        # Users can see alphas even if they contain operators/datafields not in their tier
        included_cache = cache_df.copy()
        excluded_count = 0  # We're not excluding any alphas now
        
        # Filter corresponding alphas_df to only include non-excluded
        included_alpha_ids = set(included_cache['alpha_id'].tolist())
        included_alphas_df = alphas_df[alphas_df['alpha_id'].isin(included_alpha_ids)].copy()
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
                'total_alphas': len(included_alphas_df),  # Only count included alphas
                'excluded_alphas': excluded_count,
                'total_processed': len(alphas_df),
                'regions': included_alphas_df['region_name'].value_counts().to_dict(),
                'universes': included_alphas_df['universe'].value_counts().to_dict(),
                'delays': included_alphas_df['delay'].value_counts().to_dict(),
                'neutralizations': included_alphas_df['neutralization'].value_counts().to_dict() if 'neutralization' in included_alphas_df.columns else {},
                'min_date_added': included_alphas_df['date_added'].min().strftime('%Y-%m-%d') if 'date_added' in included_alphas_df.columns and not included_alphas_df['date_added'].isna().all() else None,
                'max_date_added': included_alphas_df['date_added'].max().strftime('%Y-%m-%d') if 'date_added' in included_alphas_df.columns and not included_alphas_df['date_added'].isna().all() else None
            }
        }
        
        # Aggregate from cached data
        operator_unique_usage = {}
        operator_nominal_usage = {}
        datafield_unique_usage = {}
        datafield_nominal_usage = {}
        datafield_categories = {}
        datafield_region_pairs = set()  # Track unique (datafield, region) pairs
        
        # Create mapping of alpha_id to region, delay, and universe for quick lookup
        alpha_region_map = {}
        alpha_delay_map = {}
        alpha_universe_map = {}
        if 'region_name' in included_alphas_df.columns:
            alpha_region_map = dict(zip(included_alphas_df['alpha_id'], included_alphas_df['region_name']))
        if 'delay' in included_alphas_df.columns:
            alpha_delay_map = dict(zip(included_alphas_df['alpha_id'], included_alphas_df['delay']))
        if 'universe' in included_alphas_df.columns:
            alpha_universe_map = dict(zip(included_alphas_df['alpha_id'], included_alphas_df['universe']))
        
        for _, row in included_cache.iterrows():
            alpha_id = row['alpha_id']
            alpha_region = alpha_region_map.get(alpha_id, None)
            alpha_delay = alpha_delay_map.get(alpha_id, None)
            alpha_universe = alpha_universe_map.get(alpha_id, None)
            
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
                
                # Track unique (datafield, region) pairs
                if alpha_region:
                    datafield_region_pairs.add((df, alpha_region))
                
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
        results['datafields']['region_specific_count'] = len(datafield_region_pairs)  # Count unique datafield-region combinations
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
    
    def get_datafield_recommendations(self, selected_region: Optional[str] = None, selected_data_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get datafield recommendations based on submitted alphas.
        Identifies datafields available in multiple regions and recommends where they could be used.
        
        Args:
            selected_region: Filter recommendations for a specific region
            selected_data_type: Filter by datafield type (MATRIX, VECTOR, GROUP)
            
        Returns:
            Dictionary containing:
            - datafield_usage: Dict mapping datafield_id to regions where used
            - datafield_availability: Dict mapping datafield_id to regions where available
            - recommendations: List of recommendation dicts with datafield info and suggested regions
        """
        try:
            # Get database connection
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                # Query to get datafield usage by region
                # Note: We intentionally don't filter by ac.excluded to show all alphas
                # Users can see alphas even if they contain operators/datafields not in their tier
                query = text("""
                    SELECT
                        r.region_name,
                        a.alpha_id,
                        a.universe,
                        a.delay,
                        ac.datafields_unique
                    FROM alpha_analysis_cache ac
                    JOIN alphas a ON ac.alpha_id = a.alpha_id
                    JOIN regions r ON a.region_id = r.region_id
                    WHERE ac.datafields_unique IS NOT NULL
                    AND a.alpha_type IN ('REGULAR', 'SUPER')
                """)
                
                result = connection.execute(query)
                rows = result.fetchall()
            
            # Process datafield usage by region
            datafield_usage_by_region = {}  # {datafield_id: {region: [alpha_ids]}}
            
            for row in rows:
                region = row.region_name
                alpha_id = row.alpha_id
                datafields_json = row.datafields_unique
                
                if datafields_json:
                    datafields = json.loads(datafields_json) if isinstance(datafields_json, str) else datafields_json
                    for df_id in datafields:
                        if df_id not in datafield_usage_by_region:
                            datafield_usage_by_region[df_id] = {}
                        if region not in datafield_usage_by_region[df_id]:
                            datafield_usage_by_region[df_id][region] = []
                        datafield_usage_by_region[df_id][region].append(alpha_id)
            
            # Get cached datafield availability data
            try:
                cached_data = self._get_cached_datafield_availability(selected_data_type)
                datafield_availability = cached_data['availability']
                datafield_descriptions = cached_data['descriptions']
                
                logger.info(f"Using cached datafield data: {len(datafield_availability)} datafields")
                
            except Exception as db_error:
                logger.error(f"Failed to load datafield availability: {db_error}")
                raise Exception("Cannot generate datafield recommendations without database access. Please ensure datafields are populated.")
            
            # Also build description-based mapping for fields with same description
            description_to_datafields = {}  # {description: [datafield_ids]}
            for df_id, desc in datafield_descriptions.items():
                if desc and desc != "No field description":
                    if desc not in description_to_datafields:
                        description_to_datafields[desc] = []
                    description_to_datafields[desc].append(df_id)
            
            # Generate recommendations
            recommendations = []
            
            for df_id, regions_used in datafield_usage_by_region.items():
                # Get all regions where this datafield is available
                available_regions = set(datafield_availability.get(df_id, {}).keys())
                matching_datafields = {df_id: available_regions.copy()}
                
                # Also check for same description fields
                description = datafield_descriptions.get(df_id, '')
                if description and description != "No field description":
                    # Find all datafields with same description
                    similar_datafields = description_to_datafields.get(description, [])
                    for similar_df in similar_datafields:
                        if similar_df != df_id:  # Don't include the same datafield
                            similar_regions = set(datafield_availability.get(similar_df, {}).keys())
                            available_regions.update(similar_regions)
                            if similar_regions:
                                matching_datafields[similar_df] = similar_regions
                
                # Find regions where available but not used
                used_regions = set(regions_used.keys())
                recommended_regions = available_regions - used_regions
                
                if recommended_regions:
                    # Count total alphas using this datafield
                    total_alphas = sum(len(alphas) for alphas in regions_used.values())
                    
                    # Build detailed availability info showing which datafield IDs are available in each region
                    availability_details = {}
                    for rec_region in recommended_regions:
                        availability_details[rec_region] = []
                        for match_df, match_regions in matching_datafields.items():
                            if rec_region in match_regions:
                                availability_details[rec_region].append(match_df)
                    
                    # Get data_type from the first region where this datafield is available
                    data_type = 'Unknown'
                    for region_info in datafield_availability.get(df_id, {}).values():
                        data_type = region_info.get('data_type', 'Unknown')
                        break
                    
                    recommendation = {
                        'datafield_id': df_id,
                        'description': datafield_descriptions.get(df_id, 'No description'),
                        'data_type': data_type,
                        'used_in_regions': list(used_regions),
                        'available_in_regions': list(available_regions),
                        'recommended_regions': list(recommended_regions),
                        'alpha_count': total_alphas,
                        'usage_details': {region: len(alphas) for region, alphas in regions_used.items()},
                        'matching_datafields': matching_datafields,  # All datafields that match (by ID or description)
                        'availability_details': availability_details  # Which specific datafield IDs are available in each recommended region
                    }
                    
                    # Filter by selected region if specified
                    if selected_region:
                        if selected_region in recommended_regions:
                            recommendations.append(recommendation)
                    else:
                        recommendations.append(recommendation)
            
            # Sort recommendations by alpha count (most used first)
            recommendations.sort(key=lambda x: x['alpha_count'], reverse=True)
            
            return {
                'recommendations': recommendations,
                'total_datafields_analyzed': len(datafield_usage_by_region),
                'total_recommendations': len(recommendations),
                'datafield_usage': datafield_usage_by_region,
                'datafield_availability': datafield_availability
            }
            
        except Exception as e:
            logger.error(f"Error getting datafield recommendations: {e}")
            return {
                'recommendations': [],
                'total_datafields_analyzed': 0,
                'total_recommendations': 0,
                'error': str(e)
            }
    
    def clear_analysis_cache(self, alpha_ids: Optional[List[str]] = None) -> None:
        """
        Clear analysis cache.
        
        Args:
            alpha_ids: Specific alphas to clear (None for all)
        """
        try:
            db_engine = self._get_db_engine()
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
    
    def get_alphas_containing_operators(self, operators: List[str], region: Optional[str] = None) -> List[str]:
        """
        Return alpha IDs containing ANY of the specified operators.
        
        Args:
            operators: List of operators to match
            region: Optional region filter
            
        Returns:
            List of alpha IDs containing any of the specified operators
        """
        if not operators:
            return []
        
        try:
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                # Build query with region filter if specified
                query = """
                SELECT DISTINCT ac.alpha_id 
                FROM alpha_analysis_cache ac
                """
                
                params = {'operators': operators}
                
                if region:
                    query += """
                    JOIN alphas a ON ac.alpha_id = a.alpha_id
                    JOIN regions r ON a.region_id = r.region_id
                    WHERE r.region_name = :region
                    AND ac.operators_unique ?| ARRAY[:operators]::text[]
                    """
                    params['region'] = region
                else:
                    query += """
                    WHERE ac.operators_unique ?| ARRAY[:operators]::text[]
                    """
                
                result = connection.execute(text(query), params)
                alpha_ids = [row.alpha_id for row in result]
                
                logger.info(f"Found {len(alpha_ids)} alphas containing operators: {operators}")
                return alpha_ids
                
        except Exception as e:
            logger.error(f"Error getting alphas containing operators: {e}")
            return []
    
    def get_alphas_containing_datafields(self, datafields: List[str], region: Optional[str] = None) -> List[str]:
        """
        Return alpha IDs containing ANY of the specified datafields.
        
        Args:
            datafields: List of datafields to match
            region: Optional region filter
            
        Returns:
            List of alpha IDs containing any of the specified datafields
        """
        if not datafields:
            return []
        
        try:
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                # Build query with region filter if specified
                query = """
                SELECT DISTINCT ac.alpha_id 
                FROM alpha_analysis_cache ac
                """
                
                params = {'datafields': datafields}
                
                if region:
                    query += """
                    JOIN alphas a ON ac.alpha_id = a.alpha_id
                    JOIN regions r ON a.region_id = r.region_id
                    WHERE r.region_name = :region
                    AND ac.datafields_unique ?| ARRAY[:datafields]::text[]
                    """
                    params['region'] = region
                else:
                    query += """
                    WHERE ac.datafields_unique ?| ARRAY[:datafields]::text[]
                    """
                
                result = connection.execute(text(query), params)
                alpha_ids = [row.alpha_id for row in result]
                
                logger.info(f"Found {len(alpha_ids)} alphas containing datafields: {datafields}")
                return alpha_ids
                
        except Exception as e:
            logger.error(f"Error getting alphas containing datafields: {e}")
            return []
    
    def get_available_operators_for_region(self, region: Optional[str] = None) -> List[str]:
        """
        Get sorted list of all operators used in the region.
        
        Args:
            region: Optional region filter
            
        Returns:
            Sorted list of all operators used in the region
        """
        try:
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                # Build query with region filter if specified
                query = """
                SELECT DISTINCT jsonb_array_elements_text(ac.operators_unique) as operator
                FROM alpha_analysis_cache ac
                """
                
                params = {}
                
                if region:
                    query += """
                    JOIN alphas a ON ac.alpha_id = a.alpha_id
                    JOIN regions r ON a.region_id = r.region_id
                    WHERE r.region_name = :region
                    """
                    params['region'] = region
                else:
                    query += """
                    """
                
                query += " ORDER BY operator"
                
                result = connection.execute(text(query), params)
                operators = [row.operator for row in result if row.operator]
                
                logger.info(f"Found {len(operators)} unique operators for region: {region or 'all'}")
                return operators
                
        except Exception as e:
            logger.error(f"Error getting available operators for region: {e}")
            return []
    
    def get_available_datafields_for_region(self, region: Optional[str] = None) -> List[str]:
        """
        Get sorted list of all datafields used in the region.
        
        Args:
            region: Optional region filter
            
        Returns:
            Sorted list of all datafields used in the region
        """
        try:
            db_engine = self._get_db_engine()
            with db_engine.connect() as connection:
                # Build query with region filter if specified
                query = """
                SELECT DISTINCT jsonb_array_elements_text(ac.datafields_unique) as datafield
                FROM alpha_analysis_cache ac
                """
                
                params = {}
                
                if region:
                    query += """
                    JOIN alphas a ON ac.alpha_id = a.alpha_id
                    JOIN regions r ON a.region_id = r.region_id
                    WHERE r.region_name = :region
                    """
                    params['region'] = region
                else:
                    query += """
                    """
                
                query += " ORDER BY datafield"
                
                result = connection.execute(text(query), params)
                datafields = [row.datafield for row in result if row.datafield]
                
                logger.info(f"Found {len(datafields)} unique datafields for region: {region or 'all'}")
                return datafields
                
        except Exception as e:
            logger.error(f"Error getting available datafields for region: {e}")
            return []