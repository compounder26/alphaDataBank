#!/usr/bin/env python3
"""
Platform Data Fetcher Module

This module fetches operators and datafields from WorldQuant Brain API
based on user's tier and permissions. Provides caching functionality
for performance optimization.
"""

import json
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
from threading import Lock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legacy.ace import start_session
from legacy.helpful_functions import get_datafields
from database.schema import get_connection
from config.api_config import (
    DATAFIELDS_DATA_TYPES, DATAFIELDS_REGIONS, DATAFIELDS_UNIVERSES, DATAFIELDS_DELAYS,
    DATAFIELDS_MAX_WORKERS, DATAFIELDS_RETRY_WAIT, DATAFIELDS_MAX_BACKOFF_SECONDS
)

# Thread-safe lock for shared data structures
data_lock = Lock()

logger = logging.getLogger(__name__)

class PlatformDataFetcher:
    """
    Fetches operators and datafields from WorldQuant Brain API 
    with caching capabilities.
    """
    
    def __init__(self, cache_dir: str = "data"):
        """
        Initialize the fetcher.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.operators_cache_file = os.path.join(cache_dir, "operators_dynamic.json")
        self.datafields_cache_file = os.path.join(cache_dir, "datafields_dynamic.csv")
        self.metadata_file = os.path.join(cache_dir, ".cache_metadata.json")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        self.brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")
        
        # Deduplication tracking for datafields (thread-safe)
        self._seen_datafields = set()  # Track seen (id, region, delay) tuples
        self._fetch_lock = Lock()  # Thread-safe access to shared deduplication data
    
    def _process_datafields_with_deduplication(self, datafields_df: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process datafields DataFrame and apply deduplication based on (id, region, delay).
        
        Args:
            datafields_df: DataFrame with datafields data
            params: Parameter dictionary with region, delay, etc.
            
        Returns:
            List of unique datafield dictionaries (deduplicated)
        """
        if datafields_df is None or datafields_df.empty:
            return []
        
        results = []
        combo_name = f"{params['data_type']}/{params['region']}/{params['universe']}/delay_{params['delay']}"
        
        with self._fetch_lock:  # Thread-safe access to deduplication data
            initial_seen_count = len(self._seen_datafields)
            
            for _, row in datafields_df.iterrows():
                try:
                    datafield_id = row.get('id', '')
                    if not datafield_id:
                        logger.debug(f"Skipping row with no ID in {combo_name}")
                        continue
                    
                    # Create unique key based on (id, region, delay)
                    unique_key = (datafield_id, params['region'], params['delay'])
                    
                    if unique_key not in self._seen_datafields:
                        # New unique datafield - add to seen set and results
                        self._seen_datafields.add(unique_key)
                        
                        # Extract core datafield information
                        result = {
                            'id': datafield_id,
                            'description': row.get('description', '') or row.get('name', '') or row.get('title', ''),
                            'dataset': row.get('dataset', {}),
                            'category': row.get('category', {}),
                            'subcategory': row.get('subcategory', {}),
                            'region': params['region'],
                            'delay': params['delay'],
                            'universe': params['universe'],
                            'type': row.get('type', ''),
                            'coverage': row.get('coverage', None),
                            'userCount': row.get('userCount', None),
                            'alphaCount': row.get('alphaCount', None),
                            'pyramidMultiplier': row.get('pyramidMultiplier', None),
                            'themes': row.get('themes', {}),
                            'data_type': params['data_type'],  # MATRIX/VECTOR/GROUP
                            'fetch_region': params['region'],
                            'fetch_universe': params['universe'],
                            'fetch_delay': params['delay'],
                            'fetch_timestamp': datetime.now(),
                        }
                        results.append(result)
                    else:
                        logger.debug(f"Skipping duplicate datafield {datafield_id} for {params['region']}/delay_{params['delay']} in {combo_name}")
                        
                except Exception as e:
                    logger.warning(f"Error processing row in {combo_name}: {e}")
                    continue
            
            new_unique_count = len(self._seen_datafields) - initial_seen_count
            
            if results:
                logger.debug(f"✅ {combo_name}: {len(results)} unique datafields (filtered {len(datafields_df) - len(results)} duplicates)")
            else:
                if len(datafields_df) > 0:
                    logger.debug(f"✅ {combo_name}: No new unique datafields ({len(datafields_df)} were duplicates)")
                else:
                    logger.debug(f"✅ {combo_name}: No datafields in DataFrame")
        
        return results
    
    def _filter_operators(self, operators_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter operators based on scope and category.
        
        Exclude operators that:
        1. Have category == "Special"
        2. Have scope containing only "COMBO" 
        3. Have scope containing only "SELECTION"
        
        Args:
            operators_data: List of operator dictionaries with name, scope, category
            
        Returns:
            Filtered list of operator dictionaries
        """
        filtered_operators = []
        excluded_count = 0
        excluded_reasons = {"Special": 0, "OnlyCOMBO": 0, "OnlySELECTION": 0}
        
        for op in operators_data:
            scope = op.get('scope', [])
            category = op.get('category', '')
            name = op.get('name', '')
            
            # Exclusion logic
            should_exclude = False
            reason = None
            
            # Check for Special category
            if category == "Special":
                should_exclude = True
                reason = "Special"
            # Check for operators with only COMBO scope
            elif scope == ["COMBO"]:
                should_exclude = True
                reason = "OnlyCOMBO"
            # Check for operators with only SELECTION scope
            elif scope == ["SELECTION"]:
                should_exclude = True
                reason = "OnlySELECTION"
            
            if should_exclude:
                excluded_count += 1
                excluded_reasons[reason] += 1
                logger.debug(f"Excluding operator '{name}' - {reason}")
            else:
                filtered_operators.append(op)
        
        logger.info(f"Operator filtering results:")
        logger.info(f"  Original operators: {len(operators_data)}")
        logger.info(f"  Excluded operators: {excluded_count}")
        logger.info(f"    Special category: {excluded_reasons['Special']}")
        logger.info(f"    Only COMBO scope: {excluded_reasons['OnlyCOMBO']}")
        logger.info(f"    Only SELECTION scope: {excluded_reasons['OnlySELECTION']}")
        logger.info(f"  Final operators: {len(filtered_operators)}")
        
        return filtered_operators
    
    def fetch_operators(self, session=None) -> List[Dict[str, Any]]:
        """
        Fetch operators from the API.
        
        Args:
            session: Optional authenticated session. If None, will create new session.
            
        Returns:
            List of operator dictionaries
        """
        if session is None:
            logger.info("Creating new session for operators fetch")
            session = start_session()
        
        try:
            logger.info("Fetching operators from /operators endpoint")
            response = session.get(f"{self.brain_api_url}/operators")
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch operators: HTTP {response.status_code} - {response.text}")
            
            operators_data = response.json()
            logger.info(f"Successfully fetched {len(operators_data)} operators")
            
            # Apply filtering to exclude unwanted operators
            filtered_operators = self._filter_operators(operators_data)
            
            return filtered_operators
            
        except Exception as e:
            logger.error(f"Error fetching operators: {e}")
            raise
    
    def fetch_datafields_comprehensive(self, session=None, max_workers=200) -> pd.DataFrame:
        """
        Fetch comprehensive datafields from the API across all regions, universes, and data types.
        Uses parallel fetching for performance optimization.
        
        Args:
            session: Optional authenticated session
            max_workers: Number of concurrent threads (default: 50)
            
        Returns:
            DataFrame with all available datafields including description column
        """
        if session is None:
            logger.info("Creating new session for comprehensive datafields fetch")
            session = start_session()
        
        # Configuration for comprehensive fetching (from config)
        DATA_TYPES = DATAFIELDS_DATA_TYPES
        REGIONS = DATAFIELDS_REGIONS
        UNIVERSES = DATAFIELDS_UNIVERSES
        delays = DATAFIELDS_DELAYS
        
        try:
            logger.info(f"Starting comprehensive datafields fetch with {max_workers} workers")
            logger.info("Optimized for dataset-based approach - fetching all data types per region/universe/delay")
            
            # Generate parameter combinations WITHOUT iterating data_types
            # The dataset-based approach in get_datafields() will fetch all data types automatically
            fetch_params = []
            for region in REGIONS:
                for universe in UNIVERSES:
                    for delay in delays:
                        # Fetch all data types in one call using dataset-based approach
                        fetch_params.append({
                            'data_types': DATA_TYPES,  # Pass all data types to be handled by dataset approach
                            'region': region, 
                            'universe': universe,
                            'delay': delay
                        })
            
            logger.info(f"📊 Total combinations to process: {len(fetch_params)} (optimized from {len(fetch_params) * len(DATA_TYPES)})")
            
            # Shared results list
            all_datafields = []
            successful_fetches = 0
            failed_fetches = 0
            
            # Use ThreadPoolExecutor for parallel fetching with enhanced workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all fetch tasks
                future_to_params = {
                    executor.submit(self._fetch_single_datafield_combo, session, params): params
                    for params in fetch_params
                }
                
                # Collect results as they complete with progress bar
                with tqdm(total=len(fetch_params), desc="Fetching datafields") as pbar:
                    for future in as_completed(future_to_params):
                        params = future_to_params[future]
                        try:
                            results_list = future.result()  # Now returns List[Dict]
                            if results_list:
                                with data_lock:  # Thread-safe access to shared list
                                    all_datafields.extend(results_list)
                                successful_fetches += 1
                                pbar.set_postfix({
                                    'Total': len(all_datafields),
                                    'Success': successful_fetches,
                                    'Failed': failed_fetches,
                                    'Current': f"{params['data_type'][:3]}/{params['region']}/{params['universe'][:6]}/d{params['delay']}"
                                })
                            else:
                                failed_fetches += 1
                                logger.debug(f"⚠️ No data returned for {params['data_type']}-{params['region']}-{params['universe']}-delay{params['delay']}")
                        except Exception as e:
                            failed_fetches += 1
                            logger.error(f"Task failed for {params}: {e}")
                        finally:
                            pbar.update(1)
            
            logger.info(f"Completed fetching: {successful_fetches} successful, {failed_fetches} failed")
            
            if not all_datafields:
                logger.warning("No datafields were collected!")
                return pd.DataFrame(columns=['id', 'description', 'dataset', 'category', 'subcategory', 'region', 'delay', 'universe', 'type', 'data_type'])
            
            # Create final DataFrame from collected results
            logger.info("Creating DataFrame from collected results...")
            combined_df = pd.DataFrame(all_datafields)
            
            # Remove duplicates based on datafield_id, delay, and region combination
            # This preserves region-specific datafields (e.g., subindustry in USA vs ASI)
            if 'id' in combined_df.columns and 'delay' in combined_df.columns and 'fetch_region' in combined_df.columns:
                initial_count = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=['id', 'delay', 'fetch_region']).reset_index(drop=True)
                final_count = len(combined_df)
                
                logger.info(f"📋 Total datafields collected: {initial_count}")
                logger.info(f"📋 Unique datafields after deduplication: {final_count}")
                logger.info(f"📋 Duplicates removed: {initial_count - final_count}")
                
                # Show statistics by data type and delay (like reference script)
                if not combined_df.empty:
                    logger.info("\n📊 Statistics by data type:")
                    if 'data_type' in combined_df.columns:
                        type_stats = combined_df.groupby('data_type').size().sort_values(ascending=False)
                        for dtype, count in type_stats.items():
                            logger.info(f"   {dtype}: {count:,} datafields")
                    
                    logger.info("\n📊 Statistics by delay:")
                    delay_stats = combined_df.groupby('delay').size().sort_values(ascending=False)
                    for delay, count in delay_stats.items():
                        logger.info(f"   Delay {delay}: {count:,} datafields")
                        
            else:
                logger.warning("Cannot deduplicate: missing 'id' or 'delay' columns")
            
            logger.info(f"🎉 Successfully fetched comprehensive datafields: {len(combined_df)} unique datafields")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error in comprehensive datafields fetch: {e}")
            raise
    
    def _fetch_single_datafield_combo(self, session, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch datafields for a single parameter combination with robust error handling.
        Now optimized to fetch all data types in one call using dataset-based approach.
        
        Args:
            session: Authenticated session
            params: Dictionary with data_types (list), region, universe, delay
            
        Returns:
            List of datafield dictionaries or empty list if permanently failed
        """
        combo_name = f"ALL_TYPES/{params['region']}/{params['universe']}/delay_{params['delay']}"
        retry_wait_seconds = DATAFIELDS_RETRY_WAIT
        max_backoff_seconds = DATAFIELDS_MAX_BACKOFF_SECONDS
        attempt = 0
        
        while True:  # Indefinite retry loop for transient errors
            attempt += 1
            try:
                logger.debug(f"Fetching {combo_name}, attempt {attempt}")
                
                # Check and refresh session if needed
                session_to_use = session
                try:
                    from legacy import ace
                    if hasattr(ace, 'check_session_and_relogin'):
                        session_to_use = ace.check_session_and_relogin(session)
                except ImportError:
                    # If ace module not available, use original session
                    pass
                except Exception as e:
                    logger.debug(f"Session refresh failed for {combo_name}: {e}, using original session")
                
                # Fetch all data types for this region/universe/delay combination
                # The dataset-based approach in get_datafields() will handle all datasets automatically
                all_results = []
                successful_types = 0
                failed_types = 0
                
                for data_type in params['data_types']:
                    try:
                        logger.debug(f"Fetching {data_type} datafields for {params['region']}/{params['universe']}/delay_{params['delay']}")
                        
                        # Use the dataset-based get_datafields function with timeout protection
                        datafields_df = get_datafields(
                            session_to_use,
                            instrument_type="EQUITY",  # Always EQUITY
                            region=params['region'],
                            delay=params['delay'],
                            universe=params['universe'],
                            theme="false",
                            dataset_id="",  # Empty = use dataset-based approach
                            data_type=data_type,  # MATRIX/VECTOR/GROUP
                            search=""
                        )
                        
                        # Create params for deduplication processing
                        type_params = {
                            'data_type': data_type,
                            'region': params['region'],
                            'universe': params['universe'],
                            'delay': params['delay']
                        }
                        
                        # Use deduplication helper to process datafields
                        type_results = self._process_datafields_with_deduplication(datafields_df, type_params)
                        all_results.extend(type_results)
                        successful_types += 1
                        
                    except KeyboardInterrupt:
                        logger.info(f"Keyboard interrupt received while fetching {data_type} for {combo_name}. Stopping.")
                        raise  # Re-raise KeyboardInterrupt immediately
                        
                    except Exception as e:
                        failed_types += 1
                        logger.warning(f"Failed to fetch {data_type} for {combo_name}: {e}")
                        # Continue to next data type, but don't fail entire combination
                        continue
                
                logger.debug(f"Completed {combo_name}: {successful_types} successful, {failed_types} failed data types")
                
                # Return results even if some data types failed
                return all_results
                    
            except KeyboardInterrupt:
                logger.info(f"Keyboard interrupt received for {combo_name}. Stopping.")
                raise  # Re-raise KeyboardInterrupt
                
            except Exception as e:
                # Categorize errors for better handling
                error_msg = str(e).lower()
                
                # Check for authentication-related errors
                if any(auth_error in error_msg for auth_error in ['authentication', 'unauthorized', '401', '403']):
                    logger.error(f"❌ Authentication error for {combo_name}: {e}. Not retrying.")
                    return []  # Stop retrying on auth errors
                
                # Check for rate limiting
                if '429' in error_msg or 'rate limit' in error_msg:
                    logger.warning(f"Rate limit hit for {combo_name}: {e}. Waiting 60s before retry...")
                    time.sleep(60)
                    continue  # Skip normal backoff for rate limits
                
                # Check for persistent connectivity issues vs no data scenarios
                error_msg_lower = str(e).lower()
                
                # If it's clearly a "no data" scenario, don't retry excessively
                if any(no_data_indicator in error_msg_lower for no_data_indicator in [
                    'no datasets found', 'empty dataframe', 'no datafields', 'no results'
                ]):
                    if attempt >= 2:  # Quick retry for no-data scenarios
                        logger.info(f"No data available for {combo_name} after {attempt} attempts: {e}")
                        return []
                
                # For connectivity/server errors, use reasonable retry limit
                if any(connectivity_error in error_msg_lower for connectivity_error in [
                    'timeout', 'connection', 'network', 'remote end closed', 'connection aborted'
                ]):
                    if attempt >= 6:  # More retries for connectivity issues
                        logger.error(f"❌ Connectivity issues persist for {combo_name} after {attempt} attempts: {e}")
                        return []
                
                # For other errors, use standard retry limit
                if attempt >= 20:  # Standard retry limit for unknown errors
                    logger.error(f"❌ Failed after {attempt} attempts for {combo_name}: {e}")
                    return []
                
                wait_duration = min(retry_wait_seconds * min(attempt, 8), max_backoff_seconds)
                logger.warning(f"❌ Attempt {attempt} failed for {combo_name}: {e}. Retrying in {wait_duration}s...")
                
                try:
                    time.sleep(wait_duration)
                except KeyboardInterrupt:
                    logger.info(f"Keyboard interrupt during retry wait for {combo_name}. Stopping.")
                    raise
    
    def fetch_datafields(self, session=None, region: str = "USA", universe: str = "TOP3000", 
                        delay: int = 1, data_type: str = 'MATRIX') -> pd.DataFrame:
        """
        Legacy method for backward compatibility. Use fetch_datafields_comprehensive for complete data.
        
        Args:
            session: Optional authenticated session
            region: Region filter
            universe: Universe filter
            delay: Delay filter
            data_type: Type of data (MATRIX/VECTOR/GROUP)
            
        Returns:
            DataFrame with datafields
        """
        if session is None:
            logger.info("Creating new session for datafields fetch")
            session = start_session()
        
        try:
            logger.info(f"Fetching datafields for data_type={data_type}, region={region}, universe={universe}, delay={delay}")
            
            # Use existing get_datafields function from helpful_functions.py
            datafields_df = get_datafields(
                session,
                instrument_type="EQUITY",  # Always EQUITY
                region=region,
                delay=delay,
                universe=universe,
                theme="false",
                dataset_id="",
                data_type=data_type,  # MATRIX/VECTOR/GROUP
                search=""
            )
            
            logger.info(f"Successfully fetched {len(datafields_df)} datafields")
            return datafields_df
            
        except Exception as e:
            logger.error(f"Error fetching datafields: {e}")
            raise
    
    def save_operators_cache(self, operators_data: List[Dict[str, Any]]) -> str:
        """
        Save operators data to cache file.
        
        Args:
            operators_data: List of operator dictionaries
            
        Returns:
            Path to saved cache file
        """
        try:
            # Create cache structure with metadata
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "count": len(operators_data),
                "operators": operators_data
            }
            
            with open(self.operators_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved {len(operators_data)} operators to cache: {self.operators_cache_file}")
            
            # Update metadata
            self._update_cache_metadata("operators", datetime.now())
            
            return self.operators_cache_file
            
        except Exception as e:
            logger.error(f"Error saving operators cache: {e}")
            raise
    
    def save_datafields_cache(self, datafields_df: pd.DataFrame) -> bool:
        """
        Save datafields DataFrame to database only.
        
        Args:
            datafields_df: DataFrame with datafields data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save to database (only storage)
            db_success = self.save_datafields_to_db(datafields_df)
            if db_success:
                logger.info("Successfully saved datafields to database")
                # Update metadata
                self._update_cache_metadata("datafields", datetime.now())
                return True
            else:
                logger.error("Failed to save datafields to database")
                return False
            
        except Exception as e:
            logger.error(f"Error saving datafields: {e}")
            raise
    
    def save_datafields_to_db(self, datafields_df: pd.DataFrame) -> bool:
        """
        Save datafields DataFrame to PostgreSQL database using slim 7-column schema.
        
        Args:
            datafields_df: DataFrame with datafields data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from sqlalchemy import text
            import json
            
            # Get database connection
            db_engine = get_connection()
            
            # Clear existing datafields (for fresh import)
            with db_engine.connect() as connection:
                with connection.begin():
                    # Clear existing data
                    connection.execute(text("DELETE FROM datafields"))
                    logger.info("Cleared existing datafields from database")
                    
                    # Prepare data for insertion
                    insert_count = 0
                    batch_size = 1000
                    
                    # Process datafields in batches for better performance
                    for i in range(0, len(datafields_df), batch_size):
                        batch_df = datafields_df.iloc[i:i+batch_size]
                        batch_data = []
                        
                        for _, row in batch_df.iterrows():
                            # Extract dataset_id from dataset field
                            dataset_id = None
                            if 'dataset' in row and pd.notna(row['dataset']):
                                if isinstance(row['dataset'], dict):
                                    dataset_id = row['dataset'].get('id')
                                elif isinstance(row['dataset'], str):
                                    try:
                                        dataset_dict = json.loads(row['dataset'])
                                        dataset_id = dataset_dict.get('id')
                                    except:
                                        pass
                            
                            # Extract data_category from category field
                            data_category = None
                            if 'category' in row and pd.notna(row['category']):
                                if isinstance(row['category'], dict):
                                    data_category = row['category'].get('name') or row['category'].get('id')
                                elif isinstance(row['category'], str):
                                    try:
                                        category_dict = json.loads(row['category'])
                                        data_category = category_dict.get('name') or category_dict.get('id')
                                    except:
                                        data_category = str(row['category'])
                            
                            batch_data.append({
                                'datafield_id': str(row.get('id', '')),
                                'dataset_id': str(dataset_id) if dataset_id else None,
                                'data_category': str(data_category) if data_category else None,
                                'data_type': str(row.get('data_type', '')) if pd.notna(row.get('data_type')) else None,
                                'delay': int(row.get('delay')) if pd.notna(row.get('delay')) and str(row.get('delay')).isdigit() else None,
                                'region': str(row.get('fetch_region', '')) if pd.notna(row.get('fetch_region')) else None,
                                'data_description': str(row.get('description', '')) if pd.notna(row.get('description')) else None
                            })
                        
                        if batch_data:
                            # Insert batch using SQLAlchemy - slim schema with composite primary key
                            connection.execute(
                                text("""
                                    INSERT INTO datafields (
                                        datafield_id, dataset_id, data_category, data_type, delay, region, data_description
                                    ) VALUES (
                                        :datafield_id, :dataset_id, :data_category, :data_type, :delay, :region, :data_description
                                    )
                                    ON CONFLICT (datafield_id, region) DO UPDATE SET
                                        dataset_id = EXCLUDED.dataset_id,
                                        data_category = EXCLUDED.data_category,
                                        data_type = EXCLUDED.data_type,
                                        delay = EXCLUDED.delay,
                                        data_description = EXCLUDED.data_description
                                """),
                                batch_data
                            )
                            insert_count += len(batch_data)
                            
                            if i % (batch_size * 10) == 0:  # Log progress every 10 batches
                                logger.info(f"Inserted {insert_count}/{len(datafields_df)} datafields to database")
                    
                    logger.info(f"Successfully saved {insert_count} datafields to database")
                    return True
                    
        except Exception as e:
            logger.error(f"Error saving datafields to database: {e}")
            return False
    
    def load_datafields_from_db(self) -> Optional[pd.DataFrame]:
        """
        Load datafields from PostgreSQL database.
        
        Returns:
            DataFrame with datafields, or None if database is empty/error
        """
        try:
            from sqlalchemy import text
            
            db_engine = get_connection()
            
            with db_engine.connect() as connection:
                # Check if datafields table exists and has data
                result = connection.execute(text("SELECT COUNT(*) FROM datafields"))
                count = result.scalar()
                
                if count == 0:
                    logger.info("No datafields found in database")
                    return None
                
                # Load all datafields - slim schema
                query = text("""
                    SELECT datafield_id as id, dataset_id, data_category, data_type, delay, region, data_description as description
                    FROM datafields
                    ORDER BY datafield_id, region
                """)
                
                datafields_df = pd.read_sql(query, connection)
                logger.info(f"Loaded {len(datafields_df)} datafields from database")
                return datafields_df
                
        except Exception as e:
            logger.error(f"Error loading datafields from database: {e}")
            return None
    
    def load_cached_operators(self) -> Optional[List[str]]:
        """
        Load operators from cache file.
        
        Returns:
            List of operator names, or None if cache doesn't exist/is invalid
        """
        try:
            if not os.path.exists(self.operators_cache_file):
                logger.info("No operators cache file found")
                return None
            
            with open(self.operators_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            if 'operators' not in cache_data:
                logger.warning("Invalid cache format - missing operators key")
                return None
            
            # Extract just the operator names
            operators_list = [op['name'] for op in cache_data['operators']]
            
            logger.info(f"Loaded {len(operators_list)} operators from cache")
            return operators_list
            
        except Exception as e:
            logger.error(f"Error loading operators cache: {e}")
            return None
    
    def load_cached_datafields(self) -> Optional[pd.DataFrame]:
        """
        Load datafields from database first, fallback to cache file.
        
        Returns:
            DataFrame with datafields, or None if no cache exists/is invalid
        """
        # Try database first (much faster)
        datafields_df = self.load_datafields_from_db()
        if datafields_df is not None:
            return datafields_df
            
        # Fallback to CSV file
        try:
            if not os.path.exists(self.datafields_cache_file):
                logger.info("No datafields cache file found")
                return None
            
            logger.info("Database is empty, loading from CSV file")
            datafields_df = pd.read_csv(self.datafields_cache_file)
            
            logger.info(f"Loaded {len(datafields_df)} datafields from CSV cache")
            return datafields_df
            
        except Exception as e:
            logger.error(f"Error loading datafields from CSV cache: {e}")
            return None
    
    def is_cache_fresh(self, cache_type: str, max_age_days: int = 7) -> bool:
        """
        Check if cache is fresh enough to use.
        
        Args:
            cache_type: 'operators' or 'datafields'
            max_age_days: Maximum age in days before cache is considered stale
            
        Returns:
            True if cache is fresh, False otherwise
        """
        try:
            if not os.path.exists(self.metadata_file):
                return False
            
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if cache_type not in metadata:
                return False
            
            cache_time = datetime.fromisoformat(metadata[cache_type]['timestamp'])
            age = datetime.now() - cache_time
            
            is_fresh = age.days < max_age_days
            logger.info(f"{cache_type} cache is {age.days} days old - {'fresh' if is_fresh else 'stale'}")
            
            return is_fresh
            
        except Exception as e:
            logger.error(f"Error checking cache freshness: {e}")
            return False
    
    def _update_cache_metadata(self, cache_type: str, timestamp: datetime):
        """Update cache metadata file with timestamp."""
        try:
            metadata = {}
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            metadata[cache_type] = {
                'timestamp': timestamp.isoformat(),
                'file': self.operators_cache_file if cache_type == 'operators' else self.datafields_cache_file
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not update cache metadata: {e}")
    
    def fetch_and_cache_all(self, force_refresh: bool = False) -> Tuple[str, bool]:
        """
        Fetch operators and datafields and cache them.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Tuple of (operators_cache_path, datafields_saved_to_db)
        """
        operators_cache_path = None
        datafields_saved = False
        
        try:
            # If force_refresh, delete existing cache files FIRST
            if force_refresh:
                logger.info("Force refresh requested - deleting existing operators cache")
                
                if os.path.exists(self.operators_cache_file):
                    logger.info(f"Deleting old operators cache: {self.operators_cache_file}")
                    os.remove(self.operators_cache_file)
                
                # Also clear metadata file
                if os.path.exists(self.metadata_file):
                    logger.info(f"Deleting cache metadata: {self.metadata_file}")
                    os.remove(self.metadata_file)
            
            # Create single session for both requests
            logger.info("Starting authenticated session")
            session = start_session()
            
            # Fetch and cache operators
            if force_refresh or not self.is_cache_fresh("operators"):
                logger.info("Fetching fresh operators data")
                operators_data = self.fetch_operators(session)
                if operators_data:
                    operators_cache_path = self.save_operators_cache(operators_data)
                    logger.info(f"Successfully updated operators cache")
                else:
                    logger.error("Failed to fetch operators data")
                    if force_refresh:
                        raise Exception("Failed to fetch operators with force_refresh=True")
                    operators_cache_path = self.operators_cache_file
            else:
                logger.info("Using cached operators data")
                operators_cache_path = self.operators_cache_file
            
            # Fetch and save comprehensive datafields to database
            if force_refresh or not self.is_cache_fresh("datafields"):
                logger.info("Fetching comprehensive datafields data")
                datafields_df = self.fetch_datafields_comprehensive(session)
                if datafields_df is not None and not datafields_df.empty:
                    datafields_saved = self.save_datafields_cache(datafields_df)
                    logger.info(f"Successfully saved datafields to database with {len(datafields_df)} entries")
                else:
                    logger.error("Failed to fetch datafields data or empty result")
                    if force_refresh:
                        raise Exception("Failed to fetch datafields with force_refresh=True")
                    datafields_saved = False
            else:
                logger.info("Using cached datafields data from database")
                datafields_saved = True
            
            # Verify data was actually updated when force_refresh is True
            if force_refresh:
                if not os.path.exists(operators_cache_path):
                    raise Exception(f"Operators cache file not created: {operators_cache_path}")
                if not datafields_saved:
                    raise Exception("Datafields not saved to database")
                
                # Check operators file size
                operators_size = os.path.getsize(operators_cache_path)
                if operators_size < 100:  # Operators file should be at least 100 bytes
                    raise Exception(f"Operators cache file seems empty: {operators_size} bytes")
                
                logger.info(f"Cache refresh successful - Operators: {operators_size} bytes, Datafields: saved to database")
            
            return operators_cache_path, datafields_saved
            
        except Exception as e:
            logger.error(f"Error in fetch_and_cache_all: {e}")
            # If force_refresh failed, ensure we don't leave partial updates
            if force_refresh:
                logger.error("Force refresh failed - cache may be in inconsistent state")
            raise
    
    def import_csv_to_database(self, csv_path: Optional[str] = None) -> bool:
        """
        Import existing datafields CSV file to database.
        
        Args:
            csv_path: Path to CSV file. If None, uses default cache file path.
            
        Returns:
            True if successful, False otherwise
        """
        if csv_path is None:
            csv_path = self.datafields_cache_file
            
        try:
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return False
            
            logger.info(f"Importing datafields from CSV: {csv_path}")
            datafields_df = pd.read_csv(csv_path)
            
            if datafields_df.empty:
                logger.warning("CSV file is empty")
                return False
            
            # Save to database
            success = self.save_datafields_to_db(datafields_df)
            if success:
                logger.info(f"Successfully imported {len(datafields_df)} datafields from CSV to database")
            else:
                logger.error("Failed to import CSV to database")
            
            return success
            
        except Exception as e:
            logger.error(f"Error importing CSV to database: {e}")
            return False


def create_operators_txt_from_api(operators_data: List[Dict[str, Any]], 
                                 output_path: str = "data/operators_dynamic.txt") -> str:
    """
    Create operators.txt file from API data for backward compatibility.
    
    Args:
        operators_data: List of operator dictionaries from API
        output_path: Path to output txt file
        
    Returns:
        Path to created file
    """
    try:
        # Extract just the operator names
        operator_names = [op['name'] for op in operators_data]
        
        # Create comma-separated string like the original format
        operators_txt = ', '.join(sorted(operator_names))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(operators_txt)
        
        logger.info(f"Created operators.txt file with {len(operator_names)} operators at: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating operators.txt file: {e}")
        raise


# Convenience functions for direct usage
def fetch_dynamic_operators() -> Optional[List[str]]:
    """
    Convenience function to fetch operators directly.
    
    Returns:
        List of operator names or None if failed
    """
    try:
        fetcher = PlatformDataFetcher()
        operators_data = fetcher.fetch_operators()
        return [op['name'] for op in operators_data]
    except Exception as e:
        logger.error(f"Error in fetch_dynamic_operators: {e}")
        return None


def fetch_dynamic_datafields(region: str = "USA") -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch datafields directly.
    
    Args:
        region: Region to fetch data for
        
    Returns:
        DataFrame with datafields or None if failed
    """
    try:
        fetcher = PlatformDataFetcher()
        return fetcher.fetch_datafields(region=region)
    except Exception as e:
        logger.error(f"Error in fetch_dynamic_datafields: {e}")
        return None