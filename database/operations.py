"""
Database operations module for alpha data.
"""
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import io
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Set
from .schema import get_connection, get_region_id
from sqlalchemy import text

logger = logging.getLogger(__name__)

def insert_alpha(alpha_data: Dict[str, Any], region: str) -> None:
    """
    Insert alpha data into the database.
    
    Args:
        alpha_data: Dictionary containing alpha metadata
        region: Region name for the alpha
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin(): # Handles transaction commit/rollback
                region_id = get_region_id(connection, region)
                
                # Insert new alpha
                insert_stmt = text("""
                INSERT INTO alphas (
                    alpha_id, region_id, alpha_type, prod_correlation, self_correlation, 
                    is_sharpe, is_fitness, is_returns, is_drawdown,
                    is_longcount, is_shortcount, is_turnover, is_margin,
                    rn_sharpe, rn_fitness, code, description, universe, delay,
                    neutralization, decay, date_added, last_updated
                ) VALUES (
                    :alpha_id, :region_id, :alpha_type, :prod_correlation, :self_correlation, 
                    :is_sharpe, :is_fitness, :is_returns, :is_drawdown,
                    :is_longcount, :is_shortcount, :is_turnover, :is_margin,
                    :rn_sharpe, :rn_fitness, :code, :description, :universe, :delay,
                    :neutralization, :decay, :date_added, :last_updated
                )
                """)
                connection.execute(insert_stmt, {
                    'alpha_id': alpha_data['alpha_id'], 
                    'region_id': region_id,
                    'alpha_type': alpha_data.get('alpha_type'),
                    'prod_correlation': alpha_data.get('prod_correlation'),
                    'self_correlation': alpha_data.get('self_correlation'),
                    'is_sharpe': alpha_data.get('is_sharpe'), 
                    'is_fitness': alpha_data.get('is_fitness'),
                    'is_returns': alpha_data.get('is_returns'), 
                    'is_drawdown': alpha_data.get('is_drawdown'),
                    'is_longcount': alpha_data.get('is_longcount'), 
                    'is_shortcount': alpha_data.get('is_shortcount'),
                    'is_turnover': alpha_data.get('is_turnover'), 
                    'is_margin': alpha_data.get('is_margin'),
                    'rn_sharpe': alpha_data.get('rn_sharpe'), 
                    'rn_fitness': alpha_data.get('rn_fitness'),
                    'code': alpha_data.get('code'), 
                    'description': alpha_data.get('description'),
                    'universe': alpha_data.get('universe'), 
                    'delay': alpha_data.get('delay'),
                    'neutralization': alpha_data.get('neutralization'), 
                    'decay': alpha_data.get('decay'),
                    'date_added': alpha_data.get('date_added'),
                    'last_updated': alpha_data.get('last_updated')
                })
                pass # Alpha inserted successfully
            # Transaction commits here if no exception, or rolls back on exception
    except Exception as e:
        logger.error(f"Error inserting alpha {alpha_data.get('alpha_id', 'unknown')}: {e}")
        raise

def insert_multiple_alphas(alphas_data: List[Dict[str, Any]], region: str) -> None:
    """
    Insert multiple alphas into the database.
    
    Args:
        alphas_data: List of dictionaries containing alpha metadata
        region: Region name for the alphas
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin(): # Handles transaction commit/rollback
                region_id = get_region_id(connection, region)
                
                # Get list of existing alpha IDs for the specific region
                # It's better to filter by region_id if alphas are region-specific in the table
                # Assuming 'alphas' table has a 'region_id' column for this optimization.
                # If not, the original SELECT without WHERE is fine but less efficient.
                # For now, sticking to a general SELECT alpha_id for broader compatibility if region_id isn't on alphas table for some reason.
                existing_alphas_stmt = text("SELECT alpha_id FROM alphas WHERE region_id = :region_id")
                existing_alphas_result = connection.execute(existing_alphas_stmt, {"region_id": region_id})
                existing_alpha_ids = {row[0] for row in existing_alphas_result.fetchall()}
                
                # Filter out alphas that already exist
                new_alphas_to_insert_data = [
                    alpha for alpha in alphas_data 
                    if alpha['alpha_id'] not in existing_alpha_ids
                ]
                
                if not new_alphas_to_insert_data:
                    logger.info("No new alphas to insert for region_id {region_id}")
                    return
                
                # Prepare data for bulk insert
                columns = [
                    'alpha_id', 'region_id', 'alpha_type', 'prod_correlation', 'self_correlation',
                    'is_sharpe', 'is_fitness', 'is_returns', 'is_drawdown',
                    'is_longcount', 'is_shortcount', 'is_turnover', 'is_margin',
                    'rn_sharpe', 'rn_fitness', 'code', 'description', 'universe', 'delay',
                    'neutralization', 'decay', 'date_added', 'last_updated'
                ]
                
                insert_stmt_sql = f"""
                INSERT INTO alphas ({', '.join(columns)}) VALUES ({', '.join([':' + col for col in columns])})
                """
                insert_stmt = text(insert_stmt_sql)

                values_for_insert = []
                for alpha in new_alphas_to_insert_data:
                    values_for_insert.append({
                        'alpha_id': alpha['alpha_id'], 
                        'region_id': region_id,
                        'alpha_type': alpha.get('alpha_type'),
                        'prod_correlation': alpha.get('prod_correlation'),
                        'self_correlation': alpha.get('self_correlation'),
                        'is_sharpe': alpha.get('is_sharpe'), 
                        'is_fitness': alpha.get('is_fitness'),
                        'is_returns': alpha.get('is_returns'), 
                        'is_drawdown': alpha.get('is_drawdown'),
                        'is_longcount': alpha.get('is_longcount'), 
                        'is_shortcount': alpha.get('is_shortcount'),
                        'is_turnover': alpha.get('is_turnover'), 
                        'is_margin': alpha.get('is_margin'),
                        'rn_sharpe': alpha.get('rn_sharpe'), 
                        'rn_fitness': alpha.get('rn_fitness'),
                        'code': alpha.get('code'), 
                        'description': alpha.get('description'),
                        'universe': alpha.get('universe'), 
                        'delay': alpha.get('delay'),
                        'neutralization': alpha.get('neutralization'), 
                        'decay': alpha.get('decay')
                    })
                
                if values_for_insert:
                    connection.execute(insert_stmt, values_for_insert)
                    logger.info(f"Inserted {len(values_for_insert)} new alphas into database for region_id {region_id}")
            # Transaction commits here if no exception, or rolls back on exception
    except Exception as e:
        logger.error(f"Error inserting multiple alphas: {e}")
        raise

def get_alpha_type_and_status(alpha_id: str, region: str) -> Optional[Dict[str, Any]]:
    """
    Get alpha type and check if PNL data exists for a specific alpha in a region.

    Args:
        alpha_id: Alpha ID
        region: Region name

    Returns:
        A dictionary {'alpha_type': type, 'pnl_exists': bool} if alpha found, else None.
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            region_id = get_region_id(connection, region)

            # Get alpha type
            stmt_alpha_details = text("""
                SELECT alpha_type 
                FROM alphas 
                WHERE alpha_id = :alpha_id AND region_id = :region_id
            """)
            alpha_result = connection.execute(stmt_alpha_details, {"alpha_id": alpha_id, "region_id": region_id}).fetchone()

            if not alpha_result:
                # Alpha not found in this region
                return None

            alpha_type = alpha_result[0]
            
            # Check if PNL data exists for this alpha in the region-specific PNL table
            pnl_table_name = f"pnl_{region.lower()}"
            stmt_pnl_check = text(f"""
                SELECT 1 
                FROM {pnl_table_name} 
                WHERE alpha_id = :alpha_id 
                LIMIT 1
            """)
            pnl_exists_result = connection.execute(stmt_pnl_check, {"alpha_id": alpha_id}).fetchone()
            pnl_exists = pnl_exists_result is not None

            # Return the alpha type and PNL existence status
            return {"alpha_type": alpha_type, "pnl_exists": pnl_exists}

    except Exception as e:
        logger.error(f"Error getting alpha type and PNL status for {alpha_id} in {region}: {e}")
        # Optionally, re-raise or handle more gracefully depending on desired error propagation
        raise

def insert_pnl_data(alpha_id: str, pnl_df: pd.DataFrame, region: str) -> None:
    """
    Insert PNL data for an alpha into the region-specific PNL table.
    
    Args:
        alpha_id: Alpha ID
        pnl_df: DataFrame with Date index and PNL columns
        region: Region name
    """
    if pnl_df.empty:
        logger.warning(f"Empty PNL data for alpha {alpha_id}, skipping")
        return
    
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin(): # Handles transaction commit/rollback
                table_name = f"pnl_{region.lower()}"
                
                pnl_col = None
                for col in pnl_df.columns:
                    if 'pnl' in col.lower():
                        pnl_col = col
                        break
                
                if not pnl_col:
                    logger.error(f"No PNL column found in DataFrame for alpha {alpha_id}")
                    return
                
                # Prepare data for insertion
                pnl_records_to_insert = []
                for _, row in pnl_df.iterrows():
                    # Use lowercase column names to match API response
                    pnl_record = {
                        'alpha_id': alpha_id,
                        'date': row.name,  # Use the index which is the date
                        'pnl': row['pnl'] if 'pnl' in row else None  # Only single PNL column
                    }
                    pnl_records_to_insert.append(pnl_record)
                
                # SQL statement with parameterized query
                insert_stmt_sql = f"""
                INSERT INTO {table_name} (alpha_id, date, pnl)
                VALUES (:alpha_id, :date, :pnl)
                ON CONFLICT (alpha_id, date) DO NOTHING
                """
                insert_stmt = text(insert_stmt_sql)
                
                connection.execute(insert_stmt, pnl_records_to_insert)
                logger.info(f"Attempted insertion of {len(pnl_records_to_insert)} PNL records for alpha {alpha_id} in region {region}. Check DB for actual count due to ON CONFLICT.")
            # Transaction commits here if no exception, or rolls back on exception
    except Exception as e:
        logger.error(f"Error inserting PNL data for alpha {alpha_id}: {e}")
        raise


def insert_multiple_pnl_data(pnl_data_dict: Dict[str, pd.DataFrame], region: str) -> None:
    """
    Insert PNL data for multiple alphas in batch into the region-specific PNL table.
    This is much more efficient than inserting one alpha at a time when dealing with many alphas.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to PNL DataFrame
        region: Region name
    """
    if not pnl_data_dict:
        logger.warning(f"No PNL data provided for batch insertion in region {region}, skipping")
        return
    
    # Count valid dataframes (non-empty)
    valid_dfs = sum(1 for df in pnl_data_dict.values() if not df.empty)
    if valid_dfs == 0:
        logger.warning(f"All PNL dataframes are empty for region {region}, skipping batch insertion")
        return
    
    logger.info(f"Starting batch insertion of PNL data for {valid_dfs} alphas in region {region}")
    
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin(): # Single transaction for all inserts
                # Prepare table name
                table_name = f"pnl_{region.lower()}"
                
                # SQL statement with parameterized query
                insert_stmt_sql = f"""
                INSERT INTO {table_name} (alpha_id, date, pnl)
                VALUES (:alpha_id, :date, :pnl)
                ON CONFLICT (alpha_id, date) DO NOTHING
                """
                insert_stmt = text(insert_stmt_sql)
                
                # Prepare all records at once
                all_pnl_records = []
                alpha_record_counts = {}
                
                for alpha_id, pnl_df in pnl_data_dict.items():
                    if pnl_df.empty:
                        logger.debug(f"Empty PNL data for alpha {alpha_id}, skipping in batch")
                        continue
                    
                    # Reset index to get date as a column if needed (handle lowercase date column)
                    if pnl_df.index.name == 'date':
                        pnl_df = pnl_df.reset_index()
                    
                    # Track records per alpha for logging
                    record_count = len(pnl_df)
                    alpha_record_counts[alpha_id] = record_count
                    
                    # Add records to the batch
                    for idx, row in pnl_df.iterrows():
                        pnl_record = {
                            'alpha_id': alpha_id,
                            'date': row.name if pnl_df.index.name == 'date' else row['date'],
                            'pnl': row['pnl'] if 'pnl' in row else None
                        }
                        all_pnl_records.append(pnl_record)
                
                # Execute batch insert in a single operation
                if all_pnl_records:
                    connection.execute(insert_stmt, all_pnl_records)
                    logger.info(f"Batch inserted {len(all_pnl_records)} total PNL records for {len(alpha_record_counts)} alphas in region {region}")
                    for alpha_id, count in alpha_record_counts.items():
                        logger.debug(f"  - Alpha {alpha_id}: {count} records")
                else:
                    logger.warning(f"No valid PNL records to insert for region {region} after processing")
            # Transaction commits here if no exception, or rolls back on exception
    except Exception as e:
        logger.error(f"Error in batch PNL insertion for region {region}: {e}")
        raise

def insert_multiple_pnl_data_optimized(pnl_data_dict: Dict[str, pd.DataFrame], region: str) -> None:
    """
    Optimized PNL data insertion using PostgreSQL COPY command.
    This method is significantly faster for large datasets compared to the standard INSERT.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to PNL DataFrame
        region: Region name
    """
    if not pnl_data_dict:
        logger.warning(f"No PNL data provided for batch insertion in region {region}, skipping")
        return
    
    # Count valid dataframes
    valid_df_count = sum(1 for df in pnl_data_dict.values() if not df.empty)
    
    if valid_df_count == 0:
        logger.warning(f"No valid PNL dataframes for batch insertion in region {region}, skipping")
        return
    
    try:
        db_engine = get_connection()
        table_name = f"pnl_{region.lower()}"
        
        # Create a single large DataFrame with all records
        all_records = []
        alpha_record_counts = {}
        
        for alpha_id, df in pnl_data_dict.items():
            if df.empty:
                continue
                
            # Prepare records
            records = df.reset_index().rename(columns={'index': 'date'})
            records['alpha_id'] = alpha_id
            all_records.append(records[['alpha_id', 'date', 'pnl']])
            alpha_record_counts[alpha_id] = len(records)
        
        if not all_records:
            logger.warning(f"No valid PNL records to insert for region {region}")
            return
            
        # Combine all records
        combined_records = pd.concat(all_records, ignore_index=True)
        
        # Extract connection parameters from SQLAlchemy engine and create proper psycopg2 connection
        url = db_engine.url
        db_params = {
            'dbname': url.database,
            'user': url.username,
            'password': url.password,
            'host': url.host,
            'port': url.port
        }
        # Use raw psycopg2 connection for COPY
        conn = psycopg2.connect(**db_params)
        conn.autocommit = False
        cursor = conn.cursor()
        
        try:
            # Create a temporary table for data staging
            cursor.execute(f"CREATE TEMP TABLE temp_{table_name} (LIKE {table_name} INCLUDING ALL) ON COMMIT DROP")
            
            # Prepare data as CSV in memory
            csv_data = io.StringIO()
            combined_records.to_csv(csv_data, index=False, header=False, sep='\t')
            csv_data.seek(0)
            
            # Use COPY to load data into temp table
            cursor.copy_from(
                csv_data,
                f"temp_{table_name}",
                columns=['alpha_id', 'date', 'pnl']
            )
            
            # Insert from temp table to real table with conflict handling
            cursor.execute(f"""
                INSERT INTO {table_name} (alpha_id, date, pnl)
                SELECT alpha_id, date, pnl FROM temp_{table_name}
                ON CONFLICT (alpha_id, date) DO NOTHING
            """)
            
            # Commit the transaction
            conn.commit()
            logger.info(f"Completed batch PNL insertion for {len(alpha_record_counts)} alphas in region {region}")
            for alpha_id, count in alpha_record_counts.items():
                logger.debug(f"  - Alpha {alpha_id}: {count} records")
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Error in optimized batch PNL insertion for region {region}: {e}")
        raise

def get_all_alpha_ids_by_region_basic(region: str) -> List[str]:
    """
    Get all alpha IDs for a specific region (includes REGULAR and SUPER alphas).
    
    Args:
        region: Region name
    
    Returns:
        List of alpha IDs (all types)
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            region_id = get_region_id(connection, region) # Pass SQLAlchemy connection
            stmt = text("SELECT alpha_id FROM alphas WHERE region_id = :region_id")
            result = connection.execute(stmt, {"region_id": region_id})
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Error getting alpha IDs for region {region}: {e}")
        raise

def get_alpha_ids_for_pnl_processing(region: str) -> List[str]:
    """
    Get REGULAR and SUPER alpha IDs for a specific region that do not have PNL data yet.

    Args:
        region: Region name

    Returns:
        List of alpha IDs that are 'REGULAR' or 'SUPER' and need PNL processing.
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            region_id = get_region_id(connection, region)
            pnl_table_name = f"pnl_{region.lower()}"
            
            # Select REGULAR and SUPER alpha_ids that are not in the PNL table for this region
            stmt = text(f"""
                SELECT a.alpha_id
                FROM alphas a
                LEFT JOIN {pnl_table_name} pnl ON a.alpha_id = pnl.alpha_id
                WHERE a.region_id = :region_id
                  AND a.alpha_type IN ('REGULAR', 'SUPER')
                  AND pnl.alpha_id IS NULL
            """)
            result = connection.execute(stmt, {"region_id": region_id})
            alpha_ids = [row[0] for row in result.fetchall()]
            logger.info(f"Found {len(alpha_ids)} REGULAR/SUPER alphas in region {region} needing PNL processing.")
            return alpha_ids
    except Exception as e:
        logger.error(f"Error getting REGULAR/SUPER alpha IDs for PNL processing in region {region}: {e}")
        raise

def get_regular_alpha_ids_for_pnl_processing(region: str) -> List[str]:
    """
    Legacy function for backward compatibility. Now includes SUPER alphas.
    
    Args:
        region: Region name
        
    Returns:
        List of alpha IDs that are 'REGULAR' or 'SUPER' and need PNL processing.
    """
    return get_alpha_ids_for_pnl_processing(region)

def get_pnl_data_for_alphas(alpha_ids: List[str], region: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Get PNL data for multiple alphas in a region.
    
    Args:
        alpha_ids: List of alpha IDs
        region: Region name
    
    Returns:
        Dictionary mapping alpha_id to a dictionary containing the PNL DataFrame under the 'df' key
        Format: {alpha_id: {'df': dataframe}}
    """
    if not alpha_ids:
        return {}
    
    try:
        db_engine = get_connection() # This returns a SQLAlchemy Engine
        result = {}
        table_name = f"pnl_{region.lower()}"
        alphas_with_data = 0
        
        for alpha_id in alpha_ids:
            # Using named parameters for pd.read_sql with SQLAlchemy engine
            query = f"""
            SELECT date, pnl FROM {table_name}
            WHERE alpha_id = :alpha_id
            ORDER BY date
            """
            
            # pd.read_sql can take a SQLAlchemy Engine directly
            # Wrap query with text() to ensure named parameters are processed by SQLAlchemy
            df = pd.read_sql(text(query), db_engine, params={"alpha_id": alpha_id})
            if not df.empty:
                # Convert date to datetime and set as index
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                # Store dataframe in a nested dictionary under 'df' key to match expected format
                # in the correlation calculation code
                result[alpha_id] = {'df': df}
                alphas_with_data += 1
            else:
                # Still include the alpha in the result but with an empty dataframe
                # This avoids KeyError when accessing alphas without data
                result[alpha_id] = {'df': None}
        
        if alphas_with_data == 0:
            logger.warning(f"No PNL data found for any of the {len(alpha_ids)} alphas in region {region}")
        else:
            logger.info(f"Retrieved PNL data for {alphas_with_data} out of {len(alpha_ids)} alphas in region {region}")
        
        return result
    except Exception as e:
        logger.error(f"Error getting PNL data for alphas in region {region}: {e}")
        raise

def calculate_and_store_correlations(region: str) -> None:
    """
    Calculate and store correlation summary statistics for all alphas in a region.
    Uses more accurate correlation calculation using daily returns
    based on portfolio value, and handling of different time periods.
    
    Args:
        region: Region name
    """
    try:
        # Get all alpha IDs for the region
        alpha_ids = get_all_alpha_ids_by_region_basic(region)
        
        if not alpha_ids:
            logger.warning(f"No alphas found for region {region}")
            return
        
        # Get PNL data for all alphas
        logger.info(f"Retrieving PNL data for {len(alpha_ids)} alphas in region {region}")
        pnl_data = get_pnl_data_for_alphas(alpha_ids, region)
        
        if not pnl_data:
            logger.warning(f"No PNL data found for region {region}")
            return
        
        logger.info(f"Calculating correlations for {len(pnl_data)} alphas in region {region}")
        
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin(): # Handles transaction commit/rollback
                # Helper functions for correlation calculation
                def calculate_daily_pnl(cumulative_pnl_series):
                    """Calculate daily PNL from cumulative PNL series."""
                    daily_pnl = cumulative_pnl_series.diff()
                    daily_pnl.iloc[0] = cumulative_pnl_series.iloc[0]
                    return daily_pnl
                
                def calculate_returns(daily_pnl_series, initial_value=10_000_000):
                    """Calculate daily returns from daily PNL series."""
                    portfolio_value = initial_value + daily_pnl_series.cumsum()
                    return daily_pnl_series / portfolio_value.shift(1)
                
                def calculate_alpha_correlation(alpha_id, alpha_pnl, other_id, other_pnl):
                    """Calculate correlation between two alpha PNLs."""
                    if alpha_id == other_id:
                        return None  # Skip self-correlation
                    
                    # Get overlapping dates
                    common_dates = alpha_pnl.index.intersection(other_pnl.index)
                    if len(common_dates) < 20:  # Require at least 20 common dates for meaningful correlation
                        # Not enough common dates for meaningful correlation
                        return None
                    
                    # Use overlapping period for both alphas
                    alpha_pnl_common = alpha_pnl.loc[common_dates, 'pnl']
                    other_pnl_common = other_pnl.loc[common_dates, 'pnl']
                    
                    # Calculate daily PNL and returns
                    alpha_daily_pnl = calculate_daily_pnl(alpha_pnl_common)
                    alpha_returns = calculate_returns(alpha_daily_pnl)
                    
                    other_daily_pnl = calculate_daily_pnl(other_pnl_common)
                    other_returns = calculate_returns(other_daily_pnl)
                    
                    # Filter out invalid values
                    valid = ~(alpha_returns.isna() | other_returns.isna() | 
                              np.isinf(alpha_returns) | np.isinf(other_returns))
                    alpha_returns_clean = alpha_returns[valid]
                    other_returns_clean = other_returns[valid]
                    
                    if len(alpha_returns_clean) < 20:
                        # Not enough valid returns data
                        return None
                    
                    # Calculate correlation
                    from scipy import stats
                    corr = stats.pearsonr(alpha_returns_clean, other_returns_clean)[0]
                    if np.isnan(corr):
                        return None
                    
                    return corr
                
                # Process each alpha
                for alpha_id, alpha_pnl in pnl_data.items():
                    # Calculate correlation with all other alphas
                    correlations = []
                    
                    # For each alpha, identify the time window to use
                    # Aim for a 4-year window if possible, or use all available data
                    end_date = alpha_pnl.index.max()
                    start_date = end_date - pd.DateOffset(years=4)
                    
                    # Filter alpha PNL to the window
                    windowed_alpha_pnl = alpha_pnl
                    if len(alpha_pnl.loc[alpha_pnl.index >= start_date]) >= 60:  # At least 60 data points needed
                        windowed_alpha_pnl = alpha_pnl.loc[alpha_pnl.index >= start_date]
                    
                    # Calculate correlation with all other alphas
                    for other_id, other_pnl in pnl_data.items():
                        corr = calculate_alpha_correlation(alpha_id, windowed_alpha_pnl, other_id, other_pnl)
                        if corr is not None:
                            correlations.append(corr)
                    
                    if not correlations:
                        continue
                    
                    min_corr = min(correlations)
                    max_corr = max(correlations)
                    avg_corr = sum(correlations) / len(correlations)
                    median_corr = np.median(correlations)
                    
                    table_name = f"correlation_{region.lower()}"
                    # Using named parameters for the upsert query
                    upsert_query_sql = f"""
                    INSERT INTO {table_name} (alpha_id, min_correlation, max_correlation, avg_correlation, median_correlation)
                    VALUES (:alpha_id, :min_corr, :max_corr, :avg_corr, :median_corr)
                    ON CONFLICT (alpha_id) DO UPDATE SET
                        min_correlation = :min_corr,
                        max_correlation = :max_corr,
                        avg_correlation = :avg_corr,
                        median_correlation = :median_corr,
                        last_updated = CURRENT_TIMESTAMP
                    """
                    upsert_stmt = text(upsert_query_sql)
                    connection.execute(upsert_stmt, {
                        "alpha_id": alpha_id, 
                        "min_corr": min_corr, 
                        "max_corr": max_corr, 
                        "avg_corr": avg_corr,
                        "median_corr": median_corr
                    })
                
                pass # Updated correlation statistics
            # Transaction commits here if no exception, or rolls back on exception
    except Exception as e:
        logger.error(f"Error calculating correlations for region {region}: {e}")
        raise

def get_correlation_statistics(region: str) -> pd.DataFrame:
    """
    Get correlation statistics for all alphas in a region.
    
    Args:
        region: Region name
    
    Returns:
        DataFrame with correlation statistics
    """
    try:
        db_engine = get_connection() # This returns a SQLAlchemy Engine
        table_name = f"correlation_{region.lower()}"
        # Using named parameters for pd.read_sql with SQLAlchemy engine
        query = f"""
        SELECT a.alpha_id, a.is_sharpe, a.is_fitness, c.min_correlation, 
               c.max_correlation, c.avg_correlation, c.median_correlation, c.last_updated
        FROM alphas a
        JOIN {table_name} c ON a.alpha_id = c.alpha_id
        JOIN regions r ON a.region_id = r.region_id
        WHERE r.region_name = :region_name
        ORDER BY c.avg_correlation
        """
        
        # pd.read_sql can take a SQLAlchemy Engine directly
        # Wrap query with text() to ensure named parameters are processed by SQLAlchemy
        df = pd.read_sql(text(query), db_engine, params={"region_name": region})
        return df
    except Exception as e:
        logger.error(f"Error getting correlation statistics for region {region}: {e}")
        raise
