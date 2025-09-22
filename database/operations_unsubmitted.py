"""
Database operations module for unsubmitted alpha data.
"""
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import io
import time
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Set
from .schema import get_connection, get_region_id
from sqlalchemy import text

logger = logging.getLogger(__name__)

def insert_unsubmitted_alpha(alpha_data: Dict[str, Any], region: str) -> None:
    """
    Insert unsubmitted alpha data into the database.
    
    Args:
        alpha_data: Dictionary containing unsubmitted alpha metadata
        region: Region name for the alpha
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin():
                region_id = get_region_id(connection, region)
                
                # Insert new unsubmitted alpha
                insert_stmt = text("""
                INSERT INTO alphas_unsubmitted (
                    alpha_id, region_id, alpha_type, self_correlation,
                    is_sharpe, is_fitness, is_returns, is_drawdown,
                    is_longcount, is_shortcount, is_turnover, is_margin,
                    rn_sharpe, rn_fitness, code, description, universe, delay,
                    neutralization, decay, date_added, last_updated
                ) VALUES (
                    :alpha_id, :region_id, :alpha_type, :self_correlation,
                    :is_sharpe, :is_fitness, :is_returns, :is_drawdown,
                    :is_longcount, :is_shortcount, :is_turnover, :is_margin,
                    :rn_sharpe, :rn_fitness, :code, :description, :universe, :delay,
                    :neutralization, :decay, :date_added, :last_updated
                )
                """)
                connection.execute(insert_stmt, {
                    'alpha_id': alpha_data['alpha_id'], 
                    'region_id': region_id,
                    'alpha_type': alpha_data.get('alpha_type', 'UNSUBMITTED'),
                    'self_correlation': alpha_data.get('self_correlation'),  # Will be None initially
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
    except Exception as e:
        logger.error(f"Error inserting unsubmitted alpha {alpha_data.get('alpha_id', 'unknown')}: {e}")
        raise

def insert_multiple_unsubmitted_alphas(alphas_data: List[Dict[str, Any]], region: str) -> None:
    """
    Insert multiple unsubmitted alphas into the database.
    
    Args:
        alphas_data: List of dictionaries containing unsubmitted alpha metadata
        region: Region name for the alphas
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin():
                region_id = get_region_id(connection, region)
                
                # Prepare data for batch insertion (all alphas, let DB handle duplicates)
                alpha_values = []
                for alpha_data in alphas_data:
                    alpha_values.append((
                        alpha_data['alpha_id'],
                        region_id,
                        alpha_data.get('alpha_type', 'UNSUBMITTED'),
                        alpha_data.get('self_correlation'),  # Will be None initially
                        alpha_data.get('is_sharpe'),
                        alpha_data.get('is_fitness'),
                        alpha_data.get('is_returns'),
                        alpha_data.get('is_drawdown'),
                        alpha_data.get('is_longcount'),
                        alpha_data.get('is_shortcount'),
                        alpha_data.get('is_turnover'),
                        alpha_data.get('is_margin'),
                        alpha_data.get('rn_sharpe'),
                        alpha_data.get('rn_fitness'),
                        alpha_data.get('code'),
                        alpha_data.get('description'),
                        alpha_data.get('universe'),
                        alpha_data.get('delay'),
                        alpha_data.get('neutralization'),
                        alpha_data.get('decay'),
                        alpha_data.get('date_added'),
                        alpha_data.get('last_updated')
                    ))
                
                # Batch insert using execute_values with conflict handling
                insert_query = """
                INSERT INTO alphas_unsubmitted (
                    alpha_id, region_id, alpha_type, self_correlation,
                    is_sharpe, is_fitness, is_returns, is_drawdown,
                    is_longcount, is_shortcount, is_turnover, is_margin,
                    rn_sharpe, rn_fitness, code, description, universe, delay,
                    neutralization, decay, date_added, last_updated
                ) VALUES %s
                ON CONFLICT (alpha_id) DO NOTHING
                """
                
                # Use raw psycopg2 for execute_values
                raw_conn = connection.connection
                with raw_conn.cursor() as cursor:
                    execute_values(cursor, insert_query, alpha_values, page_size=1000)
                
                # Note: We can't easily count how many were actually inserted due to ON CONFLICT DO NOTHING
                # But we can report how many we attempted to insert
                logger.info(f"Processed {len(alphas_data)} unsubmitted alphas for region {region} (duplicates automatically skipped)")
                
    except Exception as e:
        logger.error(f"Error inserting multiple unsubmitted alphas for region {region}: {e}")
        raise

def get_unsubmitted_alpha_type_and_status(alpha_id: str, region: str) -> Optional[Dict[str, Any]]:
    """
    Get unsubmitted alpha type and check if PNL data exists for a specific alpha in a region.

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
                FROM alphas_unsubmitted 
                WHERE alpha_id = :alpha_id AND region_id = :region_id
            """)
            alpha_result = connection.execute(stmt_alpha_details, {"alpha_id": alpha_id, "region_id": region_id}).fetchone()

            if not alpha_result:
                return None

            alpha_type = alpha_result[0]
            
            # Check if PNL data exists for this alpha in the region-specific PNL table
            pnl_table_name = f"pnl_unsubmitted_{region.lower()}"
            stmt_pnl_check = text(f"""
                SELECT 1 
                FROM {pnl_table_name} 
                WHERE alpha_id = :alpha_id 
                LIMIT 1
            """)
            pnl_exists_result = connection.execute(stmt_pnl_check, {"alpha_id": alpha_id}).fetchone()
            pnl_exists = pnl_exists_result is not None

            return {"alpha_type": alpha_type, "pnl_exists": pnl_exists}

    except Exception as e:
        logger.error(f"Error getting unsubmitted alpha type and PNL status for {alpha_id} in {region}: {e}")
        raise

def get_unsubmitted_alpha_ids_for_pnl_processing(region: str) -> List[str]:
    """
    Get unsubmitted alpha IDs for a specific region that do not have PNL data yet.

    Args:
        region: Region name

    Returns:
        List of unsubmitted alpha IDs that need PNL processing.
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            region_id = get_region_id(connection, region)
            pnl_table_name = f"pnl_unsubmitted_{region.lower()}"
            
            stmt = text(f"""
                SELECT a.alpha_id
                FROM alphas_unsubmitted a
                LEFT JOIN {pnl_table_name} pnl ON a.alpha_id = pnl.alpha_id
                WHERE a.region_id = :region_id
                  AND pnl.alpha_id IS NULL
            """)
            result = connection.execute(stmt, {"region_id": region_id})
            alpha_ids = [row[0] for row in result.fetchall()]
            logger.info(f"Found {len(alpha_ids)} unsubmitted alphas in region {region} needing PNL processing.")
            return alpha_ids
    except Exception as e:
        logger.error(f"Error getting unsubmitted alpha IDs for PNL processing in region {region}: {e}")
        raise

def insert_unsubmitted_pnl_data(alpha_id: str, pnl_df: pd.DataFrame, region: str) -> None:
    """
    Insert unsubmitted PNL data into the region-specific table.
    
    Args:
        alpha_id: Alpha ID
        pnl_df: DataFrame containing PNL data with date index and pnl column
        region: Region name
    """
    if pnl_df.empty:
        logger.warning(f"No PNL data to insert for unsubmitted alpha {alpha_id} in region {region}")
        return
    
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin():
                table_name = f"pnl_unsubmitted_{region.lower()}"
                
                # Prepare data for insertion
                # itertuples() returns (index, col1, col2, ...) so we need to unpack properly
                pnl_values = [(alpha_id, row.Index.strftime('%Y-%m-%d'), float(row.pnl)) 
                             for row in pnl_df.itertuples()]
                
                # Use raw psycopg2 for execute_values
                raw_conn = connection.connection
                with raw_conn.cursor() as cursor:
                    insert_query = f"""
                    INSERT INTO {table_name} (alpha_id, date, pnl) 
                    VALUES %s 
                    ON CONFLICT (alpha_id, date) DO UPDATE SET pnl = EXCLUDED.pnl
                    """
                    execute_values(cursor, insert_query, pnl_values, page_size=1000)
                
                logger.info(f"Inserted {len(pnl_values)} PNL records for unsubmitted alpha {alpha_id} in region {region}")
                
    except Exception as e:
        logger.error(f"Error inserting unsubmitted PNL data for alpha {alpha_id} in region {region}: {e}")
        raise

def insert_multiple_unsubmitted_pnl_data_optimized(pnl_data_dict: Dict[str, pd.DataFrame], region: str) -> None:
    """
    Optimized batch insertion of unsubmitted PNL data using PostgreSQL COPY command.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to PNL DataFrame
        region: Region name
    """
    if not pnl_data_dict:
        logger.info("No unsubmitted PNL data to insert")
        return
        
    try:
        db_engine = get_connection()
        table_name = f"pnl_unsubmitted_{region.lower()}"
        
        # Prepare all data in memory first
        all_data = []
        for alpha_id, pnl_df in pnl_data_dict.items():
            if not pnl_df.empty:
                for date, row in pnl_df.iterrows():
                    all_data.append(f"{alpha_id}\t{date.strftime('%Y-%m-%d')}\t{row['pnl']}")
        
        if not all_data:
            logger.info("No valid unsubmitted PNL data to insert after processing")
            return
        
        # Use raw psycopg2 connection for COPY
        raw_conn = db_engine.raw_connection()
        try:
            with raw_conn.cursor() as cursor:
                # Create a temporary table for bulk insert
                temp_table = f"temp_pnl_unsubmitted_{region.lower()}_{int(time.time())}"
                
                cursor.execute(f"""
                CREATE TEMP TABLE {temp_table} (
                    alpha_id VARCHAR(50),
                    date DATE,
                    pnl FLOAT
                )
                """)
                
                # Use COPY for fast insertion into temp table
                data_io = io.StringIO('\n'.join(all_data))
                cursor.copy_from(data_io, temp_table, columns=('alpha_id', 'date', 'pnl'))
                
                # Insert from temp table with conflict handling
                cursor.execute(f"""
                INSERT INTO {table_name} (alpha_id, date, pnl)
                SELECT alpha_id, date, pnl FROM {temp_table}
                ON CONFLICT (alpha_id, date) DO UPDATE SET pnl = EXCLUDED.pnl
                """)
                
                raw_conn.commit()
        finally:
            raw_conn.close()
        
        total_records = len(all_data)
        logger.info(f"Batch inserted {total_records} unsubmitted PNL records across {len(pnl_data_dict)} alphas in region {region}")
        
    except Exception as e:
        logger.error(f"Error in optimized batch unsubmitted PNL insertion for region {region}: {e}")
        raise

def get_all_unsubmitted_alpha_ids_by_region(region: str) -> List[str]:
    """
    Get all unsubmitted alpha IDs for a specific region.
    
    Args:
        region: Region name
    
    Returns:
        List of unsubmitted alpha IDs
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            region_id = get_region_id(connection, region)
            stmt = text("SELECT alpha_id FROM alphas_unsubmitted WHERE region_id = :region_id")
            result = connection.execute(stmt, {"region_id": region_id})
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        logger.error(f"Error getting unsubmitted alpha IDs for region {region}: {e}")
        raise

def get_unsubmitted_pnl_data_for_alphas(alpha_ids: List[str], region: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Get PNL data for multiple unsubmitted alphas in a region.
    
    Args:
        alpha_ids: List of alpha IDs
        region: Region name
    
    Returns:
        Dictionary mapping alpha_id to a dictionary containing the PNL DataFrame under the 'df' key
        Format: {alpha_id: {'df': dataframe}}
    """
    if not alpha_ids:
        logger.info("No unsubmitted alpha IDs provided to get_unsubmitted_pnl_data_for_alphas.")
        return {}
    
    try:
        db_engine = get_connection()
        table_name = f"pnl_unsubmitted_{region.lower()}"
        
        with db_engine.connect() as connection:
            # Construct IN clause for alpha IDs
            placeholders = ','.join([f"':alpha_id_{i}'" for i in range(len(alpha_ids))])
            query = f"""
            SELECT alpha_id, date, pnl 
            FROM {table_name} 
            WHERE alpha_id IN ({placeholders})
            ORDER BY alpha_id, date
            """
            
            # Create parameter dictionary
            params = {f'alpha_id_{i}': alpha_id for i, alpha_id in enumerate(alpha_ids)}
            
            stmt = text(query.replace("':alpha_id_", ":alpha_id_").replace("'", ""))
            result = connection.execute(stmt, params)
            
            # Group results by alpha_id
            pnl_data = {}
            for row in result.fetchall():
                alpha_id, date, pnl = row
                if alpha_id not in pnl_data:
                    pnl_data[alpha_id] = {'dates': [], 'pnls': [], 'alpha_ids': []}
                pnl_data[alpha_id]['dates'].append(date)
                pnl_data[alpha_id]['pnls'].append(pnl)
                pnl_data[alpha_id]['alpha_ids'].append(alpha_id)
            
            # Convert to DataFrames
            final_data = {}
            for alpha_id, data in pnl_data.items():
                df = pd.DataFrame({
                    'alpha_id': data['alpha_ids'],
                    'pnl': data['pnls']
                }, index=pd.to_datetime(data['dates']))
                df.index.name = 'date'
                final_data[alpha_id] = {'df': df}
            
            logger.info(f"Retrieved unsubmitted PNL data for {len(final_data)} alphas in region {region}")
            return final_data
            
    except Exception as e:
        logger.error(f"Error getting unsubmitted PNL data for alphas in region {region}: {e}")
        raise

def update_unsubmitted_alpha_self_correlation(alpha_id: str, region: str, self_correlation: float) -> None:
    """
    Update the self_correlation field for an unsubmitted alpha.
    
    Args:
        alpha_id: Alpha ID
        region: Region name 
        self_correlation: The max correlation value from correlation table
    """
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin():
                region_id = get_region_id(connection, region)
                
                update_stmt = text("""
                UPDATE alphas_unsubmitted 
                SET self_correlation = :self_correlation, last_updated = CURRENT_TIMESTAMP
                WHERE alpha_id = :alpha_id AND region_id = :region_id
                """)
                
                result = connection.execute(update_stmt, {
                    'alpha_id': alpha_id,
                    'region_id': region_id,
                    'self_correlation': self_correlation
                })
                
                if result.rowcount > 0:
                    logger.debug(f"Updated self_correlation for unsubmitted alpha {alpha_id} in region {region}: {self_correlation}")
                else:
                    logger.warning(f"No unsubmitted alpha found to update: {alpha_id} in region {region}")
                    
    except Exception as e:
        logger.error(f"Error updating self_correlation for unsubmitted alpha {alpha_id} in region {region}: {e}")
        raise

def update_multiple_unsubmitted_alpha_self_correlations(correlation_results: Dict[str, Dict[str, Any]], region: str) -> None:
    """
    Update self_correlation values for multiple unsubmitted alphas.
    
    Args:
        correlation_results: Dictionary mapping alpha IDs to correlation results
        region: Region name
    """
    if not correlation_results:
        logger.info("No correlation results to update in alphas_unsubmitted table")
        return
        
    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            with connection.begin():
                region_id = get_region_id(connection, region)
                
                # Prepare batch update data
                update_values = []
                for alpha_id, result in correlation_results.items():
                    max_correlation = result.get('max_correlation')
                    if max_correlation is not None:
                        update_values.append((alpha_id, region_id, max_correlation))
                
                if not update_values:
                    logger.info("No valid correlation values to update")
                    return
                
                # Use raw psycopg2 for batch update
                raw_conn = connection.connection
                with raw_conn.cursor() as cursor:
                    # Use psycopg2's execute_values for batch update
                    update_query = """
                    UPDATE alphas_unsubmitted SET 
                        self_correlation = data.corr,
                        last_updated = CURRENT_TIMESTAMP
                    FROM (VALUES %s) AS data(alpha_id, region_id, corr)
                    WHERE alphas_unsubmitted.alpha_id = data.alpha_id 
                      AND alphas_unsubmitted.region_id = data.region_id
                    """
                    execute_values(cursor, update_query, update_values, page_size=1000)
                    
                logger.info(f"Updated self_correlation for {len(update_values)} unsubmitted alphas in region {region}")
                
    except Exception as e:
        logger.error(f"Error updating multiple self_correlations for region {region}: {e}")
        raise

# Optimized bulk loading functions
def get_unsubmitted_pnl_data_bulk(alpha_ids: List[str], region: str) -> Dict[str, pd.DataFrame]:
    """
    Load PNL data for multiple unsubmitted alphas in a single database query.
    This is significantly faster than loading alphas one by one.

    Args:
        alpha_ids: List of unsubmitted alpha IDs to load
        region: Region identifier

    Returns:
        Dictionary of alpha_id -> DataFrame with PNL data
    """
    if not alpha_ids:
        return {}

    table_name = f'pnl_unsubmitted_{region.lower()}'

    try:
        db_engine = get_connection()

        # Use ANY() for efficient bulk loading
        query = text(f"""
            SELECT alpha_id, date, pnl
            FROM {table_name}
            WHERE alpha_id = ANY(:alpha_ids)
            ORDER BY alpha_id, date
        """)

        with db_engine.connect() as connection:
            # Execute single query for all alphas
            logger.info(f"Executing bulk unsubmitted PNL query for {len(alpha_ids)} alphas in region {region}")
            df = pd.read_sql(
                query,
                connection,
                params={'alpha_ids': alpha_ids}
            )

        if df.empty:
            logger.warning(f"No unsubmitted PNL data found for any of the {len(alpha_ids)} alphas")
            return {}

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Group by alpha_id and create dictionary
        result = {}
        for alpha_id, group_df in df.groupby('alpha_id'):
            pnl_df = group_df[['date', 'pnl']].set_index('date').sort_index()
            result[str(alpha_id)] = pnl_df

        logger.info(f"Successfully loaded unsubmitted PNL data for {len(result)} alphas in a single query")
        return result

    except Exception as e:
        # Check if it's a table not found error
        error_msg = str(e)
        if "does not exist" in error_msg or "UndefinedTable" in error_msg:
            logger.warning(f"Table {table_name} does not exist for region {region}. This region may not support unsubmitted alphas.")
        else:
            logger.error(f"Error in bulk unsubmitted PNL loading: {e}")
        return {}

def get_unsubmitted_pnl_data_optimized(alpha_ids: List[str], region: str) -> Dict[str, Dict]:
    """
    Load unsubmitted PNL data optimized for correlation calculation.
    Returns the same format as get_unsubmitted_pnl_data_for_alphas but much faster.

    Args:
        alpha_ids: List of unsubmitted alpha IDs
        region: Region identifier

    Returns:
        Dictionary compatible with existing correlation code
    """
    pnl_data = get_unsubmitted_pnl_data_bulk(alpha_ids, region)

    # Convert to expected format
    result = {}
    for alpha_id, df in pnl_data.items():
        result[alpha_id] = {'df': df}

    return result