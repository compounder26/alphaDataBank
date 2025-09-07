"""
Database schema and initialization module.
"""
import psycopg2
# from psycopg2.extensions import connection as Connection # No longer using this specific type hint for get_connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, Connection as SQLAlchemyConnection
from sqlalchemy import text # For executing text-based SQL
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import List
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database_config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, REGIONS

logger = logging.getLogger(__name__)

# --- SQLAlchemy Engine Setup ---
try:
    db_connection_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    db_engine = create_engine(
        db_connection_str,
        pool_size=20,           # Base pool size (increased from default 5)
        max_overflow=30,        # Overflow connections (increased from default 10)  
        pool_pre_ping=True,     # Validate connections before use
        pool_recycle=3600,      # Recycle connections every hour
        echo=False,             # Set to True for SQL debugging
        connect_args={"options": "-c timezone=utc"}  # Set timezone for consistency
    )
    logger.info("SQLAlchemy engine created successfully with enhanced connection pooling.")
except Exception as e:
    logger.error(f"Failed to create SQLAlchemy engine: {e}")
    # Depending on desired behavior, you might want to raise this or exit
    # For now, we'll let it proceed, and errors will occur upon use if engine creation failed.
    db_engine = None # Ensure db_engine is defined even if creation fails

def create_database() -> None:
    """Create the database specified by DB_NAME if it doesn't exist."""
    # Connection parameters to connect to the default 'postgres' database
    # This is necessary because you can't be connected to a database to create that same database.
    connection_params_postgres_db = {
        'user': DB_USER,
        'password': DB_PASSWORD,
        'host': DB_HOST,
        'port': DB_PORT,
        'database': 'postgres'  # Connect to the default 'postgres' db to create the new one
    }
    
    conn = None  # Initialize conn to None for the finally block
    try:
        conn = psycopg2.connect(**connection_params_postgres_db)
        # Set the connection to autocommit mode.
        # This ensures that CREATE DATABASE is not run inside a transaction block.
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        with conn.cursor() as cursor:
            # Check if the target database (from your config's DB_NAME) exists
            # It's good practice to use DB_NAME from your config for the database name.
            # Assuming DB_NAME is 'alpha_database' as per your script's intent.
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
            if not cursor.fetchone():
                # If using DB_NAME from config, this is generally safe.
                # For dynamic names, consider using psycopg2.sql.Identifier for safety.
                cursor.execute(f"CREATE DATABASE {DB_NAME}")
                logger.info(f"Database '{DB_NAME}' created successfully.")
            else:
                logger.info(f"Database '{DB_NAME}' already exists.")
    except psycopg2.Error as e:
        logger.error(f"Error during database creation: {e}")
        raise  # Re-raise the exception after logging
    except Exception as e:
        logger.error(f"An unexpected error occurred during database creation: {e}")
        raise # Re-raise unexpected exceptions
    finally:
        if conn:
            conn.close()

def init_schema() -> None:
    """Initialize the database schema."""
    try:
        connection_params_main_db = {
            'user': DB_USER,
            'password': DB_PASSWORD,
            'host': DB_HOST,
            'port': DB_PORT,
            'database': DB_NAME
        }
        with psycopg2.connect(**connection_params_main_db) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                # Create regions table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS regions (
                    region_id SERIAL PRIMARY KEY,
                    region_name VARCHAR(10) UNIQUE NOT NULL
                )
                """)
                
                # Insert regions if they don't exist
                for region in REGIONS:
                    cursor.execute(
                        "INSERT INTO regions (region_name) VALUES (%s) ON CONFLICT (region_name) DO NOTHING",
                        (region,)
                    )
                
                # Create alphas table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS alphas (
                    alpha_id VARCHAR(50) PRIMARY KEY,
                    region_id INTEGER REFERENCES regions(region_id),
                    alpha_type VARCHAR(15), -- ADDED: Type of alpha (REGULAR/SUPER/UNSUBMITTED)
                    prod_correlation FLOAT, -- ADDED: Production correlation from metadata
                    self_correlation FLOAT, -- ADDED: Self correlation from metadata
                    is_sharpe FLOAT,
                    is_fitness FLOAT,
                    is_returns FLOAT,
                    is_drawdown FLOAT,
                    is_longcount INTEGER,
                    is_shortcount INTEGER,
                    is_turnover FLOAT,
                    is_margin FLOAT,
                    rn_sharpe FLOAT,
                    rn_fitness FLOAT,
                    code TEXT,
                    description TEXT,
                    universe VARCHAR(50),
                    delay INTEGER,
                    neutralization VARCHAR(50),
                    decay VARCHAR(50),
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP
                )
                """)
                
                # Create PNL and correlation tables for each region
                for region in REGIONS:
                    # Create PNL table for this region
                    cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS pnl_{region.lower()} (
                        pnl_id SERIAL PRIMARY KEY,
                        alpha_id VARCHAR(50) REFERENCES alphas(alpha_id) ON DELETE CASCADE,
                        date DATE NOT NULL,
                        pnl FLOAT NOT NULL,
                        UNIQUE(alpha_id, date)
                    )
                    """)
                    
                    # Create correlation summary table for this region
                    cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS correlation_{region.lower()} (
                        correlation_id SERIAL PRIMARY KEY,
                        alpha_id VARCHAR(50) REFERENCES alphas(alpha_id) ON DELETE CASCADE,
                        min_correlation FLOAT,
                        max_correlation FLOAT,
                        avg_correlation FLOAT,
                        median_correlation FLOAT,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(alpha_id)
                    )
                    """)
                    
                    # Create indices for performance
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_pnl_{region.lower()}_alpha_id ON pnl_{region.lower()} (alpha_id)")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_pnl_{region.lower()}_date ON pnl_{region.lower()} (date)")
                
                logger.info("Database schema initialized successfully")
    except psycopg2.Error as e:
        logger.error(f"Error initializing schema: {e}")
        raise

def get_connection() -> Engine:
    """Get the SQLAlchemy engine for the database."""
    if db_engine is None:
        logger.error("SQLAlchemy engine is not available. Check initial configuration.")
        raise RuntimeError("Database engine not initialized")
    return db_engine

def get_region_id(conn: SQLAlchemyConnection, region_name: str) -> int:
    """Get the region_id for a given region name."""
    # For SQLAlchemy, you can execute directly on the connection
    # The %s placeholder is typical for psycopg2; SQLAlchemy uses named parameters (:param) or positional based on dialect.
    # For psycopg2 dialect (default for postgresql+psycopg2), %s should still work with text().
    stmt = text("SELECT region_id FROM regions WHERE region_name = :region_name")
    db_result = conn.execute(stmt, {"region_name": region_name})
    row = db_result.fetchone()
    if row:
        return row[0]
    raise ValueError(f"Region {region_name} not found in database")

def init_unsubmitted_schema() -> None:
    """Initialize the database schema for unsubmitted alphas."""
    try:
        connection_params_main_db = {
            'user': DB_USER,
            'password': DB_PASSWORD,
            'host': DB_HOST,
            'port': DB_PORT,
            'database': DB_NAME
        }
        with psycopg2.connect(**connection_params_main_db) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                # Create unsubmitted alphas table (no correlation fields from API)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS alphas_unsubmitted (
                    alpha_id VARCHAR(50) PRIMARY KEY,
                    region_id INTEGER REFERENCES regions(region_id),
                    alpha_type VARCHAR(15), -- Always 'UNSUBMITTED' for this table
                    self_correlation FLOAT, -- Max correlation with submitted alphas from correlation table
                    is_sharpe FLOAT,
                    is_fitness FLOAT,
                    is_returns FLOAT,
                    is_drawdown FLOAT,
                    is_longcount INTEGER,
                    is_shortcount INTEGER,
                    is_turnover FLOAT,
                    is_margin FLOAT,
                    rn_sharpe FLOAT,
                    rn_fitness FLOAT,
                    code TEXT,
                    description TEXT,
                    universe VARCHAR(50),
                    delay INTEGER,
                    neutralization VARCHAR(50),
                    decay VARCHAR(50),
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP
                )
                """)
                
                # Add self_correlation column if it doesn't exist (migration for existing databases)
                cursor.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                  WHERE table_name='alphas_unsubmitted' AND column_name='self_correlation') THEN
                        ALTER TABLE alphas_unsubmitted ADD COLUMN self_correlation FLOAT;
                    END IF;
                END $$;
                """)
                
                # Create PNL and correlation tables for each region for unsubmitted alphas
                for region in REGIONS:
                    # Create PNL table for unsubmitted alphas in this region
                    cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS pnl_unsubmitted_{region.lower()} (
                        pnl_id SERIAL PRIMARY KEY,
                        alpha_id VARCHAR(50) REFERENCES alphas_unsubmitted(alpha_id) ON DELETE CASCADE,
                        date DATE NOT NULL,
                        pnl FLOAT NOT NULL,
                        UNIQUE(alpha_id, date)
                    )
                    """)
                    
                    # Create correlation summary table for unsubmitted alphas
                    # This stores max correlation with submitted alphas only
                    cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS correlation_unsubmitted_{region.lower()} (
                        correlation_id SERIAL PRIMARY KEY,
                        alpha_id VARCHAR(50) REFERENCES alphas_unsubmitted(alpha_id) ON DELETE CASCADE,
                        max_correlation_with_submitted FLOAT,
                        best_correlated_submitted_alpha VARCHAR(50),
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(alpha_id)
                    )
                    """)
                    
                    # Create indices for performance
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_pnl_unsubmitted_{region.lower()}_alpha_id ON pnl_unsubmitted_{region.lower()} (alpha_id)")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_pnl_unsubmitted_{region.lower()}_date ON pnl_unsubmitted_{region.lower()} (date)")
                    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_correlation_unsubmitted_{region.lower()}_alpha_id ON correlation_unsubmitted_{region.lower()} (alpha_id)")
                
                logger.info("Unsubmitted alphas database schema initialized successfully")
    except psycopg2.Error as e:
        logger.error(f"Error initializing unsubmitted schema: {e}")
        raise

def initialize_database() -> None:
    """Main function to initialize the database."""
    create_database()
    init_schema()
    logger.info("Database initialization complete")

def init_analysis_schema() -> None:
    """Initialize the database schema for alpha expression analysis."""
    try:
        connection_params_main_db = {
            'user': DB_USER,
            'password': DB_PASSWORD,
            'host': DB_HOST,
            'port': DB_PORT,
            'database': DB_NAME
        }
        with psycopg2.connect(**connection_params_main_db) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                # Create alpha analysis cache table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS alpha_analysis_cache (
                    cache_id SERIAL PRIMARY KEY,
                    alpha_id VARCHAR(50) REFERENCES alphas(alpha_id) ON DELETE CASCADE,
                    operators_unique JSONB,
                    operators_nominal JSONB,
                    datafields_unique JSONB,
                    datafields_nominal JSONB,
                    excluded BOOLEAN DEFAULT FALSE,
                    exclusion_reason TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(alpha_id)
                )
                """)
                
                # Add missing columns if they don't exist (migration for existing databases)
                cursor.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                  WHERE table_name='alpha_analysis_cache' AND column_name='excluded') THEN
                        ALTER TABLE alpha_analysis_cache ADD COLUMN excluded BOOLEAN DEFAULT FALSE;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                  WHERE table_name='alpha_analysis_cache' AND column_name='exclusion_reason') THEN
                        ALTER TABLE alpha_analysis_cache ADD COLUMN exclusion_reason TEXT;
                    END IF;
                END $$;
                """)
                
                # Create analysis summary table for aggregated results
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_summary (
                    summary_id SERIAL PRIMARY KEY,
                    region_id INTEGER REFERENCES regions(region_id),
                    universe VARCHAR(50),
                    delay INTEGER,
                    alpha_type VARCHAR(15),
                    analysis_type VARCHAR(20), -- 'operators' or 'datafields'
                    item_name VARCHAR(200),
                    item_category VARCHAR(100),
                    unique_count INTEGER,
                    nominal_count INTEGER,
                    alpha_ids JSONB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create indices for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alpha_analysis_cache_alpha_id ON alpha_analysis_cache (alpha_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_summary_region ON analysis_summary (region_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_summary_type ON analysis_summary (analysis_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_summary_item ON analysis_summary (item_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_summary_filters ON analysis_summary (region_id, universe, delay, alpha_type)")
                
                # Additional performance indexes for common query patterns
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alphas_region_universe ON alphas (region_id, universe)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alphas_region_delay ON alphas (region_id, delay)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alphas_code_notnull ON alphas (alpha_id) WHERE code IS NOT NULL AND code != ''")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alphas_date_added ON alphas (date_added)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alphas_composite_filters ON alphas (region_id, universe, delay, alpha_type) WHERE code IS NOT NULL")
                
                # Create slim datafields table with only 6 essential columns
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS datafields (
                    datafield_id VARCHAR(255),
                    dataset_id VARCHAR(255),
                    data_category VARCHAR(255),
                    data_type VARCHAR(50),
                    delay INTEGER,
                    region VARCHAR(10),
                    data_description TEXT,
                    PRIMARY KEY (datafield_id, region)
                )
                """)
                
                # Create indexes for common datafield query patterns
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datafields_dataset_id ON datafields(dataset_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datafields_data_category ON datafields(data_category)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datafields_data_type ON datafields(data_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datafields_delay ON datafields(delay)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datafields_region ON datafields(region)")
                
                logger.info("Alpha analysis database schema initialized successfully (including datafields table)")
    except psycopg2.Error as e:
        logger.error(f"Error initializing analysis schema: {e}")
        raise

def initialize_unsubmitted_database() -> None:
    """Main function to initialize the unsubmitted alphas database."""
    create_database()  # Ensure main database exists
    init_unsubmitted_schema()
    logger.info("Unsubmitted alphas database initialization complete")

def initialize_analysis_database() -> None:
    """Main function to initialize the alpha analysis database."""
    create_database()  # Ensure main database exists
    init_analysis_schema()
    logger.info("Alpha analysis database initialization complete")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_database()
