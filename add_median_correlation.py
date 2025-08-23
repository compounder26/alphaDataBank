"""
Script to add median_correlation column to correlation tables.
"""
import psycopg2
from config.database_config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
from database.regions import REGIONS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_median_correlation_column():
    """Add median_correlation column to correlation tables if it doesn't exist."""
    try:
        connection_params = {
            'user': DB_USER,
            'password': DB_PASSWORD,
            'host': DB_HOST,
            'port': DB_PORT,
            'database': DB_NAME
        }
        
        with psycopg2.connect(**connection_params) as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                for region in REGIONS:
                    table_name = f'correlation_{region.lower()}'
                    # Check if column exists
                    cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND column_name = 'median_correlation'
                    """, (table_name,))
                    
                    if not cursor.fetchone():
                        # Add the column if it doesn't exist
                        cursor.execute(f"""
                        ALTER TABLE {table_name}
                        ADD COLUMN median_correlation FLOAT
                        """)
                        logger.info(f"Added median_correlation column to {table_name}")
                    else:
                        logger.info(f"median_correlation column already exists in {table_name}")
                        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        raise

if __name__ == "__main__":
    add_median_correlation_column()
