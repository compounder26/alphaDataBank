"""
Script to initialize the alpha database.
"""
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from database.schema import initialize_database, initialize_analysis_database, initialize_unsubmitted_database
from utils.helpers import setup_logging

if __name__ == "__main__":
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Initialize all database schemas
    logging.info("Initializing Alpha Database...")
    initialize_database()
    logging.info("Main database schema initialized")
    
    logging.info("Initializing Analysis Database Schema...")
    initialize_analysis_database()
    logging.info("Analysis database schema initialized")
    
    logging.info("Initializing Unsubmitted Alphas Database Schema...")
    initialize_unsubmitted_database()
    logging.info("Unsubmitted alphas database schema initialized")
    
    logging.info("🎉 Complete database initialization finished!")
