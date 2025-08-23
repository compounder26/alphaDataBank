"""
Script to initialize the alpha database.
"""
import sys
import os
import logging
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from database.schema import initialize_database
from utils.helpers import setup_logging

if __name__ == "__main__":
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Initialize database
    logging.info("Initializing Alpha Database...")
    initialize_database()
    logging.info("Database initialization complete")
