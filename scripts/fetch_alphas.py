"""
Script to fetch alphas from the WorldQuant Brain API and store them in the database.
"""
import sys
import os
import logging
import argparse
from pathlib import Path

# Setup project path
sys.path.append(str(Path(__file__).parent.parent))
from utils.bootstrap import setup_project_path
setup_project_path()

from api.auth import get_authenticated_session, check_session_valid
from api.alpha_fetcher import scrape_alphas_by_region
from database.operations import insert_multiple_alphas
from utils.helpers import setup_logging
from config.database_config import REGIONS

def main():
    parser = argparse.ArgumentParser(description='Fetch alphas from the WorldQuant Brain API')
    parser.add_argument('--region', choices=REGIONS, help='Region to fetch alphas for')
    parser.add_argument('--all', action='store_true', help='Fetch alphas for all regions')
    args = parser.parse_args()
    
    if not args.region and not args.all:
        parser.error('Either --region or --all must be specified')
    
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Get authenticated session
    session = get_authenticated_session()
    if not session:
        logging.error("Failed to get authenticated session. Exiting.")
        return
    
    if not check_session_valid(session):
        logging.error("Session is not valid. Please re-authenticate.")
        return
    
    # Determine which regions to fetch
    regions_to_fetch = REGIONS if args.all else [args.region]
    
    for region in regions_to_fetch:
        logging.info(f"Fetching alphas for region {region}...")
        
        # Scrape alphas for this region
        alphas = scrape_alphas_by_region(session, region)
        
        if not alphas:
            logging.warning(f"No alphas found for region {region}")
            continue
        
        logging.info(f"Found {len(alphas)} alphas for region {region}")
        
        # Insert alphas into database
        try:
            insert_multiple_alphas(alphas, region)
            logging.info(f"Successfully stored alphas for region {region} in database")
        except Exception as e:
            logging.error(f"Error storing alphas for region {region}: {e}")
    
    logging.info("Alpha fetching complete")

if __name__ == "__main__":
    main()
