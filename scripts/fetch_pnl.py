"""
Script to fetch PNL data for alphas and store it in the database.
"""
import sys
import os
import logging
import argparse
import time
from pathlib import Path

# Setup project path
sys.path.append(str(Path(__file__).parent.parent))
from utils.bootstrap import setup_project_path
setup_project_path()

from api.auth import get_authenticated_session, check_session_valid
from api.alpha_fetcher import get_alpha_pnl
from database.operations import get_all_alpha_ids_by_region, insert_pnl_data
from utils.helpers import setup_logging
from config.database_config import REGIONS

def main():
    parser = argparse.ArgumentParser(description='Fetch PNL data for alphas')
    parser.add_argument('--region', choices=REGIONS, help='Region to fetch PNL data for')
    parser.add_argument('--all', action='store_true', help='Fetch PNL data for all regions')
    parser.add_argument('--alpha-id', help='Fetch PNL data for a specific alpha ID')
    args = parser.parse_args()
    
    if not args.region and not args.all and not args.alpha_id:
        parser.error('Either --region, --all, or --alpha-id must be specified')
    
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
    
    # If a specific alpha ID is provided
    if args.alpha_id:
        if not args.region:
            logging.error("Region must be specified when fetching PNL for a specific alpha ID")
            return
        
        logging.info(f"Fetching PNL data for alpha {args.alpha_id} in region {args.region}...")
        pnl_df = get_alpha_pnl(session, args.alpha_id)
        
        if pnl_df.empty:
            logging.warning(f"No PNL data found for alpha {args.alpha_id}")
        else:
            logging.info(f"Retrieved {len(pnl_df)} PNL records for alpha {args.alpha_id}")
            insert_pnl_data(args.alpha_id, pnl_df, args.region)
            logging.info(f"PNL data for alpha {args.alpha_id} stored in database")
        
        return
    
    # Determine which regions to fetch
    regions_to_fetch = REGIONS if args.all else [args.region]
    
    for region in regions_to_fetch:
        logging.info(f"Fetching PNL data for region {region}...")
        
        # Get all alpha IDs for this region
        alpha_ids = get_all_alpha_ids_by_region(region)
        
        if not alpha_ids:
            logging.warning(f"No alphas found for region {region}")
            continue
        
        logging.info(f"Found {len(alpha_ids)} alphas for region {region}")
        
        # Fetch PNL data for each alpha
        for i, alpha_id in enumerate(alpha_ids):
            logging.info(f"Fetching PNL data for alpha {alpha_id} ({i+1}/{len(alpha_ids)})...")
            
            try:
                pnl_df = get_alpha_pnl(session, alpha_id)
                
                if pnl_df.empty:
                    logging.warning(f"No PNL data found for alpha {alpha_id}")
                    continue
                
                logging.info(f"Retrieved {len(pnl_df)} PNL records for alpha {alpha_id}")
                
                # Store PNL data in database
                insert_pnl_data(alpha_id, pnl_df, region)
                logging.info(f"PNL data for alpha {alpha_id} stored in database")
                
                # Rate limiting
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error fetching or storing PNL data for alpha {alpha_id}: {e}")
    
    logging.info("PNL data fetching complete")

if __name__ == "__main__":
    main()
