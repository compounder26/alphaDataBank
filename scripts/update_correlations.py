"""
Script to calculate and update correlation statistics for alphas.
"""
import sys
import os
import logging
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from database.operations import calculate_and_store_correlations
from utils.helpers import setup_logging
from config.database_config import REGIONS

def main():
    parser = argparse.ArgumentParser(description='Calculate and update correlation statistics for alphas')
    parser.add_argument('--region', choices=REGIONS, help='Region to update correlations for')
    parser.add_argument('--all', action='store_true', help='Update correlations for all regions')
    args = parser.parse_args()
    
    if not args.region and not args.all:
        parser.error('Either --region or --all must be specified')
    
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Determine which regions to update
    regions_to_update = REGIONS if args.all else [args.region]
    
    for region in regions_to_update:
        logging.info(f"Calculating correlations for region {region}...")
        
        try:
            calculate_and_store_correlations(region)
            logging.info(f"Successfully updated correlation statistics for region {region}")
        except Exception as e:
            logging.error(f"Error calculating correlations for region {region}: {e}")
    
    logging.info("Correlation calculation complete")

if __name__ == "__main__":
    main()
