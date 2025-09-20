#!/usr/bin/env python3
"""
Unified correlation calculation script.

This script consolidates functionality from multiple correlation scripts:
- update_correlations.py (basic submitted correlations)
- update_correlations_optimized.py (optimized submitted correlations) 
- calculate_unsubmitted_correlations.py (unsubmitted vs submitted)
- calculate_cross_correlation.py (cross-correlation with CSV export)

Usage examples:
    # Calculate correlations for submitted alphas
    python scripts/calculate_correlations.py --mode submitted --region USA
    
    # Calculate correlations for unsubmitted vs submitted alphas
    python scripts/calculate_correlations.py --mode unsubmitted --region USA
    
    # Cross-correlation analysis with CSV export
    python scripts/calculate_correlations.py --mode cross --alpha-ids alpha1,alpha2,alpha3 --csv-export results.csv
    
    # All regions for submitted alphas
    python scripts/calculate_correlations.py --mode submitted --all-regions
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from analysis.correlation.correlation_engine import CorrelationEngine
from utils.helpers import setup_logging
from config.database_config import REGIONS
from api.auth import get_authenticated_session

def main():
    parser = argparse.ArgumentParser(
        description='Unified correlation calculation for alpha strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode submitted --region USA
  %(prog)s --mode unsubmitted --region USA  
  %(prog)s --mode cross --alpha-ids alpha1,alpha2 --csv-export results.csv
  %(prog)s --mode submitted --all-regions
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['submitted', 'unsubmitted', 'cross'], 
                       required=True,
                       help='Correlation calculation mode')
    
    # Region options
    parser.add_argument('--region', 
                       choices=REGIONS, 
                       help='Region to process (required for submitted/unsubmitted modes)')
    parser.add_argument('--all-regions', 
                       action='store_true',
                       help='Process all regions (for submitted/unsubmitted modes)')
    
    # Cross-correlation options
    parser.add_argument('--alpha-ids',
                       help='Comma-separated list of alpha IDs for cross-correlation analysis')
    parser.add_argument('--reference-alpha',
                       help='Reference alpha ID for 1-to-many comparison')
    parser.add_argument('--csv-export',
                       help='Path to export cross-correlation results as CSV')
    
    # Performance options
    parser.add_argument('--max-workers',
                       type=int,
                       default=4,
                       help='Maximum number of worker threads (default: 4)')
    parser.add_argument('--window-days',
                       type=int,
                       default=400,
                       help='Window size in days for correlation calculation (default: 400)')
    parser.add_argument('--no-cython',
                       action='store_true',
                       help='Disable Cython acceleration, use Python fallback')
    
    # Logging options
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['submitted', 'unsubmitted']:
        if not args.region and not args.all_regions:
            parser.error(f'--mode {args.mode} requires either --region or --all-regions')
    
    if args.mode == 'cross':
        if not args.alpha_ids:
            parser.error('--mode cross requires --alpha-ids')
    
    # Set up logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Initialize correlation engine
    correlation_engine = CorrelationEngine(use_cython=not args.no_cython)
    
    try:
        if args.mode == 'submitted':
            # Submitted alphas correlation calculation
            regions_to_process = REGIONS if args.all_regions else [args.region]
            
            for region in regions_to_process:
                logger.info(f"Calculating submitted alpha correlations for region: {region}")
                correlation_engine.calculate_batch_submitted(
                    region=region, 
                    max_workers=args.max_workers,
                    window_days=args.window_days
                )
                logger.info(f"Successfully completed correlations for region: {region}")
        
        elif args.mode == 'unsubmitted':
            # Unsubmitted vs submitted correlations
            regions_to_process = REGIONS if args.all_regions else [args.region]
            
            for region in regions_to_process:
                logger.info(f"Calculating unsubmitted vs submitted correlations for region: {region}")
                correlation_engine.calculate_unsubmitted_vs_submitted(
                    region=region,
                    max_workers=args.max_workers
                )
                logger.info(f"Successfully completed unsubmitted correlations for region: {region}")
        
        elif args.mode == 'cross':
            # Cross-correlation analysis
            alpha_ids = [alpha_id.strip() for alpha_id in args.alpha_ids.split(',')]
            logger.info(f"Starting cross-correlation analysis for {len(alpha_ids)} alphas")
            
            # Get authenticated session if needed
            session = get_authenticated_session()
            if session is None:
                logger.error("Failed to get authenticated session for API calls")
                return 1
            
            results = correlation_engine.calculate_cross_correlation_analysis(
                alpha_ids=alpha_ids,
                reference_alpha_id=args.reference_alpha,
                csv_export_path=args.csv_export,
                session=session
            )
            
            if results:
                logger.info(f"Cross-correlation analysis completed for {len(results)} alphas")
                
                # Print top correlations
                sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                logger.info("Top 10 correlations:")
                for alpha_id, correlation in sorted_results[:10]:
                    logger.info(f"  {alpha_id}: {correlation:.4f}")
            else:
                logger.warning("No correlation results generated")
        
        logger.info("Correlation calculation completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Correlation calculation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error in correlation calculation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())