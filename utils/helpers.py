"""
Helper utilities for the Alpha DataBank system.

This module provides common utility functions used across the codebase,
primarily for logging configuration and reporting functionality.

Note: Previous correlation calculation functions have been moved to the
optimized implementation in scripts/update_correlations_optimized.py.
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for Alpha DataBank components.
    
    This function configures the Python logging system to output logs
    at the specified level. Logs are always written to console (stdout)
    and optionally to a file if the log_file parameter is specified.
    
    Args:
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
        log_file: Optional file path to save logs
        
    Raises:
        ValueError: If an invalid log level is provided
        
    Example:
        >>> setup_logging(log_level="DEBUG", log_file="logs/alpha_databank.log")
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = []
    
    # Always log to console
    handlers.append(logging.StreamHandler())
    
    # Log to file if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers
    )
    
    logger.debug("Logging configuration initialized successfully")

# The correlation calculation functions previously defined here have been 
# moved to more optimized implementations in scripts/update_correlations_optimized.py

def get_overlapping_dates(pnl1: pd.DataFrame, pnl2: pd.DataFrame) -> List[datetime]:
    """
    Get overlapping dates between two PNL DataFrames.
    
    Args:
        pnl1: First PNL DataFrame with datetime index
        pnl2: Second PNL DataFrame with datetime index
        
    Returns:
        List of overlapping dates sorted chronologically
    """
    return sorted(list(pnl1.index.intersection(pnl2.index)))

def print_correlation_report(correlation_stats: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted report of correlation statistics to the console.
    
    This function takes the calculated correlation statistics for alphas and
    formats them into a readable tabular report showing minimum, maximum,
    and average correlation values for each alpha.
    
    Args:
        correlation_stats: Dictionary mapping alpha_id to statistics dictionary,
                          which should contain keys 'min_correlation', 'max_correlation',
                          and 'avg_correlation'
    
    Example output:
        ==== Correlation Statistics Report ====
        Alpha ID   Min Corr   Max Corr   Avg Corr   Count
        ---------------------------------------------
        ALPHA001   -0.2345    0.5678     0.1234     42
        ALPHA002   -0.4567    0.7890     0.3456     38
        =============================================
    """
    if not correlation_stats:
        print("No correlation statistics to report")
        return
        
    # Check if 'count' is in the stats to determine if we show the count column
    show_count = any('count' in stats for _, stats in correlation_stats.items())
    
    print("\n==== Correlation Statistics Report ====")
    
    if show_count:
        header = f"{'Alpha ID':<10} {'Min Corr':<10} {'Max Corr':<10} {'Avg Corr':<10} {'Count':<8}"
        separator = "-" * 53
    else:
        header = f"{'Alpha ID':<10} {'Min Corr':<10} {'Max Corr':<10} {'Avg Corr':<10}"
        separator = "-" * 45
        
    print(header)
    print(separator)
    
    for alpha_id, stats in correlation_stats.items():
        min_val = stats.get('min_correlation', stats.get('min_corr', 0.0))
        max_val = stats.get('max_correlation', stats.get('max_corr', 0.0))
        avg_val = stats.get('avg_correlation', stats.get('avg_corr', 0.0))
        
        if show_count and 'count' in stats:
            count = stats['count']
            print(f"{alpha_id:<10} {min_val:<10.4f} {max_val:<10.4f} {avg_val:<10.4f} {count:<8}")
        else:
            print(f"{alpha_id:<10} {min_val:<10.4f} {max_val:<10.4f} {avg_val:<10.4f}")
        
    if show_count:
        print("=" * 53)
    else:
        print("=" * 45)
    
    # Print summary of report
    print(f"Total alphas in report: {len(correlation_stats)}")

