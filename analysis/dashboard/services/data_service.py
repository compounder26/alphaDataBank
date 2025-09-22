"""
Data Service

Core data loading, caching, and region management for the dashboard.
Extracted from visualization_server.py with full backward compatibility.
"""

import os
import sys
import json
import glob
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the project root to the path for imports (preserve original behavior)
# Setup project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.bootstrap import setup_project_path
setup_project_path()

from analysis.analysis_operations import AnalysisOperations
from database.schema import get_connection
from config.database_config import REGIONS
from sqlalchemy import text

from ..utils import cached


# Global operators list for dynamic operators (preserve original globals)
DYNAMIC_OPERATORS_LIST = None
DYNAMIC_DATAFIELDS_LIST = None
_analysis_ops_instance = None


def create_analysis_operations(operators_file: Optional[str] = None,
                              operators_list: Optional[List[str]] = None,
                              available_datafields_list: Optional[List[str]] = None) -> AnalysisOperations:
    """
    Create or return singleton AnalysisOperations instance with current global settings.
    This prevents creating multiple instances and reduces connection overhead.

    IMPORTANT: When operators_list is provided from operators_dynamic.json, these are
    TIER-SPECIFIC operators that the user has access to. The API already filters these
    based on the user's tier.

    Args:
        operators_file: Path to operators file (optional)
        operators_list: List of operators (optional) - TIER-SPECIFIC from API
        available_datafields_list: List of available datafields (optional) - TIER-SPECIFIC from API

    Returns:
        AnalysisOperations instance
    """
    global _analysis_ops_instance
    if _analysis_ops_instance is None:
        # Use provided parameters or fall back to globals
        if operators_file is None:
            from ..config import DEFAULT_OPERATORS_FILE
            operators_file = DEFAULT_OPERATORS_FILE

        if operators_list is None:
            operators_list = DYNAMIC_OPERATORS_LIST

        if available_datafields_list is None:
            available_datafields_list = DYNAMIC_DATAFIELDS_LIST

        # CRITICAL: Pass the tier-specific lists to AnalysisOperations
        # These lists represent what the user ACTUALLY has access to
        _analysis_ops_instance = AnalysisOperations(operators_file, operators_list, available_datafields_list)

        if operators_list:
            print(f"Created AnalysisOperations with {len(operators_list)} tier-specific operators")
        if available_datafields_list:
            print(f"Created AnalysisOperations with {len(available_datafields_list)} tier-specific datafields")
        else:
            print("WARNING: No tier-specific datafields provided - alphas may not be filtered correctly!")

    return _analysis_ops_instance


def set_tier_operators_and_datafields(operators_list: List[str], datafields_list: Optional[List[str]] = None):
    """
    Set the global tier-specific operators and datafields lists.

    Args:
        operators_list: List of operators available to the user's tier
        datafields_list: Optional list of datafields available to the user's tier
    """
    global DYNAMIC_OPERATORS_LIST, DYNAMIC_DATAFIELDS_LIST
    DYNAMIC_OPERATORS_LIST = operators_list
    DYNAMIC_DATAFIELDS_LIST = datafields_list
    print(f"Set tier operators: {len(operators_list) if operators_list else 0} operators")
    print(f"Set tier datafields: {len(datafields_list) if datafields_list else 0} datafields")
    # Reset singleton to pick up new settings
    reset_analysis_operations()


def reset_analysis_operations():
    """Reset the singleton instance (useful for testing or config changes)."""
    global _analysis_ops_instance
    _analysis_ops_instance = None


@cached(ttl=300)  # Cache for 5 minutes
def get_alpha_details_for_clustering(alpha_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch alpha details for clustering hover information.

    Args:
        alpha_ids: List of alpha IDs to fetch details for

    Returns:
        Dictionary mapping alpha_id to alpha details
    """
    # Return empty dict if no alpha IDs provided
    if not alpha_ids:
        return {}

    try:
        db_engine = get_connection()
        with db_engine.connect() as connection:
            # Get alpha details
            placeholders = ','.join([f":alpha_{i}" for i in range(len(alpha_ids))])
            query = text(f"""
                SELECT
                    a.alpha_id,
                    a.code,
                    a.universe,
                    a.delay,
                    a.is_sharpe,
                    a.is_fitness,
                    a.is_returns,
                    a.neutralization,
                    a.decay,
                    r.region_name
                FROM alphas a
                JOIN regions r ON a.region_id = r.region_id
                WHERE a.alpha_id IN ({placeholders})
            """)

            params = {f'alpha_{i}': alpha_id for i, alpha_id in enumerate(alpha_ids)}
            result = connection.execute(query, params)

            alpha_details = {}
            for row in result:
                alpha_details[row.alpha_id] = {
                    'code': row.code or '',
                    'universe': row.universe or 'N/A',
                    'delay': row.delay if row.delay is not None else 'N/A',
                    'is_sharpe': row.is_sharpe if row.is_sharpe is not None else 0,
                    'is_fitness': row.is_fitness if row.is_fitness is not None else 0,
                    'is_returns': row.is_returns if row.is_returns is not None else 0,
                    'neutralization': row.neutralization or 'N/A',
                    'decay': row.decay or 'N/A',
                    'region_name': row.region_name or 'N/A'
                }

            return alpha_details
    except Exception as e:
        print(f"Error fetching alpha details: {e}")
        return {}


def load_clustering_data(filepath: str) -> Dict[str, Any]:
    """
    Load clustering data from a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary with clustering data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


@cached(ttl=600)  # Cache for 10 minutes
def load_all_region_data() -> Dict[str, Any]:
    """
    Load clustering data for all available regions.

    Returns:
        Dictionary with region names as keys and clustering data as values
    """
    all_region_data = {}

    for region in REGIONS:
        # Look for the latest clustering file for this region
        pattern = f"analysis/clustering/alpha_clustering_{region}_*.json"
        files = glob.glob(pattern)

        if files:
            latest_file = max(files, key=os.path.getctime)
            try:
                region_data = load_clustering_data(latest_file)
                if region_data:
                    all_region_data[region] = region_data
                    print(f"[OK] Loaded {region}: {region_data.get('alpha_count', 0)} alphas")
            except Exception as e:
                print(f"[ERROR] Failed to load {region}: {e}")
        else:
            print(f"[MISSING] No clustering data found for {region}")

    print(f"Successfully loaded clustering data for {len(all_region_data)} regions")
    return all_region_data


def get_available_regions_from_files() -> List[str]:
    """
    Get list of regions that have clustering data files available.

    Returns:
        List of region names with available data
    """
    available_regions = []

    for region in REGIONS:
        pattern = f"analysis/clustering/alpha_clustering_{region}_*.json"
        files = glob.glob(pattern)
        if files:
            available_regions.append(region)

    return available_regions


def load_tier_specific_datafields() -> List[str]:
    """
    Load tier-specific datafields from the database.
    These are the datafields that were fetched from the API for this user's tier.

    Returns:
        List of datafield IDs available to the user
    """
    try:
        from database.schema import get_connection
        from sqlalchemy import text

        db_engine = get_connection()
        with db_engine.connect() as connection:
            # Get all unique datafield IDs from the database
            # These were populated by renew_genius.py and are tier-specific
            query = text("""
                SELECT DISTINCT datafield_id
                FROM datafields
                WHERE datafield_id IS NOT NULL AND datafield_id != ''
                ORDER BY datafield_id
            """)

            result = connection.execute(query)
            datafield_ids = [row[0] for row in result.fetchall()]

            print(f"Loaded {len(datafield_ids)} tier-specific datafields from database")
            return datafield_ids

    except Exception as e:
        print(f"Error loading tier-specific datafields: {e}")
        return []


def load_operators_data(operators_file: str) -> List[str]:
    """
    Load operators data from file with support for different formats.

    Args:
        operators_file: Path to operators file (JSON or TXT)

    Returns:
        List of operator names
    """
    global DYNAMIC_OPERATORS_LIST

    try:
        operators_file_clean = operators_file.strip().lower()

        if operators_file_clean.endswith('.json'):
            # Handle JSON format (like operators_dynamic.json)
            with open(operators_file, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'operators' in data:
                # Extract operator names from API response format
                operators = [op['name'] for op in data['operators']]
                print(f"Loaded {len(operators)} operators from JSON file")
            elif isinstance(data, list):
                # Direct list of operator names
                operators = data
                print(f"Loaded {len(operators)} operators from JSON list")
            else:
                raise ValueError(f"Unsupported JSON format in {operators_file}")
        else:
            # Handle traditional TXT format
            with open(operators_file, 'r') as f:
                operators = [op.strip() for op in f.read().split(',')]
            print(f"Loaded {len(operators)} operators from TXT file")

        # Validate operator count to catch parsing errors
        if len(operators) > 1000:
            print(f"Warning: Suspicious operator count: {len(operators)} - possible parsing error")
            # Show sample to help debug
            sample_ops = operators[:10]
            print(f"Sample operators: {sample_ops}")

        DYNAMIC_OPERATORS_LIST = operators
        return operators

    except Exception as e:
        print(f"Error loading operators: {e}")
        return []


def validate_clustering_data(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate clustering data structure.

    Args:
        data: Clustering data to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required top-level keys
    required_keys = ['alpha_count', 'region', 'timestamp']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    # Validate alpha count
    if 'alpha_count' in data:
        if not isinstance(data['alpha_count'], int) or data['alpha_count'] <= 0:
            errors.append("Invalid alpha_count: must be positive integer")

    # Validate region
    if 'region' in data:
        if not isinstance(data['region'], str) or not data['region'].strip():
            errors.append("Invalid region: must be non-empty string")

    # Check for coordinate data
    coordinate_keys = ['mds_coords', 'tsne_coords', 'umap_coords', 'pca_coords']
    if not any(key in data for key in coordinate_keys):
        errors.append("No coordinate data found")

    # Validate coordinate data structure
    for coord_key in coordinate_keys:
        if coord_key in data and not isinstance(data[coord_key], dict):
            errors.append(f"Invalid {coord_key}: must be dictionary")

    return len(errors) == 0, errors


def get_data_file_info(filepath: str) -> Dict[str, Any]:
    """
    Get information about a data file.

    Args:
        filepath: Path to the file

    Returns:
        File information dictionary
    """
    try:
        stat = os.stat(filepath)
        return {
            'filepath': filepath,
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'exists': True
        }
    except Exception as e:
        return {
            'filepath': filepath,
            'error': str(e),
            'exists': False
        }


def cleanup_cached_data():
    """Clean up cached data and reset instances."""
    from ..utils import clear_cache

    # Clear utility caches
    clear_cache()

    # Reset analysis operations instance
    reset_analysis_operations()

    print("Cleaned up cached data and reset instances")