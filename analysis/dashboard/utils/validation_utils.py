"""
Validation Utilities

Input validation and data integrity checks for the dashboard.
"""

import re
from datetime import datetime
from typing import Any, List, Optional, Union, Dict


def validate_alpha_id(alpha_id: str) -> bool:
    """
    Validate alpha ID format.

    Args:
        alpha_id: Alpha identifier to validate

    Returns:
        True if valid format
    """
    if not alpha_id or not isinstance(alpha_id, str):
        return False

    # Alpha IDs are typically numeric or alphanumeric
    # Adjust pattern based on actual format requirements
    pattern = r'^[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, alpha_id.strip()))


def validate_region(region: str, valid_regions: List[str]) -> bool:
    """
    Validate region against allowed regions.

    Args:
        region: Region to validate
        valid_regions: List of valid regions

    Returns:
        True if valid region
    """
    return region in valid_regions if region else False


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate date range.

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        True if valid date range
    """
    try:
        if start_date and end_date:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            return start <= end
        return True  # Allow partial ranges
    except (ValueError, AttributeError):
        return False


def validate_numeric_input(value: Any, min_val: float = None,
                          max_val: float = None) -> bool:
    """
    Validate numeric input within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        True if valid
    """
    try:
        num_val = float(value)
        if min_val is not None and num_val < min_val:
            return False
        if max_val is not None and num_val > max_val:
            return False
        return True
    except (ValueError, TypeError):
        return False


def validate_clustering_method(method: str) -> bool:
    """
    Validate clustering visualization method.

    Args:
        method: Clustering method name

    Returns:
        True if valid method
    """
    valid_methods = ['mds', 'tsne', 'umap', 'pca', 'heatmap']
    return method in valid_methods


def validate_distance_metric(metric: str) -> bool:
    """
    Validate distance metric.

    Args:
        metric: Distance metric name

    Returns:
        True if valid metric
    """
    valid_metrics = ['simple', 'euclidean', 'angular']
    return metric in valid_metrics


def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> bool:
    """
    Validate file path and extension.

    Args:
        file_path: File path to validate
        allowed_extensions: List of allowed file extensions

    Returns:
        True if valid file path
    """
    if not file_path or not isinstance(file_path, str):
        return False

    # Check for dangerous path traversal
    if '..' in file_path or file_path.startswith('/'):
        return False

    if allowed_extensions:
        extension = file_path.lower().split('.')[-1]
        return extension in [ext.lower().lstrip('.') for ext in allowed_extensions]

    return True


def validate_json_structure(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate JSON data structure has required keys.

    Args:
        data: Dictionary to validate
        required_keys: List of required keys

    Returns:
        True if all required keys present
    """
    if not isinstance(data, dict):
        return False

    return all(key in data for key in required_keys)


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


def sanitize_user_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input text.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""

    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text)

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized.strip()


def validate_port_number(port: Union[str, int]) -> bool:
    """
    Validate port number.

    Args:
        port: Port number to validate

    Returns:
        True if valid port
    """
    try:
        port_num = int(port)
        return 1024 <= port_num <= 65535  # Valid user port range
    except (ValueError, TypeError):
        return False


def validate_analysis_filters(filters: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate analysis filter parameters.

    Args:
        filters: Filter parameters to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Validate region if provided
    if 'region' in filters and filters['region']:
        from ..config import AVAILABLE_REGIONS
        if not validate_region(filters['region'], AVAILABLE_REGIONS):
            errors.append(f"Invalid region: {filters['region']}")

    # Validate date range if provided
    if 'date_from' in filters and 'date_to' in filters:
        if not validate_date_range(filters.get('date_from'), filters.get('date_to')):
            errors.append("Invalid date range")

    # Validate delay if provided
    if 'delay' in filters and filters['delay'] is not None:
        if not validate_numeric_input(filters['delay'], min_val=0, max_val=30):
            errors.append("Invalid delay: must be between 0 and 30")

    return len(errors) == 0, errors


def validate_callback_inputs(*args) -> bool:
    """
    Validate callback inputs are not None/empty.

    Args:
        args: Arguments to validate

    Returns:
        True if all arguments are valid
    """
    return all(arg is not None for arg in args)