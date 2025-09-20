"""
Data Transformation Utilities

Helper functions for data processing and transformation.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string.

    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str) if json_str else default
    except (json.JSONDecodeError, TypeError):
        return default


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_alpha_expression(code: str, max_length: int = 300) -> str:
    """
    Format alpha expression for display.

    Args:
        code: Alpha expression code
        max_length: Maximum length for display

    Returns:
        Formatted expression
    """
    if not code:
        return 'No expression available'

    # Break at logical points for better readability
    formatted_code = code.replace(';', ';<br>').replace(', ', ',<br>')

    # Limit to reasonable length for hover
    if len(formatted_code) > max_length:
        formatted_code = formatted_code[:max_length] + '...'

    return formatted_code


def create_hover_text(alpha_id: str, alpha_details: Dict[str, Any],
                     cluster_info: str = "", match_info: str = "") -> str:
    """
    Create enhanced hover text for alpha points.

    Args:
        alpha_id: Alpha identifier
        alpha_details: Alpha details dictionary
        cluster_info: Cluster information
        match_info: Highlighting match information

    Returns:
        Formatted hover text
    """
    code = alpha_details.get('code', '')
    formatted_code = format_alpha_expression(code)

    hover_text = f"""<b>{alpha_id}</b>{' (' + match_info + ')' if match_info else ''}<br>
Expression: <br>{formatted_code}<br>
{cluster_info}<br>
Universe: {alpha_details.get('universe', 'N/A')}<br>
Delay: {alpha_details.get('delay', 'N/A')}<br>
Sharpe: {alpha_details.get('is_sharpe', 0):.3f}<br>
Fitness: {alpha_details.get('is_fitness', 0):.3f}<br>
Returns: {alpha_details.get('is_returns', 0):.3f}<br>
Neutralization: {alpha_details.get('neutralization', 'N/A')}<br>
Decay: {alpha_details.get('decay', 'N/A')}<br>
Region: {alpha_details.get('region_name', 'N/A')}"""

    return hover_text


def extract_dataset_from_datafield(datafield: str) -> str:
    """
    Extract dataset ID from datafield name.

    Args:
        datafield: Datafield name

    Returns:
        Dataset ID
    """
    if '.' in datafield:
        return datafield.split('.')[0]
    else:
        # Try to extract from first part before underscore
        parts = datafield.split('_')
        return parts[0] if len(parts) > 1 else 'unknown'


def calculate_usage_percentage(used: int, total: int) -> float:
    """
    Calculate usage percentage safely.

    Args:
        used: Number used
        total: Total number

    Returns:
        Usage percentage
    """
    return (used / total * 100) if total > 0 else 0.0


def sort_dict_by_values(data_dict: Dict[str, Union[int, float]],
                       reverse: bool = True) -> List[tuple]:
    """
    Sort dictionary by values.

    Args:
        data_dict: Dictionary to sort
        reverse: Sort in descending order

    Returns:
        List of (key, value) tuples sorted by value
    """
    return sorted(data_dict.items(), key=lambda x: x[1], reverse=reverse)


def group_small_categories(data: List[tuple], min_threshold: int = 5,
                          other_label: str = "Others") -> List[tuple]:
    """
    Group small categories into 'Others' category.

    Args:
        data: List of (category, count) tuples
        min_threshold: Minimum count to keep category separate
        other_label: Label for grouped small categories

    Returns:
        List with small categories grouped
    """
    large_categories = [(cat, count) for cat, count in data if count >= min_threshold]
    small_categories = [(cat, count) for cat, count in data if count < min_threshold]

    result = large_categories
    if small_categories:
        others_count = sum(count for _, count in small_categories)
        result.append((other_label, others_count))

    return result


def safe_divide(numerator: Union[int, float], denominator: Union[int, float],
                default: float = 0.0) -> float:
    """
    Safely divide two numbers.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def clean_display_name(name: str, remove_prefixes: List[str] = None) -> str:
    """
    Clean display name by removing prefixes and formatting.

    Args:
        name: Original name
        remove_prefixes: List of prefixes to remove

    Returns:
        Cleaned display name
    """
    if not name:
        return name

    display_name = name.replace('_', ' ').title()

    if remove_prefixes:
        for prefix in remove_prefixes:
            prefix_title = prefix.title() + ' '
            if display_name.startswith(prefix_title):
                display_name = display_name.replace(prefix_title, '')

    return display_name


def validate_numeric_range(value: Any, min_val: float = None,
                          max_val: float = None, default: float = None) -> Optional[float]:
    """
    Validate numeric value is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value if invalid

    Returns:
        Valid numeric value or default
    """
    try:
        num_val = float(value)
        if min_val is not None and num_val < min_val:
            return default
        if max_val is not None and num_val > max_val:
            return default
        return num_val
    except (ValueError, TypeError):
        return default