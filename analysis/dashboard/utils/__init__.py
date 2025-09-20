"""
Utilities package for Alpha Dashboard.

Provides caching, data transformation, plotting, and validation utilities.
"""

from .cache_utils import (
    cached,
    clear_cache,
    cleanup_cache,
    get_cache_stats
)

from .data_utils import (
    safe_json_loads,
    safe_float_conversion,
    truncate_text,
    format_alpha_expression,
    create_hover_text,
    extract_dataset_from_datafield,
    calculate_usage_percentage,
    sort_dict_by_values,
    group_small_categories,
    safe_divide,
    clean_display_name,
    validate_numeric_range
)

from .plot_utils import (
    get_cluster_color_map,
    create_scatter_trace,
    create_highlight_trace,
    add_plot_annotations,
    apply_chart_template,
    create_bar_chart,
    create_pie_chart,
    scale_treemap_values,
    create_correlation_heatmap,
    add_performance_overlay
)

from .validation_utils import (
    validate_alpha_id,
    validate_region,
    validate_date_range,
    validate_numeric_input,
    validate_clustering_method,
    validate_distance_metric,
    validate_file_path,
    validate_json_structure,
    validate_clustering_data,
    sanitize_user_input,
    validate_port_number,
    validate_analysis_filters,
    validate_callback_inputs
)

__all__ = [
    # Cache utilities
    'cached',
    'clear_cache',
    'cleanup_cache',
    'get_cache_stats',

    # Data utilities
    'safe_json_loads',
    'safe_float_conversion',
    'truncate_text',
    'format_alpha_expression',
    'create_hover_text',
    'extract_dataset_from_datafield',
    'calculate_usage_percentage',
    'sort_dict_by_values',
    'group_small_categories',
    'safe_divide',
    'clean_display_name',
    'validate_numeric_range',

    # Plot utilities
    'get_cluster_color_map',
    'create_scatter_trace',
    'create_highlight_trace',
    'add_plot_annotations',
    'apply_chart_template',
    'create_bar_chart',
    'create_pie_chart',
    'scale_treemap_values',
    'create_correlation_heatmap',
    'add_performance_overlay',

    # Validation utilities
    'validate_alpha_id',
    'validate_region',
    'validate_date_range',
    'validate_numeric_input',
    'validate_clustering_method',
    'validate_distance_metric',
    'validate_file_path',
    'validate_json_structure',
    'validate_clustering_data',
    'sanitize_user_input',
    'validate_port_number',
    'validate_analysis_filters',
    'validate_callback_inputs'
]