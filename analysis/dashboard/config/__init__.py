"""
Configuration package for Alpha Dashboard.

Centralizes all configuration settings including dashboard settings,
plot configurations, and styling options.
"""

from .dashboard_config import (
    TEMPLATE,
    DEFAULT_OPERATORS_FILE,
    DASHBOARD_SETTINGS,
    DATA_SETTINGS,
    UI_SETTINGS,
    FILTER_SETTINGS,
    PERFORMANCE_SETTINGS,
    ASSET_SETTINGS,
    CLUSTER_COLORS,
    AVAILABLE_REGIONS
)

from .plot_config import (
    DEFAULT_TEMPLATE,
    CHART_DIMENSIONS,
    COLOR_SCHEMES,
    PCA_FEATURE_COLORS,
    CHART_STYLE_DEFAULTS,
    HEATMAP_SETTINGS,
    TREEMAP_SETTINGS,
    ANIMATION_SETTINGS,
    get_plotly_layout_defaults,
    get_scatter_trace_defaults,
    get_bar_chart_defaults,
    get_heatmap_defaults
)

__all__ = [
    # Dashboard config
    'TEMPLATE',
    'DEFAULT_OPERATORS_FILE',
    'DASHBOARD_SETTINGS',
    'DATA_SETTINGS',
    'UI_SETTINGS',
    'FILTER_SETTINGS',
    'PERFORMANCE_SETTINGS',
    'ASSET_SETTINGS',
    'CLUSTER_COLORS',
    'AVAILABLE_REGIONS',

    # Plot config
    'DEFAULT_TEMPLATE',
    'CHART_DIMENSIONS',
    'COLOR_SCHEMES',
    'PCA_FEATURE_COLORS',
    'CHART_STYLE_DEFAULTS',
    'HEATMAP_SETTINGS',
    'TREEMAP_SETTINGS',
    'ANIMATION_SETTINGS',
    'get_plotly_layout_defaults',
    'get_scatter_trace_defaults',
    'get_bar_chart_defaults',
    'get_heatmap_defaults'
]