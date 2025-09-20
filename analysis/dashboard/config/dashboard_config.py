"""
Dashboard Configuration

Central configuration for the Alpha Analysis Dashboard.
"""

import os

# Default plotting template
TEMPLATE = 'plotly_white'

# Default file paths - now point to dynamic files, can be overridden by command line arguments
DEFAULT_OPERATORS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    'data', 'operators_dynamic.json'
)

# Dashboard settings
DASHBOARD_SETTINGS = {
    'title': 'Alpha Analysis Dashboard',
    'default_port': 8050,
    'auto_open_browser': True,
    'browser_delay_seconds': 2
}

# Data loading settings
DATA_SETTINGS = {
    'max_alpha_display': 100,  # Maximum alphas to display in modals
    'cache_timeout_minutes': 30,
    'preload_analysis_data': True
}

# UI settings
UI_SETTINGS = {
    'default_chart_height': '85vh',
    'modal_max_height': '500px',
    'scrollable_max_height': '400px',
    'charts_per_row': 2,
    'show_debug_info': False
}

# Filter settings
FILTER_SETTINGS = {
    'date_format': 'MM/DD/YYYY',
    'default_top_items': 20,
    'max_treemap_items': 50
}

# Performance settings
PERFORMANCE_SETTINGS = {
    'enable_caching': True,
    'max_concurrent_callbacks': 10,
    'chart_animation_duration': 300
}

# Asset settings
ASSET_SETTINGS = {
    'external_stylesheets': ['assets/css/custom.css'],
    'external_scripts': ['assets/js/custom.js'],
    'include_print_css': True
}

# Color schemes for clusters
CLUSTER_COLORS = [
    '#e41a1c',  # Red
    '#377eb8',  # Blue
    '#4daf4a',  # Green
    '#984ea3',  # Purple
    '#ff7f00',  # Orange
    '#ffff33',  # Yellow
    '#a65628',  # Brown
    '#f781bf',  # Pink
    '#66c2a5',  # Teal
    '#fc8d62',  # Light orange
    '#8da0cb',  # Light blue
    '#e78ac3',  # Light pink
    '#a6d854',  # Yellow-green
    '#ffd92f',  # Gold
    '#e5c494',  # Beige
    '#b3b3b3'   # Light gray
]

# Available regions (should match database configuration)
AVAILABLE_REGIONS = ['USA', 'EUR', 'JPN', 'CHN', 'AMR', 'ASI', 'GLB', 'HKG', 'KOR', 'TWN']