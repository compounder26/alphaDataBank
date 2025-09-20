"""
Dashboard Application Factory

Main application factory for the Alpha Analysis Dashboard.
Integrates all refactored components while preserving original functionality.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add the project root to the path for imports (preserve original behavior)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import dash
import dash_bootstrap_components as dbc

# Import refactored modules
from .config import DASHBOARD_SETTINGS, DEFAULT_OPERATORS_FILE, TEMPLATE
from .services import (
    load_all_region_data, create_analysis_operations, reset_analysis_operations,
    get_analysis_service, get_clustering_service
)
from .layouts import create_main_layout, create_analysis_tab_content, create_clustering_tab_content
from .callbacks import register_all_callbacks


# Global variables for backward compatibility (preserve original globals)
OPERATORS_FILE = DEFAULT_OPERATORS_FILE
DYNAMIC_OPERATORS_LIST = None
_analysis_ops_instance = None


def create_visualization_app(data: Optional[Dict[str, Any]] = None,
                           operators_list: Optional[List[str]] = None) -> dash.Dash:
    """
    Create a Dash app for visualizing the clustering data.

    PRESERVES EXACT FUNCTIONALITY from original visualization_server.py create_visualization_app()
    while using the refactored modular architecture.

    Args:
        data: Dictionary with clustering data (legacy support)
        operators_list: Optional list of operators to use (overrides operators file)

    Returns:
        Dash app
    """
    # Initialize analysis operations (preserve original logic)
    reset_analysis_operations()

    analysis_ops = None
    try:
        analysis_ops = create_analysis_operations(OPERATORS_FILE, operators_list)
        if operators_list:
            print(f"[OK] Analysis operations initialized with {len(operators_list)} dynamic operators")
        else:
            print("[OK] Analysis operations initialized with static operators file")
    except Exception as e:
        print(f"Warning: Could not initialize analysis operations: {e}")

    # Load all region data (preserve original behavior)
    all_region_data = load_all_region_data()

    # Extract metadata for dashboard display
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_regions = len(all_region_data)

    # Get available regions for the dropdown
    available_regions = list(all_region_data.keys())
    if not available_regions and data:
        # Fallback to single region if provided (legacy support)
        available_regions = [data.get('region', 'USA')]
        all_region_data = {available_regions[0]: data}

    print(f"Dashboard initialized with {total_regions} regions: {', '.join(available_regions)}")

    # Initialize Dash app with assets support
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        assets_folder='assets'  # Enable assets directory
    )

    # Set app title
    app.title = DASHBOARD_SETTINGS['title']

    # Create main layout using refactored layout modules
    app.layout = create_main_layout(
        available_regions=available_regions,
        all_region_data=all_region_data,
        analysis_ops_available=analysis_ops is not None
    )

    # Register all callbacks using refactored callback modules
    layout_functions = {
        'create_analysis_tab_content': lambda: create_analysis_tab_content(),
        'create_clustering_tab_content': lambda: create_clustering_tab_content()
    }

    register_all_callbacks(app, layout_functions)

    return app


def create_app() -> dash.Dash:
    """
    Create a Dash app with default settings for production deployment.
    This function is used by wsgi.py for WSGI server deployment.

    PRESERVES EXACT FUNCTIONALITY from original visualization_server.py create_app()

    Returns:
        Dash app configured for production
    """
    # Use default operators file (preserve original logic)
    global OPERATORS_FILE, DYNAMIC_OPERATORS_LIST
    OPERATORS_FILE = DEFAULT_OPERATORS_FILE

    # Try to load dynamic operators if JSON file exists (preserve original logic)
    if os.path.exists(OPERATORS_FILE) and OPERATORS_FILE.endswith('.json'):
        try:
            import json
            with open(OPERATORS_FILE, 'r') as f:
                operators_data = json.load(f)

            if isinstance(operators_data, dict) and 'operators' in operators_data:
                DYNAMIC_OPERATORS_LIST = [op['name'] for op in operators_data['operators']]
                print(f"[OK] Production mode: Loaded {len(DYNAMIC_OPERATORS_LIST)} dynamic operators")
            elif isinstance(operators_data, list):
                DYNAMIC_OPERATORS_LIST = operators_data
                print(f"[OK] Production mode: Loaded {len(DYNAMIC_OPERATORS_LIST)} operators")
        except Exception as e:
            print(f"[WARNING] Production mode: Using default operators (error loading JSON: {e})")

    # Create and return the app without clustering data (analysis-only mode)
    print("[STARTING] Initializing dashboard in production mode...")
    app = create_visualization_app(data=None, operators_list=DYNAMIC_OPERATORS_LIST)
    return app


def get_app_info() -> Dict[str, Any]:
    """
    Get information about the current app configuration.

    Returns:
        App configuration information
    """
    return {
        'operators_file': OPERATORS_FILE,
        'dynamic_operators_count': len(DYNAMIC_OPERATORS_LIST) if DYNAMIC_OPERATORS_LIST else 0,
        'analysis_ops_available': _analysis_ops_instance is not None,
        'dashboard_title': DASHBOARD_SETTINGS['title'],
        'template': TEMPLATE
    }


def reset_app_globals():
    """Reset global app state (useful for testing)."""
    global _analysis_ops_instance, DYNAMIC_OPERATORS_LIST
    _analysis_ops_instance = None
    DYNAMIC_OPERATORS_LIST = None
    reset_analysis_operations()


# Preserve original globals for backward compatibility
def set_operators_file(file_path: str):
    """Set the operators file path."""
    global OPERATORS_FILE
    OPERATORS_FILE = file_path


def set_dynamic_operators_list(operators: List[str]):
    """Set the dynamic operators list."""
    global DYNAMIC_OPERATORS_LIST
    DYNAMIC_OPERATORS_LIST = operators