"""
Main Dashboard Layout

Main dashboard structure with tabs, stores, and modal components.
Extracted from visualization_server.py with preserved functionality.
"""

from typing import List, Dict, Any
from dash import html, dcc
import dash_bootstrap_components as dbc

from ..components import (
    create_dashboard_header,
    create_loading_wrapper,
    create_main_detail_modal,
    create_datafield_detail_modal,
    create_tab_selector
)


def create_main_layout(available_regions: List[str] = None,
                      all_region_data: Dict[str, Any] = None,
                      analysis_ops_available: bool = False) -> dbc.Container:
    """
    Create the main dashboard layout with tabs and data stores.

    Args:
        available_regions: List of available clustering regions
        all_region_data: All region clustering data
        analysis_ops_available: Whether analysis operations are available

    Returns:
        Main dashboard container
    """
    # Set defaults
    if available_regions is None:
        available_regions = []
    if all_region_data is None:
        all_region_data = {}

    return dbc.Container([
        # Dashboard header
        create_dashboard_header(),

        # Main tabs
        dbc.Row([
            dbc.Col([
                create_tab_selector(
                    tabs=[
                        {'label': '📊 Expression Analysis', 'value': 'analysis-tab'},
                        {'label': '🎯 Alpha Clustering', 'value': 'clustering-tab'},
                    ],
                    element_id='main-tabs',
                    default_value='analysis-tab'
                ),

                # Tab content with loading
                create_loading_wrapper(
                    content=html.Div(id='tab-content'),
                    loading_id="loading-main-content"
                )
            ], width=12)
        ]),

        # Hidden stores for data (preserve exact structure from original)
        *create_data_stores(available_regions, all_region_data, analysis_ops_available),

        # Modal components for interactive features
        create_main_detail_modal(),
        create_datafield_detail_modal(),

    ], fluid=True)


def create_data_stores(available_regions: List[str],
                      all_region_data: Dict[str, Any],
                      analysis_ops_available: bool) -> List[dcc.Store]:
    """
    Create all data stores for the dashboard (preserve original structure exactly).

    Args:
        available_regions: List of available regions
        all_region_data: All region clustering data
        analysis_ops_available: Whether analysis operations are available

    Returns:
        List of Store components
    """
    return [
        # Analysis data stores
        dcc.Store(id='analysis-data', data={}),
        dcc.Store(id='preloaded-analysis-data', data={}),
        dcc.Store(id='analysis-filters', data={'region': None, 'universe': None, 'delay': None}),

        # Auto-trigger for initial data load
        dcc.Interval(id='initial-load-trigger', interval=1000, n_intervals=0, max_intervals=1),

        # Clustering data stores
        dcc.Store(id='all-region-data', data=all_region_data),
        dcc.Store(id='available-regions', data=available_regions),
        dcc.Store(id='selected-clustering-region', data=available_regions[0] if available_regions else None),

        # Current region data stores (for backward compatibility)
        dcc.Store(id='current-mds-data', data=[]),
        dcc.Store(id='mds-data-simple', data=[]),
        dcc.Store(id='mds-data-euclidean', data=[]),
        dcc.Store(id='mds-data-angular', data=[]),
        dcc.Store(id='current-tsne-data', data=[]),
        dcc.Store(id='current-umap-data', data=[]),
        dcc.Store(id='current-pca-data', data=[]),
        dcc.Store(id='current-pca-info', data={}),
        dcc.Store(id='current-metadata', data=[]),

        # Cluster profiles for interpretability
        dcc.Store(id='tsne-cluster-profiles', data={}),
        dcc.Store(id='umap-cluster-profiles', data={}),
        dcc.Store(id='pca-cluster-profiles', data={}),
        dcc.Store(id='main-cluster-profiles', data={}),

        # Pre-calculated heatmap data for all distance metrics
        dcc.Store(id='heatmap-data-simple', data={}),
        dcc.Store(id='heatmap-data-euclidean', data={}),
        dcc.Store(id='heatmap-data-angular', data={}),

        # UI state stores
        dcc.Store(id='selected-alpha', data=None),
        dcc.Store(id='analysis-ops', data={'available': analysis_ops_available}),

        # Store for view states and expanded lists
        dcc.Store(id='operators-view-mode', data='top20'),
        dcc.Store(id='datafields-view-mode', data='top20'),

        # Store for highlighting feature
        dcc.Store(id='operator-highlighted-alphas', data=[]),
        dcc.Store(id='datafield-highlighted-alphas', data=[]),
        dcc.Store(id='available-operators', data=[]),
        dcc.Store(id='available-datafields', data=[])
    ]


def create_empty_tab_content() -> html.Div:
    """
    Create empty tab content placeholder.

    Returns:
        Empty content div
    """
    return html.Div("Select a tab to begin analysis", className="text-center text-muted p-5")


def create_unavailable_analysis_content() -> dbc.Alert:
    """
    Create content for when analysis is unavailable.

    Returns:
        Alert component explaining unavailability
    """
    return dbc.Alert([
        html.H4("Analysis Unavailable", className="alert-heading"),
        html.P("The analysis system could not be initialized. Please check:"),
        html.Ul([
            html.Li("Dynamic operators/datafields data is available (run with --renew if needed)"),
            html.Li("Database connection is working"),
            html.Li("Analysis schema is initialized")
        ])
    ], color="warning")


def get_tab_content_id_mapping() -> Dict[str, str]:
    """
    Get mapping of tab values to content creation functions.

    Returns:
        Dictionary mapping tab values to layout function names
    """
    return {
        'analysis-tab': 'create_analysis_tab_content',
        'clustering-tab': 'create_clustering_tab_content'
    }