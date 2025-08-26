"""
Alpha Analysis & Clustering Visualization Server.

This module provides a comprehensive web interface for visualizing alpha clustering results
and analyzing operator/datafield usage patterns in alpha expressions.
It includes interactive dashboards for both clustering visualization and expression analysis.
"""
import os
import sys
import json
import argparse
import webbrowser
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import analysis modules
from analysis.analysis_operations import AnalysisOperations
from database.schema import initialize_analysis_database, get_connection
from sqlalchemy import text

# Default styling
TEMPLATE = 'plotly_white'

# Analysis configuration  
OPERATORS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'operators.txt')
DATAFIELDS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'all_datafields_comprehensive.csv')

def get_alpha_details_for_clustering(alpha_ids: List[str]) -> Dict[str, Dict]:
    """Fetch alpha details for clustering hover information."""
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

def load_all_region_data() -> Dict[str, Any]:
    """
    Load clustering data for all available regions.
    
    Returns:
        Dictionary with region names as keys and clustering data as values
    """
    import glob
    from config.database_config import REGIONS
    
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

def create_visualization_app(data: Optional[Dict[str, Any]] = None) -> dash.Dash:
    """
    Create a Dash app for visualizing the clustering data.
    
    Args:
        data: Dictionary with clustering data
        
    Returns:
        Dash app
    """
    # Initialize analysis operations
    analysis_ops = None
    try:
        analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
    except Exception as e:
        print(f"Warning: Could not initialize analysis operations: {e}")
    
    # Load all region data instead of single region
    all_region_data = load_all_region_data()
    
    # Extract metadata for dashboard display
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_regions = len(all_region_data)
    
    # Get available regions for the dropdown
    available_regions = list(all_region_data.keys())
    if not available_regions and data:
        # Fallback to single region if provided
        available_regions = [data.get('region', 'USA')]
        all_region_data = {available_regions[0]: data}
    
    print(f"Dashboard initialized with {total_regions} regions: {', '.join(available_regions)}")
    
    # Initialize Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Set app title  
    app.title = "Alpha Analysis Dashboard"
    
    # Create layout with tabs
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Alpha Analysis Dashboard", className="text-center my-4"),
                html.P(f"Generated on: {timestamp} | Analysis & Clustering Platform", 
                      className="text-center text-muted mb-4"),
            ], width=12)
        ]),
        
        # Main tabs
        dbc.Row([
            dbc.Col([
                dcc.Tabs(id='main-tabs', value='analysis-tab', children=[
                    dcc.Tab(label='📊 Expression Analysis', value='analysis-tab'),
                    dcc.Tab(label='🎯 Alpha Clustering', value='clustering-tab'),
                ], className='mb-4'),
                
                # Tab content with loading
                dcc.Loading(
                    id="loading-main-content",
                    type="default",
                    children=[html.Div(id='tab-content')]
                )
            ], width=12)
        ]),
        
        # Hidden stores for data
        dcc.Store(id='analysis-data', data={}),
        dcc.Store(id='preloaded-analysis-data', data={}),
        dcc.Store(id='analysis-filters', data={'region': None, 'universe': None, 'delay': None}),
        # Auto-trigger for initial data load
        dcc.Interval(id='initial-load-trigger', interval=1000, n_intervals=0, max_intervals=1),
        
        # Store all region clustering data
        dcc.Store(id='all-region-data', data=all_region_data),
        dcc.Store(id='available-regions', data=available_regions),
        dcc.Store(id='selected-clustering-region', data=available_regions[0] if available_regions else None),
        # Current region data (will be updated when region is selected)
        dcc.Store(id='current-mds-data', data=[]),
        dcc.Store(id='current-tsne-data', data=[]),
        dcc.Store(id='current-umap-data', data=[]),
        dcc.Store(id='current-pca-data', data=[]),
        dcc.Store(id='current-metadata', data=[]),
        dcc.Store(id='selected-alpha', data=None),
        dcc.Store(id='analysis-ops', data={'available': analysis_ops is not None}),
        # Store for view states and expanded lists
        dcc.Store(id='operators-view-mode', data='top20'),  # 'top20', 'all', 'usage-analysis'
        dcc.Store(id='datafields-view-mode', data='top20'),  # 'top20', 'all'
        
        # Modal components for interactive features
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id="detail-modal-title")),
            dbc.ModalBody([
                dcc.Loading(
                    id="modal-loading",
                    type="circle",
                    color="#007bff",
                    delay_show=0,      # Show immediately when loading starts
                    delay_hide=400,    # Keep visible for at least 400ms
                    children=html.Div(id="detail-modal-body")
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="detail-modal-close", className="ms-auto", n_clicks=0)
            ])
        ], id="detail-modal", is_open=False, size="lg"),
        
    ], fluid=True)
    
    # Callback to render tab content
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value'),
        State('analysis-ops', 'data')
    )
    def render_tab_content(active_tab, analysis_ops_data):
        if active_tab == 'analysis-tab':
            if not analysis_ops_data.get('available', False):
                return dbc.Alert([
                    html.H4("Analysis Unavailable", className="alert-heading"),
                    html.P("The analysis system could not be initialized. Please check:"),
                    html.Ul([
                        html.Li("operators.txt file exists in project root"),
                        html.Li("all_datafields_comprehensive.csv file exists in project root"),
                        html.Li("Database connection is working"),
                        html.Li("Analysis schema is initialized")
                    ])
                ], color="warning")
            
            return create_analysis_tab_content()
        
        elif active_tab == 'clustering-tab':
            return create_clustering_tab_content()
        
        return html.Div("Select a tab to begin analysis")
    
    def create_analysis_tab_content():
        """Create the analysis tab content."""
        return dbc.Row([
            # Filters sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analysis Filters"),
                    dbc.CardBody([
                        html.Div([
                            html.Label("Region:", className="form-label"),
                            dcc.Dropdown(
                                id='region-filter',
                                placeholder="All regions",
                                clearable=True
                            )
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Label("Universe:", className="form-label"),
                            dcc.Dropdown(
                                id='universe-filter',
                                placeholder="All universes",
                                clearable=True
                            )
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Label("Delay:", className="form-label"),
                            dcc.Dropdown(
                                id='delay-filter',
                                placeholder="All delays",
                                clearable=True
                            )
                        ], className="mb-3"),
                        
                        html.Div([
                            html.Label("Dates:", className="form-label"),
                            dcc.DatePickerRange(
                                id='dates-filter',
                                start_date_placeholder_text="Start date",
                                end_date_placeholder_text="End date",
                                clearable=True,
                                display_format='MM/DD/YYYY',
                                style={'width': '100%'}
                            )
                        ], className="mb-3"),
                        
                        dbc.Button("Apply Filters", id="apply-filters-btn", color="primary", className="w-100 mb-2")
                    ])
                ], className="mb-3"),
                
                # Analysis summary card
                dbc.Card([
                    dbc.CardHeader("Analysis Summary"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-analysis-summary",
                            type="default",
                            children=[html.Div(id="analysis-summary")]
                        )
                    ])
                ])
            ], width=3),
            
            # Main analysis content
            dbc.Col([
                dcc.Tabs(id='analysis-subtabs', value='operators-subtab', children=[
                    dcc.Tab(label='\u2699\ufe0f Operators', value='operators-subtab'),
                    dcc.Tab(label='\ud83d\udcc8 Datafields', value='datafields-subtab'),
                    dcc.Tab(label='\u2696\ufe0f Neutralization', value='neutralization-subtab'),
                    dcc.Tab(label='\ud83d\udd04 Cross Analysis', value='cross-subtab'),
                ], className='mb-3'),
                
                # Analysis subtab content with loading
                dcc.Loading(
                    id="loading-analysis-subtabs",
                    type="default",
                    children=[html.Div(id='analysis-subtab-content')]
                )
            ], width=9)
        ])
    
    def create_clustering_tab_content():
        """Create the clustering tab content."""
        return dbc.Row([
            dbc.Col([
                # Region Selector Card
                dbc.Card([
                    dbc.CardHeader("Region Selection"),
                    dbc.CardBody([
                        html.Label("Select Region:", className="form-label"),
                        dcc.Dropdown(
                            id='clustering-region-selector',
                            placeholder="Select clustering region...",
                            clearable=False
                        ),
                        html.Div(id='clustering-region-info', className="mt-2 small text-muted")
                    ])
                ], className="mb-3"),
                
                # Visualization Method Card
                dbc.Card([
                    dbc.CardHeader("Visualization Method"),
                    dbc.CardBody([
                        dcc.RadioItems(
                            id='method-selector',
                            options=[
                                {'label': 'MDS on Correlation Matrix', 'value': 'mds'},
                                {'label': 't-SNE on Performance Features', 'value': 'tsne'},
                                {'label': 'UMAP on Performance Features', 'value': 'umap'},
                                {'label': 'PCA on Performance Features', 'value': 'pca'},
                            ],
                            value='mds',
                            inline=True,
                            className="mb-3"
                        ),
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Alpha Details"),
                    dbc.CardBody([
                        html.Div(id='alpha-details', className="p-3")
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Clustering Visualization"),
                    dbc.CardBody([
                        # Clustering plot with loading
                        dcc.Loading(
                            id="loading-clustering-plot",
                            type="default",
                            children=[dcc.Graph(id='clustering-plot', style={'height': '85vh'})]
                        )
                    ])
                ])
            ], width=9)
        ])
    
    # Preload data callback - runs immediately on startup
    @app.callback(
        Output('preloaded-analysis-data', 'data'),
        Input('initial-load-trigger', 'n_intervals'),
        State('analysis-ops', 'data')
    )
    def preload_analysis_data(n_intervals, analysis_ops_data):
        if not analysis_ops_data.get('available', False):
            return {}
        
        try:
            # Preload all analysis data without filters for faster access
            temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
            results = temp_analysis_ops.get_analysis_summary()
            return results
        except Exception as e:
            print(f"Error preloading data: {e}")
            # The AnalysisOperations.get_analysis_summary() now handles missing table initialization
            # so this should work on retry. Return empty dict for now.
            return {}

    # Analysis callbacks
    @app.callback(
        [Output('region-filter', 'options'),
         Output('universe-filter', 'options'),
         Output('delay-filter', 'options'),
         Output('dates-filter', 'min_date_allowed'),
         Output('dates-filter', 'max_date_allowed'),
         Output('dates-filter', 'start_date'),
         Output('dates-filter', 'end_date')],
        Input('preloaded-analysis-data', 'data'),
        State('analysis-ops', 'data')
    )
    def populate_filter_options(preloaded_data, analysis_ops_data):
        if not analysis_ops_data.get('available', False) or not preloaded_data:
            return [], [], [], None, None, None, None
        
        try:
            # Use preloaded data to populate filter options faster
            metadata = preloaded_data.get('metadata', {})
            
            regions = [{'label': r, 'value': r} for r in sorted(metadata.get('regions', {}).keys())]
            universes = [{'label': u, 'value': u} for u in sorted(metadata.get('universes', {}).keys())]
            delays = [{'label': str(d), 'value': d} for d in sorted(metadata.get('delays', {}).keys())]
            
            # Get date range for date picker
            min_date = metadata.get('min_date_added')
            max_date = metadata.get('max_date_added')
            
            return regions, universes, delays, min_date, max_date, None, None
        except:
            return [], [], [], None, None, None, None
    
    # Region selector callback for clustering tab
    @app.callback(
        [Output('clustering-region-selector', 'options'),
         Output('clustering-region-selector', 'value')],
        Input('available-regions', 'data')
    )
    def populate_clustering_region_options(available_regions):
        if not available_regions:
            return [], None
        
        options = [{'label': region, 'value': region} for region in available_regions]
        default_value = available_regions[0] if available_regions else None
        return options, default_value
    
    # Callback to update clustering data when region is selected
    @app.callback(
        [Output('current-mds-data', 'data'),
         Output('current-tsne-data', 'data'),
         Output('current-umap-data', 'data'),
         Output('current-pca-data', 'data'),
         Output('current-metadata', 'data'),
         Output('clustering-region-info', 'children')],
        Input('clustering-region-selector', 'value'),
        State('all-region-data', 'data')
    )
    def update_clustering_data_for_region(selected_region, all_region_data):
        if not selected_region or not all_region_data or selected_region not in all_region_data:
            return [], [], [], [], [], "No data available"
        
        region_data = all_region_data[selected_region]
        
        # Convert coordinate dictionaries to DataFrames
        mds_coords = pd.DataFrame.from_dict(region_data.get('mds_coords', {}), orient='index')
        tsne_coords = pd.DataFrame.from_dict(region_data.get('tsne_coords', {}), orient='index')
        umap_coords = pd.DataFrame.from_dict(region_data.get('umap_coords', {}), orient='index')
        pca_coords = pd.DataFrame.from_dict(region_data.get('pca_coords', {}), orient='index')
        metadata = pd.DataFrame.from_dict(region_data.get('alpha_metadata', {}), orient='index')
        
        # Convert to dict format for storage
        mds_data = mds_coords.reset_index().to_dict('records') if not mds_coords.empty else []
        tsne_data = tsne_coords.reset_index().to_dict('records') if not tsne_coords.empty else []
        umap_data = umap_coords.reset_index().to_dict('records') if not umap_coords.empty else []
        pca_data = pca_coords.reset_index().to_dict('records') if not pca_coords.empty else []
        metadata_data = metadata.reset_index().to_dict('records') if not metadata.empty else []
        
        # Create info text
        alpha_count = region_data.get('alpha_count', 0)
        timestamp = region_data.get('timestamp', 'Unknown')
        info_text = f"{alpha_count} alphas | Generated: {timestamp}"
        
        return mds_data, tsne_data, umap_data, pca_data, metadata_data, info_text
    
    @app.callback(
        [Output('analysis-data', 'data'),
         Output('analysis-summary', 'children')],
        [Input('apply-filters-btn', 'n_clicks'),
         Input('initial-load-trigger', 'n_intervals')],
        [State('region-filter', 'value'),
         State('universe-filter', 'value'),
         State('delay-filter', 'value'),
         State('dates-filter', 'start_date'),
         State('dates-filter', 'end_date'),
         State('preloaded-analysis-data', 'data'),
         State('analysis-ops', 'data')]
    )
    def update_analysis_data(n_clicks, n_intervals, region, universe, delay, date_from, date_to, preloaded_data, analysis_ops_data):
        if not analysis_ops_data.get('available', False):
            return {}, "Analysis not available"
        
        try:
            # Use preloaded data if no filters are applied (faster)
            if not any([region, universe, delay, date_from, date_to]) and preloaded_data:
                results = preloaded_data
            else:
                # Get filtered analysis results
                temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
                results = temp_analysis_ops.get_analysis_summary(region, universe, delay, date_from, date_to)
            
            # Create summary
            metadata = results.get('metadata', {})
            total_alphas = metadata.get('total_alphas', 0)
            
            summary = dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong(f"Total Alphas: {total_alphas}")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Top Operators:"),
                    html.Ul([
                        html.Li(f"{op}: {count} alphas") 
                        for op, count in results.get('operators', {}).get('top_operators', [])[:5]
                    ])
                ]),
                dbc.ListGroupItem([
                    html.Strong("Top Datafields:"),
                    html.Ul([
                        html.Li(f"{df}: {count} alphas")
                        for df, count in results.get('datafields', {}).get('top_datafields', [])[:5]
                    ])
                ])
            ], flush=True)
            
            return results, summary
        except Exception as e:
            return {}, f"Error loading analysis: {str(e)}"
    
    @app.callback(
        Output('analysis-subtab-content', 'children'),
        [Input('analysis-subtabs', 'value'),
         Input('analysis-data', 'data'),
         Input('operators-view-mode', 'data'),
         Input('datafields-view-mode', 'data')]
    )
    def render_analysis_subtab_content(active_subtab, analysis_data, operators_view_mode, datafields_view_mode):
        if not analysis_data:
            return html.Div("Apply filters to load analysis data", className="text-muted text-center p-4")
        
        if active_subtab == 'operators-subtab':
            return create_operators_content(analysis_data, operators_view_mode)
        elif active_subtab == 'datafields-subtab':
            return create_datafields_content(analysis_data, datafields_view_mode)
        elif active_subtab == 'neutralization-subtab':
            return create_neutralization_content(analysis_data)
        elif active_subtab == 'cross-subtab':
            return create_cross_analysis_content(analysis_data)
        
        return html.Div()
    
    def create_operators_content(analysis_data, view_mode='top20'):
        """Create operators analysis content with different view modes."""
        operators_data = analysis_data.get('operators', {})
        top_operators = operators_data.get('top_operators', [])
        metadata = analysis_data.get('metadata', {})
        
        if not top_operators:
            return html.Div("No operator data available", className="text-muted text-center p-4")
        
        # Add view mode selector at the top
        view_selector = dbc.Card([
            dbc.CardHeader("View Options"),
            dbc.CardBody([
                dbc.RadioItems(
                    id='operators-view-selector',
                    options=[
                        {'label': 'Top 20 Most Used', 'value': 'top20'},
                        {'label': 'All Used Operators', 'value': 'all'},
                        {'label': 'Usage Analysis (All Platform Operators)', 'value': 'usage-analysis'}
                    ],
                    value=view_mode,
                    inline=True
                )
            ])
        ], className="mb-3")
        
        if view_mode == 'usage-analysis':
            return html.Div([view_selector, create_usage_analysis_content(analysis_data)])
        elif view_mode == 'all':
            return html.Div([view_selector, create_all_operators_content(analysis_data)])
        else:
            return html.Div([view_selector, create_top20_operators_content(analysis_data)])
    
    # Callback to handle operators view mode changes
    @app.callback(
        Output('operators-view-mode', 'data'),
        Input('operators-view-selector', 'value')
    )
    def update_operators_view_mode(selected_mode):
        return selected_mode
    
    def create_usage_analysis_content(analysis_data):
        """Create comprehensive usage analysis showing all platform operators."""
        operators_data = analysis_data.get('operators', {})
        used_operators = dict(operators_data.get('top_operators', []))
        
        # Load all platform operators
        try:
            with open(OPERATORS_FILE, 'r') as f:
                all_operators = [op.strip() for op in f.read().split(',')]
        except:
            return html.Div("Error loading operators file", className="text-danger")
        
        # Categorize operators
        frequently_used = [(op, count) for op, count in used_operators.items() if count >= 10]
        rarely_used = [(op, count) for op, count in used_operators.items() if 1 <= count < 10]
        never_used = [(op, 0) for op in all_operators if op not in used_operators]
        
        frequently_used.sort(key=lambda x: x[1], reverse=True)
        rarely_used.sort(key=lambda x: x[1], reverse=True)
        never_used.sort()
        
        return dbc.Row([
            dbc.Col([
                html.H5(f"📊 Operator Usage Summary ({len(all_operators)} total operators)", className="text-primary"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Frequently Used", className="text-success"),
                                html.H4(len(frequently_used), className="text-success"),
                                html.Small("≥10 uses")
                            ])
                        ])
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Rarely Used", className="text-warning"),
                                html.H4(len(rarely_used), className="text-warning"),
                                html.Small("1-9 uses")
                            ])
                        ])
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Never Used", className="text-danger"),
                                html.H4(len(never_used), className="text-danger"),
                                html.Small("0 uses")
                            ])
                        ])
                    ], width=4)
                ], className="mb-4"),
                
                # Detailed lists
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.Div([
                            dbc.Badge(f"{op} ({count})", color="success", className="me-1 mb-1")
                            for op, count in frequently_used
                        ])
                    ], title=f"Frequently Used Operators ({len(frequently_used)})", item_id="frequent"),
                    
                    dbc.AccordionItem([
                        html.Div([
                            dbc.Badge(f"{op} ({count})", color="warning", className="me-1 mb-1")
                            for op, count in rarely_used
                        ])
                    ], title=f"Rarely Used Operators ({len(rarely_used)})", item_id="rare"),
                    
                    dbc.AccordionItem([
                        html.Div([
                            dbc.Badge(op, color="danger", className="me-1 mb-1")
                            for op, _ in never_used  # Show ALL never used operators
                        ], style={'max-height': '400px', 'overflow-y': 'auto'})  # Add scroll for very long lists
                    ], title=f"Never Used Operators ({len(never_used)})", item_id="never")
                ], active_item="frequent")
            ])
        ])
    
    def create_all_operators_content(analysis_data):
        """Create view showing all used operators with their counts."""
        operators_data = analysis_data.get('operators', {})
        all_used_operators = operators_data.get('top_operators', [])
        metadata = analysis_data.get('metadata', {})
        
        # Create bar chart with all operators
        df = pd.DataFrame({
            'operator': [op for op, _ in all_used_operators],
            'count': [count for _, count in all_used_operators]
        })
        
        fig = px.bar(
            df,
            x='count',
            y='operator',
            orientation='h',
            title=f"All {len(all_used_operators)} Used Operators (Click bars for details)",
            labels={'count': 'Number of Alphas Using', 'operator': 'Operator'},
            template=TEMPLATE,
            height=max(800, len(all_used_operators) * 25)  # Dynamic height based on operator count
        )
        fig.update_layout(clickmode='event+select')
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Used in %{x} alphas<br>Click for details<extra></extra>',
            marker_color='steelblue'
        )
        
        total_unique_ops = len(operators_data.get('unique_usage', {}))
        total_nominal = sum(operators_data.get('nominal_usage', {}).values())
        total_alphas = metadata.get('total_alphas', 0)
        
        return dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id='all-operators-chart', figure=fig, style={'height': f'{fig.layout.height}px'})
                ], id="all-operators-container", style={
                    'overflow': 'auto',
                    'border': '2px solid #ccc',
                    'border-radius': '4px',
                    'min-width': '500px',
                    'min-height': '700px',
                    'max-width': '100%'
                }),
            ], width=12)
        ])
    
    def create_top20_operators_content(analysis_data):
        """Create the original top 20 operators view."""
        operators_data = analysis_data.get('operators', {})
        top_operators = operators_data.get('top_operators', [])
        metadata = analysis_data.get('metadata', {})
        
        # Create DataFrame for proper Plotly usage
        import pandas as pd
        df = pd.DataFrame({
            'operator': [op for op, _ in top_operators[:20]],
            'count': [count for _, count in top_operators[:20]]
        })
        
        # Create interactive bar chart
        fig = px.bar(
            df,
            x='count',
            y='operator',
            orientation='h',
            title="Top 20 Most Used Operators (Click bars for details)",
            labels={'count': 'Number of Alphas Using', 'operator': 'Operator'},
            template=TEMPLATE
        )
        fig.update_layout(height=600, clickmode='event+select')
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Used in %{x} alphas<br>Click for details<extra></extra>',
            marker_color='steelblue'
        )
        
        # Calculate statistics
        total_unique_ops = len(operators_data.get('unique_usage', {}))
        total_nominal = sum(operators_data.get('nominal_usage', {}).values())
        total_alphas = metadata.get('total_alphas', 0)
        avg_ops_per_alpha = total_nominal / total_alphas if total_alphas > 0 else 0
        
        return dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Graph(id='operators-chart', figure=fig)
                ], id="operators-chart-container", style={
                    'overflow': 'hidden',
                    'border': '2px dashed #ccc',
                    'border-radius': '4px',
                    'min-width': '400px',
                    'min-height': '500px',
                    'max-width': '100%',
                    'position': 'relative'
                }),
            ], width=8),
            dbc.Col([
                html.H5("📊 Usage Statistics"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong(f"Total Unique Operators: "),
                        html.Span(f"{total_unique_ops}", className="badge bg-primary ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong(f"Total Operator Instances: "),
                        html.Span(f"{total_nominal:,}", className="badge bg-success ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong(f"Average per Alpha: "),
                        html.Span(f"{avg_ops_per_alpha:.1f}", className="badge bg-info ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong(f"Total Alphas: "),
                        html.Span(f"{total_alphas}", className="badge bg-secondary ms-2")
                    ])
                ], flush=True, className="mb-3"),
                
                html.Hr(),
                html.H6("💡 Interaction Tips"),
                dbc.Alert([
                    html.Ul([
                        html.Li("Click on bars to see breakdown by region/universe/delay"),
                        html.Li("Hover for usage details"),
                        html.Li("Modal shows alpha expressions using the operator")
                    ], className="mb-0")
                ], color="light")
            ], width=4)
        ])
    
    def create_datafields_content(analysis_data, view_mode='top20'):
        """Create datafields analysis content with different view modes."""
        datafields_data = analysis_data.get('datafields', {})
        top_datafields = datafields_data.get('top_datafields', [])
        by_category = datafields_data.get('by_category', {})
        metadata = analysis_data.get('metadata', {})
        
        if not top_datafields:
            return html.Div("No datafield data available", className="text-muted text-center p-4")
        
        # Add view mode selector
        view_selector = dbc.Card([
            dbc.CardHeader("View Options"),
            dbc.CardBody([
                dbc.RadioItems(
                    id='datafields-view-selector',
                    options=[
                        {'label': 'Top 20 Most Used', 'value': 'top20'},
                        {'label': 'All Used Datafields', 'value': 'all'},
                        {'label': 'All Used Datasets', 'value': 'datasets'}
                    ],
                    value=view_mode,
                    inline=True
                )
            ])
        ], className="mb-3")
        
        if view_mode == 'all':
            return html.Div([view_selector, create_all_datafields_content(analysis_data)])
        elif view_mode == 'datasets':
            return html.Div([view_selector, create_all_datasets_content(analysis_data)])
        else:
            return html.Div([view_selector, create_top20_datafields_content(analysis_data)])
    
    # Callback to handle datafields view mode changes
    @app.callback(
        Output('datafields-view-mode', 'data'),
        Input('datafields-view-selector', 'value')
    )
    def update_datafields_view_mode(selected_mode):
        return selected_mode
    
    def create_all_datafields_content(analysis_data):
        """Create view showing all used datafields with their counts."""
        datafields_data = analysis_data.get('datafields', {})
        all_used_datafields = datafields_data.get('top_datafields', [])
        by_category = datafields_data.get('by_category', {})
        metadata = analysis_data.get('metadata', {})
        
        # Create bar chart with all datafields
        df1 = pd.DataFrame({
            'datafield': [df for df, _ in all_used_datafields],
            'count': [count for _, count in all_used_datafields]
        })
        
        # Create interactive bar chart for all datafields
        fig1 = px.bar(
            df1,
            x='count',
            y='datafield',
            orientation='h',
            title=f"All {len(all_used_datafields)} Used Datafields (Click bars for details)",
            labels={'count': 'Number of Alphas Using', 'datafield': 'Datafield'},
            template=TEMPLATE,
            height=max(800, len(all_used_datafields) * 20)  # Dynamic height
        )
        fig1.update_layout(clickmode='event+select')
        fig1.update_traces(
            hovertemplate='<b>%{y}</b><br>Used in %{x} alphas<br>Click for details<extra></extra>',
            marker_color='darkgreen'
        )
        
        return dbc.Row([
            dbc.Col([
                dcc.Graph(id='all-datafields-chart', figure=fig1, style={'height': f'{fig1.layout.height}px'}),
            ], width=12)
        ])
    
    def create_all_datasets_content(analysis_data):
        """Create view showing all used datasets with their counts."""
        datafields_data = analysis_data.get('datafields', {})
        unique_usage = datafields_data.get('unique_usage', {})
        metadata = analysis_data.get('metadata', {})
        
        if not unique_usage:
            return html.Div("No dataset data available", className="text-muted text-center p-4")
        
        # Calculate dataset usage counts
        dataset_counts = {}
        try:
            temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
            print(f"Processing {len(unique_usage)} datafields for dataset analysis")
            
            for df, alphas in unique_usage.items():
                if df in temp_analysis_ops.parser.datafields:
                    dataset_id = temp_analysis_ops.parser.datafields[df]['dataset_id']
                    if dataset_id not in dataset_counts:
                        dataset_counts[dataset_id] = 0
                    dataset_counts[dataset_id] += len(alphas)
            
            datasets_list = list(sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True))
            print(f"Found {len(datasets_list)} datasets: {[name for name, _ in datasets_list[:10]]}...")
        except Exception as e:
            print(f"Error processing datasets: {e}")
            return html.Div("Error processing dataset data", className="text-danger text-center p-4")
        
        if not datasets_list:
            return html.Div("No dataset data found", className="text-muted text-center p-4")
        
        # Create DataFrame for datasets chart
        df = pd.DataFrame({
            'dataset': [dataset for dataset, _ in datasets_list],
            'count': [count for _, count in datasets_list]
        })
        
        # Create interactive bar chart for all datasets
        fig = px.bar(
            df,
            x='count',
            y='dataset',
            orientation='h',
            title=f"All Used Datasets ({len(datasets_list)} total) - Click bars for details",
            labels={'count': 'Total Datafield Instances', 'dataset': 'Dataset ID'},
            template=TEMPLATE
        )
        
        # Calculate dynamic height for proper visibility
        min_height = 400
        bar_height = max(25, int(400 / len(datasets_list))) if len(datasets_list) > 0 else 25
        calculated_height = max(min_height, len(datasets_list) * bar_height + 100)
        
        fig.update_layout(
            height=calculated_height, 
            clickmode='event+select',
            yaxis={'categoryorder': 'total ascending'}
        )
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>%{x} total datafield instances<br>Click for details<extra></extra>',
            marker_color='steelblue'
        )
        
        return dbc.Row([
            dbc.Col([
                dcc.Graph(id='all-datasets-chart', figure=fig, style={'height': f'{fig.layout.height}px'}),
            ], width=12)
        ])
    
    def create_top20_datafields_content(analysis_data):
        """Create the original top 20 datafields view."""
        datafields_data = analysis_data.get('datafields', {})
        top_datafields = datafields_data.get('top_datafields', [])
        by_category = datafields_data.get('by_category', {})
        metadata = analysis_data.get('metadata', {})
        
        # Create DataFrame for datafields chart
        df1 = pd.DataFrame({
            'datafield': [df for df, _ in top_datafields[:20]],
            'count': [count for _, count in top_datafields[:20]]
        })
        
        # Create interactive bar chart for top datafields
        fig1 = px.bar(
            df1,
            x='count',
            y='datafield',
            orientation='h',
            title="Top 20 Most Used Datafields (Click bars for details)",
            labels={'count': 'Number of Alphas Using', 'datafield': 'Datafield'},
            template=TEMPLATE
        )
        fig1.update_layout(height=500, clickmode='event+select')
        fig1.update_traces(
            hovertemplate='<b>%{y}</b><br>Used in %{x} alphas<br>Click for details<extra></extra>',
            marker_color='darkgreen'
        )
        
        # Create enhanced pie chart for categories
        category_counts = {}
        dataset_counts = {}
        
        # Calculate category and dataset counts
        try:
            temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
            unique_usage = datafields_data.get('unique_usage', {})
            print(f"Processing {len(unique_usage)} datafields for dataset/category analysis")
            
            for df, alphas in unique_usage.items():
                if df in temp_analysis_ops.parser.datafields:
                    dataset_id = temp_analysis_ops.parser.datafields[df]['dataset_id']
                    category = temp_analysis_ops.parser.datafields[df]['data_category']
                    
                    # Count dataset usage (including empty dataset_ids)
                    if dataset_id is not None:  # More permissive - only exclude None
                        dataset_key = dataset_id if dataset_id else 'unknown'
                        dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + len(alphas)
                    
                    # Count category usage  
                    if category:
                        category_counts[category] = category_counts.get(category, 0) + len(alphas)
                else:
                    # For datafields not in parser, try to extract from name
                    if '.' in df:
                        dataset_key = df.split('.')[0]
                        dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + len(alphas)
            
            print(f"Found {len(dataset_counts)} datasets: {list(dataset_counts.keys())[:10]}...")
            print(f"Top datasets: {sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
            
        except Exception as e:
            print(f"Error calculating category/dataset counts: {e}")
            import traceback
            traceback.print_exc()
            
            # Improved fallback logic
            print("Falling back to by_category data extraction...")
            for category, datafields in by_category.items():
                total_alphas = set()
                for datafield, alphas in datafields.items():
                    total_alphas.update(alphas)
                    # Extract dataset from datafield name
                    if '.' in datafield:
                        dataset_key = datafield.split('.')[0]
                    else:
                        # Try to extract from first part before underscore
                        parts = datafield.split('_')
                        dataset_key = parts[0] if len(parts) > 1 else 'unknown'
                    
                    dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + len(alphas)
                
                category_counts[category] = len(total_alphas)
        
        # Create enhanced category pie chart with better readability
        if category_counts:
            # Truncate category names for better display
            truncated_names = [name[:15] + "..." if len(name) > 15 else name for name in category_counts.keys()]
            
            fig2 = px.pie(
                values=list(category_counts.values()),
                names=truncated_names,
                title="Usage by Category",
                template=TEMPLATE
            )
            fig2.update_layout(
                height=300,
                font=dict(size=10),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01,
                    font=dict(size=9)
                )
            )
            fig2.update_traces(
                textposition='auto',
                textinfo='percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
                hoverinfo='label+percent+value'
            )
        else:
            fig2 = go.Figure()
            fig2.update_layout(
                title="No Category Data Available",
                height=300,
                annotations=[dict(text="No data to display", x=0.5, y=0.5, showarrow=False)]
            )
        
        # Create dataset treemap
        if dataset_counts:
            # Limit to top 20 datasets for readability
            sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"Creating treemap with {len(sorted_datasets)} datasets")
            
            # Create treemap with proper parameters
            fig3 = px.treemap(
                names=[name for name, _ in sorted_datasets],
                parents=["" for _ in sorted_datasets],  # All datasets are children of root
                values=[value for _, value in sorted_datasets],
                title="Top 20 Datasets by Usage (Click to zoom, use bar charts for details)"
            )
            fig3.update_traces(
                textinfo="label+value",
                textposition="middle center"
            )
            fig3.update_layout(height=450)
        else:
            print("WARNING: dataset_counts is empty - treemap will be empty")
            fig3 = go.Figure()
        
        # Calculate statistics
        total_unique_dfs = len(datafields_data.get('unique_usage', {}))
        total_nominal = sum(datafields_data.get('nominal_usage', {}).values())
        total_alphas = metadata.get('total_alphas', 0)
        avg_dfs_per_alpha = total_nominal / total_alphas if total_alphas > 0 else 0
        
        
        # Create plot containers
        plot_containers = [
                # Main datafields chart
                html.Div([
                    html.Div([
                        html.H6("🔄 Top 20 Datafields", className="mb-0"),
                        html.Div([], style={'display': 'flex'})
                    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                    dcc.Graph(id='datafields-chart', figure=fig1, style={'height': 'calc(100% - 40px)'})
                ], style={
                    'height': '100%',
                    'border': '1px solid #ddd',
                    'border-radius': '8px',
                    'padding': '10px',
                    'background-color': 'white'
                }),
                
                # Statistics panel
                html.Div([
                    html.H6("📈 Usage Statistics", className="mb-2"),
                    dbc.ListGroup([
                        dbc.ListGroupItem([
                            html.Strong(f"Total Unique Datafields: "),
                            html.Span(f"{total_unique_dfs}", className="badge bg-primary ms-2")
                        ]),
                        dbc.ListGroupItem([
                            html.Strong(f"Total Datafield Instances: "),
                            html.Span(f"{total_nominal:,}", className="badge bg-success ms-2")
                        ]),
                        dbc.ListGroupItem([
                            html.Strong(f"Average per Alpha: "),
                            html.Span(f"{avg_dfs_per_alpha:.1f}", className="badge bg-info ms-2")
                        ]),
                        dbc.ListGroupItem([
                            html.Strong(f"Dataset Categories: "),
                            html.Span(f"{len(category_counts)}", className="badge bg-warning ms-2")
                        ])
                    ], flush=True)
                ], style={
                    'height': '100%',
                    'border': '1px solid #ddd',
                    'border-radius': '8px',
                    'padding': '10px',
                    'background-color': 'white'
                }),
                
                # Category pie chart
                html.Div([
                    html.Div([
                        html.H6("📊 Usage by Category", className="mb-0"),
                        html.Div([], style={'display': 'flex'})
                    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                    dcc.Graph(id='category-chart', figure=fig2, style={'height': 'calc(100% - 40px)'})
                ], style={
                    'height': '100%',
                    'border': '1px solid #ddd',
                    'border-radius': '8px',
                    'padding': '10px',
                    'background-color': 'white'
                }),
                
                # Dataset treemap
                html.Div([
                    html.Div([
                        html.H6("🗂️ Top 20 Datasets", className="mb-0"),
                        html.Div([], style={'display': 'flex'})
                    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '10px'}),
                    dcc.Graph(id='dataset-treemap', figure=fig3, style={'height': 'calc(100% - 40px)'})
                ], style={
                    'height': '100%',
                    'border': '1px solid #ddd',
                    'border-radius': '8px',
                    'padding': '10px',
                    'background-color': 'white'
                })
        ]
        
        # Create enhanced responsive layout with inline styles
        main_content = html.Div([
            # Grid container with enhanced plots
            html.Div([
                # Main datafields chart
                html.Div([
                    plot_containers[0]
                ], style={
                    'grid-column': '1 / 3',
                    'grid-row': '1',
                    'height': '600px'
                }),
                
                # Statistics panel
                html.Div([
                    plot_containers[1]
                ], style={
                    'grid-column': '1',
                    'grid-row': '2',
                    'height': '400px'
                }),
                
                # Category pie chart
                html.Div([
                    plot_containers[2]
                ], style={
                    'grid-column': '2',
                    'grid-row': '2',
                    'height': '400px'
                }),
                
                # Dataset treemap
                html.Div([
                    plot_containers[3]
                ], style={
                    'grid-column': '1 / 3',
                    'grid-row': '3',
                    'height': '500px'
                }),
            ], style={
                'display': 'grid',
                'grid-template-columns': '1fr 1fr',
                'grid-template-rows': 'auto auto auto',
                'gap': '15px',
                'padding': '15px',
                'min-height': '1200px'
            }),
            
        ])
        
        return main_content
    
    def create_cross_analysis_content(analysis_data):
        """Create cross-analysis content."""
        return dbc.Row([
            dbc.Col([
                html.H4("Cross-Analysis Dashboard"),
                html.P("Coming soon: Correlation between operator and datafield usage patterns."),
                dbc.Alert([
                    html.H6("Available Features:", className="alert-heading"),
                    html.Ul([
                        html.Li("Operator-Datafield co-occurrence matrix"),
                        html.Li("Alpha complexity analysis"),
                        html.Li("Strategy pattern identification"),
                        html.Li("Performance correlation with expression complexity")
                    ])
                ], color="info")
            ])
        ])
    
    def create_neutralization_content(analysis_data):
        """Create neutralization analysis content."""
        metadata = analysis_data.get('metadata', {})
        neutralizations = metadata.get('neutralizations', {})
        total_alphas = metadata.get('total_alphas', 0)
        
        if not neutralizations:
            return dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H4("No Neutralization Data Available", className="alert-heading"),
                        html.P("Neutralization information is not available in the current dataset."),
                    ], color="warning")
                ])
            ])
        
        # Create pie chart for neutralization breakdown
        labels = list(neutralizations.keys())
        values = list(neutralizations.values())
        
        pie_fig = px.pie(
            values=values,
            names=labels,
            title="Alpha Distribution by Neutralization Type",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        pie_fig.update_traces(textposition='inside', textinfo='percent+label')
        pie_fig.update_layout(showlegend=True, height=500)
        
        # Create bar chart for neutralization breakdown
        bar_fig = px.bar(
            x=values,
            y=labels,
            orientation='h',
            title="Neutralization Usage Count",
            color=values,
            color_continuous_scale='viridis'
        )
        bar_fig.update_traces(texttemplate='%{x}', textposition='outside')
        bar_fig.update_layout(
            xaxis_title="Number of Alphas",
            yaxis_title="Neutralization Type",
            showlegend=False,
            height=max(400, len(labels) * 50)
        )
        
        return dbc.Row([
            dbc.Col([
                dcc.Graph(id="neutralization-pie-chart", figure=pie_fig)
            ], width=6),
            dbc.Col([
                dcc.Graph(id="neutralization-bar-chart", figure=bar_fig)
            ], width=6),
            dbc.Col([
                html.H5("🎯 Neutralization Statistics"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong(f"Total Neutralization Types: "),
                        html.Span(f"{len(neutralizations)}", className="badge bg-primary ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong(f"Most Common: "),
                        html.Span(f"{max(neutralizations, key=neutralizations.get) if neutralizations else 'N/A'}", className="badge bg-success ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong(f"Least Common: "),
                        html.Span(f"{min(neutralizations, key=neutralizations.get) if neutralizations else 'N/A'}", className="badge bg-warning ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong(f"Total Alphas: "),
                        html.Span(f"{total_alphas}", className="badge bg-secondary ms-2")
                    ])
                ], className="mb-3"),
                
                html.H6("📋 Detailed Breakdown"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong(f"{neut_type}: "),
                        html.Span(f"{count} alphas", className="me-2"),
                        html.Span(f"({count/total_alphas*100:.1f}%)" if total_alphas > 0 else "(0%)", className="text-muted small")
                    ]) for neut_type, count in sorted(neutralizations.items(), key=lambda x: x[1], reverse=True)
                ])
            ], width=12, className="mt-4")
        ])
    
    
    # Callback to handle "Show all alphas" button clicks
    @app.callback(
        Output({'type': 'alpha-list-container', 'operator': dash.MATCH}, 'children'),
        Input({'type': 'show-all-alphas-btn', 'operator': dash.MATCH}, 'n_clicks'),
        [State('analysis-data', 'data'),
         State({'type': 'show-all-alphas-btn', 'operator': dash.MATCH}, 'id')]
    )
    def expand_alpha_list(n_clicks, analysis_data, btn_id):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        
        operator = btn_id['operator']
        operators_data = analysis_data.get('operators', {})
        alphas_using = operators_data.get('unique_usage', {}).get(operator, [])
        
        if len(alphas_using) <= 50:
            raise dash.exceptions.PreventUpdate
        
        # Return expanded list with all alphas and a "show less" button
        return [
            html.Div([
                dbc.Badge(
                    alpha_id,
                    id={'type': 'alpha-badge', 'index': alpha_id},
                    color="primary",
                    className="me-1 mb-1",
                    style={'cursor': 'pointer'}
                ) for alpha_id in alphas_using  # Show all alpha IDs
            ], style={'max-height': '200px', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'padding': '10px', 'border-radius': '4px'}),
            html.Div([
                dbc.Button(
                    "Show less",
                    id={'type': 'show-less-alphas-btn', 'operator': operator},
                    color="secondary",
                    size="sm",
                    className="mt-2"
                )
            ])
        ]
    
    # Callback to handle "Show less" button clicks
    @app.callback(
        Output({'type': 'alpha-list-container', 'operator': dash.MATCH}, 'children', allow_duplicate=True),
        Input({'type': 'show-less-alphas-btn', 'operator': dash.MATCH}, 'n_clicks'),
        [State('analysis-data', 'data'),
         State({'type': 'show-less-alphas-btn', 'operator': dash.MATCH}, 'id')],
        prevent_initial_call=True
    )
    def collapse_alpha_list(n_clicks, analysis_data, btn_id):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        
        operator = btn_id['operator']
        operators_data = analysis_data.get('operators', {})
        alphas_using = operators_data.get('unique_usage', {}).get(operator, [])
        
        # Return all alphas in scrollable container
        return [
            html.Div([
                dbc.Badge(
                    alpha_id,
                    id={'type': 'alpha-badge', 'index': alpha_id},
                    color="primary",
                    className="me-1 mb-1",
                    style={'cursor': 'pointer'}
                ) for alpha_id in alphas_using  # Show all alpha IDs
            ], style={'max-height': '200px', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'padding': '10px', 'border-radius': '4px'}),
            html.Div([
                dbc.Button(
                    "Show less",
                    id={'type': 'show-less-alphas-btn', 'operator': operator},
                    color="secondary",
                    size="sm",
                    className="mt-2"
                )
            ])
        ]
    
    
    # Callback to handle "Show all alphas" button clicks for datafields
    @app.callback(
        Output({'type': 'alpha-list-container-df', 'datafield': dash.MATCH}, 'children'),
        Input({'type': 'show-all-alphas-btn-df', 'datafield': dash.MATCH}, 'n_clicks'),
        [State('analysis-data', 'data'),
         State({'type': 'show-all-alphas-btn-df', 'datafield': dash.MATCH}, 'id')]
    )
    def expand_alpha_list_df(n_clicks, analysis_data, btn_id):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        
        datafield = btn_id['datafield']
        datafields_data = analysis_data.get('datafields', {})
        alphas_using = datafields_data.get('unique_usage', {}).get(datafield, [])
        
        if len(alphas_using) <= 50:
            raise dash.exceptions.PreventUpdate
        
        # Return expanded list with all alphas and a "show less" button
        return [
            html.Div([
                dbc.Badge(
                    alpha_id,
                    id={'type': 'alpha-badge', 'index': alpha_id},
                    color="success",
                    className="me-1 mb-1",
                    style={'cursor': 'pointer'}
                ) for alpha_id in alphas_using  # Show all alpha IDs
            ], style={'max-height': '200px', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'padding': '10px', 'border-radius': '4px'}),
            html.Div([
                dbc.Button(
                    "Show less",
                    id={'type': 'show-less-alphas-btn-df', 'datafield': datafield},
                    color="secondary",
                    size="sm",
                    className="mt-2"
                )
            ])
        ]
    
    # Callback to handle "Show less" button clicks for datafields
    @app.callback(
        Output({'type': 'alpha-list-container-df', 'datafield': dash.MATCH}, 'children', allow_duplicate=True),
        Input({'type': 'show-less-alphas-btn-df', 'datafield': dash.MATCH}, 'n_clicks'),
        [State('analysis-data', 'data'),
         State({'type': 'show-less-alphas-btn-df', 'datafield': dash.MATCH}, 'id')],
        prevent_initial_call=True
    )
    def collapse_alpha_list_df(n_clicks, analysis_data, btn_id):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        
        datafield = btn_id['datafield']
        datafields_data = analysis_data.get('datafields', {})
        alphas_using = datafields_data.get('unique_usage', {}).get(datafield, [])
        
        # Return all alphas in scrollable container
        return [
            html.Div([
                dbc.Badge(
                    alpha_id,
                    id={'type': 'alpha-badge', 'index': alpha_id},
                    color="success",
                    className="me-1 mb-1",
                    style={'cursor': 'pointer'}
                ) for alpha_id in alphas_using  # Show all alpha IDs
            ], style={'max-height': '200px', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'padding': '10px', 'border-radius': '4px'}),
            html.Div([
                dbc.Button(
                    "Show less",
                    id={'type': 'show-less-alphas-btn-df', 'datafield': datafield},
                    color="secondary",
                    size="sm",
                    className="mt-2"
                )
            ])
        ]
    
    
    
    
    

    # Clustering callback (updated for multi-region support)
    @app.callback(
        Output('clustering-plot', 'figure'),
        Input('method-selector', 'value'),
        Input('current-mds-data', 'data'),
        Input('current-tsne-data', 'data'),
        Input('current-umap-data', 'data'),
        Input('current-pca-data', 'data'),
        Input('selected-alpha', 'data')
    )
    def update_plot(method, mds_data, tsne_data, umap_data, pca_data, selected_alpha):
        # Select the appropriate data based on the method
        if method == 'mds':
            plot_data = pd.DataFrame(mds_data)
            title = "MDS on Correlation Matrix"
        elif method == 'tsne':
            plot_data = pd.DataFrame(tsne_data)
            title = "t-SNE on Performance Features"
        elif method == 'umap':
            plot_data = pd.DataFrame(umap_data)
            title = "UMAP on Performance Features"
        elif method == 'pca':
            plot_data = pd.DataFrame(pca_data)
            title = "PCA on Performance Features"
        
        # Fetch alpha details for enhanced hover information
        if not plot_data.empty:
            alpha_ids = plot_data['index'].tolist()
            alpha_details = get_alpha_details_for_clustering(alpha_ids)
            
            # Create enhanced hover text
            hover_texts = []
            for alpha_id in alpha_ids:
                details = alpha_details.get(alpha_id, {})
                code = details.get('code', '')
                
                # Format expression with line breaks for better readability
                # Split long expressions into multiple lines at logical break points
                formatted_code = code
                if len(code) > 60:
                    # Break at semicolons, commas, and operators for better readability
                    formatted_code = code.replace(';', ';<br>').replace(', ', ',<br>')
                    # Limit to reasonable length for hover
                    if len(formatted_code) > 300:
                        formatted_code = formatted_code[:300] + '...'
                
                # Format hover text with better structure
                hover_text = f"""<b>{alpha_id}</b><br>
                Expression: <br>{formatted_code}<br>
                <br>
                Universe: {details.get('universe', 'N/A')}<br>
                Delay: {details.get('delay', 'N/A')}<br>
                Sharpe: {details.get('is_sharpe', 0):.3f}<br>
                Fitness: {details.get('is_fitness', 0):.3f}<br>
                Returns: {details.get('is_returns', 0):.3f}<br>
                Neutralization: {details.get('neutralization', 'N/A')}<br>
                Decay: {details.get('decay', 'N/A')}<br>
                Region: {details.get('region_name', 'N/A')}"""
                hover_texts.append(hover_text)
        else:
            hover_texts = []
        
        # Create scatter plot with enhanced hover
        fig = px.scatter(
            plot_data, 
            x='x', 
            y='y',
            hover_name='index',  # Use alpha_id as hover name
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            template=TEMPLATE
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            hovermode='closest',
            clickmode='event+select'
        )
        
        # Update traces with enhanced hover information
        fig.update_traces(
            marker=dict(size=10, opacity=0.8),
            hovertext=hover_texts,
            customdata=alpha_ids,  # Store clean alpha_ids for click handling
            hovertemplate='%{hovertext}<br>Click to view on WorldQuant Brain<extra></extra>'
        )
        
        # Highlight selected alpha if any
        if selected_alpha:
            selected_index = selected_alpha.get('index')
            if selected_index in plot_data['index'].values:
                selected_point = plot_data[plot_data['index'] == selected_index]
                
                fig.add_trace(go.Scatter(
                    x=selected_point['x'],
                    y=selected_point['y'],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15,
                        line=dict(width=2, color='black')
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        fig.update_layout(
            annotations=[
                dict(
                    text="Click on a point to open its WorldQuant Brain link",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.1
                )
            ]
        )
        
        return fig
    
    # Callback to update alpha details when a point is clicked
    @app.callback(
        Output('alpha-details', 'children'),
        Output('selected-alpha', 'data'),
        Input('clustering-plot', 'clickData'),
        State('current-metadata', 'data')
    )
    def display_alpha_details(clickData, metadata_data):
        if not clickData:
            return "Click on a point to see alpha details.", None
        
        # Get the alpha_id from the clicked point
        point = clickData['points'][0]
        alpha_id = point.get('customdata', point.get('hovertext', 'unknown'))
        
        # Create WorldQuant Brain URL
        wq_url = f"https://platform.worldquantbrain.com/alpha/{alpha_id}"
        
        # Fetch detailed alpha information from database
        try:
            alpha_details_dict = get_alpha_details_for_clustering([alpha_id])
            alpha_info = alpha_details_dict.get(alpha_id, {})
            
            if alpha_info:
                # Create comprehensive details with proper formatting
                details = [
                    html.H4(f"Alpha: {alpha_id}", className="text-primary mb-3"),
                    
                    # WorldQuant Brain button
                    html.Div([
                        html.A(
                            dbc.Button(
                                "View on WorldQuant Brain", 
                                color="primary", 
                                size="sm",
                                className="mb-3"
                            ),
                            href=wq_url,
                            target="_blank"
                        )
                    ]),
                    
                    # Performance metrics in a clean grid
                    html.H6("📊 Performance Metrics", className="text-success mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Small([html.Strong("Sharpe: "), f"{alpha_info.get('is_sharpe', 0):.3f}"], className="d-block"),
                            html.Small([html.Strong("Fitness: "), f"{alpha_info.get('is_fitness', 0):.3f}"], className="d-block"),
                            html.Small([html.Strong("Returns: "), f"{alpha_info.get('is_returns', 0):.3f}"], className="d-block"),
                        ], width=6),
                        dbc.Col([
                            html.Small([html.Strong("Universe: "), alpha_info.get('universe', 'N/A')], className="d-block"),
                            html.Small([html.Strong("Delay: "), str(alpha_info.get('delay', 'N/A'))], className="d-block"),
                            html.Small([html.Strong("Region: "), alpha_info.get('region_name', 'N/A')], className="d-block"),
                        ], width=6),
                    ], className="mb-3"),
                    
                    # Additional settings
                    html.H6("⚙️ Settings", className="text-info mb-2"),
                    html.Div([
                        html.Small([html.Strong("Neutralization: "), alpha_info.get('neutralization', 'N/A')], className="d-block"),
                        html.Small([html.Strong("Decay: "), str(alpha_info.get('decay', 'N/A'))], className="d-block"),
                    ], className="mb-3"),
                    
                    # Full expression in a code block
                    html.H6("📝 Expression", className="text-warning mb-2"),
                    html.Div([
                        html.Code(
                            alpha_info.get('code', 'No expression available'),
                            style={
                                'white-space': 'pre-wrap',
                                'word-wrap': 'break-word',
                                'font-size': '0.75rem',
                                'background-color': '#f8f9fa',
                                'padding': '8px',
                                'border-radius': '4px',
                                'display': 'block',
                                'max-height': '150px',
                                'overflow-y': 'auto',
                                'border': '1px solid #dee2e6'
                            }
                        )
                    ])
                ]
            else:
                # Fallback when no detailed info is available
                details = [
                    html.H4(f"Alpha: {alpha_id}", className="text-primary mb-3"),
                    html.A(
                        dbc.Button(
                            "View on WorldQuant Brain", 
                            color="primary", 
                            size="sm",
                            className="mb-3"
                        ),
                        href=wq_url,
                        target="_blank"
                    ),
                    html.P("Detailed information not available in cache.", className="text-muted")
                ]
        except Exception as e:
            print(f"Error fetching alpha details: {e}")
            # Error fallback
            details = [
                html.H4(f"Alpha: {alpha_id}", className="text-primary mb-3"),
                html.A(
                    dbc.Button(
                        "View on WorldQuant Brain", 
                        color="primary", 
                        size="sm"
                    ),
                    href=wq_url,
                    target="_blank"
                ),
                html.P(f"Error loading details: {str(e)}", className="text-danger small")
            ]
        
        # Return details and selected alpha
        return details, {'index': alpha_id}
    
    # Modal close callback
    @app.callback(
        Output("detail-modal", "is_open"),
        [Input("detail-modal-close", "n_clicks")],
        [State("detail-modal", "is_open")],
        prevent_initial_call=True
    )
    def close_modal(n_clicks, is_open):
        if n_clicks:
            return False
        return is_open
    
    # Top 20 operators chart click handler
    @app.callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("operators-chart", "clickData")],
        [State("analysis-data", "data")],
        prevent_initial_call=True
    )
    def handle_operator_click(operators_chart_click, analysis_data):
        if not operators_chart_click or not operators_chart_click.get('points'):
            raise dash.exceptions.PreventUpdate
        
        click_data = operators_chart_click
        chart_type = "Top 20 Operators"
            
        # Get clicked operator name
        point = click_data['points'][0]
        operator = point['y']  # y-axis contains operator name for horizontal bars
        count = point['x']     # x-axis contains the count
        
        # Get operator data
        operators_data = analysis_data.get('operators', {})
        unique_usage = operators_data.get('unique_usage', {})
        alphas_using = unique_usage.get(operator, [])
        
        # Create modal content
        title = f"Operator Details: {operator}"
        
        body_content = [
            html.H5(f"📊 Usage Statistics", className="mb-3"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Operator: "),
                    html.Code(operator, className="bg-light p-1")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Used in: "),
                    html.Span(f"{count} alphas", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Chart: "),
                    html.Span(chart_type, className="text-muted")
                ])
            ], flush=True, className="mb-4"),
            
            html.H6(f"🔗 Alphas Using This Operator ({len(alphas_using)})", className="mb-3"),
        ]
        
        if alphas_using:
            # Show all alphas in a scrollable container
            alpha_badges = [
                dbc.Badge(
                    alpha_id,
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                    target="_blank",
                    color="primary",
                    className="me-1 mb-1",
                    style={'cursor': 'pointer', 'text-decoration': 'none'}
                ) for alpha_id in alphas_using
            ]
            
            body_content.append(
                html.Div(alpha_badges, className="mb-3", style={'max-height': '400px', 'overflow-y': 'auto'})
            )
        else:
            body_content.append(
                dbc.Alert("No alpha data available for this operator.", color="warning")
            )
        
        return True, title, body_content
    
    # All operators chart click handler
    @app.callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("all-operators-chart", "clickData")],
        [State("analysis-data", "data")],
        prevent_initial_call=True
    )
    def handle_all_operators_click(all_operators_click, analysis_data):
        if not all_operators_click or not all_operators_click.get('points'):
            raise dash.exceptions.PreventUpdate
        
        click_data = all_operators_click
        chart_type = "All Operators"
        
        # Get clicked operator name
        point = click_data['points'][0]
        operator = point['y']  # y-axis contains operator name for horizontal bars
        count = point['x']     # x-axis contains the count
        
        # Get operator data
        operators_data = analysis_data.get('operators', {})
        unique_usage = operators_data.get('unique_usage', {})
        alphas_using = unique_usage.get(operator, [])
        
        # Create modal content
        title = f"Operator Details: {operator}"
        
        body_content = [
            html.H5(f"📊 Usage Statistics", className="mb-3"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Operator: "),
                    html.Code(operator, className="bg-light p-1")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Used in: "),
                    html.Span(f"{count} alphas", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Chart: "),
                    html.Span(chart_type, className="text-muted")
                ])
            ], flush=True, className="mb-4"),
            
            html.H6(f"🔗 Alphas Using This Operator ({len(alphas_using)})", className="mb-3"),
        ]
        
        if alphas_using:
            # Show all alphas in a scrollable container
            alpha_badges = [
                dbc.Badge(
                    alpha_id,
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                    target="_blank",
                    color="primary",
                    className="me-1 mb-1",
                    style={'cursor': 'pointer', 'text-decoration': 'none'}
                ) for alpha_id in alphas_using
            ]
            
            body_content.append(
                html.Div(alpha_badges, className="mb-3", style={'max-height': '400px', 'overflow-y': 'auto'})
            )
        else:
            body_content.append(
                dbc.Alert("No alpha data available for this operator.", color="warning")
            )
        
        return True, title, body_content
    
    # Top 20 datafields chart click handler
    @app.callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("datafields-chart", "clickData")],
        [State("analysis-data", "data")],
        prevent_initial_call=True
    )
    def handle_datafields_chart_click(datafields_click, analysis_data):
        if not datafields_click or not datafields_click.get('points'):
            raise dash.exceptions.PreventUpdate
        
        # Get clicked datafield name
        point = datafields_click['points'][0]
        datafield = point['y']  # y-axis contains datafield name for horizontal bars
        count = point['x']     # x-axis contains the count
        
        return handle_datafield_detail_click(datafield, count, "Top 20 Datafields", analysis_data)
    
    # All datafields chart click handler
    @app.callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("all-datafields-chart", "clickData")],
        [State("analysis-data", "data")],
        prevent_initial_call=True
    )
    def handle_all_datafields_chart_click(all_datafields_click, analysis_data):
        if not all_datafields_click or not all_datafields_click.get('points'):
            raise dash.exceptions.PreventUpdate
        
        # Get clicked datafield name
        point = all_datafields_click['points'][0]
        datafield = point['y']  # y-axis contains datafield name for horizontal bars
        count = point['x']     # x-axis contains the count
        
        return handle_datafield_detail_click(datafield, count, "All Datafields", analysis_data)
    
    # All datasets chart click handler
    @app.callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("all-datasets-chart", "clickData")],
        [State("analysis-data", "data")],
        prevent_initial_call=True
    )
    def handle_all_datasets_chart_click(all_datasets_click, analysis_data):
        if not all_datasets_click or not all_datasets_click.get('points'):
            raise dash.exceptions.PreventUpdate
        
        # Get clicked dataset name
        point = all_datasets_click['points'][0]
        dataset = point['y']  # y-axis contains dataset name for horizontal bars
        count = point['x']    # x-axis contains the count
        
        return handle_dataset_click(dataset, count, analysis_data)
    
    def handle_datafield_detail_click(datafield, count, chart_type, analysis_data):
        """Handle click on datafield charts."""
        datafields_data = analysis_data.get('datafields', {})
        unique_usage = datafields_data.get('unique_usage', {})
        alphas_using = unique_usage.get(datafield, [])
        
        # Get datafield metadata
        try:
            temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
            df_info = temp_analysis_ops.parser.datafields.get(datafield, {})
            dataset_id = df_info.get('dataset_id', 'Unknown')
            category = df_info.get('data_category', 'Unknown')
        except:
            dataset_id = 'Unknown'
            category = 'Unknown'
        
        # Create modal content
        title = f"Datafield Details: {datafield}"
        
        body_content = [
            html.H5(f"📊 Usage Statistics", className="mb-3"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Datafield: "),
                    html.Code(datafield, className="bg-light p-1")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Used in: "),
                    html.Span(f"{count} alphas", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Dataset: "),
                    html.Span(dataset_id, className="badge bg-success ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Category: "),
                    html.Span(category, className="badge bg-info ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Chart: "),
                    html.Span(chart_type, className="text-muted")
                ])
            ], flush=True, className="mb-4"),
            
            html.H6(f"🔗 Alphas Using This Datafield ({len(alphas_using)})", className="mb-3"),
        ]
        
        if alphas_using:
            # Show all alphas in a scrollable container
            alpha_badges = [
                dbc.Badge(
                    alpha_id,
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                    target="_blank",
                    color="success",
                    className="me-1 mb-1",
                    style={'cursor': 'pointer', 'text-decoration': 'none'}
                ) for alpha_id in alphas_using
            ]
            
            body_content.append(
                html.Div(alpha_badges, className="mb-3", style={'max-height': '400px', 'overflow-y': 'auto'})
            )
        else:
            body_content.append(
                dbc.Alert("No alpha data available for this datafield.", color="warning")
            )
        
        return True, title, body_content
    
    def handle_dataset_click(dataset, count, analysis_data):
        """Handle click on dataset chart."""
        datafields_data = analysis_data.get('datafields', {})
        unique_usage = datafields_data.get('unique_usage', {})
        
        # Use pre-processed dataset mappings for fast lookup
        dataset_mappings = datafields_data.get('dataset_mappings', {})
        dataset_to_datafields = dataset_mappings.get('dataset_to_datafields', {})
        
        # Get datafields for this dataset (FAST - no loops needed)
        dataset_datafields = []
        total_alphas = set()
        
        datafields_in_dataset = dataset_to_datafields.get(dataset, [])
        for df in datafields_in_dataset:
            alphas = unique_usage.get(df, [])
            if alphas:  # Only include if alphas exist
                dataset_datafields.append((df, len(alphas)))
                total_alphas.update(alphas)
        
        dataset_datafields.sort(key=lambda x: x[1], reverse=True)
        
        # Create modal content
        title = f"Dataset Details: {dataset}"
        
        body_content = [
            html.H5(f"📊 Dataset Statistics", className="mb-3"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Dataset ID: "),
                    html.Code(dataset, className="bg-light p-1")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Total Datafield Instances: "),
                    html.Span(f"{count}", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Unique Datafields: "),
                    html.Span(f"{len(dataset_datafields)}", className="badge bg-info ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Unique Alphas: "),
                    html.Span(f"{len(total_alphas)}", className="badge bg-success ms-2")
                ])
            ], flush=True, className="mb-4"),
            
            html.H6(f"🔍 Datafields in This Dataset ({len(dataset_datafields)})", className="mb-3"),
        ]
        
        if dataset_datafields:
            # Show all datafields as a scrollable table
            datafield_items = []
            for df, df_count in dataset_datafields:  # Show all datafields
                datafield_items.append(
                    dbc.ListGroupItem([
                        html.Strong(df),
                        html.Span(f"{df_count} alphas", className="badge bg-secondary ms-2 float-end")
                    ])
                )
            
            body_content.extend([
                html.Div([
                    dbc.ListGroup(datafield_items, flush=True)
                ], style={'max-height': '300px', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'border-radius': '4px'}, className="mb-3"),
                
                html.H6(f"🔗 All Alphas Using This Dataset ({len(total_alphas)} total)", className="mb-3"),
                html.Div([
                    dbc.Badge(
                        alpha_id,
                        href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                        target="_blank",
                        color="warning",
                        className="me-1 mb-1",
                        style={'cursor': 'pointer', 'text-decoration': 'none'}
                    ) for alpha_id in list(total_alphas)
                ], className="mb-3", style={'max-height': '400px', 'overflow-y': 'auto'})
            ])
        else:
            body_content.append(
                dbc.Alert("No datafield data available for this dataset.", color="warning")
            )
        
        return True, title, body_content
    
    # Category pie chart click handler (treemap removed for better UX)
    @app.callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("category-chart", "clickData")],
        [State("analysis-data", "data")],
        prevent_initial_call=True
    )
    def handle_category_click(category_click, analysis_data):
        if not category_click or not category_click.get('points'):
            raise dash.exceptions.PreventUpdate
            
        datafields_data = analysis_data.get('datafields', {})
        unique_usage = datafields_data.get('unique_usage', {})
        
        # Handle pie chart click
        point = category_click['points'][0]
        category = point['label']
        count = point['value']
        
        return handle_category_detail_click(category, count, unique_usage)
    
    def handle_category_detail_click(category, count, unique_usage):
        """Handle click on category pie chart."""
        try:
            temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
            category_datafields = []
            total_alphas = set()
            
            for df, alphas in unique_usage.items():
                if df in temp_analysis_ops.parser.datafields:
                    df_info = temp_analysis_ops.parser.datafields[df]
                    if df_info.get('data_category') == category:
                        category_datafields.append((df, len(alphas)))
                        total_alphas.update(alphas)
                        
            category_datafields.sort(key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"Error processing category {category}: {e}")
            category_datafields = []
            total_alphas = set()
        
        # Create modal content
        title = f"Category Details: {category}"
        
        body_content = [
            html.H5(f"📊 Category Statistics", className="mb-3"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Category: "),
                    html.Code(category, className="bg-light p-1")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Total Usage: "),
                    html.Span(f"{count}", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Unique Datafields: "),
                    html.Span(f"{len(category_datafields)}", className="badge bg-info ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Unique Alphas: "),
                    html.Span(f"{len(total_alphas)}", className="badge bg-success ms-2")
                ])
            ], flush=True, className="mb-4"),
            
            html.H6(f"🔍 Top Datafields in This Category", className="mb-3"),
        ]
        
        if category_datafields:
            # Show all datafields
            datafield_items = []
            for df, df_count in category_datafields:  # Show all datafields
                datafield_items.append(
                    dbc.ListGroupItem([
                        html.Strong(df),
                        html.Span(f"{df_count} alphas", className="badge bg-secondary ms-2 float-end")
                    ])
                )
            
            body_content.extend([
                html.Div([
                    dbc.ListGroup(datafield_items, flush=True)
                ], style={'max-height': '300px', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'border-radius': '4px'}, className="mb-3"),
                
                html.H6(f"🔗 All Alphas in This Category ({len(total_alphas)} total)", className="mb-3"),
                html.Div([
                    dbc.Badge(
                        alpha_id,
                        href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                        target="_blank",
                        color="info",
                        className="me-1 mb-1",
                        style={'cursor': 'pointer', 'text-decoration': 'none'}
                    ) for alpha_id in list(total_alphas)
                ], className="mb-3", style={'max-height': '400px', 'overflow-y': 'auto'})
            ])
        else:
            body_content.append(
                dbc.Alert("No datafield data available for this category.", color="warning")
            )
        
        return True, title, body_content
    
    # Neutralization pie chart click handler
    @app.callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("neutralization-pie-chart", "clickData")],
        [State("analysis-data", "data")],
        prevent_initial_call=True
    )
    def handle_neutralization_pie_click(pie_click, analysis_data):
        if not pie_click or not pie_click.get('points'):
            raise dash.exceptions.PreventUpdate
        
        point = pie_click['points'][0]
        neutralization = point['label']
        count = point['value']
        
        return handle_neutralization_detail_click(neutralization, count, analysis_data)
    
    # Neutralization bar chart click handler
    @app.callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("neutralization-bar-chart", "clickData")],
        [State("analysis-data", "data")],
        prevent_initial_call=True
    )
    def handle_neutralization_bar_click(bar_click, analysis_data):
        if not bar_click or not bar_click.get('points'):
            raise dash.exceptions.PreventUpdate
        
        point = bar_click['points'][0]
        neutralization = point['y']  # y-axis contains neutralization name for horizontal bars
        count = point['x']           # x-axis contains the count
        
        return handle_neutralization_detail_click(neutralization, count, analysis_data)
    
    def handle_neutralization_detail_click(neutralization, count, analysis_data):
        """Handle click on neutralization charts."""
        # Get all alphas and filter by neutralization
        alpha_details = {}
        try:
            # Get alpha details from clustering if available
            alpha_details = get_alpha_details_for_clustering([])  # Get all alphas
        except Exception as e:
            print(f"Error getting alpha details: {e}")
        
        # Filter alphas by neutralization
        matching_alphas = []
        try:
            db_engine = get_connection()
            with db_engine.connect() as connection:
                query = text("""
                    SELECT alpha_id, code, universe, delay, is_sharpe, is_fitness, is_returns,
                           neutralization, decay, r.region_name
                    FROM alphas a
                    JOIN regions r ON a.region_id = r.region_id
                    WHERE a.neutralization = :neutralization
                    ORDER BY a.alpha_id
                """)
                
                result = connection.execute(query, {'neutralization': neutralization})
                
                for row in result:
                    matching_alphas.append({
                        'alpha_id': row.alpha_id,
                        'code': row.code or '',
                        'universe': row.universe or 'N/A',
                        'delay': row.delay if row.delay is not None else 'N/A',
                        'is_sharpe': row.is_sharpe if row.is_sharpe is not None else 0,
                        'is_fitness': row.is_fitness if row.is_fitness is not None else 0,
                        'is_returns': row.is_returns if row.is_returns is not None else 0,
                        'neutralization': row.neutralization or 'N/A',
                        'decay': row.decay or 'N/A',
                        'region_name': row.region_name or 'N/A'
                    })
                    
        except Exception as e:
            print(f"Error fetching alphas for neutralization {neutralization}: {e}")
        
        # Create modal content
        title = f"Neutralization Details: {neutralization}"
        
        body_content = [
            html.H5(f"📊 Neutralization Statistics", className="mb-3"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Neutralization Type: "),
                    html.Code(neutralization, className="bg-light p-1")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Total Alphas: "),
                    html.Span(f"{count}", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Average Performance: "),
                    html.Span(f"{sum(alpha['is_sharpe'] for alpha in matching_alphas) / len(matching_alphas):.3f}" if matching_alphas else "N/A", className="badge bg-success ms-2")
                ]),
            ], flush=True, className="mb-4"),
            
            html.H6(f"🔍 Alphas Using {neutralization}", className="mb-3"),
        ]
        
        if matching_alphas:
            # Show alpha summary statistics
            universes = {}
            regions = {}
            delays = {}
            
            for alpha in matching_alphas:
                universe = alpha.get('universe', 'N/A')
                region = alpha.get('region_name', 'N/A')
                delay = str(alpha.get('delay', 'N/A'))
                
                universes[universe] = universes.get(universe, 0) + 1
                regions[region] = regions.get(region, 0) + 1
                delays[delay] = delays.get(delay, 0) + 1
            
            body_content.extend([
                html.H6("📈 Distribution Summary", className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        html.Strong("By Universe:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in sorted(universes.items())])
                    ], width=4),
                    dbc.Col([
                        html.Strong("By Region:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in sorted(regions.items())])
                    ], width=4),
                    dbc.Col([
                        html.Strong("By Delay:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in sorted(delays.items())])
                    ], width=4),
                ], className="mb-4"),
                
                html.H6(f"📋 Alpha List ({len(matching_alphas)} total)", className="mb-2"),
                html.Div([
                    dbc.Badge(
                        alpha['alpha_id'],
                        id={'type': 'alpha-badge', 'index': alpha['alpha_id']},
                        color="primary",
                        className="me-1 mb-1",
                        style={'cursor': 'pointer'},
                        title=f"Sharpe: {alpha['is_sharpe']:.3f} | Universe: {alpha['universe']} | Region: {alpha['region_name']}"
                    ) for alpha in matching_alphas[:100]  # Limit to first 100 for performance
                ], style={'max-height': '200px', 'overflow-y': 'auto', 'border': '1px solid #dee2e6', 'padding': '10px', 'border-radius': '4px'}, className="mb-3"),
            ])
            
            if len(matching_alphas) > 100:
                body_content.append(
                    dbc.Alert(f"Showing first 100 alphas out of {len(matching_alphas)} total.", color="info", className="mt-2")
                )
        else:
            body_content.append(
                dbc.Alert("No alpha data available for this neutralization type.", color="warning")
            )
        
        return True, title, body_content
    
    return app

def open_browser(port, delay=1):
    """Open browser after a delay."""
    time.sleep(delay)
    webbrowser.open(f'http://localhost:{port}')

def main():
    """
    Run the analysis and visualization server.
    """
    parser = argparse.ArgumentParser(description="Alpha Analysis & Clustering Dashboard")
    parser.add_argument("--data-file", type=str, help="Path to the clustering data JSON file (optional)")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--init-db", action="store_true", help="Initialize analysis database schema")
    
    args = parser.parse_args()
    
    # Initialize database if requested
    if args.init_db:
        print("Initializing analysis database schema...")
        try:
            initialize_analysis_database()
            print("Analysis database schema initialized successfully.")
        except Exception as e:
            print(f"Error initializing database: {e}")
            return
    
    # Load clustering data if provided
    data = None
    if args.data_file:
        if not os.path.isfile(args.data_file):
            print(f"Warning: Clustering data file not found: {args.data_file}")
            print("Starting in analysis-only mode...")
        else:
            try:
                data = load_clustering_data(args.data_file)
                print(f"Loaded clustering data for region: {data.get('region', 'Unknown')}")
            except Exception as e:
                print(f"Warning: Could not load clustering data: {e}")
                print("Starting in analysis-only mode...")
    
    # Create app
    print("Initializing dashboard...")
    app = create_visualization_app(data)
    
    # Start browser opener thread
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port, 2), daemon=True).start()
    
    # Run server
    mode = "Analysis & Clustering" if data else "Analysis-only"
    print(f"Starting {mode} dashboard on port {args.port}...")
    print(f"Dashboard URL: http://localhost:{args.port}")
    
    if not args.no_browser:
        print("Browser will open automatically in 2 seconds...")
    
    try:
        app.run(debug=args.debug, port=args.port, host='127.0.0.1')
    except Exception as e:
        print(f"Error starting server: {e}")
        print("\nTroubleshooting tips:")
        print(f"- Check if port {args.port} is already in use")
        print("- Try a different port with --port argument")
        print("- Ensure all required dependencies are installed")

if __name__ == "__main__":
    main()
