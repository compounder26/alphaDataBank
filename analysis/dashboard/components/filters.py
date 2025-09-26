"""
Filter Components

Reusable filter components for the dashboard.
"""

from typing import List, Dict, Any, Optional
from dash import html, dcc
import dash_bootstrap_components as dbc

from .base_components import create_info_card, create_action_button

def create_analysis_filters() -> dbc.Card:
    """
    Create analysis filters card with all filter components.

    Returns:
        Card component with analysis filters
    """
    return create_info_card(
        title="Analysis Filters",
        content=[
            create_region_filter(),
            create_universe_filter(),
            create_delay_filter(),
            create_dates_filter(),
            create_action_button(
                "Apply Filters",
                button_id="apply-filters-btn",
                color="primary",
                className="w-100 mb-2"
            )
        ]
    )

def create_region_filter(element_id: str = 'region-filter',
                        placeholder: str = "All regions",
                        clearable: bool = True) -> html.Div:
    """
    Create region filter dropdown.

    Args:
        element_id: Element ID
        placeholder: Placeholder text
        clearable: Whether dropdown is clearable

    Returns:
        Filter div component
    """
    return html.Div([
        html.Label("Region:", className="form-label"),
        dcc.Dropdown(
            id=element_id,
            placeholder=placeholder,
            clearable=clearable
        )
    ], className="mb-3")

def create_universe_filter(element_id: str = 'universe-filter',
                          placeholder: str = "All universes",
                          clearable: bool = True) -> html.Div:
    """
    Create universe filter dropdown.

    Args:
        element_id: Element ID
        placeholder: Placeholder text
        clearable: Whether dropdown is clearable

    Returns:
        Filter div component
    """
    return html.Div([
        html.Label("Universe:", className="form-label"),
        dcc.Dropdown(
            id=element_id,
            placeholder=placeholder,
            clearable=clearable
        )
    ], className="mb-3")

def create_delay_filter(element_id: str = 'delay-filter',
                       placeholder: str = "All delays",
                       clearable: bool = True) -> html.Div:
    """
    Create delay filter dropdown.

    Args:
        element_id: Element ID
        placeholder: Placeholder text
        clearable: Whether dropdown is clearable

    Returns:
        Filter div component
    """
    return html.Div([
        html.Label("Delay:", className="form-label"),
        dcc.Dropdown(
            id=element_id,
            placeholder=placeholder,
            clearable=clearable
        )
    ], className="mb-3")

def create_dates_filter(element_id: str = 'dates-filter',
                       display_format: str = 'MM/DD/YYYY') -> html.Div:
    """
    Create dates filter component.

    Args:
        element_id: Element ID
        display_format: Date display format

    Returns:
        Filter div component
    """
    return html.Div([
        html.Label("Dates:", className="form-label"),
        dcc.DatePickerRange(
            id=element_id,
            start_date_placeholder_text="Start date",
            end_date_placeholder_text="End date",
            clearable=True,
            display_format=display_format,
            style={'width': '100%'}
        )
    ], className="mb-3")

def create_clustering_region_selector(element_id: str = 'clustering-region-selector',
                                    placeholder: str = "Select clustering region...") -> html.Div:
    """
    Create clustering region selector.

    Args:
        element_id: Element ID
        placeholder: Placeholder text

    Returns:
        Region selector component
    """
    return html.Div([
        html.Label("Select Region:", className="form-label"),
        dcc.Dropdown(
            id=element_id,
            placeholder=placeholder,
            clearable=False
        ),
        html.Div(id='clustering-region-info', className="mt-2 small text-muted")
    ])

def create_distance_metric_selector(element_id: str = 'distance-metric',
                                   default_value: str = 'euclidean',
                                   container_id: str = 'distance-metric-container') -> html.Div:
    """
    Create distance metric selector.

    Args:
        element_id: Element ID
        default_value: Default selected value
        container_id: Container ID for show/hide

    Returns:
        Distance metric selector
    """
    options = [
        {'label': 'Simple (1 - corr)', 'value': 'simple'},
        {'label': 'Euclidean √(2(1-corr))', 'value': 'euclidean'},
        {'label': 'Angular √(0.5(1-corr))', 'value': 'angular'},
    ]

    return html.Div([
        html.Label("Distance Metric:", className="form-label"),
        dcc.RadioItems(
            id=element_id,
            options=options,
            value=default_value,
            className="mb-2"
        ),
    ], id=container_id, style={'margin-bottom': '15px'})

def create_highlighting_filters() -> dbc.Card:
    """
    Create alpha highlighting filters card.

    Returns:
        Card component with highlighting filters
    """
    return create_info_card(
        title="Alpha Highlighting",
        content=[
            # Operator highlighting section
            html.Div([
                html.Label("Highlight alphas using operators:", className="form-label"),
                dcc.Dropdown(
                    id='operator-highlight-selector',
                    placeholder="Select operators to highlight...",
                    multi=True,
                    searchable=True
                ),
            ], className="mb-3"),

            # Datafield highlighting section
            html.Div([
                html.Label("Highlight alphas using datafields:", className="form-label"),
                dcc.Dropdown(
                    id='datafield-highlight-selector',
                    placeholder="Select datafields to highlight...",
                    multi=True,
                    searchable=True
                ),
            ], className="mb-3"),

            # Clear highlights button
            create_action_button(
                "Clear Highlights",
                button_id="clear-highlights-btn",
                color="outline-secondary",
                size="sm"
            )
        ]
    )

def create_display_options_card() -> dbc.Card:
    """
    Create display options card.

    Returns:
        Card component with display options
    """
    return create_info_card(
        title="Display Options",
        content=[
            dcc.RadioItems(
                id='cluster-color-mode',
                options=[
                    {'label': 'Color by Cluster', 'value': 'cluster'},
                    {'label': 'Single Color', 'value': 'single'}
                ],
                value='cluster',
                inline=True,
                className="mb-2"
            ),
            html.Small(
                "Choose whether to color points by cluster assignment or use a single color.",
                className="text-muted"
            )
        ]
    )

def create_method_explanation_card() -> dbc.Card:
    """
    Create method explanation card.

    Returns:
        Card component for method explanations
    """
    return dbc.Card([
        dbc.CardHeader("Method Explanation", className="bg-info text-white"),
        dbc.CardBody([
            html.Div(id='method-explanation', className="mb-3")
        ])
    ], className="mb-3")

def create_recommendation_filters() -> dbc.Card:
    """
    Create recommendation filters card.

    Returns:
        Card component with recommendation filters
    """
    return create_info_card(
        title="Filters",
        content=[
            dbc.Row([
                dbc.Col([
                    html.Label("Target Region:", className="form-label"),
                    dcc.Dropdown(
                        id='recommendation-region-filter',
                        options=[{'label': 'All Regions', 'value': 'all'}] +
                               [{'label': region, 'value': region} for region in
                                ['USA', 'EUR', 'JPN', 'CHN', 'AMR', 'ASI', 'GLB', 'HKG', 'KOR', 'TWN']],
                        value='all',
                        clearable=False,
                        placeholder="Select target region..."
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Datafield Type:", className="form-label"),
                    dcc.Dropdown(
                        id='recommendation-type-filter',
                        options=[
                            {'label': 'All Types', 'value': 'all'},
                            {'label': 'Matrix', 'value': 'MATRIX'},
                            {'label': 'Vector', 'value': 'VECTOR'},
                            {'label': 'Group', 'value': 'GROUP'}
                        ],
                        value='all',
                        clearable=False,
                        placeholder="Select datafield type..."
                    )
                ], md=4),
                dbc.Col([
                    create_action_button(
                        "Refresh Recommendations",
                        button_id="refresh-recommendations-btn",
                        color="primary",
                        className="mt-4"
                    )
                ], md=4)
            ])
        ]
    )

def create_tab_selector(tabs: List[Dict[str, str]], element_id: str,
                       default_value: str, className: str = 'mb-4') -> dcc.Tabs:
    """
    Create tab selector component.

    Args:
        tabs: List of tab dictionaries with 'label' and 'value'
        element_id: Element ID
        default_value: Default selected tab
        className: CSS classes

    Returns:
        Tabs component
    """
    tab_components = [
        dcc.Tab(label=tab['label'], value=tab['value'])
        for tab in tabs
    ]

    return dcc.Tabs(
        id=element_id,
        value=default_value,
        children=tab_components,
        className=className
    )

def create_multi_select_dropdown(element_id: str, placeholder: str,
                                options: List[Dict[str, str]] = None,
                                searchable: bool = True) -> dcc.Dropdown:
    """
    Create multi-select dropdown.

    Args:
        element_id: Element ID
        placeholder: Placeholder text
        options: Dropdown options
        searchable: Whether dropdown is searchable

    Returns:
        Multi-select Dropdown component
    """
    return dcc.Dropdown(
        id=element_id,
        placeholder=placeholder,
        multi=True,
        searchable=searchable,
        options=options or []
    )

def create_single_select_dropdown(element_id: str, placeholder: str,
                                 options: List[Dict[str, str]] = None,
                                 clearable: bool = True,
                                 searchable: bool = False) -> dcc.Dropdown:
    """
    Create single-select dropdown.

    Args:
        element_id: Element ID
        placeholder: Placeholder text
        options: Dropdown options
        clearable: Whether dropdown is clearable
        searchable: Whether dropdown is searchable

    Returns:
        Single-select Dropdown component
    """
    return dcc.Dropdown(
        id=element_id,
        placeholder=placeholder,
        clearable=clearable,
        searchable=searchable,
        options=options or []
    )

def create_filter_reset_button(button_id: str = "reset-filters-btn") -> dbc.Button:
    """
    Create filter reset button.

    Args:
        button_id: Button ID

    Returns:
        Reset button component
    """
    return create_action_button(
        "Reset Filters",
        button_id=button_id,
        color="outline-secondary",
        size="sm"
    )

def create_filter_summary(applied_filters: Dict[str, Any]) -> html.Div:
    """
    Create summary of applied filters.

    Args:
        applied_filters: Dictionary of applied filters

    Returns:
        Filter summary component
    """
    if not any(applied_filters.values()):
        return html.Div()

    filter_items = []
    for key, value in applied_filters.items():
        if value:
            display_key = key.replace('_', ' ').title()
            filter_items.append(
                dbc.Badge(f"{display_key}: {value}", color="info", className="me-1 mb-1")
            )

    if not filter_items:
        return html.Div()

    return html.Div([
        html.Small("Applied Filters:", className="text-muted"),
        html.Div(filter_items, className="mt-1")
    ], className="mb-3")