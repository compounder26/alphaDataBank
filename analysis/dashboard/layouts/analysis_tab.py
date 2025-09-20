"""
Analysis Tab Layout

Layout structure for the expression analysis tab.
Extracted from visualization_server.py with preserved functionality.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..components import (
    create_analysis_filters,
    create_info_card,
    create_loading_wrapper,
    create_tab_selector
)

# Import content creation functions from compatibility bridge
from ..compatibility_bridge import (
    create_operators_content,
    create_datafields_content,
    create_neutralization_content,
    create_cross_analysis_content,
    get_dataset_treemap_sidebar_info
)


def create_analysis_tab_content() -> dbc.Row:
    """
    Create the analysis tab content layout.

    Returns:
        Row component with analysis tab structure
    """
    return dbc.Row([
        # Filters sidebar
        dbc.Col([
            # Analysis filters card
            create_analysis_filters(),

            # Analysis summary card
            create_info_card(
                title="Analysis Summary",
                content=create_loading_wrapper(
                    content=html.Div(id="analysis-summary"),
                    loading_id="loading-analysis-summary"
                )
            ),

            # Conditional treemap information panels (only shown in treemap view)
            html.Div(id='treemap-sidebar-info', children=[], className="mt-3")
        ], width=3),

        # Main analysis content
        dbc.Col([
            # Analysis subtabs
            create_tab_selector(
                tabs=[
                    {'label': '⚙️ Operators', 'value': 'operators-subtab'},
                    {'label': '📈 Datafields', 'value': 'datafields-subtab'},
                    {'label': '⚖️ Neutralization', 'value': 'neutralization-subtab'},
                    {'label': '🔄 Cross Analysis', 'value': 'cross-subtab'},
                ],
                element_id='analysis-subtabs',
                default_value='operators-subtab'
            ),

            # Analysis subtab content with loading
            create_loading_wrapper(
                content=html.Div(id='analysis-subtab-content'),
                loading_id="loading-analysis-subtabs"
            )
        ], width=9)
    ])


def create_operators_view_selector() -> dbc.Card:
    """
    Create operators view mode selector.

    Returns:
        Card component with view options
    """
    return create_info_card(
        title="View Options",
        content=dbc.RadioItems(
            id='operators-view-selector',
            options=[
                {'label': 'Top 20 Most Used', 'value': 'top20'},
                {'label': 'All Used Operators', 'value': 'all'},
                {'label': 'Usage Analysis (All Platform Operators)', 'value': 'usage-analysis'}
            ],
            value='top20',
            inline=True
        ),
        className="mb-3"
    )


def create_datafields_view_selector() -> dbc.Card:
    """
    Create datafields view mode selector.

    Returns:
        Card component with view options
    """
    return create_info_card(
        title="View Options",
        content=dbc.RadioItems(
            id='datafields-view-selector',
            options=[
                {'label': 'Top 20 Most Used', 'value': 'top20'},
                {'label': 'All Used Datafields', 'value': 'all'},
                {'label': 'All Used Datasets', 'value': 'datasets'},
                {'label': 'Dataset Treemap', 'value': 'treemap'}
            ],
            value='top20',
            inline=True
        ),
        className="mb-3"
    )


def create_operators_usage_summary(frequently_used, rarely_used, never_used, total_operators):
    """
    Create operators usage summary with statistics.

    Args:
        frequently_used: List of frequently used operators
        rarely_used: List of rarely used operators
        never_used: List of never used operators
        total_operators: Total number of operators

    Returns:
        Usage summary component
    """
    return html.Div([
        html.H5(f"📊 Operator Usage Summary ({total_operators} total operators)", className="text-primary"),
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
                    for op, _ in never_used
                ], style={'max-height': '400px', 'overflow-y': 'auto'})
            ], title=f"Never Used Operators ({len(never_used)})", item_id="never")
        ], active_item="frequent")
    ])


def create_operators_statistics_panel(total_unique_ops, total_nominal, total_alphas):
    """
    Create operators statistics panel.

    Args:
        total_unique_ops: Total unique operators
        total_nominal: Total operator instances
        total_alphas: Total number of alphas

    Returns:
        Statistics panel component
    """
    avg_ops_per_alpha = total_nominal / total_alphas if total_alphas > 0 else 0

    return html.Div([
        html.H5("📊 Usage Statistics"),
        dbc.ListGroup([
            dbc.ListGroupItem([
                html.Strong("Total Unique Operators: "),
                html.Span(f"{total_unique_ops}", className="badge bg-primary ms-2")
            ]),
            dbc.ListGroupItem([
                html.Strong("Total Operator Instances: "),
                html.Span(f"{total_nominal:,}", className="badge bg-success ms-2")
            ]),
            dbc.ListGroupItem([
                html.Strong("Average per Alpha: "),
                html.Span(f"{avg_ops_per_alpha:.1f}", className="badge bg-info ms-2")
            ]),
            dbc.ListGroupItem([
                html.Strong("Total Alphas: "),
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
    ])


def create_datafields_grid_layout(chart_containers):
    """
    Create responsive grid layout for datafields view.

    Args:
        chart_containers: List of chart container components

    Returns:
        Grid layout component
    """
    if len(chart_containers) < 4:
        # Pad with empty divs if needed
        while len(chart_containers) < 4:
            chart_containers.append(html.Div())

    return html.Div([
        # Grid container with enhanced plots
        html.Div([
            # Main datafields chart
            html.Div([
                chart_containers[0]
            ], style={
                'grid-column': '1 / 3',
                'grid-row': '1',
                'height': '600px'
            }),

            # Statistics panel
            html.Div([
                chart_containers[1]
            ], style={
                'grid-column': '1',
                'grid-row': '2',
                'height': '400px'
            }),

            # Category pie chart
            html.Div([
                chart_containers[2]
            ], style={
                'grid-column': '2',
                'grid-row': '2',
                'height': '400px'
            }),

            # Dataset treemap
            html.Div([
                chart_containers[3]
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
        })
    ])


def create_analysis_summary_items(results):
    """
    Create analysis summary list items.

    Args:
        results: Analysis results dictionary

    Returns:
        List of ListGroupItem components
    """
    metadata = results.get('metadata', {})
    total_alphas = metadata.get('total_alphas', 0)

    # Show filter information in summary if filters are applied
    filter_info = []
    filters = metadata.get('filters', {})
    if filters.get('region'):
        filter_info.append(f"Region: {filters['region']}")
    if filters.get('universe'):
        filter_info.append(f"Universe: {filters['universe']}")
    if filters.get('delay') is not None:
        filter_info.append(f"Delay: {filters['delay']}")
    if filters.get('date_from') or filters.get('date_to'):
        date_range = f"{filters.get('date_from') or 'start'} to {filters.get('date_to') or 'end'}"
        filter_info.append(f"Date range: {date_range}")

    summary_items = [
        dbc.ListGroupItem([
            html.Strong(f"Total Alphas: {total_alphas}")
        ])
    ]

    if filter_info:
        summary_items.append(
            dbc.ListGroupItem([
                html.Strong("Active Filters: "),
                html.Small(", ".join(filter_info), className="text-muted")
            ])
        )

    summary_items.extend([
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
    ])

    return summary_items


def create_no_analysis_data_message() -> html.Div:
    """
    Create message for when no analysis data is available.

    Returns:
        No data message component
    """
    return html.Div(
        "Apply filters to load analysis data",
        className="text-muted text-center p-4"
    )