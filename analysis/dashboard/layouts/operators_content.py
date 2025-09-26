"""
Operators Content Functions

Content creation functions for operators analysis views.
"""

import pandas as pd
import plotly.express as px
from dash import html, dcc
import dash_bootstrap_components as dbc

from ..config import TEMPLATE
from ..services import load_operators_data, get_analysis_service

def create_operators_content(analysis_data, view_mode='top20'):
    """
    Create operators analysis content with different view modes.

    """
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

def create_usage_analysis_content(analysis_data):
    """
    Create comprehensive usage analysis showing all platform operators.

    """
    operators_data = analysis_data.get('operators', {})
    used_operators = dict(operators_data.get('top_operators', []))

    # Load all platform operators
    try:
        # Use analysis service to get operator usage analysis
        analysis_service = get_analysis_service()
        from ..config import DEFAULT_OPERATORS_FILE
        usage_analysis = analysis_service.get_operator_usage_analysis(DEFAULT_OPERATORS_FILE)

        if 'error' in usage_analysis:
            return html.Div(f"Error loading operators file: {usage_analysis['error']}", className="text-danger")

        frequently_used = usage_analysis['frequently_used']
        rarely_used = usage_analysis['rarely_used']
        never_used = usage_analysis['never_used']
        total_operators = usage_analysis['total_operators']

    except Exception as e:
        print(f"❌ Error loading operators: {e}")
        return html.Div(f"Error loading operators file: {str(e)}", className="text-danger")

    return dbc.Row([
        dbc.Col([
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
                        for op, _ in never_used  # Show ALL never used operators
                    ], style={'max-height': '400px', 'overflow-y': 'auto'})  # Add scroll for very long lists
                ], title=f"Never Used Operators ({len(never_used)})", item_id="never")
            ], active_item="frequent")
        ])
    ])

def create_all_operators_content(analysis_data):
    """
    Create view showing all used operators with their counts.

    """
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
    """
    Create the original top 20 operators view.

    """
    operators_data = analysis_data.get('operators', {})
    top_operators = operators_data.get('top_operators', [])
    metadata = analysis_data.get('metadata', {})

    # Create DataFrame for proper Plotly usage
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
        ], width=4)
    ])