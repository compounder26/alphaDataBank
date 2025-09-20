"""
Chart Components

Reusable chart components and graph containers for the dashboard.
Extracted from visualization_server.py with preserved styling and functionality.
"""

from typing import Any, Dict, Optional, List
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from .base_components import create_loading_wrapper, create_info_card


def create_main_clustering_plot(plot_id: str = 'clustering-plot',
                               height: str = '85vh') -> dbc.Card:
    """
    Create main clustering visualization card.

    Args:
        plot_id: Plot element ID
        height: Plot height

    Returns:
        Card component with clustering plot
    """
    return dbc.Card([
        dbc.CardHeader("Clustering Visualization"),
        dbc.CardBody([
            create_loading_wrapper(
                content=dcc.Graph(id=plot_id, style={'height': height}),
                loading_id="loading-clustering-plot"
            )
        ])
    ])


def create_pca_loadings_plot(plot_id: str = 'pca-loadings-heatmap',
                           height: str = '400px',
                           container_id: str = 'pca-loadings-container') -> html.Div:
    """
    Create PCA loadings heatmap (shown only when PCA is selected).

    Args:
        plot_id: Plot element ID
        height: Plot height
        container_id: Container ID for show/hide

    Returns:
        Div component with PCA loadings plot
    """
    return html.Div(
        id=container_id,
        children=[
            dbc.Card([
                dbc.CardHeader("PCA Feature Loadings", className="bg-secondary text-white"),
                dbc.CardBody([
                    create_loading_wrapper(
                        content=dcc.Graph(id=plot_id, style={'height': height}),
                        loading_id="loading-pca-heatmap"
                    )
                ])
            ], className="mt-3")
        ],
        style={'display': 'none'}
    )


def create_operators_chart_container(chart_id: str = 'operators-chart',
                                   container_id: str = "operators-chart-container") -> html.Div:
    """
    Create operators chart container with styling.

    Args:
        chart_id: Chart element ID
        container_id: Container ID

    Returns:
        Styled chart container
    """
    return html.Div([
        dcc.Graph(id=chart_id)
    ], id=container_id, style={
        'overflow': 'hidden',
        'border': '2px dashed #ccc',
        'border-radius': '4px',
        'min-width': '400px',
        'min-height': '500px',
        'max-width': '100%',
        'position': 'relative'
    })


def create_all_operators_chart_container(chart_id: str = 'all-operators-chart',
                                       container_id: str = "all-operators-container") -> html.Div:
    """
    Create all operators chart container with scrolling.

    Args:
        chart_id: Chart element ID
        container_id: Container ID

    Returns:
        Scrollable chart container
    """
    return html.Div([
        dcc.Graph(id=chart_id)
    ], id=container_id, style={
        'overflow': 'auto',
        'border': '2px solid #ccc',
        'border-radius': '4px',
        'min-width': '500px',
        'min-height': '700px',
        'max-width': '100%'
    })


def create_datafields_chart_container(chart_id: str, title: str,
                                    container_style: Dict[str, Any] = None) -> html.Div:
    """
    Create datafields chart container with title.

    Args:
        chart_id: Chart element ID
        title: Chart title
        container_style: Optional container styling

    Returns:
        Chart container with title
    """
    default_style = {
        'height': '100%',
        'border': '1px solid #ddd',
        'border-radius': '8px',
        'padding': '10px',
        'background-color': 'white'
    }

    if container_style:
        default_style.update(container_style)

    return html.Div([
        html.Div([
            html.H6(title, className="mb-0"),
            html.Div([], style={'display': 'flex'})
        ], style={
            'display': 'flex',
            'justify-content': 'space-between',
            'align-items': 'center',
            'margin-bottom': '10px'
        }),
        dcc.Graph(id=chart_id, style={'height': 'calc(100% - 40px)'})
    ], style=default_style)


def create_statistics_panel(stats_id: str, title: str = "Usage Statistics") -> html.Div:
    """
    Create statistics panel component.

    Args:
        stats_id: Statistics element ID
        title: Panel title

    Returns:
        Statistics panel
    """
    return html.Div([
        html.H6(f"📊 {title}", className="mb-2"),
        html.Div(id=stats_id)
    ], style={
        'height': '100%',
        'border': '1px solid #ddd',
        'border-radius': '8px',
        'padding': '10px',
        'background-color': 'white'
    })


def create_responsive_chart_grid(charts: List[Dict[str, Any]]) -> html.Div:
    """
    Create responsive grid layout for charts.

    Args:
        charts: List of chart dictionaries with 'component', 'grid_area'

    Returns:
        Grid container with charts
    """
    grid_children = []
    for chart in charts:
        grid_children.append(
            html.Div([
                chart['component']
            ], style=chart.get('style', {}))
        )

    return html.Div(
        grid_children,
        style={
            'display': 'grid',
            'grid-template-columns': '1fr 1fr',
            'grid-template-rows': 'auto auto auto',
            'gap': '15px',
            'padding': '15px',
            'min-height': '1200px'
        }
    )


def create_chart_with_loading(chart_id: str, loading_id: str,
                             height: str = '400px',
                             loading_type: str = "default") -> dcc.Loading:
    """
    Create chart component with loading indicator.

    Args:
        chart_id: Chart element ID
        loading_id: Loading component ID
        height: Chart height
        loading_type: Loading animation type

    Returns:
        Loading component with chart
    """
    return create_loading_wrapper(
        content=dcc.Graph(id=chart_id, style={'height': height}),
        loading_id=loading_id,
        loading_type=loading_type
    )


def create_empty_chart_placeholder(message: str = "No data available") -> go.Figure:
    """
    Create empty chart placeholder.

    Args:
        message: Message to display

    Returns:
        Empty Plotly figure with message
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig


def create_treemap_container(chart_id: str = 'dataset-treemap-main',
                           height: str = '750px') -> dcc.Graph:
    """
    Create treemap container for full-width display.

    Args:
        chart_id: Chart element ID
        height: Chart height

    Returns:
        Graph component for treemap
    """
    return dcc.Graph(
        id=chart_id,
        style={'height': height}
    )


def create_chart_info_panel(info_id: str, title: str,
                           description: str = None) -> dbc.Card:
    """
    Create chart information panel.

    Args:
        info_id: Info element ID
        title: Panel title
        description: Optional description

    Returns:
        Information panel card
    """
    content = [html.Div(id=info_id, className="p-2")]

    if description:
        content.insert(0, html.P(description, className="text-muted mb-3"))

    return create_info_card(
        title=title,
        content=content
    )


def create_cluster_statistics_panel() -> dbc.Card:
    """
    Create cluster statistics panel.

    Returns:
        Card component for cluster statistics
    """
    return create_info_card(
        title="Cluster Statistics",
        content=html.Div(id='cluster-statistics', className="p-2")
    )


def create_alpha_details_panel() -> dbc.Card:
    """
    Create alpha details panel.

    Returns:
        Card component for alpha details
    """
    return create_info_card(
        title="Alpha Details",
        content=html.Div(id='alpha-details', className="p-3")
    )


def create_clustering_info_card() -> dbc.Card:
    """
    Create clustering information card with explanation.

    Returns:
        Information card component
    """
    return create_info_card(
        title="Clustering Information",
        content=[
            html.P([
                "🎨 Points are colored by ", html.Strong("automatically detected groups"), " of similar trading strategies. ",
                "Each color represents alphas with similar risk/return patterns."
            ], className="mb-2"),
            html.Small([
                html.Strong("HDBSCAN algorithm"), " automatically determines the optimal number of clusters based on data density."
            ], className="text-muted")
        ]
    )


def create_chart_controls_row(controls: List[Any]) -> dbc.Row:
    """
    Create row of chart controls.

    Args:
        controls: List of control components

    Returns:
        Bootstrap Row with controls
    """
    cols = []
    col_width = 12 // len(controls) if controls else 12

    for control in controls:
        cols.append(dbc.Col(control, width=col_width))

    return dbc.Row(cols, className="mb-3")


def create_chart_explanation_alert(title: str, points: List[str],
                                 alert_type: str = "light") -> dbc.Alert:
    """
    Create chart explanation alert with bullet points.

    Args:
        title: Alert title
        points: List of explanation points
        alert_type: Alert color type

    Returns:
        Alert component with explanation
    """
    return dbc.Alert([
        html.H6(title, className="mb-2"),
        html.Ul([
            html.Li(point) for point in points
        ], className="mb-0")
    ], color=alert_type)