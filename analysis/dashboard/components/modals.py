"""
Modal Components

Reusable modal components for detailed views and interactions.
Extracted from visualization_server.py with preserved functionality.
"""

from typing import List, Dict, Any, Optional, Tuple
from dash import html, dcc
import dash_bootstrap_components as dbc

from .base_components import create_alpha_badge, create_action_button
from .tables import create_alpha_table


def create_main_detail_modal() -> dbc.Modal:
    """
    Create main detail modal for chart interactions.

    Returns:
        Main detail modal component
    """
    return dbc.Modal([
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
            create_action_button(
                "Close",
                button_id="detail-modal-close",
                color="secondary",
                className="ms-auto"
            )
        ])
    ], id="detail-modal", is_open=False, size="lg")


def create_datafield_detail_modal() -> dbc.Modal:
    """
    Create datafield detail modal for recommendations.

    Returns:
        Datafield detail modal component
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="datafield-modal-title")),
        dbc.ModalBody(id="datafield-modal-body"),
        dbc.ModalFooter([
            create_action_button(
                "Close",
                button_id="datafield-modal-close",
                color="secondary",
                className="ms-auto"
            )
        ])
    ], id="datafield-detail-modal", size="lg", is_open=False)


def create_operator_modal_content(operator: str, count: int, chart_type: str,
                                alphas_using: List[str]) -> List[Any]:
    """
    Create modal content for operator details.

    Args:
        operator: Operator name
        count: Usage count
        chart_type: Chart type name
        alphas_using: List of alphas using this operator

    Returns:
        List of modal content components
    """
    content = [
        html.H5("📊 Usage Statistics", className="mb-3"),
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
            create_alpha_badge(
                alpha_id=alpha_id,
                href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                color="primary"
            ) for alpha_id in alphas_using
        ]

        content.append(
            html.Div(
                alpha_badges,
                className="mb-3",
                style={'max-height': '400px', 'overflow-y': 'auto'}
            )
        )
    else:
        content.append(
            dbc.Alert("No alpha data available for this operator.", color="warning")
        )

    return content


def create_datafield_modal_content(datafield: str, count: int, chart_type: str,
                                 alphas_using: List[str], dataset_id: str = "Unknown",
                                 category: str = "Unknown") -> List[Any]:
    """
    Create modal content for datafield details.

    Args:
        datafield: Datafield name
        count: Usage count
        chart_type: Chart type name
        alphas_using: List of alphas using this datafield
        dataset_id: Dataset ID
        category: Data category

    Returns:
        List of modal content components
    """
    content = [
        html.H5("📊 Usage Statistics", className="mb-3"),
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
            create_alpha_badge(
                alpha_id=alpha_id,
                href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                color="success"
            ) for alpha_id in alphas_using
        ]

        content.append(
            html.Div(
                alpha_badges,
                className="mb-3",
                style={'max-height': '400px', 'overflow-y': 'auto'}
            )
        )
    else:
        content.append(
            dbc.Alert("No alpha data available for this datafield.", color="warning")
        )

    return content


def create_dataset_modal_content(dataset: str, count: int,
                               dataset_datafields: List[Tuple[str, int]],
                               total_alphas: set) -> List[Any]:
    """
    Create modal content for dataset details.

    Args:
        dataset: Dataset name
        count: Total datafield instances
        dataset_datafields: List of (datafield, count) tuples
        total_alphas: Set of all alphas using this dataset

    Returns:
        List of modal content components
    """
    content = [
        html.H5("📊 Dataset Statistics", className="mb-3"),
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
        for df, df_count in dataset_datafields:
            datafield_items.append(
                dbc.ListGroupItem([
                    html.Strong(df),
                    html.Span(f"{df_count} alphas", className="badge bg-secondary ms-2 float-end")
                ])
            )

        content.extend([
            html.Div([
                dbc.ListGroup(datafield_items, flush=True)
            ], style={
                'max-height': '300px',
                'overflow-y': 'auto',
                'border': '1px solid #dee2e6',
                'border-radius': '4px'
            }, className="mb-3"),

            html.H6(f"🔗 All Alphas Using This Dataset ({len(total_alphas)} total)", className="mb-3"),
            html.Div([
                create_alpha_badge(
                    alpha_id=alpha_id,
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                    color="warning"
                ) for alpha_id in list(total_alphas)
            ], className="mb-3", style={'max-height': '400px', 'overflow-y': 'auto'})
        ])
    else:
        content.append(
            dbc.Alert("No datafield data available for this dataset.", color="warning")
        )

    return content


def create_alpha_details_modal_content(alpha_id: str, alpha_info: Dict[str, Any],
                                     wq_url: str) -> List[Any]:
    """
    Create modal content for single alpha details.

    Args:
        alpha_id: Alpha identifier
        alpha_info: Alpha information dictionary
        wq_url: WorldQuant Brain URL

    Returns:
        List of modal content components
    """
    return [
        html.H4(f"Alpha: {alpha_id}", className="text-primary mb-3"),

        # WorldQuant Brain button
        html.Div([
            html.A(
                create_action_button(
                    "View on WorldQuant Brain",
                    button_id="wq-brain-btn",
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


def create_dual_alpha_modal_content(alpha_x: str, alpha_y: str, correlation: float,
                                  alpha_x_info: Dict[str, Any],
                                  alpha_y_info: Dict[str, Any]) -> List[Any]:
    """
    Create modal content for dual alpha comparison (heatmap).

    Args:
        alpha_x: First alpha ID
        alpha_y: Second alpha ID
        correlation: Correlation value
        alpha_x_info: First alpha information
        alpha_y_info: Second alpha information

    Returns:
        List of modal content components
    """
    return [
        html.H4(f"Correlation: {correlation:.3f}", className="text-center text-primary mb-3"),
        html.Hr(),

        # Create two columns for the two alphas
        dbc.Row([
            # Alpha X column
            dbc.Col([
                html.H5(f"Alpha X: {alpha_x}", className="text-info mb-2"),
                html.A(
                    create_action_button("View on WQ Brain", "wq-brain-x", color="primary", size="sm", className="mb-2"),
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha_x}",
                    target="_blank"
                ),

                # Performance metrics
                html.H6("📊 Performance", className="text-success mt-2 mb-1"),
                html.Small([html.Strong("Sharpe: "), f"{alpha_x_info.get('is_sharpe', 0):.3f}"], className="d-block"),
                html.Small([html.Strong("Fitness: "), f"{alpha_x_info.get('is_fitness', 0):.3f}"], className="d-block"),
                html.Small([html.Strong("Returns: "), f"{alpha_x_info.get('is_returns', 0):.3f}"], className="d-block"),

                # Settings
                html.H6("⚙️ Settings", className="text-info mt-2 mb-1"),
                html.Small([html.Strong("Universe: "), alpha_x_info.get('universe', 'N/A')], className="d-block"),
                html.Small([html.Strong("Delay: "), str(alpha_x_info.get('delay', 'N/A'))], className="d-block"),

                # Expression
                html.H6("📝 Expression", className="text-warning mt-2 mb-1"),
                html.Code(
                    alpha_x_info.get('code', 'N/A')[:100] + ('...' if len(alpha_x_info.get('code', '')) > 100 else ''),
                    style={'font-size': '0.7rem', 'white-space': 'pre-wrap', 'word-wrap': 'break-word'}
                )
            ], width=6, style={'border-right': '1px solid #dee2e6'}),

            # Alpha Y column
            dbc.Col([
                html.H5(f"Alpha Y: {alpha_y}", className="text-info mb-2"),
                html.A(
                    create_action_button("View on WQ Brain", "wq-brain-y", color="primary", size="sm", className="mb-2"),
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha_y}",
                    target="_blank"
                ),

                # Performance metrics
                html.H6("📊 Performance", className="text-success mt-2 mb-1"),
                html.Small([html.Strong("Sharpe: "), f"{alpha_y_info.get('is_sharpe', 0):.3f}"], className="d-block"),
                html.Small([html.Strong("Fitness: "), f"{alpha_y_info.get('is_fitness', 0):.3f}"], className="d-block"),
                html.Small([html.Strong("Returns: "), f"{alpha_y_info.get('is_returns', 0):.3f}"], className="d-block"),

                # Settings
                html.H6("⚙️ Settings", className="text-info mt-2 mb-1"),
                html.Small([html.Strong("Universe: "), alpha_y_info.get('universe', 'N/A')], className="d-block"),
                html.Small([html.Strong("Delay: "), str(alpha_y_info.get('delay', 'N/A'))], className="d-block"),

                # Expression
                html.H6("📝 Expression", className="text-warning mt-2 mb-1"),
                html.Code(
                    alpha_y_info.get('code', 'N/A')[:100] + ('...' if len(alpha_y_info.get('code', '')) > 100 else ''),
                    style={'font-size': '0.7rem', 'white-space': 'pre-wrap', 'word-wrap': 'break-word'}
                )
            ], width=6)
        ])
    ]


def create_category_modal_content(category: str, count: int,
                                category_datafields: List[Tuple[str, int]],
                                total_alphas: set) -> List[Any]:
    """
    Create modal content for category details.

    Args:
        category: Category name
        count: Total usage count
        category_datafields: List of (datafield, count) tuples
        total_alphas: Set of all alphas in category

    Returns:
        List of modal content components
    """
    content = [
        html.H5("📊 Category Statistics", className="mb-3"),
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

        html.H6("🔍 Top Datafields in This Category", className="mb-3"),
    ]

    if category_datafields:
        # Show all datafields
        datafield_items = []
        for df, df_count in category_datafields:
            datafield_items.append(
                dbc.ListGroupItem([
                    html.Strong(df),
                    html.Span(f"{df_count} alphas", className="badge bg-secondary ms-2 float-end")
                ])
            )

        content.extend([
            html.Div([
                dbc.ListGroup(datafield_items, flush=True)
            ], style={
                'max-height': '300px',
                'overflow-y': 'auto',
                'border': '1px solid #dee2e6',
                'border-radius': '4px'
            }, className="mb-3"),

            html.H6(f"🔗 All Alphas in This Category ({len(total_alphas)} total)", className="mb-3"),
            html.Div([
                create_alpha_badge(
                    alpha_id=alpha_id,
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                    color="info"
                ) for alpha_id in list(total_alphas)
            ], className="mb-3", style={'max-height': '400px', 'overflow-y': 'auto'})
        ])
    else:
        content.append(
            dbc.Alert("No datafield data available for this category.", color="warning")
        )

    return content


def create_neutralization_modal_content(neutralization: str, count: int,
                                       matching_alphas: List[Dict[str, Any]]) -> List[Any]:
    """
    Create modal content for neutralization details.

    Args:
        neutralization: Neutralization type
        count: Total alphas count
        matching_alphas: List of alpha information dictionaries

    Returns:
        List of modal content components
    """
    content = [
        html.H5("📊 Neutralization Statistics", className="mb-3"),
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
                html.Span(
                    f"{sum(alpha['is_sharpe'] for alpha in matching_alphas) / len(matching_alphas):.3f}" if matching_alphas else "N/A",
                    className="badge bg-success ms-2"
                )
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

        content.extend([
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
                create_alpha_badge(
                    alpha_id=alpha['alpha_id'],
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha['alpha_id']}",
                    color="primary",
                    additional_classes="",
                    id={'type': 'alpha-badge', 'index': alpha['alpha_id']},
                    title=f"Sharpe: {alpha['is_sharpe']:.3f} | Universe: {alpha['universe']} | Region: {alpha['region_name']}"
                ) for alpha in matching_alphas[:100]  # Limit to first 100 for performance
            ], style={
                'max-height': '200px',
                'overflow-y': 'auto',
                'border': '1px solid #dee2e6',
                'padding': '10px',
                'border-radius': '4px'
            }, className="mb-3"),
        ])

        if len(matching_alphas) > 100:
            content.append(
                dbc.Alert(f"Showing first 100 alphas out of {len(matching_alphas)} total.", color="info", className="mt-2")
            )
    else:
        content.append(
            dbc.Alert("No alpha data available for this neutralization type.", color="warning")
        )

    return content


def create_error_modal_content(error_message: str) -> List[Any]:
    """
    Create modal content for error display.

    Args:
        error_message: Error message to display

    Returns:
        List of modal content components
    """
    return [
        dbc.Alert([
            html.H4("Error", className="alert-heading"),
            html.P(error_message),
        ], color="danger")
    ]