"""
Table Components

Reusable table and list components for displaying data.
Extracted from visualization_server.py with preserved functionality.
"""

from typing import List, Dict, Any, Optional, Tuple
from dash import html, dcc
import dash_bootstrap_components as dbc

from .base_components import create_alpha_badge, create_data_type_badge, create_action_button, create_clickable_badge_with_action


def create_alpha_table(alphas: List[Dict[str, Any]], max_rows: int = 20,
                      show_code: bool = True, table_id: str = None) -> dbc.Table:
    """
    Create table for displaying alpha information.

    Args:
        alphas: List of alpha dictionaries
        max_rows: Maximum rows to display
        show_code: Whether to show code preview column
        table_id: Table ID

    Returns:
        Bootstrap Table component
    """
    if not alphas:
        return html.Div("No alphas to display", className="text-muted text-center p-4")

    # Create table headers
    headers = ["Alpha ID", "Sharpe", "Fitness"]
    if show_code:
        headers.append("Code Preview")

    header_row = html.Tr([html.Th(header) for header in headers])

    # Create table rows
    rows = []
    for alpha in alphas[:max_rows]:
        cells = [
            html.A(
                alpha.get('alpha_id', 'Unknown'),
                href=f"https://platform.worldquantbrain.com/alpha/{alpha.get('alpha_id')}",
                target="_blank",
                className="text-decoration-none"
            ),
            f"{alpha.get('is_sharpe', 0):.3f}" if alpha.get('is_sharpe') else "N/A",
            f"{alpha.get('is_fitness', 0):.3f}" if alpha.get('is_fitness') else "N/A"
        ]

        if show_code:
            code = alpha.get('code', '')
            code_preview = html.Code(
                code[:50] + "..." if len(code) > 50 else code,
                className="small"
            )
            cells.append(html.Td(code_preview, title=code))
        else:
            cells = [html.Td(cell) for cell in cells]

        rows.append(html.Tr(cells))

    table_props = {
        'children': [
            html.Thead([header_row]),
            html.Tbody(rows)
        ],
        'striped': True,
        'hover': True,
        'responsive': True,
        'size': "sm"
    }

    if table_id:
        table_props['id'] = table_id

    return dbc.Table(**table_props)


def create_recommendations_table(recommendations: List[Dict[str, Any]]) -> dbc.Table:
    """
    Create recommendations table with interactive elements.

    Args:
        recommendations: List of recommendation dictionaries

    Returns:
        Recommendations table component
    """
    if not recommendations:
        return html.Div("No recommendations available", className="text-muted text-center p-4")

    # Create table rows
    table_rows = []
    for idx, rec in enumerate(recommendations):
        # Create clickable region badges for used regions
        used_badges = []
        for region in rec['used_in_regions']:
            usage_count = rec['usage_details'].get(region, 0)
            badge = dbc.Badge(
                f"{region} ({usage_count} alphas)",
                color="success",
                className="me-1",
                id={'type': 'datafield-used-badge', 'idx': idx, 'region': region, 'datafield': rec['datafield_id']},
                style={'cursor': 'pointer', 'transition': 'transform 0.2s'},
                n_clicks=0
            )
            used_badges.append(badge)

        # Create detailed recommended badges with clickable functionality
        recommended_badges = []
        availability_details = rec.get('availability_details', {})
        for region in rec['recommended_regions']:
            matching_ids = availability_details.get(region, [])
            if len(matching_ids) > 1:
                # Multiple matching datafields - make it clickable
                badge = create_clickable_badge_with_action(
                    text=f"{region} ({len(matching_ids)} IDs)",
                    action_id={'type': 'datafield-region-badge', 'idx': idx, 'region': region, 'datafield': rec['datafield_id']},
                    color="primary",
                    tooltip=f"Click to view {len(matching_ids)} available datafields in {region}"
                )
            elif len(matching_ids) == 1:
                # Single datafield but still make clickable for consistency
                badge = create_clickable_badge_with_action(
                    text=region,
                    action_id={'type': 'datafield-region-badge', 'idx': idx, 'region': region, 'datafield': rec['datafield_id']},
                    color="primary",
                    tooltip=f"Click to view datafield details for {region}"
                )
            else:
                # Fallback - no matching IDs
                badge = dbc.Badge(region, color="primary", className="me-1")
            recommended_badges.append(badge)

        # Usage details
        usage_text = ", ".join([
            f"{region}: {count}"
            for region, count in rec['usage_details'].items()
        ])

        # Check if this is a description-based match
        matching_datafields = rec.get('matching_datafields', {})
        if len(matching_datafields) > 1:
            # Show indicator that this matches multiple datafield IDs
            datafield_display = html.Span([
                rec['datafield_id'],
                html.Small(f" (+{len(matching_datafields)-1} similar)", className="text-muted ms-1")
            ])
        else:
            datafield_display = rec['datafield_id']

        table_rows.append(
            html.Tr([
                html.Td(datafield_display),
                html.Td(
                    rec['description'][:100] + ('...' if len(rec['description']) > 100 else ''),
                    title=rec['description']
                ),
                html.Td(create_data_type_badge(rec.get('data_type', 'Unknown'))),
                html.Td(usage_text),
                html.Td(used_badges),
                html.Td(recommended_badges),
                html.Td(
                    create_action_button(
                        "View Alphas",
                        button_id={'type': 'view-datafield-alphas', 'index': rec['datafield_id']},
                        size="sm",
                        color="info",
                        outline=True
                    )
                )
            ])
        )

    # Create the table
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Datafield ID"),
                html.Th("Description"),
                html.Th("Type"),
                html.Th("Usage Count"),
                html.Th("Used In"),
                html.Th("Recommended For"),
                html.Th("Actions")
            ])
        ]),
        html.Tbody(table_rows)
    ], striped=True, bordered=True, hover=True, responsive=True, size="sm")


def create_statistics_list(statistics: List[Dict[str, Any]]) -> dbc.ListGroup:
    """
    Create statistics list component.

    Args:
        statistics: List of statistic dictionaries

    Returns:
        ListGroup component with statistics
    """
    items = []
    for stat in statistics:
        label = stat.get('label', '')
        value = stat.get('value', '')
        badge_color = stat.get('badge_color', 'primary')
        description = stat.get('description', '')

        item_content = [
            html.Strong(f"{label}: "),
            html.Span(str(value), className=f"badge bg-{badge_color} ms-2")
        ]

        if description:
            item_content.extend([html.Br(), html.Small(description, className="text-muted")])

        items.append(dbc.ListGroupItem(item_content))

    return dbc.ListGroup(items, flush=True)


def create_scrollable_list(items: List[Any], max_height: str = "300px",
                          list_id: str = None, className: str = "") -> html.Div:
    """
    Create scrollable list container.

    Args:
        items: List of items to display
        max_height: Maximum container height
        list_id: List container ID
        className: Additional CSS classes

    Returns:
        Scrollable div container
    """
    div_props = {
        'children': items,
        'style': {
            'max-height': max_height,
            'overflow-y': 'auto',
            'border': '1px solid #dee2e6',
            'border-radius': '4px'
        },
        'className': f"scrollable-content {className}".strip()
    }

    if list_id:
        div_props['id'] = list_id

    return html.Div(**div_props)


def create_alpha_summary_card(total_alphas: int, summary_stats: List[Dict[str, Any]]) -> dbc.Card:
    """
    Create alpha summary card with statistics.

    Args:
        total_alphas: Total number of alphas
        summary_stats: List of summary statistics

    Returns:
        Summary card component
    """
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Total Alphas Analyzed", className="text-muted"),
                    html.H3(str(total_alphas))
                ], md=4)
            ] + [
                dbc.Col([
                    html.H6(stat.get('label', ''), className="text-muted"),
                    html.H3(str(stat.get('value', '')))
                ], md=4) for stat in summary_stats[:2]  # Limit to 3 total columns
            ])
        ])
    ], className="mb-4")


def create_breakdown_table(items: List[Tuple[str, int]], title: str = "Breakdown",
                          show_percentage: bool = True, total: int = None) -> html.Div:
    """
    Create breakdown table with counts and percentages.

    Args:
        items: List of (item_name, count) tuples
        title: Table title
        show_percentage: Whether to show percentages
        total: Total for percentage calculation

    Returns:
        Breakdown table component
    """
    if not items:
        return html.Div()

    if total is None:
        total = sum(count for _, count in items)

    rows = []
    for item_name, count in items:
        cells = [html.Td(html.Strong(f"{item_name}: ")), html.Td(f"{count} alphas")]

        if show_percentage and total > 0:
            percentage = (count / total * 100)
            cells.append(html.Td(f"({percentage:.1f}%)", className="text-muted small"))

        rows.append(html.Tr(cells))

    return html.Div([
        html.H6(title, className="mb-2"),
        dbc.Table([
            html.Tbody(rows)
        ], borderless=True, size="sm")
    ])


def create_expandable_alpha_list(alpha_ids: List[str], operator: str = None,
                               datafield: str = None, initial_limit: int = 50) -> html.Div:
    """
    Create expandable alpha list with show more/less functionality.

    Args:
        alpha_ids: List of alpha IDs
        operator: Operator name (for IDs)
        datafield: Datafield name (for IDs)
        initial_limit: Initial number of alphas to show

    Returns:
        Expandable alpha list component
    """
    if not alpha_ids:
        return html.Div("No alphas found", className="text-muted")

    # Determine color and ID type based on context
    if operator:
        color = "primary"
        container_id = {'type': 'alpha-list-container', 'operator': operator}
        button_id = {'type': 'show-all-alphas-btn', 'operator': operator}
    elif datafield:
        color = "success"
        container_id = {'type': 'alpha-list-container-df', 'datafield': datafield}
        button_id = {'type': 'show-all-alphas-btn-df', 'datafield': datafield}
    else:
        color = "secondary"
        container_id = "alpha-list-container"
        button_id = "show-all-alphas-btn"

    # Show initial set of alphas
    initial_alphas = alpha_ids[:initial_limit]
    badges = [
        create_alpha_badge(
            alpha_id=alpha_id,
            href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
            color=color,
            id={'type': 'alpha-badge', 'index': alpha_id}
        ) for alpha_id in initial_alphas
    ]

    container_content = [
        html.Div(
            badges,
            style={
                'max-height': '200px',
                'overflow-y': 'auto',
                'border': '1px solid #dee2e6',
                'padding': '10px',
                'border-radius': '4px'
            }
        )
    ]

    # Add "show more" button if there are more alphas
    if len(alpha_ids) > initial_limit:
        container_content.append(
            html.Div([
                create_action_button(
                    f"Show all ({len(alpha_ids)} total)",
                    button_id=button_id,
                    color="secondary",
                    size="sm",
                    className="mt-2"
                )
            ])
        )

    return html.Div(container_content, id=container_id)


def create_datafield_details_table(datafield_info_list: List[Dict[str, Any]]) -> List[Any]:
    """
    Create detailed table for datafield information.

    Args:
        datafield_info_list: List of datafield information dictionaries

    Returns:
        List of card components with datafield details
    """
    cards = []
    for i, df_info in enumerate(datafield_info_list, 1):
        card = dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.Span(f"Datafield {i}: ", className="text-muted"),
                    html.Code(df_info['id'], className="bg-light p-1")
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Strong("Dataset: "),
                        html.Span(df_info['dataset'], className="badge bg-success ms-2")
                    ], md=6),
                    dbc.Col([
                        html.Strong("Category: "),
                        html.Span(df_info['category'], className="badge bg-info ms-2")
                    ], md=6)
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        html.Strong("Type: "),
                        html.Span(df_info['data_type'], className="badge bg-secondary ms-2")
                    ], md=6),
                    dbc.Col([
                        html.Strong("Delay: "),
                        html.Span(str(df_info['delay']), className="badge bg-warning ms-2")
                    ], md=6)
                ])
            ])
        ], className="mb-3")
        cards.append(card)

    return cards


def create_usage_breakdown_list(breakdown_data: Dict[str, int],
                              title: str = "Usage Breakdown") -> html.Div:
    """
    Create usage breakdown list.

    Args:
        breakdown_data: Dictionary of item -> count
        title: List title

    Returns:
        Breakdown list component
    """
    if not breakdown_data:
        return html.Div()

    items = []
    for item, count in sorted(breakdown_data.items()):
        items.append(html.Li(f"{item}: {count}"))

    return html.Div([
        html.Strong(f"{title}:"),
        html.Ul(items)
    ])


def create_cluster_breakdown_list(cluster_info: Dict[str, List[str]],
                                total_alphas: int) -> dbc.ListGroup:
    """
    Create cluster breakdown list with color indicators.

    Args:
        cluster_info: Dictionary mapping cluster names to alpha lists
        total_alphas: Total number of alphas

    Returns:
        ListGroup with cluster breakdown
    """
    from ..config import COLOR_SCHEMES

    # Custom sorting function for proper cluster ordering
    def cluster_sort_key(item):
        cluster_name, alphas = item
        if cluster_name == "Outliers":
            return (1, 0)  # Sort Outliers last
        else:
            # Extract numeric part from "Cluster X"
            try:
                cluster_num = int(cluster_name.split()[-1])
                return (0, cluster_num)  # Sort clusters numerically
            except (ValueError, IndexError):
                return (2, cluster_name)  # Unknown format

    cluster_items = []
    colors = COLOR_SCHEMES['cluster_colors']
    cluster_idx = 0

    for cluster_name, alphas in sorted(cluster_info.items(), key=cluster_sort_key):
        if cluster_name == "Outliers":
            color = COLOR_SCHEMES['outlier_color']
        else:
            color = colors[cluster_idx % len(colors)]
            cluster_idx += 1

        cluster_items.append(
            dbc.ListGroupItem([
                html.Span([
                    html.Span("●", style={"color": color, "font-size": "16px", "font-weight": "bold"}),
                    f" {cluster_name}: {len(alphas)} alphas",
                ]),
                html.Small(f" ({len(alphas)/total_alphas*100:.1f}%)", className="text-muted")
            ])
        )

    return dbc.ListGroup(cluster_items, flush=True)


def create_dataset_info_list(dataset_datafields: List[Tuple[str, int]],
                           max_height: str = "300px") -> html.Div:
    """
    Create dataset information list.

    Args:
        dataset_datafields: List of (datafield, count) tuples
        max_height: Maximum container height

    Returns:
        Scrollable dataset info list
    """
    if not dataset_datafields:
        return html.Div("No datafields available", className="text-muted")

    datafield_items = []
    for df, df_count in dataset_datafields:
        datafield_items.append(
            dbc.ListGroupItem([
                html.Strong(df),
                html.Span(f"{df_count} alphas", className="badge bg-secondary ms-2 float-end")
            ])
        )

    return html.Div([
        dbc.ListGroup(datafield_items, flush=True)
    ], style={
        'max-height': max_height,
        'overflow-y': 'auto',
        'border': '1px solid #dee2e6',
        'border-radius': '4px'
    })


def create_performance_summary_table(alphas: List[Dict[str, Any]],
                                   max_rows: int = 10) -> dbc.Table:
    """
    Create performance summary table.

    Args:
        alphas: List of alpha dictionaries
        max_rows: Maximum rows to show

    Returns:
        Performance summary table
    """
    if not alphas:
        return html.Div("No performance data available", className="text-muted")

    rows = []
    for alpha in alphas[:max_rows]:
        rows.append(html.Tr([
            html.Td(alpha.get('alpha_id', 'Unknown')),
            html.Td(f"{alpha.get('is_sharpe', 0):.3f}"),
            html.Td(f"{alpha.get('is_fitness', 0):.3f}"),
            html.Td(alpha.get('universe', 'N/A')),
            html.Td(alpha.get('region_name', 'N/A'))
        ]))

    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Alpha ID"),
                html.Th("Sharpe"),
                html.Th("Fitness"),
                html.Th("Universe"),
                html.Th("Region")
            ])
        ]),
        html.Tbody(rows)
    ], striped=True, hover=True, responsive=True, size="sm")


def create_region_alpha_breakdown(usage_info: Dict[str, List[str]],
                                limit_per_region: int = 20) -> List[Any]:
    """
    Create breakdown of alphas by region.

    Args:
        usage_info: Dictionary mapping regions to alpha lists
        limit_per_region: Maximum alphas to show per region

    Returns:
        List of region breakdown components
    """
    content_parts = []

    for region, alpha_ids in usage_info.items():
        region_section = html.Div([
            html.H5(f"{region} Region ({len(alpha_ids)} alphas)", className="mb-2"),
            html.Div([
                create_alpha_badge(
                    alpha_id=alpha_id,
                    href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                    color="primary"
                ) for alpha_id in alpha_ids[:limit_per_region]
            ], className="mb-3"),
            html.Small(
                f"Showing {min(limit_per_region, len(alpha_ids))} of {len(alpha_ids)} alphas"
            ) if len(alpha_ids) > limit_per_region else None,
            html.Hr()
        ])
        content_parts.append(region_section)

    return content_parts


def create_interactive_badge_list(items: List[Dict[str, Any]],
                                badge_type: str = "clickable") -> html.Div:
    """
    Create interactive badge list.

    Args:
        items: List of item dictionaries with badge properties
        badge_type: Type of badges to create

    Returns:
        Div with interactive badges
    """
    badges = []
    for item in items:
        if badge_type == "clickable":
            badge = create_clickable_badge_with_action(
                text=item.get('text', ''),
                action_id=item.get('action_id', {}),
                color=item.get('color', 'primary'),
                tooltip=item.get('tooltip', '')
            )
        else:
            badge = create_alpha_badge(
                alpha_id=item.get('text', ''),
                color=item.get('color', 'primary'),
                href=item.get('href'),
                **item.get('props', {})
            )
        badges.append(badge)

    return html.Div(badges)