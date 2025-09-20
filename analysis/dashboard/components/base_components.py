"""
Base UI Components

Common UI elements used throughout the dashboard.
Extracted from visualization_server.py with preserved styling and functionality.
"""

from typing import List, Dict, Any, Optional, Union
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_dashboard_header(title: str = "Alpha Analysis Dashboard",
                          subtitle: str = None) -> dbc.Row:
    """
    Create dashboard header with title and subtitle.

    Args:
        title: Main dashboard title
        subtitle: Optional subtitle

    Returns:
        Bootstrap Row component
    """
    if subtitle is None:
        from datetime import datetime
        subtitle = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Analysis & Clustering Platform"

    return dbc.Row([
        dbc.Col([
            html.H1(title, className="text-center my-4"),
            html.P(subtitle, className="text-center text-muted mb-4"),
        ], width=12)
    ])


def create_info_card(title: str, content: Any, header_color: str = None,
                    className: str = "mb-3") -> dbc.Card:
    """
    Create information card with header and content.

    Args:
        title: Card title
        content: Card content (can be HTML components)
        header_color: Optional header color class
        className: CSS classes for the card

    Returns:
        Bootstrap Card component
    """
    header_props = {"children": title}
    if header_color:
        header_props["className"] = header_color

    return dbc.Card([
        dbc.CardHeader(**header_props),
        dbc.CardBody(content)
    ], className=className)


def create_statistics_card(title: str, stats: List[Dict[str, Any]],
                          className: str = "mb-3") -> dbc.Card:
    """
    Create statistics card with list of statistics.

    Args:
        title: Card title
        stats: List of stat dictionaries with 'label', 'value', 'badge_color'
        className: CSS classes for the card

    Returns:
        Bootstrap Card component
    """
    stat_items = []
    for stat in stats:
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

        stat_items.append(dbc.ListGroupItem(item_content))

    return create_info_card(
        title=title,
        content=dbc.ListGroup(stat_items, flush=True),
        className=className
    )


def create_method_selector(methods: List[Dict[str, Any]], selected_value: str = 'mds',
                          element_id: str = 'method-selector') -> dcc.RadioItems:
    """
    Create method selection radio items.

    Args:
        methods: List of method dictionaries with 'label', 'value', 'disabled'
        selected_value: Default selected value
        element_id: Element ID

    Returns:
        RadioItems component
    """
    return dcc.RadioItems(
        id=element_id,
        options=methods,
        value=selected_value,
        inline=False,
        className="mb-3"
    )


def create_view_mode_selector(view_modes: List[Dict[str, str]], selected_mode: str,
                             element_id: str, inline: bool = True) -> dbc.RadioItems:
    """
    Create view mode selector.

    Args:
        view_modes: List of view mode options
        selected_mode: Currently selected mode
        element_id: Element ID
        inline: Whether to display inline

    Returns:
        RadioItems component
    """
    return dbc.RadioItems(
        id=element_id,
        options=view_modes,
        value=selected_mode,
        inline=inline
    )


def create_alpha_badge(alpha_id: str, color: str = "primary",
                      href: str = None, clickable: bool = True,
                      additional_classes: str = "", **kwargs) -> dbc.Badge:
    """
    Create standardized alpha badge.

    Args:
        alpha_id: Alpha identifier
        color: Badge color
        href: Optional URL for external link
        clickable: Whether badge should be clickable
        additional_classes: Additional CSS classes
        **kwargs: Additional properties

    Returns:
        Bootstrap Badge component
    """
    base_classes = f"me-1 mb-1 {additional_classes}".strip()

    if clickable:
        base_classes += " alpha-badge"

    badge_props = {
        'children': alpha_id,
        'color': color,
        'className': base_classes,
        **kwargs
    }

    if clickable:
        badge_props['style'] = {'cursor': 'pointer', 'text-decoration': 'none'}

    if href:
        badge_props['href'] = href
        badge_props['target'] = "_blank"

    return dbc.Badge(**badge_props)


def create_alpha_list_container(alpha_ids: List[str], color: str = "primary",
                               max_height: str = "200px", show_limit: int = 50,
                               container_id: str = None) -> html.Div:
    """
    Create scrollable container for alpha lists.

    Args:
        alpha_ids: List of alpha IDs
        color: Badge color
        max_height: Maximum container height
        show_limit: Maximum alphas to show initially
        container_id: Container ID

    Returns:
        Div component with alpha badges
    """
    # Create badges for alphas
    badges = []
    for i, alpha_id in enumerate(alpha_ids):
        if show_limit and i >= show_limit:
            break

        badge = create_alpha_badge(
            alpha_id=alpha_id,
            color=color,
            href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
            id={'type': 'alpha-badge', 'index': alpha_id}
        )
        badges.append(badge)

    container_props = {
        'children': badges,
        'className': "scrollable-content",
        'style': {
            'max-height': max_height,
            'overflow-y': 'auto',
            'border': '1px solid #dee2e6',
            'padding': '10px',
            'border-radius': '4px'
        }
    }

    if container_id:
        container_props['id'] = container_id

    container = html.Div(**container_props)

    # Add "show more" button if we have more alphas
    if show_limit and len(alpha_ids) > show_limit:
        return html.Div([
            container,
            html.Div([
                dbc.Button(
                    f"Show all ({len(alpha_ids)} total)",
                    color="secondary",
                    size="sm",
                    className="mt-2"
                )
            ])
        ])

    return container


def create_loading_wrapper(content: Any, loading_id: str,
                          loading_type: str = "default") -> dcc.Loading:
    """
    Create loading wrapper for content.

    Args:
        content: Content to wrap with loading
        loading_id: Loading component ID
        loading_type: Loading animation type

    Returns:
        Loading component
    """
    return dcc.Loading(
        id=loading_id,
        type=loading_type,
        children=content
    )


def create_expandable_list(items: List[str], title: str, color: str = "primary",
                          default_show: int = 10, badge_type: str = "text") -> dbc.AccordionItem:
    """
    Create expandable list in accordion format.

    Args:
        items: List of items to display
        title: Accordion title
        color: Badge color
        default_show: Number of items to show by default
        badge_type: Type of badge content ('text' or 'count')

    Returns:
        AccordionItem component
    """
    if badge_type == "count":
        badges = [
            dbc.Badge(f"{item[0]} ({item[1]})", color=color, className="me-1 mb-1")
            for item in items
        ]
    else:
        badges = [
            dbc.Badge(str(item), color=color, className="me-1 mb-1")
            for item in items
        ]

    content = html.Div(badges)
    if len(items) > default_show:
        content = html.Div(
            badges,
            style={'max-height': '400px', 'overflow-y': 'auto'}
        )

    return dbc.AccordionItem(
        content,
        title=f"{title} ({len(items)})",
        item_id=title.lower().replace(' ', '-')
    )


def create_alert_message(message: str, alert_type: str = "info",
                        heading: str = None, dismissible: bool = False) -> dbc.Alert:
    """
    Create alert message component.

    Args:
        message: Alert message
        alert_type: Alert type ('success', 'info', 'warning', 'danger')
        heading: Optional alert heading
        dismissible: Whether alert can be dismissed

    Returns:
        Bootstrap Alert component
    """
    content = [html.P(message, className="mb-0")]

    if heading:
        content.insert(0, html.H4(heading, className="alert-heading"))

    return dbc.Alert(
        content,
        color=alert_type,
        dismissable=dismissible
    )


def create_progress_indicator(current: int, total: int, label: str = "") -> html.Div:
    """
    Create progress indicator.

    Args:
        current: Current progress value
        total: Total value
        label: Optional label

    Returns:
        Div with progress indicator
    """
    percentage = (current / total * 100) if total > 0 else 0

    return html.Div([
        html.Label(label, className="form-label") if label else None,
        dbc.Progress(
            value=percentage,
            striped=True,
            animated=True if percentage < 100 else False,
            color="success" if percentage == 100 else "primary",
            className="mb-2"
        ),
        html.Small(f"{current} / {total} ({percentage:.1f}%)", className="text-muted")
    ])


def create_tooltip_badge(text: str, tooltip: str, color: str = "primary",
                        className: str = "me-1 mb-1") -> html.Span:
    """
    Create badge with tooltip.

    Args:
        text: Badge text
        tooltip: Tooltip text
        color: Badge color
        className: CSS classes

    Returns:
        Span with badge and tooltip
    """
    return html.Span([
        dbc.Badge(
            text,
            color=color,
            className=className,
            id=f"tooltip-{hash(text + tooltip)}",
            style={'cursor': 'help'}
        ),
        dbc.Tooltip(
            tooltip,
            target=f"tooltip-{hash(text + tooltip)}",
            placement="top"
        )
    ])


def create_clickable_badge_with_action(text: str, action_id: Dict[str, Any],
                                     color: str = "primary", tooltip: str = None,
                                     className: str = "me-1 mb-1 datafield-clickable-badge") -> dbc.Badge:
    """
    Create clickable badge with action ID for callbacks.

    Args:
        text: Badge text
        action_id: Action ID dictionary for pattern-matching callbacks
        color: Badge color
        tooltip: Optional tooltip text
        className: CSS classes

    Returns:
        Clickable Badge component
    """
    badge_props = {
        'children': text,
        'id': action_id,
        'color': color,
        'className': className,
        'style': {'cursor': 'pointer', 'transition': 'all 0.2s ease'},
        'n_clicks': 0
    }

    if tooltip:
        badge_props['title'] = tooltip

    return dbc.Badge(**badge_props)


def create_summary_grid(items: List[Dict[str, Any]], columns: int = 3) -> dbc.Row:
    """
    Create summary statistics grid.

    Args:
        items: List of summary items with 'title', 'value', 'color', 'description'
        columns: Number of columns in grid

    Returns:
        Bootstrap Row with summary cards
    """
    cols = []
    col_width = 12 // columns

    for item in items:
        card = dbc.Card([
            dbc.CardBody([
                html.H6(item.get('title', ''), className=f"text-{item.get('color', 'primary')}"),
                html.H4(str(item.get('value', '')), className=f"text-{item.get('color', 'primary')}"),
                html.Small(item.get('description', ''))
            ])
        ])
        cols.append(dbc.Col(card, width=col_width))

    return dbc.Row(cols, className="mb-4")


def create_data_type_badge(data_type: str) -> dbc.Badge:
    """
    Create standardized data type badge.

    Args:
        data_type: Data type ('MATRIX', 'VECTOR', 'GROUP', etc.)

    Returns:
        Styled Badge component
    """
    color_map = {
        'MATRIX': 'secondary',
        'VECTOR': 'primary',
        'GROUP': 'success'
    }

    return dbc.Badge(
        data_type,
        color=color_map.get(data_type, 'light'),
        className="text-uppercase"
    )


def create_region_info_text(alpha_count: int, timestamp: str) -> str:
    """
    Create region information text.

    Args:
        alpha_count: Number of alphas
        timestamp: Generation timestamp

    Returns:
        Formatted info text
    """
    return f"{alpha_count} alphas | Generated: {timestamp}"


def create_usage_percentage_badge(used: int, total: int) -> html.Span:
    """
    Create usage percentage badge.

    Args:
        used: Number used
        total: Total number

    Returns:
        Span with usage percentage
    """
    percentage = (used / total * 100) if total > 0 else 0

    if percentage >= 80:
        color = "success"
    elif percentage >= 50:
        color = "warning"
    else:
        color = "danger"

    return html.Span([
        html.Span(f"{used}/{total}", className="me-2"),
        dbc.Badge(f"{percentage:.1f}%", color=color, className="small")
    ])


def create_interactive_table_row(cells: List[Any], row_id: str = None) -> html.Tr:
    """
    Create interactive table row.

    Args:
        cells: List of cell contents
        row_id: Optional row ID

    Returns:
        Table row component
    """
    row_props = {
        'children': [html.Td(cell) for cell in cells]
    }

    if row_id:
        row_props['id'] = row_id

    return html.Tr(**row_props)


def create_collapsible_section(title: str, content: Any,
                              default_open: bool = False,
                              section_id: str = None) -> dbc.Collapse:
    """
    Create collapsible section.

    Args:
        title: Section title
        content: Section content
        default_open: Whether section is open by default
        section_id: Section ID

    Returns:
        Collapsible section
    """
    if section_id is None:
        section_id = f"collapse-{hash(title)}"

    return html.Div([
        dbc.Button(
            title,
            id=f"{section_id}-toggle",
            className="mb-2",
            color="outline-primary",
            n_clicks=0
        ),
        dbc.Collapse(
            content,
            id=section_id,
            is_open=default_open
        )
    ])


def create_legend_component(legend_items: List[Dict[str, str]]) -> html.Div:
    """
    Create legend component for charts.

    Args:
        legend_items: List of legend items with 'color', 'label', 'description'

    Returns:
        Legend component
    """
    legend_elements = []

    for item in legend_items:
        color = item.get('color', '#808080')
        label = item.get('label', '')
        description = item.get('description', '')

        legend_elements.append(
            html.Div([
                html.Div(style={
                    'width': '20px', 'height': '20px', 'background-color': color,
                    'display': 'inline-block', 'margin-right': '10px',
                    'border': '1px solid #ccc'
                }),
                html.Span(f"{label}: {description}", style={'vertical-align': 'top'})
            ], className="mb-2")
        )

    return html.Div(legend_elements)


def create_section_header(title: str, icon: str = "", subtitle: str = "") -> html.Div:
    """
    Create section header with optional icon and subtitle.

    Args:
        title: Section title
        icon: Optional emoji icon
        subtitle: Optional subtitle

    Returns:
        Section header div
    """
    header_content = f"{icon} {title}".strip()

    elements = [html.H5(header_content, className="mb-2")]

    if subtitle:
        elements.append(html.P(subtitle, className="text-muted mb-3"))

    return html.Div(elements)


def create_empty_state_message(message: str, icon: str = "📊",
                              suggestion: str = None) -> html.Div:
    """
    Create empty state message.

    Args:
        message: Main message
        icon: Optional icon
        suggestion: Optional suggestion text

    Returns:
        Empty state component
    """
    content = [
        html.H4(f"{icon} {message}", className="text-muted text-center"),
    ]

    if suggestion:
        content.append(
            html.P(suggestion, className="text-muted text-center small")
        )

    return html.Div(content, className="p-5")


def create_action_button(text: str, button_id: Any, color: str = "primary",
                        size: str = "md", outline: bool = False,
                        className: str = "") -> dbc.Button:
    """
    Create action button with consistent styling.

    Args:
        text: Button text
        button_id: Button ID (can be dict for pattern-matching)
        color: Button color
        size: Button size
        outline: Whether button is outline style
        className: Additional CSS classes

    Returns:
        Bootstrap Button component
    """
    return dbc.Button(
        text,
        id=button_id,
        color=color,
        size=size,
        outline=outline,
        className=className,
        n_clicks=0
    )