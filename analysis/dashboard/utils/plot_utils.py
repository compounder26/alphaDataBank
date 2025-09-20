"""
Plot Utilities

Helper functions for Plotly chart creation and styling.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

from ..config import COLOR_SCHEMES, get_plotly_layout_defaults, get_scatter_trace_defaults


def get_cluster_color_map(unique_clusters: List[str]) -> Dict[str, str]:
    """
    Create color mapping for clusters.

    Args:
        unique_clusters: List of unique cluster names

    Returns:
        Dictionary mapping cluster names to colors
    """
    colors = COLOR_SCHEMES['cluster_colors']
    color_map = {}

    # Custom sorting for cluster strings (Cluster 0, 1, 2... then Outlier)
    def cluster_sort_key(cluster_str):
        if cluster_str == 'Outlier':
            return (1, 0)  # Sort Outlier last
        else:
            # Extract numeric part from "Cluster X"
            try:
                cluster_num = int(cluster_str.split()[-1])
                return (0, cluster_num)  # Sort clusters numerically
            except (ValueError, IndexError):
                return (2, cluster_str)  # Unknown format

    sorted_clusters = sorted(unique_clusters, key=cluster_sort_key)

    cluster_idx = 0
    for cluster in sorted_clusters:
        if cluster == 'Outlier':
            color_map[cluster] = COLOR_SCHEMES['outlier_color']
        else:
            color_map[cluster] = colors[cluster_idx % len(colors)]
            cluster_idx += 1

    return color_map


def create_scatter_trace(x_data: List[float], y_data: List[float],
                        text: List[str], hover_text: List[str],
                        customdata: List[str], name: str,
                        color: str, size: int = 12) -> go.Scatter:
    """
    Create a standardized scatter trace.

    Args:
        x_data: X coordinates
        y_data: Y coordinates
        text: Text labels
        hover_text: Hover text
        customdata: Custom data for callbacks
        name: Trace name
        color: Marker color
        size: Marker size

    Returns:
        Plotly Scatter trace
    """
    trace_defaults = get_scatter_trace_defaults()

    return go.Scatter(
        x=x_data,
        y=y_data,
        mode=trace_defaults['mode'],
        name=name,
        marker=dict(
            color=color,
            size=size,
            line=trace_defaults['marker']['line']
        ),
        customdata=customdata,
        text=text,
        hovertext=hover_text,
        hovertemplate=trace_defaults['hovertemplate']
    )


def create_highlight_trace(x_data: List[float], y_data: List[float],
                          color: str, name: str, size: int = 12,
                          opacity: float = 0.7) -> go.Scatter:
    """
    Create a highlight trace for selected points.

    Args:
        x_data: X coordinates
        y_data: Y coordinates
        color: Highlight color
        name: Trace name
        size: Marker size
        opacity: Marker opacity

    Returns:
        Plotly Scatter trace for highlights
    """
    return go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(
            color=color,
            size=size,
            opacity=opacity,
            line=dict(width=1, color='darkgreen' if 'green' in color else 'black')
        ),
        hoverinfo='skip',
        showlegend=True,
        name=name
    )


def add_plot_annotations(fig: go.Figure, annotations: List[str],
                        positions: List[Tuple[float, float]] = None) -> None:
    """
    Add annotations to a plot.

    Args:
        fig: Plotly figure
        annotations: List of annotation texts
        positions: List of (x, y) positions for annotations
    """
    if not positions:
        # Default positions at bottom of plot
        positions = [(0, -0.12 - i * 0.04) for i in range(len(annotations))]

    plot_annotations = []
    for i, (text, (x, y)) in enumerate(zip(annotations, positions)):
        plot_annotations.append(dict(
            text=text,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            font=dict(size=11 if i == 0 else 10),
            xanchor="left"
        ))

    fig.update_layout(
        annotations=plot_annotations,
        margin=dict(b=80 + len(annotations) * 20)  # Add margin for annotations
    )


def apply_chart_template(fig: go.Figure, title: str,
                        x_label: str = None, y_label: str = None,
                        height: int = None, width: int = None) -> None:
    """
    Apply standard template to chart.

    Args:
        fig: Plotly figure
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        height: Chart height
        width: Chart width
    """
    layout_defaults = get_plotly_layout_defaults()

    update_dict = {
        'title': title,
        **layout_defaults
    }

    if x_label:
        update_dict['xaxis_title'] = x_label
    if y_label:
        update_dict['yaxis_title'] = y_label
    if height:
        update_dict['height'] = height
    if width:
        update_dict['width'] = width

    fig.update_layout(**update_dict)


def create_bar_chart(data: List[Tuple[str, Union[int, float]]],
                     title: str, x_label: str = "Count",
                     y_label: str = "Item", color: str = "steelblue",
                     height: int = 600, max_items: int = None) -> go.Figure:
    """
    Create a horizontal bar chart.

    Args:
        data: List of (label, value) tuples
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        color: Bar color
        height: Chart height
        max_items: Maximum number of items to show

    Returns:
        Plotly Figure
    """
    if max_items:
        data = data[:max_items]

    labels, values = zip(*data) if data else ([], [])

    fig = px.bar(
        x=values,
        y=labels,
        orientation='h',
        title=title,
        labels={'x': x_label, 'y': y_label}
    )

    fig.update_traces(
        marker_color=color,
        hovertemplate=f'<b>%{{y}}</b><br>{x_label}: %{{x}}<br>Click for details<extra></extra>'
    )

    # Dynamic height based on number of items
    calculated_height = max(height, len(labels) * 25 + 100) if labels else height

    apply_chart_template(fig, title, x_label, y_label, calculated_height)
    fig.update_layout(clickmode='event+select')

    return fig


def create_pie_chart(data: Dict[str, Union[int, float]], title: str,
                    color_sequence: List[str] = None) -> go.Figure:
    """
    Create a pie chart.

    Args:
        data: Dictionary of label -> value
        title: Chart title
        color_sequence: Custom color sequence

    Returns:
        Plotly Figure
    """
    if not data:
        return go.Figure()

    labels = list(data.keys())
    values = list(data.values())

    # Truncate labels for better display
    truncated_labels = [
        label[:15] + "..." if len(label) > 15 else label
        for label in labels
    ]

    fig = px.pie(
        values=values,
        names=truncated_labels,
        title=title,
        color_discrete_sequence=color_sequence
    )

    fig.update_traces(
        textposition='auto',
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
        hoverinfo='label+percent+value'
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            font=dict(size=9)
        ),
        height=350
    )

    return fig


def scale_treemap_values(values: List[Union[int, float]],
                        min_size: int = 20, scale_factor: float = 10) -> List[float]:
    """
    Apply scaling to treemap values for better visibility.

    Args:
        values: Original values
        min_size: Minimum size guarantee
        scale_factor: Scaling factor for square root scaling

    Returns:
        Scaled values
    """
    scaled_values = []
    for value in values:
        if value == 0:
            scaled_values.append(min_size)
        else:
            # Square root scaling with minimum size
            scaled_value = max(np.sqrt(value) * scale_factor, min_size)
            scaled_values.append(scaled_value)

    return scaled_values


def create_correlation_heatmap(correlation_matrix: np.ndarray,
                              alpha_ids: List[str], title: str = "Correlation Heatmap") -> go.Figure:
    """
    Create a correlation heatmap.

    Args:
        correlation_matrix: Correlation matrix
        alpha_ids: Alpha IDs for labels
        title: Chart title

    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=alpha_ids,
        y=alpha_ids,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        colorbar=dict(title="Correlation"),
        hovertemplate="Alpha X: %{x}<br>Alpha Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Alpha ID", tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(title="Alpha ID", tickfont=dict(size=8)),
        height=800,
        width=900,
        clickmode='event+select',
        hovermode='closest'
    )

    return fig


def add_performance_overlay(fig: go.Figure, performance_values: List[float],
                           performance_metric: str = "performance") -> None:
    """
    Add performance color overlay to scatter plot.

    Args:
        fig: Plotly figure to modify
        performance_values: Performance values for color mapping
        performance_metric: Name of performance metric
    """
    if not performance_values or all(v is None for v in performance_values):
        return

    # Filter out None values for color scale
    valid_values = [v for v in performance_values if v is not None]
    if not valid_values:
        return

    # Update traces with performance coloring
    for trace in fig.data:
        if hasattr(trace, 'marker'):
            trace.marker.color = performance_values
            trace.marker.colorscale = 'Viridis'
            trace.marker.colorbar = dict(title=performance_metric.title())
            trace.marker.showscale = True