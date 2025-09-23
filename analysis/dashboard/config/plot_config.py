"""
Plot Configuration

Configuration for Plotly charts and visualizations.
"""

import plotly.graph_objects as go

# Default plotting template
DEFAULT_TEMPLATE = 'plotly_white'

# Chart dimensions
CHART_DIMENSIONS = {
    'clustering_plot_height': '85vh',
    'pca_loadings_height': '400px',
    'modal_chart_height': '300px',
    'treemap_height': '750px',
    'operators_chart_height': 600,
    'datafields_chart_height': 500
}

# Color schemes
COLOR_SCHEMES = {
    'cluster_colors': [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#ffff33', '#a65628', '#f781bf',
        '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
        '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'
    ],
    'outlier_color': '#808080',
    'single_color': '#377eb8',
    'operator_highlight': 'green',
    'datafield_highlight': 'darkgreen',
    'selected_alpha': 'red'
}

# PCA feature category colors
PCA_FEATURE_COLORS = {
    'spiked': '#1f77b4',      # Blue - Factor models
    'multiscale': '#2ca02c',  # Green - Temporal patterns
    'risk': '#d62728',        # Red - Risk metrics
    'metadata': '#ff7f0e',    # Orange - Metadata
    'regime': '#9467bd'       # Purple - Regime features
}

# Chart styling defaults
CHART_STYLE_DEFAULTS = {
    'marker_size': 12,
    'marker_line_width': 1,
    'marker_line_color': 'white',
    'hover_template_suffix': '<extra></extra>',
    'font_size': 12,
    'title_font_size': 16,
    'axis_title_font_size': 14
}

# Heatmap settings
HEATMAP_SETTINGS = {
    'colorscale': 'RdBu',
    'zmid': 0,
    'text_font_size': 8,
    'show_text': True,
    'hover_format': '.3f'
}

# Treemap settings
TREEMAP_SETTINGS = {
    'text_font_size': 18,
    'text_font_color': 'white',
    'text_font_family': 'Arial Black',
    'border_width': 1,
    'border_color': 'white',
    'min_text_size': 8
}

# Animation settings
ANIMATION_SETTINGS = {
    'transition_duration': 300,
    'transition_easing': 'cubic-in-out',
    'frame_duration': 500
}

def get_plotly_layout_defaults():
    """Get default layout settings for Plotly charts."""
    return {
        'template': DEFAULT_TEMPLATE,
        'hovermode': 'closest',
        'clickmode': 'event+select',
        'font': {'size': CHART_STYLE_DEFAULTS['font_size']},
        'title': {'font': {'size': CHART_STYLE_DEFAULTS['title_font_size']}},
        'xaxis': {'title': {'font': {'size': CHART_STYLE_DEFAULTS['axis_title_font_size']}}},
        'yaxis': {'title': {'font': {'size': CHART_STYLE_DEFAULTS['axis_title_font_size']}}}
    }

def get_scatter_trace_defaults():
    """Get default settings for scatter plot traces."""
    return {
        'mode': 'markers',
        'marker': {
            'size': CHART_STYLE_DEFAULTS['marker_size'],
            'line': {
                'width': CHART_STYLE_DEFAULTS['marker_line_width'],
                'color': CHART_STYLE_DEFAULTS['marker_line_color']
            }
        },
        'hovertemplate': '%{hovertext}<br>Click to view on WorldQuant Brain<extra></extra>'
    }

def get_bar_chart_defaults():
    """Get default settings for bar charts."""
    return {
        'orientation': 'h',
        'hovertemplate': '<b>%{y}</b><br>Used in %{x} alphas<br>Click for details<extra></extra>',
        'marker_color': 'steelblue'
    }

def get_heatmap_defaults():
    """Get default settings for heatmaps."""
    return {
        'colorscale': HEATMAP_SETTINGS['colorscale'],
        'zmid': HEATMAP_SETTINGS['zmid'],
        'texttemplate': "%{text}",
        'textfont': {"size": HEATMAP_SETTINGS['text_font_size']},
        'hovertemplate': "Alpha X: %{x}<br>Alpha Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    }