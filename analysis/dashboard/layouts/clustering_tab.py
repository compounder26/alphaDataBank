"""
Clustering Tab Layout

Layout structure for the clustering visualization tab.
Extracted from visualization_server.py with preserved functionality.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..components import (
    create_info_card,
    create_clustering_region_selector,
    create_method_selector,
    create_distance_metric_selector,
    create_highlighting_filters,
    create_display_options_card,
    create_clustering_info_card,
    create_cluster_statistics_panel,
    create_alpha_details_panel,
    create_method_explanation_card,
    create_main_clustering_plot,
    create_pca_loadings_plot
)


def create_clustering_tab_content(advanced_clustering_available: bool = True) -> dbc.Row:
    """
    Create the clustering tab content layout.

    Args:
        advanced_clustering_available: Whether advanced clustering is available

    Returns:
        Row component with clustering tab structure
    """
    return dbc.Row([
        # Left sidebar with controls
        dbc.Col([
            # Region Selector Card
            create_info_card(
                title="Region Selection",
                content=create_clustering_region_selector()
            ),

            # Visualization Method Card
            create_visualization_method_card(advanced_clustering_available),

            # Clustering Information Card
            create_clustering_info_card(),

            # Alpha Highlighting Card
            create_highlighting_filters(),

            # Display Options Card
            create_display_options_card(),

            # Cluster Statistics Panel
            create_cluster_statistics_panel(),

            # Alpha Details Panel
            create_alpha_details_panel()
        ], width=3),

        # Main visualization area
        dbc.Col([
            # Method Explanation Card
            create_method_explanation_card(),

            # Main clustering plot
            create_main_clustering_plot(),

            # PCA Feature Loadings Heatmap (shown only when PCA is selected)
            create_pca_loadings_plot()
        ], width=9)
    ])


def create_visualization_method_card(advanced_clustering_available: bool = True) -> dbc.Card:
    """
    Create visualization method selection card.

    Args:
        advanced_clustering_available: Whether advanced clustering methods are available

    Returns:
        Card component with method selector
    """
    method_options = [
        {'label': 'MDS on Correlation Matrix', 'value': 'mds'},
        {'label': 't-SNE on Performance Features', 'value': 'tsne'},
        {'label': 'UMAP on Performance Features', 'value': 'umap'},
        {'label': 'PCA on Performance Features', 'value': 'pca'},
        {'label': 'Correlation Heatmap', 'value': 'heatmap', 'disabled': not advanced_clustering_available},
    ]

    content = [
        create_method_selector(
            methods=method_options,
            selected_value='mds'
        )
    ]

    # Add distance metric selector for advanced clustering
    if advanced_clustering_available:
        content.extend([
            html.Hr(),
            create_distance_metric_selector()
        ])

    return create_info_card(
        title="Visualization Method",
        content=content,
        className="mb-4"
    )


def create_clustering_explanation_text():
    """
    Create clustering explanation text.

    Returns:
        List of paragraph components with explanation
    """
    return [
        html.P([
            "🎨 Points are colored by ", html.Strong("automatically detected groups"), " of similar trading strategies. ",
            "Each color represents alphas with similar risk/return patterns."
        ], className="mb-2"),
        html.Small([
            html.Strong("HDBSCAN algorithm"), " automatically determines the optimal number of clusters based on data density."
        ], className="text-muted")
    ]


def get_clustering_method_options(advanced_available=True):
    """
    Get clustering method options based on availability.

    Args:
        advanced_available: Whether advanced methods are available

    Returns:
        List of method option dictionaries
    """
    options = [
        {'label': 'MDS on Correlation Matrix', 'value': 'mds'},
        {'label': 't-SNE on Performance Features', 'value': 'tsne'},
        {'label': 'UMAP on Performance Features', 'value': 'umap'},
        {'label': 'PCA on Performance Features', 'value': 'pca'}
    ]

    if advanced_available:
        options.append({'label': 'Correlation Heatmap', 'value': 'heatmap'})
    else:
        options.append({'label': 'Correlation Heatmap', 'value': 'heatmap', 'disabled': True})

    return options


def get_distance_metric_options():
    """
    Get distance metric options.

    Returns:
        List of distance metric option dictionaries
    """
    return [
        {'label': 'Simple (1 - corr)', 'value': 'simple'},
        {'label': 'Euclidean √(2(1-corr))', 'value': 'euclidean'},
        {'label': 'Angular √(0.5(1-corr))', 'value': 'angular'},
    ]


def should_show_distance_metric_selector(method, advanced_available):
    """
    Determine if distance metric selector should be shown.

    Args:
        method: Selected clustering method
        advanced_available: Whether advanced clustering is available

    Returns:
        Style dictionary for container visibility
    """
    if advanced_available and method == 'mds':
        return {'margin-bottom': '15px', 'display': 'block'}
    else:
        return {'display': 'none'}


def should_show_pca_loadings(method):
    """
    Determine if PCA loadings should be shown.

    Args:
        method: Selected clustering method

    Returns:
        Style dictionary for container visibility
    """
    if method == 'pca':
        return {'display': 'block'}
    else:
        return {'display': 'none'}