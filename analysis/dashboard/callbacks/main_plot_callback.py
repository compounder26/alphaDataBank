"""
Main Clustering Plot Callback

The critical main clustering plot update callback.
Extracted from visualization_server.py lines 2804-3551 with logic preservation.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic
from ..services import get_chart_service


def register_main_plot_callback(app: dash.Dash):
    """
    Register the main clustering plot callback.

    CRITICAL: This is the core visualization callback that handles all clustering methods.
    Maintains exact compatibility with original visualization_server.py.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    # Check if advanced clustering is available
    try:
        from analysis.clustering.clustering_algorithms import calculate_rolling_correlation_matrix
        ADVANCED_CLUSTERING_AVAILABLE = True
    except ImportError:
        ADVANCED_CLUSTERING_AVAILABLE = False

    @callback_wrapper.safe_callback(
        Output('clustering-plot', 'figure'),
        [Input('method-selector', 'value'),
         Input('current-mds-data', 'data'),
         Input('current-tsne-data', 'data'),
         Input('current-umap-data', 'data'),
         Input('current-pca-data', 'data'),
         Input('selected-alpha', 'data'),
         Input('all-region-data', 'data'),
         Input('selected-clustering-region', 'data'),
         Input('operator-highlighted-alphas', 'data'),
         Input('datafield-highlighted-alphas', 'data'),
         Input('cluster-color-mode', 'value')],
        [State('current-pca-info', 'data'),
         State('distance-metric', 'value'),
         State('heatmap-data-simple', 'data'),
         State('heatmap-data-euclidean', 'data'),
         State('heatmap-data-angular', 'data')],
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_plot(method, mds_data, tsne_data, umap_data, pca_data, selected_alpha,
                   all_region_data, selected_region, operator_alphas, datafield_alphas, color_mode,
                   pca_info, distance_metric, heatmap_simple, heatmap_euclidean, heatmap_angular):
        """
        Update main clustering plot.

        SIMPLIFIED from visualization_server.py lines 2824-3551
        Uses chart service for complex logic while preserving exact behavior.
        """
        # Handle distance metric availability
        if not ADVANCED_CLUSTERING_AVAILABLE:
            distance_metric = 'euclidean'  # Default fallback
        # Handle advanced clustering methods using pre-calculated data
        if method == 'heatmap' and ADVANCED_CLUSTERING_AVAILABLE:
            # Heatmap always uses the same correlation matrix
            data = heatmap_euclidean  # They're all the same, just use one

            # Create heatmap from pre-calculated data
            if data and 'correlation_matrix' in data and 'alpha_ids' in data:
                fig = go.Figure(data=go.Heatmap(
                    z=data['correlation_matrix'],
                    x=data['alpha_ids'],
                    y=data['alpha_ids'],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(data['correlation_matrix'], 2),
                    texttemplate="%{text}",
                    textfont={"size": 8},
                    colorbar=dict(title="Correlation"),
                    hovertemplate="Alpha X: %{x}<br>Alpha Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
                ))

                fig.update_layout(
                    title="Correlation Heatmap",
                    xaxis=dict(title="Alpha ID", tickangle=45, tickfont=dict(size=8)),
                    yaxis=dict(title="Alpha ID", tickfont=dict(size=8)),
                    template='plotly_white',
                    height=800,
                    width=900,
                    clickmode='event+select',
                    hovermode='closest'
                )
                return fig

            # If no data available, show message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No pre-calculated {method.upper()} data available.<br>Please regenerate clustering data.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(template='plotly_white')
            return fig

        # Select the appropriate data based on the method
        data_mapping = {
            'mds': mds_data,
            'tsne': tsne_data,
            'umap': umap_data,
            'pca': pca_data
        }

        plot_data_raw = data_mapping.get(method, [])
        if not plot_data_raw:
            return _create_empty_plot_figure(method)

        plot_data = pd.DataFrame(plot_data_raw)

        # Check if plot_data is valid
        if plot_data.empty or 'x' not in plot_data.columns or 'y' not in plot_data.columns:
            return _create_empty_plot_figure(method)

        # Use chart service for plot creation
        chart_service = get_chart_service()

        return chart_service.create_clustering_plot(
            method=method,
            plot_data=plot_data,
            pca_info=pca_info,
            operator_alphas=operator_alphas,
            datafield_alphas=datafield_alphas,
            selected_alpha=selected_alpha,
            color_mode=color_mode,
            distance_metric=distance_metric
        )

    @callback_wrapper.safe_callback(
        [Output('alpha-details', 'children'),
         Output('selected-alpha', 'data')],
        Input('clustering-plot', 'clickData'),
        [State('current-metadata', 'data'),
         State('method-selector', 'value')],
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def display_alpha_details(clickData, metadata_data, method):
        """
        Display alpha details when a point is clicked.

        SIMPLIFIED from visualization_server.py lines 3801-4014
        """
        if not clickData:
            return "Click on a point to see alpha details.", None

        # Get the clicked point data
        point = clickData['points'][0]

        # Handle heatmap clicks (dual alpha display)
        if method == 'heatmap':
            return _handle_heatmap_click(point)

        # Handle other visualization methods (single alpha display)
        alpha_id = point.get('customdata') or point.get('text') or 'unknown'

        # Create WorldQuant Brain URL
        wq_url = f"https://platform.worldquantbrain.com/alpha/{alpha_id}"

        # Use modal content creation from components
        from ..components import create_alpha_details_modal_content
        from ..services import get_alpha_details_for_clustering

        # Fetch detailed alpha information
        try:
            alpha_details_dict = get_alpha_details_for_clustering([alpha_id])
            alpha_info = alpha_details_dict.get(alpha_id, {})

            if alpha_info:
                details = create_alpha_details_modal_content(alpha_id, alpha_info, wq_url)
            else:
                # Fallback when no detailed info is available
                details = [
                    html.H4(f"Alpha: {alpha_id}", className="text-primary mb-3"),
                    html.A(
                        dbc.Button("View on WorldQuant Brain", color="primary", size="sm"),
                        href=wq_url, target="_blank"
                    ),
                    html.P("Detailed information not available in cache.", className="text-muted")
                ]
        except Exception as e:
            print(f"Error fetching alpha details: {e}")
            details = [
                html.H4(f"Alpha: {alpha_id}", className="text-primary mb-3"),
                html.A(
                    dbc.Button("View on WorldQuant Brain", color="primary", size="sm"),
                    href=wq_url, target="_blank"
                ),
                html.P(f"Error loading details: {str(e)}", className="text-danger small")
            ]

        return details, {'index': alpha_id}


def _create_empty_plot_figure(method):
    """Create empty plot figure with message."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"No data available for {method.upper()} clustering.<br>Please wait for data to load or try a different method.",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        title=f"Select a clustering method",
        template='plotly_white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig


def _handle_heatmap_click(point):
    """Handle heatmap click for dual alpha display."""
    from ..components import create_dual_alpha_modal_content
    from ..services import get_alpha_details_for_clustering

    # For heatmap, x and y contain the alpha IDs
    alpha_x = point.get('x')
    alpha_y = point.get('y')
    correlation = point.get('z', 0)

    if alpha_x and alpha_y:
        # Fetch details for both alphas
        try:
            alpha_details_dict = get_alpha_details_for_clustering([alpha_x, alpha_y])
            alpha_x_info = alpha_details_dict.get(alpha_x, {})
            alpha_y_info = alpha_details_dict.get(alpha_y, {})

            # Create dual alpha display
            details = create_dual_alpha_modal_content(alpha_x, alpha_y, correlation, alpha_x_info, alpha_y_info)

            return details, {'x': alpha_x, 'y': alpha_y, 'correlation': correlation}

        except Exception as e:
            print(f"Error fetching dual alpha details: {e}")
            return html.P(f"Error loading details: {str(e)}", className="text-danger"), None
    else:
        return "Unable to extract alpha IDs from heatmap click.", None


# Export for easy registration
__all__ = ['register_main_plot_callback']