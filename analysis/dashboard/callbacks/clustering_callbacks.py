"""
Clustering Callbacks

Clustering visualization and coordinate update callbacks.
Extracted from visualization_server.py lines 724-855 with exact logic preservation.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic
from ..services import get_clustering_service


def register_clustering_callbacks(app: dash.Dash):
    """
    Register clustering-related callbacks.

    CRITICAL: These callbacks handle region selection, coordinate updates, and method switching.
    Maintains exact compatibility with original visualization_server.py.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        [Output('current-mds-data', 'data'),
         Output('mds-data-simple', 'data'),
         Output('mds-data-euclidean', 'data'),
         Output('mds-data-angular', 'data'),
         Output('current-tsne-data', 'data'),
         Output('current-umap-data', 'data'),
         Output('current-pca-data', 'data'),
         Output('current-pca-info', 'data'),
         Output('current-metadata', 'data'),
         Output('heatmap-data-simple', 'data'),
         Output('heatmap-data-euclidean', 'data'),
         Output('heatmap-data-angular', 'data'),
         Output('tsne-cluster-profiles', 'data'),
         Output('umap-cluster-profiles', 'data'),
         Output('pca-cluster-profiles', 'data'),
         Output('main-cluster-profiles', 'data'),
         Output('clustering-region-info', 'children')],
        Input('clustering-region-selector', 'value'),
        State('all-region-data', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_clustering_data_for_region(selected_region, all_region_data):
        """
        Update clustering data when region is selected.

        EXACT COPY from visualization_server.py lines 725-797
        Preserves all data transformation logic and coordinate handling.
        """
        if not selected_region or not all_region_data or selected_region not in all_region_data:
            return [], [], [], [], [], [], [], {}, {}, {}, {}, {}, {}, {}, {}, {}, "No data available"

        region_data = all_region_data[selected_region]

        # Load pre-calculated MDS data for all distance metrics
        mds_simple = pd.DataFrame.from_dict(region_data.get('mds_coords_simple', {}), orient='index')
        mds_euclidean = pd.DataFrame.from_dict(region_data.get('mds_coords_euclidean', region_data.get('mds_coords', {})), orient='index')
        mds_angular = pd.DataFrame.from_dict(region_data.get('mds_coords_angular', {}), orient='index')

        # Convert coordinate dictionaries to DataFrames
        tsne_coords = pd.DataFrame.from_dict(region_data.get('tsne_coords', {}), orient='index')
        umap_coords = pd.DataFrame.from_dict(region_data.get('umap_coords', {}), orient='index')
        pca_coords = pd.DataFrame.from_dict(region_data.get('pca_coords', {}), orient='index')
        pca_info = region_data.get('pca_info', {})  # Load PCA analysis information
        metadata = pd.DataFrame.from_dict(region_data.get('alpha_metadata', {}), orient='index')

        # Convert to dict format for storage
        mds_data_simple = mds_simple.reset_index().to_dict('records') if not mds_simple.empty else []
        mds_data_euclidean = mds_euclidean.reset_index().to_dict('records') if not mds_euclidean.empty else []
        mds_data_angular = mds_angular.reset_index().to_dict('records') if not mds_angular.empty else []

        # Default MDS is euclidean for backward compatibility
        mds_data_current = mds_data_euclidean

        tsne_data = tsne_coords.reset_index().to_dict('records') if not tsne_coords.empty else []
        umap_data = umap_coords.reset_index().to_dict('records') if not umap_coords.empty else []
        pca_data = pca_coords.reset_index().to_dict('records') if not pca_coords.empty else []
        metadata_data = metadata.reset_index().to_dict('records') if not metadata.empty else []

        # Load pre-calculated heatmap data for all distance metrics
        heatmap_simple = region_data.get('heatmap_data_simple', {})
        heatmap_euclidean = region_data.get('heatmap_data_euclidean', {})
        heatmap_angular = region_data.get('heatmap_data_angular', {})

        # Load cluster profiles for interpretability
        tsne_profiles = region_data.get('tsne_cluster_profiles', {})
        umap_profiles = region_data.get('umap_cluster_profiles', {})
        pca_profiles = region_data.get('pca_cluster_profiles', {})
        main_profiles = region_data.get('main_cluster_profiles', {})

        # Create info text
        alpha_count = region_data.get('alpha_count', 0)
        timestamp = region_data.get('timestamp', 'Unknown')
        info_text = f"{alpha_count} alphas | Generated: {timestamp}"

        return (mds_data_current, mds_data_simple, mds_data_euclidean, mds_data_angular,
                tsne_data, umap_data, pca_data, pca_info, metadata_data,
                heatmap_simple, heatmap_euclidean, heatmap_angular,
                tsne_profiles, umap_profiles, pca_profiles, main_profiles,
                info_text)

    @callback_wrapper.safe_callback(
        Output('current-mds-data', 'data', allow_duplicate=True),
        [Input('distance-metric', 'value'),
         State('mds-data-simple', 'data'),
         State('mds-data-euclidean', 'data'),
         State('mds-data-angular', 'data')],
        prevent_initial_call=True  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_mds_with_distance_metric(distance_metric, mds_simple, mds_euclidean, mds_angular):
        """
        Update MDS data based on selected distance metric.

        EXACT COPY from visualization_server.py lines 835-855
        Preserves all distance metric selection logic.
        """
        if not distance_metric:
            return []

        # Select the appropriate pre-calculated MDS data
        if distance_metric == 'simple':
            return mds_simple
        elif distance_metric == 'angular':
            return mds_angular
        else:  # euclidean is default
            return mds_euclidean


def register_clustering_ui_callbacks(app: dash.Dash):
    """
    Register clustering UI control callbacks.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        Output('distance-metric-container', 'style'),
        Input('method-selector', 'value'),
        prevent_initial_call=True  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def toggle_distance_metric_visibility(method):
        """
        Show distance metric selector only for MDS visualization.

        EXACT COPY from visualization_server.py lines 809-820
        """
        # Check if advanced clustering is available (from config)
        try:
            from analysis.clustering.clustering_algorithms import calculate_rolling_correlation_matrix
            ADVANCED_CLUSTERING_AVAILABLE = True
        except ImportError:
            ADVANCED_CLUSTERING_AVAILABLE = False

        if ADVANCED_CLUSTERING_AVAILABLE and method == 'mds':
            return {'margin-bottom': '15px', 'display': 'block'}
        else:
            return {'display': 'none'}

    @callback_wrapper.safe_callback(
        Output('pca-loadings-container', 'style'),
        Input('method-selector', 'value'),
        prevent_initial_call=True  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def toggle_pca_loadings_visibility(method):
        """
        Show PCA loadings heatmap only when PCA method is selected.

        EXACT COPY from visualization_server.py lines 822-833
        """
        if method == 'pca':
            return {'display': 'block'}
        else:
            return {'display': 'none'}


# Export for easy registration
__all__ = ['register_clustering_callbacks', 'register_clustering_ui_callbacks']