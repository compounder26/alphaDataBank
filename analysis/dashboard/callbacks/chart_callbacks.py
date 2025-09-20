"""
Chart Interaction Callbacks

Main clustering plot and chart interaction callbacks.
Extracted from visualization_server.py lines 2515-3640 with exact logic preservation.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic
from ..services import get_chart_service, get_clustering_service, get_alpha_details_for_clustering


def register_chart_interaction_callbacks(app: dash.Dash):
    """
    Register chart interaction callbacks.

    CRITICAL: These callbacks handle the main clustering plot and method explanations.
    Maintains exact compatibility with original visualization_server.py.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        Output('method-explanation', 'children'),
        Input('method-selector', 'value'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_method_explanation(method):
        """
        Provide mathematical explanation for the selected clustering method.

        EXACT COPY from visualization_server.py lines 2516-2629
        Preserves all method explanations and mathematical content.
        """
        explanations = {
            'mds': [
                html.H5("📊 Multidimensional Scaling (MDS) on Correlation Matrix", className="text-primary"),
                html.P([
                    html.B("Input Data: "),
                    html.Span("Uses correlation matrix of alpha PnL returns (not performance features like Sharpe ratio). ", className="text-info"),
                    "Converts correlations to distances using d_ij = √(2(1 - ρ_ij)) where ρ is the correlation coefficient."
                ]),
                html.P([
                    html.B("What it shows: "),
                    "Maps alphas to 2D space where distance represents correlation dissimilarity. ",
                    "Unlike other methods, MDS directly measures alpha overlap rather than performance similarity."
                ]),
                html.P([
                    html.B("Why useful: "),
                    "Alphas close together are highly correlated (avoid combining), while distant alphas are uncorrelated (good for diversification). ",
                    "The Euclidean distance in the plot directly corresponds to portfolio diversification benefit."
                ]),
                html.P([
                    html.B("Mathematics: "),
                    "Minimizes stress function S = Σ_ij (d_ij - δ_ij)² where d_ij is the 2D distance and δ_ij is the original correlation distance. "
                ], className="small text-muted"),
            ],

            'tsne': [
                html.H5("🔬 t-SNE on Performance Features", className="text-primary"),
                html.P([
                    html.B("What it shows: "),
                    "Non-linear projection emphasizing local structure - alphas with similar risk-return profiles cluster together. ",
                    "Uses Sharpe ratio, volatility, drawdown, skewness, and kurtosis as features."
                ]),
                html.P([
                    html.B("Why useful: "),
                    "Reveals natural groupings of strategies with similar behavior patterns. ",
                    "Tight clusters indicate redundant strategies; isolated points represent unique approaches worth including."
                ]),
                html.P([
                    html.B("Mathematics: "),
                    "Minimizes KL divergence between probability distributions: KL(P||Q) = Σ_ij p_ij log(p_ij/q_ij). ",
                    "Uses Student's t-distribution in embedding space for heavy-tailed flexibility."
                ], className="small text-muted"),
            ],

            'umap': [
                html.H5("🗺️ UMAP on Performance Features", className="text-primary"),
                html.P([
                    html.B("What it shows: "),
                    "Preserves both local and global structure - maintains meaningful distances between clusters. ",
                    "Superior to t-SNE for understanding relationships between different strategy groups."
                ]),
                html.P([
                    html.B("Why useful: "),
                    "Inter-cluster distances are meaningful - can identify which strategy groups are most different. ",
                    "Useful for hierarchical portfolio construction across multiple strategy types."
                ]),
                html.P([
                    html.B("Mathematics: "),
                    "Constructs fuzzy topological representation using k-NN graph, then optimizes layout via cross-entropy. ",
                    "Balances local structure preservation (n_neighbors) with global structure (min_dist parameter)."
                ], className="small text-muted"),
            ],

            'pca': [
                html.H5("📐 PCA on Performance Features", className="text-primary"),
                html.P([
                    html.B("What it shows: "),
                    "Linear projection onto principal components - PC1 typically captures risk-return trade-off, PC2 captures style factors. ",
                    "Preserves global structure and relative distances between all alphas."
                ]),
                html.Div(id="pca-dynamic-info", className="mb-3"),  # Dynamic PCA information will be inserted here
                html.P([
                    html.B("Why useful: "),
                    "Interpretable axes - can understand what drives separation between strategies. ",
                    "Linear nature means portfolio combinations behave predictably in this space."
                ]),
                html.P([
                    html.B("Mathematics: "),
                    "Eigendecomposition of covariance matrix: Σ = VΛV^T where columns of V are principal components. ",
                    "Projects data onto eigenvectors with largest eigenvalues to maximize variance."
                ], className="small text-muted"),
            ],

            'heatmap': [
                html.H5("Correlation Heatmap", className="text-primary"),
                html.P([
                    html.B("What it shows: "),
                    "Full N×N correlation matrix with hierarchical clustering reordering. ",
                    "Red = positive correlation (redundant), Blue = negative (natural hedges), White = uncorrelated."
                ]),
                html.P([
                    html.B("Why useful: "),
                    "Direct view of all pairwise relationships - identify blocks of similar strategies. ",
                    "Diagonal blocks reveal strategy families; off-diagonal patterns show cross-dependencies."
                ]),
                html.P([
                    html.B("Mathematics: "),
                    "Pearson correlation of percentage returns: ρ = Cov(r_i, r_j) / (σ_i × σ_j). ",
                ], className="small text-muted"),
            ],
        }

        explanation = explanations.get(method, [html.P("Select a visualization method to see its explanation.")])

        return html.Div([
            *explanation
        ])

    @callback_wrapper.safe_callback(
        Output('cluster-statistics', 'children'),
        [Input('current-pca-data', 'data'),
         Input('current-umap-data', 'data'),
         Input('current-tsne-data', 'data'),
         Input('current-mds-data', 'data'),
         Input('pca-cluster-profiles', 'data'),
         Input('umap-cluster-profiles', 'data'),
         Input('tsne-cluster-profiles', 'data'),
         Input('main-cluster-profiles', 'data'),
         Input('method-selector', 'value')],
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_cluster_statistics(pca_data, umap_data, tsne_data, mds_data,
                                 pca_profiles, umap_profiles, tsne_profiles, main_profiles, method):
        """
        Update cluster statistics display.

        EXACT COPY from visualization_server.py lines 2631-2802
        Preserves all cluster analysis and statistics generation logic.
        """
        # Handle heatmap method (no cluster statistics)
        if method == 'heatmap':
            return html.P("Cluster statistics not available for correlation heatmap.", className="text-info")

        # Get the appropriate data based on selected method
        data_mapping = {
            'pca': pca_data,
            'umap': umap_data,
            'tsne': tsne_data,
            'mds': mds_data
        }

        data = data_mapping.get(method, {})
        if not data:
            return html.P("No clustering data available.", className="text-warning")

        # Convert list data (records format) to DataFrame to extract cluster information
        if isinstance(data, list):
            if not data:
                return html.P("No clustering data available.", className="text-warning")
            try:
                df = pd.DataFrame(data)
                if 'cluster' not in df.columns:
                    return html.P("No cluster information available for this method.", className="text-warning")

                # Extract cluster information from DataFrame
                cluster_info = {}
                has_clusters = False

                for _, row in df.iterrows():
                    alpha_id = row.get('index', 'Unknown')
                    cluster_val = row.get('cluster')

                    if cluster_val is not None and not (isinstance(cluster_val, float) and pd.isna(cluster_val)):
                        has_clusters = True
                        if cluster_val >= 0:
                            cluster_name = f"Cluster {int(cluster_val)}"
                        else:
                            cluster_name = "Outliers"

                        if cluster_name not in cluster_info:
                            cluster_info[cluster_name] = []
                        cluster_info[cluster_name].append(alpha_id)

            except Exception as e:
                return html.P(f"Error processing cluster data: {str(e)}", className="text-danger")

        # Handle dictionary format (legacy support)
        elif isinstance(data, dict):
            cluster_info = {}
            has_clusters = False

            for alpha_id, alpha_data in data.items():
                if isinstance(alpha_data, dict) and 'cluster' in alpha_data:
                    cluster_val = alpha_data['cluster']
                    if cluster_val is not None and not (isinstance(cluster_val, float) and pd.isna(cluster_val)):
                        has_clusters = True
                        if cluster_val >= 0:
                            cluster_name = f"Cluster {int(cluster_val)}"
                        else:
                            cluster_name = "Outliers"

                        if cluster_name not in cluster_info:
                            cluster_info[cluster_name] = []
                        cluster_info[cluster_name].append(alpha_id)
        else:
            return html.P("Invalid clustering data format.", className="text-warning")

        if not has_clusters or not cluster_info:
            return html.P("No cluster information available.", className="text-warning")

        # Calculate statistics
        total_alphas = sum(len(alphas) for alphas in cluster_info.values())
        n_clusters = len([k for k in cluster_info.keys() if k != "Outliers"])
        n_outliers = len(cluster_info.get("Outliers", []))

        # Create statistics display
        stats_components = [
            html.H6("Cluster Summary", className="mb-2"),
            html.P([
                html.Strong("Total Alphas: "), f"{total_alphas}",
                html.Br(),
                html.Strong("Clusters Found: "), f"{n_clusters}",
                html.Br(),
                html.Strong("Outliers: "), f"{n_outliers} ({n_outliers/total_alphas*100:.1f}%)" if total_alphas > 0 else "0"
            ], className="mb-3")
        ]

        # Add cluster breakdown with color scheme
        if cluster_info:
            stats_components.append(html.H6("Cluster Breakdown", className="mb-2"))

            cluster_items = []
            # Use the same color scheme as the plot - enhanced with more distinct colors
            colors = [
                '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                '#ff7f00', '#ffff33', '#a65628', '#f781bf',
                '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
                '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'
            ]

            # Custom sorting function for proper cluster ordering (0, 1, 2... then Outliers)
            def cluster_sort_key(item):
                cluster_name, alphas = item
                if cluster_name == "Outliers":
                    return (1, 0)  # Sort Outliers last
                else:
                    # Extract numeric part from "Cluster X"
                    cluster_num = int(cluster_name.split()[-1])
                    return (0, cluster_num)  # Sort clusters numerically

            cluster_idx = 0
            for cluster_name, alphas in sorted(cluster_info.items(), key=cluster_sort_key):
                if cluster_name == "Outliers":
                    color = "#808080"  # Gray for outliers
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

            stats_components.append(
                dbc.ListGroup(cluster_items, flush=True)
            )

        return html.Div(stats_components)

    @callback_wrapper.safe_callback(
        Output('pca-dynamic-info', 'children'),
        Input('method-selector', 'value'),
        State('current-pca-info', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_pca_info(method, pca_info):
        """
        Update PCA dynamic information.

        EXACT COPY from visualization_server.py lines 3554-3640
        Preserves all PCA information display logic.
        """
        if method != 'pca' or not pca_info:
            return []

        content = []

        # Add variance explained information
        if 'variance_explained' in pca_info:
            var_exp = pca_info['variance_explained']
            pc1_var = var_exp.get('pc1', 0) * 100
            pc2_var = var_exp.get('pc2', 0) * 100
            total_var = var_exp.get('total_2d', 0) * 100

            content.append(html.P([
                html.B("Variance Explained: "),
                f"PC1: {pc1_var:.1f}%, PC2: {pc2_var:.1f}% (2D captures {total_var:.1f}% of variance)"
            ], className="text-info"))

        # Enhanced interpretation with category information
        if 'interpretation' in pca_info:
            interp = pca_info['interpretation']
            if interp.get('pc1') and interp['pc1'] != "Mixed factors":
                content.append(html.P([
                    html.B("PC1 Drivers: "),
                    interp['pc1']
                ], className="small text-success"))
            if interp.get('pc2') and interp['pc2'] != "Mixed factors":
                content.append(html.P([
                    html.B("PC2 Drivers: "),
                    interp['pc2']
                ], className="small text-success"))

        # Add feature contributions with color coding
        if 'top_features' in pca_info:
            top_features = pca_info['top_features']

            if top_features.get('pc1'):
                pc1_items = []
                for feat, contrib in top_features['pc1']:
                    # Determine category color
                    color = "#666"
                    if feat.startswith('spiked_'): color = "#1f77b4"
                    elif feat.startswith('multiscale_'): color = "#2ca02c"
                    elif feat.startswith('risk_'): color = "#d62728"
                    elif feat.startswith('metadata_'): color = "#ff7f0e"

                    clean_name = feat.replace('spiked_', '').replace('multiscale_', '').replace('risk_', '').replace('metadata_', '')
                    pc1_items.append(html.Span([
                        html.Span("●", style={"color": color}),
                        f" {clean_name} ({contrib:.2f})"
                    ], className="me-2"))

                content.append(html.P([
                    html.B("PC1 Top Contributors: "),
                    html.Div(pc1_items, style={"line-height": "1.6"})
                ], className="small"))

            if top_features.get('pc2'):
                pc2_items = []
                for feat, contrib in top_features['pc2']:
                    # Determine category color
                    color = "#666"
                    if feat.startswith('spiked_'): color = "#1f77b4"
                    elif feat.startswith('multiscale_'): color = "#2ca02c"
                    elif feat.startswith('risk_'): color = "#d62728"
                    elif feat.startswith('metadata_'): color = "#ff7f0e"

                    clean_name = feat.replace('spiked_', '').replace('multiscale_', '').replace('risk_', '').replace('metadata_', '')
                    pc2_items.append(html.Span([
                        html.Span("●", style={"color": color}),
                        f" {clean_name} ({contrib:.2f})"
                    ], className="me-2"))

                content.append(html.P([
                    html.B("PC2 Top Contributors: "),
                    html.Div(pc2_items, style={"line-height": "1.6"})
                ], className="small"))

        return content

    @callback_wrapper.safe_callback(
        Output('pca-loadings-heatmap', 'figure'),
        Input('method-selector', 'value'),
        State('current-pca-info', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_pca_loadings_heatmap(method, pca_info):
        """
        Update PCA loadings heatmap.

        EXACT COPY from visualization_server.py lines 3641-3799
        Preserves all PCA loadings visualization logic.
        """
        if method != 'pca' or not pca_info or 'loadings' not in pca_info:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No PCA loadings data available",
                showarrow=False,
                xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=16)
            )
            return fig

        # Use chart service for complex PCA loadings chart creation
        chart_service = get_chart_service()
        return chart_service.create_pca_loadings_heatmap(pca_info)


# Export for easy registration
__all__ = ['register_chart_interaction_callbacks']