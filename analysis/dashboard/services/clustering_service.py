"""
Clustering Service

Business logic for clustering operations, coordinate transformations, and data processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .data_service import load_all_region_data
from ..models import ClusteringData, CoordinateData
from ..utils import cached

class ClusteringService:
    """Service for handling clustering operations and coordinate data."""

    def __init__(self):
        """Initialize clustering service."""
        self._region_data_cache = {}

    @cached(ttl=600)  # Cache for 10 minutes
    def load_region_clustering_data(self, region: str = None) -> Dict[str, ClusteringData]:
        """
        Load clustering data for all regions or a specific region.

        Args:
            region: Specific region to load (if None, loads all)

        Returns:
            Dictionary mapping region names to ClusteringData
        """
        all_region_data = load_all_region_data()
        result = {}

        for region_name, raw_data in all_region_data.items():
            if region is None or region == region_name:
                try:
                    clustering_data = ClusteringData.from_legacy_data(region_name, raw_data)
                    result[region_name] = clustering_data
                except Exception as e:
                    print(f"Error converting clustering data for {region_name}: {e}")

        return result

    def get_coordinate_data_for_method(self, clustering_data: ClusteringData,
                                     method: str, distance_metric: str = 'euclidean') -> List[Dict[str, Any]]:
        """
        Get coordinate data for a specific clustering method.

        Args:
            clustering_data: Clustering data for the region
            method: Clustering method ('mds', 'tsne', 'umap', 'pca')
            distance_metric: Distance metric for MDS ('simple', 'euclidean', 'angular')

        Returns:
            List of coordinate records
        """
        if method == 'mds':
            return clustering_data.get_current_mds_data(distance_metric)
        elif method == 'tsne':
            return clustering_data.tsne_coords.to_records()
        elif method == 'umap':
            return clustering_data.umap_coords.to_records()
        elif method == 'pca':
            return clustering_data.pca_coords.to_records()
        else:
            return []

    def get_heatmap_data(self, clustering_data: ClusteringData,
                        distance_metric: str = 'euclidean') -> Dict[str, Any]:
        """
        Get heatmap data for correlation visualization.

        Args:
            clustering_data: Clustering data for the region
            distance_metric: Distance metric ('simple', 'euclidean', 'angular')

        Returns:
            Heatmap data dictionary
        """
        metric_map = {
            'simple': clustering_data.heatmap_simple,
            'euclidean': clustering_data.heatmap_euclidean,
            'angular': clustering_data.heatmap_angular
        }
        return metric_map.get(distance_metric, clustering_data.heatmap_euclidean)

    def get_cluster_statistics(self, coordinate_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate cluster statistics from coordinate data.

        Args:
            coordinate_data: Coordinate data in records format

        Returns:
            Cluster statistics dictionary
        """
        if not coordinate_data:
            return {'error': 'No data available'}

        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(coordinate_data)

            if 'cluster' not in df.columns:
                return {'error': 'No cluster information available'}

            # Extract cluster information
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

            if not has_clusters or not cluster_info:
                return {'error': 'No cluster information available'}

            # Calculate statistics
            total_alphas = sum(len(alphas) for alphas in cluster_info.values())
            n_clusters = len([k for k in cluster_info.keys() if k != "Outliers"])
            n_outliers = len(cluster_info.get("Outliers", []))

            return {
                'cluster_info': cluster_info,
                'total_alphas': total_alphas,
                'n_clusters': n_clusters,
                'n_outliers': n_outliers,
                'outlier_percentage': safe_divide(n_outliers, total_alphas) * 100 if total_alphas > 0 else 0
            }

        except Exception as e:
            return {'error': f'Error processing cluster data: {str(e)}'}

    def get_method_explanation(self, method: str) -> Dict[str, Any]:
        """
        Get explanation for clustering method.

        Args:
            method: Clustering method name

        Returns:
            Method explanation data
        """
        explanations = {
            'mds': {
                'title': '📊 Multidimensional Scaling (MDS) on Correlation Matrix',
                'input_data': 'Uses correlation matrix of alpha PnL returns (not performance features like Sharpe ratio). Converts correlations to distances using d_ij = √(2(1 - ρ_ij)) where ρ is the correlation coefficient.',
                'what_shows': 'Maps alphas to 2D space where distance represents correlation dissimilarity. Unlike other methods, MDS directly measures alpha overlap rather than performance similarity.',
                'why_useful': 'Alphas close together are highly correlated (avoid combining), while distant alphas are uncorrelated (good for diversification). The Euclidean distance in the plot directly corresponds to portfolio diversification benefit.',
                'mathematics': 'Minimizes stress function S = Σ_ij (d_ij - δ_ij)² where d_ij is the 2D distance and δ_ij is the original correlation distance.'
            },

            'tsne': {
                'title': '🔬 t-SNE on Performance Features',
                'what_shows': 'Non-linear projection emphasizing local structure - alphas with similar risk-return profiles cluster together. Uses Sharpe ratio, volatility, drawdown, skewness, and kurtosis as features.',
                'why_useful': 'Reveals natural groupings of strategies with similar behavior patterns. Tight clusters indicate redundant strategies; isolated points represent unique approaches worth including.',
                'mathematics': 'Minimizes KL divergence between probability distributions: KL(P||Q) = Σ_ij p_ij log(p_ij/q_ij). Uses Student\'s t-distribution in embedding space for heavy-tailed flexibility.'
            },

            'umap': {
                'title': '🗺️ UMAP on Performance Features',
                'what_shows': 'Preserves both local and global structure - maintains meaningful distances between clusters. Superior to t-SNE for understanding relationships between different strategy groups.',
                'why_useful': 'Inter-cluster distances are meaningful - can identify which strategy groups are most different. Useful for hierarchical portfolio construction across multiple strategy types.',
                'mathematics': 'Constructs fuzzy topological representation using k-NN graph, then optimizes layout via cross-entropy. Balances local structure preservation (n_neighbors) with global structure (min_dist parameter).'
            },

            'pca': {
                'title': '📐 PCA on Performance Features',
                'what_shows': 'Linear projection onto principal components - PC1 typically captures risk-return trade-off, PC2 captures style factors. Preserves global structure and relative distances between all alphas.',
                'why_useful': 'Interpretable axes - can understand what drives separation between strategies. Linear nature means portfolio combinations behave predictably in this space.',
                'mathematics': 'Eigendecomposition of covariance matrix: Σ = VΛV^T where columns of V are principal components. Projects data onto eigenvectors with largest eigenvalues to maximize variance.'
            },

            'heatmap': {
                'title': 'Correlation Heatmap',
                'what_shows': 'Full N×N correlation matrix with hierarchical clustering reordering. Red = positive correlation (redundant), Blue = negative (natural hedges), White = uncorrelated.',
                'why_useful': 'Direct view of all pairwise relationships - identify blocks of similar strategies. Diagonal blocks reveal strategy families; off-diagonal patterns show cross-dependencies.',
                'mathematics': 'Pearson correlation of percentage returns: ρ = Cov(r_i, r_j) / (σ_i × σ_j).'
            }
        }

        return explanations.get(method, {
            'title': 'Unknown Method',
            'what_shows': 'Select a valid visualization method.',
            'why_useful': '',
            'mathematics': ''
        })

    def update_region_selection(self, all_region_data: Dict[str, ClusteringData],
                              selected_region: str) -> Dict[str, Any]:
        """
        Update region selection and return all coordinate stores for the region.

        Args:
            all_region_data: All region clustering data
            selected_region: Newly selected region

        Returns:
            Dictionary with all coordinate stores for the region
        """
        if not selected_region or selected_region not in all_region_data:
            # Return empty data for all stores
            return {
                'current-mds-data': [],
                'mds-data-simple': [],
                'mds-data-euclidean': [],
                'mds-data-angular': [],
                'current-tsne-data': [],
                'current-umap-data': [],
                'current-pca-data': [],
                'current-pca-info': {},
                'current-metadata': [],
                'heatmap-data-simple': {},
                'heatmap-data-euclidean': {},
                'heatmap-data-angular': {},
                'tsne-cluster-profiles': {},
                'umap-cluster-profiles': {},
                'pca-cluster-profiles': {},
                'main-cluster-profiles': {},
                'clustering-region-info': "No data available"
            }

        clustering_data = all_region_data[selected_region]

        # Create info text
        alpha_count = clustering_data.alpha_count
        timestamp = clustering_data.timestamp or 'Unknown'
        info_text = f"{alpha_count} alphas | Generated: {timestamp}"

        return {
            'current-mds-data': clustering_data.mds_euclidean.to_records(),
            'mds-data-simple': clustering_data.mds_simple.to_records(),
            'mds-data-euclidean': clustering_data.mds_euclidean.to_records(),
            'mds-data-angular': clustering_data.mds_angular.to_records(),
            'current-tsne-data': clustering_data.tsne_coords.to_records(),
            'current-umap-data': clustering_data.umap_coords.to_records(),
            'current-pca-data': clustering_data.pca_coords.to_records(),
            'current-pca-info': clustering_data.pca_info,
            'current-metadata': [
                {'index': k, **v} for k, v in clustering_data.alpha_metadata.items()
            ],
            'heatmap-data-simple': clustering_data.heatmap_simple,
            'heatmap-data-euclidean': clustering_data.heatmap_euclidean,
            'heatmap-data-angular': clustering_data.heatmap_angular,
            'tsne-cluster-profiles': clustering_data.tsne_cluster_profiles,
            'umap-cluster-profiles': clustering_data.umap_cluster_profiles,
            'pca-cluster-profiles': clustering_data.pca_cluster_profiles,
            'main-cluster-profiles': clustering_data.main_cluster_profiles,
            'clustering-region-info': info_text
        }

    def prepare_plot_data(self, method: str, coordinate_data: List[Dict[str, Any]],
                         pca_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare plot data for visualization.

        Args:
            method: Clustering method
            coordinate_data: Coordinate data in records format
            pca_info: PCA information (for PCA method)

        Returns:
            Prepared plot data
        """
        if not coordinate_data:
            return {'empty': True, 'message': f'No data available for {method.upper()} clustering'}

        plot_data = pd.DataFrame(coordinate_data)

        # Check required columns
        if 'x' not in plot_data.columns or 'y' not in plot_data.columns:
            return {'empty': True, 'message': f'Invalid coordinate data for {method}'}

        # Set axis labels based on method
        if method == 'pca' and pca_info and 'variance_explained' in pca_info:
            var_exp = pca_info['variance_explained']
            pc1_var = var_exp.get('pc1', 0) * 100  # Convert to percentage
            pc2_var = var_exp.get('pc2', 0) * 100
            total_var = var_exp.get('total_2d', 0) * 100

            title = f"PCA on Performance Features (Total Variance: {total_var:.1f}%)"
            x_label = f"PC1 ({pc1_var:.1f}%)"
            y_label = f"PC2 ({pc2_var:.1f}%)"

            # Add interpretation hints to the labels if available
            if 'interpretation' in pca_info:
                interp = pca_info['interpretation']
                if interp.get('pc1') and interp['pc1'] != "Mixed factors":
                    x_label += f": {interp['pc1']}"
                if interp.get('pc2') and interp['pc2'] != "Mixed factors":
                    y_label += f": {interp['pc2']}"
        else:
            method_titles = {
                'mds': 'MDS on Correlation Matrix',
                'tsne': 't-SNE on Performance Features',
                'umap': 'UMAP on Performance Features',
                'pca': 'PCA on Performance Features'
            }
            title = method_titles.get(method, f'{method.upper()} Clustering')
            x_label = 'Dimension 1'
            y_label = 'Dimension 2'

        return {
            'plot_data': plot_data,
            'title': title,
            'x_label': x_label,
            'y_label': y_label,
            'empty': False
        }

    def get_cluster_color_mapping(self, plot_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get cluster color mapping for visualization.

        Args:
            plot_data: Plot data DataFrame

        Returns:
            Color mapping and cluster information
        """
        from ..utils import get_cluster_color_map

        if 'cluster' not in plot_data.columns:
            return {'has_clusters': False}

        # Check if we have cluster data
        has_clusters = plot_data['cluster'].notna().any()

        if not has_clusters:
            return {'has_clusters': False}

        # Create cluster string labels
        plot_data_with_clusters = plot_data.copy()
        plot_data_with_clusters['cluster_str'] = plot_data_with_clusters['cluster'].apply(
            lambda x: f'Cluster {int(x)}' if x >= 0 else 'Outlier'
        )

        # Get unique clusters and color mapping
        unique_clusters = sorted(
            plot_data_with_clusters['cluster_str'].unique(),
            key=self._cluster_sort_key
        )
        color_map = get_cluster_color_map(unique_clusters)

        return {
            'has_clusters': True,
            'plot_data_with_clusters': plot_data_with_clusters,
            'unique_clusters': unique_clusters,
            'color_map': color_map
        }

    def _cluster_sort_key(self, cluster_str: str) -> Tuple[int, int]:
        """Sort key for cluster strings (Cluster 0, 1, 2... then Outlier)."""
        if cluster_str == 'Outlier':
            return (1, 0)  # Sort Outlier last
        else:
            try:
                # Extract numeric part from "Cluster X"
                cluster_num = int(cluster_str.split()[-1])
                return (0, cluster_num)  # Sort clusters numerically
            except (ValueError, IndexError):
                return (2, 0)  # Unknown format

    def apply_highlights_to_plot_data(self, plot_data: pd.DataFrame,
                                    operator_alphas: List[str] = None,
                                    datafield_alphas: List[str] = None,
                                    selected_alpha: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply highlighting information to plot data.

        Args:
            plot_data: Plot data DataFrame
            operator_alphas: List of alphas highlighted by operators
            datafield_alphas: List of alphas highlighted by datafields
            selected_alpha: Currently selected alpha

        Returns:
            Highlighting data for visualization
        """
        highlight_info = {
            'operator_matches': pd.DataFrame(),
            'datafield_matches': pd.DataFrame(),
            'selected_point': pd.DataFrame()
        }

        if not plot_data.empty and 'index' in plot_data.columns:
            # Operator highlights
            if operator_alphas:
                operator_matches = plot_data[plot_data['index'].isin(operator_alphas)]
                highlight_info['operator_matches'] = operator_matches

            # Datafield highlights
            if datafield_alphas:
                datafield_matches = plot_data[plot_data['index'].isin(datafield_alphas)]
                highlight_info['datafield_matches'] = datafield_matches

            # Selected alpha highlight
            if selected_alpha:
                selected_index = selected_alpha.get('index')
                if selected_index and selected_index in plot_data['index'].values:
                    selected_point = plot_data[plot_data['index'] == selected_index]
                    highlight_info['selected_point'] = selected_point

        return highlight_info

    def get_pca_dynamic_info(self, pca_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get dynamic PCA information for display.

        Args:
            pca_info: PCA analysis information

        Returns:
            Formatted PCA information
        """
        if not pca_info:
            return {}

        result = {}

        # Add variance explained information
        if 'variance_explained' in pca_info:
            var_exp = pca_info['variance_explained']
            pc1_var = var_exp.get('pc1', 0) * 100
            pc2_var = var_exp.get('pc2', 0) * 100
            total_var = var_exp.get('total_2d', 0) * 100

            result['variance_info'] = f"PC1: {pc1_var:.1f}%, PC2: {pc2_var:.1f}% (2D captures {total_var:.1f}% of variance)"

        # Enhanced interpretation with category information
        if 'interpretation' in pca_info:
            interp = pca_info['interpretation']
            result['interpretation'] = {}
            if interp.get('pc1') and interp['pc1'] != "Mixed factors":
                result['interpretation']['pc1'] = interp['pc1']
            if interp.get('pc2') and interp['pc2'] != "Mixed factors":
                result['interpretation']['pc2'] = interp['pc2']

        # Add feature contributions with color coding
        if 'top_features' in pca_info:
            result['feature_contributions'] = self._format_pca_features(pca_info['top_features'])

        return result

    def _format_pca_features(self, top_features: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Dict[str, Any]]]:
        """Format PCA feature contributions for display."""
        from ..config import PCA_FEATURE_COLORS

        formatted = {}

        for component, features in top_features.items():
            component_features = []
            for feat, contrib in features:
                # Determine category color
                color = "#666"  # Default
                for category, cat_color in PCA_FEATURE_COLORS.items():
                    if feat.startswith(f'{category}_'):
                        color = cat_color
                        break

                clean_name = feat
                # Handle compound prefix risk_regime_ first
                clean_name = clean_name.replace('risk_regime_', '')
                # Then handle other prefixes
                for prefix in ['spiked_', 'multiscale_', 'risk_', 'metadata_', 'regime_']:
                    clean_name = clean_name.replace(prefix, '')

                component_features.append({
                    'name': clean_name,
                    'contribution': contrib,
                    'color': color,
                    'full_name': feat
                })

            formatted[component] = component_features

        return formatted

# Global service instance
_clustering_service_instance = None

def get_clustering_service() -> ClusteringService:
    """Get singleton clustering service instance."""
    global _clustering_service_instance
    if _clustering_service_instance is None:
        _clustering_service_instance = ClusteringService()
    return _clustering_service_instance

def reset_clustering_service():
    """Reset clustering service instance (for testing)."""
    global _clustering_service_instance
    _clustering_service_instance = None