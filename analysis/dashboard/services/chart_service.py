"""
Chart Service

Chart data preparation and Plotly chart creation for the dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math
from typing import Dict, List, Any, Optional, Tuple

from .data_service import get_alpha_details_for_clustering
from ..utils import (
    create_hover_text, get_cluster_color_map, create_scatter_trace,
    create_highlight_trace, add_plot_annotations, apply_chart_template,
    create_bar_chart, create_pie_chart, scale_treemap_values,
    create_correlation_heatmap
)
from ..config import TEMPLATE, COLOR_SCHEMES, CHART_DIMENSIONS

class ChartService:
    """Service for creating and managing dashboard charts."""

    def __init__(self):
        """Initialize chart service."""
        pass

    def create_clustering_plot(self, method: str, plot_data: pd.DataFrame,
                              pca_info: Dict[str, Any] = None,
                              operator_alphas: List[str] = None,
                              datafield_alphas: List[str] = None,
                              selected_alpha: Dict[str, Any] = None,
                              color_mode: str = 'cluster',
                              distance_metric: str = 'euclidean') -> go.Figure:
        """
        Create clustering visualization plot.

        Args:
            method: Clustering method ('mds', 'tsne', 'umap', 'pca', 'heatmap')
            plot_data: Plot data DataFrame
            pca_info: PCA information (for PCA method)
            operator_alphas: List of operator-highlighted alphas
            datafield_alphas: List of datafield-highlighted alphas
            selected_alpha: Currently selected alpha
            color_mode: Coloring mode ('cluster' or 'single')
            distance_metric: Distance metric for MDS

        Returns:
            Plotly Figure
        """
        # Handle empty data
        if plot_data.empty or 'x' not in plot_data.columns or 'y' not in plot_data.columns:
            return self._create_empty_plot(method)

        # Set title and axis labels
        title, x_label, y_label = self._get_plot_labels(method, pca_info, distance_metric)

        # Fetch alpha details for hover information
        alpha_ids = plot_data['index'].tolist() if 'index' in plot_data.columns else []
        alpha_details = get_alpha_details_for_clustering(alpha_ids)

        # Create enhanced hover text
        hover_texts = self._create_hover_texts(alpha_ids, alpha_details, operator_alphas, datafield_alphas)

        # Create the main scatter plot
        fig = self._create_scatter_plot(plot_data, hover_texts, color_mode)

        # Add highlighting traces
        self._add_highlight_traces(fig, plot_data, operator_alphas, datafield_alphas, selected_alpha)

        # Update layout and annotations
        apply_chart_template(fig, title, x_label, y_label)
        self._add_plot_annotations(fig, operator_alphas, datafield_alphas)

        return fig

    def create_heatmap_plot(self, heatmap_data: Dict[str, Any]) -> go.Figure:
        """
        Create correlation heatmap plot.

        Args:
            heatmap_data: Heatmap data with correlation matrix and alpha IDs

        Returns:
            Plotly Figure
        """
        if not heatmap_data or 'correlation_matrix' not in heatmap_data or 'alpha_ids' not in heatmap_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No pre-calculated heatmap data available.<br>Please regenerate clustering data.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )
            return fig

        return create_correlation_heatmap(
            heatmap_data['correlation_matrix'],
            heatmap_data['alpha_ids'],
            "Correlation Heatmap"
        )

    def create_operators_chart(self, operators_data: List[Tuple[str, int]],
                              chart_type: str = 'top20', max_items: int = 20) -> go.Figure:
        """
        Create operators bar chart.

        Args:
            operators_data: List of (operator, count) tuples
            chart_type: Chart type ('top20', 'all')
            max_items: Maximum items to display

        Returns:
            Plotly Figure
        """
        if chart_type == 'top20':
            data = operators_data[:max_items]
            title = f"Top {min(max_items, len(operators_data))} Most Used Operators (Click bars for details)"
        else:
            data = operators_data
            title = f"All {len(operators_data)} Used Operators (Click bars for details)"

        return create_bar_chart(
            data=data,
            title=title,
            x_label="Number of Alphas Using",
            y_label="Operator",
            color="steelblue",
            height=600 if chart_type == 'top20' else max(800, len(data) * 25)
        )

    def create_datafields_chart(self, datafields_data: List[Tuple[str, int]],
                               chart_type: str = 'top20', max_items: int = 20) -> go.Figure:
        """
        Create datafields bar chart.

        Args:
            datafields_data: List of (datafield, count) tuples
            chart_type: Chart type ('top20', 'all')
            max_items: Maximum items to display

        Returns:
            Plotly Figure
        """
        if chart_type == 'top20':
            data = datafields_data[:max_items]
            title = f"Top {min(max_items, len(datafields_data))} Most Used Datafields (Click bars for details)"
        else:
            data = datafields_data
            title = f"All {len(datafields_data)} Used Datafields (Click bars for details)"

        return create_bar_chart(
            data=data,
            title=title,
            x_label="Number of Alphas Using",
            y_label="Datafield",
            color="darkgreen",
            height=500 if chart_type == 'top20' else max(800, len(data) * 20)
        )

    def create_datasets_chart(self, datasets_data: List[Tuple[str, int]]) -> go.Figure:
        """
        Create datasets bar chart.

        Args:
            datasets_data: List of (dataset, count) tuples

        Returns:
            Plotly Figure
        """
        return create_bar_chart(
            data=datasets_data,
            title=f"All Used Datasets ({len(datasets_data)} total) - Click bars for details",
            x_label="Total Datafield Instances",
            y_label="Dataset ID",
            color="steelblue",
            height=max(400, len(datasets_data) * 25 + 100)
        )

    def create_category_pie_chart(self, category_data: Dict[str, int]) -> go.Figure:
        """
        Create category pie chart.

        Args:
            category_data: Dictionary of category -> count

        Returns:
            Plotly Figure
        """
        if not category_data:
            fig = go.Figure()
            fig.update_layout(
                title="No Category Data Available",
                height=300,
                annotations=[dict(text="No data to display", x=0.5, y=0.5, showarrow=False)]
            )
            return fig

        return create_pie_chart(
            data=category_data,
            title="Usage by Category",
            color_sequence=px.colors.qualitative.Set3
        )

    def create_dataset_treemap(self, dataset_stats: List[Dict[str, Any]]) -> go.Figure:
        """
        Create enhanced treemap for datasets.

        Args:
            dataset_stats: List of dataset statistics

        Returns:
            Plotly Figure
        """
        if not dataset_stats:
            return go.Figure()

        # Use all datasets (not limited)
        all_stats = dataset_stats

        # Calculate total area for text display threshold
        total_datafields = sum([stat['total_datafields'] for stat in all_stats])

        # Create enhanced color mapping based on alpha usage
        colors = self._create_treemap_colors(all_stats)
        text_info = self._create_treemap_text(all_stats, total_datafields)
        hover_text = self._create_treemap_hover(all_stats)

        # Apply scaling to improve visibility of all datasets
        scaled_values = scale_treemap_values([stat['total_datafields'] for stat in all_stats])

        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=[stat['dataset_id'] for stat in all_stats],
            values=scaled_values,
            parents=["" for _ in all_stats],
            text=text_info,
            textinfo="text",
            textposition="middle center",
            textfont=dict(size=18, color="white", family="Arial Black"),
            marker=dict(
                colors=colors,
                line=dict(width=1, color="white")
            ),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
            maxdepth=1
        ))

        fig.update_layout(
            title={
                'text': f"Dataset Treemap: Size=Datafield Count (scaled), Color=Alpha Usage (All {len(all_stats)} datasets)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=750,
            font=dict(size=20),
            template=TEMPLATE,
            margin=dict(t=80, l=10, r=10, b=10),
            uniformtext=dict(minsize=8, mode='hide')
        )

        return fig

    def create_pca_loadings_heatmap(self, pca_info: Dict[str, Any]) -> go.Figure:
        """
        Create PCA loadings heatmap.

        Args:
            pca_info: PCA analysis information

        Returns:
            Plotly Figure
        """
        if not pca_info or 'loadings' not in pca_info:
            fig = go.Figure()
            fig.add_annotation(
                text="No PCA loadings data available",
                showarrow=False,
                xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=16)
            )
            return fig

        loadings = pca_info['loadings']
        feature_names = loadings.get('feature_names', [])
        pc1_loadings = loadings.get('pc1', {})
        pc2_loadings = loadings.get('pc2', {})

        if not feature_names or not pc1_loadings or not pc2_loadings:
            fig = go.Figure()
            fig.add_annotation(
                text="Incomplete PCA loadings data",
                showarrow=False,
                xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=16)
            )
            return fig

        # Get feature categories for color-coding
        feature_categories = pca_info.get('feature_categories', {})

        # Create enhanced feature display names and categories
        enhanced_display_names, feature_category_labels = self._process_pca_features(
            feature_names, feature_categories
        )

        # Create loadings matrix for heatmap
        pc1_values = [pc1_loadings.get(feature, 0) for feature in feature_names]
        pc2_values = [pc2_loadings.get(feature, 0) for feature in feature_names]
        loadings_matrix = [pc1_values, pc2_values]
        components = ['PC1', 'PC2']

        # Create custom hover text
        hover_text = self._create_pca_hover_text(
            components, loadings_matrix, feature_names,
            feature_category_labels, enhanced_display_names
        )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=loadings_matrix,
            x=enhanced_display_names,
            y=components,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Loading Value"),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        ))

        # Add category color bars and formatting
        self._format_pca_loadings_chart(fig, pca_info, feature_category_labels, feature_names)

        return fig

    def _create_empty_plot(self, method: str) -> go.Figure:
        """Create empty plot with message."""
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
            template=TEMPLATE,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    def _get_plot_labels(self, method: str, pca_info: Dict[str, Any] = None,
                        distance_metric: str = 'euclidean') -> Tuple[str, str, str]:
        """Get plot title and axis labels."""
        if method == 'mds':
            distance_label = {'simple': 'Simple', 'euclidean': 'Euclidean', 'angular': 'Angular'}.get(distance_metric, 'Euclidean')
            title = f"MDS on Correlation Matrix ({distance_label} distance)"
            x_label, y_label = "Dimension 1", "Dimension 2"
        elif method == 'tsne':
            title = "t-SNE on Performance Features"
            x_label, y_label = "Dimension 1", "Dimension 2"
        elif method == 'umap':
            title = "UMAP on Performance Features"
            x_label, y_label = "Dimension 1", "Dimension 2"
        elif method == 'pca':
            if pca_info and 'variance_explained' in pca_info:
                var_exp = pca_info['variance_explained']
                pc1_var = var_exp.get('pc1', 0) * 100
                pc2_var = var_exp.get('pc2', 0) * 100
                total_var = var_exp.get('total_2d', 0) * 100
                title = f"PCA on Performance Features (Total Variance: {total_var:.1f}%)"
                x_label = f"PC1 ({pc1_var:.1f}%)"
                y_label = f"PC2 ({pc2_var:.1f}%)"

                # Add interpretation hints if available
                if 'interpretation' in pca_info:
                    interp = pca_info['interpretation']
                    if interp.get('pc1') and interp['pc1'] != "Mixed factors":
                        x_label += f": {interp['pc1']}"
                    if interp.get('pc2') and interp['pc2'] != "Mixed factors":
                        y_label += f": {interp['pc2']}"
            else:
                title = "PCA on Performance Features"
                x_label, y_label = "PC1", "PC2"
        else:
            title = f"{method.upper()} Clustering"
            x_label, y_label = "Dimension 1", "Dimension 2"

        return title, x_label, y_label

    def _create_hover_texts(self, alpha_ids: List[str], alpha_details: Dict[str, Any],
                           operator_alphas: List[str] = None,
                           datafield_alphas: List[str] = None) -> List[str]:
        """Create enhanced hover texts."""
        hover_texts = []
        for alpha_id in alpha_ids:
            details = alpha_details.get(alpha_id, {})

            # Add highlighting information
            match_info = []
            if operator_alphas and alpha_id in operator_alphas:
                match_info.append("● Operator Match")
            if datafield_alphas and alpha_id in datafield_alphas:
                match_info.append("● Datafield Match")

            match_text = " | ".join(match_info)
            hover_text = create_hover_text(alpha_id, details, "", match_text)
            hover_texts.append(hover_text)

        return hover_texts

    def _create_scatter_plot(self, plot_data: pd.DataFrame, hover_texts: List[str],
                            color_mode: str) -> go.Figure:
        """Create the main scatter plot."""
        fig = go.Figure()

        if 'cluster' in plot_data.columns and color_mode == 'cluster':
            # Cluster coloring mode
            plot_data_with_clusters = plot_data.copy()
            plot_data_with_clusters['cluster_str'] = plot_data_with_clusters['cluster'].apply(
                lambda x: f'Cluster {int(x)}' if x >= 0 else 'Outlier'
            )

            # Get color mapping
            unique_clusters = sorted(
                plot_data_with_clusters['cluster_str'].unique(),
                key=self._cluster_sort_key
            )
            color_map = get_cluster_color_map(unique_clusters)

            # Add one trace per cluster for proper legend colors
            for cluster_name in unique_clusters:
                cluster_data = plot_data_with_clusters[plot_data_with_clusters['cluster_str'] == cluster_name]

                # Get hover texts for this cluster's alphas
                cluster_hover_texts = self._get_cluster_hover_texts(
                    cluster_data['index'].tolist(), plot_data['index'].tolist(), hover_texts, cluster_name
                )

                fig.add_trace(create_scatter_trace(
                    x_data=cluster_data['x'].tolist(),
                    y_data=cluster_data['y'].tolist(),
                    text=cluster_data['index'].tolist(),
                    hover_text=cluster_hover_texts,
                    customdata=cluster_data['index'].tolist(),
                    name=cluster_name,
                    color=color_map[cluster_name]
                ))
        else:
            # Single color mode
            fig.add_trace(create_scatter_trace(
                x_data=plot_data['x'].tolist(),
                y_data=plot_data['y'].tolist(),
                text=plot_data['index'].tolist() if 'index' in plot_data.columns else [],
                hover_text=hover_texts,
                customdata=plot_data['index'].tolist() if 'index' in plot_data.columns else [],
                name='Alphas',
                color=COLOR_SCHEMES['single_color']
            ))

        return fig

    def _add_highlight_traces(self, fig: go.Figure, plot_data: pd.DataFrame,
                             operator_alphas: List[str] = None,
                             datafield_alphas: List[str] = None,
                             selected_alpha: Dict[str, Any] = None):
        """Add highlighting traces to the plot."""
        # Add operator highlights
        if operator_alphas and not plot_data.empty and 'index' in plot_data.columns:
            operator_matches = plot_data[plot_data['index'].isin(operator_alphas)]
            if not operator_matches.empty:
                fig.add_trace(create_highlight_trace(
                    x_data=operator_matches['x'].tolist(),
                    y_data=operator_matches['y'].tolist(),
                    color=COLOR_SCHEMES['operator_highlight'],
                    name='Operator Match'
                ))

        # Add datafield highlights
        if datafield_alphas and not plot_data.empty and 'index' in plot_data.columns:
            datafield_matches = plot_data[plot_data['index'].isin(datafield_alphas)]
            if not datafield_matches.empty:
                fig.add_trace(create_highlight_trace(
                    x_data=datafield_matches['x'].tolist(),
                    y_data=datafield_matches['y'].tolist(),
                    color=COLOR_SCHEMES['datafield_highlight'],
                    name='Datafield Match'
                ))

        # Highlight selected alpha (highest priority)
        if selected_alpha and not plot_data.empty and 'index' in plot_data.columns:
            selected_index = selected_alpha.get('index')
            if selected_index in plot_data['index'].values:
                selected_point = plot_data[plot_data['index'] == selected_index]

                fig.add_trace(go.Scatter(
                    x=selected_point['x'],
                    y=selected_point['y'],
                    mode='markers',
                    marker=dict(
                        color=COLOR_SCHEMES['selected_alpha'],
                        size=15,
                        line=dict(width=2, color='black')
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))

    def _add_plot_annotations(self, fig: go.Figure, operator_alphas: List[str] = None,
                             datafield_alphas: List[str] = None):
        """Add annotations to the plot."""
        annotations = ["Click on a point to open its WorldQuant Brain link"]

        # Add highlighting legend if there are highlights
        if operator_alphas or datafield_alphas:
            legend_parts = []
            if operator_alphas:
                legend_parts.append("<span style='color:green'>●</span> Operator Match")
            if datafield_alphas:
                legend_parts.append("<span style='color:darkgreen'>●</span> Datafield Match")
            legend_parts.append("<span style='color:red'>●</span> Selected")
            annotations.append(" | ".join(legend_parts))

        add_plot_annotations(fig, annotations)

    def _cluster_sort_key(self, cluster_str: str) -> Tuple[int, int]:
        """Sort key for cluster strings."""
        if cluster_str == 'Outlier':
            return (1, 0)
        else:
            try:
                cluster_num = int(cluster_str.split()[-1])
                return (0, cluster_num)
            except (ValueError, IndexError):
                return (2, 0)

    def _get_cluster_hover_texts(self, cluster_alpha_ids: List[str], all_alpha_ids: List[str],
                                all_hover_texts: List[str], cluster_name: str) -> List[str]:
        """Get hover texts for a specific cluster."""
        cluster_hover_texts = []
        for alpha_id in cluster_alpha_ids:
            try:
                alpha_idx = all_alpha_ids.index(alpha_id)
                if alpha_idx < len(all_hover_texts):
                    cluster_hover_texts.append(all_hover_texts[alpha_idx])
                else:
                    cluster_hover_texts.append(f"<b>{alpha_id}</b><br>Cluster: {cluster_name}")
            except (ValueError, IndexError):
                cluster_hover_texts.append(f"<b>{alpha_id}</b><br>Cluster: {cluster_name}")

        return cluster_hover_texts

    def _create_treemap_colors(self, dataset_stats: List[Dict[str, Any]]) -> List[str]:
        """Create color mapping for treemap based on alpha usage."""
        colors = []
        max_usage = max([d['alpha_usage_count'] for d in dataset_stats]) if dataset_stats else 1

        for stat in dataset_stats:
            if stat['alpha_usage_count'] == 0:
                colors.append('#808080')  # Grey for unused datasets
            else:
                usage_count = stat['alpha_usage_count']
                if usage_count <= 5:
                    # Light blue to medium blue for 1-5 alphas
                    intensity = usage_count / 5
                    blue_val = int(200 - (intensity * 50))
                    colors.append(f'rgb(100, 150, {blue_val})')
                elif usage_count <= 20:
                    # Medium blue to dark blue for 6-20 alphas
                    intensity = (usage_count - 5) / 15
                    blue_val = int(150 - (intensity * 50))
                    colors.append(f'rgb(80, 130, {blue_val})')
                else:
                    # Dark blue to purple for 20+ alphas
                    intensity = min((usage_count - 20) / 30, 1.0)
                    red_val = int(80 + (intensity * 50))
                    blue_val = int(100 - (intensity * 30))
                    colors.append(f'rgb({red_val}, 100, {blue_val})')

        return colors

    def _create_treemap_text(self, dataset_stats: List[Dict[str, Any]], total_datafields: int) -> List[str]:
        """Create text labels for treemap."""
        # Apply scaling and smart text display
        scaled_values = scale_treemap_values([stat['total_datafields'] for stat in dataset_stats])
        total_scaled = sum(scaled_values)
        text_info = []

        for i, stat in enumerate(dataset_stats):
            scaled_percentage = (scaled_values[i] / total_scaled * 100) if total_scaled > 0 else 0

            if scaled_percentage >= 1.5:
                text_info.append(f"{stat['dataset_id']}<br>{stat['alpha_usage_count']} alphas")
            elif scaled_percentage >= 0.8:
                text_info.append(f"{stat['dataset_id']}<br>{stat['alpha_usage_count']}")
            elif scaled_percentage >= 0.4:
                text_info.append(stat['dataset_id'])
            else:
                dataset_id = stat['dataset_id']
                if len(dataset_id) > 8:
                    text_info.append(dataset_id[:8] + "...")
                else:
                    text_info.append(dataset_id)

        return text_info

    def _create_treemap_hover(self, dataset_stats: List[Dict[str, Any]]) -> List[str]:
        """Create hover text for treemap."""
        hover_text = []
        for stat in dataset_stats:
            hover_info = (
                f"<b>{stat['dataset_id']}</b><br>"
                f"Total Datafields: {stat['total_datafields']}<br>"
                f"Used Datafields: {stat['used_datafields']}<br>"
                f"Usage Rate: {stat['usage_percentage']:.1f}%<br>"
                f"Alphas Using: {stat['alpha_usage_count']}<br>"
                f"Status: {'Used' if stat['is_used'] else 'Unused'}"
            )
            hover_text.append(hover_info)

        return hover_text

    def _process_pca_features(self, feature_names: List[str],
                             feature_categories: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Process PCA features for display."""
        enhanced_display_names = []
        feature_category_labels = []

        for feature in feature_names:
            # First remove the prefix (handle compound prefixes like risk_regime_)
            clean_name = feature
            # Handle compound prefix risk_regime_ first
            if feature.startswith('risk_regime_'):
                clean_name = feature[len('risk_regime_'):]
            else:
                # Then handle single prefixes
                for prefix in ['risk_', 'metadata_', 'spiked_', 'multiscale_', 'regime_']:
                    if feature.startswith(prefix):
                        clean_name = feature[len(prefix):]
                        break

            # Then convert to title case
            display_name = clean_name.replace('_', ' ').title()

            # Determine category
            category = 'other'
            for cat_name, cat_features in feature_categories.items():
                if feature in cat_features:
                    category = cat_name
                    break

            enhanced_display_names.append(display_name)
            feature_category_labels.append(category)

        return enhanced_display_names, feature_category_labels

    def _create_pca_hover_text(self, components: List[str], loadings_matrix: List[List[float]],
                              feature_names: List[str], feature_category_labels: List[str],
                              enhanced_display_names: List[str]) -> List[List[str]]:
        """Create hover text for PCA loadings heatmap."""
        hover_text = []
        for i, component in enumerate(components):
            component_text = []
            values = loadings_matrix[i]
            for j, (feature, value, category, display_name) in enumerate(
                zip(feature_names, values, feature_category_labels, enhanced_display_names)
            ):
                component_text.append(
                    f'<b>{component}</b><br>'
                    f'Feature: {display_name}<br>'
                    f'Category: {category.title()}<br>'
                    f'Loading: {value:.3f}<br>'
                    f'<i>Full name: {feature}</i>'
                )
            hover_text.append(component_text)

        return hover_text

    def _format_pca_loadings_chart(self, fig: go.Figure, pca_info: Dict[str, Any],
                                  feature_category_labels: List[str], feature_names: List[str]):
        """Format PCA loadings chart with categories and styling."""
        from ..config import PCA_FEATURE_COLORS

        # Add category color bars at the bottom
        category_counts = {cat: feature_category_labels.count(cat) for cat in set(feature_category_labels)}

        # Create category annotations
        total_features = len(feature_names)
        category_text = []
        for category, count in category_counts.items():
            if count > 0:
                color = PCA_FEATURE_COLORS.get(category, '#808080')
                category_text.append(f'<span style="color:{color};">●</span> {category.title()}: {count}')

        subtitle = f"Total Features: {total_features} | " + " | ".join(category_text)

        # Add variance explained information to title
        title = "Enhanced PCA Feature Loadings"
        if 'variance_explained' in pca_info:
            var_exp = pca_info['variance_explained']
            pc1_var = var_exp.get('pc1', 0) * 100
            pc2_var = var_exp.get('pc2', 0) * 100
            title = f"Enhanced PCA Feature Loadings (PC1: {pc1_var:.1f}%, PC2: {pc2_var:.1f}%)"

        # Add colored rectangles for feature categories on x-axis
        shapes = []
        current_x = -0.5
        for i, category in enumerate(feature_category_labels):
            color = PCA_FEATURE_COLORS.get(category, '#808080')
            shapes.append(dict(
                type="rect",
                x0=current_x, x1=current_x + 1,
                y0=-2.8, y1=-2.6,
                fillcolor=color,
                opacity=0.6,
                line=dict(width=0)
            ))
            current_x += 1

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>{subtitle}</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Features (Color-coded by Category)",
            yaxis_title="Principal Components",
            template=TEMPLATE,
            height=450,
            xaxis=dict(
                tickangle=45,
                range=[-0.5, len(feature_names) - 0.5]
            ),
            yaxis=dict(
                range=[-3, 2]  # Make room for category bars
            ),
            shapes=shapes
        )

# Global service instance
_chart_service_instance = None

def get_chart_service() -> ChartService:
    """Get singleton chart service instance."""
    global _chart_service_instance
    if _chart_service_instance is None:
        _chart_service_instance = ChartService()
    return _chart_service_instance

def reset_chart_service():
    """Reset chart service instance (for testing)."""
    global _chart_service_instance
    _chart_service_instance = None