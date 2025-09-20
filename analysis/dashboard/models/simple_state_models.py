"""
Simple State Models

Simplified state models without pydantic dependency for compatibility testing.
This allows testing the refactored structure while working on dependency issues.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd


class AnalysisFilters:
    """Analysis filter parameters with basic validation."""

    def __init__(self, region: str = None, universe: str = None, delay: int = None,
                 date_from: str = None, date_to: str = None):
        self.region = region
        self.universe = universe
        self.delay = delay
        self.date_from = date_from
        self.date_to = date_to

        # Basic validation
        if delay is not None and (delay < 0 or delay > 30):
            raise ValueError("Delay must be between 0 and 30")

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy filter format for backward compatibility."""
        return {
            'region': self.region,
            'universe': self.universe,
            'delay': self.delay,
            'date_from': self.date_from,
            'date_to': self.date_to
        }

    @classmethod
    def from_legacy_dict(cls, data: Dict[str, Any]) -> 'AnalysisFilters':
        """Create from legacy filter format."""
        return cls(**{k: v for k, v in data.items() if v is not None})


class CoordinateData:
    """Clustering coordinate data with format conversion."""

    def __init__(self, coordinates: Dict[str, Dict[str, float]] = None):
        self.coordinates = coordinates or {}

    def to_records_format(self) -> List[Dict[str, Any]]:
        """Convert to records format for Dash stores."""
        if not self.coordinates:
            return []

        df = pd.DataFrame.from_dict(self.coordinates, orient='index')
        return df.reset_index().to_dict('records')

    @classmethod
    def from_records_format(cls, records: List[Dict[str, Any]]) -> 'CoordinateData':
        """Create from records format."""
        if not records:
            return cls()

        df = pd.DataFrame(records)
        if 'index' in df.columns:
            df.set_index('index', inplace=True)

        coords = df.to_dict('index')
        return cls(coordinates=coords)

    @classmethod
    def from_dict_format(cls, data: Dict[str, Dict[str, float]]) -> 'CoordinateData':
        """Create from dictionary format."""
        return cls(coordinates=data)


class AnalysisData:
    """Analysis results data structure."""

    def __init__(self, metadata: Dict[str, Any] = None, operators: Dict[str, Any] = None,
                 datafields: Dict[str, Any] = None):
        self.metadata = metadata or {}
        self.operators = operators or {}
        self.datafields = datafields or {}

    def get_total_alphas(self) -> int:
        """Get total number of alphas."""
        return self.metadata.get('total_alphas', 0)

    def get_filter_info(self) -> Dict[str, Any]:
        """Get applied filter information."""
        return self.metadata.get('filters', {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format."""
        return {
            'metadata': self.metadata,
            'operators': self.operators,
            'datafields': self.datafields
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisData':
        """Create from legacy analysis data format."""
        return cls(
            metadata=data.get('metadata', {}),
            operators=data.get('operators', {}),
            datafields=data.get('datafields', {})
        )


class DashboardState:
    """Simple dashboard state container for testing."""

    def __init__(self):
        self.analysis_data = AnalysisData()
        self.preloaded_analysis_data = AnalysisData()
        self.analysis_filters = AnalysisFilters()
        self.all_region_data = {}
        self.available_regions = []
        self.selected_clustering_region = None
        self.analysis_ops_available = False

    def get_legacy_stores(self) -> Dict[str, Any]:
        """Get all data in legacy store format for backward compatibility."""
        return {
            'analysis-data': self.analysis_data.to_dict(),
            'preloaded-analysis-data': self.preloaded_analysis_data.to_dict(),
            'analysis-filters': self.analysis_filters.to_legacy_dict(),
            'available-regions': self.available_regions,
            'selected-clustering-region': self.selected_clustering_region,
            'analysis-ops': {'available': self.analysis_ops_available},
        }


class ViewState:
    """UI view state management."""

    def __init__(self, operators_view_mode: str = 'top20', datafields_view_mode: str = 'top20',
                 selected_alpha: str = None, selected_clustering_region: str = None,
                 distance_metric: str = 'euclidean', clustering_method: str = 'mds'):
        self.operators_view_mode = operators_view_mode
        self.datafields_view_mode = datafields_view_mode
        self.selected_alpha = selected_alpha
        self.selected_clustering_region = selected_clustering_region
        self.distance_metric = distance_metric
        self.clustering_method = clustering_method

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format."""
        return {
            'operators_view_mode': self.operators_view_mode,
            'datafields_view_mode': self.datafields_view_mode,
            'selected_alpha': self.selected_alpha,
            'selected_clustering_region': self.selected_clustering_region,
            'distance_metric': self.distance_metric,
            'clustering_method': self.clustering_method
        }


class HighlightState:
    """Alpha highlighting state."""

    def __init__(self, operator_highlighted_alphas: List[str] = None,
                 datafield_highlighted_alphas: List[str] = None,
                 available_operators: List[str] = None,
                 available_datafields: List[str] = None):
        self.operator_highlighted_alphas = operator_highlighted_alphas or []
        self.datafield_highlighted_alphas = datafield_highlighted_alphas or []
        self.available_operators = available_operators or []
        self.available_datafields = available_datafields or []

    def get_all_highlighted_alphas(self) -> List[str]:
        """Get all highlighted alphas from all sources."""
        all_highlighted = set()
        all_highlighted.update(self.operator_highlighted_alphas)
        all_highlighted.update(self.datafield_highlighted_alphas)
        return list(all_highlighted)

    def clear_highlights(self):
        """Clear all highlights."""
        self.operator_highlighted_alphas.clear()
        self.datafield_highlighted_alphas.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format."""
        return {
            'operator_highlighted_alphas': self.operator_highlighted_alphas,
            'datafield_highlighted_alphas': self.datafield_highlighted_alphas,
            'available_operators': self.available_operators,
            'available_datafields': self.available_datafields
        }


class ClusteringData:
    """Complete clustering data for a region."""

    def __init__(self, region: str = '', alpha_count: int = 0, timestamp: str = None, **kwargs):
        self.region = region
        self.alpha_count = alpha_count
        self.timestamp = timestamp

        # Store all other data as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_current_mds_data(self, distance_metric: str = 'euclidean') -> List[Dict[str, Any]]:
        """Get MDS data based on distance metric."""
        # Convert from legacy format to records
        metric_map = {
            'simple': getattr(self, 'mds_coords_simple', {}),
            'euclidean': getattr(self, 'mds_coords_euclidean', getattr(self, 'mds_coords', {})),
            'angular': getattr(self, 'mds_coords_angular', {})
        }

        data = metric_map.get(distance_metric, {})
        if data:
            df = pd.DataFrame.from_dict(data, orient='index')
            return df.reset_index().to_dict('records') if not df.empty else []
        return []

    def to_legacy_stores(self) -> Dict[str, Any]:
        """Convert to legacy store format."""
        # Convert coordinate dictionaries to DataFrames and then to records
        def coords_to_records(coords_data):
            if coords_data:
                df = pd.DataFrame.from_dict(coords_data, orient='index')
                return df.reset_index().to_dict('records') if not df.empty else []
            return []

        return {
            'current-mds-data': coords_to_records(getattr(self, 'mds_coords_euclidean', getattr(self, 'mds_coords', {}))),
            'mds-data-simple': coords_to_records(getattr(self, 'mds_coords_simple', {})),
            'mds-data-euclidean': coords_to_records(getattr(self, 'mds_coords_euclidean', getattr(self, 'mds_coords', {}))),
            'mds-data-angular': coords_to_records(getattr(self, 'mds_coords_angular', {})),
            'current-tsne-data': coords_to_records(getattr(self, 'tsne_coords', {})),
            'current-umap-data': coords_to_records(getattr(self, 'umap_coords', {})),
            'current-pca-data': coords_to_records(getattr(self, 'pca_coords', {})),
            'current-pca-info': getattr(self, 'pca_info', {}),
            'current-metadata': [
                {'index': k, **v} for k, v in getattr(self, 'alpha_metadata', {}).items()
            ],
            'tsne-cluster-profiles': getattr(self, 'tsne_cluster_profiles', {}),
            'umap-cluster-profiles': getattr(self, 'umap_cluster_profiles', {}),
            'pca-cluster-profiles': getattr(self, 'pca_cluster_profiles', {}),
            'main-cluster-profiles': getattr(self, 'main_cluster_profiles', {}),
            'heatmap-data-simple': getattr(self, 'heatmap_data_simple', {}),
            'heatmap-data-euclidean': getattr(self, 'heatmap_data_euclidean', {}),
            'heatmap-data-angular': getattr(self, 'heatmap_data_angular', {}),
        }

    @classmethod
    def from_legacy_data(cls, region: str, legacy_data: Dict[str, Any]) -> 'ClusteringData':
        """Create from legacy all-region-data format."""
        return cls(
            region=region,
            alpha_count=legacy_data.get('alpha_count', 0),
            timestamp=legacy_data.get('timestamp'),
            mds_coords_simple=legacy_data.get('mds_coords_simple', {}),
            mds_coords_euclidean=legacy_data.get('mds_coords_euclidean', legacy_data.get('mds_coords', {})),
            mds_coords_angular=legacy_data.get('mds_coords_angular', {}),
            tsne_coords=legacy_data.get('tsne_coords', {}),
            umap_coords=legacy_data.get('umap_coords', {}),
            pca_coords=legacy_data.get('pca_coords', {}),
            pca_info=legacy_data.get('pca_info', {}),
            alpha_metadata=legacy_data.get('alpha_metadata', {}),
            tsne_cluster_profiles=legacy_data.get('tsne_cluster_profiles', {}),
            umap_cluster_profiles=legacy_data.get('umap_cluster_profiles', {}),
            pca_cluster_profiles=legacy_data.get('pca_cluster_profiles', {}),
            main_cluster_profiles=legacy_data.get('main_cluster_profiles', {}),
            heatmap_data_simple=legacy_data.get('heatmap_data_simple', {}),
            heatmap_data_euclidean=legacy_data.get('heatmap_data_euclidean', {}),
            heatmap_data_angular=legacy_data.get('heatmap_data_angular', {}),
        )