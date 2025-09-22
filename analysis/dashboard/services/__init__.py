"""
Services Package

Business logic layer for the Alpha Dashboard.
Provides clean interfaces for data operations, analysis, clustering, and recommendations.
"""

from typing import Dict, Any

from .data_service import (
    create_analysis_operations,
    reset_analysis_operations,
    get_alpha_details_for_clustering,
    load_clustering_data,
    load_all_region_data,
    get_available_regions_from_files,
    load_operators_data,
    load_tier_specific_datafields,
    validate_clustering_data,
    get_data_file_info,
    cleanup_cached_data,
    set_tier_operators_and_datafields
)

from .analysis_service import (
    AnalysisService,
    get_analysis_service,
    reset_analysis_service
)

from .clustering_service import (
    ClusteringService,
    get_clustering_service,
    reset_clustering_service
)

from .chart_service import (
    ChartService,
    get_chart_service,
    reset_chart_service
)

from .recommendation_service import (
    RecommendationService,
    get_recommendation_service,
    reset_recommendation_service
)

__all__ = [
    # Data service functions
    'create_analysis_operations',
    'reset_analysis_operations',
    'get_alpha_details_for_clustering',
    'load_clustering_data',
    'load_all_region_data',
    'get_available_regions_from_files',
    'load_operators_data',
    'load_tier_specific_datafields',
    'validate_clustering_data',
    'get_data_file_info',
    'cleanup_cached_data',
    'set_tier_operators_and_datafields',

    # Service classes and factories
    'AnalysisService',
    'get_analysis_service',
    'reset_analysis_service',

    'ClusteringService',
    'get_clustering_service',
    'reset_clustering_service',

    'ChartService',
    'get_chart_service',
    'reset_chart_service',

    'RecommendationService',
    'get_recommendation_service',
    'reset_recommendation_service'
]


def reset_all_services():
    """Reset all service instances (useful for testing)."""
    reset_analysis_service()
    reset_clustering_service()
    reset_chart_service()
    reset_recommendation_service()
    reset_analysis_operations()
    cleanup_cached_data()
    print(" All services reset and caches cleared")


def get_all_services() -> Dict[str, Any]:
    """Get all service instances for centralized access."""
    return {
        'analysis': get_analysis_service(),
        'clustering': get_clustering_service(),
        'chart': get_chart_service(),
        'recommendation': get_recommendation_service()
    }