"""
Layouts Package

Page and tab layout structures for the Alpha Dashboard.
Provides organized layout components extracted from the original monolithic file.
"""

from .main_layout import (
    create_main_layout,
    create_data_stores,
    create_empty_tab_content,
    create_unavailable_analysis_content,
    get_tab_content_id_mapping
)

from .analysis_tab import (
    create_analysis_tab_content,
    create_operators_view_selector,
    create_datafields_view_selector,
    create_operators_usage_summary,
    create_operators_statistics_panel,
    create_datafields_grid_layout,
    create_analysis_summary_items,
    create_no_analysis_data_message
)

from .clustering_tab import (
    create_clustering_tab_content,
    create_visualization_method_card,
    create_clustering_explanation_text,
    get_clustering_method_options,
    get_distance_metric_options,
    should_show_distance_metric_selector,
    should_show_pca_loadings
)

from .recommendations_tab import (
    create_recommendations_tab_content,
    create_recommendations_summary_card,
    create_recommendations_display,
    create_recommendation_error_display,
    create_no_recommendations_message,
    create_neutralization_content
)

__all__ = [
    # Main layout
    'create_main_layout',
    'create_data_stores',
    'create_empty_tab_content',
    'create_unavailable_analysis_content',
    'get_tab_content_id_mapping',

    # Analysis tab
    'create_analysis_tab_content',
    'create_operators_view_selector',
    'create_datafields_view_selector',
    'create_operators_usage_summary',
    'create_operators_statistics_panel',
    'create_datafields_grid_layout',
    'create_analysis_summary_items',
    'create_no_analysis_data_message',

    # Clustering tab
    'create_clustering_tab_content',
    'create_visualization_method_card',
    'create_clustering_explanation_text',
    'get_clustering_method_options',
    'get_distance_metric_options',
    'should_show_distance_metric_selector',
    'should_show_pca_loadings',

    # Recommendations tab
    'create_recommendations_tab_content',
    'create_recommendations_summary_card',
    'create_recommendations_display',
    'create_recommendation_error_display',
    'create_no_recommendations_message',
    'create_neutralization_content'
]