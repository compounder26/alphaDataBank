"""
Components Package

Reusable UI components for the Alpha Dashboard.
Provides modular, testable components for the dashboard.
"""

from .base_components import (
    create_dashboard_header,
    create_info_card,
    create_statistics_card,
    create_method_selector,
    create_view_mode_selector,
    create_alpha_badge,
    create_alpha_list_container,
    create_loading_wrapper,
    create_expandable_list,
    create_alert_message,
    create_progress_indicator,
    create_tooltip_badge,
    create_clickable_badge_with_action,
    create_summary_grid,
    create_data_type_badge,
    create_region_info_text,
    create_usage_percentage_badge,
    create_interactive_table_row,
    create_collapsible_section,
    create_legend_component,
    create_section_header,
    create_empty_state_message,
    create_action_button,
    create_dataset_statistics_panel
)

from .filters import (
    create_analysis_filters,
    create_region_filter,
    create_universe_filter,
    create_delay_filter,
    create_dates_filter,
    create_clustering_region_selector,
    create_distance_metric_selector,
    create_highlighting_filters,
    create_display_options_card,
    create_method_explanation_card,
    create_recommendation_filters,
    create_tab_selector,
    create_multi_select_dropdown,
    create_single_select_dropdown,
    create_filter_reset_button,
    create_filter_summary
)

from .charts import (
    create_main_clustering_plot,
    create_pca_loadings_plot,
    create_operators_chart_container,
    create_all_operators_chart_container,
    create_datafields_chart_container,
    create_statistics_panel,
    create_responsive_chart_grid,
    create_chart_with_loading,
    create_empty_chart_placeholder,
    create_treemap_container,
    create_chart_info_panel,
    create_cluster_statistics_panel,
    create_alpha_details_panel,
    create_clustering_info_card,
    create_chart_controls_row,
    create_chart_explanation_alert
)

from .modals import (
    create_main_detail_modal,
    create_datafield_detail_modal,
    create_operator_modal_content,
    create_datafield_modal_content,
    create_dataset_modal_content,
    create_alpha_details_modal_content,
    create_dual_alpha_modal_content,
    create_category_modal_content,
    create_neutralization_modal_content,
    create_error_modal_content
)

from .tables import (
    create_alpha_table,
    create_recommendations_table,
    create_statistics_list,
    create_scrollable_list,
    create_alpha_summary_card,
    create_breakdown_table,
    create_expandable_alpha_list,
    create_datafield_details_table,
    create_usage_breakdown_list,
    create_cluster_breakdown_list,
    create_dataset_info_list,
    create_performance_summary_table,
    create_region_alpha_breakdown,
    create_interactive_badge_list
)

__all__ = [
    # Base components
    'create_dashboard_header',
    'create_info_card',
    'create_statistics_card',
    'create_method_selector',
    'create_view_mode_selector',
    'create_alpha_badge',
    'create_alpha_list_container',
    'create_loading_wrapper',
    'create_expandable_list',
    'create_alert_message',
    'create_progress_indicator',
    'create_tooltip_badge',
    'create_clickable_badge_with_action',
    'create_summary_grid',
    'create_data_type_badge',
    'create_region_info_text',
    'create_usage_percentage_badge',
    'create_interactive_table_row',
    'create_collapsible_section',
    'create_legend_component',
    'create_section_header',
    'create_empty_state_message',
    'create_action_button',
    'create_dataset_statistics_panel',

    # Filter components
    'create_analysis_filters',
    'create_region_filter',
    'create_universe_filter',
    'create_delay_filter',
    'create_dates_filter',
    'create_clustering_region_selector',
    'create_distance_metric_selector',
    'create_highlighting_filters',
    'create_display_options_card',
    'create_method_explanation_card',
    'create_recommendation_filters',
    'create_tab_selector',
    'create_multi_select_dropdown',
    'create_single_select_dropdown',
    'create_filter_reset_button',
    'create_filter_summary',

    # Chart components
    'create_main_clustering_plot',
    'create_pca_loadings_plot',
    'create_operators_chart_container',
    'create_all_operators_chart_container',
    'create_datafields_chart_container',
    'create_statistics_panel',
    'create_responsive_chart_grid',
    'create_chart_with_loading',
    'create_empty_chart_placeholder',
    'create_treemap_container',
    'create_chart_info_panel',
    'create_cluster_statistics_panel',
    'create_alpha_details_panel',
    'create_clustering_info_card',
    'create_chart_controls_row',
    'create_chart_explanation_alert',

    # Modal components
    'create_main_detail_modal',
    'create_datafield_detail_modal',
    'create_operator_modal_content',
    'create_datafield_modal_content',
    'create_dataset_modal_content',
    'create_alpha_details_modal_content',
    'create_dual_alpha_modal_content',
    'create_category_modal_content',
    'create_neutralization_modal_content',
    'create_error_modal_content',

    # Table components
    'create_alpha_table',
    'create_recommendations_table',
    'create_statistics_list',
    'create_scrollable_list',
    'create_alpha_summary_card',
    'create_breakdown_table',
    'create_expandable_alpha_list',
    'create_datafield_details_table',
    'create_usage_breakdown_list',
    'create_cluster_breakdown_list',
    'create_dataset_info_list',
    'create_performance_summary_table',
    'create_region_alpha_breakdown',
    'create_interactive_badge_list'
]