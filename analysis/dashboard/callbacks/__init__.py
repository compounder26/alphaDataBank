"""
Callbacks Package

Organized callback system for the Alpha Dashboard.
Provides safe, phased callback registration with dependency management.
"""

from .base_callbacks import CallbackRegistry, CallbackWrapper
from .tab_callbacks import register_tab_callbacks, register_content_callbacks
from .data_callbacks import register_data_loading_callbacks, register_clustering_region_callbacks
from .clustering_callbacks import register_clustering_callbacks, register_clustering_ui_callbacks
from .highlight_callbacks import register_highlighting_callbacks, register_analysis_subtab_callbacks
from .chart_callbacks import register_chart_interaction_callbacks
from .main_plot_callback import register_main_plot_callback
from .modal_callbacks import register_modal_callbacks, register_close_modal_callbacks
from .recommendation_callbacks import register_recommendation_callbacks

# Import pattern-matching callbacks (these are handled separately due to complexity)
try:
    from .pattern_matching_callbacks import register_pattern_matching_callbacks
    PATTERN_MATCHING_AVAILABLE = True
except ImportError:
    PATTERN_MATCHING_AVAILABLE = False
    register_pattern_matching_callbacks = None


def register_all_callbacks(app, layout_functions=None):
    """
    Register all callbacks for the dashboard.

    CRITICAL: Follows the safe extraction phases to maintain functionality.

    Args:
        app: Dash application instance
        layout_functions: Dictionary of layout creation functions
    """
    print("= Registering dashboard callbacks...")

    # Phase 1: Tab callbacks (SAFE - no dependencies)
    if layout_functions:
        register_tab_callbacks(
            app,
            layout_functions.get('create_analysis_tab_content'),
            layout_functions.get('create_clustering_tab_content')
        )
    register_content_callbacks(app)
    print(" Phase 1: Tab callbacks registered")

    # Phase 2: Data loading callbacks (SAFE - minimal dependencies)
    register_data_loading_callbacks(app)
    register_clustering_region_callbacks(app)
    print(" Phase 2: Data loading callbacks registered")

    # Phase 3: Clustering callbacks (SAFE - depend on data)
    register_clustering_callbacks(app)
    register_clustering_ui_callbacks(app)
    print(" Phase 3: Clustering callbacks registered")

    # Phase 4: Highlighting callbacks (MODERATE RISK - depend on clustering)
    register_highlighting_callbacks(app)
    register_analysis_subtab_callbacks(app)
    print(" Phase 4: Highlighting and analysis callbacks registered")

    # Phase 5: Chart interaction callbacks (MODERATE RISK - depend on highlighting)
    register_chart_interaction_callbacks(app)
    register_main_plot_callback(app)
    register_modal_callbacks(app)
    register_close_modal_callbacks(app)
    register_recommendation_callbacks(app)
    print(" Phase 5: Chart interaction callbacks registered")

    # Phase 6: Pattern-matching callbacks (HIGH RISK - extract last)
    if PATTERN_MATCHING_AVAILABLE:
        try:
            register_pattern_matching_callbacks(app)
            print(" Phase 6: Pattern-matching callbacks registered")
        except Exception as e:
            print(f"� Pattern-matching callbacks failed to register: {e}")
    else:
        print("� Pattern-matching callbacks not available - will use original implementations")

        pass  # All callbacks registered successfully


def get_callback_registration_status():
    """
    Get status of callback registration for debugging.

    Returns:
        Dictionary with registration status
    """
    return {
        'tab_callbacks': True,
        'data_callbacks': True,
        'clustering_callbacks': True,
        'highlighting_callbacks': True,
        'chart_callbacks': True,
        'pattern_matching_callbacks': PATTERN_MATCHING_AVAILABLE
    }


def register_callbacks_by_phase(app, phase: int, **kwargs):
    """
    Register callbacks by specific phase for testing.

    Args:
        app: Dash application instance
        phase: Phase number (1-6)
        **kwargs: Additional arguments for specific phases
    """
    if phase == 1:
        register_tab_callbacks(app, **kwargs)
        register_content_callbacks(app)
    elif phase == 2:
        register_data_loading_callbacks(app)
        register_clustering_region_callbacks(app)
    elif phase == 3:
        register_clustering_callbacks(app)
        register_clustering_ui_callbacks(app)
    elif phase == 4:
        register_highlighting_callbacks(app)
        register_analysis_subtab_callbacks(app)
    elif phase == 5:
        register_chart_interaction_callbacks(app)
    elif phase == 6 and PATTERN_MATCHING_AVAILABLE:
        register_pattern_matching_callbacks(app)
    else:
        raise ValueError(f"Invalid phase: {phase}")


__all__ = [
    'CallbackRegistry',
    'CallbackWrapper',
    'register_all_callbacks',
    'register_callbacks_by_phase',
    'get_callback_registration_status',

    # Individual registration functions
    'register_tab_callbacks',
    'register_content_callbacks',
    'register_data_loading_callbacks',
    'register_clustering_region_callbacks',
    'register_clustering_callbacks',
    'register_clustering_ui_callbacks',
    'register_highlighting_callbacks',
    'register_analysis_subtab_callbacks',
    'register_chart_interaction_callbacks'
]