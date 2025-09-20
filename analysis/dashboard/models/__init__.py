"""
Dashboard Models

Data models and state management for the visualization dashboard.
Provides both modern typed interfaces and backward compatibility with legacy stores.
"""

# Use simple Python classes (no external dependencies)
from .simple_state_models import (
    AnalysisFilters,
    CoordinateData,
    AnalysisData,
    DashboardState,
    ViewState,
    HighlightState,
    ClusteringData
)

from .state_manager import (
    StateManager,
    get_state_manager,
    initialize_state_manager,
    reset_state_manager
)

from .transition_utils import (
    CallbackTransition,
    StoreToStateMapper,
    create_store_sync_callback,
    migrate_callback_to_state_manager,
    validate_state_consistency,
    debug_state_summary
)

__all__ = [
    # State Models
    'AnalysisFilters',
    'CoordinateData',
    'ClusteringData',
    'AnalysisData',
    'ViewState',
    'HighlightState',
    'DashboardState',

    # State Manager
    'StateManager',
    'get_state_manager',
    'initialize_state_manager',
    'reset_state_manager',

    # Transition Utilities
    'CallbackTransition',
    'StoreToStateMapper',
    'create_store_sync_callback',
    'migrate_callback_to_state_manager',
    'validate_state_consistency',
    'debug_state_summary'
]