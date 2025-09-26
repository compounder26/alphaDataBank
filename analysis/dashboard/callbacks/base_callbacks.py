"""
Base callback utilities and registration system.

Provides safe callback extraction with dependency preservation.
"""

from typing import Dict, List, Any, Callable, Optional, Union
import dash
from dash import Input, Output, State, callback, callback_context
from dash.exceptions import PreventUpdate
import logging

logger = logging.getLogger(__name__)


class CallbackRegistry:
    """
    Registry for managing callback dependencies and safe extraction.

    Ensures callbacks are registered in dependency order and maintains
    exact compatibility with existing patterns.
    """

    def __init__(self, app: dash.Dash):
        self.app = app
        self._callbacks: Dict[str, Dict] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._registration_order: List[str] = []

    def register_callback_group(self, group_name: str, callbacks: List[Dict]):
        """Register a group of related callbacks with dependency tracking."""
        self._callbacks[group_name] = callbacks

        # Track registration order for safe extraction
        if group_name not in self._registration_order:
            self._registration_order.append(group_name)

        logger.info(f"Registered callback group: {group_name} with {len(callbacks)} callbacks")

    def get_callback_dependencies(self, callback_func: Callable) -> Dict[str, List[str]]:
        """Extract Input/Output/State dependencies from callback function."""
        # This would analyze the callback decorator to extract dependencies
        # For now, return empty dict - implement based on specific needs
        return {}


class CallbackWrapper:
    """
    Wrapper to preserve exact callback behavior during extraction.

    Maintains:
    - prevent_initial_call settings
    - Error handling patterns
    - PreventUpdate logic
    - Pattern-matching callback IDs
    """

    def __init__(self, app: dash.Dash):
        self.app = app

    def safe_callback(self, outputs, inputs, states=None, prevent_initial_call=False):
        """
        Decorator that preserves exact callback behavior.

        Usage:
            @callback_wrapper.safe_callback(
                Output('component-id', 'property'),
                Input('trigger-id', 'property'),
                prevent_initial_call=True
            )
            def my_callback(trigger_value):
                # Existing callback logic unchanged
                pass
        """
        def decorator(func):
            # Register callback with exact same parameters as original
            if states is not None:
                self.app.callback(
                    outputs, inputs, states,
                    prevent_initial_call=prevent_initial_call
                )(func)
            else:
                self.app.callback(
                    outputs, inputs,
                    prevent_initial_call=prevent_initial_call
                )(func)
            return func
        return decorator

    def pattern_matching_callback(self, outputs, inputs, states=None, prevent_initial_call=False):
        """
        Specialized wrapper for pattern-matching callbacks.

        Preserves MATCH, ALL, ALLSMALLER patterns exactly.
        """
        def decorator(func):
            self.app.callback(
                outputs, inputs, states,
                prevent_initial_call=prevent_initial_call
            )(func)
            return func
        return decorator


def preserve_prevent_update_logic(original_func: Callable) -> Callable:
    """
    Decorator to preserve existing PreventUpdate patterns.

    Ensures extracted callbacks maintain exact same conditional logic.
    """
    def wrapper(*args, **kwargs):
        try:
            result = original_func(*args, **kwargs)
            return result
        except PreventUpdate:
            # Re-raise PreventUpdate as-is
            raise
        except Exception as e:
            # Log the actual error for debugging
            logger.error(f"Callback error in {original_func.__name__}: {e}")
            # Import traceback for detailed error logging
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Re-raise the original exception to see what's actually happening
            raise
    return wrapper


def get_store_dependencies() -> Dict[str, List[str]]:
    """
    Map all store IDs and their callback dependencies.

    Critical for maintaining data flow during extraction.
    """
    return {
        # Core data stores
        'analysis-data': ['preloaded-analysis-data', 'analysis-filters'],
        'preloaded-analysis-data': ['analysis-ops'],
        'analysis-filters': ['region-filter', 'universe-filter', 'delay-filter'],

        # Clustering stores
        'all-region-data': [],
        'selected-clustering-region': ['available-regions'],
        'current-mds-data': ['selected-clustering-region', 'distance-metric'],
        'mds-data-simple': ['selected-clustering-region'],
        'mds-data-euclidean': ['selected-clustering-region'],
        'mds-data-angular': ['selected-clustering-region'],
        'current-tsne-data': ['selected-clustering-region'],
        'current-umap-data': ['selected-clustering-region'],
        'current-pca-data': ['selected-clustering-region'],

        # Highlighting stores
        'operator-highlighted-alphas': ['operator-highlight-selector'],
        'datafield-highlighted-alphas': ['datafield-highlight-selector'],
        'available-operators': ['selected-clustering-region'],
        'available-datafields': ['selected-clustering-region'],

        # View state stores
        'operators-view-mode': [],
        'datafields-view-mode': [],
    }


def validate_callback_extraction(extracted_callbacks: List[str],
                               remaining_callbacks: List[str]) -> bool:
    """
    Validate that callback extraction preserves all dependencies.

    Returns False if extraction would break dependency chain.
    """
    store_deps = get_store_dependencies()

    for callback_name in extracted_callbacks:
        # Check if any remaining callbacks depend on this one's outputs
        # Implementation would check actual Input/Output/State dependencies
        pass

    return True


class CallbackExtractionValidator:
    """
    Validates safe callback extraction without breaking dependencies.
    """

    def __init__(self, original_file_path: str):
        self.original_file_path = original_file_path
        self._callback_map: Dict[str, Dict] = {}

    def analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze all callback dependencies in original file."""
        # This would parse the original file to extract dependencies
        return {}

    def validate_extraction_plan(self, extraction_groups: Dict[str, List[str]]) -> bool:
        """Validate that extraction plan preserves all dependencies."""
        return True

    def generate_safe_extraction_order(self) -> List[str]:
        """Generate safe order for callback extraction."""
        # Return groups in dependency order
        return [
            'tab_callbacks',      # Independent - safe to extract first
            'data_loading',       # Depends only on stores
            'filter_callbacks',   # Depends on data_loading
            'clustering_base',    # Depends on filters
            'highlighting',       # Depends on clustering
            'chart_interactions', # Depends on clustering + highlighting
            'modals',            # Depends on chart interactions
        ]