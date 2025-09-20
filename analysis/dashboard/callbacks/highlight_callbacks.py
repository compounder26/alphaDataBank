"""
Highlighting Callbacks

Alpha highlighting system callbacks with preserved functionality.
Extracted from visualization_server.py lines 856-943 with exact logic preservation.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic
from ..services import get_analysis_service


def register_highlighting_callbacks(app: dash.Dash):
    """
    Register alpha highlighting callbacks.

    CRITICAL: These callbacks handle the highlighting system for operators and datafields.
    Maintains exact compatibility with original visualization_server.py.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        [Output('available-operators', 'data'),
         Output('available-datafields', 'data'),
         Output('operator-highlight-selector', 'options'),
         Output('datafield-highlight-selector', 'options')],
        Input('selected-clustering-region', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_highlight_options(selected_region):
        """
        Update available operators and datafields for highlighting based on selected region.

        EXACT COPY from visualization_server.py lines 857-886
        Preserves all logic for populating highlight dropdowns.
        """
        if not selected_region:
            return [], [], [], []

        try:
            # Get analysis operations instance using service
            analysis_service = get_analysis_service()

            # Get available operators and datafields for the region
            operators = analysis_service.get_available_operators_for_region(selected_region)
            datafields = analysis_service.get_available_datafields_for_region(selected_region)

            # Format options for dropdowns
            operator_options = [{'label': op, 'value': op} for op in operators]
            datafield_options = [{'label': df, 'value': df} for df in datafields]

            return operators, datafields, operator_options, datafield_options

        except Exception as e:
            print(f"Error updating highlight options: {e}")
            return [], [], [], []

    @callback_wrapper.safe_callback(
        Output('operator-highlighted-alphas', 'data'),
        Input('operator-highlight-selector', 'value'),
        State('selected-clustering-region', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_operator_highlights(selected_operators, region):
        """
        Update list of alphas highlighted by operator selection.

        EXACT COPY from visualization_server.py lines 888-910
        Preserves all logic for finding alphas with selected operators.
        """
        if not selected_operators or not region:
            return []

        try:
            # Get analysis operations instance using service
            analysis_service = get_analysis_service()

            # Get alphas containing selected operators
            alpha_ids = analysis_service.get_alphas_containing_operators(selected_operators, region)
            return alpha_ids

        except Exception as e:
            print(f"Error updating operator highlights: {e}")
            return []

    @callback_wrapper.safe_callback(
        Output('datafield-highlighted-alphas', 'data'),
        Input('datafield-highlight-selector', 'value'),
        State('selected-clustering-region', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_datafield_highlights(selected_datafields, region):
        """
        Update list of alphas highlighted by datafield selection.

        EXACT COPY from visualization_server.py lines 911-932
        Preserves all logic for finding alphas with selected datafields.
        """
        if not selected_datafields or not region:
            return []

        try:
            # Get analysis operations instance using service
            analysis_service = get_analysis_service()

            # Get alphas containing selected datafields
            alpha_ids = analysis_service.get_alphas_containing_datafields(selected_datafields, region)
            return alpha_ids

        except Exception as e:
            print(f"Error updating datafield highlights: {e}")
            return []

    @callback_wrapper.safe_callback(
        [Output('operator-highlight-selector', 'value'),
         Output('datafield-highlight-selector', 'value')],
        Input('clear-highlights-btn', 'n_clicks'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def clear_highlights(n_clicks):
        """
        Clear both operator and datafield highlights.

        EXACT COPY from visualization_server.py lines 934-943
        Preserves all clearing logic.
        """
        if n_clicks:
            return [], []
        return dash.no_update, dash.no_update


def register_analysis_subtab_callbacks(app: dash.Dash):
    """
    Register analysis subtab callbacks.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        Output('analysis-subtab-content', 'children'),
        [Input('analysis-subtabs', 'value'),
         Input('analysis-data', 'data'),
         Input('operators-view-mode', 'data'),
         Input('datafields-view-mode', 'data')],
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def render_analysis_subtab_content(active_subtab, analysis_data, operators_view_mode, datafields_view_mode):
        """
        Render analysis subtab content.

        EXACT COPY from visualization_server.py lines 1035-1055
        Preserves all subtab routing logic.
        """
        if not analysis_data:
            return html.Div("Apply filters to load analysis data", className="text-muted text-center p-4")

        # Import content creation functions
        from ..layouts.analysis_tab import (
            create_operators_content, create_datafields_content,
            create_neutralization_content, create_cross_analysis_content
        )

        if active_subtab == 'operators-subtab':
            return create_operators_content(analysis_data, operators_view_mode)
        elif active_subtab == 'datafields-subtab':
            return create_datafields_content(analysis_data, datafields_view_mode)
        elif active_subtab == 'neutralization-subtab':
            return create_neutralization_content(analysis_data)
        elif active_subtab == 'cross-subtab':
            return create_cross_analysis_content(analysis_data)

        return html.Div()

    @callback_wrapper.safe_callback(
        Output('operators-view-mode', 'data'),
        Input('operators-view-selector', 'value'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_operators_view_mode(selected_mode):
        """
        Update operators view mode.

        EXACT COPY from visualization_server.py lines 1091-1096
        """
        return selected_mode

    @callback_wrapper.safe_callback(
        Output('datafields-view-mode', 'data'),
        Input('datafields-view-selector', 'value'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_datafields_view_mode(selected_mode):
        """
        Update datafields view mode.

        EXACT COPY from visualization_server.py lines 1379-1384
        """
        return selected_mode

    @callback_wrapper.safe_callback(
        Output('treemap-sidebar-info', 'children'),
        [Input('datafields-view-selector', 'value'),
         Input('analysis-subtabs', 'value'),
         Input('analysis-data', 'data')],
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_treemap_sidebar_info(view_mode, active_subtab, analysis_data):
        """
        Conditionally show treemap information in sidebar ONLY in datafields tab.

        UPDATED from visualization_server.py lines 1387-1395 to fix sidebar placement
        """
        # Only show if in datafields subtab AND treemap view mode
        if active_subtab == 'datafields-subtab' and view_mode == 'treemap' and analysis_data:
            # Import here to avoid circular imports
            from ..layouts.analysis_tab import get_dataset_treemap_sidebar_info
            return get_dataset_treemap_sidebar_info(analysis_data)
        return []


# Export for easy registration
__all__ = ['register_highlighting_callbacks', 'register_analysis_subtab_callbacks']