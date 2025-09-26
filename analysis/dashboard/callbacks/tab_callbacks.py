"""
Tab and content rendering callbacks.

These callbacks are relatively independent and safe to extract first.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic

def register_tab_callbacks(app: dash.Dash, create_analysis_tab_content: callable,
                          create_clustering_tab_content: callable):
    """
    Register all tab-related callbacks.

    Args:
        app: Dash application instance
        create_analysis_tab_content: Function to create analysis tab content
        create_clustering_tab_content: Function to create clustering tab content
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value'),
        State('analysis-ops', 'data')
    )
    @preserve_prevent_update_logic
    def render_tab_content(active_tab, analysis_ops_data):
        """
        Render tab content based on active tab.

        Preserves all logic and error handling.
        """
        if active_tab == 'analysis-tab':
            if not analysis_ops_data.get('available', False):
                return dbc.Alert([
                    html.H4("Analysis Unavailable", className="alert-heading"),
                    html.P("The analysis system could not be initialized. Please check:"),
                    html.Ul([
                        html.Li("Dynamic operators/datafields data is available (run with --renew if needed)"),
                        html.Li("Database connection is working"),
                        html.Li("Analysis schema is initialized")
                    ])
                ], color="warning")

            return create_analysis_tab_content()

        elif active_tab == 'clustering-tab':
            return create_clustering_tab_content()

        return html.Div("Select a tab to begin analysis")

def register_content_callbacks(app: dash.Dash):
    """
    Register content-specific callbacks that don't depend on data stores.

    These are safe to extract as they have minimal dependencies.
    """
    # Add any other tab-related callbacks here
    # that don't depend on complex data flows
    pass

# Export for easy registration
__all__ = ['register_tab_callbacks', 'register_content_callbacks']