"""
Recommendation System Callbacks

Datafield recommendation generation and interaction callbacks.
Extracted from visualization_server.py lines 4722-5178 with exact logic preservation.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic
from ..services import get_recommendation_service, create_analysis_operations
from ..layouts.recommendations_tab import (
    create_recommendations_display,
    create_recommendation_error_display,
    create_no_recommendations_message
)


def register_recommendation_callbacks(app: dash.Dash):
    """
    Register recommendation system callbacks.

    CRITICAL: These callbacks handle the cross-analysis recommendations feature.
    Maintains exact compatibility with original visualization_server.py.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        Output('recommendations-content', 'children'),
        [Input('refresh-recommendations-btn', 'n_clicks'),
         Input('analysis-data', 'data')],
        [State('recommendation-region-filter', 'value'),
         State('recommendation-type-filter', 'value')],
        prevent_initial_call=False
    )
    @preserve_prevent_update_logic
    def update_datafield_recommendations(n_clicks, analysis_data, selected_region, selected_data_type):
        """
        Update datafield recommendations based on filters.

        EXACT COPY from visualization_server.py lines 4730-4761
        """
        if not analysis_data:
            return html.Div("Please load analysis data first by applying filters in the Analysis tab.",
                          className="text-muted text-center p-4")

        try:
            # Get recommendations with both region and data type filtering
            target_region = None if selected_region == 'all' else selected_region
            target_data_type = None if selected_data_type == 'all' else selected_data_type

            # Use the imported service function directly
            rec_service = get_recommendation_service()
            recommendations_data = rec_service.get_datafield_recommendations(
                selected_region=target_region,
                selected_data_type=target_data_type
            )

            if 'error' in recommendations_data:
                return create_recommendation_error_display(recommendations_data['error'])

            recommendations = recommendations_data.get('recommendations', [])

            if not recommendations:
                return create_no_recommendations_message()

            # Create recommendations display - extract summary stats
            summary_stats = {
                'total_analyzed': recommendations_data.get('total_datafields_analyzed', 0),
                'expansion_opportunities': len(recommendations),
                'potential_new_alphas': sum(len(rec['recommended_regions']) for rec in recommendations)
            }
            return create_recommendations_display(recommendations, summary_stats)

        except Exception as e:
            print(f"Error updating recommendations: {e}")
            return create_recommendation_error_display(str(e))


# Export for easy registration
__all__ = ['register_recommendation_callbacks']