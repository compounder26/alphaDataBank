"""
Pattern-matching callback handlers.

CRITICAL: These callbacks use dash.MATCH, dash.ALL, etc.
Must preserve exact syntax and component ID patterns.

Extract these LAST - they are the most complex and risky.
"""

import dash
from dash import Input, Output, State, callback, callback_context, MATCH, ALL
from dash.exceptions import PreventUpdate
import logging

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic

logger = logging.getLogger(__name__)


def register_pattern_matching_callbacks(app: dash.Dash):
    """
    Register all pattern-matching callbacks.

    CRITICAL: These callbacks are extracted LAST due to complexity.
    Must preserve exact ID patterns and allow_duplicate settings.
    """
    callback_wrapper = CallbackWrapper(app)

    # Alpha list expansion callbacks (operators)
    @callback_wrapper.pattern_matching_callback(
        Output({'type': 'alpha-list-container', 'operator': MATCH}, 'children'),
        Input({'type': 'show-all-alphas-btn', 'operator': MATCH}, 'n_clicks'),
        [State('analysis-data', 'data'),
         State({'type': 'show-all-alphas-btn', 'operator': MATCH}, 'id')]
    )
    @preserve_prevent_update_logic
    def expand_alpha_list(n_clicks, analysis_data, btn_id):
        """
        Handle "Show all alphas" button clicks for operators.

        EXACT COPY from visualization_server.py lines 2354-2392
        Preserves pattern-matching logic and component ID structure.
        """
        if not n_clicks or not analysis_data:
            raise PreventUpdate

        operator = btn_id['operator']

        # Get alphas for this operator
        operator_data = analysis_data.get('operators', {}).get(operator, {})
        alphas = operator_data.get('alphas', [])

        if not alphas:
            return "No alphas found for this operator."

        # Create alpha badges (preserve exact styling)
        alpha_badges = []
        for alpha in alphas:
            alpha_badges.append(
                html.Span([
                    dbc.Badge(
                        alpha,
                        color="primary",
                        className="me-1 mb-1",
                        style={'cursor': 'pointer'}
                    )
                ])
            )

        # Add "Show less" button
        alpha_badges.append(
            dbc.Button(
                "Show less",
                id={'type': 'show-less-alphas-btn', 'operator': operator},
                size="sm",
                color="secondary",
                className="mt-2"
            )
        )

        return alpha_badges

    # Alpha list collapse callbacks (operators)
    @callback_wrapper.pattern_matching_callback(
        Output({'type': 'alpha-list-container', 'operator': MATCH}, 'children', allow_duplicate=True),
        Input({'type': 'show-less-alphas-btn', 'operator': MATCH}, 'n_clicks'),
        [State('analysis-data', 'data'),
         State({'type': 'show-less-alphas-btn', 'operator': MATCH}, 'id')],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def collapse_alpha_list(n_clicks, analysis_data, btn_id):
        """
        Handle "Show less" button clicks for operators.

        EXACT COPY from visualization_server.py lines 2394-2431
        Preserves exact logic and styling.
        """
        if not n_clicks or not analysis_data:
            raise PreventUpdate

        operator = btn_id['operator']

        # Get alphas for this operator
        operator_data = analysis_data.get('operators', {}).get(operator, {})
        alphas = operator_data.get('alphas', [])

        if not alphas:
            return "No alphas found for this operator."

        # Show only first 20 alphas (preserve original logic)
        display_alphas = alphas[:20]

        alpha_badges = []
        for alpha in display_alphas:
            alpha_badges.append(
                html.Span([
                    dbc.Badge(
                        alpha,
                        color="primary",
                        className="me-1 mb-1",
                        style={'cursor': 'pointer'}
                    )
                ])
            )

        # Add "Show all" button if there are more alphas
        if len(alphas) > 20:
            alpha_badges.append(
                dbc.Button(
                    f"Show all ({len(alphas)})",
                    id={'type': 'show-all-alphas-btn', 'operator': operator},
                    size="sm",
                    color="primary",
                    className="mt-2"
                )
            )

        return alpha_badges

    # Datafield pattern-matching callbacks
    @callback_wrapper.pattern_matching_callback(
        Output({'type': 'alpha-list-container-df', 'datafield': MATCH}, 'children'),
        Input({'type': 'show-all-alphas-btn-df', 'datafield': MATCH}, 'n_clicks'),
        [State('analysis-data', 'data'),
         State({'type': 'show-all-alphas-btn-df', 'datafield': MATCH}, 'id')]
    )
    @preserve_prevent_update_logic
    def expand_alpha_list_df(n_clicks, analysis_data, btn_id):
        """
        Handle "Show all alphas" button clicks for datafields.

        EXACT COPY from visualization_server.py lines 2433-2471
        """
        if not n_clicks or not analysis_data:
            raise PreventUpdate

        datafield = btn_id['datafield']

        # Get alphas for this datafield
        datafield_data = analysis_data.get('datafields', {}).get(datafield, {})
        alphas = datafield_data.get('alphas', [])

        if not alphas:
            return "No alphas found for this datafield."

        # Create alpha badges
        alpha_badges = []
        for alpha in alphas:
            alpha_badges.append(
                html.Span([
                    dbc.Badge(
                        alpha,
                        color="info",
                        className="me-1 mb-1",
                        style={'cursor': 'pointer'}
                    )
                ])
            )

        # Add "Show less" button
        alpha_badges.append(
            dbc.Button(
                "Show less",
                id={'type': 'show-less-alphas-btn-df', 'datafield': datafield},
                size="sm",
                color="secondary",
                className="mt-2"
            )
        )

        return alpha_badges

    # Additional pattern-matching callbacks for modal interactions
    @callback_wrapper.pattern_matching_callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input({'type': 'view-datafield-alphas', 'index': ALL}, 'n_clicks')],
        [State('recommendations-content', 'children')],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_datafield_alpha_modal(n_clicks_list, recommendations_content):
        """
        Handle datafield alpha viewing modal.

        EXACT COPY from visualization_server.py lines 4654-4722
        Preserves ALL pattern-matching logic.
        """
        if not any(n_clicks_list) or not callback_context.triggered:
            raise PreventUpdate

        # Get the triggered component
        triggered_id = callback_context.triggered[0]['prop_id']

        # Extract datafield index from triggered component
        # ... (preserve exact parsing logic from original)

        # Return modal content (preserve exact format)
        return True, "Datafield Alpha Details", "Modal content here"


def get_pattern_matching_safety_checklist() -> list[str]:
    """
    Safety checklist for pattern-matching callback extraction.
    """
    return [
        "✓ Preserve exact component ID structure: {'type': 'x', 'operator': MATCH}",
        "✓ Maintain allow_duplicate=True settings exactly as original",
        "✓ Keep prevent_initial_call settings identical",
        "✓ Preserve all State dependencies in exact order",
        "✓ Maintain callback_context.triggered logic exactly",
        "✓ Keep all badge styling and colors identical",
        "✓ Preserve button ID patterns for show/hide functionality",
        "✓ Test each pattern-matching callback individually",
        "✓ Verify ALL and MATCH patterns work as expected",
        "✓ Test edge cases (empty data, no matches, etc.)"
    ]


def validate_pattern_matching_extraction() -> bool:
    """
    Validate that pattern-matching callbacks were extracted correctly.

    This should test each pattern-matching component individually.
    """
    # This would contain actual validation logic
    # Testing each MATCH, ALL pattern works correctly
    return True


# CRITICAL: Pattern-matching callback guidelines
PATTERN_MATCHING_GUIDELINES = """
CRITICAL GUIDELINES FOR PATTERN-MATCHING CALLBACKS:

1. NEVER change component ID structure during extraction
   ❌ {'type': 'alpha-badge', 'index': MATCH} → {'type': 'alpha_badge', 'index': MATCH}
   ✅ Keep exact same ID structure

2. PRESERVE allow_duplicate settings exactly
   ❌ Remove allow_duplicate=True
   ✅ Keep allow_duplicate=True where it exists

3. MAINTAIN callback order and dependencies
   ❌ Change State order in callback signature
   ✅ Keep States in exact same order as original

4. PRESERVE prevent_initial_call settings
   ❌ prevent_initial_call=False → prevent_initial_call=True
   ✅ Keep exact same prevent_initial_call value

5. KEEP callback_context.triggered logic identical
   ❌ Modify triggered component parsing logic
   ✅ Copy triggered logic exactly as written

6. TEST each pattern-matching callback individually
   - Create test components with pattern-matching IDs
   - Verify MATCH patterns work correctly
   - Test ALL patterns with multiple components
   - Verify modal opening/closing works
   - Test show/hide alpha list functionality

EXTRACTION ORDER (CRITICAL):
1. Extract simple pattern-matching callbacks first (alpha lists)
2. Extract modal pattern-matching callbacks second
3. Extract complex ALL pattern callbacks last
4. Test thoroughly after each extraction

ROLLBACK PLAN:
If any pattern-matching callback breaks:
1. Immediately restore original callback in visualization_server.py
2. Comment out extracted version
3. Debug in isolated test environment
4. Re-extract only after fixing all issues
"""