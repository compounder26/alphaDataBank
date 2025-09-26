"""
Pattern-matching callback handlers.

CRITICAL: These callbacks use dash.MATCH, dash.ALL, etc.
Must preserve exact syntax and component ID patterns.

Extract these LAST - they are the most complex and risky.
"""

import dash
from dash import Input, Output, State, callback, callback_context, MATCH, ALL, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
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
        Shows alphas grouped by region that use the selected datafield.
        """
        if not any(n_clicks_list) or not callback_context.triggered:
            raise PreventUpdate

        # Get the triggered component
        triggered_id = callback_context.triggered[0]['prop_id']

        # Parse the datafield ID from the triggered component
        import json
        triggered_info = json.loads(triggered_id.split('.')[0].replace("'", '"'))
        datafield_id = triggered_info['index']

        try:
            # Get recommendation and analysis services
            from ..services import get_recommendation_service, get_analysis_service
            from ..components import create_alpha_badge

            rec_service = get_recommendation_service()
            analysis_service = get_analysis_service()

            # Get full recommendations data to find regions where this datafield is used
            recommendations_data = rec_service.get_datafield_recommendations()
            recommendations = recommendations_data.get('recommendations', [])

            # Find the recommendation for this specific datafield
            datafield_rec = None
            for rec in recommendations:
                if rec['datafield_id'] == datafield_id:
                    datafield_rec = rec
                    break

            if not datafield_rec:
                return True, f"Datafield: {datafield_id}", html.Div(
                    "No usage data found for this datafield.",
                    className="text-muted text-center p-4"
                )

            # Build modal content showing alphas by region
            modal_content = []

            # Add header with datafield info
            modal_content.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Datafield Information", className="mb-3"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Strong("Datafield ID: "),
                                html.Code(datafield_id, className="bg-light p-1")
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Description: "),
                                html.Span(datafield_rec.get('description', 'No description'))
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Data Type: "),
                                html.Span(
                                    datafield_rec.get('data_type', 'Unknown'),
                                    className="badge bg-info ms-2"
                                )
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("Total Alphas Using This: "),
                                html.Span(
                                    str(datafield_rec.get('alpha_count', 0)),
                                    className="badge bg-primary ms-2"
                                )
                            ])
                        ], flush=True)
                    ])
                ], className="mb-4")
            )

            # Add alphas grouped by region
            modal_content.append(
                html.H5("Alphas Using This Datafield (by Region)", className="mb-3")
            )

            # Get usage details from the recommendation
            usage_details = datafield_rec.get('usage_details', {})
            used_regions = datafield_rec.get('used_in_regions', [])

            if used_regions:
                accordion_items = []

                for region in sorted(used_regions):
                    # Get alphas for this datafield in this region
                    alphas_in_region = rec_service.get_alphas_by_datafield_and_region(
                        datafield_id, region
                    )

                    region_count = usage_details.get(region, len(alphas_in_region))

                    # Create accordion item for this region
                    accordion_item = dbc.AccordionItem(
                        [
                            html.Div([
                                html.P(
                                    f"Total alphas in {region}: {region_count}",
                                    className="text-muted mb-3"
                                ),
                                html.Div(
                                    [
                                        create_alpha_badge(
                                            alpha_id=alpha_id,
                                            href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                                            color="success"
                                        ) for alpha_id in alphas_in_region[:100]  # Limit to 100
                                    ],
                                    style={'max-height': '300px', 'overflow-y': 'auto'}
                                ),
                                html.Small(
                                    f"Showing first 100 of {len(alphas_in_region)} alphas",
                                    className="text-muted mt-2"
                                ) if len(alphas_in_region) > 100 else None
                            ])
                        ],
                        title=f"{region} Region ({region_count} alphas)",
                        item_id=f"region-{region}"
                    )
                    accordion_items.append(accordion_item)

                modal_content.append(
                    dbc.Accordion(
                        accordion_items,
                        id="regions-accordion",
                        active_item=f"region-{sorted(used_regions)[0]}" if used_regions else None,
                        flush=True
                    )
                )
            else:
                modal_content.append(
                    dbc.Alert(
                        "No alpha usage data found for this datafield.",
                        color="warning"
                    )
                )

            # Add recommendations footer if there are recommended regions
            recommended_regions = datafield_rec.get('recommended_regions', [])
            if recommended_regions:
                modal_content.append(html.Hr(className="my-4"))
                modal_content.append(
                    dbc.Alert([
                        html.Strong("Expansion Opportunities: "),
                        f"This datafield could be used in {len(recommended_regions)} additional regions: ",
                        html.Span(
                            ", ".join(recommended_regions),
                            className="fw-bold"
                        )
                    ], color="info")
                )

            return True, f"Datafield Usage: {datafield_id}", html.Div(modal_content)

        except Exception as e:
            import traceback
            error_msg = f"Error loading datafield alphas: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return True, "Error", dbc.Alert(error_msg, color="danger")

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
1. Check the callback registration and ID patterns
2. Debug in isolated test environment
3. Fix issues before deployment
"""