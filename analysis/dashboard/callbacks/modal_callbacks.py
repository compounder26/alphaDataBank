"""
Modal Interaction Callbacks

Modal handling callbacks for chart clicks and detail displays.
"""

import dash
from dash import html, dcc, Input, Output, State, callback, MATCH, ALL
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic
from ..components import (
    create_operator_modal_content, create_datafield_modal_content,
    create_dataset_modal_content, create_category_modal_content,
    create_neutralization_modal_content, create_error_modal_content,
    create_alpha_badge
)
from ..services import get_analysis_service, get_recommendation_service

def register_modal_callbacks(app: dash.Dash):
    """
    Register all modal interaction callbacks.

    CRITICAL: These callbacks handle chart clicks and modal content generation.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        [Output("detail-modal", "is_open"),
         Output("detail-modal-title", "children"),
         Output("detail-modal-body", "children")],
        [Input("operators-chart", "clickData"),
         Input("detail-modal-close", "n_clicks")],
        [State("detail-modal", "is_open"),
         State("analysis-data", "data")],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_operator_click(operators_chart_click, close_clicks, is_open, analysis_data):
        """
        Handle operator chart clicks to show modal.

        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "detail-modal-close":
            return False, "", ""

        if trigger_id == "operators-chart" and operators_chart_click:
            try:
                operator = operators_chart_click['points'][0]['y']
                count = operators_chart_click['points'][0]['x']

                # Get alphas using this operator
                operators_data = analysis_data.get('operators', {})
                unique_usage = operators_data.get('unique_usage', {})
                alphas_using = unique_usage.get(operator, [])

                # Create modal content
                content = create_operator_modal_content(
                    operator, count, "Top 20 Operators", alphas_using
                )

                return True, f"Operator: {operator}", content

            except (KeyError, IndexError, TypeError) as e:
                return True, "Error", create_error_modal_content(f"Error processing operator click: {str(e)}")

        raise PreventUpdate

    @callback_wrapper.safe_callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("all-operators-chart", "clickData"),
         Input("detail-modal-close", "n_clicks")],
        [State("detail-modal", "is_open"),
         State("analysis-data", "data")],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_all_operators_click(all_operators_click, close_clicks, is_open, analysis_data):
        """
        Handle all operators chart clicks.

        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "detail-modal-close":
            return False, "", ""

        if trigger_id == "all-operators-chart" and all_operators_click:
            try:
                operator = all_operators_click['points'][0]['y']
                count = all_operators_click['points'][0]['x']

                # Get alphas using this operator
                operators_data = analysis_data.get('operators', {})
                unique_usage = operators_data.get('unique_usage', {})
                alphas_using = unique_usage.get(operator, [])

                content = create_operator_modal_content(
                    operator, count, "All Operators", alphas_using
                )

                return True, f"Operator: {operator}", content

            except (KeyError, IndexError, TypeError) as e:
                return True, "Error", create_error_modal_content(f"Error processing operator click: {str(e)}")

        raise PreventUpdate

    @callback_wrapper.safe_callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("datafields-chart", "clickData"),
         Input("detail-modal-close", "n_clicks")],
        [State("detail-modal", "is_open"),
         State("analysis-data", "data")],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_datafields_chart_click(datafields_click, close_clicks, is_open, analysis_data):
        """
        Handle datafields chart clicks.

        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "detail-modal-close":
            return False, "", ""

        if trigger_id == "datafields-chart" and datafields_click:
            try:
                datafield = datafields_click['points'][0]['y']
                count = datafields_click['points'][0]['x']

                # Get alphas using this datafield
                datafields_data = analysis_data.get('datafields', {})
                unique_usage = datafields_data.get('unique_usage', {})
                alphas_using = unique_usage.get(datafield, [])

                # Get datafield details
                try:
                    analysis_service = get_analysis_service()
                    analysis_ops = analysis_service._get_analysis_ops()
                    datafield_info = analysis_ops.parser.datafields.get(datafield, {})
                    dataset_id = datafield_info.get('dataset_id', 'Unknown')
                    category = datafield_info.get('data_category', 'Unknown')
                except:
                    dataset_id = 'Unknown'
                    category = 'Unknown'

                content = create_datafield_modal_content(
                    datafield, count, "Top 20 Datafields", alphas_using, dataset_id, category
                )

                return True, f"Datafield: {datafield}", content

            except (KeyError, IndexError, TypeError) as e:
                return True, "Error", create_error_modal_content(f"Error processing datafield click: {str(e)}")

        raise PreventUpdate

    @callback_wrapper.safe_callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("all-datafields-chart", "clickData"),
         Input("detail-modal-close", "n_clicks")],
        [State("detail-modal", "is_open"),
         State("analysis-data", "data")],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_all_datafields_chart_click(all_datafields_click, close_clicks, is_open, analysis_data):
        """
        Handle all datafields chart clicks.

        Handles clicks on the "All Used Datafields" chart to show datafield details.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print(f"DEBUG: All datafields chart callback triggered by: {trigger_id}")

        if trigger_id == "detail-modal-close":
            return False, "", ""

        if trigger_id == "all-datafields-chart" and all_datafields_click:
            print(f"DEBUG: Processing all-datafields-chart click")
            try:
                datafield = all_datafields_click['points'][0]['y']
                count = all_datafields_click['points'][0]['x']

                # Get alphas using this datafield
                datafields_data = analysis_data.get('datafields', {})
                unique_usage = datafields_data.get('unique_usage', {})
                alphas_using = unique_usage.get(datafield, [])

                # Get datafield details
                try:
                    analysis_service = get_analysis_service()
                    analysis_ops = analysis_service._get_analysis_ops()
                    datafield_info = analysis_ops.parser.datafields.get(datafield, {})
                    dataset_id = datafield_info.get('dataset_id', 'Unknown')
                    category = datafield_info.get('data_category', 'Unknown')
                except:
                    dataset_id = 'Unknown'
                    category = 'Unknown'

                content = create_datafield_modal_content(
                    datafield, count, "All Used Datafields", alphas_using, dataset_id, category
                )

                return True, f"Datafield: {datafield}", content

            except (KeyError, IndexError, TypeError) as e:
                return True, "Error", create_error_modal_content(f"Error processing datafield click: {str(e)}")

        raise PreventUpdate

    @callback_wrapper.safe_callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("all-datasets-chart", "clickData"),
         Input("detail-modal-close", "n_clicks")],
        [State("detail-modal", "is_open"),
         State("analysis-data", "data")],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_all_datasets_chart_click(all_datasets_click, close_clicks, is_open, analysis_data):
        """
        Handle datasets chart clicks.

        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "detail-modal-close":
            return False, "", ""

        if trigger_id == "all-datasets-chart" and all_datasets_click:
            try:
                dataset = all_datasets_click['points'][0]['y']
                count = all_datasets_click['points'][0]['x']

                # Get dataset details from analysis data
                datafields_data = analysis_data.get('datafields', {})
                unique_usage = datafields_data.get('unique_usage', {})

                # Find all datafields for this dataset
                try:
                    analysis_service = get_analysis_service()
                    analysis_ops = analysis_service._get_analysis_ops()

                    dataset_datafields = []
                    total_alphas = set()

                    for df, alphas in unique_usage.items():
                        if df in analysis_ops.parser.datafields:
                            df_dataset = analysis_ops.parser.datafields[df]['dataset_id']
                            if df_dataset == dataset:
                                dataset_datafields.append((df, len(alphas)))
                                total_alphas.update(alphas)

                    # Sort by count
                    dataset_datafields.sort(key=lambda x: x[1], reverse=True)

                except Exception as e:
                    print(f"Error getting dataset details: {e}")
                    dataset_datafields = []
                    total_alphas = set()

                content = create_dataset_modal_content(
                    dataset, count, dataset_datafields, total_alphas
                )

                return True, f"Dataset: {dataset}", content

            except (KeyError, IndexError, TypeError) as e:
                return True, "Error", create_error_modal_content(f"Error processing dataset click: {str(e)}")

        raise PreventUpdate

    @callback_wrapper.safe_callback(
        [Output("detail-modal", "is_open", allow_duplicate=True),
         Output("detail-modal-title", "children", allow_duplicate=True),
         Output("detail-modal-body", "children", allow_duplicate=True)],
        [Input("neutralization-pie-chart", "clickData"),
         Input("neutralization-bar-chart", "clickData"),
         Input("detail-modal-close", "n_clicks")],
        [State("detail-modal", "is_open"),
         State("analysis-data", "data")],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_neutralization_click(pie_click, bar_click, close_clicks, is_open, analysis_data):
        """
        Handle neutralization chart clicks.

        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        print(f"DEBUG: Neutralization callback triggered by: {trigger_id}")

        if trigger_id == "detail-modal-close":
            return False, "", ""

        # Check which chart was clicked
        if trigger_id == "neutralization-pie-chart" and pie_click:
            print(f"DEBUG: Processing neutralization-pie-chart click")
            try:
                neutralization = pie_click['points'][0]['label']
                count = pie_click['points'][0]['value']

                # Get matching alphas
                from ..services import get_recommendation_service
                rec_service = get_recommendation_service()
                matching_alphas = rec_service.get_alphas_by_neutralization(neutralization, limit=100)

                content = create_neutralization_modal_content(
                    neutralization, count, matching_alphas
                )

                return True, f"Neutralization: {neutralization}", content

            except (KeyError, IndexError, TypeError) as e:
                return True, "Error", create_error_modal_content(f"Error processing neutralization click: {str(e)}")

        elif trigger_id == "neutralization-bar-chart" and bar_click:
            print(f"DEBUG: Processing neutralization-bar-chart click")
            try:
                neutralization = bar_click['points'][0]['y']
                count = bar_click['points'][0]['x']

                # Get matching alphas
                from ..services import get_recommendation_service
                rec_service = get_recommendation_service()
                matching_alphas = rec_service.get_alphas_by_neutralization(neutralization, limit=100)

                content = create_neutralization_modal_content(
                    neutralization, count, matching_alphas
                )

                return True, f"Neutralization: {neutralization}", content

            except (KeyError, IndexError, TypeError) as e:
                return True, "Error", create_error_modal_content(f"Error processing neutralization click: {str(e)}")

        raise PreventUpdate

    @callback_wrapper.safe_callback(
        [Output("datafield-detail-modal", "is_open", allow_duplicate=True),
         Output("datafield-modal-title", "children", allow_duplicate=True),
         Output("datafield-modal-body", "children", allow_duplicate=True)],
        [Input({'type': 'datafield-used-badge', 'idx': ALL, 'region': ALL, 'datafield': ALL}, 'n_clicks')],
        [State("datafield-detail-modal", "is_open")],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_datafield_used_badge_click(n_clicks_list, is_open):
        """
        Handle clicks on datafield-used-badge (green badges showing where datafield is used).
        Shows alphas using the datafield in the clicked region.
        """
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks_list):
            raise PreventUpdate

        # Get the triggered badge info
        triggered_id = ctx.triggered[0]['prop_id']
        badge_info = eval(triggered_id.split('.')[0])  # Extract the dictionary ID

        datafield = badge_info['datafield']
        region = badge_info['region']

        try:
            # Get alphas using this datafield in this region
            rec_service = get_recommendation_service()
            analysis_service = get_analysis_service()

            # Get alphas for this datafield and region
            alphas_in_region = rec_service.get_alphas_by_datafield_and_region(datafield, region)

            # Create modal content
            content = [
                html.H5(f"📊 Datafield Usage in {region}", className="mb-3"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong("Datafield: "),
                        html.Code(datafield, className="bg-light p-1")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Region: "),
                        html.Span(region, className="badge bg-success ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Total Alphas: "),
                        html.Span(f"{len(alphas_in_region)}", className="badge bg-primary ms-2")
                    ])
                ], flush=True, className="mb-4"),

                html.H6(f"🔗 Alphas Using This Datafield in {region}", className="mb-3"),
            ]

            if alphas_in_region:
                alpha_badges = [
                    create_alpha_badge(
                        alpha_id=alpha_id,
                        href=f"https://platform.worldquantbrain.com/alpha/{alpha_id}",
                        color="success"
                    ) for alpha_id in alphas_in_region[:100]  # Limit to 100 for performance
                ]

                content.append(
                    html.Div(
                        alpha_badges,
                        className="mb-3",
                        style={'max-height': '400px', 'overflow-y': 'auto'}
                    )
                )

                if len(alphas_in_region) > 100:
                    content.append(
                        dbc.Alert(f"Showing first 100 of {len(alphas_in_region)} alphas", color="info")
                    )
            else:
                content.append(
                    dbc.Alert("No alpha data available.", color="warning")
                )

            return True, f"Datafield {datafield} in {region}", content

        except Exception as e:
            return True, "Error", create_error_modal_content(f"Error loading datafield details: {str(e)}")

    @callback_wrapper.safe_callback(
        [Output("datafield-detail-modal", "is_open", allow_duplicate=True),
         Output("datafield-modal-title", "children", allow_duplicate=True),
         Output("datafield-modal-body", "children", allow_duplicate=True)],
        [Input({'type': 'datafield-region-badge', 'idx': ALL, 'region': ALL, 'datafield': ALL}, 'n_clicks')],
        [State("datafield-detail-modal", "is_open"),
         State("recommendations-content", "children")],
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def handle_datafield_region_badge_click(n_clicks_list, is_open, recommendations_content):
        """
        Handle clicks on datafield-region-badge (blue badges showing recommended regions).
        Shows available datafield IDs in the target region.
        """
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks_list):
            raise PreventUpdate

        # Get the triggered badge info
        triggered_id = ctx.triggered[0]['prop_id']
        badge_info = eval(triggered_id.split('.')[0])  # Extract the dictionary ID

        datafield = badge_info['datafield']
        target_region = badge_info['region']
        idx = badge_info['idx']

        try:
            # Get the recommendation service and fetch fresh recommendations data
            rec_service = get_recommendation_service()

            # Get the full recommendations data to access availability_details
            recommendations_data = rec_service.get_datafield_recommendations()
            recommendations = recommendations_data.get('recommendations', [])

            # Find the specific recommendation for this datafield
            matching_rec = None
            for rec in recommendations:
                if rec['datafield_id'] == datafield:
                    matching_rec = rec
                    break

            if not matching_rec:
                # Fallback to the old method if recommendation not found
                matching_datafields = rec_service.get_matching_datafields_in_region(datafield, target_region)
            else:
                # Use the availability_details from the recommendation
                availability_details = matching_rec.get('availability_details', {})
                matching_ids = availability_details.get(target_region, [])

                # Get detailed info for each matching datafield
                matching_datafields = []
                if matching_ids:
                    analysis_ops = rec_service._get_analysis_ops()
                    db_engine = analysis_ops._get_db_engine()
                    with db_engine.connect() as connection:
                        from sqlalchemy import text
                        # Build IN clause for datafield IDs
                        placeholders = ', '.join([f':df_{i}' for i in range(len(matching_ids))])
                        query = text(f"""
                            SELECT DISTINCT datafield_id, data_description, dataset_id,
                                   data_category, data_type, delay
                            FROM datafields
                            WHERE region = :region
                            AND datafield_id IN ({placeholders})
                        """)

                        # Build parameters dict
                        params = {'region': target_region}
                        for i, df_id in enumerate(matching_ids):
                            params[f'df_{i}'] = df_id

                        result = connection.execute(query, params)
                        for row in result:
                            matching_datafields.append({
                                'id': row.datafield_id,
                                'description': row.data_description or 'No description',
                                'dataset': row.dataset_id or 'Unknown',
                                'category': row.data_category or 'Unknown',
                                'data_type': row.data_type or 'Unknown',
                                'delay': row.delay or 0
                            })

            # Create modal content
            content = [
                html.H5(f"📊 Datafield Availability in {target_region}", className="mb-3"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong("Source Datafield: "),
                        html.Code(datafield, className="bg-light p-1")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Target Region: "),
                        html.Span(target_region, className="badge bg-primary ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("Available Datafield IDs: "),
                        html.Span(f"{len(matching_datafields)}", className="badge bg-info ms-2")
                    ])
                ], flush=True, className="mb-4"),
            ]

            if matching_datafields:
                content.append(html.H6("🔍 Available Datafields", className="mb-3"))

                # Create cards for each matching datafield
                for i, df_info in enumerate(matching_datafields[:10], 1):  # Limit to 10 for display
                    card = dbc.Card([
                        dbc.CardHeader([
                            html.H6([
                                html.Span(f"Option {i}: ", className="text-muted"),
                                html.Code(df_info.get('id', 'Unknown'), className="bg-light p-1")
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Strong("Dataset: "),
                                    html.Span(df_info.get('dataset', 'Unknown'), className="badge bg-success ms-2")
                                ], md=6),
                                dbc.Col([
                                    html.Strong("Category: "),
                                    html.Span(df_info.get('category', 'Unknown'), className="badge bg-info ms-2")
                                ], md=6)
                            ], className="mb-2"),
                            html.P(df_info.get('description', 'No description available'),
                                  className="text-muted small mb-0")
                        ])
                    ], className="mb-3")
                    content.append(card)

                if len(matching_datafields) > 10:
                    content.append(
                        dbc.Alert(f"Showing first 10 of {len(matching_datafields)} available datafields", color="info")
                    )

                content.append(
                    dbc.Alert([
                        html.Strong("💡 Tip: "),
                        f"You can use any of these datafields to create new alphas in {target_region}"
                    ], color="light", className="mt-3")
                )
            else:
                content.append(
                    dbc.Alert(f"No matching datafields found in {target_region}", color="warning")
                )

            return True, f"Datafield Options for {target_region}", content

        except Exception as e:
            return True, "Error", create_error_modal_content(f"Error loading datafield options: {str(e)}")

def register_close_modal_callbacks(app: dash.Dash):
    """Register modal close callbacks."""
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        Output("detail-modal", "is_open", allow_duplicate=True),
        Input("detail-modal-close", "n_clicks"),
        State("detail-modal", "is_open"),
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def close_modal(n_clicks, is_open):
        """
        Close main detail modal.

        """
        if n_clicks:
            return False
        return dash.no_update

    @callback_wrapper.safe_callback(
        Output("datafield-detail-modal", "is_open"),
        Input("datafield-modal-close", "n_clicks"),
        State("datafield-detail-modal", "is_open"),
        prevent_initial_call=True
    )
    @preserve_prevent_update_logic
    def close_datafield_modal(n_clicks, is_open):
        """
        Close datafield detail modal.

        """
        if n_clicks:
            return False
        return dash.no_update

# Export for easy registration
__all__ = ['register_modal_callbacks', 'register_close_modal_callbacks']