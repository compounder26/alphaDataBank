"""
Modal Interaction Callbacks

Modal handling callbacks for chart clicks and detail displays.
Extracted from visualization_server.py lines 4037-5178 with exact logic preservation.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic
from ..components import (
    create_operator_modal_content, create_datafield_modal_content,
    create_dataset_modal_content, create_category_modal_content,
    create_neutralization_modal_content, create_error_modal_content
)
from ..services import get_analysis_service


def register_modal_callbacks(app: dash.Dash):
    """
    Register all modal interaction callbacks.

    CRITICAL: These callbacks handle chart clicks and modal content generation.
    Maintains exact compatibility with original visualization_server.py.

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

        EXACT COPY from visualization_server.py lines 4037-4108
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

        EXACT COPY from visualization_server.py lines 4109-4180
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

        EXACT COPY from visualization_server.py lines 4181-4200
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

        EXACT COPY from visualization_server.py lines 4221-4231
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

        EXACT COPY from visualization_server.py lines 4500-4528
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

        EXACT COPY from visualization_server.py lines 4023-4027
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

        EXACT COPY from visualization_server.py lines 5174-5178
        """
        if n_clicks:
            return False
        return dash.no_update


# Export for easy registration
__all__ = ['register_modal_callbacks', 'register_close_modal_callbacks']