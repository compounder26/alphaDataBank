"""
Data Loading Callbacks

Data loading and preloading callbacks with preserved functionality.
Extracted from visualization_server.py lines 658-709 with exact logic preservation.
"""

import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from .base_callbacks import CallbackWrapper, preserve_prevent_update_logic
from ..services import get_analysis_service


def register_data_loading_callbacks(app: dash.Dash):
    """
    Register data loading callbacks.

    CRITICAL: These callbacks handle the initial data loading and filter population.
    Maintains exact compatibility with original visualization_server.py.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        Output('preloaded-analysis-data', 'data'),
        Input('initial-load-trigger', 'n_intervals'),
        State('analysis-ops', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def preload_analysis_data(n_intervals, analysis_ops_data):
        """
        Preload analysis data callback.

        EXACT COPY from visualization_server.py lines 658-677
        Preserves all logic, error handling, and data structures.
        """
        if not analysis_ops_data.get('available', False):
            return {}

        try:
            # Use the refactored service but maintain exact behavior
            analysis_service = get_analysis_service()
            results = analysis_service.get_analysis_summary()
            return results
        except Exception as e:
            print(f"Error preloading data: {e}")
            # The AnalysisOperations.get_analysis_summary() now handles missing table initialization
            # so this should work on retry. Return empty dict for now.
            return {}

    @callback_wrapper.safe_callback(
        [Output('region-filter', 'options'),
         Output('universe-filter', 'options'),
         Output('delay-filter', 'options'),
         Output('dates-filter', 'min_date_allowed'),
         Output('dates-filter', 'max_date_allowed'),
         Output('dates-filter', 'start_date'),
         Output('dates-filter', 'end_date')],
        Input('preloaded-analysis-data', 'data'),
        State('analysis-ops', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def populate_filter_options(preloaded_data, analysis_ops_data):
        """
        Populate filter options callback.

        EXACT COPY from visualization_server.py lines 679-709
        Preserves all logic for populating dropdowns and date ranges.
        """
        if not analysis_ops_data.get('available', False) or not preloaded_data:
            return [], [], [], None, None, None, None

        try:
            # Use preloaded data to populate filter options faster
            metadata = preloaded_data.get('metadata', {})

            regions = [{'label': r, 'value': r} for r in sorted(metadata.get('regions', {}).keys())]
            universes = [{'label': u, 'value': u} for u in sorted(metadata.get('universes', {}).keys())]
            delays = [{'label': str(d), 'value': d} for d in sorted(metadata.get('delays', {}).keys())]

            # Get date range for date picker
            min_date = metadata.get('min_date_added')
            max_date = metadata.get('max_date_added')

            return regions, universes, delays, min_date, max_date, None, None
        except:
            return [], [], [], None, None, None, None

    @callback_wrapper.safe_callback(
        [Output('analysis-data', 'data'),
         Output('analysis-summary', 'children')],
        [Input('apply-filters-btn', 'n_clicks'),
         Input('initial-load-trigger', 'n_intervals')],
        [State('region-filter', 'value'),
         State('universe-filter', 'value'),
         State('delay-filter', 'value'),
         State('dates-filter', 'start_date'),
         State('dates-filter', 'end_date'),
         State('preloaded-analysis-data', 'data'),
         State('analysis-ops', 'data')],
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_analysis_data(n_clicks, n_intervals, region, universe, delay, date_from, date_to, preloaded_data, analysis_ops_data):
        """
        Update analysis data callback.

        EXACT COPY from visualization_server.py lines 945-1034
        Preserves all logic for data filtering and summary creation.
        """
        if not analysis_ops_data.get('available', False):
            return {}, "Analysis not available"

        try:
            # Use preloaded data if no filters are applied (faster)
            if not any([region, universe, delay, date_from, date_to]) and preloaded_data:
                results = preloaded_data
            else:
                # Get filtered analysis results using service
                analysis_service = get_analysis_service()
                results = analysis_service.get_analysis_summary(region, universe, delay, date_from, date_to)

            # Add filter information to results metadata
            if 'metadata' not in results:
                results['metadata'] = {}
            results['metadata']['filters'] = {
                'region': region,
                'universe': universe,
                'delay': delay,
                'date_from': date_from,
                'date_to': date_to
            }

            # Create summary (EXACT LOGIC from original)
            metadata = results.get('metadata', {})
            total_alphas = metadata.get('total_alphas', 0)
            excluded_alphas = metadata.get('excluded_alphas', 0)
            total_processed = metadata.get('total_processed', 0)

            # Show filter information in summary if filters are applied
            filter_info = []
            if region:
                filter_info.append(f"Region: {region}")
            if universe:
                filter_info.append(f"Universe: {universe}")
            if delay is not None:
                filter_info.append(f"Delay: {delay}")
            if date_from or date_to:
                date_range = f"{date_from or 'start'} to {date_to or 'end'}"
                filter_info.append(f"Date range: {date_range}")

            # Build summary items starting with total alphas
            summary_items = [
                dbc.ListGroupItem([
                    html.Strong(f"Total Alphas: {total_alphas}")
                ])
            ]

            # Add exclusion information if there are excluded alphas
            if excluded_alphas > 0:
                exclusion_text = f"Excluded (tier restrictions): {excluded_alphas}"
                if total_processed > 0:
                    percentage = (excluded_alphas / total_processed) * 100
                    exclusion_text += f" ({percentage:.1f}%)"

                summary_items.append(
                    dbc.ListGroupItem([
                        html.Strong(exclusion_text),
                        html.Br(),
                        html.Small("Alphas excluded due to unavailable operators/datafields for your tier",
                                  className="text-muted")
                    ])
                )

            if filter_info:
                summary_items.append(
                    dbc.ListGroupItem([
                        html.Strong("Active Filters: "),
                        html.Small(", ".join(filter_info), className="text-muted")
                    ])
                )

            summary_items.extend([
                dbc.ListGroupItem([
                    html.Strong("Top Operators:"),
                    html.Ul([
                        html.Li(f"{op}: {count} alphas")
                        for op, count in results.get('operators', {}).get('top_operators', [])[:5]
                    ])
                ]),
                dbc.ListGroupItem([
                    html.Strong("Top Datafields:"),
                    html.Ul([
                        html.Li(f"{df}: {count} alphas")
                        for df, count in results.get('datafields', {}).get('top_datafields', [])[:5]
                    ])
                ])
            ])

            summary = dbc.ListGroup(summary_items, flush=True)

            return results, summary
        except Exception as e:
            return {}, f"Error loading analysis: {str(e)}"


def register_clustering_region_callbacks(app: dash.Dash):
    """
    Register clustering region selection callbacks.

    Args:
        app: Dash application instance
    """
    callback_wrapper = CallbackWrapper(app)

    @callback_wrapper.safe_callback(
        [Output('clustering-region-selector', 'options'),
         Output('clustering-region-selector', 'value')],
        Input('available-regions', 'data'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def populate_clustering_region_options(available_regions):
        """
        Populate clustering region options.

        EXACT COPY from visualization_server.py lines 711-723
        """
        if not available_regions:
            return [], None

        options = [{'label': region, 'value': region} for region in available_regions]
        default_value = available_regions[0] if available_regions else None
        return options, default_value

    @callback_wrapper.safe_callback(
        Output('selected-clustering-region', 'data'),
        Input('clustering-region-selector', 'value'),
        prevent_initial_call=False  # CRITICAL: Must match original
    )
    @preserve_prevent_update_logic
    def update_selected_clustering_region(selected_region):
        """
        Update the selected clustering region store.

        EXACT COPY from visualization_server.py lines 800-807
        """
        return selected_region


# Export for easy registration
__all__ = ['register_data_loading_callbacks', 'register_clustering_region_callbacks']