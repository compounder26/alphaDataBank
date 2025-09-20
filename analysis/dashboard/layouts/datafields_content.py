"""
Datafields Content Functions

Content creation functions for datafields analysis views.
Extracted from visualization_server.py lines 1341-2200 with exact logic preservation.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
from dash import html, dcc
import dash_bootstrap_components as dbc

from ..config import TEMPLATE
from ..services import create_analysis_operations, get_analysis_service


def create_datafields_content(analysis_data, view_mode='top20'):
    """
    Create datafields analysis content with different view modes.

    EXACT COPY from visualization_server.py lines 1341-1376
    """
    datafields_data = analysis_data.get('datafields', {})
    top_datafields = datafields_data.get('top_datafields', [])
    by_category = datafields_data.get('by_category', {})
    metadata = analysis_data.get('metadata', {})

    if not top_datafields:
        return html.Div("No datafield data available", className="text-muted text-center p-4")

    # Add view mode selector
    view_selector = dbc.Card([
        dbc.CardHeader("View Options"),
        dbc.CardBody([
            dbc.RadioItems(
                id='datafields-view-selector',
                options=[
                    {'label': 'Top 20 Most Used', 'value': 'top20'},
                    {'label': 'All Used Datafields', 'value': 'all'},
                    {'label': 'All Used Datasets', 'value': 'datasets'},
                    {'label': 'Dataset Treemap', 'value': 'treemap'}
                ],
                value=view_mode,
                inline=True
            )
        ])
    ], className="mb-3")

    if view_mode == 'all':
        return html.Div([view_selector, create_all_datafields_content(analysis_data)])
    elif view_mode == 'datasets':
        return html.Div([view_selector, create_all_datasets_content(analysis_data)])
    elif view_mode == 'treemap':
        return html.Div([view_selector, create_dataset_treemap_content(analysis_data)])
    else:
        return html.Div([view_selector, create_top20_datafields_content(analysis_data)])


def create_all_datafields_content(analysis_data):
    """
    Create view showing all used datafields with their counts.

    EXACT COPY from visualization_server.py lines 1397-1431
    """
    datafields_data = analysis_data.get('datafields', {})
    all_used_datafields = datafields_data.get('top_datafields', [])
    by_category = datafields_data.get('by_category', {})
    metadata = analysis_data.get('metadata', {})

    # Create bar chart with all datafields
    df1 = pd.DataFrame({
        'datafield': [df for df, _ in all_used_datafields],
        'count': [count for _, count in all_used_datafields]
    })

    # Create interactive bar chart for all datafields
    fig1 = px.bar(
        df1,
        x='count',
        y='datafield',
        orientation='h',
        title=f"All {len(all_used_datafields)} Used Datafields (Click bars for details)",
        labels={'count': 'Number of Alphas Using', 'datafield': 'Datafield'},
        template=TEMPLATE,
        height=max(800, len(all_used_datafields) * 20)  # Dynamic height
    )
    fig1.update_layout(clickmode='event+select')
    fig1.update_traces(
        hovertemplate='<b>%{y}</b><br>Used in %{x} alphas<br>Click for details<extra></extra>',
        marker_color='darkgreen'
    )

    return dbc.Row([
        dbc.Col([
            dcc.Graph(id='all-datafields-chart', figure=fig1, style={'height': f'{fig1.layout.height}px'}),
        ], width=12)
    ])


def create_all_datasets_content(analysis_data):
    """
    Create view showing all used datasets with their counts.

    EXACT COPY from visualization_server.py lines 1433-1500
    """
    datafields_data = analysis_data.get('datafields', {})
    unique_usage = datafields_data.get('unique_usage', {})
    metadata = analysis_data.get('metadata', {})

    if not unique_usage:
        return html.Div("No dataset data available", className="text-muted text-center p-4")

    # Calculate dataset usage counts
    dataset_counts = {}
    try:
        analysis_service = get_analysis_service()
        temp_analysis_ops = analysis_service._get_analysis_ops()
        print(f"Processing {len(unique_usage)} datafields for dataset analysis")

        for df, alphas in unique_usage.items():
            if df in temp_analysis_ops.parser.datafields:
                dataset_id = temp_analysis_ops.parser.datafields[df]['dataset_id']
                if dataset_id not in dataset_counts:
                    dataset_counts[dataset_id] = 0
                dataset_counts[dataset_id] += len(alphas)

        datasets_list = list(sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True))
        print(f"Found {len(datasets_list)} datasets: {[name for name, _ in datasets_list[:10]]}...")
    except Exception as e:
        print(f"Error processing datasets: {e}")
        return html.Div("Error processing dataset data", className="text-danger text-center p-4")

    if not datasets_list:
        return html.Div("No dataset data found", className="text-muted text-center p-4")

    # Create DataFrame for datasets chart
    df = pd.DataFrame({
        'dataset': [dataset for dataset, _ in datasets_list],
        'count': [count for _, count in datasets_list]
    })

    # Create interactive bar chart for all datasets
    fig = px.bar(
        df,
        x='count',
        y='dataset',
        orientation='h',
        title=f"All Used Datasets ({len(datasets_list)} total) - Click bars for details",
        labels={'count': 'Total Datafield Instances', 'dataset': 'Dataset ID'},
        template=TEMPLATE
    )

    # Calculate dynamic height for proper visibility
    min_height = 400
    bar_height = max(25, int(400 / len(datasets_list))) if len(datasets_list) > 0 else 25
    calculated_height = max(min_height, len(datasets_list) * bar_height + 100)

    fig.update_layout(
        height=calculated_height,
        clickmode='event+select',
        yaxis={'categoryorder': 'total ascending'}
    )
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>%{x} total datafield instances<br>Click for details<extra></extra>',
        marker_color='steelblue'
    )

    return dbc.Row([
        dbc.Col([
            dcc.Graph(id='all-datasets-chart', figure=fig, style={'height': f'{fig.layout.height}px'}),
        ], width=12)
    ])


def create_top20_datafields_content(analysis_data):
    """
    Create the original top 20 datafields view.

    EXTRACTED from visualization_server.py lines 1502-1800 (complex grid layout)
    """
    datafields_data = analysis_data.get('datafields', {})
    top_datafields = datafields_data.get('top_datafields', [])
    by_category = datafields_data.get('by_category', {})
    metadata = analysis_data.get('metadata', {})

    # Create DataFrame for datafields chart
    df1 = pd.DataFrame({
        'datafield': [df for df, _ in top_datafields[:20]],
        'count': [count for _, count in top_datafields[:20]]
    })

    # Create interactive bar chart for top datafields
    fig1 = px.bar(
        df1,
        x='count',
        y='datafield',
        orientation='h',
        title="Top 20 Most Used Datafields (Click bars for details)",
        labels={'count': 'Number of Alphas Using', 'datafield': 'Datafield'},
        template=TEMPLATE
    )
    fig1.update_layout(height=500, clickmode='event+select')
    fig1.update_traces(
        hovertemplate='<b>%{y}</b><br>Used in %{x} alphas<br>Click for details<extra></extra>',
        marker_color='darkgreen'
    )

    # Create enhanced pie chart for categories and dataset analysis
    category_counts, dataset_counts = _calculate_category_and_dataset_counts(datafields_data)

    # Create category pie chart
    fig2 = _create_category_pie_chart(category_counts)

    # Create dataset treemap
    fig3 = _create_dataset_treemap(dataset_counts)

    # Calculate statistics
    total_unique_dfs = len(datafields_data.get('unique_usage', {}))
    total_nominal = sum(datafields_data.get('nominal_usage', {}).values())
    total_alphas = metadata.get('total_alphas', 0)
    avg_dfs_per_alpha = total_nominal / total_alphas if total_alphas > 0 else 0

    # Get region-specific count from analysis data
    total_region_specific_dfs = datafields_data.get('region_specific_count', total_unique_dfs)

    # Create responsive grid layout (simplified from original complex grid)
    return dbc.Row([
        # Main datafields chart
        dbc.Col([
            dcc.Graph(id='datafields-chart', figure=fig1)
        ], width=6),

        # Statistics and category charts
        dbc.Col([
            html.H6("📈 Usage Statistics", className="mb-2"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Total Unique Datafields: "),
                    html.Span(f"{total_unique_dfs}", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Average per Alpha: "),
                    html.Span(f"{avg_dfs_per_alpha:.1f}", className="badge bg-info ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Dataset Categories: "),
                    html.Span(f"{len(category_counts)}", className="badge bg-warning ms-2")
                ]),
            ], flush=True, className="mb-3"),

            dcc.Graph(id='category-chart', figure=fig2, style={'height': '300px'}),
        ], width=6),

        # Dataset treemap (full width)
        dbc.Col([
            dcc.Graph(id='dataset-treemap', figure=fig3, style={'height': '400px'})
        ], width=12)
    ])


def create_dataset_treemap_content(analysis_data):
    """
    Create enhanced treemap view showing datasets with usage patterns.

    SIMPLIFIED from visualization_server.py lines 1957-2107
    """
    # Use chart service for treemap creation
    from ..services import get_analysis_service

    analysis_service = get_analysis_service()

    # Extract filter information from analysis_data metadata
    metadata = analysis_data.get('metadata', {})
    filters = metadata.get('filters', {})

    # Calculate dataset statistics
    dataset_info = analysis_service.calculate_dataset_statistics(
        analysis_data,
        filters.get('region'),
        filters.get('universe'),
        filters.get('delay'),
        filters.get('date_from'),
        filters.get('date_to')
    )

    if 'error' in dataset_info:
        return html.Div(f"Error loading dataset data: {dataset_info['error']}",
                      className="text-danger text-center p-4")

    dataset_stats = dataset_info['dataset_stats']
    if not dataset_stats:
        return html.Div("No dataset data available", className="text-muted text-center p-4")

    # Use chart service to create treemap
    from ..services import get_chart_service
    chart_service = get_chart_service()
    fig = chart_service.create_dataset_treemap(dataset_stats)

    return dcc.Graph(id='dataset-treemap-main', figure=fig, style={'height': '750px'})


def get_dataset_treemap_sidebar_info(analysis_data):
    """
    Get sidebar information panels for the dataset treemap view.

    EXTRACTED from visualization_server.py lines 2109-2202
    """
    from ..services import get_analysis_service

    analysis_service = get_analysis_service()

    # Extract filter information
    metadata = analysis_data.get('metadata', {})
    filters = metadata.get('filters', {})

    # Calculate statistics
    dataset_info = analysis_service.calculate_dataset_statistics(
        analysis_data,
        filters.get('region'),
        filters.get('universe'),
        filters.get('delay'),
        filters.get('date_from'),
        filters.get('date_to')
    )

    if 'error' in dataset_info or not dataset_info['dataset_stats']:
        return []

    # Create statistics panel
    stats_panel = dbc.Card([
        dbc.CardHeader("📊 Dataset Statistics"),
        dbc.CardBody([
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Total Datasets: "),
                    html.Span(f"{dataset_info['total_datasets']}", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Datasets with Alpha Usage: "),
                    html.Span(f"{dataset_info['used_datasets']}", className="badge bg-success ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Unused Datasets: "),
                    html.Span(f"{dataset_info['unused_datasets']}", className="badge bg-secondary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Usage Rate: "),
                    html.Span(
                        f"{(dataset_info['used_datasets'] / dataset_info['total_datasets'] * 100):.1f}%" if dataset_info['total_datasets'] > 0 else "0%",
                        className="badge bg-info ms-2"
                    )
                ])
            ], flush=True, className="mb-3"),

            html.Hr(),
            html.H6("🎯 Top Used Datasets"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong(f"{stat['dataset_id']}: "),
                    html.Span(f"{stat['alpha_usage_count']} alphas", className="me-2"),
                    html.Span(
                        f"({stat['used_datafields']}/{stat['total_datafields']} fields)",
                        className="text-muted small"
                    )
                ]) for stat in [d for d in dataset_info['dataset_stats'] if d['is_used']][:10]
            ], flush=True) if any(d['is_used'] for d in dataset_info['dataset_stats']) else html.Div("No used datasets found", className="text-muted")
        ])
    ], className="mb-3")

    # Create legend panel
    legend_panel = dbc.Card([
        dbc.CardHeader("🎨 Legend"),
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Div(style={
                        'width': '20px', 'height': '20px', 'background-color': '#808080',
                        'display': 'inline-block', 'margin-right': '10px', 'border': '1px solid #ccc'
                    }),
                    html.Span("Unused datasets (no alphas)", style={'vertical-align': 'top'})
                ], className="mb-2"),
                html.Div([
                    html.Div(style={
                        'width': '20px', 'height': '20px',
                        'background': 'linear-gradient(to right, rgb(100,150,200), rgb(130,100,70))',
                        'display': 'inline-block', 'margin-right': '10px', 'border': '1px solid #ccc'
                    }),
                    html.Span("Used datasets (color by alpha count)", style={'vertical-align': 'top'})
                ], className="mb-2")
            ]),
            html.Hr(),
            html.H6("💡 Usage Tips"),
            dbc.Alert([
                html.Ul([
                    html.Li("Size of each rectangle = Total datafields in dataset"),
                    html.Li("Color = Alpha usage (grey = unused, blue-purple gradient = used)"),
                    html.Li("Text shown only for larger datasets to avoid clutter"),
                    html.Li("Hover for detailed statistics on all datasets"),
                    html.Li("Click to zoom in on individual datasets")
                ], className="mb-0")
            ], color="light")
        ])
    ])

    return [stats_panel, legend_panel]


def create_dataset_treemap_content(analysis_data):
    """
    Create enhanced treemap view showing datasets with usage patterns.

    EXTRACTED from visualization_server.py lines 1957-2107
    """
    # Use the chart service for treemap creation
    from ..services import get_chart_service, get_analysis_service

    chart_service = get_chart_service()
    analysis_service = get_analysis_service()

    # Extract filter information from analysis_data metadata
    metadata = analysis_data.get('metadata', {})
    filters = metadata.get('filters', {})

    # Calculate dataset statistics
    dataset_info = analysis_service.calculate_dataset_statistics(
        analysis_data,
        filters.get('region'),
        filters.get('universe'),
        filters.get('delay'),
        filters.get('date_from'),
        filters.get('date_to')
    )

    if 'error' in dataset_info:
        return html.Div(f"Error loading dataset data: {dataset_info['error']}",
                      className="text-danger text-center p-4")

    dataset_stats = dataset_info['dataset_stats']
    if not dataset_stats:
        return html.Div("No dataset data available", className="text-muted text-center p-4")

    # Create treemap using chart service
    fig = chart_service.create_dataset_treemap(dataset_stats)

    return dcc.Graph(id='dataset-treemap-main', figure=fig, style={'height': '750px'})


def create_top20_datafields_content(analysis_data):
    """
    Create the original top 20 datafields view with grid layout.

    SIMPLIFIED from visualization_server.py lines 1502-1800 (removed complex grid)
    """
    datafields_data = analysis_data.get('datafields', {})
    top_datafields = datafields_data.get('top_datafields', [])
    metadata = analysis_data.get('metadata', {})

    # Create DataFrame for datafields chart
    df1 = pd.DataFrame({
        'datafield': [df for df, _ in top_datafields[:20]],
        'count': [count for _, count in top_datafields[:20]]
    })

    # Create interactive bar chart for top datafields
    fig1 = px.bar(
        df1,
        x='count',
        y='datafield',
        orientation='h',
        title="Top 20 Most Used Datafields (Click bars for details)",
        labels={'count': 'Number of Alphas Using', 'datafield': 'Datafield'},
        template=TEMPLATE
    )
    fig1.update_layout(height=500, clickmode='event+select')
    fig1.update_traces(
        hovertemplate='<b>%{y}</b><br>Used in %{x} alphas<br>Click for details<extra></extra>',
        marker_color='darkgreen'
    )

    # Calculate category and dataset counts for additional charts
    category_counts, dataset_counts = _calculate_category_and_dataset_counts(datafields_data)

    # Create category pie chart
    fig2 = _create_category_pie_chart(category_counts)

    # Create dataset treemap
    fig3 = _create_dataset_treemap(dataset_counts)

    # Calculate statistics
    total_unique_dfs = len(datafields_data.get('unique_usage', {}))
    total_nominal = sum(datafields_data.get('nominal_usage', {}).values())
    total_alphas = metadata.get('total_alphas', 0)
    avg_dfs_per_alpha = total_nominal / total_alphas if total_alphas > 0 else 0

    return dbc.Row([
        # Main datafields chart
        dbc.Col([
            dcc.Graph(id='datafields-chart', figure=fig1)
        ], width=6),

        # Statistics panel
        dbc.Col([
            html.H6("📈 Usage Statistics", className="mb-2"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Total Unique Datafields: "),
                    html.Span(f"{total_unique_dfs}", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Average per Alpha: "),
                    html.Span(f"{avg_dfs_per_alpha:.1f}", className="badge bg-info ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Dataset Categories: "),
                    html.Span(f"{len(category_counts)}", className="badge bg-warning ms-2")
                ]),
            ], flush=True, className="mb-3"),

            dcc.Graph(id='category-chart', figure=fig2, style={'height': '300px'})
        ], width=6),

        # Dataset treemap
        dbc.Col([
            dcc.Graph(id='dataset-treemap', figure=fig3, style={'height': '400px'})
        ], width=12)
    ])


def _calculate_category_and_dataset_counts(datafields_data):
    """Calculate category and dataset counts for charts."""
    category_counts = {}
    dataset_counts = {}

    try:
        analysis_service = get_analysis_service()
        temp_analysis_ops = analysis_service._get_analysis_ops()
        unique_usage = datafields_data.get('unique_usage', {})

        for df, alphas in unique_usage.items():
            if df in temp_analysis_ops.parser.datafields:
                dataset_id = temp_analysis_ops.parser.datafields[df]['dataset_id']
                category = temp_analysis_ops.parser.datafields[df]['data_category']

                # Count dataset usage
                if dataset_id:
                    dataset_key = dataset_id if dataset_id else 'unknown'
                    dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + len(alphas)

                # Count category usage
                if category:
                    category_counts[category] = category_counts.get(category, 0) + len(alphas)
            else:
                # Fallback: extract dataset from datafield name
                if '.' in df:
                    dataset_key = df.split('.')[0]
                    dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + len(alphas)

    except Exception as e:
        print(f"Error calculating category/dataset counts: {e}")
        # Fallback logic using by_category data
        by_category = datafields_data.get('by_category', {})
        for category, datafields in by_category.items():
            total_alphas = set()
            for datafield, alphas in datafields.items():
                total_alphas.update(alphas)
                # Extract dataset from datafield name
                if '.' in datafield:
                    dataset_key = datafield.split('.')[0]
                else:
                    parts = datafield.split('_')
                    dataset_key = parts[0] if len(parts) > 1 else 'unknown'

                dataset_counts[dataset_key] = dataset_counts.get(dataset_key, 0) + len(alphas)

            category_counts[category] = len(total_alphas)

    return category_counts, dataset_counts


def _create_category_pie_chart(category_counts):
    """Create category pie chart."""
    if category_counts:
        # Truncate category names for better display
        truncated_names = [name[:15] + "..." if len(name) > 15 else name for name in category_counts.keys()]

        fig2 = px.pie(
            values=list(category_counts.values()),
            names=truncated_names,
            template=TEMPLATE
        )
        fig2.update_layout(
            height=350,
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01,
                font=dict(size=9)
            )
        )
        fig2.update_traces(
            textposition='auto',
            textinfo='percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
            hoverinfo='label+percent+value'
        )
    else:
        fig2 = go.Figure()
        fig2.update_layout(
            title="No Category Data Available",
            height=300,
            annotations=[dict(text="No data to display", x=0.5, y=0.5, showarrow=False)]
        )

    return fig2


def _create_dataset_treemap(dataset_counts):
    """Create dataset treemap."""
    if dataset_counts:
        # Limit to top 20 datasets for readability
        sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        # Create treemap with proper parameters
        fig3 = px.treemap(
            names=[name for name, _ in sorted_datasets],
            parents=["" for _ in sorted_datasets],  # All datasets are children of root
            values=[value for _, value in sorted_datasets],
            title="Top 20 Datasets by Usage (Click to zoom, use bar charts for details)"
        )
        fig3.update_traces(
            textinfo="label+value",
            textposition="middle center",
            textfont=dict(size=18)  # 2x bigger font
        )
        fig3.update_layout(
            height=450,
            font=dict(size=16)
        )
    else:
        fig3 = go.Figure()

    return fig3