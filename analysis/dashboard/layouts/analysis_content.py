"""
Analysis Content Functions

Content creation functions for neutralization and cross-analysis views.
Extracted from visualization_server.py lines 2204-2350 with exact logic preservation.
"""

import plotly.express as px
from dash import html, dcc
import dash_bootstrap_components as dbc

from ..components import create_recommendation_filters, create_loading_wrapper


def create_cross_analysis_content(analysis_data):
    """
    Create cross-analysis content with datafield recommendations.

    EXACT COPY from visualization_server.py lines 2204-2265
    """
    return dbc.Row([
        dbc.Col([
            html.H4("Datafield Recommendations", className="mb-3"),
            html.P("Discover datafields used in your submitted alphas and identify regions where they could be expanded.",
                   className="text-muted mb-4"),

            # Filters section
            create_recommendation_filters(),

            # Loading indicator and recommendations content
            create_loading_wrapper(
                content=html.Div(id='recommendations-content'),
                loading_id="loading-recommendations"
            )
        ])
    ])


def create_neutralization_content(analysis_data):
    """
    Create neutralization analysis content.

    EXACT COPY from visualization_server.py lines 2267-2350
    """
    metadata = analysis_data.get('metadata', {})
    neutralizations = metadata.get('neutralizations', {})
    total_alphas = metadata.get('total_alphas', 0)

    if not neutralizations:
        return dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H4("No Neutralization Data Available", className="alert-heading"),
                    html.P("Neutralization information is not available in the current dataset."),
                ], color="warning")
            ])
        ])

    # Create pie chart for neutralization breakdown
    labels = list(neutralizations.keys())
    values = list(neutralizations.values())

    pie_fig = px.pie(
        values=values,
        names=labels,
        title="Alpha Distribution by Neutralization Type",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(showlegend=True, height=500)

    # Create bar chart for neutralization breakdown
    bar_fig = px.bar(
        x=values,
        y=labels,
        orientation='h',
        title="Neutralization Usage Count",
        color=values,
        color_continuous_scale='viridis'
    )
    bar_fig.update_traces(texttemplate='%{x}', textposition='outside')
    bar_fig.update_layout(
        xaxis_title="Number of Alphas",
        yaxis_title="Neutralization Type",
        showlegend=False,
        height=max(400, len(labels) * 50)
    )

    return dbc.Row([
        dbc.Col([
            dcc.Graph(id="neutralization-pie-chart", figure=pie_fig)
        ], width=6),
        dbc.Col([
            dcc.Graph(id="neutralization-bar-chart", figure=bar_fig)
        ], width=6),
        dbc.Col([
            html.H5("🎯 Neutralization Statistics"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Total Neutralization Types: "),
                    html.Span(f"{len(neutralizations)}", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Most Common: "),
                    html.Span(f"{max(neutralizations, key=neutralizations.get) if neutralizations else 'N/A'}", className="badge bg-success ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Least Common: "),
                    html.Span(f"{min(neutralizations, key=neutralizations.get) if neutralizations else 'N/A'}", className="badge bg-warning ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Total Alphas: "),
                    html.Span(f"{total_alphas}", className="badge bg-secondary ms-2")
                ])
            ], className="mb-3"),

            html.H6("📋 Detailed Breakdown"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong(f"{neut_type}: "),
                    html.Span(f"{count} alphas", className="me-2"),
                    html.Span(f"({count/total_alphas*100:.1f}%)" if total_alphas > 0 else "(0%)", className="text-muted small")
                ]) for neut_type, count in sorted(neutralizations.items(), key=lambda x: x[1], reverse=True)
            ])
        ], width=12, className="mt-4")
    ])