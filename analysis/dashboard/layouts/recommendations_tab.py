"""
Recommendations Tab Layout

Layout structure for the datafield recommendations and cross-analysis tab.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc

from ..components import (
    create_info_card,
    create_recommendation_filters,
    create_loading_wrapper,
    create_alpha_summary_card,
    create_recommendations_table
)

def create_recommendations_tab_content() -> dbc.Row:
    """
    Create the recommendations tab content layout.

    Returns:
        Row component with recommendations tab structure
    """
    return dbc.Row([
        dbc.Col([
            html.H4("Datafield Recommendations", className="mb-3"),
            html.P(
                "Discover datafields used in your submitted alphas and identify regions where they could be expanded.",
                className="text-muted mb-4"
            ),

            # Filters section
            create_recommendation_filters(),

            # Loading indicator and recommendations content
            create_loading_wrapper(
                content=html.Div(id='recommendations-content'),
                loading_id="loading-recommendations"
            )
        ])
    ])

def create_recommendations_summary_card(summary_stats):
    """
    Create recommendations summary card.

    Args:
        summary_stats: Summary statistics dictionary

    Returns:
        Summary card component
    """
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Total Datafields Analyzed", className="text-muted"),
                    html.H3(str(summary_stats.get('total_analyzed', 0)))
                ], md=4),
                dbc.Col([
                    html.H6("Expansion Opportunities", className="text-muted"),
                    html.H3(str(summary_stats.get('expansion_opportunities', 0)))
                ], md=4),
                dbc.Col([
                    html.H6("Potential New Alphas", className="text-muted"),
                    html.H3(str(summary_stats.get('potential_new_alphas', 0)))
                ], md=4)
            ])
        ])
    ], className="mb-4")

def create_recommendations_display(recommendations, summary_stats):
    """
    Create complete recommendations display.

    Args:
        recommendations: List of recommendation dictionaries
        summary_stats: Summary statistics

    Returns:
        Complete recommendations display
    """
    if not recommendations:
        return dbc.Alert(
            "No datafield recommendations found matching the current filters. Try adjusting the filters or analyzing more alphas.",
            color="info"
        )

    return html.Div([
        create_recommendations_summary_card(summary_stats),
        html.H5("Datafield Expansion Opportunities", className="mb-3"),
        html.P(
            "The table below shows datafields you've used in submitted alphas and regions where they could be expanded. Click on region badges to see detailed datafield information.",
            className="text-muted"
        ),
        create_recommendations_table(recommendations)
    ])

def create_recommendation_error_display(error_message):
    """
    Create error display for recommendations.

    Args:
        error_message: Error message to display

    Returns:
        Alert component with error
    """
    return dbc.Alert(f"Error loading recommendations: {error_message}", color="danger")

def create_no_recommendations_message():
    """
    Create message when no recommendations are available.

    Returns:
        Alert component with message
    """
    return dbc.Alert(
        "No datafield recommendations found matching the current filters. Try adjusting the filters or analyzing more alphas.",
        color="info"
    )

def create_neutralization_content(neutralization_data):
    """
    Create neutralization analysis content.

    Args:
        neutralization_data: Neutralization analysis data

    Returns:
        Row component with neutralization content
    """
    if not neutralization_data.get('available', False):
        return dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H4("No Neutralization Data Available", className="alert-heading"),
                    html.P("Neutralization information is not available in the current dataset."),
                ], color="warning")
            ])
        ])

    neutralizations = neutralization_data['neutralizations']
    total_alphas = neutralization_data['total_alphas']
    statistics = neutralization_data['statistics']
    breakdown = neutralization_data['breakdown']

    return dbc.Row([
        # Pie chart
        dbc.Col([
            dcc.Graph(id="neutralization-pie-chart")
        ], width=6),

        # Bar chart
        dbc.Col([
            dcc.Graph(id="neutralization-bar-chart")
        ], width=6),

        # Statistics panel
        dbc.Col([
            html.H5("🎯 Neutralization Statistics"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Strong("Total Neutralization Types: "),
                    html.Span(f"{statistics['total_types']}", className="badge bg-primary ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Most Common: "),
                    html.Span(f"{statistics['most_common']}", className="badge bg-success ms-2")
                ]),
                dbc.ListGroupItem([
                    html.Strong("Least Common: "),
                    html.Span(f"{statistics['least_common']}", className="badge bg-warning ms-2")
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
                    html.Span(
                        f"({count/total_alphas*100:.1f}%)" if total_alphas > 0 else "(0%)",
                        className="text-muted small"
                    )
                ]) for neut_type, count in breakdown
            ])
        ], width=12, className="mt-4")
    ])