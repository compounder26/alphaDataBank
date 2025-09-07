"""
Example integration of clustering method explanations into the Dash dashboard.
This shows how to add informative tooltips and help text to the visualization interface.
"""

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from method_explanations import get_method_explanation, DASHBOARD_METHOD_MAPPING
from method_summaries import get_summary, get_portfolio_insights

def create_method_info_card(method_key: str):
    """
    Create an informative card component for a clustering method.
    
    Parameters
    ----------
    method_key : str
        The key identifying the clustering method
    
    Returns
    -------
    dbc.Card
        A Bootstrap card component with method explanation
    """
    explanation = get_method_explanation(method_key)
    summary = get_summary(method_key)
    insights = get_portfolio_insights(method_key)
    
    card = dbc.Card([
        dbc.CardHeader(html.H4(explanation['title'])),
        dbc.CardBody([
            html.H6("Quick Summary", className="text-muted"),
            html.P(summary, style={'fontSize': '0.9em'}),
            
            html.Hr(),
            
            html.H6("Mathematical Formula", className="text-muted"),
            html.Code(explanation['key_formula'], 
                     style={'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '5px'}),
            
            html.Hr(),
            
            html.H6("Portfolio Construction Tips", className="text-muted"),
            html.Ul([html.Li(insight, style={'fontSize': '0.9em'}) for insight in insights[:3]]),
            
            html.Hr(),
            
            dbc.Accordion([
                dbc.AccordionItem(
                    html.P(explanation['explanation'], style={'fontSize': '0.85em'}),
                    title="Detailed Mathematical Explanation",
                    item_id="math_details"
                ),
                dbc.AccordionItem(
                    html.Pre(explanation['interpretation'], 
                            style={'fontSize': '0.85em', 'whiteSpace': 'pre-wrap'}),
                    title="Interpretation Guide",
                    item_id="interpretation"
                )
            ], start_collapsed=True, flush=True)
        ])
    ], className="mb-3")
    
    return card

def create_method_dropdown_with_help():
    """
    Create a dropdown menu for method selection with integrated help tooltips.
    
    Returns
    -------
    dbc.Row
        A row containing the dropdown and help button
    """
    dropdown_options = [
        {"label": display_name, "value": method_key}
        for display_name, method_key in DASHBOARD_METHOD_MAPPING.items()
    ]
    
    return dbc.Row([
        dbc.Col([
            dbc.Label("Select Clustering Method", html_for="method-dropdown"),
            dcc.Dropdown(
                id="method-dropdown",
                options=dropdown_options,
                value="mds_correlation",
                clearable=False
            )
        ], width=10),
        dbc.Col([
            dbc.Button(
                "?", 
                id="method-help-button",
                color="info",
                size="sm",
                className="mt-4",
                outline=True
            ),
            dbc.Popover(
                id="method-help-popover",
                target="method-help-button",
                trigger="hover",
                placement="left"
            )
        ], width=2)
    ])

def create_comparison_table():
    """
    Create a comparison table of all clustering methods.
    
    Returns
    -------
    dbc.Table
        A formatted table comparing key aspects of each method
    """
    from method_summaries import COMPUTATIONAL_NOTES
    
    rows = []
    for display_name, method_key in DASHBOARD_METHOD_MAPPING.items():
        comp_notes = COMPUTATIONAL_NOTES.get(method_key, {})
        explanation = get_method_explanation(method_key)
        
        rows.append(
            html.Tr([
                html.Td(html.Strong(display_name)),
                html.Td(explanation.get('mathematical_name', 'N/A'), style={'fontSize': '0.85em'}),
                html.Td(comp_notes.get('complexity', 'N/A'), style={'fontSize': '0.85em'}),
                html.Td(comp_notes.get('scalability', 'N/A'), style={'fontSize': '0.85em'}),
                html.Td(comp_notes.get('stability', 'N/A'), style={'fontSize': '0.85em'}),
                html.Td(
                    dbc.Button(
                        "Details",
                        id=f"details-{method_key}",
                        size="sm",
                        color="secondary",
                        outline=True
                    )
                )
            ])
        )
    
    table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Method"),
                html.Th("Technical Name"),
                html.Th("Complexity"),
                html.Th("Scalability"),
                html.Th("Stability"),
                html.Th("Info")
            ])
        ]),
        html.Tbody(rows)
    ], striped=True, bordered=True, hover=True, responsive=True, size="sm")
    
    return table

# Example callback for updating help popover
def register_help_callbacks(app):
    """
    Register callbacks for method help functionality.
    
    Parameters
    ----------
    app : dash.Dash
        The Dash application instance
    """
    @app.callback(
        Output("method-help-popover", "children"),
        Input("method-dropdown", "value")
    )
    def update_help_popover(method_key):
        if not method_key:
            return "Select a method to see details."
        
        summary = get_summary(method_key)
        explanation = get_method_explanation(method_key)
        
        return [
            html.H6(explanation['title'], className="mb-2"),
            html.P(summary, style={'fontSize': '0.85em'}),
            html.Hr(),
            html.Small(f"Formula: {explanation['key_formula']}", className="text-muted")
        ]

# Example of adding method explanations to existing plots
def add_explanation_to_plot(fig, method_key):
    """
    Add method explanation as annotation to a plotly figure.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The plotly figure to annotate
    method_key : str
        The clustering method key
    
    Returns
    -------
    plotly.graph_objects.Figure
        The annotated figure
    """
    explanation = get_method_explanation(method_key)
    
    # Add subtitle with mathematical name
    fig.update_layout(
        title={
            'text': f"{explanation['title']}<br><sub>{explanation['mathematical_name']}</sub>",
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    # Add formula as annotation
    fig.add_annotation(
        text=f"<i>{explanation['key_formula']}</i>",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="left"
    )
    
    return fig

# Quick reference card for dashboard sidebar
def create_quick_reference_card():
    """
    Create a collapsible quick reference card for the sidebar.
    
    Returns
    -------
    dbc.Card
        A card with quick tips for using clustering visualizations
    """
    return dbc.Card([
        dbc.CardHeader(
            html.H5("Quick Reference Guide", className="mb-0"),
            style={'backgroundColor': '#f8f9fa'}
        ),
        dbc.Collapse(
            dbc.CardBody([
                html.H6("Correlation-based Methods", className="text-primary mt-2"),
                html.Ul([
                    html.Li("MDS: Linear projection, preserves correlation distances"),
                    html.Li("HRP: Hierarchical portfolio construction"),
                    html.Li("MST: Network view of strongest correlations"),
                    html.Li("Heatmap: Full correlation matrix view")
                ], style={'fontSize': '0.85em'}),
                
                html.H6("Performance-based Methods", className="text-primary mt-3"),
                html.Ul([
                    html.Li("PCA: Linear, orthogonal components"),
                    html.Li("t-SNE: Non-linear, local structure"),
                    html.Li("UMAP: Non-linear, preserves global structure")
                ], style={'fontSize': '0.85em'}),
                
                html.H6("Key Insights", className="text-primary mt-3"),
                html.Ul([
                    html.Li("Distance = Dissimilarity in all 2D projections"),
                    html.Li("Clusters = Similar strategies (avoid combining)"),
                    html.Li("Outliers = Unique alphas (good for diversification)"),
                    html.Li("Color = Performance metric (Sharpe, returns, etc.)")
                ], style={'fontSize': '0.85em'}),
                
                html.Hr(),
                
                html.P([
                    html.Strong("Pro Tip: "),
                    "Combine insights from multiple methods for robust portfolio construction. "
                    "No single visualization tells the complete story."
                ], style={'fontSize': '0.85em', 'fontStyle': 'italic'})
            ]),
            id="reference-collapse",
            is_open=False
        )
    ])

if __name__ == "__main__":
    # Example: Print all method summaries for documentation
    print("=" * 80)
    print("CLUSTERING METHOD SUMMARIES FOR DOCUMENTATION")
    print("=" * 80)
    
    for display_name, method_key in DASHBOARD_METHOD_MAPPING.items():
        print(f"\n### {display_name}")
        print(get_summary(method_key))
        print(f"\n**Key Formula:** {get_method_explanation(method_key)['key_formula']}")
        print("-" * 40)