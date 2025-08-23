"""
Alpha Clustering Visualization Server.

This module provides a web interface for visualizing the alpha clustering results.
It allows for interactive exploration of the data through hover tooltips and clickable points.
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Default styling
COLORS = px.colors.qualitative.Plotly
TEMPLATE = 'plotly_white'

def load_clustering_data(filepath: str) -> Dict[str, Any]:
    """
    Load clustering data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary with clustering data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

def create_visualization_app(data: Dict[str, Any]) -> dash.Dash:
    """
    Create a Dash app for visualizing the clustering data.
    
    Args:
        data: Dictionary with clustering data
        
    Returns:
        Dash app
    """
    # Extract data
    region = data['region']
    timestamp = data['timestamp']
    alpha_count = data['alpha_count']
    
    # Convert coordinate dictionaries to DataFrames
    mds_coords = pd.DataFrame.from_dict(data['mds_coords'], orient='index')
    tsne_coords = pd.DataFrame.from_dict(data['tsne_coords'], orient='index')
    umap_coords = pd.DataFrame.from_dict(data['umap_coords'], orient='index')
    pca_coords = pd.DataFrame.from_dict(data['pca_coords'], orient='index')
    
    # Merge with metadata if available
    metadata = pd.DataFrame.from_dict(data.get('alpha_metadata', {}), orient='index')
    
    # Initialize Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Set app title
    app.title = f"Alpha Clustering - {region}"
    
    # Create layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1(f"Alpha Clustering - {region}", className="text-center my-4"),
                html.P(f"Generated on: {timestamp} | Total Alphas: {alpha_count}", 
                      className="text-center text-muted mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Visualization Method"),
                    dbc.CardBody([
                        dcc.RadioItems(
                            id='method-selector',
                            options=[
                                {'label': 'MDS on Correlation Matrix', 'value': 'mds'},
                                {'label': 't-SNE on Performance Features', 'value': 'tsne'},
                                {'label': 'UMAP on Performance Features', 'value': 'umap'},
                                {'label': 'PCA on Performance Features', 'value': 'pca'},
                            ],
                            value='mds',
                            inline=True,
                            className="mb-3"
                        ),
                    ])
                ], className="mb-4"),
                
                dbc.Card([
                    dbc.CardHeader("Alpha Details"),
                    dbc.CardBody([
                        html.Div(id='alpha-details', className="p-3")
                    ])
                ])
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Clustering Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id='clustering-plot', style={'height': '80vh'})
                    ])
                ])
            ], width=9)
        ]),
        
        # Store the data in the app
        dcc.Store(id='mds-data', data=mds_coords.reset_index().to_dict('records')),
        dcc.Store(id='tsne-data', data=tsne_coords.reset_index().to_dict('records')),
        dcc.Store(id='umap-data', data=umap_coords.reset_index().to_dict('records')),
        dcc.Store(id='pca-data', data=pca_coords.reset_index().to_dict('records')),
        dcc.Store(id='metadata', data=metadata.reset_index().to_dict('records')),
        dcc.Store(id='selected-alpha', data=None)
    ], fluid=True)
    
    # Callback to update the plot based on the selected method
    @app.callback(
        Output('clustering-plot', 'figure'),
        Input('method-selector', 'value'),
        Input('mds-data', 'data'),
        Input('tsne-data', 'data'),
        Input('umap-data', 'data'),
        Input('pca-data', 'data'),
        Input('selected-alpha', 'data')
    )
    def update_plot(method, mds_data, tsne_data, umap_data, pca_data, selected_alpha):
        # Select the appropriate data based on the method
        if method == 'mds':
            plot_data = pd.DataFrame(mds_data)
            title = "MDS on Correlation Matrix"
        elif method == 'tsne':
            plot_data = pd.DataFrame(tsne_data)
            title = "t-SNE on Performance Features"
        elif method == 'umap':
            plot_data = pd.DataFrame(umap_data)
            title = "UMAP on Performance Features"
        elif method == 'pca':
            plot_data = pd.DataFrame(pca_data)
            title = "PCA on Performance Features"
        
        # Create scatter plot
        fig = px.scatter(
            plot_data, 
            x='x', 
            y='y',
            hover_name='index',  # Use alpha_id as hover name
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            template=TEMPLATE
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            hovermode='closest',
            clickmode='event+select'
        )
        
        # Update traces for better interactivity
        fig.update_traces(
            marker=dict(size=10, opacity=0.8),
            hovertemplate='<b>%{hovertext}</b><br>Click to view on WorldQuant Brain<extra></extra>'
        )
        
        # Highlight selected alpha if any
        if selected_alpha:
            selected_index = selected_alpha.get('index')
            if selected_index in plot_data['index'].values:
                selected_point = plot_data[plot_data['index'] == selected_index]
                
                fig.add_trace(go.Scatter(
                    x=selected_point['x'],
                    y=selected_point['y'],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15,
                        line=dict(width=2, color='black')
                    ),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        fig.update_layout(
            annotations=[
                dict(
                    text="Click on a point to open its WorldQuant Brain link",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.1
                )
            ]
        )
        
        return fig
    
    # Callback to update alpha details when a point is clicked
    @app.callback(
        Output('alpha-details', 'children'),
        Output('selected-alpha', 'data'),
        Input('clustering-plot', 'clickData'),
        State('metadata', 'data')
    )
    def display_alpha_details(clickData, metadata_data):
        if not clickData:
            return "Click on a point to see alpha details.", None
        
        # Get the alpha_id from the clicked point
        point = clickData['points'][0]
        alpha_id = point['hovertext']
        
        # Create WorldQuant Brain URL
        wq_url = f"https://platform.worldquantbrain.com/alpha/{alpha_id}"
        
        # Find metadata for this alpha
        metadata_df = pd.DataFrame(metadata_data)
        if not metadata_df.empty and 'index' in metadata_df.columns and alpha_id in metadata_df['index'].values:
            alpha_metadata = metadata_df[metadata_df['index'] == alpha_id].iloc[0].to_dict()
            
            # Create details card
            details = [
                html.H5(f"Alpha ID: {alpha_id}"),
                html.Hr(),
                # Add button to open WorldQuant Brain in new tab
                html.A(
                    dbc.Button(
                        "View on WorldQuant Brain", 
                        color="primary", 
                        className="mb-3"
                    ),
                    href=wq_url,
                    target="_blank"  # Open in new tab
                )
            ]
            
            # Add metadata fields
            for key, value in alpha_metadata.items():
                if key != 'index':
                    if isinstance(value, (int, float)):
                        # Format numbers nicely
                        formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    details.append(html.P([
                        html.Strong(f"{key.replace('_', ' ').title()}: "),
                        formatted_value
                    ]))
        else:
            details = [
                html.H5(f"Alpha ID: {alpha_id}"),
                html.Hr(),
                # Add button even if no metadata is available
                html.A(
                    dbc.Button(
                        "View on WorldQuant Brain", 
                        color="primary", 
                        className="mb-3"
                    ),
                    href=wq_url,
                    target="_blank"  # Open in new tab
                ),
                html.P("No additional metadata available for this alpha.")
            ]
        
        # Return details and selected alpha
        return details, {'index': alpha_id}
    
    return app

def main():
    """
    Run the visualization server.
    """
    parser = argparse.ArgumentParser(description="Alpha Clustering Visualization")
    parser.add_argument("data_file", type=str, help="Path to the clustering data JSON file")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.isfile(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        return
    
    # Load data
    data = load_clustering_data(args.data_file)
    
    # Create app
    app = create_visualization_app(data)
    
    # Run server
    print(f"Starting visualization server for region {data['region']} on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser to view the visualization.")
    app.run(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main()
