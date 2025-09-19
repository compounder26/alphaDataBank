"""Advanced clustering plot creation functions."""

import numpy as np
import plotly.graph_objects as go
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import traceback


def create_advanced_clustering_plot(method, all_region_data, selected_region, distance_metric='euclidean'):
    """Create advanced clustering visualizations (HRP, MST, Heatmap).
    
    Parameters
    ----------
    method : str
        Visualization method ('hrp', 'mst', or 'heatmap')
    all_region_data : dict
        Data for all regions (not currently used)
    selected_region : str
        Selected region for analysis
    distance_metric : str
        Distance metric to use ('simple', 'euclidean', or 'angular')
    
    Returns
    -------
    plotly.graph_objects.Figure
        The visualization figure
    """
    TEMPLATE = 'plotly_white'
    
    try:
        # Import clustering analysis functions
        from analysis.clustering.clustering_algorithms import (
            calculate_correlation_matrix
        )
        from database.operations import (
            get_regular_alpha_ids_by_region,
            get_pnl_data_for_alphas
        )
        
        # Get PNL data for the selected region
        if not selected_region:
            selected_region = 'USA'
        
        print(f"Creating {method} plot for region {selected_region} with {distance_metric} distance")
        
        # Get alpha IDs for the region
        alpha_ids = get_regular_alpha_ids_by_region(selected_region)
        if not alpha_ids:
            return go.Figure().add_annotation(
                text=f"No alphas found for region {selected_region}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=20)
            )
        
        # Get PNL data for these alphas
        pnl_data = get_pnl_data_for_alphas(alpha_ids, selected_region)
        if not pnl_data:
            return go.Figure().add_annotation(
                text=f"No PNL data available for region {selected_region}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=20)
            )
        
        print(f"Fetched PNL data for {len(pnl_data)} alphas")
        
        # Calculate correlation matrix
        corr_matrix = calculate_correlation_matrix(pnl_data)
        
        # Extract alpha_ids from the correlation matrix index
        alpha_ids = list(corr_matrix.index) if not corr_matrix.empty else []
        
        print(f"Calculated correlation matrix of shape {corr_matrix.shape}")
        
        # Convert correlation to distance based on selected metric
        if distance_metric == 'simple':
            dist_matrix = 1 - corr_matrix
        elif distance_metric == 'euclidean':
            dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        elif distance_metric == 'angular':
            dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        else:
            dist_matrix = np.sqrt(2 * (1 - corr_matrix))  # Default to Euclidean
        
        # Ensure no negative distances (can happen with numerical errors)
        dist_matrix = np.maximum(dist_matrix, 0)
        
        # Create visualization based on method
        if method == 'mst':
            # Minimum Spanning Tree - show network graph
            print("Creating MST network graph")
            
            # Create graph from correlation matrix
            G = nx.Graph()
            
            # Add nodes
            for alpha_id in alpha_ids:
                G.add_node(alpha_id)
            
            # Add edges with weights (using distance)
            for i in range(len(alpha_ids)):
                for j in range(i+1, len(alpha_ids)):
                    # Use distance as weight - use iloc for DataFrame indexing
                    G.add_edge(alpha_ids[i], alpha_ids[j], weight=dist_matrix.iloc[i, j])
            
            # Find minimum spanning tree
            mst = nx.minimum_spanning_tree(G)
            
            print(f"MST has {len(mst.nodes())} nodes and {len(mst.edges())} edges")
            
            # Create layout for visualization
            pos = nx.spring_layout(mst, k=2, iterations=50, seed=42)
            
            # Create plotly figure
            fig = go.Figure()
            
            # Add edges
            edge_traces = []
            for edge in mst.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                # Get correlation for this edge
                idx0 = alpha_ids.index(edge[0])
                idx1 = alpha_ids.index(edge[1])
                correlation = corr_matrix[idx0, idx1]
                
                # Create color based on correlation
                if correlation > 0.7:
                    color = 'red'  # High correlation
                elif correlation > 0.3:
                    color = 'orange'  # Medium correlation
                else:
                    color = 'green'  # Low correlation
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=2, color=color),
                    hovertemplate=f"{edge[0]} - {edge[1]}<br>Correlation: {correlation:.3f}<extra></extra>",
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            # Add all edge traces
            for trace in edge_traces:
                fig.add_trace(trace)
            
            # Add nodes
            node_x = [pos[node][0] for node in mst.nodes()]
            node_y = [pos[node][1] for node in mst.nodes()]
            node_text = [str(node) for node in mst.nodes()]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(
                    size=12,
                    color='lightblue',
                    line=dict(color='darkblue', width=2)
                ),
                hovertemplate="Alpha: %{text}<extra></extra>",
                showlegend=False
            ))
            
            # Add legend for edge colors
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='red', width=2),
                name='High Correlation (>0.7)'
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='orange', width=2),
                name='Medium Correlation (0.3-0.7)'
            ))
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='green', width=2),
                name='Low Correlation (<0.3)'
            ))
            
            fig.update_layout(
                title=f"Minimum Spanning Tree ({distance_metric} distance)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template=TEMPLATE,
                height=600,
                hovermode='closest',
                showlegend=True
            )
            
        elif method == 'heatmap':
            # Correlation heatmap
            print("Creating correlation heatmap")
            
            # Limit size for readability
            max_alphas = 50
            if len(alpha_ids) > max_alphas:
                print(f"Limiting heatmap to first {max_alphas} alphas for readability")
                # Use iloc for DataFrame slicing
                corr_matrix = corr_matrix.iloc[:max_alphas, :max_alphas]
                alpha_ids = alpha_ids[:max_alphas]
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=alpha_ids,
                y=alpha_ids,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 8},
                colorbar=dict(title="Correlation"),
                hovertemplate="Alpha X: %{x}<br>Alpha Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=f"Correlation Heatmap (using {distance_metric} distance metric)",
                xaxis=dict(title="Alpha ID", tickangle=45, tickfont=dict(size=8)),
                yaxis=dict(title="Alpha ID", tickfont=dict(size=8)),
                template=TEMPLATE,
                height=800,
                width=900
            )
        
        else:
            return go.Figure().add_annotation(
                text=f"Unknown clustering method: {method}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=20)
            )
        
        print(f"Successfully created {method} visualization")
        return fig
        
    except Exception as e:
        print(f"Error creating advanced clustering plot: {str(e)}")
        traceback.print_exc()
        
        return go.Figure().add_annotation(
            text=f"Error creating {method} visualization: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )