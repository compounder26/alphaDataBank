# Alpha Clustering Analysis

This module provides tools for analyzing the similarities and differences between alphas using various dimensionality reduction techniques.

## Overview

The alpha clustering analysis allows visualization of alpha relationships using two different methodologies:

### Method 1: Correlation-based Clustering (MDS)
- Calculates pairwise Pearson correlation matrix from the daily PnL data of the alphas
- Converts correlations to dissimilarities (D_ij = 1 - C_ij)
- Applies Multidimensional Scaling (MDS) to visualize the relationships in 2D space

### Method 2: Performance Feature-based Clustering
Applies dimensionality reduction to a set of performance features:
- Sharpe Ratio (from database)
- Maximum Drawdown (from database)
- Returns (from database)
- Volatility (calculated)
- Percentage of Positive PnL Days (calculated)
- Calmar Ratio (calculated)
- Skewness of daily returns (calculated)
- Kurtosis of daily returns (calculated)

Offers three different dimensionality reduction techniques:
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- PCA (Principal Component Analysis)

## Files

- `clustering_analysis.py`: Main analysis module for generating clustering data
- `visualization_server.py`: Interactive web visualization server using Dash/Plotly
- `README.md`: This documentation file

## Requirements

The clustering module requires the following packages (beyond the core alphaDataBank dependencies):
- scikit-learn
- umap-learn
- dash
- dash-bootstrap-components
- plotly

You can install these dependencies with:

```bash
pip install scikit-learn umap-learn dash dash-bootstrap-components plotly
```

## Usage

### Step 1: Generate clustering data

Run the clustering analysis to generate visualization data for a specific region:

```bash
python -m analysis.clustering.clustering_analysis --region USA
```

Available regions: USA, EUR, JPN, CHN, AMR, ASI, GLB, HKG, KOR, TWN

This will create a JSON file with the clustering results in the clustering directory.

### Step 2: Start the visualization server

Run the visualization server with the generated JSON file:

```bash
python -m analysis.clustering.visualization_server path/to/alpha_clustering_USA_timestamp.json
```

Then open http://localhost:8050 in your browser to view the interactive visualization.

## Visualization Features

The web interface provides:
- Interactive scatter plots showing alpha relationships
- Ability to switch between different clustering methods (MDS, t-SNE, UMAP, PCA)
- Hover tooltips showing alpha IDs
- Detailed information panel when clicking on alpha points
- Highlighting of selected alphas
