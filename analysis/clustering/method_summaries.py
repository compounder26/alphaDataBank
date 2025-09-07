"""
Concise summaries of clustering methods for dashboard tooltips and quick reference.
Each summary is 3-4 sentences optimized for quantitative analysts.
"""

CLUSTERING_METHOD_SUMMARIES = {
    "mds_correlation": (
        "MDS projects alphas into 2D space where Euclidean distances approximate d_ij = √(2(1-ρ_ij)), "
        "preserving the correlation structure from daily returns. Alphas close together are highly "
        "correlated and offer little diversification benefit, while distant alphas are uncorrelated "
        "or negatively correlated. This linear dimensionality reduction is ideal for identifying "
        "clusters of redundant strategies and finding truly orthogonal alpha sources."
    ),
    
    "tsne_performance": (
        "t-SNE uses KL divergence minimization with Student's t-distribution to embed high-dimensional "
        "performance features (Sharpe, drawdown, skewness) into 2D while preserving local neighborhoods. "
        "Unlike linear methods, t-SNE can separate alphas with similar average performance but different "
        "risk profiles or market regime dependencies. Tight clusters indicate strategies with nearly "
        "identical performance characteristics that may be redundant in a portfolio."
    ),
    
    "umap_performance": (
        "UMAP constructs a fuzzy topological representation using k-nearest neighbors and optimizes "
        "a cross-entropy objective to preserve both local and global structure of the performance manifold. "
        "Superior to t-SNE for maintaining meaningful inter-cluster distances while preserving local "
        "neighborhoods, UMAP reveals hierarchical relationships between strategy families. The algorithm's "
        "ability to preserve global topology makes cluster distances interpretable for portfolio construction."
    ),
    
    "pca_performance": (
        "PCA projects standardized performance metrics onto orthogonal principal components that capture "
        "maximum variance, with PC1 and PC2 typically explaining 40-60% of total variation. The first "
        "component often represents the risk-return trade-off while subsequent components capture style "
        "factors or regime sensitivities. Linear nature preserves relative distances, making PCA ideal "
        "for understanding the primary drivers of performance variation across your alpha universe."
    ),
    
    "hierarchical_risk_parity": (
        "HRP applies hierarchical clustering using correlation distance d = √(0.5(1-ρ)) followed by "
        "recursive bisection with inverse-variance allocation to construct robust portfolios without "
        "requiring expected returns. The dendrogram visualizes the correlation hierarchy where height "
        "represents dissimilarity - combining alphas from different branches maximizes diversification. "
        "This method avoids the numerical instability of Markowitz optimization while adapting weights "
        "to the correlation structure, making it ideal for practical portfolio construction."
    ),
    
    "minimum_spanning_tree": (
        "The MST connects all alphas with minimum total correlation distance Σ√(2(1-ρ_ij)), revealing "
        "the essential structure of dependencies by filtering out redundant connections. Hub nodes with "
        "many edges represent strategies correlated with multiple others (avoid over-allocation), while "
        "leaf nodes indicate unique alphas offering maximum diversification. The MST forms the backbone "
        "for advanced portfolio methods and helps identify which correlations are structurally important."
    ),
    
    "correlation_heatmap": (
        "Displays the full N×N Pearson correlation matrix of daily returns with hierarchical clustering "
        "to reorder alphas, revealing block structures of strategy families. Red blocks indicate redundant "
        "strategies (ρ > 0.7) that should not be combined, while blue regions show natural hedges (ρ < -0.3). "
        "White areas represent uncorrelated alphas (|ρ| < 0.3) ideal for diversification - combining "
        "strategies from different blocks creates robust portfolios with stable risk-adjusted returns."
    )
}

# Portfolio construction insights for each method
PORTFOLIO_INSIGHTS = {
    "mds_correlation": [
        "Select alphas from different regions of the plot for maximum diversification",
        "Avoid combining multiple alphas from the same cluster",
        "Outliers often represent unique strategies worth higher allocation",
        "Monitor cluster stability over time to detect regime changes"
    ],
    
    "tsne_performance": [
        "Clusters reveal strategies with similar risk-return profiles",
        "Combine alphas from different clusters to diversify performance drivers",
        "Outliers may indicate strategies with unique market exposures",
        "Use for regime-aware portfolio construction"
    ],
    
    "umap_performance": [
        "Inter-cluster distances indicate true diversification potential",
        "Build portfolios by sampling from each major cluster",
        "Central regions often contain balanced, all-weather strategies",
        "Edge regions may contain specialized, high-conviction alphas"
    ],
    
    "pca_performance": [
        "Combine alphas from different quadrants for orthogonal exposures",
        "PC1 typically represents risk level - balance across this axis",
        "PC2 often captures style - mix for style diversification",
        "Monitor loadings to understand performance drivers"
    ],
    
    "hierarchical_risk_parity": [
        "Allocate across major branches first, then within branches",
        "Cutting the dendrogram at different heights gives different granularity",
        "Higher cuts create broader strategy buckets",
        "Use inverse-variance weighting within each cluster"
    ],
    
    "minimum_spanning_tree": [
        "Prioritize leaf nodes for unique exposures",
        "Limit allocation to hub nodes to avoid concentration",
        "Path length indicates correlation transitivity",
        "Monitor tree stability to detect structural breaks"
    ],
    
    "correlation_heatmap": [
        "Target correlation of 0.3 or less between portfolio components",
        "Blue-red pairs can provide natural hedging",
        "Block diagonal structure reveals natural strategy groupings",
        "Off-diagonal patterns indicate cross-strategy dependencies"
    ]
}

# Mathematical complexity and computational notes
COMPUTATIONAL_NOTES = {
    "mds_correlation": {
        "complexity": "O(n²) for correlation, O(n³) for eigendecomposition",
        "scalability": "Good for up to 1000 alphas",
        "stability": "Stable, deterministic solution",
        "parameters": "No hyperparameters to tune"
    },
    
    "tsne_performance": {
        "complexity": "O(n² log n) with Barnes-Hut approximation",
        "scalability": "Good for up to 10,000 alphas",
        "stability": "Stochastic - set random_state for reproducibility",
        "parameters": "Perplexity (5-50), learning_rate (10-1000)"
    },
    
    "umap_performance": {
        "complexity": "O(n^1.14) with approximation algorithms",
        "scalability": "Excellent - handles 100,000+ alphas",
        "stability": "More stable than t-SNE, still stochastic",
        "parameters": "n_neighbors (5-50), min_dist (0.0-0.99)"
    },
    
    "pca_performance": {
        "complexity": "O(min(n³, p³)) where p = features",
        "scalability": "Excellent for any dataset size",
        "stability": "Completely deterministic",
        "parameters": "n_components (typically 2-10)"
    },
    
    "hierarchical_risk_parity": {
        "complexity": "O(n² log n) for clustering, O(n) for allocation",
        "scalability": "Good for portfolios up to 500 assets",
        "stability": "Deterministic given correlation matrix",
        "parameters": "Linkage method, number of clusters"
    },
    
    "minimum_spanning_tree": {
        "complexity": "O(n² log n) with efficient algorithms",
        "scalability": "Good for up to 5000 alphas",
        "stability": "Unique solution for distinct correlations",
        "parameters": "None - fully determined by correlations"
    },
    
    "correlation_heatmap": {
        "complexity": "O(n²) for correlation, O(n² log n) for clustering",
        "scalability": "Visualization limited to ~200 alphas for clarity",
        "stability": "Deterministic correlation, slight variation in clustering",
        "parameters": "Clustering method, distance metric"
    }
}

def get_summary(method_key: str) -> str:
    """Get concise summary for a clustering method."""
    return CLUSTERING_METHOD_SUMMARIES.get(method_key, "Method description not available.")

def get_portfolio_insights(method_key: str) -> list:
    """Get portfolio construction insights for a method."""
    return PORTFOLIO_INSIGHTS.get(method_key, ["No specific insights available."])

def get_computational_notes(method_key: str) -> dict:
    """Get computational complexity and parameter notes."""
    return COMPUTATIONAL_NOTES.get(method_key, {
        "complexity": "Unknown",
        "scalability": "Unknown",
        "stability": "Unknown",
        "parameters": "Unknown"
    })