"""
Clustering Method Documentation Module.

This module provides comprehensive documentation for all clustering visualization methods
used in the AlphaDataBank dashboard. It serves as a unified source of truth for:

- Mathematical explanations for clustering methods with detailed algorithm descriptions
- Concise summaries for dashboard tooltips and quick reference
- Portfolio construction insights for practical alpha strategy selection
- Computational complexity notes and parameter guidance
- Unified access functions for retrieving method information

The module combines detailed mathematical explanations (originally from method_explanations.py)
with practical summaries and insights (originally from method_summaries.py) to provide
both theoretical understanding and practical application guidance for quantitative analysts
working with alpha strategy clustering and portfolio construction.

Key Features:
- Detailed mathematical formulations for 7 core clustering methods
- Concise 3-4 sentence summaries optimized for dashboard display
- Portfolio construction insights for each visualization method
- Computational complexity and scalability information
- Unified API for accessing all documentation types

Methods Documented:
- MDS on Correlation Matrix: Linear dimensionality reduction preserving correlation distances
- t-SNE on Performance Features: Non-linear embedding emphasizing local neighborhoods
- UMAP on Performance Features: Balanced local/global structure preservation
- PCA on Performance Features: Orthogonal decomposition of performance variance
- Hierarchical Risk Parity: Tree-based portfolio construction with inverse-variance weighting
- Minimum Spanning Tree: Network analysis revealing essential correlation structure
- Correlation Heatmap: Full matrix visualization with hierarchical ordering

Usage:
    # Get detailed mathematical explanation
    explanation = get_method_explanation('mds_correlation')

    # Get concise summary for tooltips
    summary = get_method_summary('tsne_performance')

    # Get both explanation and summary
    info = get_method_info('umap_performance')

    # Get portfolio construction insights
    insights = get_portfolio_insights('hierarchical_risk_parity')

    # Get computational complexity notes
    notes = get_computational_notes('minimum_spanning_tree')
"""

# Detailed explanations (from method_explanations.py)
CLUSTERING_METHOD_EXPLANATIONS = {
    "mds_correlation": {
        "title": "MDS on Correlation Matrix",
        "mathematical_name": "Multidimensional Scaling (Classical MDS)",
        "explanation": (
            "MDS finds a 2D representation where distances between points approximate the original "
            "dissimilarities d_ij = sqrt(2(1 - ρ_ij)), where ρ_ij is the Pearson correlation of daily "
            "percentage returns between alphas i and j. The algorithm minimizes the stress function "
            "Σ(d_ij - δ_ij)² where δ_ij are the Euclidean distances in the reduced space. "
            "This visualization reveals natural clusters of alphas with similar return patterns, helping "
            "identify diversification opportunities - alphas far apart have low correlation and provide "
            "better portfolio diversification when combined."
        ),
        "interpretation": (
            "• Distance represents correlation: nearby alphas are highly correlated (ρ > 0.7)\n"
            "• Clusters indicate redundant strategies that capture similar market inefficiencies\n"
            "• Outliers represent unique alphas offering maximum diversification benefit\n"
            "• Color coding by performance metrics helps identify high-Sharpe, uncorrelated alphas"
        ),
        "key_formula": "d_ij = √(2(1 - ρ_ij)) where ρ_ij = corr(returns_i, returns_j)"
    },

    "tsne_performance": {
        "title": "t-SNE on Performance Features",
        "mathematical_name": "t-Distributed Stochastic Neighbor Embedding",
        "explanation": (
            "t-SNE models pairwise similarities as conditional probabilities p_j|i = exp(-||x_i - x_j||²/2σ_i²) "
            "in high-dimensional space and q_ij = (1 + ||y_i - y_j||²)^(-1) using Student's t-distribution "
            "in the low-dimensional embedding. The algorithm minimizes KL divergence KL(P||Q) = Σp_ij log(p_ij/q_ij) "
            "between these distributions. Applied to performance features (Sharpe, max drawdown, skew, kurtosis), "
            "t-SNE excels at preserving local neighborhoods, revealing clusters of alphas with similar risk-return "
            "profiles that may not be apparent from correlation alone."
        ),
        "interpretation": (
            "• Tight clusters indicate alphas with similar performance characteristics\n"
            "• Perplexity parameter (5-50) controls the effective number of neighbors\n"
            "• Non-linear mapping can separate alphas that linear methods like PCA cannot\n"
            "• Useful for identifying regime-specific strategies (e.g., momentum vs mean-reversion)"
        ),
        "key_formula": "min KL(P||Q) where q_ij ∝ (1 + ||y_i - y_j||²)^(-1)"
    },

    "umap_performance": {
        "title": "UMAP on Performance Features",
        "mathematical_name": "Uniform Manifold Approximation and Projection",
        "explanation": (
            "UMAP constructs a weighted k-neighbor graph using a fuzzy set membership function "
            "ρ_i = min{d(x_i, x_j) : j ∈ kNN(i)} and edge weights w_ij = exp(-(d_ij - ρ_i)/σ_i). "
            "It then optimizes a low-dimensional layout to preserve both local (via attractive forces) "
            "and global (via repulsive forces) structure using cross-entropy CE = Σ[w_ij log(w_ij/v_ij) + "
            "(1-w_ij)log((1-w_ij)/(1-v_ij))]. UMAP typically preserves global structure better than t-SNE "
            "while maintaining superior local neighborhood preservation compared to PCA."
        ),
        "interpretation": (
            "• Preserves both local clusters and global topology of the performance manifold\n"
            "• Distance between clusters is meaningful (unlike t-SNE)\n"
            "• n_neighbors parameter (5-50) controls local vs global structure trade-off\n"
            "• Reveals hierarchical relationships between strategy families"
        ),
        "key_formula": "min CE(high-d graph, low-d graph) with fuzzy set membership"
    },

    "pca_performance": {
        "title": "PCA on Performance Features",
        "mathematical_name": "Principal Component Analysis",
        "explanation": (
            "PCA performs eigendecomposition of the covariance matrix C = (1/n)X^T X of standardized "
            "performance features to find orthogonal directions of maximum variance. The first two "
            "principal components PC₁ and PC₂ are the eigenvectors corresponding to the largest eigenvalues "
            "λ₁ and λ₂, capturing the percentage of variance λ_i/Σλ_j. For alpha strategies, PC₁ often "
            "represents the risk-return trade-off while PC₂ might capture style factors like momentum vs "
            "mean reversion or sensitivity to different market regimes."
        ),
        "interpretation": (
            "• Linear projection preserves global structure and relative distances\n"
            "• Explained variance ratio indicates how much information is retained\n"
            "• Component loadings reveal which features drive the separation\n"
            "• Orthogonal components represent uncorrelated sources of performance variation"
        ),
        "key_formula": "Y = XW where W = [v₁, v₂] are top eigenvectors of Cov(X)"
    },

    "hierarchical_risk_parity": {
        "title": "Hierarchical Risk Parity (HRP)",
        "mathematical_name": "Lopez de Prado's HRP Algorithm",
        "explanation": (
            "HRP combines hierarchical clustering with risk parity allocation in three steps: "
            "(1) Build a dendrogram using distance d_ij = √(0.5(1 - ρ_ij)) and single linkage, "
            "(2) Quasi-diagonalize the correlation matrix to place similar assets together, "
            "(3) Apply recursive bisection with inverse-variance weighting w_i ∝ 1/σ_i within clusters. "
            "This approach avoids the numerical instability of mean-variance optimization while "
            "producing diversified portfolios that adapt to the correlation structure. The dendrogram "
            "height represents correlation distance, revealing the hierarchical relationship between strategies."
        ),
        "interpretation": (
            "• Vertical axis shows correlation distance (0 = perfect correlation, √2 = uncorrelated)\n"
            "• Horizontal branching reveals natural strategy groupings\n"
            "• Earlier splits indicate more distinct strategy families\n"
            "• Optimal portfolio combines strategies from different branches for diversification"
        ),
        "key_formula": "d_ij = √(0.5(1 - ρ_ij)), weights via recursive inverse-variance bisection"
    },

    "minimum_spanning_tree": {
        "title": "Minimum Spanning Tree",
        "mathematical_name": "MST on Correlation Network",
        "explanation": (
            "The MST connects all alphas using edges with minimum total distance d_ij = √(2(1 - ρ_ij)), "
            "creating a tree structure with N-1 edges for N alphas using Kruskal's or Prim's algorithm. "
            "This filtered network preserves only the strongest relationships, revealing the backbone "
            "of dependencies between strategies. Central nodes (high degree) represent strategies that "
            "correlate with many others, while peripheral nodes indicate unique, diversifying alphas. "
            "The MST is the foundation for many portfolio construction methods including HRP."
        ),
        "interpretation": (
            "• Edge length/thickness represents correlation strength (shorter = higher correlation)\n"
            "• Hub nodes (many connections) indicate strategies correlated with multiple others\n"
            "• Leaf nodes (single connection) represent unique, diversifying strategies\n"
            "• Path length between nodes indicates correlation transitivity"
        ),
        "key_formula": "min Σd_ij subject to tree constraint, d_ij = √(2(1 - ρ_ij))"
    },

    "correlation_heatmap": {
        "title": "Correlation Heatmap",
        "mathematical_name": "Pearson Correlation Matrix Visualization",
        "explanation": (
            "The heatmap displays the full N×N correlation matrix where each cell (i,j) shows "
            "ρ_ij = cov(r_i, r_j)/(σ_i × σ_j), the Pearson correlation between daily percentage "
            "returns of alphas i and j. Hierarchical clustering reorders rows/columns to place "
            "similar alphas adjacent, creating block structures that reveal strategy families. "
            "The color scale from blue (ρ = -1) through white (ρ = 0) to red (ρ = 1) provides "
            "immediate visual identification of correlation patterns, redundancies, and diversification opportunities."
        ),
        "interpretation": (
            "• Red blocks indicate clusters of highly correlated strategies (redundant)\n"
            "• Blue regions show negatively correlated alphas (natural hedges)\n"
            "• White/light areas represent uncorrelated strategies (good for diversification)\n"
            "• Dendrograms on axes show the hierarchical clustering structure"
        ),
        "key_formula": "ρ_ij = Σ[(r_it - μ_i)(r_jt - μ_j)] / (σ_i × σ_j × T)"
    }
}

# Concise summaries (from method_summaries.py)
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

# Quick reference for dashboard integration
DASHBOARD_METHOD_MAPPING = {
    "MDS (Correlation Matrix)": "mds_correlation",
    "t-SNE (Performance)": "tsne_performance",
    "UMAP (Performance)": "umap_performance",
    "PCA (Performance)": "pca_performance",
    "Hierarchical Risk Parity": "hierarchical_risk_parity",
    "Minimum Spanning Tree": "minimum_spanning_tree",
    "Correlation Heatmap": "correlation_heatmap"
}

# Unified access functions
def get_method_explanation(method_name: str) -> dict:
    """
    Get detailed mathematical explanation for a clustering method.

    Parameters
    ----------
    method_name : str
        The key identifying the clustering method

    Returns
    -------
    dict
        Dictionary containing title, explanation, interpretation, and formula
    """
    return CLUSTERING_METHOD_EXPLANATIONS.get(method_name, {
        "title": "Unknown Method",
        "mathematical_name": "Unknown",
        "explanation": "No explanation available for this method.",
        "interpretation": "No interpretation guidelines available.",
        "key_formula": "N/A"
    })

def get_method_summary(method_name: str) -> str:
    """
    Get concise summary for a clustering method.

    Parameters
    ----------
    method_name : str
        The key identifying the clustering method

    Returns
    -------
    str
        Concise summary suitable for tooltips and quick reference
    """
    return CLUSTERING_METHOD_SUMMARIES.get(method_name, "Method description not available.")

def get_all_method_names() -> list:
    """
    Get list of all documented clustering methods.

    Returns
    -------
    list
        Sorted list of all method keys that have documentation
    """
    explanations_keys = set(CLUSTERING_METHOD_EXPLANATIONS.keys())
    summaries_keys = set(CLUSTERING_METHOD_SUMMARIES.keys())
    return sorted(explanations_keys.union(summaries_keys))

def get_method_info(method_name: str) -> dict:
    """
    Get both explanation and summary for a method.

    Parameters
    ----------
    method_name : str
        The key identifying the clustering method

    Returns
    -------
    dict
        Dictionary containing both detailed explanation and concise summary
    """
    return {
        'explanation': get_method_explanation(method_name),
        'summary': get_method_summary(method_name)
    }

def get_portfolio_insights(method_name: str) -> list:
    """
    Get portfolio construction insights for a clustering method.

    Parameters
    ----------
    method_name : str
        The key identifying the clustering method

    Returns
    -------
    list
        List of practical insights for portfolio construction
    """
    return PORTFOLIO_INSIGHTS.get(method_name, ["No specific insights available."])

def get_computational_notes(method_name: str) -> dict:
    """
    Get computational complexity and parameter notes for a method.

    Parameters
    ----------
    method_name : str
        The key identifying the clustering method

    Returns
    -------
    dict
        Dictionary with complexity, scalability, stability, and parameter information
    """
    return COMPUTATIONAL_NOTES.get(method_name, {
        "complexity": "Unknown",
        "scalability": "Unknown",
        "stability": "Unknown",
        "parameters": "Unknown"
    })

def get_dashboard_method_key(display_name: str) -> str:
    """
    Convert dashboard display name to internal method key.

    Parameters
    ----------
    display_name : str
        Display name as shown in dashboard dropdown

    Returns
    -------
    str
        Internal method key for accessing documentation
    """
    return DASHBOARD_METHOD_MAPPING.get(display_name, display_name.lower().replace(" ", "_").replace("(", "").replace(")", ""))

def format_explanation_for_display(method_name: str, include_formula: bool = True) -> str:
    """
    Format a detailed explanation for display in the dashboard.

    Parameters
    ----------
    method_name : str
        The key identifying the clustering method
    include_formula : bool, optional
        Whether to include the mathematical formula, by default True

    Returns
    -------
    str
        Formatted explanation text suitable for dashboard display
    """
    method = get_method_explanation(method_name)

    output = f"**{method['title']}**\n\n"
    output += f"{method['explanation']}\n\n"
    output += f"**How to Interpret:**\n{method['interpretation']}\n"

    if include_formula and 'key_formula' in method:
        output += f"\n**Key Formula:** {method['key_formula']}"

    return output

def get_all_explanations() -> dict:
    """
    Get all clustering method explanations.

    Returns
    -------
    dict
        Complete dictionary of all detailed method explanations
    """
    return CLUSTERING_METHOD_EXPLANATIONS

def get_all_summaries() -> dict:
    """
    Get all clustering method summaries.

    Returns
    -------
    dict
        Complete dictionary of all concise method summaries
    """
    return CLUSTERING_METHOD_SUMMARIES

def get_all_portfolio_insights() -> dict:
    """
    Get all portfolio construction insights.

    Returns
    -------
    dict
        Complete dictionary of portfolio insights for all methods
    """
    return PORTFOLIO_INSIGHTS

def get_all_computational_notes() -> dict:
    """
    Get all computational complexity notes.

    Returns
    -------
    dict
        Complete dictionary of computational notes for all methods
    """
    return COMPUTATIONAL_NOTES