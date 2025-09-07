"""
Mathematical explanations for clustering visualization methods in the Alpha DataBank dashboard.
Each explanation is designed for quantitative analysts and portfolio managers.
"""

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

def get_method_explanation(method_key: str) -> dict:
    """
    Retrieve the explanation for a specific clustering method.
    
    Parameters
    ----------
    method_key : str
        The key identifying the clustering method
    
    Returns
    -------
    dict
        Dictionary containing title, explanation, interpretation, and formula
    """
    return CLUSTERING_METHOD_EXPLANATIONS.get(method_key, {
        "title": "Unknown Method",
        "explanation": "No explanation available for this method.",
        "interpretation": "No interpretation guidelines available.",
        "key_formula": "N/A"
    })

def get_all_explanations() -> dict:
    """
    Get all clustering method explanations.
    
    Returns
    -------
    dict
        Complete dictionary of all method explanations
    """
    return CLUSTERING_METHOD_EXPLANATIONS

def format_explanation_for_display(method_key: str, include_formula: bool = True) -> str:
    """
    Format an explanation for display in the dashboard.
    
    Parameters
    ----------
    method_key : str
        The key identifying the clustering method
    include_formula : bool
        Whether to include the mathematical formula
    
    Returns
    -------
    str
        Formatted explanation text suitable for dashboard display
    """
    method = get_method_explanation(method_key)
    
    output = f"**{method['title']}**\n\n"
    output += f"{method['explanation']}\n\n"
    output += f"**How to Interpret:**\n{method['interpretation']}\n"
    
    if include_formula and 'key_formula' in method:
        output += f"\n**Key Formula:** {method['key_formula']}"
    
    return output

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