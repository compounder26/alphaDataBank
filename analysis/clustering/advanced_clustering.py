"""
Advanced clustering methods for alpha strategy analysis.

This module implements sophisticated clustering techniques including:
- Hierarchical Risk Parity (HRP) clustering
- Dynamic Time Warping (DTW) for different alpha lifespans
- Rolling correlation windows for time-varying relationships
- Enhanced risk metrics for better clustering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
import logging
import warnings

# Import correlation engine for consistent calculations
from correlation.correlation_engine import CorrelationEngine

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def calculate_rolling_correlation_matrix(
    pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]], 
    windows: List[int] = [60, 120, 252],
    min_common_days: int = 30,
    use_cython: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Calculate multi-scale rolling correlation matrix with time-weighted averaging.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to PnL data
        windows: List of rolling window sizes in days
        min_common_days: Minimum number of common days required
        use_cython: Whether to use Cython acceleration
        
    Returns:
        Tuple of (final weighted correlation matrix, dict of correlation matrices by window)
    """
    correlation_engine = CorrelationEngine(use_cython=use_cython)
    
    # Extract alpha IDs and prepare data
    alpha_ids = []
    pnl_series = {}
    
    for alpha_id, data in pnl_data_dict.items():
        if data is None or 'df' not in data or data['df'] is None:
            continue
        df = data['df']
        if 'pnl' not in df.columns or len(df) < min_common_days:
            continue
        pnl_series[alpha_id] = df['pnl']
        alpha_ids.append(alpha_id)
    
    if len(alpha_ids) < 2:
        logger.warning(f"Not enough alphas for rolling correlation: {len(alpha_ids)}")
        return pd.DataFrame(), {}
    
    # Calculate correlations for each window size
    window_correlations = {}
    
    for window in windows:
        logger.info(f"Calculating rolling correlations with {window}-day window")
        
        # Initialize correlation matrix for this window
        n_alphas = len(alpha_ids)
        window_corr_matrix = pd.DataFrame(
            np.eye(n_alphas), 
            index=alpha_ids, 
            columns=alpha_ids
        )
        
        # Calculate pairwise rolling correlations
        for i, alpha_id1 in enumerate(alpha_ids):
            pnl1 = pnl_series[alpha_id1]
            
            for j, alpha_id2 in enumerate(alpha_ids):
                if i >= j:  # Skip diagonal and lower triangle
                    continue
                    
                pnl2 = pnl_series[alpha_id2]
                
                # Find common dates
                common_dates = pnl1.index.intersection(pnl2.index)
                
                if len(common_dates) < window:
                    continue
                
                # Calculate rolling correlations
                rolling_corrs = []
                for end_idx in range(window, len(common_dates) + 1):
                    start_idx = end_idx - window
                    window_dates = common_dates[start_idx:end_idx]
                    
                    # Get windowed PnL arrays
                    windowed_pnl1 = pnl1.loc[window_dates].values
                    windowed_pnl2 = pnl2.loc[window_dates].values
                    
                    # Calculate correlation for this window
                    corr = correlation_engine.calculate_pairwise(windowed_pnl1, windowed_pnl2)
                    if corr is not None:
                        rolling_corrs.append(corr)
                
                if rolling_corrs:
                    # Weight recent correlations more heavily (exponential decay)
                    weights = np.exp(np.linspace(-2, 0, len(rolling_corrs)))
                    weights /= weights.sum()
                    
                    # Calculate weighted average correlation
                    weighted_corr = np.average(rolling_corrs, weights=weights)
                    
                    # Store in symmetric matrix
                    window_corr_matrix.loc[alpha_id1, alpha_id2] = weighted_corr
                    window_corr_matrix.loc[alpha_id2, alpha_id1] = weighted_corr
        
        window_correlations[f'window_{window}'] = window_corr_matrix
    
    # Combine multiple timescales with weights
    # Shorter windows get higher weight for recent behavior
    window_weights = {
        'window_60': 0.4,
        'window_120': 0.35,
        'window_252': 0.25
    }
    
    # Calculate weighted average of all window correlations
    final_corr = pd.DataFrame(np.zeros((len(alpha_ids), len(alpha_ids))), 
                             index=alpha_ids, columns=alpha_ids)
    
    total_weight = 0
    for window_key, corr_matrix in window_correlations.items():
        if window_key in window_weights:
            weight = window_weights[window_key]
            final_corr += corr_matrix * weight
            total_weight += weight
    
    if total_weight > 0:
        final_corr /= total_weight
    
    # Ensure diagonal is 1
    np.fill_diagonal(final_corr.values, 1.0)
    
    return final_corr, window_correlations


def hierarchical_risk_parity_clustering(
    returns_df: pd.DataFrame,
    linkage_method: str = 'single'
) -> Tuple[np.ndarray, Dict[int, List[str]], np.ndarray]:
    """
    Implement Lopez de Prado's Hierarchical Risk Parity (HRP) clustering method.
    
    Args:
        returns_df: DataFrame with returns (alphas as columns, dates as rows)
        linkage_method: Linkage method for hierarchical clustering
        
    Returns:
        Tuple of (linkage matrix, cluster assignments, optimal weights)
    """
    # Step 1: Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Step 2: Convert correlation to distance
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    
    # Step 3: Perform hierarchical clustering
    condensed_dist = squareform(dist_matrix)
    linkage_matrix = sch.linkage(condensed_dist, method=linkage_method)
    
    # Step 4: Quasi-diagonalization (reorder correlation matrix)
    def get_quasi_diag(link):
        """Extract quasi-diagonal ordering from linkage matrix."""
        link = link.astype(int)
        sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_idx.max() >= num_items:
            sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_idx[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_idx = pd.concat([sort_idx, df0])
            sort_idx = sort_idx.sort_index()
            sort_idx.index = range(sort_idx.shape[0])
        
        return sort_idx.tolist()
    
    sort_idx = get_quasi_diag(linkage_matrix)
    
    # Step 5: Recursive bisection for weight allocation
    def get_recursive_bisection(cov, sort_idx):
        """Calculate HRP weights through recursive bisection."""
        w = pd.Series(1, index=sort_idx)
        c_items = [sort_idx]
        
        while len(c_items) > 0:
            # Bisect the items in two clusters
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), 
                      (len(i) // 2, len(i))) if len(i) > 1]
            
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1] if i + 1 < len(c_items) else []
                
                if not c_items1:
                    continue
                    
                # Calculate cluster variances
                c_var0 = get_cluster_var(cov, c_items0)
                c_var1 = get_cluster_var(cov, c_items1)
                
                # Allocate weights inversely proportional to cluster variance
                alpha = 1 - c_var0 / (c_var0 + c_var1)
                
                w[c_items0] *= alpha
                w[c_items1] *= 1 - alpha
        
        return w
    
    def get_cluster_var(cov, c_items):
        """Calculate variance of a cluster."""
        cov_slice = cov.loc[c_items, c_items].values
        w = np.ones(len(c_items)) / len(c_items)
        return np.dot(w, np.dot(cov_slice, w))
    
    # Calculate covariance matrix with shrinkage
    lw = LedoitWolf()
    cov_matrix = pd.DataFrame(
        lw.fit(returns_df).covariance_,
        index=returns_df.columns,
        columns=returns_df.columns
    )
    
    # Get HRP weights
    sorted_columns = [returns_df.columns[i] for i in sort_idx]
    hrp_weights = get_recursive_bisection(cov_matrix, sorted_columns)
    
    # Step 6: Extract clusters from dendrogram
    # Cut the dendrogram to get optimal number of clusters
    max_d = np.percentile(linkage_matrix[:, 2], 70)  # Cut at 70th percentile
    clusters = sch.fcluster(linkage_matrix, max_d, criterion='distance')
    
    # Create cluster dictionary
    cluster_dict = {}
    for idx, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(returns_df.columns[idx])
    
    return linkage_matrix, cluster_dict, hrp_weights.values


def calculate_advanced_risk_metrics(
    pnl_series: pd.Series,
    market_returns: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics for clustering.
    
    Args:
        pnl_series: Series of cumulative PnL values
        market_returns: Optional market returns for regime analysis
        
    Returns:
        Dictionary of risk metrics
    """
    # Calculate returns from cumulative PnL
    returns = pnl_series.pct_change().dropna()
    
    if len(returns) < 20:
        return {}
    
    metrics = {}
    
    # Basic statistics
    metrics['mean_return'] = returns.mean()
    metrics['std_return'] = returns.std()
    metrics['skewness'] = returns.skew()
    metrics['excess_kurtosis'] = returns.kurtosis()
    
    # Downside risk metrics
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        metrics['downside_deviation'] = downside_returns.std()
        metrics['sortino_ratio'] = metrics['mean_return'] / metrics['downside_deviation'] * np.sqrt(252)
    
    # Value at Risk and Conditional VaR
    metrics['var_95'] = returns.quantile(0.05)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
    
    # Maximum drawdown and duration
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    metrics['max_drawdown'] = drawdown.min()
    
    # Drawdown duration
    drawdown_start = None
    max_duration = 0
    current_duration = 0
    
    for i, dd in enumerate(drawdown):
        if dd < 0:
            if drawdown_start is None:
                drawdown_start = i
            current_duration = i - drawdown_start
        else:
            if current_duration > max_duration:
                max_duration = current_duration
            drawdown_start = None
            current_duration = 0
    
    metrics['max_drawdown_duration'] = max(max_duration, current_duration)
    
    # Calmar ratio
    if metrics['max_drawdown'] != 0:
        annual_return = metrics['mean_return'] * 252
        metrics['calmar_ratio'] = annual_return / abs(metrics['max_drawdown'])
    
    # Hurst exponent (measure of trending vs mean reversion)
    try:
        from hurst import compute_Hc
        H, c, data = compute_Hc(returns.values, kind='price', simplified=True)
        metrics['hurst_exponent'] = H
    except:
        # If hurst package not available, use simple R/S analysis
        metrics['hurst_exponent'] = calculate_hurst_simple(returns.values)
    
    # Stability score (inverse of rolling volatility's volatility)
    rolling_vol = returns.rolling(20).std()
    if len(rolling_vol.dropna()) > 0:
        metrics['stability_score'] = 1 / (1 + rolling_vol.std())
    
    # Autocorrelation structure
    metrics['acf_lag1'] = returns.autocorr(lag=1) if len(returns) > 1 else 0
    metrics['acf_lag5'] = returns.autocorr(lag=5) if len(returns) > 5 else 0
    metrics['acf_lag20'] = returns.autocorr(lag=20) if len(returns) > 20 else 0
    
    # Regime-dependent metrics
    if market_returns is not None and len(market_returns) > 0:
        # Align dates
        common_dates = returns.index.intersection(market_returns.index)
        if len(common_dates) > 20:
            aligned_returns = returns.loc[common_dates]
            aligned_market = market_returns.loc[common_dates]
            
            # Bull/bear performance
            bull_mask = aligned_market > 0
            bear_mask = aligned_market <= 0
            
            if bull_mask.sum() > 0:
                metrics['bull_performance'] = aligned_returns[bull_mask].mean()
            if bear_mask.sum() > 0:
                metrics['bear_performance'] = aligned_returns[bear_mask].mean()
            
            # Crisis beta (correlation during market downturns)
            crisis_threshold = aligned_market.quantile(0.1)
            crisis_mask = aligned_market <= crisis_threshold
            if crisis_mask.sum() > 5:
                crisis_corr = aligned_returns[crisis_mask].corr(aligned_market[crisis_mask])
                metrics['crisis_beta'] = crisis_corr
    
    return metrics


def calculate_hurst_simple(returns: np.ndarray) -> float:
    """
    Simple Hurst exponent calculation using R/S analysis.
    
    Args:
        returns: Array of returns
        
    Returns:
        Hurst exponent estimate
    """
    if len(returns) < 20:
        return 0.5  # Default to random walk
    
    # Calculate R/S for different time scales
    scales = []
    rs_values = []
    
    for scale in range(10, min(len(returns) // 2, 100), 5):
        rs_list = []
        
        for start in range(0, len(returns) - scale, scale):
            segment = returns[start:start + scale]
            
            # Calculate mean-adjusted cumulative sum
            mean = np.mean(segment)
            cumsum = np.cumsum(segment - mean)
            
            # Calculate range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Calculate standard deviation
            S = np.std(segment, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            scales.append(scale)
            rs_values.append(np.mean(rs_list))
    
    if len(scales) < 2:
        return 0.5
    
    # Fit log(R/S) = H * log(n) + constant
    log_scales = np.log(scales)
    log_rs = np.log(rs_values)
    
    # Linear regression
    coeffs = np.polyfit(log_scales, log_rs, 1)
    H = coeffs[0]
    
    # Bound between 0 and 1
    return np.clip(H, 0, 1)


def apply_ledoit_wolf_shrinkage(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Ledoit-Wolf shrinkage to correlation matrix for improved stability.
    
    Args:
        correlation_matrix: Original correlation matrix
        
    Returns:
        Shrunk correlation matrix
    """
    try:
        # Convert correlation to covariance (assume unit variances)
        cov_matrix = correlation_matrix.values
        
        # Ensure the matrix is symmetric
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Make matrix positive semi-definite by eigenvalue clipping
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
        cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Ensure diagonal is 1 (correlation matrix property)
        np.fill_diagonal(cov_matrix, 1.0)
        
        # Apply Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        
        # LedoitWolf expects data matrix, so we create synthetic returns
        # that would produce our correlation matrix
        n_assets = cov_matrix.shape[0]
        n_samples = max(100, n_assets * 3)  # Ensure sufficient samples
        
        # Generate synthetic returns with the target correlation structure
        np.random.seed(42)  # For reproducibility
        try:
            # Suppress the specific warning about covariance matrix
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='covariance is not symmetric positive-semidefinite')
                synthetic_returns = np.random.multivariate_normal(
                    mean=np.zeros(n_assets),
                    cov=cov_matrix,
                    size=n_samples
                )
        except np.linalg.LinAlgError:
            # If still not positive definite, use identity matrix fallback
            logger.warning("Correlation matrix still not positive definite, using regularized version")
            regularized_cov = cov_matrix + 0.01 * np.eye(n_assets)
            synthetic_returns = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=regularized_cov,
                size=n_samples
            )
        
        # Apply shrinkage
        shrunk_cov = lw.fit(synthetic_returns).covariance_
        
        # Convert back to correlation matrix
        diag_sqrt = np.sqrt(np.diag(shrunk_cov))
        shrunk_corr = shrunk_cov / np.outer(diag_sqrt, diag_sqrt)
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(shrunk_corr, 1.0)
        
        shrunk_df = pd.DataFrame(shrunk_corr, 
                               index=correlation_matrix.index,
                               columns=correlation_matrix.columns)
        
        logger.info(f"Applied Ledoit-Wolf shrinkage with intensity {lw.shrinkage_:.3f}")
        
        return shrunk_df
        
    except Exception as e:
        logger.warning(f"Error applying Ledoit-Wolf shrinkage: {e}")
        return correlation_matrix


def find_optimal_clusters(
    distance_matrix: np.ndarray,
    max_clusters: int = 20,
    use_multiple_criteria: bool = True
) -> Tuple[int, Dict[int, float]]:
    """
    Determine optimal number of clusters using multiple criteria.
    
    Args:
        distance_matrix: Pairwise distance matrix
        max_clusters: Maximum number of clusters to test
        use_multiple_criteria: Use multiple cluster validation metrics
        
    Returns:
        Tuple of (optimal number of clusters, scores dictionary)
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.cluster import AgglomerativeClustering
    
    n_samples = distance_matrix.shape[0]
    max_clusters = min(max_clusters, n_samples // 2)
    
    scores = {}
    
    for n_clusters in range(2, max_clusters + 1):
        # Perform clustering (using metric instead of affinity for newer scikit-learn)
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
        except TypeError:
            # Fallback for older scikit-learn versions
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            )
        labels = clusterer.fit_predict(distance_matrix)
        
        # Calculate multiple metrics
        metrics_dict = {'n_clusters': n_clusters}
        
        # Silhouette score
        try:
            silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
            metrics_dict['silhouette'] = silhouette
        except Exception as e:
            logger.warning(f"Error calculating silhouette score for {n_clusters} clusters: {e}")
            metrics_dict['silhouette'] = 0
        
        if use_multiple_criteria:
            # Davies-Bouldin index (lower is better)
            try:
                # Convert distance matrix to feature matrix using MDS
                from sklearn.manifold import MDS
                mds = MDS(n_components=min(10, distance_matrix.shape[0] - 1), 
                         dissimilarity='precomputed', random_state=42)
                features_approx = mds.fit_transform(distance_matrix)
                
                from sklearn.metrics import davies_bouldin_score
                db_score = davies_bouldin_score(features_approx, labels)
                metrics_dict['davies_bouldin'] = db_score
            except:
                metrics_dict['davies_bouldin'] = float('inf')
            
            # Gap statistic approximation
            inertia = calculate_clustering_inertia(distance_matrix, labels)
            metrics_dict['inertia'] = inertia
            
            # Cluster balance (evenness of cluster sizes)
            cluster_sizes = np.bincount(labels)
            balance = 1 - (cluster_sizes.std() / cluster_sizes.mean()) if cluster_sizes.mean() > 0 else 0
            metrics_dict['balance'] = balance
        
        scores[n_clusters] = metrics_dict
    
    if use_multiple_criteria:
        # Multi-criteria optimization
        optimal_k = find_optimal_k_multi_criteria(scores)
    else:
        # Simple silhouette-based selection
        silhouette_values = [scores[k]['silhouette'] for k in sorted(scores.keys())]
        
        # Elbow detection on silhouette scores
        if len(silhouette_values) > 2:
            first_derivative = np.diff(silhouette_values)
            second_derivative = np.diff(first_derivative)
            
            # Find elbow point (maximum curvature)
            elbow_idx = np.argmax(np.abs(second_derivative)) + 2
            optimal_k = min(elbow_idx, max_clusters)
        else:
            optimal_k = max(scores.keys(), key=lambda k: scores[k]['silhouette'])
    
    return optimal_k, scores


def create_minimum_spanning_tree(correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a minimum spanning tree from correlation matrix.
    
    Args:
        correlation_matrix: Pairwise correlation matrix
        
    Returns:
        Dictionary with MST edges and node positions
    """
    import networkx as nx
    
    # Convert correlation to distance
    distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for node in correlation_matrix.index:
        G.add_node(node)
    
    # Add weighted edges
    for i, node1 in enumerate(correlation_matrix.index):
        for j, node2 in enumerate(correlation_matrix.columns):
            if i < j:  # Only upper triangle
                weight = distance_matrix.iloc[i, j]
                if not np.isnan(weight):
                    G.add_edge(node1, node2, weight=weight)
    
    # Compute minimum spanning tree
    mst = nx.minimum_spanning_tree(G)
    
    # Calculate layout for visualization
    pos = nx.spring_layout(mst, k=1/np.sqrt(len(mst.nodes())), iterations=50)
    
    # Extract edges with weights for visualization
    edges = []
    for edge in mst.edges(data=True):
        edges.append({
            'source': edge[0],
            'target': edge[1],
            'weight': edge[2]['weight'],
            'correlation': correlation_matrix.loc[edge[0], edge[1]]
        })
    
    return {
        'nodes': list(mst.nodes()),
        'edges': edges,
        'positions': pos,
        'graph': mst
    }


def calculate_clustering_inertia(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate clustering inertia from distance matrix.
    
    Args:
        distance_matrix: Pairwise distance matrix
        labels: Cluster labels
        
    Returns:
        Total within-cluster sum of squared distances
    """
    unique_labels = np.unique(labels)
    total_inertia = 0.0
    
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        
        if len(cluster_indices) > 1:
            # Sum of pairwise distances within this cluster
            cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            cluster_inertia = np.sum(cluster_distances) / (2 * len(cluster_indices))
            total_inertia += cluster_inertia
    
    return float(total_inertia)


def find_optimal_k_multi_criteria(scores: Dict[int, Dict[str, float]]) -> int:
    """
    Find optimal number of clusters using multiple criteria.
    
    Args:
        scores: Dictionary of clustering scores for different k values
        
    Returns:
        Optimal number of clusters
    """
    if not scores:
        return 2
    
    k_values = sorted(scores.keys())
    
    # Normalize all metrics to [0, 1]
    normalized_scores = {}
    
    for k in k_values:
        normalized_scores[k] = {}
        
        # Silhouette: higher is better, already in [-1, 1], map to [0, 1]
        silhouette = scores[k].get('silhouette', 0)
        normalized_scores[k]['silhouette'] = (silhouette + 1) / 2
        
        # Davies-Bouldin: lower is better, invert and normalize
        if 'davies_bouldin' in scores[k]:
            db_values = [scores[kk]['davies_bouldin'] for kk in k_values 
                        if 'davies_bouldin' in scores[kk] and scores[kk]['davies_bouldin'] < float('inf')]
            if db_values:
                db_max = max(db_values)
                db_min = min(db_values)
                if db_max > db_min:
                    db_normalized = 1 - (scores[k]['davies_bouldin'] - db_min) / (db_max - db_min)
                else:
                    db_normalized = 0.5
                normalized_scores[k]['davies_bouldin'] = max(0, db_normalized)
        
        # Balance: higher is better, already in [0, 1]
        if 'balance' in scores[k]:
            normalized_scores[k]['balance'] = max(0, min(1, scores[k]['balance']))
        
        # Inertia: lower is better (for gap statistic approximation)
        if 'inertia' in scores[k]:
            inertia_values = [scores[kk]['inertia'] for kk in k_values if 'inertia' in scores[kk]]
            if inertia_values:
                inertia_max = max(inertia_values)
                inertia_min = min(inertia_values)
                if inertia_max > inertia_min:
                    inertia_normalized = 1 - (scores[k]['inertia'] - inertia_min) / (inertia_max - inertia_min)
                else:
                    inertia_normalized = 0.5
                normalized_scores[k]['inertia'] = max(0, inertia_normalized)
    
    # Calculate composite score with weights
    weights = {
        'silhouette': 0.4,
        'davies_bouldin': 0.2,
        'balance': 0.2,
        'inertia': 0.2
    }
    
    composite_scores = {}
    for k in k_values:
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in normalized_scores[k]:
                score += weight * normalized_scores[k][metric]
                total_weight += weight
        
        if total_weight > 0:
            composite_scores[k] = score / total_weight
        else:
            composite_scores[k] = 0
    
    # Find k with maximum composite score
    optimal_k = max(composite_scores.keys(), key=lambda k: composite_scores[k])
    
    logger.info(f"Multi-criteria optimization selected k={optimal_k} with score {composite_scores[optimal_k]:.3f}")
    
    return optimal_k


def enhanced_hierarchical_clustering(correlation_matrix: pd.DataFrame,
                                   method: str = 'ward',
                                   apply_shrinkage: bool = True,
                                   find_optimal_k: bool = True) -> Dict[str, Any]:
    """
    Enhanced hierarchical clustering with multiple improvements.
    
    Args:
        correlation_matrix: Correlation matrix
        method: Linkage method ('ward', 'complete', 'average', 'single')
        apply_shrinkage: Apply Ledoit-Wolf shrinkage
        find_optimal_k: Automatically find optimal number of clusters
        
    Returns:
        Dictionary with clustering results
    """
    if correlation_matrix.empty:
        return {'error': 'empty_correlation_matrix'}
    
    # Apply shrinkage if requested
    if apply_shrinkage:
        logger.info("Applying Ledoit-Wolf shrinkage to correlation matrix")
        shrunk_corr = apply_ledoit_wolf_shrinkage(correlation_matrix)
    else:
        shrunk_corr = correlation_matrix.copy()
    
    # Convert correlation to distance
    if method == 'ward':
        # Ward linkage requires Euclidean distance
        distance_matrix = np.sqrt(2 * (1 - shrunk_corr))
    else:
        # Use angular distance for other methods
        distance_matrix = np.sqrt(0.5 * (1 - shrunk_corr))
    
    # Perform hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = sch.linkage(condensed_dist, method=method)
    
    results = {
        'linkage_matrix': linkage_matrix,
        'distance_matrix': distance_matrix,
        'method': method,
        'shrinkage_applied': apply_shrinkage
    }
    
    # Find optimal number of clusters if requested
    if find_optimal_k:
        logger.info("Finding optimal number of clusters")
        optimal_k, cluster_scores = find_optimal_clusters(distance_matrix.values, 
                                                        max_clusters=min(20, len(correlation_matrix) // 2))
        
        # Get cluster labels for optimal k
        labels = sch.fcluster(linkage_matrix, optimal_k, criterion='maxclust')
        
        results.update({
            'optimal_k': optimal_k,
            'cluster_scores': cluster_scores,
            'cluster_labels': labels - 1,  # Convert to 0-based indexing
            'silhouette_score': cluster_scores[optimal_k]['silhouette']
        })
        
        # Create cluster dictionary
        cluster_dict = {}
        for idx, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(correlation_matrix.index[idx])
        
        results['clusters'] = cluster_dict
    
    logger.info(f"Enhanced hierarchical clustering complete. Method: {method}, "
               f"Optimal k: {results.get('optimal_k', 'not calculated')}")
    
    return results


def create_alpha_similarity_network(correlation_matrix: pd.DataFrame,
                                   threshold: float = 0.3,
                                   layout: str = 'spring') -> Dict[str, Any]:
    """
    Create a network representation of alpha similarities.
    
    Args:
        correlation_matrix: Pairwise correlation matrix
        threshold: Correlation threshold for edge creation
        layout: Network layout algorithm ('spring', 'circular', 'random')
        
    Returns:
        Dictionary with network information
    """
    import networkx as nx
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for alpha_id in correlation_matrix.index:
        G.add_node(alpha_id)
    
    # Add edges for correlations above threshold
    edges_added = 0
    for i, alpha1 in enumerate(correlation_matrix.index):
        for j, alpha2 in enumerate(correlation_matrix.columns):
            if i < j:  # Only upper triangle
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    G.add_edge(alpha1, alpha2, 
                             weight=abs(corr),
                             correlation=corr,
                             edge_type='positive' if corr > 0 else 'negative')
                    edges_added += 1
    
    if edges_added == 0:
        logger.warning(f"No edges added with threshold {threshold}. Consider lowering the threshold.")
        return {'error': 'no_edges', 'threshold': threshold}
    
    # Calculate layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Calculate network metrics
    network_metrics = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'average_clustering': nx.average_clustering(G) if G.number_of_edges() > 0 else 0,
        'n_connected_components': nx.number_connected_components(G)
    }
    
    # Find communities using modularity-based partitioning
    try:
        import networkx.algorithms.community as community
        communities = community.greedy_modularity_communities(G)
        community_dict = {}
        for i, community_set in enumerate(communities):
            community_dict[f'community_{i}'] = list(community_set)
        network_metrics['communities'] = community_dict
        network_metrics['modularity'] = community.modularity(G, communities)
    except Exception as e:
        logger.warning(f"Could not calculate communities: {e}")
    
    # Identify central nodes
    if G.number_of_edges() > 0:
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Top 5 central nodes
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        
        network_metrics['top_degree_central'] = top_central
        network_metrics['top_betweenness_central'] = top_betweenness
    
    logger.info(f"Created similarity network: {network_metrics['n_nodes']} nodes, "
               f"{network_metrics['n_edges']} edges, density={network_metrics['density']:.3f}")
    
    return {
        'graph': G,
        'positions': pos,
        'network_metrics': network_metrics,
        'threshold': threshold,
        'layout': layout
    }