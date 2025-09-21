"""
Consolidated Alpha Clustering Algorithms Module.

This module provides comprehensive clustering and analysis tools for alpha strategies,
combining standard and advanced techniques including:
- Correlation matrix calculation with various methods
- Dimensionality reduction (PCA, t-SNE, UMAP, MDS)
- Clustering algorithms (HDBSCAN, KMeans, Hierarchical)
- Advanced techniques (HRP, MST, Network Analysis)
- Feature engineering and cluster profiling
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# Set up logging
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add the project root to the path for imports
# Setup project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.bootstrap import setup_project_path
setup_project_path()

# Import database functionality
from database.operations import (
    get_all_alpha_ids_by_region_basic,
    get_regular_alpha_ids_by_region,
    get_pnl_data_for_alphas,
    get_correlation_statistics
)
from config.database_config import REGIONS

# Import correlation engine for correct correlation calculation
from analysis.correlation.correlation_engine import CorrelationEngine

# Import scikit-learn components for dimensionality reduction
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import umap

# Import enhanced feature engineering and validation
try:
    from .feature_engineering import create_enhanced_feature_matrix
    from .validation import validate_clustering_pipeline
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from feature_engineering import create_enhanced_feature_matrix
        from validation import validate_clustering_pipeline
    except ImportError:
        logger.warning("Feature engineering and validation modules not available")
        create_enhanced_feature_matrix = None
        validate_clustering_pipeline = None

# ================================================================================
# CORRELATION CALCULATION FUNCTIONS
# ================================================================================

def apply_correlation_thresholding(correlations: pd.DataFrame, n_samples: int,
                                  threshold_multiplier: float = 2.5) -> pd.DataFrame:
    """
    Apply correlation thresholding based on Paleologo's formula: λ = K√(log n/T).

    Args:
        correlations: Correlation matrix
        n_samples: Number of time samples used to calculate correlations
        threshold_multiplier: Multiplier K in the threshold formula (default: 2.5)

    Returns:
        Thresholded correlation matrix
    """
    n_alphas = correlations.shape[0]
    if n_alphas <= 1 or n_samples <= 1:
        return correlations

    # Calculate threshold: λ = K√(log n/T)
    threshold = threshold_multiplier * np.sqrt(np.log(n_alphas) / n_samples)
    threshold = min(threshold, 0.95)  # Cap at 95% to avoid removing all correlations

    # Apply clipping operator: thresh_λ(ρ_i,j) := ρ_i,j * 1{|ρ_i,j| > λ}
    thresholded_corr = correlations * (np.abs(correlations) > threshold)

    # Ensure diagonal remains 1
    np.fill_diagonal(thresholded_corr.values, 1.0)

    logger.info(f"Applied correlation thresholding with λ={threshold:.3f}, "
               f"removed {((np.abs(correlations) <= threshold) & (correlations != 1)).sum().sum()} weak correlations")

    return thresholded_corr


def calculate_correlation_matrix(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]],
                                min_common_days: int = 60,
                                use_cython: bool = True,
                                apply_thresholding: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Calculate pairwise Pearson correlation matrix from daily PnL data with optional thresholding.
    Uses the same correlation calculation method as scripts/calculate_correlations.py

    Args:
        pnl_data_dict: Dictionary mapping alpha_id to a dictionary containing DataFrame with PnL data
                      Format: {alpha_id: {'df': dataframe}}
        min_common_days: Minimum number of common trading days required
        use_cython: Whether to use Cython acceleration (default: True)
        apply_thresholding: Apply correlation thresholding to remove weak correlations (default: False)

    Returns:
        If apply_thresholding is False: DataFrame containing the raw correlation matrix
        If apply_thresholding is True: Dictionary with 'raw' and 'thresholded' correlation matrices
    """
    # Initialize correlation engine with same method as main scripts
    correlation_engine = CorrelationEngine(use_cython=use_cython)

    # Extract alpha IDs and validate data
    alpha_ids = []
    pnl_arrays = {}  # Store numpy arrays for correlation calculation

    for alpha_id, data in pnl_data_dict.items():
        # Ensure we have a valid dataframe with at least some data
        if data is None or 'df' not in data or data['df'] is None or len(data['df']) < min_common_days:
            print(f"Warning: Skipping alpha {alpha_id} due to insufficient data")
            continue

        df = data['df']
        if 'pnl' not in df.columns:
            print(f"Warning: Missing 'pnl' column for alpha {alpha_id}")
            continue

        # Store cumulative PnL as numpy array (this is what correlation engine expects)
        pnl_arrays[alpha_id] = df['pnl'].values
        alpha_ids.append(alpha_id)

    n_alphas = len(alpha_ids)
    if n_alphas < 2:
        print(f"Warning: Not enough alphas with valid data for correlation calculation. Found {n_alphas}.")
        return pd.DataFrame()

    # Prepare correlation matrix
    corr_matrix = pd.DataFrame(np.eye(n_alphas),
                             index=alpha_ids,
                             columns=alpha_ids)

    # Calculate correlations between all pairs of alphas using the correct method
    for i in range(n_alphas):
        alpha_id1 = alpha_ids[i]
        pnl1 = pnl_arrays[alpha_id1]

        for j in range(i+1, n_alphas):
            alpha_id2 = alpha_ids[j]
            pnl2 = pnl_arrays[alpha_id2]

            try:
                # Find common dates by aligning the PnL series
                # Get original dataframes to check dates
                df1 = pnl_data_dict[alpha_id1]['df']
                df2 = pnl_data_dict[alpha_id2]['df']

                # Find common dates
                common_dates = df1.index.intersection(df2.index)

                if len(common_dates) >= min_common_days:
                    # Get aligned PnL arrays for common dates
                    aligned_pnl1 = df1.loc[common_dates, 'pnl'].values
                    aligned_pnl2 = df2.loc[common_dates, 'pnl'].values

                    # Use the correlation engine's method (same as scripts/calculate_correlations.py)
                    # This correctly calculates percentage returns internally
                    correlation = correlation_engine.calculate_pairwise(
                        aligned_pnl1,
                        aligned_pnl2
                    )

                    if correlation is not None:
                        # Store in symmetric matrix
                        corr_matrix.loc[alpha_id1, alpha_id2] = correlation
                        corr_matrix.loc[alpha_id2, alpha_id1] = correlation
                    else:
                        print(f"Warning: Correlation calculation returned None for {alpha_id1}/{alpha_id2}")
                else:
                    print(f"Warning: Not enough common dates ({len(common_dates)}) for {alpha_id1}/{alpha_id2}")

            except Exception as e:
                print(f"Warning: Error calculating correlation for {alpha_id1}/{alpha_id2}: {e}")

    # Apply correlation thresholding if requested
    if apply_thresholding and not corr_matrix.empty:
        # Estimate average number of samples from PnL data
        avg_samples = np.mean([len(data['df']) for data in pnl_data_dict.values()
                              if data and 'df' in data and data['df'] is not None])
        thresholded_matrix = apply_correlation_thresholding(corr_matrix.copy(), int(avg_samples))

        # Log information about thresholding
        n_zeroed = ((thresholded_matrix == 0) & (corr_matrix != 0)).sum().sum()
        logger.info(f"Correlation thresholding applied: zeroed {n_zeroed} out of {corr_matrix.size - len(corr_matrix)} correlations")

        return {
            'raw': corr_matrix,
            'thresholded': thresholded_matrix
        }

    return corr_matrix


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

                    window_pnl1 = pnl1.loc[window_dates].values
                    window_pnl2 = pnl2.loc[window_dates].values

                    corr = correlation_engine.calculate_pairwise(window_pnl1, window_pnl2)
                    if corr is not None:
                        rolling_corrs.append(corr)

                if rolling_corrs:
                    # Use median for robustness against outliers
                    median_corr = np.median(rolling_corrs)
                    window_corr_matrix.loc[alpha_id1, alpha_id2] = median_corr
                    window_corr_matrix.loc[alpha_id2, alpha_id1] = median_corr

        window_correlations[window] = window_corr_matrix

    # Calculate weighted average correlation matrix
    if window_correlations:
        # Weight more recent/shorter windows higher
        weights = np.array([1.0 / (i + 1) for i in range(len(windows))])
        weights = weights / weights.sum()

        weighted_corr = None
        for weight, window in zip(weights, windows):
            if window in window_correlations:
                if weighted_corr is None:
                    weighted_corr = window_correlations[window] * weight
                else:
                    weighted_corr += window_correlations[window] * weight

        return weighted_corr, window_correlations

    return pd.DataFrame(), {}


def apply_correlation_regularization(correlation_matrix: pd.DataFrame, shrinkage_factor: float = 0.1) -> pd.DataFrame:
    """
    Apply simple regularization to correlation matrix for improved stability.

    Uses a simple shrinkage approach: C_reg = (1-λ)*C + λ*I
    where C is the correlation matrix, I is identity, and λ is shrinkage factor.

    Args:
        correlation_matrix: Input correlation matrix
        shrinkage_factor: Shrinkage intensity (0.0 = no shrinkage, 1.0 = identity matrix)

    Returns:
        Regularized correlation matrix
    """
    if correlation_matrix.empty or correlation_matrix.shape[0] < 2:
        return correlation_matrix

    try:
        # Ensure the matrix is symmetric
        corr_values = correlation_matrix.values
        corr_values = (corr_values + corr_values.T) / 2

        # Apply simple shrinkage regularization
        n = corr_values.shape[0]
        identity = np.eye(n)

        # Shrinkage: (1-λ)*C + λ*I
        regularized = (1 - shrinkage_factor) * corr_values + shrinkage_factor * identity

        # Ensure diagonal is exactly 1
        np.fill_diagonal(regularized, 1.0)

        # Create DataFrame with same index/columns
        regularized_df = pd.DataFrame(
            regularized,
            index=correlation_matrix.index,
            columns=correlation_matrix.columns
        )

        logger.info(f"Applied correlation regularization with shrinkage factor: {shrinkage_factor:.3f}")
        return regularized_df

    except Exception as e:
        logger.warning(f"Correlation regularization failed: {e}, returning original matrix")
        return correlation_matrix


# ================================================================================
# DIMENSIONALITY REDUCTION FUNCTIONS
# ================================================================================

def mds_on_correlation_matrix(corr_matrix: pd.DataFrame,
                            distance_type: str = 'euclidean') -> pd.DataFrame:
    """
    Apply Multidimensional Scaling (MDS) on the correlation matrix to get 2D coordinates.

    Args:
        corr_matrix: Correlation matrix (DataFrame)
        distance_type: Type of distance metric ('euclidean', 'angular', or 'simple')

    Returns:
        DataFrame with alpha_id as index and x, y coordinates as columns
    """
    # Convert correlation to dissimilarity using proper distance metric
    if distance_type == 'euclidean':
        # Euclidean distance in correlation space: d = sqrt(2 * (1 - corr))
        # This properly handles negative correlations
        dissimilarity_matrix = np.sqrt(2 * (1 - corr_matrix))
    elif distance_type == 'angular':
        # Angular distance: d = sqrt(0.5 * (1 - corr))
        # Maps correlation to angle between vectors
        dissimilarity_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    else:
        # Simple dissimilarity (less mathematically correct)
        dissimilarity_matrix = 1 - corr_matrix

    # Ensure diagonal is 0
    np.fill_diagonal(dissimilarity_matrix.values, 0)

    # Apply MDS
    mds = MDS(n_components=2,
             dissimilarity='precomputed',
             random_state=0,
             metric=True)  # Use metric MDS for better preservation of distances

    # Transform to 2D coordinates
    coords = mds.fit_transform(dissimilarity_matrix)

    # Create DataFrame with results
    result_df = pd.DataFrame(coords,
                           index=corr_matrix.index,
                           columns=['x', 'y'])

    return result_df


def determine_optimal_components_for_clustering(method: str,
                                              features_df: pd.DataFrame,
                                              variance_threshold: float = 0.95) -> int:
    """
    Determine optimal number of components for clustering based on the dimensionality reduction method.

    Args:
        method: Dimensionality reduction method ('pca', 'tsne', 'umap')
        features_df: DataFrame with features
        variance_threshold: For PCA, variance explained threshold (default: 0.95)

    Returns:
        Optimal number of components for clustering
    """
    n_features = features_df.shape[1]
    n_samples = features_df.shape[0]

    if method.lower() == 'pca':
        # For PCA: Use components that explain up to variance_threshold of total variance
        try:
            # Quick PCA test to determine variance explained
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df.fillna(features_df.median()))

            pca_test = PCA().fit(scaled_features)
            cumsum_variance = np.cumsum(pca_test.explained_variance_ratio_)

            # Find number of components for desired variance threshold
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1

            # Cap at reasonable limits for performance and avoid overfitting
            max_components = min(50, n_features, n_samples // 2)
            n_components = min(n_components, max_components)

            print(f"  PCA: Using {n_components} components (capturing {cumsum_variance[n_components-1]:.1%} variance)")
            return max(2, n_components)  # Ensure at least 2 components

        except Exception as e:
            print(f"  Warning: PCA component estimation failed: {e}, using default")
            return min(10, n_features)

    elif method.lower() == 'tsne':
        # t-SNE works best with low dimensions (2-10), but Barnes-Hut only supports up to 3D
        optimal = min(10, n_features, n_samples // 10)  # Allow up to 10
        print(f"  t-SNE: Using {optimal} components for clustering (low-dim optimal)")
        return max(2, optimal)

    elif method.lower() == 'umap':
        # UMAP can handle more dimensions and preserves global structure better
        optimal = min(10, n_features, n_samples // 5)
        print(f"  UMAP: Using {optimal} components for clustering (preserves global structure)")
        return max(2, optimal)

    else:
        # Default fallback
        return min(5, n_features)


def apply_dimensionality_reduction(features_df: pd.DataFrame,
                                 method: str = 'tsne',
                                 n_components_for_clustering: Optional[int] = None,
                                 variance_threshold: float = 0.95,
                                 random_state: int = 42,
                                 use_adaptive_params: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Apply dimensionality reduction to the feature matrix with improved parameter selection.
    Returns both 2D coordinates for visualization and optimal-dimensional features for clustering.

    Args:
        features_df: DataFrame with features (alphas as rows, features as columns)
        method: Method to use ('pca', 'tsne', 'umap')
        n_components_for_clustering: Override for clustering dimensions (if None, auto-determined)
        variance_threshold: For PCA, variance explained threshold (default: 0.95)
        random_state: Random seed for reproducibility
        use_adaptive_params: Use data-driven parameter selection

    Returns:
        Tuple of (2D coordinates DataFrame, clustering features DataFrame, info dict)
    """
    from sklearn.preprocessing import StandardScaler
    import warnings

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df.fillna(features_df.median()))

    # Determine optimal components for clustering
    if n_components_for_clustering is None:
        n_components_for_clustering = determine_optimal_components_for_clustering(
            method, features_df, variance_threshold
        )

    info = {
        'method': method,
        'n_samples': len(features_df),
        'n_features': features_df.shape[1],
        'n_components_clustering': n_components_for_clustering,
        'n_components_viz': 2
    }

    if method.lower() == 'pca':
        # PCA for both visualization and clustering
        pca_clustering = PCA(n_components=n_components_for_clustering, random_state=random_state)
        features_clustering = pca_clustering.fit_transform(scaled_features)

        # Get 2D for visualization (might be same if n_components_for_clustering=2)
        if n_components_for_clustering == 2:
            features_2d = features_clustering
            pca_2d = pca_clustering  # Use same model for loadings
            info['variance_explained_2d'] = float(np.sum(pca_clustering.explained_variance_ratio_))
        else:
            pca_2d = PCA(n_components=2, random_state=random_state)
            features_2d = pca_2d.fit_transform(scaled_features)
            info['variance_explained_2d'] = float(np.sum(pca_2d.explained_variance_ratio_))

        info['variance_explained_clustering'] = float(np.sum(pca_clustering.explained_variance_ratio_))

        # Store PCA loadings for feature interpretation
        info['loadings'] = {
            'feature_names': list(features_df.columns),
            'pc1': dict(zip(features_df.columns, pca_2d.components_[0])),
            'pc2': dict(zip(features_df.columns, pca_2d.components_[1]))
        }
        info['variance_explained'] = {
            'pc1': float(pca_2d.explained_variance_ratio_[0]),
            'pc2': float(pca_2d.explained_variance_ratio_[1]),
            'total_2d': float(np.sum(pca_2d.explained_variance_ratio_))
        }

    elif method.lower() == 'tsne':
        # t-SNE with adaptive parameters
        n_samples = len(features_df)

        if use_adaptive_params:
            # Adaptive perplexity based on dataset size
            perplexity = min(30, max(5, n_samples // 10))
            learning_rate = max(10, n_samples / 12)  # sklearn's auto formula
        else:
            perplexity = 30
            learning_rate = 200.0

        # First reduce with PCA if high-dimensional (t-SNE recommendation)
        if features_df.shape[1] > 50:
            pca_prep = PCA(n_components=50, random_state=random_state)
            prep_features = pca_prep.fit_transform(scaled_features)
        else:
            prep_features = scaled_features

        # t-SNE for 2D visualization
        tsne_2d = TSNE(n_components=2,
                       perplexity=perplexity,
                       learning_rate=learning_rate,
                       max_iter=1000,
                       random_state=random_state,
                       method='barnes_hut')  # Fast approximation
        features_2d = tsne_2d.fit_transform(prep_features)

        # t-SNE for clustering dimensions
        if n_components_for_clustering == 2:
            features_clustering = features_2d
        elif n_components_for_clustering <= 3:
            # Barnes-Hut works up to 3D
            tsne_clustering = TSNE(n_components=n_components_for_clustering,
                                  perplexity=perplexity,
                                  learning_rate=learning_rate,
                                  max_iter=1000,
                                  random_state=random_state,
                                  method='barnes_hut')
            features_clustering = tsne_clustering.fit_transform(prep_features)
        else:
            # For >3D, use exact method (slower but works)
            tsne_clustering = TSNE(n_components=n_components_for_clustering,
                                  perplexity=perplexity,
                                  learning_rate=learning_rate,
                                  max_iter=1000,
                                  random_state=random_state,
                                  method='exact')
            features_clustering = tsne_clustering.fit_transform(prep_features)

        info['perplexity'] = perplexity
        info['learning_rate'] = learning_rate

    elif method.lower() == 'umap':
        try:
            # UMAP with adaptive parameters
            n_samples = len(features_df)

            if use_adaptive_params:
                # Adaptive parameters based on dataset size
                n_neighbors = min(15, max(2, n_samples // 20))
                min_dist = 0.1  # Balance between local and global structure
            else:
                n_neighbors = 15
                min_dist = 0.1

            # UMAP for 2D visualization
            umap_2d = umap.UMAP(n_components=2,
                               n_neighbors=n_neighbors,
                               min_dist=min_dist,
                               random_state=random_state)
            features_2d = umap_2d.fit_transform(scaled_features)

            # UMAP for clustering dimensions
            if n_components_for_clustering == 2:
                features_clustering = features_2d
            else:
                umap_clustering = umap.UMAP(n_components=n_components_for_clustering,
                                          n_neighbors=n_neighbors,
                                          min_dist=min_dist,
                                          random_state=random_state)
                features_clustering = umap_clustering.fit_transform(scaled_features)

            info['n_neighbors'] = n_neighbors
            info['min_dist'] = min_dist

        except Exception as e:
            print(f"UMAP failed: {e}, falling back to PCA")
            return apply_dimensionality_reduction(features_df, 'pca',
                                                 n_components_for_clustering,
                                                 variance_threshold,
                                                 random_state,
                                                 use_adaptive_params)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create DataFrames with results
    coords_2d = pd.DataFrame(features_2d,
                            index=features_df.index,
                            columns=['x', 'y'])

    # Clustering features DataFrame
    clustering_cols = [f'component_{i+1}' for i in range(n_components_for_clustering)]
    coords_clustering = pd.DataFrame(features_clustering,
                                    index=features_df.index,
                                    columns=clustering_cols)

    return coords_2d, coords_clustering, info


# ================================================================================
# CLUSTERING ALGORITHM FUNCTIONS
# ================================================================================

def perform_clustering(features_df: pd.DataFrame, method: str = 'hdbscan',
                      min_cluster_size: int = None, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Perform clustering on the feature matrix using HDBSCAN for trading strategies.

    Args:
        features_df: DataFrame with features (alphas as rows, features as columns)
        method: Clustering method ('hdbscan' or 'kmeans')
        min_cluster_size: Minimum cluster size (auto-calculated if None)
        random_state: Random state for reproducibility

    Returns:
        cluster_labels: Array of cluster assignments (-1 for outliers)
        probabilities: Array of cluster membership probabilities
        cluster_info: Dictionary with clustering information and metrics
    """
    print(f"Performing clustering on {features_df.shape[0]} alphas with {features_df.shape[1]} features...")

    # Robust scaling for financial data (handles outliers better than StandardScaler)
    scaler = RobustScaler(quantile_range=(5, 95))  # Ignore extreme 5% outliers
    scaled_features = scaler.fit_transform(features_df.fillna(features_df.median()))

    cluster_info = {
        'method': method,
        'scaler': 'RobustScaler(5-95%)',
        'n_features': features_df.shape[1],
        'n_alphas': features_df.shape[0]
    }

    if method.lower() == 'hdbscan':
        try:
            import hdbscan
        except ImportError:
            print("HDBSCAN not available, falling back to KMeans")
            method = 'kmeans'
        else:
            # Calculate adaptive minimum cluster size based on data size
            if min_cluster_size is None:
                data_size = len(features_df)
                if data_size <= 20:
                    min_cluster_size = 2  # Very permissive for small datasets
                elif data_size <= 50:
                    min_cluster_size = 3  # Allow smaller clusters
                elif data_size <= 100:
                    min_cluster_size = max(2, int(data_size * 0.02))  # 2% of data
                else:
                    min_cluster_size = max(3, int(data_size * 0.015))  # 1.5% for larger datasets

            print(f"  Using HDBSCAN with min_cluster_size={min_cluster_size}")

            # Try adaptive HDBSCAN settings based on data characteristics
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,  # Very permissive - single points can seed clusters
                metric='euclidean',  # Good for scaled financial features
                cluster_selection_method='leaf',  # More permissive than 'eom'
                cluster_selection_epsilon=0.0,  # Let algorithm decide on merging
                alpha=1.0,  # Standard density calculation
                prediction_data=True,  # Enable prediction for new data
                allow_single_cluster=True  # Allow single large cluster if that's what data shows
            )

            cluster_labels = clusterer.fit_predict(scaled_features)
            probabilities = getattr(clusterer, 'probabilities_', np.ones_like(cluster_labels, dtype=float))

            # HDBSCAN results
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            outlier_ratio = (cluster_labels == -1).sum() / len(cluster_labels) if len(cluster_labels) > 0 else 1.0

            print(f"  HDBSCAN result: {n_clusters_found} clusters, {outlier_ratio:.1%} outliers")

            # Store HDBSCAN-specific information
            cluster_info.update({
                'min_cluster_size': min_cluster_size,
                'min_samples': 1,
                'metric': 'euclidean',
                'algorithm': 'hdbscan',
                'condensed_tree': hasattr(clusterer, 'condensed_tree_'),
                'cluster_persistence': getattr(clusterer, 'cluster_persistence_', None)
            })

    if method.lower() == 'kmeans':
        from sklearn.cluster import KMeans

        # Determine optimal k using elbow method
        if min_cluster_size is None:
            data_size = len(features_df)
            max_k = min(10, data_size // 5)
            min_k = 2

            inertias = []
            for k in range(min_k, max_k + 1):
                kmeans_test = KMeans(n_clusters=k, random_state=random_state)
                kmeans_test.fit(scaled_features)
                inertias.append(kmeans_test.inertia_)

            # Find elbow point (simplified)
            if len(inertias) > 2:
                diffs = np.diff(inertias)
                optimal_k = np.argmin(diffs) + min_k + 1
            else:
                optimal_k = min_k
        else:
            optimal_k = min_cluster_size

        print(f"  Using KMeans with k={optimal_k}")

        kmeans = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)

        # Calculate distances to cluster centers for probability-like scores
        distances = kmeans.transform(scaled_features)
        min_distances = distances.min(axis=1)
        max_distance = min_distances.max()
        probabilities = 1 - (min_distances / max_distance)  # Convert distance to probability-like score

        cluster_info.update({
            'n_clusters': optimal_k,
            'inertia': float(kmeans.inertia_),
            'algorithm': 'kmeans'
        })

    # Calculate clustering metrics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = (cluster_labels == -1).sum() if -1 in cluster_labels else 0

    print(f"  Found {n_clusters} clusters with {n_outliers} outliers")

    # Calculate validation metrics if we have enough data
    if n_clusters > 1:
        try:
            # Filter out outliers for metrics calculation
            non_outlier_mask = cluster_labels != -1 if -1 in cluster_labels else np.ones_like(cluster_labels, dtype=bool)

            if non_outlier_mask.sum() > 1:
                silhouette = silhouette_score(scaled_features[non_outlier_mask], cluster_labels[non_outlier_mask])
                calinski_harabasz = calinski_harabasz_score(scaled_features[non_outlier_mask], cluster_labels[non_outlier_mask])

                cluster_info.update({
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski_harabasz)
                })

                print(f"  Silhouette Score: {silhouette:.3f}")
                print(f"  Calinski-Harabasz Score: {calinski_harabasz:.1f}")
        except Exception as e:
            print(f"  Warning: Could not calculate validation metrics: {e}")

    # Store cluster statistics
    if n_clusters > 0:
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
        cluster_sizes = dict(zip(unique_labels.astype(int), counts.astype(int)))
        cluster_info['cluster_sizes'] = cluster_sizes
        cluster_info['n_clusters'] = n_clusters
        cluster_info['n_outliers'] = int(n_outliers)
        cluster_info['outlier_percentage'] = float(n_outliers / len(cluster_labels) * 100) if len(cluster_labels) > 0 else 0

        print(f"  Cluster sizes: {cluster_sizes}")

    return cluster_labels, probabilities, cluster_info


def enhanced_hierarchical_clustering(correlation_matrix: pd.DataFrame,
                                    method: str = 'ward',
                                    apply_shrinkage: bool = True,
                                    find_optimal_k: bool = True) -> Dict[str, Any]:
    """
    Enhanced hierarchical clustering with automatic optimal cluster detection.

    Args:
        correlation_matrix: Correlation matrix of alphas
        method: Linkage method ('ward', 'complete', 'average', 'single')
        apply_shrinkage: Apply Ledoit-Wolf shrinkage before clustering
        find_optimal_k: Automatically find optimal number of clusters

    Returns:
        Dictionary with clustering results and metadata
    """
    if correlation_matrix.empty or correlation_matrix.shape[0] < 2:
        return {'error': 'Insufficient data for hierarchical clustering'}

    try:
        # Apply shrinkage if requested
        if apply_shrinkage:
            corr_matrix = apply_correlation_regularization(correlation_matrix)
        else:
            corr_matrix = correlation_matrix

        # Convert correlation to distance
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(distance_matrix.values, 0)

        # Ensure valid distance matrix
        distance_matrix = distance_matrix.fillna(0)
        distance_matrix = np.clip(distance_matrix, 0, 2)

        # Convert to condensed form for scipy
        condensed_distances = squareform(distance_matrix)

        # Perform hierarchical clustering
        linkage_matrix = sch.linkage(condensed_distances, method=method)

        results = {
            'linkage_matrix': linkage_matrix.tolist(),
            'method': method,
            'n_alphas': len(correlation_matrix),
            'alpha_ids': correlation_matrix.index.tolist()
        }

        # Find optimal number of clusters
        if find_optimal_k:
            optimal_k, cluster_scores = find_optimal_clusters(
                distance_matrix.values,
                linkage_matrix,
                correlation_matrix.index.tolist()
            )
            results['optimal_k'] = optimal_k
            results['cluster_scores'] = cluster_scores

            # Get cluster assignments for optimal k
            cluster_labels = sch.fcluster(linkage_matrix, optimal_k, criterion='maxclust')
            results['cluster_labels'] = cluster_labels.tolist()

            # Calculate cluster statistics
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            results['cluster_sizes'] = dict(zip(unique_labels.astype(int), counts.astype(int)))

        # Calculate dendrogram ordering
        dendrogram = sch.dendrogram(linkage_matrix, no_plot=True)
        results['dendrogram_order'] = dendrogram['leaves']

        return results

    except Exception as e:
        logger.error(f"Hierarchical clustering failed: {e}")
        return {'error': str(e)}


def hierarchical_risk_parity_clustering(
    correlation_matrix: pd.DataFrame,
    returns_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Implement Hierarchical Risk Parity (HRP) clustering for portfolio construction.
    Based on Lopez de Prado's HRP algorithm.

    Args:
        correlation_matrix: Correlation matrix of alpha returns
        returns_data: Optional returns data for risk calculation

    Returns:
        Dictionary with HRP clustering results and risk allocations
    """
    if correlation_matrix.empty or correlation_matrix.shape[0] < 2:
        return {'error': 'Insufficient data for HRP clustering'}

    try:
        n_alphas = len(correlation_matrix)

        # Step 1: Tree Clustering
        # Convert correlation to distance
        distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
        np.fill_diagonal(distance_matrix.values, 0)

        # Hierarchical clustering
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = sch.linkage(condensed_distances, method='single')

        # Step 2: Quasi-Diagonalization (Seriation)
        def get_quasi_diag(link):
            # Recursively rearrange correlation matrix
            link = link.astype(int)
            sort_idx = []

            def recurse(node):
                if node < n_alphas:
                    sort_idx.append(node)
                else:
                    left = int(link[node - n_alphas, 0])
                    right = int(link[node - n_alphas, 1])
                    recurse(left)
                    recurse(right)

            recurse(2 * n_alphas - 2)
            return sort_idx

        sorted_idx = get_quasi_diag(linkage_matrix)

        # Step 3: Recursive Bisection for weights
        def get_cluster_var(cov, c_items):
            # Calculate cluster variance
            cov_slice = cov.iloc[c_items, c_items]
            w = np.ones(len(c_items)) / len(c_items)  # Equal weight within cluster
            cluster_var = np.dot(w, np.dot(cov_slice, w))
            return cluster_var

        def recursive_bisection(cov, sort_ix):
            w = pd.Series(1.0, index=sort_ix)
            c_items = [sort_ix]  # Initialize clusters

            while len(c_items) > 0:
                # Bisect each cluster
                c_items_new = []
                for ci in c_items:
                    if len(ci) > 1:
                        # Split cluster
                        n_ci = len(ci)
                        c0 = ci[:n_ci // 2]
                        c1 = ci[n_ci // 2:]

                        # Calculate variance of each sub-cluster
                        var0 = get_cluster_var(cov, c0)
                        var1 = get_cluster_var(cov, c1)

                        # Allocate weights inversely proportional to variance
                        alpha = 1 - var0 / (var0 + var1)

                        # Update weights
                        w[c0] *= alpha
                        w[c1] *= (1 - alpha)

                        # Add sub-clusters for next iteration
                        c_items_new.extend([c0, c1])

                c_items = c_items_new

            return w

        # Calculate covariance from correlation (assume unit variance for simplicity)
        cov_matrix = correlation_matrix.copy()

        # Get HRP weights
        hrp_weights = recursive_bisection(cov_matrix, sorted_idx)
        hrp_weights = hrp_weights / hrp_weights.sum()  # Normalize

        # Create results
        results = {
            'hrp_weights': hrp_weights.to_dict(),
            'sorted_order': [correlation_matrix.index[i] for i in sorted_idx],
            'linkage_matrix': linkage_matrix.tolist(),
            'n_alphas': n_alphas
        }

        # Calculate risk contribution of each alpha
        portfolio_variance = np.dot(hrp_weights, np.dot(cov_matrix, hrp_weights))
        marginal_contributions = np.dot(cov_matrix, hrp_weights)
        risk_contributions = hrp_weights * marginal_contributions / portfolio_variance

        results['risk_contributions'] = pd.Series(
            risk_contributions,
            index=correlation_matrix.index
        ).to_dict()
        results['portfolio_risk'] = float(np.sqrt(portfolio_variance))

        # Cluster assignments based on dendrogram cut
        if n_alphas > 10:
            n_clusters = max(3, n_alphas // 10)
        else:
            n_clusters = max(2, n_alphas // 3)

        cluster_labels = sch.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        results['cluster_labels'] = cluster_labels.tolist()
        results['n_clusters'] = n_clusters

        return results

    except Exception as e:
        logger.error(f"HRP clustering failed: {e}")
        return {'error': str(e)}


def find_optimal_clusters(distance_matrix: np.ndarray,
                         linkage_matrix: np.ndarray,
                         alpha_ids: List[str]) -> Tuple[int, Dict[int, Dict[str, float]]]:
    """
    Find optimal number of clusters using multiple criteria.

    Args:
        distance_matrix: Pairwise distance matrix
        linkage_matrix: Hierarchical clustering linkage matrix
        alpha_ids: List of alpha IDs

    Returns:
        Tuple of (optimal k, scores dictionary)
    """
    n_alphas = len(alpha_ids)
    min_k = 2
    max_k = min(10, n_alphas // 2)

    scores = {}

    for k in range(min_k, max_k + 1):
        try:
            # Get cluster assignments
            labels = sch.fcluster(linkage_matrix, k, criterion='maxclust')

            # Calculate silhouette score
            if k < n_alphas:
                silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
            else:
                silhouette = -1

            # Calculate Calinski-Harabasz (using distance as feature)
            # Note: This is simplified - ideally use original features
            ch_score = 0  # Placeholder

            # Calculate inertia (within-cluster sum of squares)
            inertia = calculate_clustering_inertia(distance_matrix, labels)

            scores[k] = {
                'silhouette': float(silhouette),
                'calinski_harabasz': float(ch_score),
                'inertia': float(inertia),
                'n_clusters': k
            }
        except Exception as e:
            logger.warning(f"Failed to calculate scores for k={k}: {e}")
            scores[k] = {'silhouette': -1, 'inertia': float('inf')}

    # Find optimal k using multiple criteria
    optimal_k = find_optimal_k_multi_criteria(scores)

    return optimal_k, scores


def calculate_clustering_inertia(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate within-cluster sum of squared distances (inertia).

    Args:
        distance_matrix: Pairwise distance matrix
        labels: Cluster labels

    Returns:
        Total inertia
    """
    inertia = 0.0
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        cluster_distances = distance_matrix[mask][:, mask]
        # Sum of squared distances within cluster
        inertia += np.sum(cluster_distances ** 2) / (2 * mask.sum())

    return inertia


def find_optimal_k_multi_criteria(scores: Dict[int, Dict[str, float]]) -> int:
    """
    Find optimal k using multiple criteria with weighted voting.

    Args:
        scores: Dictionary of scores for each k

    Returns:
        Optimal number of clusters
    """
    if not scores:
        return 2

    k_values = list(scores.keys())

    # Criterion 1: Maximum silhouette score
    silhouettes = [scores[k].get('silhouette', -1) for k in k_values]
    best_silhouette_k = k_values[np.argmax(silhouettes)] if max(silhouettes) > 0 else k_values[0]

    # Criterion 2: Elbow method on inertia
    inertias = [scores[k].get('inertia', float('inf')) for k in k_values]
    if len(inertias) > 2 and min(inertias) < float('inf'):
        # Calculate second derivative to find elbow
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        if len(diffs2) > 0:
            elbow_idx = np.argmax(diffs2) + 1  # +1 because of double diff
            best_elbow_k = k_values[min(elbow_idx, len(k_values) - 1)]
        else:
            best_elbow_k = k_values[0]
    else:
        best_elbow_k = k_values[0]

    # Weighted voting (prefer lower k for simplicity when scores are similar)
    candidates = [best_silhouette_k, best_elbow_k]
    optimal_k = int(np.median(candidates))

    # Ensure valid k
    if optimal_k not in k_values:
        optimal_k = k_values[len(k_values) // 2]

    return optimal_k


def create_minimum_spanning_tree(correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a Minimum Spanning Tree (MST) from correlation matrix.
    Useful for identifying the backbone of relationships between alphas.

    Args:
        correlation_matrix: Correlation matrix

    Returns:
        Dictionary with MST edges and properties
    """
    if correlation_matrix.empty or correlation_matrix.shape[0] < 2:
        return {'error': 'Insufficient data for MST'}

    try:
        # Convert correlation to distance
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
        np.fill_diagonal(distance_matrix.values, 0)

        # Create MST using Kruskal's algorithm
        n = len(correlation_matrix)
        edges = []

        # Get all edges with weights
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((
                    distance_matrix.iloc[i, j],
                    correlation_matrix.index[i],
                    correlation_matrix.index[j],
                    correlation_matrix.iloc[i, j]  # Original correlation
                ))

        # Sort edges by weight (distance)
        edges.sort(key=lambda x: x[0])

        # Kruskal's algorithm with union-find
        parent = list(range(n))
        rank = [0] * n
        alpha_to_idx = {alpha: i for i, alpha in enumerate(correlation_matrix.index)}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True

        mst_edges = []
        total_weight = 0

        for weight, alpha1, alpha2, correlation in edges:
            idx1, idx2 = alpha_to_idx[alpha1], alpha_to_idx[alpha2]
            if union(idx1, idx2):
                mst_edges.append({
                    'source': alpha1,
                    'target': alpha2,
                    'weight': float(weight),
                    'correlation': float(correlation)
                })
                total_weight += weight
                if len(mst_edges) == n - 1:
                    break

        # Calculate node degrees (connectivity)
        node_degrees = {}
        for edge in mst_edges:
            node_degrees[edge['source']] = node_degrees.get(edge['source'], 0) + 1
            node_degrees[edge['target']] = node_degrees.get(edge['target'], 0) + 1

        # Identify hub nodes (high degree)
        avg_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
        hub_nodes = [node for node, degree in node_degrees.items() if degree > avg_degree]

        return {
            'edges': mst_edges,
            'n_nodes': n,
            'n_edges': len(mst_edges),
            'total_weight': float(total_weight),
            'average_weight': float(total_weight / len(mst_edges)) if mst_edges else 0,
            'node_degrees': node_degrees,
            'hub_nodes': hub_nodes,
            'average_degree': float(avg_degree)
        }

    except Exception as e:
        logger.error(f"MST creation failed: {e}")
        return {'error': str(e)}


def create_alpha_similarity_network(correlation_matrix: pd.DataFrame,
                                   threshold: float = 0.3) -> Dict[str, Any]:
    """
    Create a similarity network from correlation matrix.

    Args:
        correlation_matrix: Correlation matrix
        threshold: Minimum correlation to create an edge

    Returns:
        Dictionary with network structure and metrics
    """
    if correlation_matrix.empty:
        return {'error': 'Empty correlation matrix'}

    try:
        # Create edges for correlations above threshold
        edges = []
        n = len(correlation_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    edges.append({
                        'source': correlation_matrix.index[i],
                        'target': correlation_matrix.index[j],
                        'weight': float(corr),
                        'abs_weight': float(abs(corr))
                    })

        # Calculate network metrics
        node_degrees = {}
        for edge in edges:
            node_degrees[edge['source']] = node_degrees.get(edge['source'], 0) + 1
            node_degrees[edge['target']] = node_degrees.get(edge['target'], 0) + 1

        # Ensure all nodes are included
        for node in correlation_matrix.index:
            if node not in node_degrees:
                node_degrees[node] = 0

        # Network statistics
        n_edges = len(edges)
        max_edges = n * (n - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        avg_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
        max_degree = max(node_degrees.values()) if node_degrees else 0

        # Identify components (simplified - for full implementation use networkx)
        components = []
        visited = set()

        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for edge in edges:
                if edge['source'] == node and edge['target'] not in visited:
                    dfs(edge['target'], component)
                elif edge['target'] == node and edge['source'] not in visited:
                    dfs(edge['source'], component)

        for node in correlation_matrix.index:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(list(component))

        return {
            'edges': edges,
            'node_degrees': node_degrees,
            'network_metrics': {
                'n_nodes': n,
                'n_edges': n_edges,
                'density': float(density),
                'average_degree': float(avg_degree),
                'max_degree': int(max_degree),
                'n_components': len(components),
                'largest_component_size': max(len(c) for c in components) if components else 0
            },
            'components': components,
            'threshold': threshold
        }

    except Exception as e:
        logger.error(f"Network creation failed: {e}")
        return {'error': str(e)}


def calculate_advanced_risk_metrics(
    returns: pd.DataFrame,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate advanced risk metrics for each alpha.

    Args:
        returns: DataFrame with returns data (alphas as columns)
        confidence_level: Confidence level for VaR/CVaR

    Returns:
        DataFrame with risk metrics for each alpha
    """
    risk_metrics = pd.DataFrame(index=returns.columns)

    for alpha in returns.columns:
        alpha_returns = returns[alpha].dropna()

        if len(alpha_returns) < 20:  # Need minimum data
            continue

        # Basic statistics
        risk_metrics.loc[alpha, 'mean_return'] = alpha_returns.mean()
        risk_metrics.loc[alpha, 'volatility'] = alpha_returns.std()
        risk_metrics.loc[alpha, 'skewness'] = alpha_returns.skew()
        risk_metrics.loc[alpha, 'kurtosis'] = alpha_returns.kurtosis()

        # Downside risk
        downside_returns = alpha_returns[alpha_returns < 0]
        risk_metrics.loc[alpha, 'downside_deviation'] = downside_returns.std() if len(downside_returns) > 0 else 0

        # Value at Risk
        var = np.percentile(alpha_returns, (1 - confidence_level) * 100)
        risk_metrics.loc[alpha, 'var_95'] = var

        # Conditional Value at Risk (Expected Shortfall)
        cvar = alpha_returns[alpha_returns <= var].mean() if len(alpha_returns[alpha_returns <= var]) > 0 else var
        risk_metrics.loc[alpha, 'cvar_95'] = cvar

        # Maximum drawdown
        cumulative = (1 + alpha_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        risk_metrics.loc[alpha, 'max_drawdown'] = drawdown.min()

        # Calmar ratio (annualized return / max drawdown)
        annualized_return = alpha_returns.mean() * 252
        if risk_metrics.loc[alpha, 'max_drawdown'] != 0:
            risk_metrics.loc[alpha, 'calmar_ratio'] = annualized_return / abs(risk_metrics.loc[alpha, 'max_drawdown'])
        else:
            risk_metrics.loc[alpha, 'calmar_ratio'] = np.nan

        # Hurst exponent (simplified)
        try:
            hurst = calculate_hurst_simple(alpha_returns.values)
            risk_metrics.loc[alpha, 'hurst_exponent'] = hurst
        except:
            risk_metrics.loc[alpha, 'hurst_exponent'] = 0.5  # Random walk default

    return risk_metrics


def calculate_hurst_simple(returns: np.ndarray) -> float:
    """
    Calculate simplified Hurst exponent to measure persistence.
    H > 0.5: Trending/persistent
    H = 0.5: Random walk
    H < 0.5: Mean-reverting

    Args:
        returns: Array of returns

    Returns:
        Hurst exponent
    """
    if len(returns) < 20:
        return 0.5

    try:
        # Create cumulative sum
        cumsum = np.cumsum(returns - np.mean(returns))

        # Calculate R/S for different lags
        lags = range(2, min(20, len(returns) // 2))
        rs = []

        for lag in lags:
            # Divide series into chunks
            n_chunks = len(returns) // lag
            if n_chunks < 2:
                continue

            chunks_rs = []
            for i in range(n_chunks):
                chunk = cumsum[i*lag:(i+1)*lag]
                if len(chunk) < 2:
                    continue
                R = chunk.max() - chunk.min()  # Range
                S = np.std(returns[i*lag:(i+1)*lag])  # Std dev
                if S > 0:
                    chunks_rs.append(R / S)

            if chunks_rs:
                rs.append(np.mean(chunks_rs))

        if len(rs) > 2:
            # Fit log(R/S) = H * log(lag) + constant
            log_lags = np.log(list(lags)[:len(rs)])
            log_rs = np.log(rs)

            # Linear regression
            A = np.vstack([log_lags, np.ones(len(log_lags))]).T
            hurst, _ = np.linalg.lstsq(A, log_rs, rcond=None)[0]

            return np.clip(hurst, 0, 1)  # Ensure valid range
        else:
            return 0.5

    except Exception as e:
        logger.warning(f"Hurst calculation failed: {e}")
        return 0.5


# ================================================================================
# FEATURE CALCULATION AND PROFILING FUNCTIONS
# ================================================================================

def calculate_all_features(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]],
                          alpha_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive feature matrix combining all advanced techniques.
    This replaces the old basic feature calculation with full enhanced features.

    Args:
        pnl_data_dict: Dictionary mapping alpha_id to a dictionary containing DataFrame with PnL data
                      Format: {alpha_id: {'df': dataframe}}
        alpha_metadata: DataFrame with alpha metadata

    Returns:
        DataFrame with alpha_id as index and all enhanced features as columns
    """
    print("Calculating comprehensive feature matrix with enhanced methods...")

    # Check if feature engineering is available
    if create_enhanced_feature_matrix is None:
        print("Warning: Enhanced feature engineering not available, using basic features")
        # Fallback to basic features
        feature_list = []

        for alpha_id, data in pnl_data_dict.items():
            if data is None or 'df' not in data or data['df'] is None:
                continue

            df = data['df']
            if 'pnl' not in df.columns or len(df) < 20:
                continue

            # Calculate basic features
            features = {}
            features['alpha_id'] = alpha_id

            # Basic statistics
            pnl_values = df['pnl'].values
            returns = np.diff(pnl_values) / pnl_values[:-1]
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

            if len(returns) > 0:
                features['mean_return'] = np.mean(returns)
                features['volatility'] = np.std(returns)
                features['sharpe'] = features['mean_return'] / features['volatility'] if features['volatility'] > 0 else 0
                features['skewness'] = pd.Series(returns).skew()
                features['kurtosis'] = pd.Series(returns).kurtosis()

                # Drawdown
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                features['max_drawdown'] = np.min(drawdown)

                feature_list.append(features)

        if not feature_list:
            return pd.DataFrame()

        enhanced_features = pd.DataFrame(feature_list).set_index('alpha_id')
    else:
        # Use the enhanced feature engineering
        enhanced_features = create_enhanced_feature_matrix(
            pnl_data_dict,
            include_spiked_cov=False,  # DISABLED: Complex PCA-based features
            include_multiscale=False,  # DISABLED: Temporal volatility features
            include_risk_metrics=True
        )

    if enhanced_features.empty:
        print("Warning: Feature calculation failed, no features generated")
        return pd.DataFrame()

    # Add metadata features if available
    if not alpha_metadata.empty:
        for alpha_id in enhanced_features.index:
            if alpha_id in alpha_metadata.index:
                # Add categorical metadata features
                for feature in ['is_sharpe', 'is_drawdown', 'is_returns']:
                    if feature in alpha_metadata.columns:
                        metadata_feature_name = f'metadata_{feature.replace("is_", "")}'
                        enhanced_features.loc[alpha_id, metadata_feature_name] = alpha_metadata.loc[alpha_id, feature]

                # Add numerical metadata features
                for feature in ['turnover_ratio', 'margin', 'fitness']:
                    if feature in alpha_metadata.columns:
                        metadata_feature_name = f'metadata_{feature}'
                        value = alpha_metadata.loc[alpha_id, feature]
                        # Handle potential non-numeric values
                        if pd.notna(value) and isinstance(value, (int, float)):
                            enhanced_features.loc[alpha_id, metadata_feature_name] = float(value)
                        else:
                            enhanced_features.loc[alpha_id, metadata_feature_name] = np.nan

    # Fill any remaining NaN values with feature medians (more robust than means)
    enhanced_features = enhanced_features.fillna(enhanced_features.median())

    print(f"Created comprehensive feature matrix: {enhanced_features.shape[0]} alphas × {enhanced_features.shape[1]} features")

    # Print feature breakdown by category
    risk_features = [col for col in enhanced_features.columns if 'risk' in col.lower() or 'var' in col or 'drawdown' in col]
    metadata_features = [col for col in enhanced_features.columns if col.startswith('metadata_')]
    stat_features = [col for col in enhanced_features.columns if col not in risk_features + metadata_features]

    print(f"  Feature breakdown:")
    print(f"    - Statistical: {len(stat_features)} features")
    print(f"    - Risk metrics: {len(risk_features)} features")
    print(f"    - Metadata: {len(metadata_features)} features")

    return enhanced_features


def profile_clusters(feature_matrix: pd.DataFrame, cluster_labels: np.ndarray,
                     feature_names: List[str]) -> Dict[str, Any]:
    """
    Profile clusters by analyzing feature distributions and generating interpretations.

    Args:
        feature_matrix: DataFrame with features used for clustering
        cluster_labels: Array of cluster assignments
        feature_names: List of feature names to analyze

    Returns:
        Dictionary with cluster profiles and interpretations
    """
    if feature_matrix.empty or len(cluster_labels) == 0:
        return {}

    profiles = {
        'cluster_statistics': {},
        'cluster_interpretations': {},
        'feature_importance': {}
    }

    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels != -1]) if -1 in unique_labels else len(unique_labels)

    print(f"Profiling {n_clusters} clusters plus outliers...")

    # Calculate overall statistics for comparison
    overall_stats = {}
    for feature in feature_names:
        if feature in feature_matrix.columns:
            overall_stats[feature] = {
                'mean': feature_matrix[feature].mean(),
                'std': feature_matrix[feature].std(),
                'median': feature_matrix[feature].median()
            }

    # Profile each cluster
    for label in unique_labels:
        cluster_name = f'cluster_{label}' if label != -1 else 'outliers'
        mask = cluster_labels == label
        cluster_data = feature_matrix[mask]

        if len(cluster_data) == 0:
            continue

        profiles['cluster_statistics'][cluster_name] = {
            'size': int(mask.sum()),
            'percentage': float(mask.sum() / len(cluster_labels) * 100),
            'feature_means': {},
            'feature_stds': {},
            'distinctive_features': []
        }

        # Analyze each feature
        for feature in feature_names:
            if feature not in cluster_data.columns:
                continue

            cluster_mean = cluster_data[feature].mean()
            cluster_std = cluster_data[feature].std()

            profiles['cluster_statistics'][cluster_name]['feature_means'][feature] = float(cluster_mean)
            profiles['cluster_statistics'][cluster_name]['feature_stds'][feature] = float(cluster_std)

            # Check if this feature is distinctive for this cluster
            if feature in overall_stats:
                overall_mean = overall_stats[feature]['mean']
                overall_std = overall_stats[feature]['std']

                if overall_std > 0:
                    z_score = (cluster_mean - overall_mean) / overall_std
                    if abs(z_score) > 1.5:  # Significantly different
                        profiles['cluster_statistics'][cluster_name]['distinctive_features'].append({
                            'feature': feature,
                            'z_score': float(z_score),
                            'direction': 'high' if z_score > 0 else 'low'
                        })

        # Generate interpretation
        interpretation = generate_cluster_interpretation(cluster_data, feature_matrix, feature_names)
        profiles['cluster_interpretations'][cluster_name] = interpretation

    # Calculate feature importance (which features best separate clusters)
    if n_clusters > 1:
        from sklearn.ensemble import RandomForestClassifier

        try:
            # Use random forest to determine feature importance
            valid_mask = cluster_labels != -1
            if valid_mask.sum() > 10:  # Need enough samples
                X = feature_matrix[valid_mask][feature_names].fillna(0)
                y = cluster_labels[valid_mask]

                rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
                rf.fit(X, y)

                importance_scores = rf.feature_importances_
                feature_importance = dict(zip(feature_names, importance_scores))

                # Sort by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                profiles['feature_importance'] = {k: float(v) for k, v in sorted_features[:10]}  # Top 10

                print(f"  Top discriminating features: {', '.join([f[0] for f in sorted_features[:5]])}")
        except Exception as e:
            print(f"  Warning: Could not calculate feature importance: {e}")

    return profiles


def generate_cluster_interpretation(cluster_data: pd.DataFrame, all_data: pd.DataFrame,
                                   feature_names: List[str]) -> str:
    """
    Generate human-readable interpretation of cluster characteristics.

    Args:
        cluster_data: DataFrame with cluster members
        all_data: DataFrame with all data for comparison
        feature_names: List of feature names

    Returns:
        String interpretation of cluster characteristics
    """
    if cluster_data.empty:
        return "Empty cluster"

    interpretations = []

    # Compare cluster statistics to overall
    for feature in feature_names[:20]:  # Limit to avoid too long descriptions
        if feature not in cluster_data.columns or feature not in all_data.columns:
            continue

        cluster_mean = cluster_data[feature].mean()
        overall_mean = all_data[feature].mean()
        overall_std = all_data[feature].std()

        if overall_std == 0:
            continue

        # Z-score difference
        z_score = (cluster_mean - overall_mean) / overall_std

        # Only include features with significant differences
        if abs(z_score) > 1.5:
            direction = "High" if z_score > 0 else "Low"

            # Clean up feature names for display
            display_name = feature.replace('risk_', '').replace('metadata_', '')
            display_name = display_name.replace('_', ' ').title()

            interpretations.append(f"{direction} {display_name} ({cluster_mean:.3f})")

    if not interpretations:
        return "No distinctive characteristics identified"

    # Return top 3 most significant differences
    return " | ".join(interpretations[:3])


# ================================================================================
# MAIN ORCHESTRATION FUNCTIONS
# ================================================================================

def generate_clustering_data(region: str,
                           apply_validation: bool = True,
                           enable_advanced_methods: bool = True) -> Dict[str, Any]:
    """
    Generate comprehensive clustering data with all enhanced features for a specific region.
    This is now the main clustering function that always uses enhanced features.

    Args:
        region: Region name
        apply_validation: Apply comprehensive validation
        enable_advanced_methods: Enable advanced clustering methods

    Returns:
        Dictionary with enhanced clustering results
    """
    try:
        # Get all alpha IDs for the region
        print(f"Getting alpha IDs for region {region}...")
        alpha_ids = get_regular_alpha_ids_by_region(region)

        if not alpha_ids:
            print(f"No alphas found for region {region}")
            return {}

        print(f"Found {len(alpha_ids)} alphas for region {region}")

        # Get PnL data for these alphas
        print("Fetching PnL data...")
        pnl_data = get_pnl_data_for_alphas(alpha_ids, region)

        if not pnl_data:
            print("No PnL data found for any alphas")
            return {}

        print(f"Fetched PnL data for {len(pnl_data)} alphas")

        # Get alpha metadata
        try:
            print("Fetching correlation statistics...")
            alpha_metadata = get_correlation_statistics(region)
            if not alpha_metadata.empty:
                alpha_metadata.set_index('alpha_id', inplace=True)
                print(f"Fetched metadata for {len(alpha_metadata)} alphas")
            else:
                print("No correlation statistics found")
        except Exception as e:
            print(f"Error fetching correlation statistics: {e}")
            alpha_metadata = pd.DataFrame()

        results = {
            'region': region,
            'timestamp': datetime.now().isoformat(),
            'alpha_count': len(alpha_ids),
            'enhanced_features_used': True,
            'validation_applied': apply_validation,
            'advanced_methods_enabled': enable_advanced_methods
        }

        # Calculate All Features (Enhanced)
        print("Calculating comprehensive feature matrix...")
        all_features = calculate_all_features(pnl_data, alpha_metadata)

        if not all_features.empty and len(all_features) >= 2:
            results['features_count'] = len(all_features.columns)
            results['feature_categories'] = {
                'statistical': len([col for col in all_features.columns if col not in
                                  [c for c in all_features.columns if 'metadata' in c or 'risk' in c.lower()]]),
                'risk': len([col for col in all_features.columns if 'risk' in col.lower() or 'var' in col or 'drawdown' in col]),
                'metadata': len([col for col in all_features.columns if col.startswith('metadata_')])
            }
            print(f"Feature categories: {results['feature_categories']}")

            # Store original features for cluster profiling
            results['feature_matrix'] = all_features.to_dict(orient='index')
            results['feature_names'] = list(all_features.columns)

            # Perform clustering on feature matrix
            print("Performing clustering analysis...")
            try:
                cluster_labels, probabilities, cluster_info = perform_clustering(
                    all_features, method='hdbscan', random_state=42
                )

                # Store clustering results
                results['cluster_labels'] = cluster_labels.tolist()
                results['cluster_probabilities'] = probabilities.tolist()
                results['cluster_info'] = cluster_info

                print(f"  Clustering completed: {cluster_info.get('n_clusters', 0)} clusters, "
                      f"{cluster_info.get('n_outliers', 0)} outliers")

                # Create mapping from alpha_id to cluster
                alpha_ids = list(all_features.index)
                cluster_mapping = {alpha_id: int(label) for alpha_id, label in zip(alpha_ids, cluster_labels)}
                results['alpha_cluster_mapping'] = cluster_mapping

                # Generate cluster profiling
                print("  Profiling main feature-based clusters...")
                main_cluster_profiles = profile_clusters(all_features, cluster_labels, list(all_features.columns))
                results['main_cluster_profiles'] = main_cluster_profiles

            except Exception as e:
                print(f"Error in clustering: {e}")
                results['cluster_labels'] = []
                results['cluster_probabilities'] = []
                results['cluster_info'] = {'error': str(e)}
                results['alpha_cluster_mapping'] = {}

            # Apply dimensionality reduction
            for method in ['tsne', 'umap', 'pca']:
                try:
                    print(f"Applying {method.upper()} with enhanced features...")

                    # Get both 2D coordinates and clustering features
                    coords_2d, coords_clustering, info = apply_dimensionality_reduction(
                        all_features, method=method, use_adaptive_params=True, variance_threshold=0.95
                    )

                    # Perform clustering on the reduced features
                    print(f"  Clustering on {coords_clustering.shape[1]}D {method.upper()} features...")
                    method_cluster_labels, method_probabilities, method_cluster_info = perform_clustering(
                        coords_clustering, method='hdbscan', random_state=42
                    )

                    # Add method-specific cluster information to 2D coordinates
                    coords_with_clusters = coords_2d.copy()
                    coords_with_clusters['cluster'] = method_cluster_labels
                    coords_with_clusters['cluster_probability'] = method_probabilities

                    # Store results
                    results[f'{method}_coords'] = coords_with_clusters.to_dict(orient='index')
                    results[f'{method}_info'] = info
                    results[f'{method}_cluster_labels'] = method_cluster_labels.tolist()
                    results[f'{method}_cluster_probabilities'] = method_probabilities.tolist()
                    results[f'{method}_cluster_info'] = method_cluster_info

                    # Create method-specific alpha-cluster mapping
                    alpha_ids = list(coords_2d.index)
                    method_cluster_mapping = {alpha_id: int(label) for alpha_id, label
                                            in zip(alpha_ids, method_cluster_labels)}
                    results[f'{method}_alpha_cluster_mapping'] = method_cluster_mapping

                    print(f"  {method.upper()}: {len(coords_2d)} alphas, "
                          f"{method_cluster_info.get('n_clusters', 0)} clusters")

                except Exception as e:
                    print(f"Error in {method} dimensionality reduction: {e}")
                    results[f'{method}_coords'] = {}
                    results[f'{method}_cluster_info'] = {'error': str(e)}
        else:
            print("Could not create comprehensive features")
            return {}

        # Enhanced Correlation Analysis
        try:
            print("Calculating enhanced correlation matrix...")
            corr_result = calculate_correlation_matrix(pnl_data, apply_thresholding=True)

            # Handle both dictionary and DataFrame returns
            if isinstance(corr_result, dict):
                corr_matrix_raw = corr_result['raw']
                corr_matrix_thresholded = corr_result['thresholded']
                logger.info("Using dual correlation matrices: raw for visualization/MDS, thresholded for clustering")
            else:
                # Backward compatibility: if thresholding wasn't applied, use the same matrix for both
                corr_matrix_raw = corr_result
                corr_matrix_thresholded = corr_result

            if not corr_matrix_thresholded.empty and corr_matrix_thresholded.shape[0] >= 2:
                # Apply correlation regularization for stability
                # For clustering: use thresholded matrix
                # For visualization/MDS: use raw matrix
                if enable_advanced_methods:
                    print("Applying correlation regularization...")
                    shrunk_corr_thresholded = apply_correlation_regularization(corr_matrix_thresholded)
                    shrunk_corr_raw = apply_correlation_regularization(corr_matrix_raw)
                    results['shrinkage_applied'] = True
                else:
                    shrunk_corr_thresholded = corr_matrix_thresholded
                    shrunk_corr_raw = corr_matrix_raw

                # Enhanced hierarchical clustering (uses thresholded matrix for cleaner clustering)
                if enable_advanced_methods:
                    print("Performing enhanced hierarchical clustering...")
                    hierarchical_results = enhanced_hierarchical_clustering(
                        shrunk_corr_thresholded,
                        method='ward',
                        apply_shrinkage=False,  # Already applied
                        find_optimal_k=True
                    )
                    results['enhanced_hierarchical'] = hierarchical_results
                    print(f"  Optimal clusters: {hierarchical_results.get('optimal_k', 'unknown')}")

                # Create similarity network (uses thresholded matrix for clear connections)
                if enable_advanced_methods:
                    print("Creating alpha similarity network...")
                    network_results = create_alpha_similarity_network(
                        shrunk_corr_thresholded, threshold=0.3
                    )
                    if 'error' not in network_results:
                        results['similarity_network'] = {
                            'network_metrics': network_results['network_metrics'],
                            'threshold': network_results['threshold']
                        }
                        print(f"  Network: {network_results['network_metrics']['n_nodes']} nodes, "
                              f"{network_results['network_metrics']['n_edges']} edges")

                # Standard MDS analysis for all distance metrics (uses RAW matrix for full distance information)
                for distance_metric in ['simple', 'euclidean', 'angular']:
                    print(f"Calculating MDS with {distance_metric} distance (using raw correlations)...")
                    mds_coords = mds_on_correlation_matrix(shrunk_corr_raw, distance_type=distance_metric)

                    if not mds_coords.empty:
                        # Add cluster information if available
                        if 'alpha_cluster_mapping' in results:
                            for alpha_id in mds_coords.index:
                                if alpha_id in results['alpha_cluster_mapping']:
                                    mds_coords.loc[alpha_id, 'cluster'] = results['alpha_cluster_mapping'][alpha_id]

                        results[f'mds_coords_{distance_metric}'] = mds_coords.to_dict(orient='index')
                        print(f"  MDS {distance_metric}: {len(mds_coords)} coordinates")

                # Store heatmap data for visualization (uses RAW matrix for accurate visualization)
                print("Storing heatmap data for correlation matrix (using raw correlations)...")
                alpha_ids = list(shrunk_corr_raw.index)

                # Store heatmap data (limit to 50 alphas for readability)
                try:
                    max_alphas = 50
                    if len(alpha_ids) > max_alphas:
                        heatmap_corr = shrunk_corr_raw.iloc[:max_alphas, :max_alphas]
                        heatmap_ids = alpha_ids[:max_alphas]
                    else:
                        heatmap_corr = shrunk_corr_raw
                        heatmap_ids = alpha_ids

                    heatmap_data = {
                        'correlation_matrix': heatmap_corr.values.tolist(),
                        'alpha_ids': heatmap_ids
                    }

                    # Store for all distance metrics (they use the same correlation matrix)
                    results['heatmap_data_simple'] = heatmap_data
                    results['heatmap_data_euclidean'] = heatmap_data
                    results['heatmap_data_angular'] = heatmap_data
                    print(f"  Heatmap data stored for {len(heatmap_ids)} alphas")

                except Exception as e:
                    print(f"  Error storing heatmap data: {e}")
                    results['heatmap_data_simple'] = {}
                    results['heatmap_data_euclidean'] = {}
                    results['heatmap_data_angular'] = {}

        except Exception as e:
            print(f"Error in correlation analysis: {e}")

        # Apply validation if requested
        if apply_validation and validate_clustering_pipeline is not None:
            print("Applying clustering validation...")
            # Extract required arguments for validation
            if 'feature_matrix' in results and 'cluster_labels' in results:
                features_df = pd.DataFrame.from_dict(results['feature_matrix'], orient='index')
                labels = np.array(results['cluster_labels'])
                validation_results = validate_clustering_pipeline(
                    features_df=features_df,
                    labels=labels,
                    pnl_data_dict=pnl_data,
                    alpha_metadata=alpha_metadata
                )
                results['validation'] = validation_results
            else:
                print("Warning: Missing feature_matrix or cluster_labels for validation")

        # Check if we have any valid results
        has_valid_results = False
        for key in ['tsne_coords', 'umap_coords', 'pca_coords', 'mds_coords_euclidean']:
            if key in results and results[key] and len(results[key]) >= 2:
                has_valid_results = True
                break

        if not has_valid_results:
            print("No valid clustering results were generated")
            return {}

        return results

    except Exception as e:
        print(f"Error generating enhanced clustering data: {e}")
        import traceback
        traceback.print_exc()
        return {}




def save_clustering_results(results: Dict[str, Any], output_dir: str = None) -> str:
    """
    Save clustering results to a JSON file.

    Args:
        results: Dictionary with clustering results
        output_dir: Directory to save the results to (default: current directory)

    Returns:
        Path to the saved file
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    region = results.get('region', 'unknown')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"alpha_clustering_{region}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert any numpy/pandas types to Python native types
    def preprocess_for_json(obj):
        """Recursively convert numpy types including dictionary keys."""
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                # Convert key
                if isinstance(key, (np.integer, np.int32, np.int64, np.intc)):
                    key = int(key)
                elif isinstance(key, (np.floating, np.float32, np.float64)):
                    key = float(key)
                elif hasattr(key, 'item'):
                    key = key.item()
                # Convert value
                new_dict[key] = preprocess_for_json(value)
            return new_dict
        elif isinstance(obj, (list, tuple)):
            return [preprocess_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64, np.intc)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('index')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return obj

    # Preprocess the entire results dictionary
    results = preprocess_for_json(results)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            # Fallback for any remaining numpy types
            if isinstance(obj, (np.integer, np.int32, np.int64, np.intc)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('index')
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            return super(NpEncoder, self).default(obj)

    with open(filepath, 'w') as f:
        json.dump(results, f, cls=NpEncoder, indent=2)

    print(f"Results saved to {filepath}")
    return filepath


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main(region: str = 'USA'):
    """
    Main function to run clustering analysis for a specific region.

    Args:
        region: Region to analyze (default: 'USA')
    """
    print(f"Running clustering analysis for region {region}")
    print("=" * 80)

    # Generate clustering data with all methods
    results = generate_clustering_data(region, apply_validation=True, enable_advanced_methods=True)

    if results:
        # Save results
        output_path = save_clustering_results(results)
        print(f"\nAnalysis complete. Results saved to: {output_path}")

        # Print summary
        print("\nSummary:")
        print(f"  Region: {results['region']}")
        print(f"  Total alphas: {results['alpha_count']}")

        if 'cluster_info' in results:
            print(f"  Main clusters: {results['cluster_info'].get('n_clusters', 0)}")
            print(f"  Outliers: {results['cluster_info'].get('n_outliers', 0)}")

        if 'enhanced_hierarchical' in results:
            print(f"  Hierarchical optimal k: {results['enhanced_hierarchical'].get('optimal_k', 'N/A')}")

        if 'similarity_network' in results:
            metrics = results['similarity_network']['network_metrics']
            print(f"  Network density: {metrics['density']:.3f}")
            print(f"  Network components: {metrics['n_components']}")
    else:
        print("No results generated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run alpha clustering analysis')
    parser.add_argument('--region', type=str, default='USA',
                       choices=REGIONS, help='Region to analyze')
    parser.add_argument('--no-advanced', action='store_true',
                       help='Disable advanced clustering methods')
    parser.add_argument('--no-validation', action='store_true',
                       help='Disable validation pipeline')

    args = parser.parse_args()

    # Run analysis
    if args.no_advanced and args.no_validation:
        # Use original method for compatibility
        results = generate_clustering_data_original(args.region)
    else:
        results = generate_clustering_data(
            args.region,
            apply_validation=not args.no_validation,
            enable_advanced_methods=not args.no_advanced
        )

    if results:
        output_path = save_clustering_results(results)
        print(f"Results saved to: {output_path}")