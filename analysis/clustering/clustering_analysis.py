"""
Alpha Clustering Analysis Module.

This module provides tools for analyzing the similarities and differences
between alphas using various dimensionality reduction techniques.
"""
import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import database functionality
from database.operations import (
    get_all_alpha_ids_by_region_basic,
    get_regular_alpha_ids_by_region,
    get_pnl_data_for_alphas,
    get_correlation_statistics
)
from config.database_config import REGIONS

# Import correlation engine for correct correlation calculation
from correlation.correlation_engine import CorrelationEngine

# Import scikit-learn components for dimensionality reduction
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# Import enhanced feature engineering and validation
try:
    from .feature_engineering import create_enhanced_feature_matrix
    from .validation import validate_clustering_pipeline
    from .advanced_clustering import (
        enhanced_hierarchical_clustering, 
        apply_ledoit_wolf_shrinkage,
        create_alpha_similarity_network
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from feature_engineering import create_enhanced_feature_matrix
    from validation import validate_clustering_pipeline
    from advanced_clustering import (
        enhanced_hierarchical_clustering, 
        apply_ledoit_wolf_shrinkage,
        create_alpha_similarity_network
    )

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
                                apply_thresholding: bool = True) -> pd.DataFrame:
    """
    Calculate pairwise Pearson correlation matrix from daily PnL data with optional thresholding.
    Uses the same correlation calculation method as scripts/calculate_correlations.py
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to a dictionary containing DataFrame with PnL data
                      Format: {alpha_id: {'df': dataframe}}
        min_common_days: Minimum number of common trading days required
        use_cython: Whether to use Cython acceleration (default: True)
        apply_thresholding: Apply correlation thresholding to remove weak correlations
        
    Returns:
        DataFrame containing the correlation matrix (optionally thresholded)
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
        corr_matrix = apply_correlation_thresholding(corr_matrix, int(avg_samples))
    
    return corr_matrix

def mds_on_correlation_matrix(corr_matrix: pd.DataFrame, 
                            distance_type: str = 'euclidean') -> pd.DataFrame:
    """
    Apply Multidimensional Scaling (MDS) on the correlation matrix to get 2D coordinates.
    
    Args:
        corr_matrix: Correlation matrix (DataFrame)
        distance_type: Type of distance metric ('euclidean' or 'angular')
        
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
        # Fallback to simple dissimilarity (less mathematically correct)
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
    
    # Use the enhanced feature engineering (disable spiked covariance - known to cause issues)
    # Disable multiscale features as they were causing temporal clustering rather than risk-based clustering
    enhanced_features = create_enhanced_feature_matrix(
        pnl_data_dict,
        include_spiked_cov=False,  # DISABLED: Complex PCA-based features causing artificial patterns
        include_multiscale=False,  # DISABLED: Temporal volatility features removed by user request
        include_risk_metrics=True
    )
    
    if enhanced_features.empty:
        print("Warning: Enhanced feature calculation failed, no features generated")
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
    spiked_features = [col for col in enhanced_features.columns if col.startswith('spiked_')]
    multiscale_features = [col for col in enhanced_features.columns if col.startswith('multiscale_')]
    risk_features = [col for col in enhanced_features.columns if col.startswith('risk_')]
    metadata_features = [col for col in enhanced_features.columns if col.startswith('metadata_')]
    
    print(f"  Feature breakdown:")
    print(f"    - Spiked covariance: {len(spiked_features)} features")
    print(f"    - Multi-scale temporal: {len(multiscale_features)} features (DISABLED)")
    print(f"    - Advanced risk metrics: {len(risk_features)} features")
    print(f"    - Metadata: {len(metadata_features)} features")
    
    return enhanced_features

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
    from sklearn.preprocessing import RobustScaler
    
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
            # HDBSCAN will determine optimal number of clusters automatically
            if min_cluster_size is None:
                # Use statistical approach: sqrt(n) scaled down for smaller datasets
                data_size = len(features_df)
                if data_size <= 20:
                    min_cluster_size = 2  # Very permissive for small datasets
                elif data_size <= 50:
                    min_cluster_size = 3  # Allow smaller clusters
                elif data_size <= 100:
                    min_cluster_size = max(2, int(data_size * 0.02))  # 2% of data
                else:
                    min_cluster_size = max(3, int(data_size * 0.015))  # 1.5% for larger datasets
                
                # No upper limit - let HDBSCAN decide
            
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
            
            # HDBSCAN results - no fallback, trust the algorithm
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            outlier_ratio = (cluster_labels == -1).sum() / len(cluster_labels) if len(cluster_labels) > 0 else 1.0
            
            print(f"  HDBSCAN result: {n_clusters_found} clusters, {outlier_ratio:.1%} outliers")
            if n_clusters_found == 0:
                print(f"    HDBSCAN determined no natural clusters exist in this data")
            elif outlier_ratio > 0.5:
                print(f"    High outlier ratio indicates diverse/unique trading strategies")
            else:
                print(f"    Found clear groupings of similar trading strategies")
            
            # Store HDBSCAN-specific information
            cluster_info.update({
                'min_cluster_size': min_cluster_size,
                'min_samples': 2,
                'metric': 'euclidean',
                'algorithm': 'hdbscan',
                'condensed_tree': hasattr(clusterer, 'condensed_tree_'),
                'cluster_persistence': getattr(clusterer, 'cluster_persistence_', None)
            })
    
    
    # Calculate clustering metrics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = (cluster_labels == -1).sum() if -1 in cluster_labels else 0
    
    print(f"  Found {n_clusters} clusters with {n_outliers} outliers")
    
    # Calculate validation metrics if we have enough data
    if n_clusters > 1:
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
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
        cluster_info['outlier_percentage'] = float(n_outliers / len(cluster_labels) * 100)
        
        print(f"  Cluster sizes: {cluster_sizes}")
        print(f"  Outlier percentage: {cluster_info['outlier_percentage']:.1f}%")
    
    return cluster_labels, probabilities, cluster_info

def add_cluster_info_to_coordinates(coordinates_dict: Dict[str, Dict[str, float]], 
                                  alpha_cluster_mapping: Dict[str, int],
                                  cluster_probabilities: np.ndarray,
                                  alpha_indices: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Add cluster information to coordinate dictionaries.
    
    Args:
        coordinates_dict: Dictionary of {alpha_id: {'x': ..., 'y': ...}}
        alpha_cluster_mapping: Dictionary mapping alpha IDs to cluster labels
        cluster_probabilities: Array of cluster membership probabilities
        alpha_indices: List of alpha IDs in the same order as cluster_probabilities
        
    Returns:
        Updated coordinates dictionary with cluster and cluster_probability added
    """
    if not alpha_cluster_mapping or not coordinates_dict:
        return coordinates_dict
    
    coords_with_clusters = coordinates_dict.copy()
    
    for alpha_id in coords_with_clusters.keys():
        if alpha_id in alpha_cluster_mapping:
            # Add cluster label
            coords_with_clusters[alpha_id]['cluster'] = alpha_cluster_mapping[alpha_id]
            
            # Add cluster probability if available
            try:
                alpha_idx = alpha_indices.index(alpha_id)
                if alpha_idx < len(cluster_probabilities):
                    coords_with_clusters[alpha_id]['cluster_probability'] = cluster_probabilities[alpha_idx]
                else:
                    coords_with_clusters[alpha_id]['cluster_probability'] = 0.0
            except (ValueError, IndexError):
                coords_with_clusters[alpha_id]['cluster_probability'] = 0.0
    
    return coords_with_clusters

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
        # For higher dimensions, we'll use exact method instead
        optimal = min(10, n_features, n_samples // 10)  # Allow up to 10, will handle in model creation
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
        features_df: DataFrame with features
        method: Method to use ('tsne', 'umap', or 'pca')
        n_components_for_clustering: Number of components for clustering (auto-determined if None)
        variance_threshold: For PCA, variance explained threshold for clustering components
        random_state: Random state for reproducibility
        use_adaptive_params: Use theoretically-grounded parameter selection (default: True)
        
    Returns:
        Tuple of:
        - DataFrame with alpha_id as index and x, y coordinates as columns (2D for visualization)
        - DataFrame with alpha_id as index and optimal components for clustering (N-dimensional)
        - Dictionary with model information (loadings, variance explained, etc.)
    """
    # Check if we have enough samples
    if features_df.empty:
        print(f"Cannot apply {method}: Empty feature dataframe")
        return pd.DataFrame(), pd.DataFrame(), {}
        
    # Determine optimal components for clustering if not provided
    if n_components_for_clustering is None:
        n_components_for_clustering = determine_optimal_components_for_clustering(
            method, features_df, variance_threshold
        )
        
    if len(features_df) < 2:
        print(f"Cannot apply {method}: Need at least 2 samples, got {len(features_df)}")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Drop any rows with NaN values
    features_df = features_df.dropna()
    if features_df.empty or len(features_df) < 2:
        print(f"Cannot apply {method}: Not enough valid samples after dropping NaNs")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    try:
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        # Configure dimensionality reduction model with improved parameter selection
        n_samples, n_features = features_df.shape
        
        if method.lower() == 'tsne':
            if use_adaptive_params:
                # Logarithmic scaling based on information theory (Paleologo's recommendation)
                perplexity = max(5, min(50, int(np.log2(n_samples) * 4)))
            else:
                # Original empirical approach
                perplexity = min(30, max(5, n_samples // 5))
            
            # Ensure perplexity is valid for t-SNE
            perplexity = min(perplexity, n_samples - 1)
            
            # Create models for both visualization (2D) and clustering (N-D)
            model_2d = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
            
            # Barnes-Hut algorithm only supports up to 3D, use exact method for higher dimensions
            if n_components_for_clustering <= 3:
                model_clustering = TSNE(n_components=n_components_for_clustering, perplexity=perplexity, 
                                      random_state=random_state + 1)  # Uses Barnes-Hut by default
            else:
                model_clustering = TSNE(n_components=n_components_for_clustering, perplexity=perplexity, 
                                      method='exact', random_state=random_state + 1)  # Use exact for >3D
            
        elif method.lower() == 'umap':
            if use_adaptive_params:
                # Square-root scaling following diversification theory
                n_neighbors = max(5, min(30, int(np.sqrt(n_samples))))
                
                # Adaptive min_dist based on eigenvalue gaps
                eigenvals = np.linalg.eigvals(features_df.corr().fillna(0))
                eigenval_gap = np.mean(np.diff(sorted(eigenvals, reverse=True)[:min(10, len(eigenvals))]))
                min_dist = max(0.05, min(0.5, abs(eigenval_gap)))
            else:
                # Original empirical approach
                n_neighbors = min(15, max(2, n_samples // 5))
                min_dist = 0.1
                
            # Create models for both visualization (2D) and clustering (N-D)
            model_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, 
                               n_jobs=1, random_state=random_state)
            model_clustering = umap.UMAP(n_components=n_components_for_clustering, n_neighbors=n_neighbors, 
                                       min_dist=min_dist, n_jobs=1, random_state=random_state + 1)
                            
        elif method.lower() == 'pca':
            # For PCA, we can efficiently get both 2D and N-D from a single fit
            # Use the maximum of 2 and n_components_for_clustering
            max_components = max(2, n_components_for_clustering)
            
            if use_adaptive_params:
                # Use BBP (Baik-Ben Arous-Péché) threshold for component validation
                gamma = n_samples / n_features if n_features > 0 else 1
                bbp_threshold = 1 + np.sqrt(gamma)
                
            # Create single PCA model with enough components for both purposes
            model_full = PCA(n_components=max_components, random_state=random_state)
            
            # For PCA, we don't need separate models - we can extract different numbers of components
            model_2d = model_full  # Will extract first 2 components 
            model_clustering = model_full  # Will extract first n_components_for_clustering components
                       
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Handle transformations differently based on method
        if method.lower() == 'pca':
            # For PCA, fit once and extract different numbers of components
            full_coords = model_full.fit_transform(scaled_features)
            
            # Extract 2D coordinates for visualization
            coords_2d = full_coords[:, :2]
            
            # Extract clustering features (first n_components_for_clustering)
            coords_clustering = full_coords[:, :n_components_for_clustering]
            
        else:
            # For t-SNE and UMAP, we need to fit both models
            print(f"  Fitting {method.upper()} for 2D visualization...")
            coords_2d = model_2d.fit_transform(scaled_features)
            
            print(f"  Fitting {method.upper()} for {n_components_for_clustering}D clustering...")
            coords_clustering = model_clustering.fit_transform(scaled_features)
        
        # Ensure 2D coordinates are properly shaped
        if coords_2d.shape[1] == 1:
            coords_2d = np.column_stack([coords_2d[:, 0], np.zeros(coords_2d.shape[0])])
        
        # Create DataFrames for both outputs
        coords_2d_df = pd.DataFrame(coords_2d, index=features_df.index, columns=['x', 'y'])
        
        # Create clustering features DataFrame with appropriate column names
        clustering_cols = [f'{method}_comp_{i+1}' for i in range(coords_clustering.shape[1])]
        coords_clustering_df = pd.DataFrame(coords_clustering, index=features_df.index, columns=clustering_cols)
        
        # Collect model information
        model_info = {
            'method': method,
            'use_adaptive_params': use_adaptive_params,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_components_for_clustering': n_components_for_clustering,
            'clustering_features_shape': coords_clustering_df.shape
        }
        
        # Add method-specific parameter info
        if method.lower() == 'tsne':
            model_info['perplexity'] = perplexity
            model_info['parameter_logic'] = 'logarithmic_scaling' if use_adaptive_params else 'empirical'
        elif method.lower() == 'umap':
            model_info['n_neighbors'] = n_neighbors
            model_info['min_dist'] = min_dist
            model_info['parameter_logic'] = 'sqrt_scaling_adaptive_dist' if use_adaptive_params else 'empirical'
        elif method.lower() == 'pca':
            model_info['n_components_visualization'] = 2
            model_info['n_components_total'] = max_components
            if use_adaptive_params:
                model_info['bbp_threshold'] = bbp_threshold
                model_info['gamma'] = gamma
                model_info['parameter_logic'] = 'bbp_threshold'
            else:
                model_info['parameter_logic'] = 'empirical'
        
        # For PCA, extract additional information with enhanced feature names
        if method.lower() == 'pca':
            # Get variance explained by each component
            variance_explained = model_full.explained_variance_ratio_
            model_info['variance_explained'] = {
                'pc1': float(variance_explained[0]),
                'pc2': float(variance_explained[1]) if len(variance_explained) > 1 else 0.0,
                'total_2d': float(variance_explained[:2].sum()),  # For 2D visualization
                'total_clustering': float(variance_explained[:n_components_for_clustering].sum())  # For clustering
            }
            
            # Get feature loadings (components) - preserve full feature names
            feature_names = list(features_df.columns)
            loadings = model_full.components_
            
            # PC1 and PC2 loadings with full feature names
            pc1_loadings = dict(zip(feature_names, loadings[0]))
            pc2_loadings = dict(zip(feature_names, loadings[1])) if loadings.shape[0] > 1 else {}
            
            model_info['loadings'] = {
                'pc1': pc1_loadings,
                'pc2': pc2_loadings,
                'feature_names': feature_names
            }
            
            # Categorize features for visualization
            feature_categories = {
                'spiked': [f for f in feature_names if f.startswith('spiked_')],
                'multiscale': [f for f in feature_names if f.startswith('multiscale_')], 
                'risk': [f for f in feature_names if f.startswith('risk_')],
                'metadata': [f for f in feature_names if f.startswith('metadata_')]
            }
            model_info['feature_categories'] = feature_categories
            
            # Calculate top contributing features for each PC
            pc1_contributions = [(feature, abs(loading)) for feature, loading in pc1_loadings.items()]
            pc1_contributions.sort(key=lambda x: x[1], reverse=True)
            
            pc2_contributions = [(feature, abs(loading)) for feature, loading in pc2_loadings.items()] if pc2_loadings else []
            pc2_contributions.sort(key=lambda x: x[1], reverse=True)
            
            model_info['top_features'] = {
                'pc1': pc1_contributions[:5],  # Top 5 features for better insight
                'pc2': pc2_contributions[:5]   # Top 5 features for better insight
            }
            
            # Generate enhanced interpretation with feature categories
            def generate_enhanced_pc_interpretation(loadings, contributions):
                if not loadings:
                    return "No data"
                
                # Group by feature type and impact
                category_impacts = {'spiked': [], 'multiscale': [], 'risk': [], 'metadata': []}
                
                for feature, loading in loadings.items():
                    if abs(loading) > 0.1:  # Only significant loadings
                        for category in category_impacts:
                            if feature.startswith(f'{category}_'):
                                impact = 'high' if loading > 0.2 else 'low' if loading < -0.2 else 'moderate'
                                category_impacts[category].append((feature.replace(f'{category}_', ''), loading, impact))
                                break
                
                # Create interpretation
                interpretations = []
                for category, features in category_impacts.items():
                    if features:
                        top_feature = max(features, key=lambda x: abs(x[1]))
                        direction = "↑" if top_feature[1] > 0 else "↓" 
                        interpretations.append(f"{category.title()}: {direction}{top_feature[0]}")
                
                return " | ".join(interpretations[:3]) or "Mixed factors"
            
            model_info['interpretation'] = {
                'pc1': generate_enhanced_pc_interpretation(pc1_loadings, pc1_contributions),
                'pc2': generate_enhanced_pc_interpretation(pc2_loadings, pc2_contributions)
            }
        
        return coords_2d_df, coords_clustering_df, model_info
        
    except Exception as e:
        print(f"Error applying {method}: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}


def profile_clusters(feature_matrix: pd.DataFrame, cluster_labels: np.ndarray, 
                    feature_names: List[str]) -> Dict[str, Any]:
    """
    Profile each cluster by calculating statistics of original features.
    Maps abstract clusters back to interpretable feature characteristics.
    
    Args:
        feature_matrix: DataFrame with alphas as index, features as columns
        cluster_labels: Array of cluster assignments (-1 for outliers)
        feature_names: List of feature column names
        
    Returns:
        Dictionary with cluster profiles and interpretations
    """
    if feature_matrix.empty or len(cluster_labels) == 0:
        return {}
    
    # Get unique clusters (excluding outliers)
    unique_clusters = [c for c in np.unique(cluster_labels) if c >= 0]
    n_outliers = (cluster_labels == -1).sum()
    
    profiles = {
        'n_clusters': len(unique_clusters),
        'n_outliers': int(n_outliers),
        'cluster_profiles': {},
        'feature_importance': {},
        'cluster_interpretations': {}
    }
    
    if len(unique_clusters) == 0:
        return profiles
    
    # Calculate profiles for each cluster
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_alphas = feature_matrix.loc[cluster_mask]
        
        if len(cluster_alphas) == 0:
            continue
            
        # Calculate statistics for this cluster
        cluster_stats = {
            'size': int(cluster_mask.sum()),
            'percentage': float(cluster_mask.sum() / len(cluster_labels) * 100),
            'feature_means': cluster_alphas.mean().to_dict(),
            'feature_stds': cluster_alphas.std().to_dict(),
            'feature_medians': cluster_alphas.median().to_dict()
        }
        
        profiles['cluster_profiles'][f'cluster_{cluster_id}'] = cluster_stats
        
        # Generate interpretation based on distinctive features
        interpretation = generate_cluster_interpretation(cluster_alphas, feature_matrix, feature_names)
        profiles['cluster_interpretations'][f'cluster_{cluster_id}'] = interpretation
    
    # Calculate feature importance across all clusters (how much each feature varies between clusters)
    if len(unique_clusters) > 1:
        feature_importance = {}
        for feature in feature_names:
            cluster_means = []
            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_means.append(feature_matrix.loc[cluster_mask, feature].mean())
            
            if len(cluster_means) > 1:
                # Use coefficient of variation as importance measure
                cv = np.std(cluster_means) / (np.abs(np.mean(cluster_means)) + 1e-8)
                feature_importance[feature] = float(cv)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        profiles['feature_importance'] = dict(sorted_features[:10])  # Top 10 most discriminating features
    
    return profiles


def generate_cluster_interpretation(cluster_data: pd.DataFrame, all_data: pd.DataFrame, 
                                  feature_names: List[str]) -> str:
    """
    Generate human-readable interpretation of what makes a cluster distinctive.
    
    Args:
        cluster_data: Features for alphas in this cluster
        all_data: Features for all alphas (for comparison)
        feature_names: List of all feature names
        
    Returns:
        Human-readable interpretation string
    """
    if cluster_data.empty or all_data.empty:
        return "Insufficient data for interpretation"
    
    interpretations = []
    
    # Find features where this cluster is significantly different from overall mean
    for feature in feature_names:
        if feature not in cluster_data.columns or feature not in all_data.columns:
            continue
            
        cluster_mean = cluster_data[feature].mean()
        overall_mean = all_data[feature].mean()
        overall_std = all_data[feature].std()
        
        if overall_std == 0:
            continue
            
        # Z-score difference
        z_score = (cluster_mean - overall_mean) / overall_std
        
        # Only include features with significant differences (|z| > 1.5)
        if abs(z_score) > 1.5:
            direction = "High" if z_score > 0 else "Low"
            
            # Clean up feature names for display
            display_name = feature.replace('risk_', '').replace('multiscale_', '').replace('metadata_', '')
            display_name = display_name.replace('_', ' ').title()
            
            interpretations.append(f"{direction} {display_name} ({cluster_mean:.3f})")
    
    if not interpretations:
        return "No distinctive characteristics identified"
    
    # Return top 3 most significant differences
    return " | ".join(interpretations[:3])

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
            'enhanced_features_used': True,  # Always true now
            'validation_applied': apply_validation,
            'advanced_methods_enabled': enable_advanced_methods
        }
        
        # Calculate All Features (Enhanced)
        print("Calculating comprehensive feature matrix...")
        all_features = calculate_all_features(pnl_data, alpha_metadata)
        
        if not all_features.empty and len(all_features) >= 2:
            results['features_count'] = len(all_features.columns)
            results['feature_categories'] = {
                'spiked': len([col for col in all_features.columns if col.startswith('spiked_')]),
                'multiscale': len([col for col in all_features.columns if col.startswith('multiscale_')]),
                'risk': len([col for col in all_features.columns if col.startswith('risk_')]),
                'metadata': len([col for col in all_features.columns if col.startswith('metadata_')])
            }
            print(f"Feature categories: {results['feature_categories']}")
            
            # Store original features for cluster profiling
            results['feature_matrix'] = all_features.to_dict(orient='index')
            results['feature_names'] = list(all_features.columns)
            print(f"Stored feature matrix with {len(all_features)} alphas and {len(all_features.columns)} features")
            
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
                
                print(f"  Clustering completed: {cluster_info.get('n_clusters', 0)} clusters, {cluster_info.get('n_outliers', 0)} outliers")
                
                # Create mapping from alpha_id to cluster
                alpha_ids = list(all_features.index)
                cluster_mapping = {alpha_id: int(label) for alpha_id, label in zip(alpha_ids, cluster_labels)}
                results['alpha_cluster_mapping'] = cluster_mapping
                
                # Generate cluster profiling for main feature-based clustering
                print("  Profiling main feature-based clusters...")
                main_cluster_profiles = profile_clusters(all_features, cluster_labels, list(all_features.columns))
                results['main_cluster_profiles'] = main_cluster_profiles
                
                # Print main cluster interpretations
                if main_cluster_profiles and 'cluster_interpretations' in main_cluster_profiles:
                    for cluster_name, interpretation in main_cluster_profiles['cluster_interpretations'].items():
                        print(f"    Main {cluster_name.replace('_', ' ').title()}: {interpretation}")
                
            except Exception as e:
                print(f"Error in clustering: {e}")
                results['cluster_labels'] = []
                results['cluster_probabilities'] = []
                results['cluster_info'] = {'error': str(e)}
                results['alpha_cluster_mapping'] = {}
            
            # Apply dimensionality reduction with enhanced clustering on reduced features
            for method in ['tsne', 'umap', 'pca']:
                try:
                    print(f"Applying {method.upper()} with enhanced features...")
                    
                    # Get both 2D coordinates and clustering features
                    coords_2d, coords_clustering, info = apply_dimensionality_reduction(
                        all_features, method=method, use_adaptive_params=True, variance_threshold=0.95
                    )
                    
                    # Perform clustering on the reduced features (not full features)
                    print(f"  Clustering on {coords_clustering.shape[1]}D {method.upper()} features...")
                    method_cluster_labels, method_probabilities, method_cluster_info = perform_clustering(
                        coords_clustering, method='hdbscan', random_state=42
                    )
                    
                    # Add method-specific cluster information to 2D coordinates
                    coords_with_clusters = coords_2d.copy()
                    coords_with_clusters['cluster'] = method_cluster_labels
                    coords_with_clusters['cluster_probability'] = method_probabilities
                    
                    # Store results with method-specific clustering
                    results[f'{method}_coords'] = coords_with_clusters.to_dict(orient='index')
                    results[f'{method}_info'] = info
                    results[f'{method}_cluster_labels'] = method_cluster_labels.tolist()
                    results[f'{method}_cluster_probabilities'] = method_probabilities.tolist()
                    results[f'{method}_cluster_info'] = method_cluster_info
                    
                    # Create method-specific alpha-cluster mapping
                    alpha_ids = list(coords_2d.index)
                    method_cluster_mapping = {alpha_id: int(label) for alpha_id, label in zip(alpha_ids, method_cluster_labels)}
                    results[f'{method}_alpha_cluster_mapping'] = method_cluster_mapping
                    
                    print(f"  {method.upper()}: {len(coords_2d)} alphas, {method_cluster_info.get('n_clusters', 0)} clusters from {coords_clustering.shape[1]}D features")
                    
                    # Generate cluster profiling for interpretability
                    print(f"  Profiling {method.upper()} clusters...")
                    cluster_profiles = profile_clusters(all_features, method_cluster_labels, list(all_features.columns))
                    results[f'{method}_cluster_profiles'] = cluster_profiles
                    
                    # Print cluster interpretations for debugging
                    if cluster_profiles and 'cluster_interpretations' in cluster_profiles:
                        for cluster_name, interpretation in cluster_profiles['cluster_interpretations'].items():
                            print(f"    {cluster_name.replace('_', ' ').title()}: {interpretation}")
                    
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
            corr_matrix = calculate_correlation_matrix(pnl_data, apply_thresholding=True)
            
            print(f"Correlation matrix shape: {corr_matrix.shape if not corr_matrix.empty else 'EMPTY'}")
            
            if not corr_matrix.empty and corr_matrix.shape[0] >= 2:
                # Apply Ledoit-Wolf shrinkage for stability
                if enable_advanced_methods:
                    print("Applying Ledoit-Wolf shrinkage...")
                    shrunk_corr = apply_ledoit_wolf_shrinkage(corr_matrix)
                    results['shrinkage_applied'] = True
                else:
                    shrunk_corr = corr_matrix
                
                # Enhanced hierarchical clustering
                if enable_advanced_methods:
                    print("Performing enhanced hierarchical clustering...")
                    hierarchical_results = enhanced_hierarchical_clustering(
                        shrunk_corr, 
                        method='ward', 
                        apply_shrinkage=False,  # Already applied
                        find_optimal_k=True
                    )
                    results['enhanced_hierarchical'] = hierarchical_results
                    print(f"  Optimal clusters: {hierarchical_results.get('optimal_k', 'unknown')}")
                
                # Create similarity network
                if enable_advanced_methods:
                    print("Creating alpha similarity network...")
                    network_results = create_alpha_similarity_network(
                        shrunk_corr, threshold=0.3
                    )
                    if 'error' not in network_results:
                        results['similarity_network'] = {
                            'network_metrics': network_results['network_metrics'],
                            'threshold': network_results['threshold']
                        }
                        print(f"  Network: {network_results['network_metrics']['n_nodes']} nodes, {network_results['network_metrics']['n_edges']} edges")
                
                # Standard MDS analysis with enhanced correlation
                distance_metrics = ['simple', 'euclidean', 'angular']
                for distance_metric in distance_metrics:
                    print(f"Calculating MDS with {distance_metric} distance...")
                    mds_coords = mds_on_correlation_matrix(shrunk_corr, distance_type=distance_metric)
                    print(f"  MDS {distance_metric}: {mds_coords.shape if not mds_coords.empty else 'EMPTY'} coordinates")
                    
                    if not mds_coords.empty:
                        # Convert to dictionary format
                        mds_dict = mds_coords.to_dict(orient='index')
                        
                        # Add cluster information if available (prefer full-feature clustering, fallback to PCA)
                        cluster_mapping_to_use = None
                        cluster_probs_to_use = None
                        cluster_source = None
                        
                        if 'alpha_cluster_mapping' in results and results['alpha_cluster_mapping']:
                            cluster_mapping_to_use = results['alpha_cluster_mapping']
                            cluster_probs_to_use = np.array(results['cluster_probabilities'])
                            cluster_source = "full-features"
                        elif 'pca_alpha_cluster_mapping' in results and results['pca_alpha_cluster_mapping']:
                            cluster_mapping_to_use = results['pca_alpha_cluster_mapping']
                            cluster_probs_to_use = np.array(results['pca_cluster_probabilities'])
                            cluster_source = "PCA-features"
                        elif 'umap_alpha_cluster_mapping' in results and results['umap_alpha_cluster_mapping']:
                            cluster_mapping_to_use = results['umap_alpha_cluster_mapping']
                            cluster_probs_to_use = np.array(results['umap_cluster_probabilities'])
                            cluster_source = "UMAP-features"
                        
                        if cluster_mapping_to_use:
                            alpha_indices = list(all_features.index)
                            mds_dict = add_cluster_info_to_coordinates(
                                mds_dict,
                                cluster_mapping_to_use,
                                cluster_probs_to_use,
                                alpha_indices
                            )
                            print(f"  Added cluster information to MDS {distance_metric} coordinates (using {cluster_source})")
                        
                        # Store with both new and legacy key names for compatibility
                        results[f'enhanced_mds_{distance_metric}'] = mds_dict
                        results[f'mds_coords_{distance_metric}'] = mds_dict
                        print(f"  Stored MDS data with {len(mds_coords)} alphas")
                    else:
                        print(f"  WARNING: MDS {distance_metric} returned empty results")
                
                # Store main MDS for backward compatibility
                if 'mds_coords_euclidean' in results:
                    results['mds_coords'] = results['mds_coords_euclidean']
                
                # Generate heatmap data from correlation matrix
                print("Generating correlation heatmap data...")
                alpha_ids = list(shrunk_corr.index)
                print(f"  Available alphas for heatmap: {len(alpha_ids)}")
                
                # Store heatmap data (limited to 50 alphas for readability)
                max_alphas = 50
                if len(alpha_ids) > max_alphas:
                    heatmap_corr = shrunk_corr.iloc[:max_alphas, :max_alphas]
                    heatmap_ids = alpha_ids[:max_alphas]
                    print(f"  Using first {max_alphas} alphas for heatmap")
                else:
                    heatmap_corr = shrunk_corr
                    heatmap_ids = alpha_ids
                    print(f"  Using all {len(alpha_ids)} alphas for heatmap")
                
                # Store heatmap data for all distance metrics
                heatmap_data = {
                    'correlation_matrix': heatmap_corr.values.tolist(),
                    'alpha_ids': heatmap_ids
                }
                results['heatmap_data'] = heatmap_data
                for metric in ['simple', 'euclidean', 'angular']:
                    results[f'heatmap_data_{metric}'] = heatmap_data
                
                print(f"  Stored heatmap data: {len(heatmap_data['alpha_ids'])} x {len(heatmap_data['alpha_ids'])} matrix")
                
                # Store enhanced correlation data
                results['correlation_thresholding_applied'] = True
                
            else:
                print("Skipping enhanced correlation analysis: insufficient data")
        except Exception as e:
            print(f"Error in enhanced correlation analysis: {e}")
            # Even if enhanced correlation fails, try to provide basic MDS/heatmap data
            print("Attempting fallback MDS/heatmap generation...")
            try:
                basic_corr = calculate_correlation_matrix(pnl_data, apply_thresholding=False)
                if not basic_corr.empty:
                    mds_coords = mds_on_correlation_matrix(basic_corr, distance_type='euclidean')
                    if not mds_coords.empty:
                        # Convert to dictionary format
                        mds_dict = mds_coords.to_dict(orient='index')
                        
                        # Add cluster information if available (prefer full-feature, fallback to method-specific)
                        cluster_mapping_to_use = None
                        cluster_probs_to_use = None
                        
                        if 'alpha_cluster_mapping' in results and results['alpha_cluster_mapping']:
                            cluster_mapping_to_use = results['alpha_cluster_mapping']
                            cluster_probs_to_use = np.array(results['cluster_probabilities'])
                        elif 'pca_alpha_cluster_mapping' in results and results['pca_alpha_cluster_mapping']:
                            cluster_mapping_to_use = results['pca_alpha_cluster_mapping']
                            cluster_probs_to_use = np.array(results['pca_cluster_probabilities'])
                        
                        if cluster_mapping_to_use and not all_features.empty:
                            alpha_indices = list(all_features.index)
                            mds_dict = add_cluster_info_to_coordinates(
                                mds_dict,
                                cluster_mapping_to_use,
                                cluster_probs_to_use,
                                alpha_indices
                            )
                            print("  Added cluster information to fallback MDS coordinates")
                        
                        results['mds_coords'] = mds_dict
                        results['mds_coords_euclidean'] = mds_dict
                        print("  Fallback MDS data generated")
                    
                    # Basic heatmap
                    alpha_ids = list(basic_corr.index)[:50]  # First 50
                    if len(alpha_ids) >= 2:
                        heatmap_corr = basic_corr.iloc[:len(alpha_ids), :len(alpha_ids)]
                        heatmap_data = {
                            'correlation_matrix': heatmap_corr.values.tolist(),
                            'alpha_ids': alpha_ids
                        }
                        results['heatmap_data'] = heatmap_data
                        results['heatmap_data_euclidean'] = heatmap_data
                        print("  Fallback heatmap data generated")
            except Exception as fallback_error:
                print(f"  Fallback also failed: {fallback_error}")
        
        # Validation Pipeline
        if apply_validation and not all_features.empty:
            print("Running validation pipeline...")
            try:
                # Use hierarchical clustering labels if available
                if 'enhanced_hierarchical' in results and 'cluster_labels' in results['enhanced_hierarchical']:
                    validation_labels = results['enhanced_hierarchical']['cluster_labels']
                    clustering_method = 'hierarchical'
                else:
                    # Fallback to simple clustering for validation
                    from sklearn.cluster import KMeans
                    n_clusters = min(5, len(all_features) // 10)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    validation_labels = kmeans.fit_predict(all_features)
                    clustering_method = 'kmeans'
                
                validation_report = validate_clustering_pipeline(
                    all_features,
                    validation_labels,
                    pnl_data_dict=pnl_data,
                    alpha_metadata=alpha_metadata,
                    clustering_method=clustering_method
                )
                
                results['validation_report'] = validation_report
                overall_score = validation_report.get('overall_validation_score', 0)
                print(f"  Validation complete. Overall score: {overall_score:.3f}")
                
            except Exception as e:
                print(f"Error in validation pipeline: {e}")
                results['validation_report'] = {'error': str(e)}
        
        # Store feature names for visualization
        if not all_features.empty:
            results['feature_names'] = list(all_features.columns)
        
        # Check if we have any valid results
        has_valid_results = False
        for key in ['tsne_coords', 'umap_coords', 'pca_coords', 'mds_coords_euclidean', 'enhanced_mds_euclidean']:
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




def generate_clustering_data_original(region: str, 
                                    pnl_data: Optional[Dict] = None,
                                    alpha_metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Original clustering data generation method (for fallback and compatibility).
    
    Args:
        region: Region name
        pnl_data: Pre-fetched PnL data (optional)
        alpha_metadata: Pre-fetched alpha metadata (optional)
        
    Returns:
        Dictionary with original clustering results
    """
    try:
        # Handle pre-fetched data
        if pnl_data is None or alpha_metadata is None:
            try:
                # Get all alpha IDs for the region
                print(f"Getting alpha IDs for region {region}...")
                alpha_ids = get_regular_alpha_ids_by_region(region)
                
                if not alpha_ids:
                    print(f"No alphas found for region {region}")
                    return {}
                
                print(f"Found {len(alpha_ids)} alphas for region {region}")
                
                if pnl_data is None:
                    # Get PnL data for these alphas
                    print("Fetching PnL data...")
                    pnl_data = get_pnl_data_for_alphas(alpha_ids, region)
                    
                    if not pnl_data:
                        print("No PnL data found for any alphas")
                        return {}
                    
                    print(f"Fetched PnL data for {len(pnl_data)} alphas")
                
                if alpha_metadata is None:
                    # Get alpha metadata (correlation statistics have useful metrics)
                    try:
                        print("Fetching correlation statistics...")
                        alpha_metadata = get_correlation_statistics(region)
                        if not alpha_metadata.empty:
                            alpha_metadata.set_index('alpha_id', inplace=True)
                            print(f"Fetched metadata for {len(alpha_metadata)} alphas")
                        else:
                            print("No correlation statistics found")
                            alpha_metadata = pd.DataFrame()
                    except Exception as e:
                        print(f"Error fetching correlation statistics: {e}")
                        alpha_metadata = pd.DataFrame()
            
            except Exception as e:
                print(f"Error fetching data: {e}")
                return {}
        
        results = {
            'region': region,
            'timestamp': datetime.now().isoformat(),
            'alpha_count': len(pnl_data),
            'method': 'original'
        }
        
        # Method 1: MDS on Correlation Matrix and Advanced Visualizations
        try:
            print("Calculating original correlation matrix...")
            corr_matrix = calculate_correlation_matrix(pnl_data, apply_thresholding=False)
            
            if not corr_matrix.empty and corr_matrix.shape[0] >= 2:
                # Pre-calculate MDS for all distance metrics
                distance_metrics = ['simple', 'euclidean', 'angular']
                
                for distance_metric in distance_metrics:
                    print(f"Calculating MDS with {distance_metric} distance...")
                    mds_coords = mds_on_correlation_matrix(corr_matrix, distance_type=distance_metric)
                    results[f'mds_coords_{distance_metric}'] = mds_coords.to_dict(orient='index')
                    print(f"  MDS coordinates calculated for {len(mds_coords)} alphas")
                
                # Keep backward compatibility - store default euclidean as 'mds_coords'
                results['mds_coords'] = results['mds_coords_euclidean']
                
                # Pre-calculate Heatmap data (just one version - correlation doesn't change)
                print("Pre-calculating Heatmap data...")
                alpha_ids = list(corr_matrix.index)
                
                # Store heatmap data (limited to 50 alphas for readability)
                try:
                    max_alphas = 50
                    if len(alpha_ids) > max_alphas:
                        heatmap_corr = corr_matrix.iloc[:max_alphas, :max_alphas]
                        heatmap_ids = alpha_ids[:max_alphas]
                    else:
                        heatmap_corr = corr_matrix
                        heatmap_ids = alpha_ids
                    
                    # Store single heatmap data (correlation matrix doesn't change)
                    results['heatmap_data'] = {
                        'correlation_matrix': heatmap_corr.values.tolist(),
                        'alpha_ids': heatmap_ids
                    }
                    print(f"  Heatmap data stored for {len(heatmap_ids)} alphas")
                    
                    # Keep backward compatibility with old structure
                    for metric in ['simple', 'euclidean', 'angular']:
                        results[f'heatmap_data_{metric}'] = results['heatmap_data']
                        
                except Exception as e:
                    print(f"  Error storing heatmap data: {e}")
                    results['heatmap_data'] = {}
                    for metric in ['simple', 'euclidean', 'angular']:
                        results[f'heatmap_data_{metric}'] = {}
                
            else:
                print("Skipping MDS and advanced visualizations: Not enough data in correlation matrix")
                results['mds_coords'] = {}
                for metric in ['simple', 'euclidean', 'angular']:
                    results[f'mds_coords_{metric}'] = {}
                    results[f'heatmap_data_{metric}'] = {}
        except Exception as e:
            print(f"Error in correlation-based calculations: {e}")
            results['mds_coords'] = {}
            for metric in ['simple', 'euclidean', 'angular']:
                results[f'mds_coords_{metric}'] = {}
                results[f'heatmap_data_{metric}'] = {}
        
        # This section has been removed - now using enhanced features only
        
        # Add metadata if available
        results['alpha_metadata'] = alpha_metadata.to_dict(orient='index') if not alpha_metadata.empty else {}
        
        # Check if we have any valid results
        has_valid_results = False
        for key in ['mds_coords', 'tsne_coords', 'umap_coords', 'pca_coords']:
            if results[key] and len(results[key]) >= 2:
                has_valid_results = True
                break
        
        if not has_valid_results:
            print("No valid clustering results were generated")
            return {}
        
        return results
    except Exception as e:
        print(f"Error generating clustering data: {e}")
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
    
    region = results['region']
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
    
    print(f"Clustering results saved to {filepath}")
    return filepath

def main(region: str = 'USA'):
    """
    Run the comprehensive clustering analysis for a specific region.
    Always uses enhanced features and advanced methods.
    
    Args:
        region: Region name (default: 'USA')
    """
    print(f"Generating comprehensive clustering data for region: {region}")
    print("Features: Spiked covariance + Multi-scale temporal + Advanced risk metrics")
    print("Methods: Theoretical parameter selection, correlation shrinkage, validation")
    
    results = generate_clustering_data(
        region,
        apply_validation=True, 
        enable_advanced_methods=True
    )
    
    if results:
        output_path = save_clustering_results(results)
        print(f"\nSaved clustering results to: {output_path}")
        
        # Print comprehensive summary
        print(f"\n=== CLUSTERING SUMMARY ===")
        print(f"Total Features: {results.get('features_count', 'N/A')}")
        if 'feature_categories' in results:
            categories = results['feature_categories']
            print(f"Feature Breakdown:")
            print(f"  - Spiked covariance: {categories.get('spiked', 0)} features")
            print(f"  - Multi-scale temporal: {categories.get('multiscale', 0)} features (DISABLED)")  
            print(f"  - Advanced risk metrics: {categories.get('risk', 0)} features")
            print(f"  - Metadata: {categories.get('metadata', 0)} features")
        
        if 'validation_report' in results:
            validation = results['validation_report']
            overall_score = validation.get('overall_validation_score', 'N/A')
            print(f"Validation Score: {overall_score}")
            
        if 'enhanced_hierarchical' in results:
            h_results = results['enhanced_hierarchical']
            print(f"Optimal Clusters: {h_results.get('optimal_k', 'N/A')}")
            print(f"Silhouette Score: {h_results.get('silhouette_score', 'N/A'):.3f}")
        
        print(f"\nTo view the results, run the visualization server with:")
        print(f"  python analysis/clustering/visualization_server.py {output_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Alpha Clustering Analysis")
    parser.add_argument("--region", type=str, default="USA",
                      choices=REGIONS,
                      help=f"Region to analyze. Available regions: {', '.join(REGIONS)}")
    args = parser.parse_args()
    
    main(args.region)
