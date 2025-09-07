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
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

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

def calculate_correlation_matrix(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                                min_common_days: int = 60,
                                use_cython: bool = True) -> pd.DataFrame:
    """
    Calculate pairwise Pearson correlation matrix from daily PnL data.
    Uses the same correlation calculation method as scripts/calculate_correlations.py
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to a dictionary containing DataFrame with PnL data
                      Format: {alpha_id: {'df': dataframe}}
        min_common_days: Minimum number of common trading days required
        use_cython: Whether to use Cython acceleration (default: True)
        
    Returns:
        DataFrame containing the correlation matrix
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

def calculate_performance_features(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                                 alpha_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance features for each alpha.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to a dictionary containing DataFrame with PnL data
                      Format: {alpha_id: {'df': dataframe}}
        alpha_metadata: DataFrame with alpha metadata
        
    Returns:
        DataFrame with alpha_id as index and performance features as columns
    """
    # Initialize features list
    features = []
    
    # Process each alpha
    for alpha_id, data in pnl_data_dict.items():
        try:
            # Skip if no PnL data or invalid structure
            if data is None or 'df' not in data or data['df'] is None:
                print(f"Warning: Missing PnL data for alpha {alpha_id}")
                continue
                
            pnl_df = data['df']
            
            if len(pnl_df) < 20:  # Skip alphas with too few data points
                print(f"Warning: Not enough data points for alpha {alpha_id}. Found {len(pnl_df)}.")
                continue
            
            # Check if pnl column exists
            if 'pnl' not in pnl_df.columns:
                print(f"Warning: Missing 'pnl' column for alpha {alpha_id}")
                continue
                
            # Extract pnl values and calculate daily changes
            daily_pnl = pnl_df['pnl'].diff().dropna()
            
            if len(daily_pnl) < 20:  # Ensure we have enough daily changes
                print(f"Warning: Not enough daily PnL changes for alpha {alpha_id}")
                continue
            
            # Calculate features
            feature_dict = {
                'alpha_id': alpha_id
            }
            
            # Add metadata features we already have
            if not alpha_metadata.empty and alpha_id in alpha_metadata.index:
                for feature in ['is_sharpe', 'is_drawdown', 'is_returns']:
                    if feature in alpha_metadata.columns:
                        feature_dict[feature.replace('is_', '')] = alpha_metadata.loc[alpha_id, feature]
            
            # Calculate additional features
            # Volatility (standard deviation of daily returns)
            feature_dict['volatility'] = daily_pnl.std()
            
            # Percentage of positive PnL days
            feature_dict['pct_positive_days'] = (daily_pnl > 0).mean()
            
            # Calmar Ratio (if max_drawdown is available and not zero)
            if 'drawdown' in feature_dict and feature_dict['drawdown'] != 0 and not np.isnan(feature_dict['drawdown']):
                annualized_return = daily_pnl.mean() * 252  # Assuming 252 trading days
                feature_dict['calmar_ratio'] = annualized_return / abs(feature_dict['drawdown'])
            
            # Skewness and Kurtosis
            feature_dict['skewness'] = daily_pnl.skew()
            feature_dict['kurtosis'] = daily_pnl.kurtosis()
            
            features.append(feature_dict)
        except Exception as e:
            print(f"Warning: Error calculating features for alpha {alpha_id}: {e}")
    
    # Convert to DataFrame
    if not features:
        print("Warning: No features could be calculated for any alphas")
        return pd.DataFrame()
        
    features_df = pd.DataFrame(features)
    features_df.set_index('alpha_id', inplace=True)
    
    # Fill NaN values with column means
    features_df = features_df.fillna(features_df.mean())
    
    return features_df

def apply_dimensionality_reduction(features_df: pd.DataFrame, 
                                 method: str = 'tsne',
                                 random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply dimensionality reduction to the feature matrix.
    
    Args:
        features_df: DataFrame with features
        method: Method to use ('tsne', 'umap', or 'pca')
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of:
        - DataFrame with alpha_id as index and x, y coordinates as columns
        - Dictionary with additional model information (loadings, variance explained, etc.)
    """
    # Check if we have enough samples
    if features_df.empty:
        print(f"Cannot apply {method}: Empty feature dataframe")
        return pd.DataFrame(), {}
        
    if len(features_df) < 2:
        print(f"Cannot apply {method}: Need at least 2 samples, got {len(features_df)}")
        return pd.DataFrame(), {}
    
    # Drop any rows with NaN values
    features_df = features_df.dropna()
    if features_df.empty or len(features_df) < 2:
        print(f"Cannot apply {method}: Not enough valid samples after dropping NaNs")
        return pd.DataFrame(), {}
    
    try:
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)
        
        # Configure dimensionality reduction model based on number of samples
        if method.lower() == 'tsne':
            # t-SNE requires perplexity to be less than n_samples - 1
            # Typically perplexity is between 5 and 50
            perplexity = min(30, max(5, len(features_df) // 5))
            # Make sure perplexity is less than n_samples - 1
            perplexity = min(perplexity, len(features_df) - 1)
            model = TSNE(n_components=2, perplexity=perplexity, 
                       random_state=random_state)
            
        elif method.lower() == 'umap':
            # UMAP requires n_neighbors to be less than n_samples
            n_neighbors = min(15, max(2, len(features_df) // 5))
            model = umap.UMAP(n_components=2, 
                            n_neighbors=n_neighbors,
                            min_dist=0.1, 
                            n_jobs=1,  # Explicit n_jobs=1 to avoid warning when using random_state
                            random_state=random_state)
                            
        elif method.lower() == 'pca':
            model = PCA(n_components=min(2, len(features_df.columns)), 
                       random_state=random_state)
                       
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Transform to 2D coordinates
        coords = model.fit_transform(scaled_features)
        
        # Create DataFrame with results
        result_df = pd.DataFrame(coords, 
                               index=features_df.index, 
                               columns=['x', 'y'])
        
        # Collect model information
        model_info = {'method': method}
        
        # For PCA, extract additional information
        if method.lower() == 'pca':
            # Get variance explained by each component
            variance_explained = model.explained_variance_ratio_
            model_info['variance_explained'] = {
                'pc1': float(variance_explained[0]),
                'pc2': float(variance_explained[1]) if len(variance_explained) > 1 else 0.0,
                'total': float(variance_explained.sum())
            }
            
            # Get feature loadings (components)
            feature_names = list(features_df.columns)
            loadings = model.components_
            
            # PC1 and PC2 loadings
            pc1_loadings = dict(zip(feature_names, loadings[0]))
            pc2_loadings = dict(zip(feature_names, loadings[1])) if loadings.shape[0] > 1 else {}
            
            model_info['loadings'] = {
                'pc1': pc1_loadings,
                'pc2': pc2_loadings,
                'feature_names': feature_names
            }
            
            # Calculate top contributing features for each PC
            pc1_contributions = [(feature, abs(loading)) for feature, loading in pc1_loadings.items()]
            pc1_contributions.sort(key=lambda x: x[1], reverse=True)
            
            pc2_contributions = [(feature, abs(loading)) for feature, loading in pc2_loadings.items()] if pc2_loadings else []
            pc2_contributions.sort(key=lambda x: x[1], reverse=True)
            
            model_info['top_features'] = {
                'pc1': pc1_contributions[:3],  # Top 3 features
                'pc2': pc2_contributions[:3]   # Top 3 features
            }
            
            # Generate interpretation hints
            def generate_pc_interpretation(loadings, contributions):
                if not loadings:
                    return "No data"
                
                positive_features = [f for f, l in loadings.items() if l > 0.2]
                negative_features = [f for f, l in loadings.items() if l < -0.2]
                
                interpretation = ""
                if positive_features:
                    interpretation += f"Higher: {', '.join(positive_features[:2])}"
                if negative_features:
                    if interpretation:
                        interpretation += " | "
                    interpretation += f"Lower: {', '.join(negative_features[:2])}"
                
                return interpretation or "Mixed factors"
            
            model_info['interpretation'] = {
                'pc1': generate_pc_interpretation(pc1_loadings, pc1_contributions),
                'pc2': generate_pc_interpretation(pc2_loadings, pc2_contributions)
            }
        
        return result_df, model_info
        
    except Exception as e:
        print(f"Error applying {method}: {e}")
        return pd.DataFrame(), {}

def generate_clustering_data(region: str) -> Dict[str, Any]:
    """
    Generate clustering data for a specific region.
    
    Args:
        region: Region name
        
    Returns:
        Dictionary with clustering results
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
        
        # Get alpha metadata (correlation statistics have useful metrics)
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
        }
        
        # Method 1: MDS on Correlation Matrix and Advanced Visualizations
        try:
            print("Calculating correlation matrix...")
            corr_matrix = calculate_correlation_matrix(pnl_data)
            
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
        
        # Method 2: Dimensionality Reduction on Performance Features
        try:
            print("Calculating performance features...")
            performance_features = calculate_performance_features(pnl_data, alpha_metadata)
            
            if not performance_features.empty and performance_features.shape[0] >= 2:
                # Initialize with empty results in case any method fails
                results['tsne_coords'] = {}
                results['umap_coords'] = {}
                results['pca_coords'] = {}
                
                # Apply different dimensionality reduction techniques
                try:
                    print("Applying t-SNE...")
                    tsne_coords, tsne_info = apply_dimensionality_reduction(performance_features, method='tsne')
                    results['tsne_coords'] = tsne_coords.to_dict(orient='index')
                    print(f"t-SNE coordinates calculated for {len(tsne_coords)} alphas")
                except Exception as e:
                    print(f"Error in t-SNE calculation: {e}")
                
                try:
                    print("Applying UMAP...")
                    umap_coords, umap_info = apply_dimensionality_reduction(performance_features, method='umap')
                    results['umap_coords'] = umap_coords.to_dict(orient='index')
                    print(f"UMAP coordinates calculated for {len(umap_coords)} alphas")
                except Exception as e:
                    print(f"Error in UMAP calculation: {e}")
                
                try:
                    print("Applying PCA...")
                    pca_coords, pca_info = apply_dimensionality_reduction(performance_features, method='pca')
                    results['pca_coords'] = pca_coords.to_dict(orient='index')
                    results['pca_info'] = pca_info  # Store PCA analysis information
                    print(f"PCA coordinates calculated for {len(pca_coords)} alphas")
                except Exception as e:
                    print(f"Error in PCA calculation: {e}")
            else:
                print("Skipping dimensionality reduction: Not enough performance features")
                results['tsne_coords'] = {}
                results['umap_coords'] = {}
                results['pca_coords'] = {}
        except Exception as e:
            print(f"Error calculating performance features: {e}")
            results['tsne_coords'] = {}
            results['umap_coords'] = {}
            results['pca_coords'] = {}
        
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
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            return super(NpEncoder, self).default(obj)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, cls=NpEncoder, indent=2)
    
    print(f"Clustering results saved to {filepath}")
    return filepath

def main(region: str = 'USA'):
    """
    Run the clustering analysis for a specific region.
    
    Args:
        region: Region name (default: 'USA')
    """
    print(f"Generating clustering data for region: {region}")
    results = generate_clustering_data(region)
    
    if results:
        output_path = save_clustering_results(results)
        print(f"Saved clustering results to: {output_path}")
        print(f"To view the results, run the visualization server with:")
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
