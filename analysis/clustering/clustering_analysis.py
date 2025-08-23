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
    get_pnl_data_for_alphas,
    get_correlation_statistics
)
from config.database_config import REGIONS

# Import scikit-learn components for dimensionality reduction
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

def calculate_correlation_matrix(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]], 
                                min_common_days: int = 60) -> pd.DataFrame:
    """
    Calculate pairwise Pearson correlation matrix from daily PnL data.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to a dictionary containing DataFrame with PnL data
                      Format: {alpha_id: {'df': dataframe}}
        min_common_days: Minimum number of common trading days required
        
    Returns:
        DataFrame containing the correlation matrix
    """
    # Extract alpha IDs and validate data
    alpha_ids = []
    for alpha_id, data in pnl_data_dict.items():
        # Ensure we have a valid dataframe with at least some data
        if data is None or 'df' not in data or data['df'] is None or len(data['df']) < min_common_days:
            print(f"Warning: Skipping alpha {alpha_id} due to insufficient data")
            continue
        alpha_ids.append(alpha_id)
    
    n_alphas = len(alpha_ids)
    if n_alphas < 2:
        print(f"Warning: Not enough alphas with valid data for correlation calculation. Found {n_alphas}.")
        return pd.DataFrame()
    
    # Prepare correlation matrix
    corr_matrix = pd.DataFrame(np.eye(n_alphas), 
                             index=alpha_ids, 
                             columns=alpha_ids)
    
    # Calculate correlations between all pairs of alphas
    for i in range(n_alphas):
        alpha_id1 = alpha_ids[i]
        alpha_data1 = pnl_data_dict[alpha_id1]
        
        # Check if data exists and has expected structure
        if alpha_data1 is None or 'df' not in alpha_data1 or alpha_data1['df'] is None:
            continue
            
        alpha_pnl1 = alpha_data1['df']
        
        for j in range(i+1, n_alphas):
            alpha_id2 = alpha_ids[j]
            alpha_data2 = pnl_data_dict[alpha_id2]
            
            # Check if data exists and has expected structure
            if alpha_data2 is None or 'df' not in alpha_data2 or alpha_data2['df'] is None:
                continue
                
            alpha_pnl2 = alpha_data2['df']
            
            # Ensure dataframes have an index (dates)
            if not hasattr(alpha_pnl1, 'index') or not hasattr(alpha_pnl2, 'index'):
                print(f"Warning: Missing index for alpha pair {alpha_id1}/{alpha_id2}")
                continue
            
            try:
                # Find common dates between the two alphas
                common_dates = alpha_pnl1.index.intersection(alpha_pnl2.index)
                
                # Only calculate correlation if we have enough common dates
                if len(common_dates) >= min_common_days:
                    # Check if 'pnl' column exists
                    if 'pnl' not in alpha_pnl1.columns or 'pnl' not in alpha_pnl2.columns:
                        print(f"Warning: Missing 'pnl' column for alpha pair {alpha_id1}/{alpha_id2}")
                        continue
                        
                    # Calculate correlation of daily returns
                    pnl1 = alpha_pnl1.loc[common_dates, 'pnl']
                    pnl2 = alpha_pnl2.loc[common_dates, 'pnl']
                    
                    # Convert PnL to daily returns (assumes starting with cumulative PnL)
                    returns1 = pnl1.diff().dropna()
                    returns2 = pnl2.diff().dropna()
                    
                    # Calculate correlation on common dates with valid returns
                    common_return_dates = returns1.index.intersection(returns2.index)
                    if len(common_return_dates) >= min_common_days:
                        correlation = returns1[common_return_dates].corr(returns2[common_return_dates])
                        
                        # Store in symmetric matrix
                        corr_matrix.loc[alpha_id1, alpha_id2] = correlation
                        corr_matrix.loc[alpha_id2, alpha_id1] = correlation
            except Exception as e:
                print(f"Warning: Error calculating correlation for {alpha_id1}/{alpha_id2}: {e}")
    
    return corr_matrix

def mds_on_correlation_matrix(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Multidimensional Scaling (MDS) on the correlation matrix to get 2D coordinates.
    
    Args:
        corr_matrix: Correlation matrix (DataFrame)
        
    Returns:
        DataFrame with alpha_id as index and x, y coordinates as columns
    """
    # Convert correlation to dissimilarity: D_ij = 1 - C_ij
    dissimilarity_matrix = 1 - corr_matrix
    
    # Ensure diagonal is 0
    np.fill_diagonal(dissimilarity_matrix.values, 0)
    
    # Apply MDS
    mds = MDS(n_components=2, 
             dissimilarity='precomputed', 
             random_state=0)
    
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
                                 random_state: int = 42) -> pd.DataFrame:
    """
    Apply dimensionality reduction to the feature matrix.
    
    Args:
        features_df: DataFrame with features
        method: Method to use ('tsne', 'umap', or 'pca')
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with alpha_id as index and x, y coordinates as columns
    """
    # Check if we have enough samples
    if features_df.empty:
        print(f"Cannot apply {method}: Empty feature dataframe")
        return pd.DataFrame()
        
    if len(features_df) < 2:
        print(f"Cannot apply {method}: Need at least 2 samples, got {len(features_df)}")
        return pd.DataFrame()
    
    # Drop any rows with NaN values
    features_df = features_df.dropna()
    if features_df.empty or len(features_df) < 2:
        print(f"Cannot apply {method}: Not enough valid samples after dropping NaNs")
        return pd.DataFrame()
    
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
        
        return result_df
        
    except Exception as e:
        print(f"Error applying {method}: {e}")
        return pd.DataFrame()

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
        alpha_ids = get_all_alpha_ids_by_region_basic(region)
        
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
        
        # Method 1: MDS on Correlation Matrix
        try:
            print("Calculating correlation matrix...")
            corr_matrix = calculate_correlation_matrix(pnl_data)
            
            if not corr_matrix.empty and corr_matrix.shape[0] >= 2:
                print("Applying MDS to correlation matrix...")
                mds_coords = mds_on_correlation_matrix(corr_matrix)
                results['mds_coords'] = mds_coords.to_dict(orient='index')
                print(f"MDS coordinates calculated for {len(mds_coords)} alphas")
            else:
                print("Skipping MDS: Not enough data in correlation matrix")
                results['mds_coords'] = {}
        except Exception as e:
            print(f"Error in MDS calculation: {e}")
            results['mds_coords'] = {}
        
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
                    tsne_coords = apply_dimensionality_reduction(performance_features, method='tsne')
                    results['tsne_coords'] = tsne_coords.to_dict(orient='index')
                    print(f"t-SNE coordinates calculated for {len(tsne_coords)} alphas")
                except Exception as e:
                    print(f"Error in t-SNE calculation: {e}")
                
                try:
                    print("Applying UMAP...")
                    umap_coords = apply_dimensionality_reduction(performance_features, method='umap')
                    results['umap_coords'] = umap_coords.to_dict(orient='index')
                    print(f"UMAP coordinates calculated for {len(umap_coords)} alphas")
                except Exception as e:
                    print(f"Error in UMAP calculation: {e}")
                
                try:
                    print("Applying PCA...")
                    pca_coords = apply_dimensionality_reduction(performance_features, method='pca')
                    results['pca_coords'] = pca_coords.to_dict(orient='index')
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
