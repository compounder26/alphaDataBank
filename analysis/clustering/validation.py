"""
Validation and Quality Assessment Framework for Alpha Clustering.
- Multi-metric stability assessment
- Parameter robustness testing
- Economic coherence validation
- Temporal stability analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

logger = logging.getLogger(__name__)


class ClusteringValidator:
    """
    Comprehensive validation framework for clustering results.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the clustering validator.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
    
    def assess_clustering_quality(self, features_df: pd.DataFrame, 
                                labels: np.ndarray,
                                distance_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Assess clustering quality using multiple metrics.
        
        Args:
            features_df: Feature matrix used for clustering
            labels: Cluster labels
            distance_matrix: Optional precomputed distance matrix
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Check if we have enough clusters and samples
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_samples = len(labels)
        
        if n_clusters < 2 or n_samples < 2:
            logger.warning("Not enough clusters or samples for quality assessment")
            return {'error': 'insufficient_data'}
        
        # Silhouette score
        try:
            if distance_matrix is not None:
                silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
            else:
                silhouette = silhouette_score(features_df, labels)
            metrics['silhouette_score'] = float(silhouette)
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {e}")
            metrics['silhouette_score'] = np.nan
        
        # Calinski-Harabasz index (only for non-precomputed metrics)
        if distance_matrix is None:
            try:
                ch_score = calinski_harabasz_score(features_df, labels)
                metrics['calinski_harabasz_score'] = float(ch_score)
            except Exception as e:
                logger.warning(f"Could not calculate Calinski-Harabasz score: {e}")
                metrics['calinski_harabasz_score'] = np.nan
        
        # Cluster size statistics
        cluster_sizes = pd.Series(labels).value_counts()
        metrics['n_clusters'] = n_clusters
        metrics['avg_cluster_size'] = float(cluster_sizes.mean())
        metrics['min_cluster_size'] = int(cluster_sizes.min())
        metrics['max_cluster_size'] = int(cluster_sizes.max())
        metrics['cluster_size_std'] = float(cluster_sizes.std())
        
        # Balance metric (how evenly distributed are cluster sizes)
        expected_size = n_samples / n_clusters
        size_deviations = abs(cluster_sizes - expected_size) / expected_size
        metrics['cluster_balance'] = 1.0 / (1.0 + size_deviations.mean())
        
        # Separation metric (average distance between cluster centers)
        if distance_matrix is None and n_clusters > 1:
            try:
                cluster_centers = []
                for label in unique_labels:
                    mask = labels == label
                    if np.sum(mask) > 0:
                        center = features_df.iloc[mask].mean()
                        cluster_centers.append(center)
                
                if len(cluster_centers) > 1:
                    centers_df = pd.DataFrame(cluster_centers)
                    center_distances = []
                    for i in range(len(cluster_centers)):
                        for j in range(i + 1, len(cluster_centers)):
                            dist = np.linalg.norm(centers_df.iloc[i] - centers_df.iloc[j])
                            center_distances.append(dist)
                    
                    metrics['avg_center_separation'] = float(np.mean(center_distances))
                    
            except Exception as e:
                logger.warning(f"Could not calculate center separation: {e}")
        
        logger.info(f"Quality assessment: Silhouette={metrics.get('silhouette_score', 'N/A'):.3f}, "
                   f"Clusters={n_clusters}, Balance={metrics.get('cluster_balance', 'N/A'):.3f}")
        
        return metrics
    
    def test_parameter_stability(self, features_df: pd.DataFrame,
                               clustering_method: str,
                               param_ranges: Dict[str, List],
                               n_trials: int = 5,
                               perturbation_std: float = 0.05) -> Dict[str, Any]:
        """
        Test clustering stability across parameter variations and data perturbations.
        
        Args:
            features_df: Feature matrix
            clustering_method: Method name ('tsne', 'umap', 'pca', 'hierarchical')
            param_ranges: Dictionary of parameter ranges to test
            n_trials: Number of perturbation trials per parameter set
            perturbation_std: Standard deviation of Gaussian noise for perturbation
            
        Returns:
            Dictionary with stability results
        """
        if features_df.empty or len(features_df) < 4:
            return {'error': 'insufficient_data'}
        
        logger.info(f"Testing parameter stability for {clustering_method} with {len(param_ranges)} parameter sets")
        
        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        param_combinations = list(itertools.product(*param_values))
        
        stability_results = []
        
        for param_combo in param_combinations:
            param_dict = dict(zip(param_names, param_combo))
            
            # Test stability with data perturbations
            trial_labels = []
            
            for trial in range(n_trials):
                # Add small random perturbation to test robustness
                perturbation = np.random.normal(0, perturbation_std, features_df.shape)
                perturbed_features = features_df + perturbation
                
                try:
                    # Perform clustering with this parameter set
                    labels = self._cluster_with_params(perturbed_features, clustering_method, param_dict)
                    if labels is not None:
                        trial_labels.append(labels)
                except Exception as e:
                    logger.warning(f"Clustering failed for params {param_dict}, trial {trial}: {e}")
            
            # Calculate stability metrics for this parameter set
            if len(trial_labels) > 1:
                stability_metrics = self._calculate_stability_metrics(trial_labels)
                stability_metrics['parameters'] = param_dict
                stability_results.append(stability_metrics)
        
        if not stability_results:
            return {'error': 'no_stable_results'}
        
        # Find most stable parameter set
        best_params = max(stability_results, key=lambda x: x.get('avg_stability', 0))
        
        results = {
            'best_parameters': best_params['parameters'],
            'best_stability_score': best_params.get('avg_stability', 0),
            'all_results': stability_results,
            'n_parameter_sets_tested': len(stability_results),
            'n_trials_per_set': n_trials
        }
        
        logger.info(f"Best stability score: {results['best_stability_score']:.3f} "
                   f"with parameters: {results['best_parameters']}")
        
        return results
    
    def _cluster_with_params(self, features_df: pd.DataFrame, 
                           method: str, 
                           params: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Perform clustering with specified method and parameters.
        
        Args:
            features_df: Feature matrix
            method: Clustering method name
            params: Parameter dictionary
            
        Returns:
            Cluster labels or None if clustering fails
        """
        try:
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df)
            
            if method == 'hierarchical':
                n_clusters = params.get('n_clusters', 5)
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clusterer.fit_predict(scaled_features)
                
            elif method == 'kmeans':
                n_clusters = params.get('n_clusters', 5)
                clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                labels = clusterer.fit_predict(scaled_features)
                
            elif method == 'tsne_clustering':
                # First apply t-SNE, then cluster
                perplexity = params.get('perplexity', 30)
                n_clusters = params.get('n_clusters', 5)
                
                tsne = TSNE(n_components=2, perplexity=perplexity, 
                           random_state=self.random_state)
                tsne_coords = tsne.fit_transform(scaled_features)
                
                clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                labels = clusterer.fit_predict(tsne_coords)
                
            elif method == 'umap_clustering':
                # First apply UMAP, then cluster
                n_neighbors = params.get('n_neighbors', 15)
                min_dist = params.get('min_dist', 0.1)
                n_clusters = params.get('n_clusters', 5)
                
                umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                     min_dist=min_dist, random_state=self.random_state)
                umap_coords = umap_model.fit_transform(scaled_features)
                
                clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
                labels = clusterer.fit_predict(umap_coords)
                
            else:
                logger.warning(f"Unknown clustering method: {method}")
                return None
                
            return labels
            
        except Exception as e:
            logger.warning(f"Clustering failed with method {method} and params {params}: {e}")
            return None
    
    def _calculate_stability_metrics(self, trial_labels: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate stability metrics across multiple clustering trials.
        
        Args:
            trial_labels: List of label arrays from different trials
            
        Returns:
            Dictionary of stability metrics
        """
        if len(trial_labels) < 2:
            return {'avg_stability': 0, 'std_stability': 0}
        
        # Calculate pairwise Adjusted Rand Index (ARI) scores
        ari_scores = []
        
        for i in range(len(trial_labels)):
            for j in range(i + 1, len(trial_labels)):
                try:
                    ari = adjusted_rand_score(trial_labels[i], trial_labels[j])
                    if not np.isnan(ari):
                        ari_scores.append(ari)
                except Exception as e:
                    logger.warning(f"Could not calculate ARI: {e}")
        
        if ari_scores:
            avg_stability = np.mean(ari_scores)
            std_stability = np.std(ari_scores)
        else:
            avg_stability = 0
            std_stability = 1  # High uncertainty
        
        return {
            'avg_stability': float(avg_stability),
            'std_stability': float(std_stability),
            'n_comparisons': len(ari_scores)
        }
    
    def assess_economic_coherence(self, features_df: pd.DataFrame,
                                labels: np.ndarray,
                                alpha_metadata: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Assess whether clusters correspond to economically meaningful groups.
        
        Args:
            features_df: Feature matrix
            labels: Cluster labels
            alpha_metadata: Optional metadata with economic indicators
            
        Returns:
            Dictionary with economic coherence metrics
        """
        coherence_metrics = {}
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return {'error': 'insufficient_clusters'}
        
        # Analyze feature coherence within clusters
        cluster_coherence = []
        
        for label in unique_labels:
            mask = labels == label
            cluster_features = features_df.iloc[mask]
            
            if len(cluster_features) > 1:
                # Calculate intra-cluster feature correlation
                feature_corrs = []
                for col in cluster_features.columns:
                    if cluster_features[col].std() > 0:  # Avoid constant features
                        # Correlation of this feature with cluster mean
                        cluster_mean = cluster_features.mean()
                        corr = cluster_features[col].corr(cluster_mean.mean())
                        if not np.isnan(corr):
                            feature_corrs.append(abs(corr))
                
                if feature_corrs:
                    cluster_coherence.append(np.mean(feature_corrs))
        
        if cluster_coherence:
            coherence_metrics['avg_feature_coherence'] = float(np.mean(cluster_coherence))
            coherence_metrics['coherence_consistency'] = 1.0 / (1.0 + np.std(cluster_coherence))
        
        # Analyze economic metadata coherence if available
        if alpha_metadata is not None and not alpha_metadata.empty:
            economic_coherence = self._analyze_metadata_coherence(labels, alpha_metadata)
            coherence_metrics.update(economic_coherence)
        
        # Risk profile coherence
        risk_coherence = self._analyze_risk_coherence(features_df, labels)
        coherence_metrics.update(risk_coherence)
        
        logger.info(f"Economic coherence assessment: "
                   f"Feature coherence={coherence_metrics.get('avg_feature_coherence', 'N/A'):.3f}")
        
        return coherence_metrics
    
    def _analyze_metadata_coherence(self, labels: np.ndarray, 
                                  metadata: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze coherence of economic metadata within clusters.
        
        Args:
            labels: Cluster labels
            metadata: Metadata DataFrame
            
        Returns:
            Dictionary of metadata coherence metrics
        """
        coherence_metrics = {}
        
        # Align metadata with labels
        common_indices = metadata.index.intersection(pd.Index(range(len(labels))))
        if len(common_indices) < 2:
            return coherence_metrics
        
        aligned_metadata = metadata.loc[common_indices]
        aligned_labels = labels[common_indices]
        
        # Analyze categorical features
        categorical_cols = aligned_metadata.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            categorical_coherence = []
            
            for col in categorical_cols:
                if aligned_metadata[col].notna().sum() > 0:
                    # Calculate purity for this categorical feature
                    purity = self._calculate_categorical_purity(aligned_labels, aligned_metadata[col])
                    categorical_coherence.append(purity)
            
            if categorical_coherence:
                coherence_metrics['categorical_purity'] = float(np.mean(categorical_coherence))
        
        # Analyze numerical features
        numerical_cols = aligned_metadata.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            numerical_coherence = []
            
            for col in numerical_cols:
                if aligned_metadata[col].notna().sum() > 0:
                    # Calculate within-cluster vs between-cluster variance ratio
                    coherence = self._calculate_numerical_coherence(aligned_labels, aligned_metadata[col])
                    if coherence is not None:
                        numerical_coherence.append(coherence)
            
            if numerical_coherence:
                coherence_metrics['numerical_coherence'] = float(np.mean(numerical_coherence))
        
        return coherence_metrics
    
    def _calculate_categorical_purity(self, labels: np.ndarray, 
                                    categorical_feature: pd.Series) -> float:
        """
        Calculate purity of categorical features within clusters.
        
        Args:
            labels: Cluster labels
            categorical_feature: Categorical feature series
            
        Returns:
            Average purity score
        """
        unique_labels = np.unique(labels)
        purities = []
        
        for label in unique_labels:
            mask = labels == label
            cluster_categories = categorical_feature[mask].dropna()
            
            if len(cluster_categories) > 0:
                # Purity = fraction of most common category in cluster
                most_common_count = cluster_categories.value_counts().iloc[0]
                purity = most_common_count / len(cluster_categories)
                purities.append(purity)
        
        return float(np.mean(purities)) if purities else 0.0
    
    def _calculate_numerical_coherence(self, labels: np.ndarray, 
                                     numerical_feature: pd.Series) -> Optional[float]:
        """
        Calculate coherence of numerical features within clusters.
        
        Args:
            labels: Cluster labels
            numerical_feature: Numerical feature series
            
        Returns:
            Coherence score (higher = more coherent)
        """
        try:
            # Calculate within-cluster variance
            unique_labels = np.unique(labels)
            within_cluster_vars = []
            cluster_sizes = []
            
            for label in unique_labels:
                mask = labels == label
                cluster_values = numerical_feature[mask].dropna()
                
                if len(cluster_values) > 1:
                    within_cluster_vars.append(cluster_values.var())
                    cluster_sizes.append(len(cluster_values))
            
            if not within_cluster_vars:
                return None
            
            # Weighted average of within-cluster variances
            total_size = sum(cluster_sizes)
            weights = [size / total_size for size in cluster_sizes]
            avg_within_var = np.average(within_cluster_vars, weights=weights)
            
            # Total variance
            total_var = numerical_feature.dropna().var()
            
            if total_var == 0:
                return 1.0  # Perfect coherence if no variance
            
            # Coherence = 1 - (within_cluster_var / total_var)
            coherence = 1.0 - (avg_within_var / total_var)
            return float(max(0.0, coherence))  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Error calculating numerical coherence: {e}")
            return None
    
    def _analyze_risk_coherence(self, features_df: pd.DataFrame, 
                              labels: np.ndarray) -> Dict[str, float]:
        """
        Analyze coherence of risk profiles within clusters.
        
        Args:
            features_df: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of risk coherence metrics
        """
        risk_metrics = {}
        
        # Look for risk-related features
        risk_features = [col for col in features_df.columns 
                        if any(keyword in col.lower() 
                              for keyword in ['vol', 'risk', 'drawdown', 'var', 'sharpe', 'sortino'])]
        
        if not risk_features:
            return risk_metrics
        
        risk_data = features_df[risk_features]
        unique_labels = np.unique(labels)
        
        # Calculate within-cluster homogeneity for risk features
        risk_homogeneities = []
        
        for label in unique_labels:
            mask = labels == label
            cluster_risk = risk_data.iloc[mask]
            
            if len(cluster_risk) > 1:
                # Calculate coefficient of variation for each risk feature
                cvs = []
                for col in cluster_risk.columns:
                    if cluster_risk[col].std() > 0 and cluster_risk[col].mean() != 0:
                        cv = cluster_risk[col].std() / abs(cluster_risk[col].mean())
                        cvs.append(cv)
                
                if cvs:
                    # Low CV indicates high homogeneity
                    avg_cv = np.mean(cvs)
                    homogeneity = 1.0 / (1.0 + avg_cv)
                    risk_homogeneities.append(homogeneity)
        
        if risk_homogeneities:
            risk_metrics['risk_profile_homogeneity'] = float(np.mean(risk_homogeneities))
        
        return risk_metrics
    
    def temporal_stability_analysis(self, pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]],
                                  features_df: pd.DataFrame,
                                  labels: np.ndarray,
                                  window_size: int = 120) -> Dict[str, Any]:
        """
        Analyze temporal stability of cluster assignments.
        
        Args:
            pnl_data_dict: PnL data dictionary
            features_df: Current feature matrix
            labels: Current cluster labels
            window_size: Rolling window size for temporal analysis
            
        Returns:
            Dictionary with temporal stability metrics
        """
        logger.info(f"Analyzing temporal stability with {window_size}-day windows")
        
        # Extract time series data
        alpha_ids = features_df.index
        temporal_data = {}
        
        for alpha_id in alpha_ids:
            if alpha_id in pnl_data_dict:
                data = pnl_data_dict[alpha_id]
                if data and 'df' in data and data['df'] is not None:
                    df = data['df']
                    if 'pnl' in df.columns and len(df) >= window_size:
                        temporal_data[alpha_id] = df['pnl']
        
        if len(temporal_data) < len(alpha_ids) // 2:
            return {'error': 'insufficient_temporal_data'}
        
        # Calculate rolling correlations to track relationship stability
        stability_scores = []
        
        # Get the first common date range
        common_dates = None
        for alpha_id, pnl_series in temporal_data.items():
            if common_dates is None:
                common_dates = pnl_series.index
            else:
                common_dates = common_dates.intersection(pnl_series.index)
        
        if len(common_dates) < window_size * 2:
            return {'error': 'insufficient_common_dates'}
        
        # Analyze stability in rolling windows
        n_windows = len(common_dates) - window_size + 1
        window_step = max(1, n_windows // 10)  # Analyze up to 10 windows
        
        for i in range(0, n_windows, window_step):
            window_dates = common_dates[i:i + window_size]
            
            # Calculate features for this time window
            window_features = self._calculate_window_features(temporal_data, window_dates, alpha_ids)
            
            if window_features is not None and len(window_features) > 2:
                # Compare with original clustering
                stability_score = self._compare_feature_stability(features_df, window_features, labels)
                if stability_score is not None:
                    stability_scores.append(stability_score)
        
        if not stability_scores:
            return {'error': 'no_stability_scores'}
        
        temporal_metrics = {
            'avg_temporal_stability': float(np.mean(stability_scores)),
            'stability_std': float(np.std(stability_scores)),
            'min_stability': float(np.min(stability_scores)),
            'max_stability': float(np.max(stability_scores)),
            'n_windows_analyzed': len(stability_scores),
            'window_size': window_size
        }
        
        logger.info(f"Temporal stability: {temporal_metrics['avg_temporal_stability']:.3f} "
                   f"(±{temporal_metrics['stability_std']:.3f}) over {len(stability_scores)} windows")
        
        return temporal_metrics
    
    def _calculate_window_features(self, temporal_data: Dict[str, pd.Series], 
                                 window_dates: pd.DatetimeIndex,
                                 alpha_ids: pd.Index) -> Optional[pd.DataFrame]:
        """
        Calculate features for a specific time window.
        
        Args:
            temporal_data: Dictionary of PnL time series
            window_dates: Date range for this window
            alpha_ids: Alpha IDs to include
            
        Returns:
            Feature matrix for this window or None if insufficient data
        """
        try:
            window_features = []
            
            for alpha_id in alpha_ids:
                if alpha_id in temporal_data:
                    pnl_series = temporal_data[alpha_id]
                    window_pnl = pnl_series.loc[window_dates]
                    
                    if len(window_pnl) >= len(window_dates) * 0.8:  # At least 80% data coverage
                        returns = window_pnl.pct_change().dropna()
                        
                        if len(returns) > 10:
                            # Calculate basic features for this window
                            features = {
                                'volatility': returns.std(),
                                'mean_return': returns.mean(),
                                'skewness': returns.skew(),
                                'kurtosis': returns.kurtosis(),
                                'max_drawdown': self._calculate_simple_drawdown(window_pnl)
                            }
                            
                            features['alpha_id'] = alpha_id
                            window_features.append(features)
            
            if len(window_features) < 2:
                return None
            
            features_df = pd.DataFrame(window_features)
            features_df.set_index('alpha_id', inplace=True)
            features_df = features_df.fillna(features_df.median())
            
            return features_df
            
        except Exception as e:
            logger.warning(f"Error calculating window features: {e}")
            return None
    
    def _calculate_simple_drawdown(self, pnl_series: pd.Series) -> float:
        """
        Calculate simple maximum drawdown.
        
        Args:
            pnl_series: PnL time series
            
        Returns:
            Maximum drawdown as a fraction
        """
        try:
            cumulative = pnl_series.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / (running_max + 1e-8)  # Avoid division by zero
            return float(drawdown.min())
        except:
            return 0.0
    
    def _compare_feature_stability(self, original_features: pd.DataFrame,
                                 window_features: pd.DataFrame,
                                 original_labels: np.ndarray) -> Optional[float]:
        """
        Compare feature similarity between original and window features.
        
        Args:
            original_features: Original feature matrix
            window_features: Window-specific feature matrix
            original_labels: Original cluster labels
            
        Returns:
            Stability score or None if comparison fails
        """
        try:
            # Find common alphas
            common_alphas = original_features.index.intersection(window_features.index)
            
            if len(common_alphas) < 3:
                return None
            
            # Align features
            orig_aligned = original_features.loc[common_alphas]
            window_aligned = window_features.loc[common_alphas]
            
            # Standardize both feature sets
            scaler1 = StandardScaler()
            scaler2 = StandardScaler()
            
            orig_scaled = scaler1.fit_transform(orig_aligned.fillna(0))
            window_scaled = scaler2.fit_transform(window_aligned.fillna(0))
            
            # Calculate correlation between feature sets
            correlations = []
            min_cols = min(orig_scaled.shape[1], window_scaled.shape[1])
            
            for i in range(min_cols):
                if orig_scaled[:, i].std() > 0 and window_scaled[:, i].std() > 0:
                    corr = np.corrcoef(orig_scaled[:, i], window_scaled[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                return float(np.mean(correlations))
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error comparing feature stability: {e}")
            return None


def validate_clustering_pipeline(features_df: pd.DataFrame,
                               labels: np.ndarray,
                               pnl_data_dict: Optional[Dict] = None,
                               alpha_metadata: Optional[pd.DataFrame] = None,
                               clustering_method: str = 'hierarchical',
                               param_ranges: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run comprehensive validation pipeline for clustering results.
    
    Args:
        features_df: Feature matrix
        labels: Cluster labels
        pnl_data_dict: Optional PnL data for temporal analysis
        alpha_metadata: Optional metadata for economic coherence
        clustering_method: Method used for clustering
        param_ranges: Parameter ranges for stability testing
        
    Returns:
        Comprehensive validation report
    """
    validator = ClusteringValidator()
    
    validation_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_alphas': len(features_df),
        'n_features': len(features_df.columns),
        'n_clusters': len(np.unique(labels)),
        'clustering_method': clustering_method
    }
    
    logger.info(f"Running validation pipeline for {len(features_df)} alphas, "
               f"{len(features_df.columns)} features, {len(np.unique(labels))} clusters")
    
    # Quality assessment
    logger.info("Assessing clustering quality...")
    quality_metrics = validator.assess_clustering_quality(features_df, labels)
    validation_report['quality_metrics'] = quality_metrics
    
    # Economic coherence
    logger.info("Assessing economic coherence...")
    coherence_metrics = validator.assess_economic_coherence(features_df, labels, alpha_metadata)
    validation_report['economic_coherence'] = coherence_metrics
    
    # Parameter stability (if ranges provided)
    if param_ranges:
        logger.info("Testing parameter stability...")
        stability_results = validator.test_parameter_stability(
            features_df, clustering_method, param_ranges
        )
        validation_report['parameter_stability'] = stability_results
    
    # Temporal stability (if PnL data provided)
    if pnl_data_dict:
        logger.info("Analyzing temporal stability...")
        temporal_results = validator.temporal_stability_analysis(
            pnl_data_dict, features_df, labels
        )
        validation_report['temporal_stability'] = temporal_results
    
    # Overall validation score
    validation_score = calculate_overall_validation_score(validation_report)
    validation_report['overall_validation_score'] = validation_score
    
    logger.info(f"Validation complete. Overall score: {validation_score:.3f}")
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            # Convert both keys and values
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
                new_dict[key] = convert_numpy_types(value)
            return new_dict
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, int, np.int32, np.int64, np.intc)):
            return int(obj)
        elif isinstance(obj, (np.floating, float, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    validation_report = convert_numpy_types(validation_report)
    
    return validation_report


def calculate_overall_validation_score(validation_report: Dict[str, Any]) -> float:
    """
    Calculate an overall validation score from multiple metrics.
    
    Args:
        validation_report: Complete validation report
        
    Returns:
        Overall validation score (0-1, higher is better)
    """
    scores = []
    weights = []
    
    # Quality metrics weight: 30%
    quality = validation_report.get('quality_metrics', {})
    if 'silhouette_score' in quality and not np.isnan(quality['silhouette_score']):
        # Convert silhouette score (-1 to 1) to (0 to 1)
        silhouette_norm = (quality['silhouette_score'] + 1) / 2
        scores.append(silhouette_norm)
        weights.append(0.3)
    
    # Economic coherence weight: 25%
    coherence = validation_report.get('economic_coherence', {})
    if 'avg_feature_coherence' in coherence:
        scores.append(coherence['avg_feature_coherence'])
        weights.append(0.25)
    
    # Parameter stability weight: 25%
    stability = validation_report.get('parameter_stability', {})
    if 'best_stability_score' in stability:
        scores.append(stability['best_stability_score'])
        weights.append(0.25)
    
    # Temporal stability weight: 20%
    temporal = validation_report.get('temporal_stability', {})
    if 'avg_temporal_stability' in temporal:
        scores.append(temporal['avg_temporal_stability'])
        weights.append(0.2)
    
    if not scores:
        return 0.0
    
    # Weighted average
    total_weight = sum(weights)
    if total_weight > 0:
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    else:
        weighted_score = np.mean(scores)
    
    return float(np.clip(weighted_score, 0, 1))