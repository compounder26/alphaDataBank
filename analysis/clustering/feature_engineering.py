"""
Enhanced Feature Engineering for Alpha Clustering.
- Spiked covariance model features
- Multi-scale temporal features with exponential weighting
- Idiosyncratic volatility normalization
- Advanced risk metrics for clustering
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)


def estimate_idiosyncratic_volatility(returns_df: pd.DataFrame, 
                                    n_factors: int = 5, 
                                    half_life: int = 60) -> pd.Series:
    """
    Estimate idiosyncratic volatilities using a fast-decay exponential weighting.
    
    Args:
        returns_df: DataFrame with returns (alphas as columns, dates as rows)
        n_factors: Number of factors for decomposition
        half_life: Half-life for exponential weighting (default: 60 days)
        
    Returns:
        Series of idiosyncratic volatilities for each alpha
    """
    if returns_df.empty or len(returns_df) < 20:
        logger.warning("Insufficient data for idiosyncratic volatility estimation")
        return pd.Series(index=returns_df.columns)
    
    # Create exponential weights (recent observations weighted more heavily)
    n_periods = len(returns_df)
    decay_factor = np.log(0.5) / half_life
    weights = np.exp(decay_factor * np.arange(n_periods - 1, -1, -1))
    weights = weights / weights.sum()
    
    # Apply weights to returns with numerical stability
    weight_matrix = np.sqrt(weights[:, np.newaxis])
    weighted_returns = returns_df * weight_matrix
    
    # Fill any remaining NaN/inf values
    weighted_returns = weighted_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Estimate factor model with weighted data
    try:
        # Check for valid data before PCA
        if weighted_returns.std().min() < 1e-10:  # Some columns are constant
            logger.warning("Some returns series are constant, using fallback volatility")
            return returns_df.std().fillna(0.01)  # Small positive default
            
        # Use PCA to identify common factors
        n_components = min(n_factors, len(returns_df.columns), weighted_returns.shape[0] - 1)
        if n_components <= 0:
            return returns_df.std().fillna(0.01)
            
        pca = PCA(n_components=n_components)
        factor_loadings = pca.fit(weighted_returns).components_.T
        
        # Calculate factor exposures and residual variances with bounds checking
        idio_vars = {}
        for i, alpha_id in enumerate(returns_df.columns):
            alpha_returns = weighted_returns.iloc[:, i].fillna(0)
            
            # Project onto factor space with safety checks
            if i < len(factor_loadings):
                factor_exposure = factor_loadings[i, :]
                
                # Calculate residual variance (idiosyncratic component)
                total_var = np.var(alpha_returns)
                if np.isfinite(total_var) and total_var > 1e-10:
                    factor_var = np.sum(factor_exposure**2 * pca.explained_variance_)
                    if np.isfinite(factor_var) and factor_var >= 0:
                        idio_var = max(0.001, total_var - factor_var)  # Prevent negative variance
                        idio_vars[alpha_id] = np.sqrt(idio_var)
                    else:
                        idio_vars[alpha_id] = np.sqrt(total_var * 0.5)  # Fallback: assume 50% idiosyncratic
                else:
                    idio_vars[alpha_id] = 0.01  # Small positive default
            else:
                idio_vars[alpha_id] = 0.01  # Small positive default
            
        return pd.Series(idio_vars)
        
    except Exception as e:
        logger.warning(f"Error in idiosyncratic volatility estimation: {e}")
        # Fallback to simple volatility
        return returns_df.std()


def calculate_spiked_covariance_features(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]],
                                       n_factors: int = 5) -> pd.DataFrame:
    """
    Extract features from the spiked covariance model.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to PnL data
        n_factors: Number of principal factors to extract
        
    Returns:
        DataFrame with spiked covariance features
    """
    # Convert PnL data to returns
    returns_data = {}
    for alpha_id, data in pnl_data_dict.items():
        if data is None or 'df' not in data or data['df'] is None:
            continue
        df = data['df']
        if 'pnl' not in df.columns or len(df) < 30:
            continue
        
        # Calculate returns from cumulative PnL with robust handling
        pnl_series = df['pnl'].fillna(method='ffill').fillna(0)  # Fill any NaN values
        
        # Handle edge cases in PnL data
        if pnl_series.abs().max() < 1e-10:  # Essentially zero PnL
            continue
            
        # Calculate percentage returns more robustly
        returns = pnl_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        # Additional cleaning - cap extreme values (winsorize at 5%/95%)
        if len(returns) > 20 and returns.std() > 1e-10:  # Non-constant returns
            lower_bound = returns.quantile(0.05)
            upper_bound = returns.quantile(0.95)
            returns = returns.clip(lower=lower_bound, upper=upper_bound)
            returns_data[alpha_id] = returns
    
    if len(returns_data) < 2:
        logger.warning("Insufficient data for spiked covariance features")
        return pd.DataFrame()
    
    # Create returns dataframe
    returns_df = pd.DataFrame(returns_data).fillna(0)
    
    # Stage 1: Estimate idiosyncratic volatilities
    idio_vols = estimate_idiosyncratic_volatility(returns_df, n_factors=n_factors)
    
    # Stage 2: Normalize returns by idiosyncratic volatility
    normalized_returns = returns_df.div(idio_vols + 1e-8, axis=1)  # Add small constant to prevent division by zero
    
    # Stage 3: Apply PCA on normalized correlation matrix
    try:
        # Filter out constant columns before correlation to avoid warnings
        non_constant_cols = []
        for col in normalized_returns.columns:
            if normalized_returns[col].std() > 1e-10:  # Non-constant threshold
                non_constant_cols.append(col)

        if len(non_constant_cols) < 2:
            logger.warning("Too few non-constant series for correlation matrix")
            return pd.DataFrame()

        # Calculate correlation only on non-constant series
        filtered_returns = normalized_returns[non_constant_cols]
        correlation_matrix = filtered_returns.corr()
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix.fillna(0).values)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Apply BBP threshold for significant factors
        n_samples, n_assets = returns_df.shape
        gamma = n_assets / n_samples if n_samples > 0 else 1
        bbp_threshold = 1 + np.sqrt(gamma)
        
        significant_factors = np.sum(eigenvals > bbp_threshold)
        significant_factors = max(1, min(significant_factors, n_factors))
        
        # Extract features
        features = {}
        alpha_ids = list(returns_df.columns)
        
        for i, alpha_id in enumerate(alpha_ids):
            # Factor loadings (exposure to principal components)
            factor_loadings = eigenvecs[i, :significant_factors]
            
            features[alpha_id] = {
                # Idiosyncratic characteristics
                'idio_vol': idio_vols.get(alpha_id, 0),
                'idio_vol_ratio': idio_vols.get(alpha_id, 0) / returns_df.std().get(alpha_id, 1),
                
                # Factor exposures
                **{f'factor_loading_{j+1}': factor_loadings[j] 
                   for j in range(len(factor_loadings))},
                
                # Eigenvalue exposures (strength of systematic risk)
                'eigenvalue_exposure': np.sum(factor_loadings**2 * eigenvals[:significant_factors]),
                
                # Factor concentration (how much variance is explained by top factor)
                'factor_concentration': factor_loadings[0]**2 * eigenvals[0] / np.sum(factor_loadings**2 * eigenvals[:significant_factors])
                if significant_factors > 0 and np.sum(factor_loadings**2 * eigenvals[:significant_factors]) > 0 else 0
            }
        
        features_df = pd.DataFrame(features).T
        
        # Add summary statistics
        features_df['n_significant_factors'] = significant_factors
        features_df['bbp_threshold'] = bbp_threshold
        features_df['gamma'] = gamma
        
        logger.info(f"Extracted spiked covariance features: {significant_factors} factors, "
                   f"BBP threshold: {bbp_threshold:.3f}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error in spiked covariance feature calculation: {e}")
        return pd.DataFrame()


def calculate_multi_scale_features(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]],
                                 windows: List[int] = [60, 120, 252]) -> pd.DataFrame:
    """
    Calculate multi-scale temporal features with exponential decay weighting.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to PnL data
        windows: List of window sizes for different time scales
        
    Returns:
        DataFrame with multi-scale features
    """
    features = []
    
    for alpha_id, data in pnl_data_dict.items():
        if data is None or 'df' not in data or data['df'] is None:
            continue
        df = data['df']
        if 'pnl' not in df.columns or len(df) < max(windows):
            continue
        
        pnl_series = df['pnl'].fillna(method='ffill').fillna(0)
        
        # Handle edge cases in PnL data
        if pnl_series.abs().max() < 1e-10:  # Essentially zero PnL
            continue
            
        # Calculate returns with robust handling
        returns = pnl_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) < max(windows) or returns.std() < 1e-10:  # Not enough data or constant
            continue
        
        alpha_features = {'alpha_id': alpha_id}
        
        # Note: Multi-scale volatility features removed as requested
        # These features were causing temporal clustering rather than risk-based clustering
        # Removed features: vol_60d, vol_120d, vol_252d and their stability metrics
        # Removed features: vol_ratio_short_long, autocorr_1d, autocorr_5d, autocorr_20d, momentum_strength
        
        features.append(alpha_features)
    
    # Return empty DataFrame since multi-scale features have been removed
    logger.info("Multi-scale features disabled - returning empty DataFrame")
    return pd.DataFrame()


def calculate_advanced_risk_metrics(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Calculate comprehensive risk metrics for clustering based on Paleologo's recommendations.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to PnL data
        
    Returns:
        DataFrame with advanced risk metrics
    """
    metrics = []
    
    for alpha_id, data in pnl_data_dict.items():
        if data is None or 'df' not in data or data['df'] is None:
            continue
        df = data['df']
        if 'pnl' not in df.columns or len(df) < 30:
            continue
        
        pnl_series = df['pnl'].fillna(method='ffill').fillna(0)
        
        # Handle edge cases in PnL data
        if pnl_series.abs().max() < 1e-10:  # Essentially zero PnL
            continue
            
        returns = pnl_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) < 20 or returns.std() < 1e-10:  # Not enough data or constant
            continue
        
        alpha_metrics = {'alpha_id': alpha_id}
        
        # Basic risk metrics
        alpha_metrics['mean_return'] = returns.mean()
        alpha_metrics['std_return'] = returns.std()
        alpha_metrics['skewness'] = returns.skew()
        alpha_metrics['excess_kurtosis'] = returns.kurtosis()
        
        # Downside risk metrics
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            alpha_metrics['downside_deviation'] = negative_returns.std()
            alpha_metrics['sortino_ratio'] = (
                alpha_metrics['mean_return'] / alpha_metrics['downside_deviation'] * np.sqrt(252)
                if alpha_metrics['downside_deviation'] > 0 else 0
            )
        
        # Value at Risk and Conditional VaR
        alpha_metrics['var_95'] = returns.quantile(0.05)
        alpha_metrics['cvar_95'] = returns[returns <= alpha_metrics['var_95']].mean()
        
        # Maximum drawdown analysis with robust handling
        cumulative_returns = (1 + returns).cumprod()
        
        # Handle edge cases in drawdown calculation
        if len(cumulative_returns) > 0 and cumulative_returns.max() > 0:
            running_max = cumulative_returns.expanding().max()
            # Avoid division by zero
            running_max = running_max.replace(0, 1e-10)
            drawdown = (cumulative_returns - running_max) / running_max
            alpha_metrics['max_drawdown'] = drawdown.min() if len(drawdown) > 0 else -0.01
        else:
            alpha_metrics['max_drawdown'] = -0.01  # Small default drawdown
        
        # Calmar ratio with bounds checking
        if (alpha_metrics['max_drawdown'] != 0 and np.isfinite(alpha_metrics['max_drawdown']) and 
            abs(alpha_metrics['max_drawdown']) > 1e-10):
            annual_return = alpha_metrics['mean_return'] * 252
            if np.isfinite(annual_return):
                alpha_metrics['calmar_ratio'] = annual_return / abs(alpha_metrics['max_drawdown'])
                # Cap extreme Calmar ratios
                alpha_metrics['calmar_ratio'] = np.clip(alpha_metrics['calmar_ratio'], -100, 100)
            else:
                alpha_metrics['calmar_ratio'] = 0
        else:
            alpha_metrics['calmar_ratio'] = 0
        
        # Hurst exponent (simple R/S analysis)
        alpha_metrics['hurst_exponent'] = calculate_hurst_exponent(returns.values)
        
        # Stability metrics
        if len(returns) >= 40:
            rolling_vol = returns.rolling(20).std()
            alpha_metrics['volatility_stability'] = 1 / (1 + rolling_vol.std()) if rolling_vol.std() > 0 else 1
        
        # Tail risk metrics
        alpha_metrics['tail_ratio'] = (
            abs(returns.quantile(0.95)) / abs(returns.quantile(0.05))
            if returns.quantile(0.05) != 0 else 1
        )
        
        # Performance consistency
        if len(returns) >= 60:
            monthly_returns = returns.groupby(returns.index.to_period('M')).sum()
            if len(monthly_returns) > 2:
                alpha_metrics['monthly_consistency'] = (monthly_returns > 0).mean()
        
        metrics.append(alpha_metrics)
    
    if not metrics:
        logger.warning("No risk metrics could be calculated")
        return pd.DataFrame()
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.set_index('alpha_id', inplace=True)
    
    # Fill NaN values with conservative defaults
    metrics_df = metrics_df.fillna({
        'mean_return': 0,
        'std_return': metrics_df['std_return'].median(),
        'skewness': 0,
        'excess_kurtosis': 0,
        'downside_deviation': metrics_df.get('downside_deviation', pd.Series()).median(),
        'sortino_ratio': 0,
        'var_95': metrics_df.get('var_95', pd.Series()).median(),
        'cvar_95': metrics_df.get('cvar_95', pd.Series()).median(),
        'max_drawdown': -0.1,  # Conservative default
        'calmar_ratio': 0,
        'hurst_exponent': 0.5,  # Random walk default
        'volatility_stability': 0.5,
        'tail_ratio': 1,
        'monthly_consistency': 0.5
    })
    
    # Add regime-based features that capture zero-variance behavior
    regime_features = extract_regime_features(pnl_data_dict, metrics_df.index)
    if not regime_features.empty:
        # Combine with risk metrics
        metrics_df = pd.concat([metrics_df, regime_features], axis=1)

    logger.info(f"Calculated advanced risk metrics for {len(metrics_df)} alphas")
    return metrics_df


def extract_regime_features(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]],
                          alpha_ids: pd.Index) -> pd.DataFrame:
    """
    Extract regime-based features that preserve zero-variance information.

    Args:
        pnl_data_dict: PnL data dictionary
        alpha_ids: Alpha IDs to process

    Returns:
        DataFrame with regime features
    """
    regime_data = []

    for alpha_id in alpha_ids:
        if alpha_id not in pnl_data_dict:
            continue

        data = pnl_data_dict[alpha_id]
        if data is None or 'df' not in data or data['df'] is None:
            continue

        df = data['df']
        if 'pnl' not in df.columns or len(df) < 30:
            continue

        pnl_series = df['pnl'].fillna(method='ffill').fillna(0)
        returns = pnl_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        if len(returns) < 20:
            continue

        regime_metrics = {'alpha_id': alpha_id}

        # Zero-loss regime features
        positive_periods = returns >= 0
        regime_metrics['zero_loss_ratio'] = positive_periods.mean()
        regime_metrics['max_consecutive_gains'] = max_consecutive_sequence(positive_periods)

        # Downside risk regime features
        negative_periods = returns < 0
        if negative_periods.any():
            regime_metrics['has_downside_periods'] = 1
            regime_metrics['downside_concentration'] = calculate_downside_concentration(returns)
        else:
            regime_metrics['has_downside_periods'] = 0
            regime_metrics['downside_concentration'] = 0

        # Volatility regime features
        regime_metrics['constant_return_periods'] = (returns == 0).mean()
        regime_metrics['low_volatility_ratio'] = (abs(returns) < returns.std() * 0.5).mean()

        regime_data.append(regime_metrics)

    if not regime_data:
        return pd.DataFrame()

    regime_df = pd.DataFrame(regime_data)
    regime_df.set_index('alpha_id', inplace=True)

    # Add prefix to distinguish from risk metrics
    regime_df.columns = [f'regime_{col}' for col in regime_df.columns]

    return regime_df


def max_consecutive_sequence(boolean_series: pd.Series) -> int:
    """Calculate maximum consecutive True values in a boolean series."""
    if boolean_series.empty:
        return 0

    # Convert to numpy for efficiency
    values = boolean_series.values
    max_count = 0
    current_count = 0

    for val in values:
        if val:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count


def calculate_downside_concentration(returns: pd.Series) -> float:
    """Calculate how concentrated the downside risk is."""
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0

    # Gini coefficient of negative returns (concentration measure)
    sorted_losses = np.sort(negative_returns.abs())
    n = len(sorted_losses)
    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * sorted_losses)) / (n * np.sum(sorted_losses)) - (n + 1) / n
    return gini


def calculate_hurst_exponent(returns: np.ndarray, max_lags: int = 50) -> float:
    """
    Calculate Hurst exponent using rescaled range analysis.
    
    Args:
        returns: Array of returns
        max_lags: Maximum number of lags to consider
        
    Returns:
        Hurst exponent (0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting)
    """
    if len(returns) < 20:
        return 0.5  # Default to random walk
    
    lags = range(2, min(max_lags, len(returns) // 2))
    rs_values = []
    
    for lag in lags:
        rs_list = []
        
        # Calculate R/S for non-overlapping segments
        for start in range(0, len(returns) - lag + 1, lag):
            segment = returns[start:start + lag]
            
            if len(segment) < lag:
                continue
            
            # Mean-adjusted cumulative sum
            mean_adj = segment - np.mean(segment)
            cum_devs = np.cumsum(mean_adj)
            
            # Range
            R = np.max(cum_devs) - np.min(cum_devs)
            
            # Standard deviation
            S = np.std(segment, ddof=1)
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(1)
    
    if len(rs_values) < 2:
        return 0.5
    
    # Linear regression on log-log plot
    log_lags = np.log(list(lags)[:len(rs_values)])
    log_rs = np.log(rs_values)
    
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
    if np.sum(valid_mask) < 2:
        return 0.5
    
    log_lags = log_lags[valid_mask]
    log_rs = log_rs[valid_mask]
    
    # Fit line: log(R/S) = H * log(lag) + constant
    coeffs = np.polyfit(log_lags, log_rs, 1)
    hurst = coeffs[0]
    
    # Bound between 0 and 1
    return np.clip(hurst, 0, 1)


def create_enhanced_feature_matrix(pnl_data_dict: Dict[str, Dict[str, pd.DataFrame]],
                                 include_spiked_cov: bool = True,
                                 include_multiscale: bool = True,
                                 include_risk_metrics: bool = True) -> pd.DataFrame:
    """
    Create a comprehensive feature matrix combining all advanced techniques.
    
    Args:
        pnl_data_dict: Dictionary mapping alpha_id to PnL data
        include_spiked_cov: Include spiked covariance model features
        include_multiscale: Include multi-scale temporal features
        include_risk_metrics: Include advanced risk metrics
        
    Returns:
        Combined feature matrix ready for clustering
    """
    feature_dfs = []
    
    if include_spiked_cov:
        logger.info("Calculating spiked covariance features...")
        spiked_features = calculate_spiked_covariance_features(pnl_data_dict)
        if not spiked_features.empty:
            spiked_features.columns = [f'spiked_{col}' for col in spiked_features.columns]
            feature_dfs.append(spiked_features)
    
    if include_multiscale:
        logger.info("Calculating multi-scale features...")
        multiscale_features = calculate_multi_scale_features(pnl_data_dict)
        if not multiscale_features.empty:
            multiscale_features.columns = [f'multiscale_{col}' for col in multiscale_features.columns]
            feature_dfs.append(multiscale_features)
    
    if include_risk_metrics:
        logger.info("Calculating advanced risk metrics...")
        risk_features = calculate_advanced_risk_metrics(pnl_data_dict)
        if not risk_features.empty:
            risk_features.columns = [f'risk_{col}' for col in risk_features.columns]
            feature_dfs.append(risk_features)
    
    if not feature_dfs:
        logger.warning("No features could be calculated")
        return pd.DataFrame()
    
    # Combine all feature sets using inner join to avoid NaN creation
    logger.info(f"Combining {len(feature_dfs)} feature sets...")
    for i, df in enumerate(feature_dfs):
        logger.info(f"  Feature set {i+1}: {df.shape[0]} alphas, {df.shape[1]} features")
    
    combined_features = feature_dfs[0]
    for df in feature_dfs[1:]:
        combined_features = combined_features.join(df, how='inner')  # Use inner join
    
    logger.info(f"After combining: {combined_features.shape[0]} alphas, {combined_features.shape[1]} features")
    
    # Check for NaN values - they shouldn't exist if features are calculated properly
    nan_counts = combined_features.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found {nan_counts.sum()} NaN values in feature matrix:")
        for feature, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"  {feature}: {count} NaN values")
        logger.warning("Filling NaN with median - this may indicate feature calculation issues")
        combined_features = combined_features.fillna(combined_features.median())
    else:
        logger.info("No NaN values found in feature matrix - good!")
    
    # Remove constant features (no information)
    constant_cols = combined_features.columns[combined_features.std() == 0]
    if len(constant_cols) > 0:
        combined_features = combined_features.drop(columns=constant_cols)
        logger.info(f"Removed {len(constant_cols)} constant features")
    
    logger.info(f"Created enhanced feature matrix: {combined_features.shape[0]} alphas × {combined_features.shape[1]} features")
    
    return combined_features