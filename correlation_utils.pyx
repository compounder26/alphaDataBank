# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport isnan, isinf
from scipy import stats

# Initialize NumPy C-API
np.import_array()

def calculate_alpha_correlation_fast(
    double[::1] alpha_pnl,
    double[::1] other_pnl,
    double initial_value=10000000.0
):
    """
    Fast Cython implementation of alpha correlation calculation.
    Uses the same formula but with compiled code for speed.

    Args:
        alpha_pnl: Numpy array of cumulative PNL for alpha
        other_pnl: Numpy array of cumulative PNL for other alpha
        initial_value: Initial portfolio value (default 10M)

    Returns:
        Correlation coefficient or None if not enough valid data points
    """
    cdef int i, n = alpha_pnl.shape[0]

    # Ensure we have enough data points
    if n < 20:
        return None

    # Arrays for calculations
    cdef double[::1] alpha_daily = np.zeros(n, dtype=np.float64)
    cdef double[::1] other_daily = np.zeros(n, dtype=np.float64)
    cdef double[::1] alpha_port_val = np.zeros(n, dtype=np.float64)
    cdef double[::1] other_port_val = np.zeros(n, dtype=np.float64)
    cdef double[::1] alpha_returns = np.zeros(n, dtype=np.float64)
    cdef double[::1] other_returns = np.zeros(n, dtype=np.float64)

    # Calculate daily PNL - EXACTLY as in original function
    alpha_daily[0] = alpha_pnl[0]  # First day's PNL is equal to cumulative
    other_daily[0] = other_pnl[0]

    for i in range(1, n):
        alpha_daily[i] = alpha_pnl[i] - alpha_pnl[i-1]  # Diff of cumulative PNL
        other_daily[i] = other_pnl[i] - other_pnl[i-1]

    # Calculate portfolio values - initial value + cumulative daily PNL
    alpha_port_val[0] = initial_value + alpha_daily[0]
    other_port_val[0] = initial_value + other_daily[0]

    for i in range(1, n):
        alpha_port_val[i] = alpha_port_val[i-1] + alpha_daily[i]
        other_port_val[i] = other_port_val[i-1] + other_daily[i]

    # Calculate returns - daily PNL divided by PREVIOUS day's portfolio value
    # Set first value to NaN to match Python's shift(1) behavior
    alpha_returns[0] = np.nan  # First return is invalid as there's no previous value
    other_returns[0] = np.nan  # This matches the behavior of shift(1) in Python

    for i in range(1, n):
        if alpha_port_val[i-1] != 0:
            alpha_returns[i] = alpha_daily[i] / alpha_port_val[i-1]
        else:
            alpha_returns[i] = np.nan

        if other_port_val[i-1] != 0:
            other_returns[i] = other_daily[i] / other_port_val[i-1]
        else:
            other_returns[i] = np.nan

    # Convert memoryviews back to numpy arrays for filtering
    alpha_returns_arr = np.asarray(alpha_returns)
    other_returns_arr = np.asarray(other_returns)

    # Filter valid values using NumPy arrays for better vectorization
    # Create a boolean mask of valid values (not NaN, not inf)
    valid_mask = np.logical_and(
        np.logical_and(~np.isnan(alpha_returns_arr), ~np.isnan(other_returns_arr)),
        np.logical_and(~np.isinf(alpha_returns_arr), ~np.isinf(other_returns_arr))
    )

    alpha_returns_clean = alpha_returns_arr[valid_mask]
    other_returns_clean = other_returns_arr[valid_mask]

    # Check if we have enough valid data points
    if alpha_returns_clean.shape[0] < 20:  # Need at least 20 points for reliable correlation
        return None

    # Calculate correlation using scipy.stats.pearsonr
    corr, _ = stats.pearsonr(alpha_returns_clean, other_returns_clean)

    if isnan(corr):
        return None

    return corr