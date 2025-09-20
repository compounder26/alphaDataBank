"""
Unified correlation calculation module for AlphaDataBank.

This module provides a single entry point for all correlation calculations:
- Submitted vs submitted alphas (regular correlations)
- Unsubmitted vs submitted alphas (finding max correlations)
- Cross-correlation analysis with CSV export
"""

from .correlation_engine import CorrelationEngine

__all__ = ['CorrelationEngine']