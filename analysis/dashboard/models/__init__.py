"""
Dashboard Models

Data models and state management for the visualization dashboard.
Provides both modern typed interfaces and backward compatibility with legacy stores.
"""

# Use simple Python classes (no external dependencies)
from .simple_state_models import (
    AnalysisFilters,
    CoordinateData,
    AnalysisData,
    DashboardState,
    ViewState,
    HighlightState,
    ClusteringData
)

__all__ = [
    # State Models
    'AnalysisFilters',
    'CoordinateData',
    'ClusteringData',
    'AnalysisData',
    'ViewState',
    'HighlightState',
    'DashboardState',
]