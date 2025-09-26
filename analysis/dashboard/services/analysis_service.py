"""
Analysis Service

Business logic for alpha analysis, operators, and datafields processing.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import text

from .data_service import create_analysis_operations
from ..utils import cached, safe_divide, sort_dict_by_values, group_small_categories
from ..config import AVAILABLE_REGIONS

class AnalysisService:
    """Service for handling analysis operations and data processing."""

    def __init__(self):
        """Initialize analysis service."""
        self.analysis_ops = None

    def _get_analysis_ops(self):
        """Get analysis operations instance."""
        if self.analysis_ops is None:
            self.analysis_ops = create_analysis_operations()
        return self.analysis_ops

    def get_analysis_summary(self, region: Optional[str] = None,
                           universe: Optional[str] = None,
                           delay: Optional[int] = None,
                           date_from: Optional[str] = None,
                           date_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive analysis summary with optional filtering.

        Args:
            region: Region filter
            universe: Universe filter
            delay: Delay filter
            date_from: Start date filter
            date_to: End date filter

        Returns:
            Analysis summary dictionary
        """
        try:
            analysis_ops = self._get_analysis_ops()
            return analysis_ops.get_analysis_summary(region, universe, delay, date_from, date_to)
        except Exception as e:
            print(f"Error getting analysis summary: {e}")
            return {
                'metadata': {'error': str(e)},
                'operators': {},
                'datafields': {}
            }

    def get_operator_usage_analysis(self, operators_file: str) -> Dict[str, Any]:
        """
        Create comprehensive usage analysis showing all platform operators.

        Args:
            operators_file: Path to operators file

        Returns:
            Usage analysis data
        """
        try:
            analysis_ops = self._get_analysis_ops()
            analysis_data = analysis_ops.get_analysis_summary()
            operators_data = analysis_data.get('operators', {})
            used_operators = dict(operators_data.get('top_operators', []))

            # Load all platform operators
            from .data_service import DYNAMIC_OPERATORS_LIST, load_operators_data

            if DYNAMIC_OPERATORS_LIST:
                all_operators = DYNAMIC_OPERATORS_LIST
                print(f"✅ Using cached operators list: {len(all_operators)} operators")
            else:
                all_operators = load_operators_data(operators_file)

            if not all_operators:
                return {'error': 'Failed to load operators data'}

            # Categorize operators
            frequently_used = [(op, count) for op, count in used_operators.items() if count >= 10]
            rarely_used = [(op, count) for op, count in used_operators.items() if 1 <= count < 10]
            never_used = [(op, 0) for op in all_operators if op not in used_operators]

            frequently_used.sort(key=lambda x: x[1], reverse=True)
            rarely_used.sort(key=lambda x: x[1], reverse=True)
            never_used.sort()

            return {
                'total_operators': len(all_operators),
                'frequently_used': frequently_used,
                'rarely_used': rarely_used,
                'never_used': never_used,
                'usage_summary': {
                    'frequent_count': len(frequently_used),
                    'rare_count': len(rarely_used),
                    'never_count': len(never_used)
                }
            }

        except Exception as e:
            print(f"Error in operator usage analysis: {e}")
            return {'error': str(e)}

    @cached(ttl=300)
    def calculate_dataset_statistics(self, analysis_data: Dict[str, Any],
                                   region: Optional[str] = None,
                                   universe: Optional[str] = None,
                                   delay: Optional[int] = None,
                                   date_from: Optional[str] = None,
                                   date_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive dataset statistics including total datafield counts
        and usage patterns for treemap visualization.

        Args:
            analysis_data: The analysis data containing datafields information
            region: Optional region filter
            universe: Optional universe filter
            delay: Optional delay filter
            date_from: Optional start date filter
            date_to: Optional end date filter

        Returns:
            Dict containing dataset usage, availability, and statistics
        """
        try:
            analysis_ops = self._get_analysis_ops()

            # If filters are provided, get fresh filtered data instead of using cached analysis_data
            if any([region, universe, delay, date_from, date_to]):
                filtered_results = analysis_ops.get_analysis_summary(region, universe, delay, date_from, date_to)
                datafields_data = filtered_results.get('datafields', {})
            else:
                # Use existing cached data
                datafields_data = analysis_data.get('datafields', {})

            unique_usage = datafields_data.get('unique_usage', {})

            # Track dataset usage (from used alphas) and total datafields per dataset
            dataset_alpha_usage = {}  # {dataset_id: set_of_alpha_ids}
            dataset_total_datafields = {}  # {dataset_id: total_datafield_count}
            dataset_used_datafields = {}  # {dataset_id: count_of_used_datafields}

            # Calculate used datafields per dataset
            for df, alphas in unique_usage.items():
                if df in analysis_ops.parser.datafields:
                    dataset_id = analysis_ops.parser.datafields[df]['dataset_id']
                    if dataset_id:
                        if dataset_id not in dataset_alpha_usage:
                            dataset_alpha_usage[dataset_id] = set()
                        dataset_alpha_usage[dataset_id].update(alphas)

                        dataset_used_datafields[dataset_id] = dataset_used_datafields.get(dataset_id, 0) + 1
                else:
                    # Fallback: extract dataset from datafield name
                    if '.' in df:
                        dataset_id = df.split('.')[0]
                        if dataset_id not in dataset_alpha_usage:
                            dataset_alpha_usage[dataset_id] = set()
                        dataset_alpha_usage[dataset_id].update(alphas)
                        dataset_used_datafields[dataset_id] = dataset_used_datafields.get(dataset_id, 0) + 1

            # Calculate total datafields per dataset, respecting region filter
            if region:
                # Get region-specific datafields directly from database
                try:
                    db_engine = analysis_ops._get_db_engine()
                    with db_engine.connect() as connection:
                        query = text("""
                            SELECT datafield_id, dataset_id
                            FROM datafields
                            WHERE region = :region
                            AND dataset_id IS NOT NULL
                            AND dataset_id != ''
                            AND datafield_id IS NOT NULL
                            AND datafield_id != ''
                        """)
                        result = connection.execute(query, {"region": region})

                        for row in result:
                            dataset_id = row.dataset_id
                            if dataset_id:
                                dataset_total_datafields[dataset_id] = dataset_total_datafields.get(dataset_id, 0) + 1


                except Exception as e:
                    print(f"Error querying region-specific datafields: {e}")
                    # Fallback: try to use all datafields from parser (not region-specific)
                    for df, info in analysis_ops.parser.datafields.items():
                        dataset_id = info.get('dataset_id')
                        if dataset_id:
                            dataset_total_datafields[dataset_id] = dataset_total_datafields.get(dataset_id, 0) + 1
                    print(f"Fallback: Using all datafields from parser, found {len(dataset_total_datafields)} datasets")
            else:
                # No region filter - get all datafields from parser
                print("No region filter: Using all datafields from parser")
                for df, info in analysis_ops.parser.datafields.items():
                    dataset_id = info.get('dataset_id')
                    if dataset_id:
                        dataset_total_datafields[dataset_id] = dataset_total_datafields.get(dataset_id, 0) + 1

            # Create comprehensive dataset statistics
            dataset_stats = []

            # When filtering is applied, we want to show all datasets with available datafields in that region
            # When no filtering, we show all datasets that either have datafields OR have been used by alphas
            if any([region, universe, delay, date_from, date_to]):
                # Filter mode: Show all datasets with datafields available in the filtered region
                all_datasets = set(dataset_total_datafields.keys())
            else:
                # No filter: Show all datasets (both used and available)
                all_datasets = set(dataset_total_datafields.keys()) | set(dataset_alpha_usage.keys())

            for dataset_id in all_datasets:
                total_datafields = dataset_total_datafields.get(dataset_id, 0)
                used_datafields = dataset_used_datafields.get(dataset_id, 0)
                alpha_usage_count = len(dataset_alpha_usage.get(dataset_id, set()))
                usage_percentage = safe_divide(used_datafields, total_datafields) * 100

                # Only include datasets that have datafields available (total_datafields > 0)
                if total_datafields > 0:
                    dataset_stats.append({
                        'dataset_id': dataset_id,
                        'total_datafields': total_datafields,
                        'used_datafields': used_datafields,
                        'alpha_usage_count': alpha_usage_count,
                        'usage_percentage': usage_percentage,
                        'is_used': alpha_usage_count > 0
                    })

            # Sort by total datafields (size)
            dataset_stats.sort(key=lambda x: x['total_datafields'], reverse=True)

            return {
                'dataset_stats': dataset_stats,
                'total_datasets': len(dataset_stats),
                'used_datasets': len([d for d in dataset_stats if d['is_used']]),
                'unused_datasets': len([d for d in dataset_stats if not d['is_used']]),
                'dataset_alpha_usage': dataset_alpha_usage
            }

        except Exception as e:
            print(f"Error calculating dataset statistics: {e}")
            return {
                'dataset_stats': [],
                'total_datasets': 0,
                'used_datasets': 0,
                'unused_datasets': 0,
                'dataset_alpha_usage': {},
                'error': str(e)
            }

    def get_datafield_recommendations(self, selected_region: Optional[str] = None,
                                    selected_data_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get datafield recommendations for expanding alpha usage.

        Args:
            selected_region: Target region filter
            selected_data_type: Data type filter

        Returns:
            Recommendations data
        """
        try:
            analysis_ops = self._get_analysis_ops()
            return analysis_ops.get_datafield_recommendations(selected_region, selected_data_type)
        except Exception as e:
            print(f"Error getting datafield recommendations: {e}")
            return {'error': str(e)}

    def get_available_operators_for_region(self, region: str) -> List[str]:
        """
        Get available operators for a specific region.

        Args:
            region: Region name

        Returns:
            List of available operators
        """
        try:
            analysis_ops = self._get_analysis_ops()
            return analysis_ops.get_available_operators_for_region(region)
        except Exception as e:
            print(f"Error getting available operators for region {region}: {e}")
            return []

    def get_available_datafields_for_region(self, region: str) -> List[str]:
        """
        Get available datafields for a specific region.

        Args:
            region: Region name

        Returns:
            List of available datafields
        """
        try:
            analysis_ops = self._get_analysis_ops()
            return analysis_ops.get_available_datafields_for_region(region)
        except Exception as e:
            print(f"Error getting available datafields for region {region}: {e}")
            return []

    def get_alphas_containing_operators(self, operators: List[str], region: str) -> List[str]:
        """
        Get alphas containing specified operators in a region.

        Args:
            operators: List of operator names
            region: Region name

        Returns:
            List of alpha IDs
        """
        try:
            analysis_ops = self._get_analysis_ops()
            return analysis_ops.get_alphas_containing_operators(operators, region)
        except Exception as e:
            print(f"Error getting alphas containing operators: {e}")
            return []

    def get_alphas_containing_datafields(self, datafields: List[str], region: str) -> List[str]:
        """
        Get alphas containing specified datafields in a region.

        Args:
            datafields: List of datafield names
            region: Region name

        Returns:
            List of alpha IDs
        """
        try:
            analysis_ops = self._get_analysis_ops()
            return analysis_ops.get_alphas_containing_datafields(datafields, region)
        except Exception as e:
            print(f"Error getting alphas containing datafields: {e}")
            return []

    def validate_analysis_filters(self, filters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate analysis filter parameters.

        Args:
            filters: Filter parameters to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate region if provided
        if 'region' in filters and filters['region']:
            if filters['region'] not in AVAILABLE_REGIONS:
                errors.append(f"Invalid region: {filters['region']}")

        # Validate delay if provided
        if 'delay' in filters and filters['delay'] is not None:
            delay = filters['delay']
            if not isinstance(delay, int) or delay < 0 or delay > 30:
                errors.append("Invalid delay: must be between 0 and 30")

        # Validate universe if provided
        if 'universe' in filters and filters['universe']:
            # Add universe validation logic if needed
            pass

        return len(errors) == 0, errors

# Global service instance
_analysis_service_instance = None

def get_analysis_service() -> AnalysisService:
    """Get singleton analysis service instance."""
    global _analysis_service_instance
    if _analysis_service_instance is None:
        _analysis_service_instance = AnalysisService()
    return _analysis_service_instance

def reset_analysis_service():
    """Reset analysis service instance (for testing)."""
    global _analysis_service_instance
    _analysis_service_instance = None