"""
Recommendation Service

Business logic for datafield recommendations and cross-analysis.
"""

from typing import Dict, List, Any, Optional
from sqlalchemy import text

from .data_service import create_analysis_operations
from ..utils import cached

class RecommendationService:
    """Service for handling datafield recommendations and cross-analysis."""

    def __init__(self):
        """Initialize recommendation service."""
        self.analysis_ops = None

    def _get_analysis_ops(self):
        """Get analysis operations instance."""
        if self.analysis_ops is None:
            self.analysis_ops = create_analysis_operations()
        return self.analysis_ops

    @cached(ttl=300)  # Cache for 5 minutes
    def get_datafield_recommendations(self, selected_region: Optional[str] = None,
                                    selected_data_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get datafield recommendations for expanding alpha usage.

        Args:
            selected_region: Target region filter
            selected_data_type: Data type filter ('MATRIX', 'VECTOR', 'GROUP')

        Returns:
            Datafield recommendations data
        """
        try:
            analysis_ops = self._get_analysis_ops()
            return analysis_ops.get_datafield_recommendations(selected_region, selected_data_type)
        except Exception as e:
            print(f"Error getting datafield recommendations: {e}")
            return {'error': str(e), 'recommendations': []}

    def get_alphas_using_datafield_in_region(self, datafield_id: str, region: str,
                                           limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get alphas using a specific datafield in a region.

        Args:
            datafield_id: Datafield identifier
            region: Region name
            limit: Maximum number of alphas to return

        Returns:
            List of alpha information dictionaries
        """
        try:
            analysis_ops = self._get_analysis_ops()
            db_engine = analysis_ops._get_db_engine()

            with db_engine.connect() as connection:
                query = text("""
                    SELECT DISTINCT
                        a.alpha_id,
                        a.code,
                        a.is_sharpe,
                        a.is_fitness,
                        a.is_returns,
                        a.universe,
                        a.delay
                    FROM alphas a
                    JOIN regions r ON a.region_id = r.region_id
                    JOIN alpha_analysis_cache ac ON a.alpha_id = ac.alpha_id
                    WHERE r.region_name = :region
                    AND ac.datafields_unique::jsonb ? :datafield_id
                    ORDER BY a.is_sharpe DESC NULLS LAST
                    LIMIT :limit
                """)

                result = connection.execute(query, {
                    'region': region,
                    'datafield_id': datafield_id,
                    'limit': limit
                })

                alphas = []
                for row in result:
                    alphas.append({
                        'alpha_id': row.alpha_id,
                        'code': row.code or '',
                        'is_sharpe': row.is_sharpe,
                        'is_fitness': row.is_fitness,
                        'is_returns': row.is_returns,
                        'universe': row.universe,
                        'delay': row.delay
                    })

                return alphas

        except Exception as e:
            print(f"Error getting alphas using datafield {datafield_id} in {region}: {e}")
            return []

    def get_datafield_details_for_region(self, datafield_id: str, region: str) -> Dict[str, Any]:
        """
        Get detailed information about a datafield in a specific region.

        Args:
            datafield_id: Datafield identifier
            region: Region name

        Returns:
            Datafield details dictionary
        """
        try:
            analysis_ops = self._get_analysis_ops()

            # Get datafield metadata from parser
            datafield_info = analysis_ops.parser.datafields.get(datafield_id, {})

            # Get usage statistics
            db_engine = analysis_ops._get_db_engine()
            with db_engine.connect() as connection:
                query = text("""
                    SELECT COUNT(DISTINCT a.alpha_id) as usage_count
                    FROM alphas a
                    JOIN regions r ON a.region_id = r.region_id
                    JOIN alpha_analysis_cache ac ON a.alpha_id = ac.alpha_id
                    WHERE r.region_name = :region
                    AND ac.datafields_unique::jsonb ? :datafield_id
                """)

                result = connection.execute(query, {
                    'region': region,
                    'datafield_id': datafield_id
                })

                usage_count = result.scalar() or 0

            return {
                'datafield_id': datafield_id,
                'region': region,
                'description': datafield_info.get('data_description', 'No description available'),
                'dataset_id': datafield_info.get('dataset_id', 'Unknown'),
                'data_category': datafield_info.get('data_category', 'Unknown'),
                'data_type': datafield_info.get('data_type', 'Unknown'),
                'delay': datafield_info.get('delay', 'Unknown'),
                'usage_count': usage_count,
                'available': usage_count > 0
            }

        except Exception as e:
            print(f"Error getting datafield details for {datafield_id} in {region}: {e}")
            return {
                'datafield_id': datafield_id,
                'region': region,
                'error': str(e)
            }

    def create_recommendations_display_data(self, recommendations: List[Dict[str, Any]],
                                          full_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create display data for recommendations table.

        Args:
            recommendations: List of recommendation dictionaries
            full_data: Complete recommendations data

        Returns:
            Formatted display data
        """
        total_analyzed = full_data.get('total_datafields_analyzed', 0)

        # Calculate summary statistics
        summary_stats = {
            'total_analyzed': total_analyzed,
            'expansion_opportunities': len(recommendations),
            'potential_new_alphas': sum(len(rec['recommended_regions']) for rec in recommendations)
        }

        # Format recommendations for table display
        formatted_recommendations = []
        for idx, rec in enumerate(recommendations):
            # Format used regions with counts
            used_badges_data = []
            for region in rec['used_in_regions']:
                usage_count = rec['usage_details'].get(region, 0)
                used_badges_data.append({
                    'region': region,
                    'usage_count': usage_count,
                    'text': f"{region} ({usage_count} alphas)"
                })

            # Format recommended regions with availability details
            recommended_badges_data = []
            availability_details = rec.get('availability_details', {})
            for region in rec['recommended_regions']:
                matching_ids = availability_details.get(region, [])
                badge_data = {
                    'region': region,
                    'matching_count': len(matching_ids),
                    'matching_ids': matching_ids
                }

                if len(matching_ids) > 1:
                    badge_data['text'] = f"{region} ({len(matching_ids)} IDs)"
                    badge_data['clickable'] = True
                    badge_data['title'] = f"Click to view {len(matching_ids)} available datafields in {region}"
                elif len(matching_ids) == 1:
                    badge_data['text'] = region
                    badge_data['clickable'] = True
                    badge_data['title'] = f"Click to view datafield details for {region}"
                else:
                    badge_data['text'] = region
                    badge_data['clickable'] = False

                recommended_badges_data.append(badge_data)

            # Check if this is a description-based match
            matching_datafields = rec.get('matching_datafields', {})
            if len(matching_datafields) > 1:
                display_name = {
                    'main': rec['datafield_id'],
                    'additional': f" (+{len(matching_datafields)-1} similar)"
                }
            else:
                display_name = {'main': rec['datafield_id'], 'additional': ''}

            formatted_rec = {
                'index': idx,
                'datafield_id': rec['datafield_id'],
                'display_name': display_name,
                'description': rec['description'],
                'data_type': rec.get('data_type', 'Unknown'),
                'alpha_count': rec['alpha_count'],
                'used_badges': used_badges_data,
                'recommended_badges': recommended_badges_data,
                'usage_details_text': ", ".join([
                    f"{region}: {count}"
                    for region, count in rec['usage_details'].items()
                ])
            }

            formatted_recommendations.append(formatted_rec)

        return {
            'summary_stats': summary_stats,
            'recommendations': formatted_recommendations
        }

    def get_neutralization_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get neutralization analysis data.

        Args:
            analysis_data: Analysis data containing metadata

        Returns:
            Neutralization analysis data
        """
        metadata = analysis_data.get('metadata', {})
        neutralizations = metadata.get('neutralizations', {})
        total_alphas = metadata.get('total_alphas', 0)

        if not neutralizations:
            return {
                'available': False,
                'message': 'No neutralization data available'
            }

        # Calculate statistics
        most_common = max(neutralizations, key=neutralizations.get) if neutralizations else 'N/A'
        least_common = min(neutralizations, key=neutralizations.get) if neutralizations else 'N/A'

        # Create sorted breakdown
        breakdown = sorted(neutralizations.items(), key=lambda x: x[1], reverse=True)

        return {
            'available': True,
            'neutralizations': neutralizations,
            'total_alphas': total_alphas,
            'statistics': {
                'total_types': len(neutralizations),
                'most_common': most_common,
                'least_common': least_common
            },
            'breakdown': breakdown
        }

    def get_alphas_by_neutralization(self, neutralization: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alphas using a specific neutralization type.

        Args:
            neutralization: Neutralization type
            limit: Maximum number of alphas to return

        Returns:
            List of alpha information
        """
        try:
            analysis_ops = self._get_analysis_ops()
            db_engine = analysis_ops._get_db_engine()

            with db_engine.connect() as connection:
                query = text("""
                    SELECT alpha_id, code, universe, delay, is_sharpe, is_fitness, is_returns,
                           neutralization, decay, r.region_name
                    FROM alphas a
                    JOIN regions r ON a.region_id = r.region_id
                    WHERE a.neutralization = :neutralization
                    ORDER BY a.alpha_id
                    LIMIT :limit
                """)

                result = connection.execute(query, {
                    'neutralization': neutralization,
                    'limit': limit
                })

                alphas = []
                for row in result:
                    alphas.append({
                        'alpha_id': row.alpha_id,
                        'code': row.code or '',
                        'universe': row.universe or 'N/A',
                        'delay': row.delay if row.delay is not None else 'N/A',
                        'is_sharpe': row.is_sharpe if row.is_sharpe is not None else 0,
                        'is_fitness': row.is_fitness if row.is_fitness is not None else 0,
                        'is_returns': row.is_returns if row.is_returns is not None else 0,
                        'neutralization': row.neutralization or 'N/A',
                        'decay': row.decay or 'N/A',
                        'region_name': row.region_name or 'N/A'
                    })

                return alphas

        except Exception as e:
            print(f"Error fetching alphas for neutralization {neutralization}: {e}")
            return []

    def get_alphas_by_datafield_and_region(self, datafield_id: str, region: str) -> List[str]:
        """
        Get list of alpha IDs using a specific datafield in a specific region.

        Args:
            datafield_id: Datafield identifier
            region: Region name

        Returns:
            List of alpha IDs
        """
        try:
            analysis_ops = self._get_analysis_ops()
            db_engine = analysis_ops._get_db_engine()

            with db_engine.connect() as connection:
                query = text("""
                    SELECT DISTINCT a.alpha_id
                    FROM alphas a
                    JOIN regions r ON a.region_id = r.region_id
                    JOIN alpha_analysis_cache ac ON a.alpha_id = ac.alpha_id
                    WHERE r.region_name = :region
                    AND ac.datafields_unique::jsonb ? :datafield_id
                    ORDER BY a.alpha_id
                """)

                result = connection.execute(query, {
                    'region': region,
                    'datafield_id': datafield_id
                })

                return [row.alpha_id for row in result]

        except Exception as e:
            print(f"Error getting alphas for datafield {datafield_id} in region {region}: {e}")
            return []

    def get_matching_datafields_in_region(self, source_datafield_id: str, target_region: str) -> List[Dict[str, Any]]:
        """
        Get matching datafields available in a target region based on a source datafield.

        Args:
            source_datafield_id: Source datafield identifier
            target_region: Target region name

        Returns:
            List of matching datafield information
        """
        try:
            analysis_ops = self._get_analysis_ops()

            # Get source datafield info
            source_info = analysis_ops.parser.datafields.get(source_datafield_id, {})
            if not source_info:
                # Try to get from database instead
                db_engine = analysis_ops._get_db_engine()
                with db_engine.connect() as connection:
                    query = text("""
                        SELECT DISTINCT data_description
                        FROM datafields
                        WHERE datafield_id = :datafield_id
                        LIMIT 1
                    """)
                    result = connection.execute(query, {'datafield_id': source_datafield_id})
                    row = result.fetchone()
                    if row:
                        source_description = row.data_description or ''
                    else:
                        source_description = ''
            else:
                source_description = source_info.get('data_description', '')

            # Find matching datafields available in target region
            matching_datafields = []

            # Query for datafields available in target region (not just used ones)
            db_engine = analysis_ops._get_db_engine()
            with db_engine.connect() as connection:
                # Get datafields available in target region from datafields table
                query = text("""
                    SELECT DISTINCT datafield_id, data_description, dataset_id,
                           data_category, data_type, delay
                    FROM datafields
                    WHERE region = :region
                    AND datafield_id IS NOT NULL AND datafield_id != ''
                """)

                result = connection.execute(query, {'region': target_region})
                region_datafields_info = {
                    row.datafield_id: {
                        'data_description': row.data_description,
                        'dataset_id': row.dataset_id,
                        'data_category': row.data_category,
                        'data_type': row.data_type,
                        'delay': row.delay
                    }
                    for row in result
                }
                region_datafields = set(region_datafields_info.keys())

            # Check for matches based on description similarity
            for df_id in region_datafields:
                df_info = region_datafields_info[df_id]
                df_description = df_info.get('data_description', '')

                # Simple matching: same description or contains source datafield base name
                if source_description and df_description:
                    # Check if descriptions match (case-insensitive)
                    if df_description.lower() == source_description.lower():
                        matching_datafields.append({
                            'id': df_id,
                            'description': df_description,
                            'dataset': df_info.get('dataset_id', 'Unknown'),
                            'category': df_info.get('data_category', 'Unknown'),
                            'data_type': df_info.get('data_type', 'Unknown'),
                            'delay': df_info.get('delay', 0)
                        })
                    # Also check if the base name matches (e.g., 'equity' part of 'equity_usa')
                    elif source_datafield_id.split('_')[0].lower() in df_id.lower():
                        matching_datafields.append({
                            'id': df_id,
                            'description': df_description,
                            'dataset': df_info.get('dataset_id', 'Unknown'),
                            'category': df_info.get('data_category', 'Unknown'),
                            'data_type': df_info.get('data_type', 'Unknown'),
                            'delay': df_info.get('delay', 0)
                        })

            # If no description-based matches, check if the exact datafield ID exists
            if not matching_datafields and source_datafield_id in region_datafields:
                df_info = region_datafields_info[source_datafield_id]
                matching_datafields.append({
                    'id': source_datafield_id,
                    'description': df_info.get('data_description', 'Same datafield available'),
                    'dataset': df_info.get('dataset_id', 'Unknown'),
                    'category': df_info.get('data_category', 'Unknown'),
                    'data_type': df_info.get('data_type', 'Unknown'),
                    'delay': df_info.get('delay', 0)
                })

            return matching_datafields[:10]  # Limit to 10 results

        except Exception as e:
            print(f"Error getting matching datafields for {source_datafield_id} in {target_region}: {e}")
            return []

# Global service instance
_recommendation_service_instance = None

def get_recommendation_service() -> RecommendationService:
    """Get singleton recommendation service instance."""
    global _recommendation_service_instance
    if _recommendation_service_instance is None:
        _recommendation_service_instance = RecommendationService()
    return _recommendation_service_instance

def reset_recommendation_service():
    """Reset recommendation service instance (for testing)."""
    global _recommendation_service_instance
    _recommendation_service_instance = None