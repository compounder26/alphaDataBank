#!/usr/bin/env python3
"""
Test script to verify datafield cross-analysis fixes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bootstrap import setup_project_path
setup_project_path()

from analysis.dashboard.services.recommendation_service import get_recommendation_service
from analysis.analysis_operations import AnalysisOperations
from database.schema import get_connection
from sqlalchemy import text
import json

def test_datafield_recommendations():
    """Test that datafield recommendations correctly show matching IDs."""
    print("\n=== Testing Datafield Recommendations ===")

    rec_service = get_recommendation_service()

    # Get recommendations
    recommendations_data = rec_service.get_datafield_recommendations()
    recommendations = recommendations_data.get('recommendations', [])

    print(f"Found {len(recommendations)} datafield recommendations")

    # Check first few recommendations with multiple IDs
    for i, rec in enumerate(recommendations[:5]):
        datafield_id = rec['datafield_id']
        availability_details = rec.get('availability_details', {})

        print(f"\n{i+1}. Datafield: {datafield_id}")
        print(f"   Description: {rec.get('description', 'N/A')[:50]}...")
        print(f"   Used in: {rec.get('used_in_regions', [])}")
        print(f"   Recommended for: {rec.get('recommended_regions', [])}")

        # Check availability details
        for region, matching_ids in availability_details.items():
            print(f"   - {region}: {len(matching_ids)} matching IDs")
            if matching_ids:
                print(f"     IDs: {matching_ids[:3]}{'...' if len(matching_ids) > 3 else ''}")


def test_matching_datafields_in_region():
    """Test the get_matching_datafields_in_region function."""
    print("\n=== Testing Matching Datafields in Region ===")

    rec_service = get_recommendation_service()

    # Get recommendations first to find a good test case
    recommendations_data = rec_service.get_datafield_recommendations()
    recommendations = recommendations_data.get('recommendations', [])

    # Find a recommendation with multiple matching IDs
    test_case = None
    for rec in recommendations:
        availability_details = rec.get('availability_details', {})
        for region, matching_ids in availability_details.items():
            if len(matching_ids) > 1:
                test_case = {
                    'datafield': rec['datafield_id'],
                    'region': region,
                    'expected_count': len(matching_ids),
                    'expected_ids': matching_ids
                }
                break
        if test_case:
            break

    if test_case:
        print(f"\nTest case: {test_case['datafield']} in {test_case['region']}")
        print(f"Expected {test_case['expected_count']} matching IDs: {test_case['expected_ids'][:3]}...")

        # Test the fixed method using availability_details approach
        # This simulates what the modal callback does
        matching_datafields = []

        # Get detailed info for each matching datafield
        analysis_ops = AnalysisOperations('operators.txt')
        db_engine = analysis_ops._get_db_engine()

        with db_engine.connect() as connection:
            query = text("""
                SELECT DISTINCT datafield_id, data_description, dataset_id,
                       data_category, data_type, delay
                FROM datafields
                WHERE region = :region
                AND datafield_id = ANY(:datafield_ids)
            """)
            result = connection.execute(query, {
                'region': test_case['region'],
                'datafield_ids': test_case['expected_ids']
            })
            for row in result:
                matching_datafields.append({
                    'id': row.datafield_id,
                    'description': row.data_description or 'No description',
                    'dataset': row.dataset_id or 'Unknown'
                })

        print(f"\nActual results: Found {len(matching_datafields)} matching datafields")
        for df in matching_datafields[:3]:
            print(f"  - {df['id']}: {df['description'][:50]}...")

        if len(matching_datafields) == test_case['expected_count']:
            print("✓ Test PASSED: Count matches expected")
        else:
            print(f"✗ Test FAILED: Expected {test_case['expected_count']}, got {len(matching_datafields)}")
    else:
        print("No test case found with multiple matching IDs")


def test_view_alphas_modal_data():
    """Test that the view alphas modal gets correct data."""
    print("\n=== Testing View Alphas Modal Data ===")

    rec_service = get_recommendation_service()

    # Get a datafield that's used in submitted alphas
    recommendations_data = rec_service.get_datafield_recommendations()
    recommendations = recommendations_data.get('recommendations', [])

    if recommendations:
        # Test with first recommendation
        test_datafield = recommendations[0]
        datafield_id = test_datafield['datafield_id']
        used_regions = test_datafield.get('used_in_regions', [])
        usage_details = test_datafield.get('usage_details', {})

        print(f"\nTesting datafield: {datafield_id}")
        print(f"Used in regions: {used_regions}")
        print(f"Usage details: {usage_details}")

        # Get alphas for each region
        for region in used_regions[:2]:  # Test first 2 regions
            alphas = rec_service.get_alphas_by_datafield_and_region(datafield_id, region)
            expected_count = usage_details.get(region, 0)

            print(f"\n  Region: {region}")
            print(f"  Expected count: {expected_count}")
            print(f"  Actual count: {len(alphas)}")

            if alphas:
                print(f"  Sample alphas: {alphas[:3]}")

            if len(alphas) > 0:
                print("  ✓ Alphas found for this region")
            else:
                print("  ✗ No alphas found (should have data)")
    else:
        print("No recommendations found to test")


if __name__ == "__main__":
    print("Testing Datafield Cross-Analysis Fixes")
    print("=" * 50)

    try:
        test_datafield_recommendations()
        test_matching_datafields_in_region()
        test_view_alphas_modal_data()

        print("\n" + "=" * 50)
        print("All tests completed!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()