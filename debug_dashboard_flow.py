#!/usr/bin/env python3
"""
Debug script to test the entire dashboard data flow.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.analysis_operations import AnalysisOperations

def test_dashboard_data_flow():
    """Test the complete data flow as it would happen in the dashboard."""
    
    print("Testing complete dashboard data flow...")
    
    # Step 1: Initialize like the dashboard does
    OPERATORS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'operators.txt')
    DATAFIELDS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'all_datafields_comprehensive.csv')
    
    # Test the file paths that the dashboard actually uses
    print(f"Dashboard OPERATORS_FILE path: {OPERATORS_FILE}")
    print(f"Dashboard DATAFIELDS_FILE path: {DATAFIELDS_FILE}")
    print(f"OPERATORS_FILE exists: {os.path.exists(OPERATORS_FILE)}")
    print(f"DATAFIELDS_FILE exists: {os.path.exists(DATAFIELDS_FILE)}")
    
    # Adjust to current directory paths for testing
    OPERATORS_FILE = os.path.join(os.getcwd(), 'operators.txt')
    DATAFIELDS_FILE = os.path.join(os.getcwd(), 'all_datafields_comprehensive.csv')
    
    print(f"\nUsing corrected paths:")
    print(f"OPERATORS_FILE: {OPERATORS_FILE}")
    print(f"DATAFIELDS_FILE: {DATAFIELDS_FILE}")
    print(f"OPERATORS_FILE exists: {os.path.exists(OPERATORS_FILE)}")
    print(f"DATAFIELDS_FILE exists: {os.path.exists(DATAFIELDS_FILE)}")
    
    # Step 2: Test preload data callback (first thing dashboard does)
    print("\n=== Testing preload data callback ===")
    try:
        temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
        preloaded_results = temp_analysis_ops.get_analysis_summary()
        
        print(f"Preloaded results keys: {list(preloaded_results.keys())}")
        datafields_data = preloaded_results.get('datafields', {})
        print(f"Datafields keys: {list(datafields_data.keys())}")
        print(f"Unique usage count: {len(datafields_data.get('unique_usage', {}))}")
        
    except Exception as e:
        print(f"ERROR in preload: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test analysis data update (when no filters applied, uses preloaded)
    print("\n=== Testing analysis data update (no filters) ===")
    
    # Simulate: not any([region, universe, delay]) and preloaded_data
    region = None
    universe = None 
    delay = None
    
    if not any([region, universe, delay]) and preloaded_results:
        results = preloaded_results
        print("Using preloaded data (no filters)")
    else:
        print("Would get filtered results")
    
    # Step 4: Test datafields content creation
    print("\n=== Testing datafields content creation ===")
    
    analysis_data = results  # This is what gets passed to create_datafields_content
    datafields_data = analysis_data.get('datafields', {})
    
    print(f"Analysis data keys: {list(analysis_data.keys())}")
    print(f"Datafields data keys: {list(datafields_data.keys())}")
    
    top_datafields = datafields_data.get('top_datafields', [])
    print(f"Top datafields count: {len(top_datafields)}")
    
    if not top_datafields:
        print("ERROR: No top_datafields found!")
        return False
        
    # Test the dataset counting logic from create_datafields_content
    category_counts = {}
    dataset_counts = {}
    
    try:
        temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
        unique_usage = datafields_data.get('unique_usage', {})
        print(f"Processing {len(unique_usage)} unique datafields...")
        
        for df, alphas in unique_usage.items():
            if df in temp_analysis_ops.parser.datafields:
                dataset_id = temp_analysis_ops.parser.datafields[df]['dataset_id']
                category = temp_analysis_ops.parser.datafields[df]['data_category']
                
                if dataset_id:
                    dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + len(alphas)
                if category:
                    category_counts[category] = category_counts.get(category, 0) + len(alphas)
        
        print(f"Final dataset_counts: {len(dataset_counts)}")
        print(f"Final category_counts: {len(category_counts)}")
        
        # Test treemap logic
        if dataset_counts:
            sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"Would create treemap with {len(sorted_datasets)} datasets")
            print("SUCCESS: Treemap would be created!")
            return True
        else:
            print("ERROR: No dataset_counts - treemap would be empty!")
            return False
            
    except Exception as e:
        print(f"ERROR in dataset counting: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dashboard_data_flow()
    sys.exit(0 if success else 1)