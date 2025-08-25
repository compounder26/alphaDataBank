#!/usr/bin/env python3
"""
Debug the actual dashboard initialization to see what's happening with file paths.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dashboard_init():
    """Test the actual dashboard initialization."""
    
    print("Testing actual dashboard initialization...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Import exactly like the dashboard does
    try:
        from analysis.clustering.visualization_server import OPERATORS_FILE, DATAFIELDS_FILE
        
        print(f"\nDashboard file paths:")
        print(f"OPERATORS_FILE: {OPERATORS_FILE}")
        print(f"DATAFIELDS_FILE: {DATAFIELDS_FILE}")
        print(f"Operators exists: {os.path.exists(OPERATORS_FILE)}")
        print(f"Datafields exists: {os.path.exists(DATAFIELDS_FILE)}")
        
        if not os.path.exists(OPERATORS_FILE) or not os.path.exists(DATAFIELDS_FILE):
            print("ERROR: Files not found with dashboard paths!")
            return False
            
        # Test analysis operations initialization like the dashboard does
        from analysis.analysis_operations import AnalysisOperations
        analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
        print("\nAnalysis operations initialized successfully!")
        
        # Test preloading like the dashboard does
        results = analysis_ops.get_analysis_summary()
        datafields_data = results.get('datafields', {})
        unique_usage = datafields_data.get('unique_usage', {})
        
        print(f"Preloaded {len(unique_usage)} unique datafields")
        
        # Test the treemap data calculation like the dashboard does
        dataset_counts = {}
        temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
        
        for df, alphas in unique_usage.items():
            if df in temp_analysis_ops.parser.datafields:
                dataset_id = temp_analysis_ops.parser.datafields[df]['dataset_id']
                if dataset_id:
                    dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + len(alphas)
        
        print(f"Dataset counts calculated: {len(dataset_counts)}")
        
        if dataset_counts:
            sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"Top 5 datasets: {sorted_datasets[:5]}")
            print("SUCCESS: Dashboard should work correctly!")
            return True
        else:
            print("ERROR: No dataset counts - treemap would be empty!")
            return False
            
    except Exception as e:
        print(f"ERROR in dashboard initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dashboard_init()
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)