#!/usr/bin/env python3
"""
Debug script to test the treemap creation logic.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import plotly.express as px
import plotly.graph_objects as go
from analysis.analysis_operations import AnalysisOperations

def test_treemap_creation():
    """Test the treemap creation logic in isolation."""
    
    print("Testing treemap creation logic...")
    
    # Initialize analysis operations
    OPERATORS_FILE = os.path.join(os.getcwd(), 'operators.txt')
    DATAFIELDS_FILE = os.path.join(os.getcwd(), 'all_datafields_comprehensive.csv')
    
    analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
    results = analysis_ops.get_analysis_summary()
    
    # Get datafields data like the visualization server
    datafields_data = results.get('datafields', {})
    print(f"Datafields data keys: {list(datafields_data.keys())}")
    print(f"Unique usage has {len(datafields_data.get('unique_usage', {}))} datafields")
    
    # Replicate the exact logic from visualization server
    category_counts = {}
    dataset_counts = {}
    
    try:
        temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
        for df, alphas in datafields_data.get('unique_usage', {}).items():
            if df in temp_analysis_ops.parser.datafields:
                dataset_id = temp_analysis_ops.parser.datafields[df]['dataset_id']
                category = temp_analysis_ops.parser.datafields[df]['data_category']
                
                if dataset_id:
                    dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + len(alphas)
                if category:
                    category_counts[category] = category_counts.get(category, 0) + len(alphas)
        
        print(f"Dataset counts calculated: {len(dataset_counts)}")
        print(f"Category counts calculated: {len(category_counts)}")
        
        # Test treemap creation
        if dataset_counts:
            print("Creating treemap...")
            # Limit to top 20 datasets for readability
            sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"Top 10 datasets: {sorted_datasets[:10]}")
            
            # Create treemap with proper parameters - test the exact code from vis server
            fig3 = px.treemap(
                names=[name for name, _ in sorted_datasets],
                values=[value for _, value in sorted_datasets],
                title="Top 20 Datasets by Usage"
            )
            fig3.update_traces(
                textinfo="label+value",
                textposition="middle center"
            )
            fig3.update_layout(height=300)
            
            print("Treemap created successfully!")
            print(f"Treemap data points: {len(sorted_datasets)}")
            return True
        else:
            print("ERROR: dataset_counts is empty!")
            return False
            
    except Exception as e:
        print(f"Error in treemap logic: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_treemap_creation()
    sys.exit(0 if success else 1)