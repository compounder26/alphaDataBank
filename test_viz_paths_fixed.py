#!/usr/bin/env python3
import os
import sys

# Add project root to path (same as visualization server does)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test the FIXED file path calculation from visualization server (2 levels up, not 3)
OPERATORS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'operators.txt')
DATAFIELDS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'all_datafields_comprehensive.csv')

print('FIXED Visualization server path calculations:')
print('OPERATORS_FILE:', OPERATORS_FILE)
print('DATAFIELDS_FILE:', DATAFIELDS_FILE)
print('Operators exists:', os.path.exists(OPERATORS_FILE))
print('Datafields exists:', os.path.exists(DATAFIELDS_FILE))

# Test analysis operations initialization
try:
    from analysis.analysis_operations import AnalysisOperations
    analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
    print('Analysis operations initialized successfully!')
    
    # Test getting summary
    results = analysis_ops.get_analysis_summary()
    datafields_data = results.get('datafields', {})
    unique_usage = datafields_data.get('unique_usage', {})
    print(f'Found {len(unique_usage)} unique datafields')
    
    # Test the treemap data specifically
    category_counts = {}
    dataset_counts = {}
    
    temp_analysis_ops = AnalysisOperations(OPERATORS_FILE, DATAFIELDS_FILE)
    for df, alphas in unique_usage.items():
        if df in temp_analysis_ops.parser.datafields:
            dataset_id = temp_analysis_ops.parser.datafields[df]['dataset_id']
            if dataset_id:
                dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + len(alphas)
    
    print(f'Dataset counts for treemap: {len(dataset_counts)}')
    if dataset_counts:
        print('TOP 5 datasets:', sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        print('SUCCESS: TREEMAP SHOULD NOW WORK!')
    else:
        print('ERROR: No dataset counts found!')
        
except Exception as e:
    print(f'ERROR initializing analysis operations: {e}')
    import traceback
    traceback.print_exc()