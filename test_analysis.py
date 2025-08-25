#!/usr/bin/env python3
"""
Test script for the alpha analysis system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_parser():
    """Test the alpha expression parser."""
    print("Testing Alpha Expression Parser...")
    
    try:
        from analysis.alpha_expression_parser import AlphaExpressionParser
        parser = AlphaExpressionParser('operators.txt', 'all_datafields_comprehensive.csv')
        
        # Test expressions
        test_cases = [
            'rank(ts_returns(close, 5) + adv20)',
            'ts_mean(volume, 10) * ts_std_dev(returns, 20)',
            'group_neutralize(rank(ts_corr(close, volume, 30)), universe)',
            'if_else(close > ts_mean(close, 5), 1, -1)'
        ]
        
        for expr in test_cases:
            result = parser.parse_expression(expr)
            print(f"  Expression: {expr}")
            print(f"    Operators: {list(result['operators']['unique'])}")
            print(f"    Datafields: {list(result['datafields']['unique'])}")
            print()
        
        print("Parser test passed!")
        return True
        
    except Exception as e:
        print(f"X Parser test failed: {e}")
        return False

def test_database_connection():
    """Test database connection and schema."""
    print("Testing Database Connection...")
    
    try:
        from database.schema import get_connection
        
        db_engine = get_connection()
        with db_engine.connect() as connection:
            from sqlalchemy import text
            result = connection.execute(text("SELECT COUNT(*) FROM alphas"))
            count = result.fetchone()[0]
            print(f"  Found {count} alphas in database")
        
        print("Database test passed!")
        return True
        
    except Exception as e:
        print(f"X Database test failed: {e}")
        return False

def test_analysis_operations():
    """Test analysis operations."""
    print("Testing Analysis Operations...")
    
    try:
        from analysis.analysis_operations import AnalysisOperations
        
        analysis_ops = AnalysisOperations('operators.txt', 'all_datafields_comprehensive.csv')
        
        # Get sample data
        alphas_df = analysis_ops.get_alphas_for_analysis()
        print(f"  Retrieved {len(alphas_df)} alphas for analysis")
        
        if len(alphas_df) > 0:
            # Test with small sample
            sample_df = alphas_df.head(5)
            results = analysis_ops.get_analysis_summary()
            
            total_ops = len(results.get('operators', {}).get('unique_usage', {}))
            total_dfs = len(results.get('datafields', {}).get('unique_usage', {}))
            
            print(f"  Found {total_ops} unique operators")
            print(f"  Found {total_dfs} unique datafields")
        
        print("Analysis operations test passed!")
        return True
        
    except Exception as e:
        print(f"X Analysis operations test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Alpha Analysis System Tests\n")
    
    tests = [
        test_parser,
        test_database_connection,
        test_analysis_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The analysis system is ready to use.")
        print("\nTo start the dashboard, run:")
        print("   python run_analysis_dashboard.py")
    else:
        print("WARNING: Some tests failed. Please check the errors above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)