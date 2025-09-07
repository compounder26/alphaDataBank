#!/usr/bin/env python3
"""
Script to clear the alpha_analysis_cache table to force re-parsing with correct logic.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database.schema import get_connection
    from sqlalchemy import text
    
    def clear_analysis_cache():
        """Clear the alpha_analysis_cache table."""
        try:
            db_engine = get_connection()
            with db_engine.connect() as connection:
                with connection.begin():
                    # Count current entries
                    count_result = connection.execute(text("SELECT COUNT(*) FROM alpha_analysis_cache"))
                    current_count = count_result.scalar()
                    
                    print(f"Found {current_count} cached analysis entries")
                    
                    if current_count > 0:
                        # Clear the cache
                        connection.execute(text("DELETE FROM alpha_analysis_cache"))
                        print(f"✅ Cleared {current_count} entries from alpha_analysis_cache")
                        print("Next dashboard run will re-parse all alphas with corrected logic")
                    else:
                        print("Cache is already empty")
                        
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
            return False
            
        return True
    
    if __name__ == "__main__":
        success = clear_analysis_cache()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("This script requires database dependencies to run")
    sys.exit(1)