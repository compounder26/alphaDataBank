#!/usr/bin/env python
"""
Utility script to regenerate clustering data for all regions.
This ensures fresh clustering analysis before starting the production server.

Usage:
    python refresh_clustering.py
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_analysis_dashboard import generate_clustering_for_regions
from config.database_config import REGIONS

def main():
    """Regenerate clustering data for specified or all regions."""
    parser = argparse.ArgumentParser(description="Regenerate clustering data")
    parser.add_argument("--regions", nargs='*', help="Specific regions to regenerate (default: all)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔄 Regenerating Clustering Data")
    print("=" * 60)
    
    try:
        # Determine which regions to process
        if args.regions:
            regions_to_process = args.regions
            print(f"📍 Processing specific regions: {', '.join(regions_to_process)}")
        else:
            regions_to_process = list(REGIONS.keys())
            print(f"🌍 Processing all regions: {', '.join(regions_to_process)}")
        
        print("\nGenerating clustering data...\n")
        
        # Generate clustering for regions
        success_count = 0
        failed_regions = []
        
        for region in regions_to_process:
            print(f"Processing {region}...")
            try:
                # Call the clustering generation function
                result = generate_clustering_for_regions([region])
                if result:
                    print(f"✅ {region}: Clustering generated successfully")
                    success_count += 1
                else:
                    print(f"⚠️ {region}: No data to cluster")
                    failed_regions.append(region)
            except Exception as e:
                print(f"❌ {region}: Failed - {str(e)}")
                failed_regions.append(region)
        
        print("\n" + "=" * 60)
        print(f"✨ Clustering Generation Complete!")
        print(f"   Successful: {success_count}/{len(regions_to_process)} regions")
        
        if failed_regions:
            print(f"   Failed/Skipped: {', '.join(failed_regions)}")
        
        print("\n🚀 Ready to start the production server:")
        print("   Windows: waitress-serve --host=127.0.0.1 --port=8050 wsgi:server")
        print("   Linux/Mac: gunicorn -w 4 -b 127.0.0.1:8050 wsgi:server")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()