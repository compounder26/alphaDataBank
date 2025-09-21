"""
Clustering utilities for alpha data analysis.

This module provides utilities for generating and managing clustering data
across different regions. Extracted from run_analysis_dashboard.py to improve
code organization and reusability.
"""
import os
import glob
from datetime import datetime
from typing import List, Tuple, Optional


def generate_all_regions_if_requested(regions_list: List[str]) -> List[str]:
    """Generate clustering data for multiple regions."""
    from analysis.clustering.clustering_algorithms import generate_clustering_data, save_clustering_results

    generated_files = []
    for region in regions_list:
        print(f"\nGenerating clustering data for {region}...")
        try:
            results = generate_clustering_data(region)
            if results:
                output_path = save_clustering_results(results)
                generated_files.append(output_path)
                print(f"Generated: {output_path}")
            else:
                print(f"Failed to generate data for {region}")
        except Exception as e:
            print(f"Error generating {region}: {e}")

    return generated_files


def delete_all_clustering_files() -> int:
    """Delete all existing clustering JSON files to force fresh recalculation."""
    print(f"\n🗑️ Cleaning up old clustering files at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Patterns for clustering files
    patterns = [
        "analysis/clustering/alpha_clustering_*.json",
        "clustering_results_*.json",
        "alpha_clustering_*.json"
    ]

    deleted_count = 0
    deleted_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                # Get file age before deletion
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getctime(file))
                os.remove(file)
                deleted_files.append((file, file_age))
                deleted_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not delete {file}: {e}")

    if deleted_count > 0:
        print(f"✅ Deleted {deleted_count} old clustering files:")
        for file, age in deleted_files[:5]:  # Show first 5 files
            age_str = f"{age.days}d {age.seconds//3600}h" if age.days > 0 else f"{age.seconds//3600}h {(age.seconds%3600)//60}m"
            print(f"   - {os.path.basename(file)} (was {age_str} old)")
        if deleted_count > 5:
            print(f"   ... and {deleted_count - 5} more files")
    else:
        print("   No clustering files found to delete")

    return deleted_count


def check_or_generate_clustering_data(region: str = 'USA') -> Optional[str]:
    """Check if clustering data exists for any region, generate if not."""
    from config.database_config import REGIONS

    # First check if any clustering data exists for the requested region
    region_files = [
        f"clustering_results_{region}.json",
        f"analysis/clustering/alpha_clustering_{region}*.json"
    ]

    # Check for existing files
    for pattern in region_files:
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getctime)
            print(f"Using existing clustering data: {latest_file}")
            return latest_file

    # No clustering data found, generate for requested region
    print(f"No clustering data found for {region}. Generating clustering data...")
    print("Available regions:", ", ".join(REGIONS))

    if region not in REGIONS:
        print(f"Warning: {region} not in available regions. Using USA as default.")
        region = 'USA'

    try:
        from analysis.clustering.clustering_algorithms import generate_clustering_data, save_clustering_results

        print(f"This may take a few minutes for region {region}...")
        results = generate_clustering_data(region)
        if results:
            output_path = save_clustering_results(results)
            print(f"Generated clustering data: {output_path}")
            return output_path
        else:
            print("Warning: Could not generate clustering data. Dashboard will run without clustering.")
            return None
    except Exception as e:
        print(f"Warning: Error generating clustering data: {e}")
        print("Dashboard will run without clustering visualization.")
        return None


def auto_generate_missing_regions(force_regenerate: bool = True) -> List[Tuple[str, Optional[str]]]:
    """Auto-generate clustering data for all regions that don't have existing files."""
    from config.database_config import REGIONS

    if force_regenerate:
        # Delete all existing clustering files first
        delete_all_clustering_files()
        print("\n🔄 Force regenerating clustering data for ALL regions...")

        # Generate for all regions since we deleted everything
        generated_files = generate_all_regions_if_requested(REGIONS)
        print(f"\n✅ Generated {len(generated_files)} new clustering files")
        return [(region, None) for region in REGIONS]

    print("Checking for existing clustering files across all regions...")

    missing_regions = []
    existing_regions = []

    for region in REGIONS:
        # Check for existing clustering files for this region
        pattern = f"analysis/clustering/alpha_clustering_{region}_*.json"
        files = glob.glob(pattern)

        if files:
            latest_file = max(files, key=os.path.getctime)
            existing_regions.append((region, latest_file))
            print(f"[OK] {region}: {latest_file}")
        else:
            missing_regions.append(region)
            print(f"[MISSING] {region}: No clustering data found")

    if missing_regions:
        print(f"\nGenerating clustering data for {len(missing_regions)} missing regions...")
        generated_files = generate_all_regions_if_requested(missing_regions)
        print(f"\nGenerated {len(generated_files)} new clustering files")
    else:
        print("\nAll regions have existing clustering data!")

    return existing_regions + [(region, None) for region in missing_regions]