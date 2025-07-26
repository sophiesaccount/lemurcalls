#!/usr/bin/env python3
"""
Script to filter out entries from JSON files in a directory based on cluster='cl' entries in reference files.
Processes all JSON files in the target directory and its subdirectories.
Supports both single reference file and reference directory with matching structure.
Both files should have the structure: {"onset": [...], "offset": [...], "cluster": [...]}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def find_cluster_cl_entries(reference_file):
    """
    Find entries in reference_file that contain cluster='cl'
    
    Args:
        reference_file (str): Path to the reference JSON file
        
    Returns:
        set: Set of indices containing cluster='cl'
    """
    cluster_cl_indices = set()
    
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the expected structure exists
        if not all(key in data for key in ['onset', 'offset', 'cluster']):
            print("Error: Reference file does not have the expected structure with 'onset', 'offset', 'cluster'")
            sys.exit(1)
        
        clusters = data['cluster']
        onsets = data['onset']
        offsets = data['offset']
        
        for i, cluster in enumerate(clusters):
            if cluster == 'cl':
                cluster_cl_indices.add(i)
                print(f"Found cluster='cl' at index {i}: onset={onsets[i]:.2f}s, offset={offsets[i]:.2f}s")
                
    except FileNotFoundError:
        print(f"Error: Reference file '{reference_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Reference file is not valid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading reference file: {e}")
        sys.exit(1)
    
    print(f"Found {len(cluster_cl_indices)} entries with cluster='cl'")
    return cluster_cl_indices


def get_reference_file_for_target(target_file, reference_directory, target_directory):
    """
    Get the corresponding reference file for a target file based on relative path
    
    Args:
        target_file (str): Path to target file
        reference_directory (str): Path to reference directory
        target_directory (str): Path to target directory
        
    Returns:
        str: Path to corresponding reference file, or None if not found
    """
    # Get relative path from target directory
    rel_path = os.path.relpath(target_file, target_directory)
    
    # Construct reference file path
    reference_file = os.path.join(reference_directory, rel_path)
    
    # Check if reference file exists
    if os.path.exists(reference_file):
        return reference_file
    else:
        return None


def filter_target_file(target_file, cluster_cl_indices, output_file=None, dry_run=False):
    """
    Remove entries from target_file that correspond to cluster='cl' entries in reference file
    
    Args:
        target_file (str): Path to the target JSON file to filter
        cluster_cl_indices (set): Set of indices to remove
        output_file (str, optional): Output file path. If None, overwrites target_file
        dry_run (bool): If True, only show what would be removed without modifying files
    
    Returns:
        int: Number of entries removed
    """
    if output_file is None:
        output_file = target_file
    
    try:
        # Read JSON data from target file
        with open(target_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the expected structure exists
        if not all(key in data for key in ['onset', 'offset', 'cluster']):
            print(f"Warning: File '{target_file}' does not have the expected structure. Skipping.")
            return 0
        
        # Filter out entries that correspond to cluster='cl' entries
        filtered_onsets = []
        filtered_offsets = []
        filtered_clusters = []
        removed_count = 0
        
        for i in range(len(data['onset'])):
            if i in cluster_cl_indices:
                print(f"  Removing entry {i}: onset={data['onset'][i]:.2f}s, offset={data['offset'][i]:.2f}s, cluster={data['cluster'][i]}")
                removed_count += 1
            else:
                filtered_onsets.append(data['onset'][i])
                filtered_offsets.append(data['offset'][i])
                filtered_clusters.append(data['cluster'][i])
        
        if not dry_run:
            # Create filtered data structure, preserving all original metadata
            filtered_data = data.copy()  # Start with all original data
            filtered_data.update({
                'onset': filtered_onsets,
                'offset': filtered_offsets,
                'cluster': filtered_clusters
            })
            
            # Write filtered content back to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            print(f"  Removed {removed_count} entries from '{target_file}'")
            print(f"  Output saved to '{output_file}'")
        else:
            print(f"  Would remove {removed_count} entries from '{target_file}'")
        
        return removed_count
        
    except FileNotFoundError:
        print(f"Error: Target file '{target_file}' not found.")
        return 0
    except json.JSONDecodeError as e:
        print(f"Error: Target file '{target_file}' is not valid JSON: {e}")
        return 0
    except Exception as e:
        print(f"Error processing target file '{target_file}': {e}")
        return 0


def find_json_files(directory):
    """
    Find all JSON files in directory and subdirectories
    
    Args:
        directory (str): Path to directory to search
        
    Returns:
        list: List of paths to JSON files
    """
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def create_default_output_directory(target_directory):
    """
    Create a default output directory name based on target directory and timestamp
    
    Args:
        target_directory (str): Path to target directory
        
    Returns:
        str: Path to default output directory
    """
    # Get the base name of the target directory
    base_name = os.path.basename(os.path.abspath(target_directory))
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create default output directory name
    default_output = f"{base_name}_filtered_{timestamp}"
    
    return default_output


def main():
    parser = argparse.ArgumentParser(
        description="Filter out entries from JSON files in a directory based on cluster='cl' entries in reference files"
    )
    parser.add_argument(
        "--reference_file",
        help="Single JSON file containing the cluster='cl' entries to identify entries to remove"
    )
    parser.add_argument(
        "--reference_directory",
        help="Directory containing reference JSON files with matching structure to target directory"
    )
    parser.add_argument(
        "--target_directory", 
        required=True,
        help="Directory containing JSON files from which to remove the corresponding entries"
    )
    parser.add_argument(
        "--output_directory",
        help="Output directory for filtered files (default: creates new directory with timestamp)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually modifying files"
    )
    parser.add_argument(
        "--exclude_reference",
        action="store_true",
        help="Exclude the reference file from processing (only applies with --reference_file)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files (use with caution!)"
    )
    
    args = parser.parse_args()
    
    # Check if either reference_file or reference_directory is provided
    if not args.reference_file and not args.reference_directory:
        print("Error: Either --reference_file or --reference_directory must be provided.")
        sys.exit(1)
    
    if args.reference_file and args.reference_directory:
        print("Error: Please provide either --reference_file OR --reference_directory, not both.")
        sys.exit(1)
    
    # Check if target directory exists
    if not os.path.exists(args.target_directory):
        print(f"Error: Target directory '{args.target_directory}' does not exist.")
        sys.exit(1)
    
    # Check if reference file/directory exists
    if args.reference_file and not os.path.exists(args.reference_file):
        print(f"Error: Reference file '{args.reference_file}' does not exist.")
        sys.exit(1)
    
    if args.reference_directory and not os.path.exists(args.reference_directory):
        print(f"Error: Reference directory '{args.reference_directory}' does not exist.")
        sys.exit(1)
    
    # Determine output directory
    if args.overwrite:
        output_directory = None
        print("WARNING: Will overwrite original files!")
    elif args.output_directory:
        output_directory = args.output_directory
    else:
        # Create default output directory
        output_directory = create_default_output_directory(args.target_directory)
        print(f"Creating default output directory: {output_directory}")
    
    print(f"Target directory: {args.target_directory}")
    if args.reference_file:
        print(f"Reference file: {args.reference_file}")
    if args.reference_directory:
        print(f"Reference directory: {args.reference_directory}")
    if output_directory:
        print(f"Output directory: {output_directory}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    print()
    
    # Find all JSON files in target directory
    json_files = find_json_files(args.target_directory)
    
    if not json_files:
        print("No JSON files found in target directory.")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for f in json_files:
        print(f"  {f}")
    print()
    
    # Process each JSON file
    total_processed = 0
    total_removed = 0
    
    for json_file in json_files:
        print(f"Processing: {json_file}")
        
        # Get reference data for this file
        if args.reference_file:
            # Single reference file for all target files
            if args.exclude_reference and os.path.abspath(json_file) == os.path.abspath(args.reference_file):
                print("  Skipping reference file (--exclude_reference)")
                continue
            
            cluster_cl_indices = find_cluster_cl_entries(args.reference_file)
        else:
            # Reference directory - find corresponding reference file
            reference_file = get_reference_file_for_target(json_file, args.reference_directory, args.target_directory)
            
            if reference_file is None:
                print(f"  Warning: No corresponding reference file found for '{json_file}'. Skipping.")
                continue
            
            print(f"  Using reference file: {reference_file}")
            cluster_cl_indices = find_cluster_cl_entries(reference_file)
        
        if not cluster_cl_indices:
            print("  No entries with cluster='cl' found in reference. Nothing to remove.")
            continue
        
        # Determine output file path
        if output_directory:
            # Create relative path from target directory
            rel_path = os.path.relpath(json_file, args.target_directory)
            output_file = os.path.join(output_directory, rel_path)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            output_file = None  # Will overwrite original file
        
        # Filter the file
        removed_count = filter_target_file(json_file, cluster_cl_indices, output_file, args.dry_run)
        
        if removed_count > 0:
            total_processed += 1
            total_removed += removed_count
        
        print()
    
    # Summary
    print("=" * 50)
    print("SUMMARY:")
    print(f"Total files processed: {total_processed}")
    print(f"Total entries removed: {total_removed}")
    if output_directory and not args.dry_run:
        print(f"Filtered files saved to: {output_directory}")
    if args.dry_run:
        print("DRY RUN - No files were actually modified")
    print("=" * 50)


if __name__ == "__main__":
    main()