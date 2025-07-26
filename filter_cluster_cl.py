#!/usr/bin/env python3
"""
Script to filter out lines from one file based on cluster='cl' entries in another file.
"""

import argparse
import os
import sys
from pathlib import Path


def find_cluster_cl_lines(reference_file):
    """
    Find line numbers in reference_file that contain cluster='cl'
    
    Args:
        reference_file (str): Path to the reference file
        
    Returns:
        set: Set of line numbers (1-indexed) containing cluster='cl'
    """
    cluster_cl_lines = set()
    
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if "cluster='cl'" in line:
                    cluster_cl_lines.add(line_num)
                    print(f"Found cluster='cl' in line {line_num}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Reference file '{reference_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading reference file: {e}")
        sys.exit(1)
    
    print(f"Found {len(cluster_cl_lines)} lines with cluster='cl'")
    return cluster_cl_lines


def filter_target_file(target_file, cluster_cl_lines, output_file=None):
    """
    Remove lines from target_file that correspond to cluster='cl' lines in reference file
    
    Args:
        target_file (str): Path to the target file to filter
        cluster_cl_lines (set): Set of line numbers to remove
        output_file (str, optional): Output file path. If None, overwrites target_file
    """
    if output_file is None:
        output_file = target_file
    
    try:
        # Read all lines from target file
        with open(target_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out lines that correspond to cluster='cl' entries
        filtered_lines = []
        removed_count = 0
        
        for line_num, line in enumerate(lines, 1):
            if line_num in cluster_cl_lines:
                print(f"Removing line {line_num}: {line.strip()}")
                removed_count += 1
            else:
                filtered_lines.append(line)
        
        # Write filtered content back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        
        print(f"Removed {removed_count} lines from '{target_file}'")
        print(f"Output saved to '{output_file}'")
        
    except FileNotFoundError:
        print(f"Error: Target file '{target_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing target file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Filter out lines from a target file based on cluster='cl' entries in a reference file"
    )
    parser.add_argument(
        "--reference_file",
        help="File containing the cluster='cl' entries to identify lines to remove"
    )
    parser.add_argument(
        "--target_file", 
        help="File from which to remove the corresponding lines"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: overwrites target_file)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually modifying files"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference file '{args.reference_file}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.target_file):
        print(f"Error: Target file '{args.target_file}' does not exist.")
        sys.exit(1)
    
    print(f"Reference file: {args.reference_file}")
    print(f"Target file: {args.target_file}")
    if args.output:
        print(f"Output file: {args.output}")
    print()
    
    # Find lines with cluster='cl' in reference file
    cluster_cl_lines = find_cluster_cl_lines(args.reference_file)
    
    if not cluster_cl_lines:
        print("No lines with cluster='cl' found. Nothing to remove.")
        return
    
    print()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print("Lines that would be removed:")
        try:
            with open(args.target_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num in cluster_cl_lines:
                        print(f"  Line {line_num}: {line.strip()}")
        except Exception as e:
            print(f"Error reading target file for dry run: {e}")
            sys.exit(1)
    else:
        # Filter the target file
        filter_target_file(args.target_file, cluster_cl_lines, args.output)


if __name__ == "__main__":
    main() 