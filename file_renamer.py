"""
Phone Recording File Renamer
Renames phone recording files from the format:
  [Dickson Lau 0489]_8330-22776622_20251014014015(8059).wav
To:
  [Dickson Lau]_20251014014015_8330-22776622_0489_(8059).wav

This reorganizes the metadata to allow proper chronological sorting.
"""

import os
import re
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple


def parse_filename(filename: str) -> Optional[dict]:
    """
    Parse the original filename format and extract metadata.
    
    Format: [Name BrokerID]_Unknown-Phone_YYYYMMDDHHmmss(Unknown).ext
    
    Args:
        filename: Original filename
        
    Returns:
        Dictionary with extracted metadata or None if parsing fails
    """
    # Pattern to match the filename format
    # [Broker Name OptionalBrokerID]_Unknown-Phone_YYYYMMDDHHmmss(Unknown).ext
    pattern = r'^\[([^\]]+?)\s*(\d+)?\]_(\d+)-(\d+)_(\d{14})\((\d+)\)(\.\w+)$'
    
    match = re.match(pattern, filename)
    if not match:
        return None
    
    broker_name = match.group(1).strip()
    broker_id = match.group(2) if match.group(2) else ""
    unknown_1 = match.group(3)  # e.g., 8330
    phone_number = match.group(4)
    datetime_str = match.group(5)
    unknown_2 = match.group(6)  # e.g., 8059
    extension = match.group(7)
    
    return {
        'broker_name': broker_name,
        'broker_id': broker_id,
        'unknown_1': unknown_1,
        'phone_number': phone_number,
        'datetime': datetime_str,
        'unknown_2': unknown_2,
        'extension': extension
    }


def generate_new_filename(metadata: dict) -> str:
    """
    Generate the new filename from extracted metadata.
    
    New format: [Name]_YYYYMMDDHHmmss_Unknown-Phone_BrokerID_(Unknown).ext
    
    Args:
        metadata: Dictionary containing extracted metadata
        
    Returns:
        New filename string
    """
    broker_name = metadata['broker_name']
    datetime_str = metadata['datetime']
    unknown_1 = metadata['unknown_1']
    phone_number = metadata['phone_number']
    broker_id = metadata['broker_id']
    unknown_2 = metadata['unknown_2']
    extension = metadata['extension']
    
    # Build new filename
    new_filename = f"[{broker_name}]_{datetime_str}_{unknown_1}-{phone_number}"
    
    # Add broker ID if it exists
    if broker_id:
        new_filename += f"_{broker_id}"
    
    # Add unknown_2 and extension
    new_filename += f"_({unknown_2}){extension}"
    
    return new_filename


def rename_file(file_path: Path, output_dir: Optional[Path] = None, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Rename a single file according to the new format.
    
    Args:
        file_path: Path to the original file
        output_dir: Optional output directory (if None, renames in place)
        dry_run: If True, only show what would be done without actually renaming
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    filename = file_path.name
    
    # Parse the filename
    metadata = parse_filename(filename)
    if not metadata:
        return False, f"Could not parse filename: {filename}"
    
    # Generate new filename
    new_filename = generate_new_filename(metadata)
    
    # Determine output path
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        new_path = output_dir / new_filename
    else:
        new_path = file_path.parent / new_filename
    
    # Check if target already exists
    if new_path.exists() and new_path != file_path:
        return False, f"Target file already exists: {new_filename}"
    
    # Perform the rename/copy
    if dry_run:
        action = "copy to" if output_dir else "rename to"
        return True, f"Would {action}: {new_filename}"
    else:
        try:
            if output_dir:
                shutil.copy2(file_path, new_path)
                action = "Copied to"
            else:
                file_path.rename(new_path)
                action = "Renamed to"
            return True, f"{action}: {new_filename}"
        except Exception as e:
            return False, f"Error processing {filename}: {str(e)}"


def process_path(input_path: str, output_dir: Optional[str] = None, dry_run: bool = False):
    """
    Process a file or directory of files.
    
    Args:
        input_path: Path to file or directory
        output_dir: Optional output directory
        dry_run: If True, only show what would be done
    """
    input_path = Path(input_path)
    output_path = Path(output_dir) if output_dir else None
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        return
    
    # Collect files to process
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        # Get all files in directory (non-recursive)
        files_to_process = [f for f in input_path.iterdir() if f.is_file()]
    
    if not files_to_process:
        print("No files found to process.")
        return
    
    # Process each file
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    print(f"\nProcessing {len(files_to_process)} file(s)...\n")
    
    for file_path in sorted(files_to_process):
        success, message = rename_file(file_path, output_path, dry_run)
        
        if success:
            print(f"✓ {message}")
            success_count += 1
        else:
            if "Could not parse" in message:
                skipped_count += 1
                if dry_run or input_path.is_dir():
                    # Only show skipped files if processing directory or in dry-run mode
                    print(f"⊘ Skipped: {file_path.name} (does not match expected format)")
            else:
                print(f"✗ {message}")
                fail_count += 1
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Successful: {success_count}")
    if skipped_count > 0:
        print(f"  Skipped: {skipped_count}")
    if fail_count > 0:
        print(f"  Failed: {fail_count}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='Rename phone recording files to a sortable format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rename a single file in place
  python file_renamer.py "recording.wav"
  
  # Rename all files in a directory
  python file_renamer.py "recordings/"
  
  # Copy renamed files to output directory
  python file_renamer.py "recordings/" -o "renamed_output/"
  
  # Dry run to preview changes
  python file_renamer.py "recordings/" --dry-run

Original format:
  [Dickson Lau 0489]_8330-22776622_20251014014015(8059).wav

New format:
  [Dickson Lau]_20251014014015_8330-22776622_0489_(8059).wav
        """
    )
    
    parser.add_argument(
        'input',
        help='Path to file or directory containing files to rename'
    )
    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        help='Output directory (if not specified, renames files in place)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without actually renaming files'
    )
    
    args = parser.parse_args()
    
    process_path(args.input, args.output_dir, args.dry_run)


if __name__ == '__main__':
    main()



