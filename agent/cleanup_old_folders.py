"""
Cleanup Script for Incorrectly Named Output Folders

This script fixes folders created by old versions of the code that had
incorrect naming conventions.

Issues fixed:
1. Removes agent/output/chopped/ (should be agent/output/chopped_segments/[filename]/)
2. Renames folders with .wav_segments suffix to remove the suffix
3. Removes transcription folders for generic names like "chopped", "chopped_segments"
"""

import os
import shutil
from pathlib import Path

# Get the agent directory
agent_dir = Path(__file__).parent.absolute()
output_dir = agent_dir / "output"

def cleanup_old_folders():
    """Clean up incorrectly named folders from old code versions."""
    
    print("🧹 Starting cleanup of old folders...")
    print(f"📂 Working directory: {agent_dir}")
    print("="*80)
    
    changes_made = False
    
    # Issue 1: Remove agent/output/chopped/ folder (wrong location)
    old_chopped = output_dir / "chopped"
    if old_chopped.exists() and old_chopped.is_dir():
        print(f"\n❌ Found incorrectly located folder: {old_chopped}")
        print(f"   This folder was created by old code and should be removed.")
        
        # List files in the folder
        files = list(old_chopped.glob("*.wav"))
        print(f"   Contains {len(files)} segment files")
        
        response = input("   Delete this folder? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(old_chopped)
            print(f"   ✅ Deleted: {old_chopped}")
            changes_made = True
        else:
            print(f"   ⏭️  Skipped")
    
    # Issue 2: Rename transcription folders with .wav_segments suffix
    transcriptions_dir = output_dir / "transcriptions"
    if transcriptions_dir.exists():
        suffixes_to_check = ['.wav_segments', '.mp3_segments', '.flac_segments', '.m4a_segments', '.ogg_segments']
        
        for folder in transcriptions_dir.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                
                # Check if folder has unwanted suffix
                for suffix in suffixes_to_check:
                    if folder_name.endswith(suffix):
                        new_name = folder_name[:-len(suffix)]
                        new_path = transcriptions_dir / new_name
                        
                        print(f"\n❌ Found incorrectly named folder: {folder.name}")
                        print(f"   Should be renamed to: {new_name}")
                        
                        if new_path.exists():
                            print(f"   ⚠️  Target folder already exists: {new_path}")
                            response = input("   Merge contents and delete old folder? (y/n): ")
                            if response.lower() == 'y':
                                # Copy files from old folder to new folder
                                for file in folder.glob("*"):
                                    if file.is_file():
                                        target_file = new_path / file.name
                                        if not target_file.exists():
                                            shutil.copy2(file, target_file)
                                            print(f"      ✅ Copied: {file.name}")
                                        else:
                                            print(f"      ⏭️  File already exists: {file.name}")
                                
                                # Delete old folder
                                shutil.rmtree(folder)
                                print(f"   ✅ Deleted old folder: {folder.name}")
                                changes_made = True
                            else:
                                print(f"   ⏭️  Skipped")
                        else:
                            response = input("   Rename this folder? (y/n): ")
                            if response.lower() == 'y':
                                folder.rename(new_path)
                                print(f"   ✅ Renamed: {folder.name} → {new_name}")
                                changes_made = True
                            else:
                                print(f"   ⏭️  Skipped")
                        
                        break  # Only process one suffix per folder
    
    # Issue 3: Remove generic transcription folders (chopped, chopped_segments)
    generic_names = ["chopped", "chopped_segments"]
    for generic_name in generic_names:
        generic_folder = transcriptions_dir / generic_name
        if generic_folder.exists() and generic_folder.is_dir():
            print(f"\n❌ Found generic transcription folder: {generic_folder.name}")
            print(f"   This folder has a generic name and should be removed.")
            
            # List files
            files = list(generic_folder.glob("*"))
            print(f"   Contains {len(files)} files")
            
            response = input("   Delete this folder? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(generic_folder)
                print(f"   ✅ Deleted: {generic_folder}")
                changes_made = True
            else:
                print(f"   ⏭️  Skipped")
    
    print("\n" + "="*80)
    if changes_made:
        print("✅ Cleanup completed! Some changes were made.")
    else:
        print("✨ No issues found or no changes were made.")
    print("="*80)


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Folder Cleanup Script" + " "*37 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    try:
        cleanup_old_folders()
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")

