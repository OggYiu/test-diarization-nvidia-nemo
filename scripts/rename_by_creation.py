"""
Rename files in a folder by their creation date.

Usage:
  python scripts/rename_by_creation.py "C:\path\to\folder" --dry-run

This script will rename files to the pattern: YYYYMMDD_HHMMSS_originalname.ext
It avoids collisions by appending an incrementing counter when necessary.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from datetime import datetime


def format_dt(dt: float) -> str:
    return datetime.fromtimestamp(dt).strftime("%Y%m%d_%H%M%S")


def rename_files(folder: Path, dry_run: bool = True, preview: int = 0):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    entries = sorted([p for p in folder.iterdir() if p.is_file()])
    changes = []

    for p in entries:
        stat = p.stat()
        # On Windows, st_ctime is creation time. On other OS it may be metadata change time.
        created = stat.st_ctime
        prefix = format_dt(created)
        new_name = f"{prefix}_{p.name}"
        new_path = folder / new_name
        # avoid collisions
        counter = 1
        while new_path.exists():
            new_name = f"{prefix}_{counter:03d}_{p.name}"
            new_path = folder / new_name
            counter += 1

        changes.append((p, new_path))

    if preview:
        for old, new in changes[:preview]:
            print(f"[PREVIEW] {old.name} -> {new.name}")
        return

    for old, new in changes:
        if dry_run:
            print(f"DRY RUN: {old.name} -> {new.name}")
        else:
            print(f"Renaming: {old.name} -> {new.name}")
            old.rename(new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing files to rename")
    parser.add_argument("--apply", action="store_true", help="Actually perform renames (default is dry-run)")
    parser.add_argument("--preview", type=int, default=0, help="Show first N proposed renames and exit")

    args = parser.parse_args()
    target = Path(args.folder).expanduser().resolve()
    rename_files(target, dry_run=not args.apply, preview=args.preview)
