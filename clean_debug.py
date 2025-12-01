#!/usr/bin/env python3
"""
Clean all generated data files.
This script removes:
- Generated markdown files (markdown_seac/*.md)
- Generated JPG images (markdown_seac/*.jpg)
- Debug images (markdown_seac/debug/)
- Hunyuan debug images (markdown_seac/debug_hunyuan/)
- Layout JSON files (markdown_seac/*.layout.json)
- Chunks JSONL file (artifacts/chunks.jsonl)
- Debug visualization images (artifacts/debug_vis/)
- Generated dataset files (artifacts/*.jsonl, artifacts/*.json)
"""

import os
import shutil
from pathlib import Path
from config import MARKDOWN_DIR, ARTIFACTS_DIR

def clean_debug_data():
    """Remove all debug and generated data."""
    items_removed = []
    
    # 1. Clean generated markdown files
    md_files = list(MARKDOWN_DIR.glob("*.md"))
    for file in md_files:
        file.unlink()
        items_removed.append(f"Removed markdown: {file.name}")
    
    # 2. Clean generated JPG files
    jpg_files = list(MARKDOWN_DIR.glob("*.jpg"))
    for file in jpg_files:
        file.unlink()
        items_removed.append(f"Removed JPG: {file.name}")
    
    # 3. Clean debug images directory
    debug_dir = MARKDOWN_DIR / "debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
        items_removed.append(f"Removed debug images: {debug_dir}")

    # 3b. Clean Hunyuan debug images directory
    hunyuan_debug_dir = MARKDOWN_DIR / "debug_hunyuan"
    if hunyuan_debug_dir.exists():
        shutil.rmtree(hunyuan_debug_dir)
        items_removed.append(f"Removed Hunyuan debug images: {hunyuan_debug_dir}")
    
    # 4. Clean layout JSON files
    layout_files = list(MARKDOWN_DIR.glob("*.layout.json"))
    for file in layout_files:
        file.unlink()
        items_removed.append(f"Removed layout JSON: {file.name}")
    
    # 5. Clean artifacts directory
    if ARTIFACTS_DIR.exists():
        # Remove chunks.jsonl
        chunks_file = ARTIFACTS_DIR / "chunks.jsonl"
        if chunks_file.exists():
            chunks_file.unlink()
            items_removed.append(f"Removed chunks: {chunks_file.name}")
        
        # Remove debug_vis directory
        debug_vis_dir = ARTIFACTS_DIR / "debug_vis"
        if debug_vis_dir.exists():
            shutil.rmtree(debug_vis_dir)
            items_removed.append(f"Removed debug visualizations: {debug_vis_dir}")
        
        # Remove all JSON and JSONL files in artifacts
        for pattern in ["*.json", "*.jsonl"]:
            for file in ARTIFACTS_DIR.glob(pattern):
                file.unlink()
                items_removed.append(f"Removed artifact: {file.name}")
    
    # 6. Report results
    if items_removed:
        print("🧹 Cleanup completed:")
        for item in items_removed:
            print(f"  ✓ {item}")
    else:
        print("✨ Nothing to clean - all directories are already clean!")

if __name__ == "__main__":
    print("Starting cleanup of all generated data...")
    clean_debug_data()
    print("\nDone!")
