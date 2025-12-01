import json
import os
from pathlib import Path

# Ensure poppler is found by adding conda bin to PATH
os.environ["PATH"] += os.pathsep + "/home/sysadmin/miniconda3/envs/clara/bin"

from config import EBOOKS_DIR, MARKDOWN_DIR, CHUNKS_PATH, ARTIFACTS_DIR
from debug_utils import visualize_chunks

def run_debug_visualization():
    print("Starting Debug Visualization...")
    
    # Load chunks
    if not CHUNKS_PATH.exists():
        print(f"Chunks file not found at {CHUNKS_PATH}")
        return
        
    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
        
    # Group chunks by source file
    chunks_by_file = {}
    for chunk in chunks:
        source = chunk.get("source", "")
        if source not in chunks_by_file:
            chunks_by_file[source] = []
        chunks_by_file[source].append(chunk)
        
    # Process each source file
    for source_md_name, file_chunks in chunks_by_file.items():
        # source is like "Ammortamento_2024.md"
        # we need the PDF path and the layout json path
        base_name = source_md_name[:-3] # remove .md
        pdf_path = EBOOKS_DIR / (base_name + ".pdf")
        layout_path = MARKDOWN_DIR / (base_name + ".layout.json")
        
        if not pdf_path.exists():
            print(f"PDF not found for {source_md_name}: {pdf_path}")
            continue
            
        if not layout_path.exists():
            print(f"Layout JSON not found for {source_md_name}: {layout_path}")
            print("Please run 01_pdf_to_markdown.py first to generate layout data.")
            continue
            
        print(f"Processing {source_md_name}...")
        with open(layout_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
            
        output_dir = ARTIFACTS_DIR / "debug_vis" / base_name
        visualize_chunks(str(pdf_path), layout_data, file_chunks, output_dir)
        print(f"Visualization saved to {output_dir}")

if __name__ == "__main__":
    run_debug_visualization()
