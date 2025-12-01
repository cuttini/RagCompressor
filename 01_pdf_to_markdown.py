import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# Ensure poppler is found by adding conda bin to PATH
os.environ["PATH"] += os.pathsep + "/home/sysadmin/miniconda3/envs/clara/bin"

# from docstrange.pipeline.neural_document_processor import NeuralDocumentProcessor
from nanonets_processor import NanonetsProcessor
from pdf2image import convert_from_path

from config import EBOOKS_DIR, MARKDOWN_DIR, MAX_PAGES_PER_PDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_pdfs(input_dir: Path, output_dir: Path, max_pages: int = None, debug: bool = False) -> None:
    # Initialize Processor
    processor = NanonetsProcessor()
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        source_path = input_dir / filename
        out_name = filename[:-4] + ".md"
        out_path = output_dir / out_name
        layout_out_path = output_dir / (filename[:-4] + ".layout.json")

        print(f"[pdf2md] Converting {source_path.name} -> {out_path.name}")
        
        # 1. Convert PDF to images
        try:
            images = convert_from_path(source_path)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            continue

        all_markdown_parts = []
        layout_data = []

        # 2. Process each page
        if max_pages:
            images = images[:max_pages]
            
        for i, img in enumerate(images):
            print(f"  Processing page {i+1}/{len(images)}...")
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if debug:
                # Save debug image
                debug_img_path = debug_dir / f"{filename[:-4]}_page_{i+1}.jpg"
                img.save(debug_img_path)
                print(f"  [DEBUG] Saved page image to {debug_img_path}")
            
            # Extract Text
            try:
                text = processor.extract_text(img)
            except Exception as e:
                logger.error(f"Failed to extract text: {e}")
                text = ""

            # Extract Layout (JSON)
            try:
                page_layout = processor.extract_layout_json(img)
            except Exception as e:
                logger.error(f"Failed to extract layout: {e}")
                page_layout = {}
            
            layout_data.append({
                "page": i + 1,
                "layout": page_layout
            })

            # Generate Markdown for this page
            page_md = []
            page_md.append(f"## Page {i+1}")
            page_md.append(text)
            
            all_markdown_parts.append("\n".join(page_md))
            
            # Clean up image memory?
            del img

        # 3. Save Markdown
        full_markdown = "\n\n---\n\n".join(all_markdown_parts)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)
            
        # 4. Save Layout Data
        with open(layout_out_path, "w", encoding="utf-8") as f:
            json.dump(layout_data, f, ensure_ascii=False, indent=2)
            
        print(f"[pdf2md] Saved layout data to {layout_out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (save intermediate images)")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_PDF, 
                        help="Maximum number of pages to process per PDF (default: config value or all pages)")
    args = parser.parse_args()

    # Process PDFs with optional page limit
    convert_pdfs(EBOOKS_DIR, MARKDOWN_DIR, max_pages=args.max_pages, debug=args.debug)
