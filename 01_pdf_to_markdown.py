import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

# Ensure poppler is found by adding conda bin to PATH
os.environ["PATH"] += os.pathsep + "/home/sysadmin/miniconda3/envs/clara/bin"

from nanonets_processor import NanonetsProcessor
from docstrange_layout_extractor import DocstrangeLayoutExtractor
from pdf2image import convert_from_path

from config import EBOOKS_DIR, MARKDOWN_DIR, MAX_PAGES_PER_PDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_table_block(block: Dict[str, Any]) -> bool:
    """Helper to check if a layout block is a table."""
    # Check common keys 'type' or 'label' depending on the extractor implementation
    b_type = block.get('type', block.get('label', '')).lower()
    return 'table' in b_type

def clean_table_start(text: str) -> str:
    """Removes the opening <table> tag and potential repeated headers."""
    # Remove the first occurrence of <table>
    # You might also want to add logic here to remove repeated <thead>...</thead> if needed
    return re.sub(r'^\s*<table>', '', text.lstrip(), count=1, flags=re.IGNORECASE)

def clean_table_end(text: str) -> str:
    """Removes the closing </table> tag from the end of the text."""
    return re.sub(r'</table>\s*$', '', text.rstrip(), count=1, flags=re.IGNORECASE)

def has_table_header(text: str) -> bool:
    """Checks if the table at the start of the text has a header."""
    # Look for <thead> or <th> tags near the start
    # We limit the search to the first few hundred characters to avoid false positives later in the text
    start_snippet = text.lstrip()[:500].lower()
    if not start_snippet.startswith('<table>'):
        return False
    
    return '<thead>' in start_snippet or '<th>' in start_snippet

def convert_pdfs(input_dir: Path, output_dir: Path, max_pages: int = None, debug: bool = False) -> None:
    # Initialize Processors
    processor = NanonetsProcessor()
    
    try:
        layout_extractor = DocstrangeLayoutExtractor()
        logger.info("DocstrangeLayoutExtractor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize DocstrangeLayoutExtractor: {e}")
        layout_extractor = None

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
        
        # State variable for table merging
        previous_page_ended_with_table = False

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
            
            # Extract Text (Nanonets)
            try:
                text = processor.extract_text(img)
            except Exception as e:
                logger.error(f"Failed to extract text: {e}")
                text = ""

            # Extract Layout (Docstrange)
            page_blocks = []
            if layout_extractor:
                try:
                    page_blocks = layout_extractor.extract_layout(img)
                except Exception as e:
                    logger.error(f"Failed to extract layout: {e}")
            
            layout_data.append({
                "page": i + 1,
                "blocks": page_blocks
            })

            # --- START SOLUTION 2: Layout-Aware Merging ---
            
            current_page_starts_with_table = False
            current_page_ends_with_table = False

            if page_blocks:
                # Check first block
                if is_table_block(page_blocks[0]):
                    current_page_starts_with_table = True
                # Check last block
                if is_table_block(page_blocks[-1]):
                    current_page_ends_with_table = True

            # Determine if we should merge with the previous page
            # Rule: Merge ONLY if previous page ended with table, current starts with table,
            # AND the current table does NOT have a header (indicating continuation).
            should_merge = (
                previous_page_ended_with_table 
                and current_page_starts_with_table 
                and len(all_markdown_parts) > 0
                and not has_table_header(text)
            )

            if should_merge:
                print(f"    -> Detecting split table. Merging Page {i} into Page {i+1}...")
                
                # 1. Modify the previous part: remove the closing </table> tag
                # We access the last added part and strip the tag
                prev_text = all_markdown_parts[-1]
                all_markdown_parts[-1] = clean_table_end(prev_text)

                # 2. Modify current text: remove the opening <table> tag
                text = clean_table_start(text)

                # 3. Handle Header: Use HTML comment instead of H2 to keep table structure valid
                page_header = f"\n\n"
                
                # 4. Append directly to the previous part (avoids the '---' separator)
                all_markdown_parts[-1] += page_header + text
            
            else:
                # Standard processing (No merge)
                page_md = []
                page_md.append(f"## Page {i+1}")
                page_md.append(text)
                all_markdown_parts.append("\n".join(page_md))

            # Update state for next iteration
            previous_page_ended_with_table = current_page_ends_with_table
            
            # --- END SOLUTION 2 ---

            # Clean up image memory
            del img

        # 3. Save Markdown
        # Note: Merged tables are already combined in 'all_markdown_parts', 
        # so standard joining works fine for the rest of the document.
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