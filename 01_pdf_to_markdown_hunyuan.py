import os
import logging
import gc
import time
from pathlib import Path
from typing import Optional, List

# Ensure poppler is found by adding conda bin to PATH
os.environ["PATH"] += os.pathsep + "/home/sysadmin/miniconda3/envs/clara/bin"

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from tqdm import tqdm

# Assuming these imports exist in your project
import json
from hunyuan_processor import HunyuanProcessor
from docstrange_layout_extractor import DocstrangeLayoutExtractor
from config import EBOOKS_DIR, MARKDOWN_DIR, MAX_PAGES_PER_PDF

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 32  # Number of pages to load into memory at once (optimized for high-end GPUs)
MAX_IMG_DIM = 2048

def resize_image_if_needed(img: Image.Image, max_dim: int = MAX_IMG_DIM) -> Image.Image:
    """Resizes image if either dimension exceeds max_dim, maintaining aspect ratio."""
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def convert_pdfs_hunyuan(input_dir: Path, output_dir: Path, max_pages: Optional[int] = None, chunk_size: int = DEFAULT_CHUNK_SIZE, debug: bool = False) -> None:
    # 1. Initialize Processor
    try:
        logger.info("Initializing HunyuanProcessor...")
        processor = HunyuanProcessor()
    except Exception as e:
        logger.critical(f"Failed to initialize HunyuanProcessor: {e}")
        return

    try:
        logger.info("Initializing DocstrangeLayoutExtractor...")
        layout_extractor = DocstrangeLayoutExtractor()
    except Exception as e:
        logger.error(f"Failed to initialize DocstrangeLayoutExtractor: {e}")
        layout_extractor = None

    output_dir.mkdir(exist_ok=True, parents=True)
    if debug:
        (output_dir / "debug_hunyuan").mkdir(exist_ok=True)

    # 2. Iterate Files
    # Convert string paths to Path objects if they aren't already
    input_dir = Path(input_dir)
    pdf_files = list(input_dir.rglob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDFs to process.")

    for pdf_path in pdf_files:
        output_path = output_dir / pdf_path.with_suffix(".md").name
        
        # Skip if output already exists (Optional: remove this check if you want to overwrite)
        if output_path.exists():
            logger.info(f"Skipping {pdf_path.name}, output already exists.")
            continue

        logger.info(f"Processing: {pdf_path.name}")

        try:
            # Get total page count first without loading images
            info = pdfinfo_from_path(pdf_path)
            total_pages = info["Pages"]
            
            # Apply max_pages limit
            pages_to_process = total_pages
            if max_pages and max_pages < total_pages:
                pages_to_process = max_pages
                logger.info(f"Limiting processing to first {pages_to_process} pages.")

            # 3. Chunked Processing Loop
            # We process in chunks to avoid loading 500 images into RAM
            progress_bar = tqdm(total=pages_to_process, desc=f"Converting {pdf_path.name}", unit="page")
            
            # Store layout data for the entire PDF
            full_layout_data = []

            # Open output file once for the entire PDF (major I/O optimization)
            with open(output_path, "w", encoding="utf-8") as md_file:
                # Write header
                md_file.write(f"# {pdf_path.stem}\n\n")
                
                for chunk_start in range(1, pages_to_process + 1, chunk_size):
                    chunk_end = min(chunk_start + chunk_size - 1, pages_to_process)
                    
                    logger.info(f"Processing chunk: Pages {chunk_start}-{chunk_end}")

                    try:
                        # Convert only the specific range of pages
                        images = convert_from_path(
                            pdf_path, 
                            first_page=chunk_start, 
                            last_page=chunk_end
                        )
                    except Exception as e:
                        logger.error(f"Failed to convert pages {chunk_start}-{chunk_end}: {e}")
                        continue

                    chunk_markdown = []

                    for i, img in enumerate(images):
                        page_num = chunk_start + i
                        
                        progress_bar.set_description(f"OCR Page {page_num}")
                        
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Debug Save
                        if debug:
                            debug_path = output_dir / "debug_hunyuan" / f"{pdf_path.stem}_page_{page_num}.jpg"
                            img.save(debug_path)
                        
                        # Extract Layout (Docstrange) - BEFORE resizing for OCR if possible, 
                        # but Docstrange expects full size usually. 
                        # Note: We should probably keep the original image for layout extraction 
                        # and use the resized one for Hunyuan if needed. 
                        # However, Hunyuan resize is only if > 2048. 
                        # Let's extract layout on the original image to be safe/accurate.
                        page_blocks = []
                        if layout_extractor:
                            try:
                                # Extract layout
                                page_blocks = layout_extractor.extract_layout(img)
                            except Exception as e:
                                logger.error(f"Failed to extract layout on page {page_num}: {e}")
                        
                        full_layout_data.append({
                            "page": page_num,
                            "blocks": page_blocks
                        })

                        # Resize for Hunyuan
                        img_resized = resize_image_if_needed(img)

                        # Extract Text
                        try:
                            start_time = time.time()
                            text = processor.extract_text(img_resized)
                            elapsed_time = time.time() - start_time
                            logger.debug(f"Processed page {page_num} in {elapsed_time:.2f}s")
                            progress_bar.set_postfix({"time": f"{elapsed_time:.2f}s"})
                        except Exception as e:
                            logger.error(f"OCR Error on page {page_num}: {e}")
                            text = f"> [Error processing page {page_num}]"

                        # Format Markdown
                        page_md = f"## Page {page_num}\n\n{text}\n\n---\n\n"
                        chunk_markdown.append(page_md)
                        
                        progress_bar.update(1)

                    # Write chunk immediately (avoids holding all pages in memory)
                    md_file.writelines(chunk_markdown)

                    # Cleanup memory
                    del images
                    gc.collect() # Force garbage collection for large image objects
            
            progress_bar.close()
            
            # Save Layout Data
            layout_out_path = output_dir / (pdf_path.stem + ".layout.json")
            with open(layout_out_path, "w", encoding="utf-8") as f:
                json.dump(full_layout_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved layout data to {layout_out_path.name}")
            
            logger.info(f"Finished {pdf_path.name}")

        except Exception as e:
            logger.error(f"Critical error processing {pdf_path.name}: {e}")
            # Optionally delete partial file if failed
            # output_path.unlink(missing_ok=True) 

if __name__ == "__main__":
    import argparse
    
    # Setup CLI
    parser = argparse.ArgumentParser(description="Convert PDFs to Markdown using Hunyuan OCR.")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_PDF, 
                        help="Limit pages per PDF (default: from config)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Number of pages to process at once (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--poppler-path", type=str, 
                        help="Optional: Path to poppler bin folder if not in PATH")

    args = parser.parse_args()

    # Handle Poppler Path safely
    if args.poppler_path:
        os.environ["PATH"] += os.pathsep + args.poppler_path
    
    # Run
    convert_pdfs_hunyuan(EBOOKS_DIR, MARKDOWN_DIR, max_pages=args.max_pages, chunk_size=args.chunk_size, debug=args.debug)