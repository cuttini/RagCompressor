import os
import logging
import gc
import time
import io
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# Ensure poppler is found by adding conda bin to PATH
os.environ["PATH"] += os.pathsep + "/home/sysadmin/miniconda3/envs/clara/bin"

from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
from tqdm import tqdm

# Assuming these imports exist in your project
import json
from hunyuan_processor import HunyuanProcessor
from docstrange_layout_extractor import DocstrangeLayoutExtractor
from config import EBOOKS_DIR, MARKDOWN_DIR, MAX_PAGES_PER_PDF, BATCH_SIZE, MAX_IMAGE_DIMENSION, ENABLE_LAYOUT_EXTRACTION, ENABLE_BATCH_INFERENCE, NUM_PARALLEL_WORKERS, CHUNK_SIZE

# Configure Logging to work with tqdm
class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that uses tqdm.write() to avoid interfering with progress bars."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

# Setup logging with tqdm-compatible handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root logger (prevents duplicates)
handler = TqdmLoggingHandler()
handler.setFormatter(logging.Formatter('%(message)s'))  # Simplified format
logger.addHandler(handler)

# Suppress library warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Constants
DEFAULT_CHUNK_SIZE = CHUNK_SIZE  # Use config value
MAX_IMG_DIM = MAX_IMAGE_DIMENSION  # Use config value

def resize_image_if_needed(img: Image.Image, max_dim: int = MAX_IMG_DIM) -> Image.Image:
    """Resizes image if either dimension exceeds max_dim, maintaining aspect ratio."""
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    return img

def convert_pdfs_hunyuan(input_dir: Path, output_dir: Path, max_pages: Optional[int] = None, chunk_size: int = DEFAULT_CHUNK_SIZE, debug: bool = False) -> None:
    # 1. Initialize Processor (only for sequential mode)
    processor = None
    if NUM_PARALLEL_WORKERS == 1:
        try:
            logger.info("🔧 Initializing HunyuanProcessor...")
            processor = HunyuanProcessor()
            logger.info("✓ Model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to initialize HunyuanProcessor: {e}")
            return

    # Initialize layout extractor if enabled
    layout_extractor = None
    if ENABLE_LAYOUT_EXTRACTION:
        try:
            logger.info("🔧 Initializing Layout Extractor...")
            layout_extractor = DocstrangeLayoutExtractor()
            logger.info("✓ Layout extractor loaded")
        except Exception as e:
            logger.error(f"⚠ Failed to initialize Layout Extractor: {e}")
            layout_extractor = None
    else:
        logger.info("⊘ Layout extraction disabled")

    output_dir.mkdir(exist_ok=True, parents=True)
    if debug:
        (output_dir / "debug_hunyuan").mkdir(exist_ok=True)

    # 2. Iterate Files
    input_dir = Path(input_dir)
    pdf_files = list(input_dir.rglob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"⚠ No PDF files found in {input_dir}")
        return

    logger.info(f"📄 Found {len(pdf_files)} PDF(s) to process")
    if NUM_PARALLEL_WORKERS > 1:
        logger.info(f"⚡ Using {NUM_PARALLEL_WORKERS} parallel workers")

    for pdf_path in pdf_files:
        output_path = output_dir / pdf_path.with_suffix(".md").name
        layout_output_path = output_dir / (pdf_path.stem + ".layout.json")
        
        # Skip if output already exists (only check .md file)
        if output_path.exists():
            logger.info(f"⊚ Skipping {pdf_path.name} (already processed)")
            continue

        logger.info(f"\n📖 Processing: {pdf_path.name}")

        try:
            # Get total page count first without loading images
            info = pdfinfo_from_path(pdf_path)
            total_pages = info["Pages"]
            
            # Apply max_pages limit
            pages_to_process = total_pages
            if max_pages and max_pages < total_pages:
                pages_to_process = max_pages
                logger.info(f"Limiting processing to first {pages_to_process} pages.")

            # 3. Pipelined Processing Loop
            # We load small chunks of images and submit them to the executor immediately.
            # We simultaneously check for completed pages and write them in order.
            
            progress_bar = tqdm(total=pages_to_process, desc=f"Converting {pdf_path.name}", unit="page")
            
            # Store layout data for the entire PDF
            full_layout_data = []

            # Create worker pool once per PDF (if using parallel processing)
            executor = None
            if NUM_PARALLEL_WORKERS > 1:
                from parallel_ocr_worker import init_worker, process_page_ocr
                executor = ProcessPoolExecutor(max_workers=NUM_PARALLEL_WORKERS, initializer=init_worker)
                logger.info(f"⚡ Started {NUM_PARALLEL_WORKERS} worker processes")
            
            # Create thread pool for background image loading
            image_loader = ThreadPoolExecutor(max_workers=1)

            try:
                # Open output file once for the entire PDF
                with open(output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(f"# {pdf_path.stem}\n\n")
                    
                    # State for pipelining
                    next_page_to_write = 1
                    pending_futures = {}  # {page_num: future}
                    
                    # Helper function for loading images
                    def load_chunk(start, end):
                        try:
                            return convert_from_path(
                                pdf_path, 
                                first_page=start, 
                                last_page=end,
                                thread_count=min(8, chunk_size)
                            )
                        except Exception as e:
                            logger.error(f"Failed to convert pages {start}-{end}: {e}")
                            return []

                    # Pre-load the first chunk
                    next_chunk_future = image_loader.submit(load_chunk, 1, min(chunk_size, pages_to_process))
                    
                    # Iterate through chunks
                    for chunk_start in range(1, pages_to_process + 1, chunk_size):
                        chunk_end = min(chunk_start + chunk_size - 1, pages_to_process)
                        
                        # 1. Get Pre-loaded Images (Producer)
                        # This waits if the loading isn't done yet, but it started long ago!
                        images = next_chunk_future.result()
                        
                        # 2. Start Loading NEXT Chunk Immediately
                        next_start = chunk_start + chunk_size
                        if next_start <= pages_to_process:
                            next_end = min(next_start + chunk_size - 1, pages_to_process)
                            next_chunk_future = image_loader.submit(load_chunk, next_start, next_end)
                        
                        if not images:
                            continue

                        # 3. Submit to Executor
                        for i, img in enumerate(images):
                            page_num = chunk_start + i
                            
                            if img.mode != 'RGB':
                                img = img.convert('RGB')

                            # Debug Save
                            if debug:
                                debug_path = output_dir / "debug_hunyuan" / f"{pdf_path.stem}_page_{page_num}.jpg"
                                img.save(debug_path)
                            
                            # Extract Layout (Docstrange)
                            page_blocks = []
                            if layout_extractor:
                                try:
                                    page_blocks = layout_extractor.extract_layout(img, extract_text_content=False)
                                except Exception as e:
                                    logger.error(f"Failed to extract layout on page {page_num}: {e}")
                            
                            full_layout_data.append({
                                "page": page_num,
                                "blocks": page_blocks
                            })

                            # Submit to OCR
                            img_resized = resize_image_if_needed(img)
                            
                            if NUM_PARALLEL_WORKERS > 1 and executor:
                                # Serialize
                                img_byte_arr = io.BytesIO()
                                img_resized.save(img_byte_arr, format='PNG')
                                img_bytes = img_byte_arr.getvalue()
                                
                                # Submit
                                future = executor.submit(process_page_ocr, (page_num, img_bytes))
                                pending_futures[page_num] = future
                            else:
                                # Sequential fallback (process immediately)
                                try:
                                    text = processor.extract_text(img_resized)
                                    # Fake a future result for consistency
                                    pending_futures[page_num] = (page_num, text, os.getpid(), 0, 0)
                                except Exception as e:
                                    logger.error(f"OCR Error on page {page_num}: {e}")
                                    pending_futures[page_num] = (page_num, f"> [Error]", os.getpid(), 0, 0)

                        # Cleanup images immediately to free RAM
                        del images
                        gc.collect()

                        # 3. Write Completed Pages (Consumer)
                        # Check if the next expected page is ready
                        while next_page_to_write in pending_futures:
                            # Get result
                            future_or_result = pending_futures[next_page_to_write]
                            
                            if NUM_PARALLEL_WORKERS > 1 and executor:
                                if not future_or_result.done():
                                    break # Next page not ready yet, go back to producing
                                result = future_or_result.result()
                            else:
                                result = future_or_result # It's already the result tuple
                            
                            # Unpack
                            page_num, text, pid, start_t, end_t = result
                            
                            # Log
                            if start_t > 0:
                                duration = end_t - start_t
                                logger.info(f"   ↳ Page {page_num} done by PID {pid} in {duration:.2f}s")
                            
                            # Write
                            page_md = f"## Page {page_num}\n\n{text}\n\n---\n\n"
                            md_file.write(page_md)
                            
                            # Cleanup future
                            del pending_futures[next_page_to_write]
                            
                            # Update progress
                            progress_bar.set_description(f"Writing p.{page_num}")
                            progress_bar.update(1)
                            
                            next_page_to_write += 1

                    # 4. Finish Remaining Pages
                    # After all chunks submitted, wait for remaining pages
                    while next_page_to_write <= pages_to_process:
                        if next_page_to_write in pending_futures:
                            future = pending_futures[next_page_to_write]
                            if NUM_PARALLEL_WORKERS > 1 and executor:
                                result = future.result() # Block until ready
                            else:
                                result = future
                                
                            page_num, text, pid, start_t, end_t = result
                            
                            if start_t > 0:
                                duration = end_t - start_t
                                logger.info(f"   ↳ Page {page_num} done by PID {pid} in {duration:.2f}s")

                            md_file.write(f"## Page {page_num}\n\n{text}\n\n---\n\n")
                            del pending_futures[next_page_to_write]
                            
                            progress_bar.set_description(f"Writing p.{page_num}")
                            progress_bar.update(1)
                            next_page_to_write += 1
                        else:
                            # Should not happen if logic is correct
                            logger.error(f"Page {next_page_to_write} missing from futures!")
                            next_page_to_write += 1
            
            finally:
                # Always cleanup the executor
                if executor:
                    executor.shutdown(wait=True)
                    logger.info("✓ Worker processes completed")
                image_loader.shutdown(wait=False)
                
            progress_bar.close()
            
            # Save Layout Data (only if enabled)
            if ENABLE_LAYOUT_EXTRACTION and layout_extractor:
                layout_out_path = output_dir / (pdf_path.stem + ".layout.json")
                with open(layout_out_path, "w", encoding="utf-8") as f:
                    json.dump(full_layout_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✓ Finished {pdf_path.name}\n")

        except Exception as e:
            logger.error(f"Critical error processing {pdf_path.name}: {e}")
            # Optionally delete partial file if failed
            # output_path.unlink(missing_ok=True) 

if __name__ == "__main__":
    import argparse
    
    # Set multiprocessing start method for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
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