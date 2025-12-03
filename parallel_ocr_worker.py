"""
Worker module for parallel OCR processing.
Each worker process initializes its own HunyuanProcessor instance.
"""
import os
import logging
from PIL import Image
from hunyuan_processor import HunyuanProcessor

# Ensure poppler is found
os.environ["PATH"] += os.pathsep + "/home/sysadmin/miniconda3/envs/clara/bin"

# Suppress worker logging (only errors)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Suppress transformers and other library warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Global worker state (initialized once per process)
_worker_processor = None

def init_worker():
    """Initialize the HunyuanProcessor for this worker process."""
    global _worker_processor
    try:
        _worker_processor = HunyuanProcessor()
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to initialize: {e}")
        raise

def process_page_ocr(page_data):
    """
    Process a single page with OCR.
    
    Args:
        page_data: Tuple of (page_num, image_bytes)
        
    Returns:
        Tuple of (page_num, text)
    """
    global _worker_processor
    
    page_num, img_bytes = page_data
    
    try:
        import time
        start_t = time.time()
        
        # Reconstruct image from bytes
        import io
        img = Image.open(io.BytesIO(img_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Process with OCR
        text = _worker_processor.extract_text(img)
        
        end_t = time.time()
        
        # Return text plus debug metadata
        return (page_num, text, os.getpid(), start_t, end_t)
        
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to process page {page_num}: {e}")
        return (page_num, f"> [Error processing page {page_num}]", os.getpid(), 0, 0)
