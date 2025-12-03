"""
Custom layout extractor using Docstrange's neural pipeline components.
This script provides access to the internal layout detection capabilities
(bounding boxes) which are not exposed by the standard Docstrange API.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image

from docstrange.pipeline.neural_document_processor import NeuralDocumentProcessor

# Configure logging
logging.basicConfig(level=logging.ERROR)  # Only errors
logger = logging.getLogger(__name__)

# Suppress library warnings
import warnings
warnings.filterwarnings('ignore')

class DocstrangeLayoutExtractor(NeuralDocumentProcessor):
    """
    Extracts layout information (bounding boxes) from images using 
    Docstrange's underlying neural models (Docling + EasyOCR).
    Inherits from NeuralDocumentProcessor to reuse initialized models.
    """
    
    def __init__(self, cache_dir: str = None, device: str = None):
        """Initialize the extractor and load models via parent class.
        
        Args:
            cache_dir: Optional directory for caching models
            device: Device to use for inference ('cuda' or 'cpu'). If None, auto-detects CUDA availability.
        """
        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        
        # Note: NeuralDocumentProcessor doesn't accept device parameter, it auto-detects
        super().__init__(cache_dir=Path(cache_dir) if cache_dir else None)
        
    def extract_layout(self, image_input: Any, extract_text_content: bool = True) -> List[Dict[str, Any]]:
        """
        Extract layout elements with bounding boxes from an image.
        
        Args:
            image_input: Path to the image file or PIL Image object.
            extract_text_content: If True, extracts text from each region (slow). If False, returns empty text.
            
        Returns:
            List of dictionaries containing:
            - text: The extracted text (or empty string if disabled)
            - bbox: [x1, y1, x2, y2]
            - label: The type of element (text, title, table, etc.)
            - confidence: Confidence score
        """
        try:
            if isinstance(image_input, str) or isinstance(image_input, Path):
                if not os.path.exists(image_input):
                    logger.error(f"Image file not found: {image_input}")
                    return []
                img = Image.open(image_input)
                should_close = True
            elif isinstance(image_input, Image.Image):
                img = image_input
                should_close = False
            else:
                logger.error(f"Unsupported image input type: {type(image_input)}")
                return []

            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Use layout_predictor from parent class
            if not self.layout_predictor:
                logger.error("Layout predictor not initialized (possibly in fallback mode)")
                return []

            # Predict layout
            layout_results = list(self.layout_predictor.predict(img))
            
            extracted_elements = []
            
            for pred in layout_results:
                # Get bounding box
                if all(k in pred for k in ['l', 't', 'r', 'b']):
                    bbox = [pred['l'], pred['t'], pred['r'], pred['b']]
                else:
                    bbox = pred.get('bbox') or pred.get('box')
                    if not bbox:
                        continue
                        
                # Get label
                label = pred.get('label', 'text').lower()
                confidence = pred.get('confidence', 1.0)
                
                # Extract text from the region using parent class method or custom logic
                text = ""
                if extract_text_content:
                    # Parent class has _extract_text_from_region but it takes bbox as list
                    text = self._extract_text_from_region(img, bbox)
                
                # If we are not extracting text, we still keep the block (it's a layout element)
                # If we ARE extracting text, we might want to filter empty ones, but let's keep consistent
                if extract_text_content and not text.strip():
                    continue
                    
                element = {
                    "text": text,
                    "bbox": [float(x) for x in bbox], # Ensure JSON serializable
                    "label": label,
                    "confidence": float(confidence)
                }
                extracted_elements.append(element)
            
            # Sort by vertical position then horizontal
            extracted_elements.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
            
            if should_close:
                img.close()
                
            return extracted_elements
                
        except Exception as e:
            logger.error(f"Layout extraction failed: {e}")
            return []

if __name__ == "__main__":
    # Test block
    import sys
    import json
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        extractor = DocstrangeLayoutExtractor()
        layout = extractor.extract_layout(image_path)
        print(json.dumps(layout, indent=2))
    else:
        print("Usage: python docstrange_layout_extractor.py <image_path>")
