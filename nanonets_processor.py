import logging
import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from config import NANONETS_MODEL_ID

logger = logging.getLogger(__name__)

class NanonetsProcessor:
    """Processor using Nanonets OCR model (VLM)."""
    
    def __init__(self, model_id: str = NANONETS_MODEL_ID):
        """Initialize the Nanonets Processor."""
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Nanonets Processor with model: {model_id} on {self.device}")
        
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            logger.info("Nanonets model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Nanonets model: {e}")
            raise

    def extract_text(self, image: Image.Image) -> str:
        """Extract plain text from image."""
        prompt = "Extract the text from the above document as if you were reading it naturally."
        return self._generate(image, prompt)

    def extract_layout_json(self, image: Image.Image) -> Dict[str, Any]:
        """Extract structured layout data (JSON) from image."""
        prompt = """Extract all information from the above document and return it as a valid JSON object.
Instructions:
- The output should be a single JSON object.
- Keys should be meaningful field names.
- If multiple similar blocks (like invoice items or line items), return a list of JSON objects under a key.
- Use strings for all values.
- Wrap page numbers using: "page_number": "1"
- Wrap watermarks using: "watermark": "CONFIDENTIAL"
- Use ☐ and ☑ for checkboxes.
"""
        json_text = self._generate(image, prompt, max_new_tokens=2048)
        return self._parse_json(json_text)

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> str:
        """Internal generation method."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
            ]
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            
            return output_text
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from generated text."""
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON block
            try:
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except:
                pass
            
            # Fallback
            return {"raw_text": text, "error": "Failed to parse JSON"}
