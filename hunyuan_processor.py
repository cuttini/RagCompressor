import torch
from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]  

    return text

class HunyuanProcessor:
    def __init__(self, model_id="tencent/HunyuanOCR", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing HunyuanProcessor with model {model_id} on {self.device}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                model_id, 
                attn_implementation="eager",
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            # self.model.eval() # device_map handles moving to device
        except Exception as e:
            logger.error(f"Failed to load HunyuanOCR model: {e}")
            raise

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text (markdown) from a PIL Image using HunyuanOCR.
        """
        try:
            logger.info(f"Processing image of size: {image.size}")
            torch.cuda.empty_cache()
            
            # Prepare inputs with the specific OCR prompt
            prompt_text = "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = self.processor(
                text=[text_input],
                images=image,
                padding=True,
                return_tensors="pt",
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
                
            # Decode
            # We need to trim input_ids from generated_ids
            input_ids_len = inputs["input_ids"].shape[1]
            generated_ids_trimmed = generated_ids[:, input_ids_len:]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return clean_repeated_substrings(output_text)
            
        except Exception as e:
            logger.exception(f"Error during HunyuanOCR inference: {e}")
            return ""

if __name__ == "__main__":
    # Simple test
    try:
        proc = HunyuanProcessor()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model loading failed: {e}")
