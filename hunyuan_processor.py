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
    def __init__(self, model_id="tencent/HunyuanOCR", device="cuda", max_new_tokens=1536):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = max_new_tokens
        
        # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx/40xx)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        
        try:
            # Suppress transformers logging
            import os
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
            self.model = HunYuanVLForConditionalGeneration.from_pretrained(
                model_id, 
                attn_implementation="sdpa",  # Use SDPA for faster attention
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.model.eval()  # Set to inference mode
        except Exception as e:
            logger.error(f"Failed to load HunyuanOCR model: {e}")
            raise

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text (markdown) from a PIL Image using HunyuanOCR.
        """
        try:
            logger.debug(f"Processing image of size: {image.size}")
            
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
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                
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

    def extract_text_batch(self, images):
        """
        Extracts text (markdown) from multiple PIL Images using HunyuanOCR in true batch mode.
        
        Args:
            images: list of PIL.Image
            
        Returns:
            list of strings (markdown for each page)
        """
        if not images:
            return []
        
        try:
            logger.debug(f"Processing batch of {len(images)} images with true batching")
            
            # Prepare the prompt text
            prompt_text = "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
            
            # Create messages for each image
            messages_batch = []
            for img in images:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                messages_batch.append(messages)
            
            # Apply chat template to each message set
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
            
            # Process all images together with padding
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            
            # Get input_ids for trimming
            if "input_ids" in inputs:
                input_ids = inputs["input_ids"]
            else:
                logger.warning("input_ids not found in inputs, using fallback")
                input_ids = inputs.get("inputs", inputs["input_ids"])
            
            # Trim input tokens from generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]
            
            # Batch decode all outputs
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Clean repeated substrings in each output
            cleaned_texts = [clean_repeated_substrings(text) for text in output_texts]
            
            return cleaned_texts
            
        except (IndexError, RuntimeError, ValueError) as e:
            # Known issue with HunyuanVLProcessor batching in some transformers versions
            logger.warning(f"Batch processing failed (likely transformers bug): {e}. Falling back to sequential processing.")
            
            # Fall back to sequential processing
            results = []
            for img in images:
                try:
                    result = self.extract_text(img)
                    results.append(result)
                except Exception as inner_e:
                    logger.error(f"Error processing single image during fallback: {inner_e}")
                    results.append("")
            return results
            
        except Exception as e:
            logger.exception(f"Unexpected error during HunyuanOCR batch inference: {e}")
            # Fall back to sequential processing
            results = []
            for img in images:
                try:
                    result = self.extract_text(img)
                    results.append(result)
                except Exception as inner_e:
                    logger.error(f"Error processing single image during fallback: {inner_e}")
                    results.append("")
            return results

if __name__ == "__main__":
    # Simple test
    try:
        proc = HunyuanProcessor()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Model loading failed: {e}")
