import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

def draw_boxes_on_page(image: Image.Image, boxes: List[Dict], color="red", width=3) -> Image.Image:
    """Draw bounding boxes on a PIL image."""
    draw = ImageDraw.Draw(image)
    for box_info in boxes:
        bbox = box_info.get('bbox')
        if not bbox:
            continue
        
        # bbox is [l, t, r, b]
        draw.rectangle(bbox, outline=color, width=width)
        
        # Optional: Draw label or ID
        label = box_info.get('label')
        if label:
            # simple text drawing
            try:
                draw.text((bbox[0], bbox[1] - 10), str(label), fill=color)
            except:
                pass
    return image

def visualize_chunks(pdf_path: str, layout_data: List[Dict], chunks: List[Dict], output_dir: Path):
    """
    Visualize semantic chunks on PDF pages.
    
    Args:
        pdf_path: Path to the original PDF.
        layout_data: List of page objects, each containing 'blocks' with 'text' and 'bbox'.
        chunks: List of semantic chunks, each containing 'content'.
        output_dir: Directory to save visualized images.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert PDF to images
    print(f"[visualize] Converting PDF {pdf_path} to images...")
    images = convert_from_path(pdf_path)
    
    # Map chunks to layout blocks
    # This is a heuristic mapping based on text overlap
    
    # Create a map of page_idx -> list of boxes to draw
    page_boxes = {i: [] for i in range(len(images))}
    
    print("[visualize] Mapping chunks to layout...")
    for chunk_idx, chunk in enumerate(chunks):
        chunk_text = chunk.get('content', '')
        # Normalize chunk text (remove markdown headers if needed, or keep them if layout has them)
        # The layout blocks usually have headers as separate blocks.
        
        # We need to find which blocks in layout_data correspond to this chunk.
        # Simple approach: Iterate all blocks in order, try to match sequence.
        
        matched_blocks = []
        
        # Flatten all blocks with page info
        all_blocks = []
        for p_idx, page in enumerate(layout_data):
            for block in page.get('blocks', []):
                block['page_idx'] = p_idx
                all_blocks.append(block)
        
        # This is O(N*M) where N=chunks, M=blocks. 
        # Better: Maintain a cursor in all_blocks since chunks are sequential (mostly).
        
        # Let's try a sliding window or just sequential search from last position.
        # Assuming chunks appear in order.
        
        # TODO: Implement robust text matching. 
        # For now, let's just assume we can find the text.
        pass 

    # Since implementing robust text alignment is complex, 
    # and we are modifying 01_pdf_to_markdown.py anyway,
    # we can try to carry over the 'block_ids' into the markdown generation 
    # and then into the chunks?
    # That would require modifying 02_semantic_chunking.py too.
    
    # Alternative: Just visualize ALL layout blocks for now to verify we have them.
    # The user asked to "box the semantic chunks".
    
    # Let's implement a simple text containment check.
    # For each chunk, find blocks whose text is contained in the chunk.
    
    for chunk in chunks:
        chunk_content = chunk.get('content', '')
        # Split chunk into lines or sentences to match against blocks
        # Blocks are usually paragraphs or headers.
        
        # Heuristic: If a block's text is a substring of the chunk, it belongs to the chunk.
        # (With some length threshold to avoid noise)
        
        for p_idx, page in enumerate(layout_data):
            for block in page.get('blocks', []):
                block_text = block.get('text', '')
                if len(block_text) > 10 and block_text in chunk_content:
                    # Match!
                    page_boxes[p_idx].append({
                        'bbox': block.get('bbox'),
                        'label': f"Chunk {chunk.get('id', '?')}"
                    })
    
    # Draw and save
    print("[visualize] Drawing and saving pages...")
    for i, img in enumerate(images):
        if i in page_boxes and page_boxes[i]:
            draw_boxes_on_page(img, page_boxes[i])
            out_path = output_dir / f"page_{i+1}_chunks.jpg"
            img.save(out_path)
            print(f"Saved {out_path}")
