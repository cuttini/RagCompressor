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

def visualize_chunks(pdf_path: str, layout_data: List[Dict], chunks: List[Dict], output_dir: Path, max_pages: int = None):
    """
    Visualize semantic chunks on PDF pages using improved matching algorithm.
    
    Strategy:
    1. For chunks with "## Page N" markers, draw ALL blocks on those pages
    2. For other chunks, use normalized text matching
    
    Args:
        pdf_path: Path to the original PDF.
        layout_data: List of page objects, each containing 'blocks' with 'text' and 'bbox'.
        chunks: List of semantic chunks, each containing 'content'.
        output_dir: Directory to save visualized images.
        max_pages: Optional limit on number of pages to process.
    """
    import re
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert PDF to images
    print(f"[visualize] Converting PDF {pdf_path} to images...")
    images = convert_from_path(pdf_path, last_page=max_pages)
    
    # Limit layout_data to max_pages if specified
    if max_pages:
        layout_data = layout_data[:max_pages]
    
    # Create a map of page_idx -> list of boxes to draw
    page_boxes = {i: [] for i in range(len(images))}
    
    print("[visualize] Mapping chunks to layout...")
    
    # Statistics
    total_blocks = sum(len(page.get('blocks', [])) for page in layout_data)
    matched_blocks_count = 0
    chunks_with_page_info = 0
    chunks_without_page_info = 0
    
    def normalize_text(text: str) -> str:
        """Normalize text for comparison by removing extra whitespace and converting to lowercase."""
        return ' '.join(text.lower().split())
    
    for chunk in chunks:
        chunk_id = chunk.get('id', '?')
        chunk_content = chunk.get('content', '')
        
        # Strategy 1: Extract page numbers from chunk content
        page_matches = re.findall(r'## Page (\d+)', chunk_content)
        
        if page_matches:
            # This chunk has explicit page markers
            chunks_with_page_info += 1
            for page_num_str in page_matches:
                page_num = int(page_num_str)
                page_idx = page_num - 1  # Convert to 0-indexed
                
                # Draw ALL blocks on this page for this chunk
                if page_idx < len(layout_data):
                    page = layout_data[page_idx]
                    for block in page.get('blocks', []):
                        bbox = block.get('bbox')
                        if bbox:
                            page_boxes[page_idx].append({
                                'bbox': bbox,
                                'label': f"C{chunk_id}"
                            })
                            matched_blocks_count += 1
        else:
            # Strategy 2: No explicit page info, use fuzzy text matching
            chunks_without_page_info += 1
            chunk_normalized = normalize_text(chunk_content)
            chunk_words = set(chunk_normalized.split())
            
            for p_idx, page in enumerate(layout_data):
                for block in page.get('blocks', []):
                    block_text = block.get('text', '')
                    block_normalized = normalize_text(block_text)
                    block_words = set(block_normalized.split())
                    
                    if not block_words:
                        continue

                    # Calculate overlap
                    intersection = chunk_words.intersection(block_words)
                    overlap_ratio = len(intersection) / len(block_words)
                    
                    # Match if significant overlap
                    # We use a threshold of 0.6 (60% of block words found in chunk)
                    if len(block_text) > 10 and overlap_ratio >= 0.6:
                        bbox = block.get('bbox')
                        if bbox:
                            page_boxes[p_idx].append({
                                'bbox': bbox,
                                'label': f"C{chunk_id}"
                            })
                            matched_blocks_count += 1
    
    # Print statistics
    print(f"[visualize] Statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Chunks with page info: {chunks_with_page_info}")
    print(f"  Chunks without page info: {chunks_without_page_info}")
    print(f"  Total layout blocks: {total_blocks}")
    print(f"  Matched blocks: {matched_blocks_count}")
    if total_blocks > 0:
        match_rate = (matched_blocks_count / total_blocks) * 100
        print(f"  Match rate: {match_rate:.1f}%")
    
    # Draw and save
    print("[visualize] Drawing and saving pages...")
    pages_saved = 0
    for i, img in enumerate(images):
        if i in page_boxes and page_boxes[i]:
            draw_boxes_on_page(img, page_boxes[i])
            out_path = output_dir / f"page_{i+1}_chunks.jpg"
            img.save(out_path)
            print(f"Saved {out_path}")
            pages_saved += 1
    
    print(f"[visualize] Saved {pages_saved} pages with visualizations")
