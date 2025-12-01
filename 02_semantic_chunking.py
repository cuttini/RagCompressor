import json
import re
from pathlib import Path
from typing import List, Dict

from transformers import AutoTokenizer

# Adjust imports based on your actual file structure
from config import MARKDOWN_DIR, CHUNKS_PATH, TOKENIZER_NAME, MAX_TOKENS_PER_CHUNK, MIN_WORDS_PER_CHUNK

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_long_text(text: str, max_tokens: int) -> List[str]:
    """Splits text into sub-paragraphs to respect max_tokens."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        candidate = (current + "\n\n" + p).strip() if current else p.strip()
        if not candidate:
            continue

        if count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if not current:
                sentences = re.split(r"(?<=[\.\?\!])\s+", p)
                sent_buf = ""
                for s in sentences:
                    cand2 = (sent_buf + " " + s).strip() if sent_buf else s.strip()
                    if count_tokens(cand2) <= max_tokens:
                        sent_buf = cand2
                    else:
                        if sent_buf:
                            chunks.append(sent_buf)
                        sent_buf = s.strip()
                if sent_buf:
                    chunks.append(sent_buf)
            else:
                chunks.append(current)
                current = p.strip()

    if current:
        chunks.append(current)
    return chunks

def chunk_markdown(md_text: str, source_file: str) -> List[Dict]:
    """Splits markdown into logical chunks, merging TOC into a single block."""
    
    # 1. Split by Level 2 headers, keeping the delimiters
    parts = re.split(r"(^## .*$)", md_text, flags=re.MULTILINE)
    
    chunks: List[Dict] = []
    current_title = "Introduzione"
    current_body = ""
    in_toc = False

    def flush(title: str, body: str):
        body = body.strip()
        if not body:
            return

        # FORCE TOC to stay as one chunk (do not split by tokens)
        if title.startswith("TOC -"):
            chunks.append({
                "source": source_file,
                "title": title.strip(),
                "content": f"{title.strip()}\n\n{body}"
            })
            return

        # Standard splitting for normal content
        for sub in split_long_text(body, MAX_TOKENS_PER_CHUNK):
            if len(sub.split()) < MIN_WORDS_PER_CHUNK:
                continue
            chunks.append({
                "source": source_file,
                "title": title.strip(),
                "content": f"{title.strip()}\n\n{sub.strip()}"
            })

    def is_toc_entry(header_text: str) -> bool:
        h = header_text.strip()
        # Case A: Explicit TOC start
        if "INDICE" in h.upper():
            return True
        # Case B: Ends with digits
        if re.search(r"\d+$", h):
            return True
        # Case C: Contains dots/leaders
        if re.search(r"[\.·]{2,}", h):
            return True
        # Case D: Implicit (starts with number)
        if re.match(r"^##\s*\d+\.", h):
            return True
        # Case E: Structural headers (Sezione, Parte, Capitolo, Appendice)
        if re.search(r"\b(Sezione|Parte|Capitolo|Appendice)\b", h, re.IGNORECASE):
            return True
        return False

    def body_has_toc_lines(body_text: str) -> bool:
        # Check if body contains lines that look like TOC entries
        lines = body_text.strip().split('\n')
        for line in lines:
            if re.search(r"(?:[\.·]{2,}|\s+pag\.|\s*»\s*)\s*\d+$", line.strip(), re.IGNORECASE):
                return True
        return False

    def clean_toc_line(text: str) -> str:
        """Removes markdown noise from TOC lines."""
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Replace dots/middle-dots/»/pag. with space
            line = re.sub(r"(?:[\.·]{2,}|\s*»\s*|\s+pag\.)", " ", line, flags=re.IGNORECASE)
            # Normalize whitespace
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    # Handle Prelude (text before first ##)
    if len(parts) > 0:
        prelude = parts[0].strip()
        if prelude:
            if "# INDICE" in prelude.upper():
                 in_toc = True
                 current_title = f"TOC - {source_file}"
                 current_body = clean_toc_line(prelude) + "\n"
            else:
                 current_body = prelude

    # Iterate in pairs: Header, Body
    # parts[0] is prelude. parts[1] is header1, parts[2] is body1, etc.
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i+1] if i+1 < len(parts) else ""
        
        header_stripped = header.strip()
        
        # Skip "Page X" headers entirely
        if re.match(r"^## Page \d+$", header_stripped):
            # Check for INDICE in this body too!
            if "# INDICE" in body.upper() and not in_toc:
                 if current_title != f"TOC - {source_file}":
                    flush(current_title, current_body)
                 
                 in_toc = True
                 current_title = f"TOC - {source_file}"
                 current_body = "" # Start fresh TOC body
                 
                 # Clean body
                 body = clean_toc_line(body)
                 current_body += body + "\n"
                 continue

            # Append body to current_body (cleaning if in TOC)
            if in_toc:
                 body = clean_toc_line(body)
            current_body += body + "\n"
            continue

        # --- TOC DETECTION ---
        # 1. Explicit Start
        has_indice_in_body = "# INDICE" in body.upper()

        if re.match(r"^##\s*Indice", header_stripped, re.IGNORECASE) or (has_indice_in_body and not in_toc):
            if current_title != f"TOC - {source_file}":
                flush(current_title, current_body)
                current_title = f"TOC - {source_file}"
                current_body = ""
            
            in_toc = True
            
            # If header was "## Indice", we skip it (it's just a title). 
            # If body had "# INDICE", we keep the body content (cleaned).
            if in_toc:
                body = clean_toc_line(body)
            current_body += body + "\n"
            continue

        # --- INSIDE TOC LOGIC ---
        if in_toc:
            # Look-Ahead: Check Header OR Body
            is_header_toc = is_toc_entry(header_stripped)
            is_body_toc = body_has_toc_lines(body)
            
            # Noise Check (Italics)
            is_noise = re.match(r"^##\s*\*.*\*$", header_stripped)

            if is_header_toc or is_body_toc or is_noise:
                # It is part of TOC (or noise we want to skip/merge)
                
                if not is_noise:
                    # Clean and append Header
                    clean_header = clean_toc_line(header_stripped.replace("##", "").strip())
                    current_body += f"{clean_header}\n"
                
                # Clean and append Body
                clean_body = clean_toc_line(body)
                current_body += f"{clean_body}\n"
                continue
            else:
                # Not a TOC entry. TOC ended.
                in_toc = False
                # Fall through to standard flush

        # --- STANDARD FLUSH ---
        flush(current_title, current_body)
        current_title = header_stripped
        current_body = body + "\n"

    # Final flush
    flush(current_title, current_body)
    return chunks

def process_all_markdown(md_dir: Path, out_path: Path) -> None:
    all_chunks: List[Dict] = []
    for md_file in sorted(md_dir.glob("*.md")):
        print(f"[chunking] Processo {md_file.name}")
        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()
        chunks = chunk_markdown(md_text, source_file=md_file.name)
        all_chunks.extend(chunks)

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, ch in enumerate(all_chunks):
            ch_out = {"id": idx, **ch}
            f.write(json.dumps(ch_out, ensure_ascii=False) + "\n")
    print(f"[chunking] Salvati {len(all_chunks)} chunk in {out_path}")

if __name__ == "__main__":
    process_all_markdown(MARKDOWN_DIR, CHUNKS_PATH)