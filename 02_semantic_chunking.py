import json
import re
from pathlib import Path
from typing import List, Dict

from transformers import AutoTokenizer

from config import MARKDOWN_DIR, CHUNKS_PATH, TOKENIZER_NAME, MAX_TOKENS_PER_CHUNK, MIN_WORDS_PER_CHUNK

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def split_long_text(text: str, max_tokens: int) -> List[str]:
    """Splitta in sottoparagraﬁ per non sforare max_tokens."""
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
            # se il singolo paragrafo è troppo lungo, split per frasi
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
    """Splitta un markdown in chunk logici basati su ## e max_tokens."""
    # separiamo per sezioni di livello 2
    parts = re.split(r"(^## .*$)", md_text, flags=re.MULTILINE)
    # parts = [prelude, '## titolo1', body1, '## titolo2', body2, ...]
    chunks: List[Dict] = []
    current_title = "Introduzione"
    current_body = ""

    def flush(title: str, body: str):
        body = body.strip()
        if not body:
            return
        # eventuale ulteriore split per token
        for sub in split_long_text(body, MAX_TOKENS_PER_CHUNK):
            if len(sub.split()) < MIN_WORDS_PER_CHUNK:
                continue
            chunks.append(
                {
                    "source": source_file,
                    "title": title.strip(),
                    "content": f"{title.strip()}\n\n{sub.strip()}",
                }
            )

    for part in parts:
        if part.startswith("## "):
            flush(current_title, current_body)
            current_title = part.strip()
            current_body = ""
        else:
            current_body += part

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
