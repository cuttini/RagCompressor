import json
import re
from pathlib import Path
from typing import List, Dict

from transformers import AutoTokenizer

# Adjust imports based on your actual file structure
from config import (MARKDOWN_DIR, CHUNKS_PATH, TOKENIZER_NAME, 
                    MAX_TOKENS_PER_CHUNK, MIN_WORDS_PER_CHUNK, MIN_CONTENT_CHARS,
                    ALLOW_TABLE_OVERFLOW, MAX_TABLE_TOKENS)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def extract_html_tables(text: str) -> List[tuple]:
    """
    Identifica tutte le tabelle HTML complete nel testo.
    
    Args:
        text: Testo che può contenere tabelle HTML
        
    Returns:
        Lista di tuple (start_pos, end_pos, table_html)
    """
    tables = []
    pattern = r'<table[^>]*>(.*?)</table>'
    
    for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        start_pos = match.start()
        end_pos = match.end()
        table_html = match.group(0)
        tables.append((start_pos, end_pos, table_html))
    
    return tables

def contains_complete_table(text: str, tables: List[tuple]) -> bool:
    """
    Verifica se il testo contiene una tabella HTML completa.
    
    Args:
        text: Testo da verificare
        tables: Lista di tabelle estratte dal testo originale
        
    Returns:
        True se il testo contiene almeno una tabella completa
    """
    # Cerca tag di apertura e chiusura tabella
    has_table_start = '<table' in text.lower()
    has_table_end = '</table>' in text.lower()
    
    return has_table_start and has_table_end

def repair_malformed_tables(text: str) -> str:
    """
    Ripara tabelle HTML malformate dall'OCR chiudendo tag aperti.
    
    L'OCR può generare tag <table> senza corrispondenti </table>.
    Questa funzione identifica le sezioni problematiche e le ripara.
    
    Args:
        text: Testo con possibili tabelle malformate
        
    Returns:
        Testo con tabelle riparate
    """
    # Trova tutte le posizioni di <table> e </table>
    table_opens = [(m.start(), m.group()) for m in re.finditer(r'<table[^>]*>', text, re.IGNORECASE)]
    table_closes = [m.start() for m in re.finditer(r'</table>', text, re.IGNORECASE)]
    
    if len(table_opens) == len(table_closes):
        # Numero uguale, probabilmente tutto OK
        return text
    
    # Ricostruiamo il testo riparando le tabelle
    repaired = text
    insertions = []  # Lista di (posizione, testo_da_inserire)
    
    # Stack-based matching per trovare tabelle non chiuse
    open_stack = []
    close_idx = 0
    
    for open_pos, open_tag in table_opens:
        # Trova il prossimo </table> dopo questa apertura
        next_close = None
        for close_pos in table_closes[close_idx:]:
            if close_pos > open_pos:
                next_close = close_pos
                close_idx = table_closes.index(close_pos) + 1
                break
        
        if next_close is None:
            # Tag non chiuso! Cerchiamo dove dovrebbe finire
            # Euristica: chiudi prima del prossimo ## header o del prossimo <table>
            search_start = open_pos + len(open_tag)
            
            # Cerca il prossimo header o tabella
            next_section = re.search(r'\n## |<table', text[search_start:], re.IGNORECASE)
            
            if next_section:
                close_position = search_start + next_section.start()
            else:
                # Fine del testo
                close_position = len(text)
            
            # Inserisci </table> prima della nuova sezione
            insertions.append((close_position, '\n</table>\n'))
    
    # Applica le inserzioni in ordine inverso per non invalidare le posizioni
    for pos, tag in sorted(insertions, reverse=True):
        repaired = repaired[:pos] + tag + repaired[pos:]
    
    if insertions:
        print(f"[INFO] Riparate {len(insertions)} tabelle malformate")
    
    return repaired

def get_actual_content_length(text: str) -> int:
    """
    Calculate actual content length excluding metadata headers and markdown formatting.
    
    Args:
        text: Full chunk text including metadata
        
    Returns:
        Length of actual text content
    """
    # Remove metadata header [CONTESTO]...[/CONTESTO]
    content = re.sub(r'\[CONTESTO\].*?\[/CONTESTO\]', '', text, flags=re.DOTALL)
    # Remove markdown headers (##, ###, etc.)
    content = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
    # Remove horizontal rules
    content = re.sub(r'^---+$', '', content, flags=re.MULTILINE)
    # Strip whitespace
    content = content.strip()
    return len(content)

def extract_metadata(filename: str) -> Dict[str, str]:
    """
    Estrae metadati strutturati dal nome del file markdown.
    
    Args:
        filename: Nome del file (es. "Manuale_Operativo_IVA_2025_Digitale_PS.md")
    
    Returns:
        Dict con chiavi: 'year', 'topic', 'source_cleaned'
    """
    # 1. ESTRAZIONE ANNO con Regex
    year_match = re.search(r"(20\d{2})", filename)
    year = year_match.group(1) if year_match else "N/A"
    
    # 2. DEDUZIONE AMBITO basato su keyword
    filename_upper = filename.upper()
    
    if "IVA" in filename_upper:
        topic = "IVA / Imposta Valore Aggiunto"
    elif "AMMORTAMENT" in filename_upper:  # Copre "Ammortamenti", "Ammortamento"
        topic = "Ammortamenti / Cespiti"
    elif "CEDOLINO" in filename_upper or "PAGA" in filename_upper or "PAGHE" in filename_upper:
        topic = "Lavoro / Paghe"
    elif "ISA" in filename_upper:
        topic = "ISA / Indici Affidabilità"
    elif "CU" in filename_upper and "CERTIFICAZIONE" in filename_upper:
        topic = "Certificazione Unica"
    elif "BILANCIO" in filename_upper:
        topic = "Bilancio / Contabilità"
    elif "ANTIRICIC" in filename_upper:
        topic = "Antiriciclaggio"
    elif "INPS" in filename_upper or "ENASARCO" in filename_upper or "PREVIDEN" in filename_upper:
        topic = "Previdenza / Contributi"
    elif "IMMOBILIAR" in filename_upper or "LOCAZION" in filename_upper or "IMU" in filename_upper:
        topic = "Immobili / Locazioni"
    elif "TERZO SETTORE" in filename_upper or "ETS" in filename_upper:
        topic = "Enti Terzo Settore"
    elif "PRIVACY" in filename_upper or "GDPR" in filename_upper:
        topic = "Privacy / GDPR"
    else:
        topic = "Fisco Generico"
    
    # 3. PULIZIA NOME FILE per campo FONTE
    # Rimuove estensione .md
    source_cleaned = filename.replace(".md", "")
    # Sostituisce underscore con spazi
    source_cleaned = source_cleaned.replace("_", " ")
    # Rimuove suffissi tecnici comuni
    source_cleaned = re.sub(r"\s+(Digitale|PS|UO)\s*$", "", source_cleaned, flags=re.IGNORECASE)
    source_cleaned = source_cleaned.strip()
    
    return {
        "year": year,
        "topic": topic,
        "source_cleaned": source_cleaned
    }

def split_long_text(text: str, max_tokens: int) -> List[str]:
    """
    Splits text into sub-paragraphs to respect max_tokens.
    TABLE-AWARE: Preserves complete HTML tables without fragmentation.
    
    Strategy:
    1. Extract all complete tables from text
    2. Replace tables with placeholders 
    3. Split remaining text by paragraphs normally
    4. Restore tables as dedicated chunks
    """
    # 1. Extract complete tables and replace with placeholders
    tables = extract_html_tables(text)
    
    if not tables:
        # No tables, use original logic
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
    
    # 2. Table-aware splitting: replace tables with placeholders
    modified_text = text
    table_placeholders = {}
    
    for idx, (start_pos, end_pos, table_html) in enumerate(tables):
        placeholder = f"__TABLE_PLACEHOLDER_{idx}__"
        table_placeholders[placeholder] = table_html
    
    # Replace tables in reverse order to preserve positions
    for idx in range(len(tables) - 1, -1, -1):
        start_pos, end_pos, table_html = tables[idx]
        placeholder = f"__TABLE_PLACEHOLDER_{idx}__"
        modified_text = modified_text[:start_pos] + placeholder + modified_text[end_pos:]
    
    # 3. Split modified text normally
    paragraphs = re.split(r"\n\s*\n", modified_text)
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
    
    # 4. Restore placeholders with actual tables
    final_chunks = []
    for chunk in chunks:
        # Check if chunk contains table placeholders
        has_placeholder = any(ph in chunk for ph in table_placeholders.keys())
        
        if has_placeholder:
            # Restore all tables in this chunk
            restored = chunk
            for placeholder, table_html in table_placeholders.items():
                if placeholder in restored:
                    restored = restored.replace(placeholder, table_html)
            
            # Check if restored chunk is too large
            restored_tokens = count_tokens(restored)
            if ALLOW_TABLE_OVERFLOW and restored_tokens <= MAX_TABLE_TOKENS:
                final_chunks.append(restored)
            elif restored_tokens <= max_tokens:
                final_chunks.append(restored)
            else:
                # Table too large, force split
                print(f"[WARNING] Chunk con tabella troppo grande ({restored_tokens} tokens), split forzato")
                # Try to split by keeping tables separate
                for placeholder, table_html in table_placeholders.items():
                    if placeholder in chunk:
                        # Add text before table
                        before_table = restored.split(table_html)[0].strip()
                        if before_table and not before_table.startswith('__TABLE_'):
                            final_chunks.append(before_table)
                        # Add table separately
                        final_chunks.append(table_html)
                        # Add text after table (if any)
                        parts = restored.split(table_html)
                        if len(parts) > 1:
                            after_table = parts[1].strip()
                            if after_table and not '__TABLE_' in after_table:
                                final_chunks.append(after_table)
                        break
        else:
            final_chunks.append(chunk)
    
    return final_chunks

def chunk_markdown(md_text: str, source_file: str) -> List[Dict]:
    """
    Splits markdown into logical chunks, merging TOC into a single block.
    Ogni chunk include un header di metadati strutturati per il RAG.
    TABLE-AWARE: Ripara e preserva tabelle HTML complete.
    """
    
    # REPAIR MALFORMED HTML TABLES FIRST
    md_text = repair_malformed_tables(md_text)
    
    # PROTECT TABLES FROM SECTION SPLITTING
    # Extract all tables and replace with placeholders BEFORE splitting by sections
    tables_in_document = extract_html_tables(md_text)
    table_placeholders_global = {}
    
    for idx, (start_pos, end_pos, table_html) in enumerate(tables_in_document):
        placeholder = f"__GLOBAL_TABLE_PH_{idx}__"
        table_placeholders_global[placeholder] = table_html
    
    # Replace tables in reverse order to preserve positions
    modified_md_text = md_text
    for idx in range(len(tables_in_document) - 1, -1, -1):
        start_pos, end_pos, table_html = tables_in_document[idx]
        placeholder = f"__GLOBAL_TABLE_PH_{idx}__"
        modified_md_text = modified_md_text[:start_pos] + placeholder + modified_md_text[end_pos:]
    
    # Estrai metadati dal nome del file
    metadata = extract_metadata(source_file)
    
    # 1. Split by Level 2 headers, keeping the delimiters (now on modified text)
    parts = re.split(r"(^## .*$)", modified_md_text, flags=re.MULTILINE)
    
    chunks: List[Dict] = []
    current_title = "Introduzione"
    current_section_title = "Introduzione"  # Track the actual section (non-Page header)
    current_body = ""
    current_page = "N/A"  # Traccia la pagina corrente
    in_toc = False

    def clean_title(title: str) -> str:
        """Remove markdown characters from title to make it plain text."""
        # Remove ## headers
        title = re.sub(r'^#+\s*', '', title)
        # Remove bold/italic markers
        title = re.sub(r'[*_]+', '', title)
        return title.strip()

    def flush(title: str, body: str, page_number: str = "N/A"):
        body = body.strip()
        if not body:
            return

        # REMOVE TOC: Do not create chunks for TOC sections
        if title.startswith("TOC -"):
            return
        
        # REMOVE BOILERPLATE: Skip sections with non-relevant administrative content
        section_title_lower = clean_title(title).lower()
        boilerplate_patterns = [
            r'^introduzione$',
            r'istruzioni?\s+(per|di)\s+attivare',
            r'e-book\s+(on-?line|gratuito)',
            r'^come\s+usare',
            r'^avvertenze?$',
            r'^note\s+(editoriali?|legali?)',
            r'^copyright',
            r'^colophon',
            r'^responsabile\s+editoria',
            r'^centro\s+studi',
            r'^autori?$',
            r'^ringraziamenti?$',
            r'^prefazione$',
            r'^presentazione$',
            r'codice\s+(di\s+)?attivazione',
            r'registra(zione|ti)',
            r'^indirizzo\s+web',
        ]
        
        for pattern in boilerplate_patterns:
            if re.search(pattern, section_title_lower, re.IGNORECASE):
                return
        
        # ALSO REMOVE: Check if body looks like a TOC (even if title doesn't indicate it)
        # This catches mini-TOCs within sections
        if body_has_toc_lines(body):
            # If most of the content looks like TOC, skip it
            lines = body.strip().split('\n')
            toc_line_count = 0
            for line in lines:
                # Only count as TOC if pattern is at END of line (not citations in middle)
                if re.search(r"(?:[\\.·]{2,}|\s+pag\.|\s*»\s*)\s*\d+$", line.strip(), re.IGNORECASE):
                    toc_line_count += 1
            # If more than 30% of lines are TOC-like, skip this chunk
            if len(lines) > 0 and (toc_line_count / len(lines)) > 0.3:
                return

        # Prepara il titolo pulito per la sezione
        section_title = clean_title(title)

        # RESTORE GLOBAL TABLE PLACEHOLDERS in body before splitting
        body_with_tables = body
        for placeholder, table_html in table_placeholders_global.items():
            if placeholder in body_with_tables:
                body_with_tables = body_with_tables.replace(placeholder, table_html)

        # Standard splitting for normal content (now with restored tables)
        for sub in split_long_text(body_with_tables, MAX_TOKENS_PER_CHUNK):
            if len(sub.split()) < MIN_WORDS_PER_CHUNK:
                continue
            
            # INIETTA HEADER DI METADATI STRUTTURATI
            metadata_header = f"""[CONTESTO]
FONTE: {metadata['source_cleaned']}
ANNO: {metadata['year']}
AMBITO: {metadata['topic']}
SEZIONE: {section_title}
PAGINA: {page_number}
[/CONTESTO]

"""
            
            # Assembla il contenuto finale con header + titolo + testo
            final_content = f"{metadata_header}{title.strip()}\n\n{sub.strip()}"
            
            # Validate actual content length (excluding metadata and formatting)
            actual_content_len = get_actual_content_length(final_content)
            if actual_content_len < MIN_CONTENT_CHARS:
                continue  # Skip chunks with insufficient actual content
            
            chunks.append({
                "source": source_file,
                "title": section_title,
                "content": final_content
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
    
    def should_merge_with_previous(prev_text: str, new_text: str, same_section: bool) -> bool:
        """
        Determina se il nuovo testo dovrebbe essere unito al precedente.
        
        Args:
            prev_text: Testo precedente (già accumulato)
            new_text: Nuovo testo da considerare
            same_section: True se siamo nella stessa sezione logica
        
        Returns:
            True se i testi dovrebbero essere uniti
        """
        # Se non siamo nella stessa sezione, non unire
        if not same_section:
            return False
        
        # Se il testo precedente è vuoto, non c'è nulla da unire
        if not prev_text.strip():
            return False
        
        # Se il precedente termina con punteggiatura forte, non unire
        if re.search(r"[.!?:]\s*$", prev_text.rstrip()):
            return False
        
        # Se il nuovo testo inizia con minuscola o parole di continuazione
        new_stripped = new_text.lstrip()
        if new_stripped and (
            new_stripped[0].islower() or
            re.match(r"^(che|di cui|inoltre|tuttavia|quindi|pertanto|il quale|la quale|dei|delle|degli|dal|dalla)", 
                     new_stripped, re.IGNORECASE)
        ):
            return True
        
        # Default: unisci se non c'è punteggiatura forte
        return True

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
        if re.match(r"^## Page (\d+)$", header_stripped):
            # Estrai il numero di pagina
            page_match = re.match(r"^## Page (\d+)$", header_stripped)
            if page_match:
                current_page = page_match.group(1)
            
            # Check for INDICE in this body too!
            if "# INDICE" in body.upper() and not in_toc:
                 if current_title != f"TOC - {source_file}":
                    flush(current_title, current_body, current_page)
                 
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
            else:
                # IMPROVED MERGE LOGIC for cross-page paragraphs
                # Use section context to decide whether to merge
                if should_merge_with_previous(current_body, body, True):
                    # Merge con uno spazio, rimuovendo whitespace superfluo
                    current_body = current_body.rstrip() + " " + body.lstrip()
                else:
                    # Mantieni separato (preservando paragrafi)
                    current_body = current_body.rstrip() + "\n\n" + body.lstrip()
            continue

        # --- TOC DETECTION ---
        # 1. Explicit Start
        has_indice_in_body = "# INDICE" in body.upper()

        if re.match(r"^##\s*Indice", header_stripped, re.IGNORECASE) or (has_indice_in_body and not in_toc):
            if current_title != f"TOC - {source_file}":
                flush(current_title, current_body, current_page)
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
        flush(current_title, current_body, current_page)
        current_section_title = header_stripped  # Track the real section title
        current_title = header_stripped
        current_body = body + "\n"

    # Final flush
    flush(current_title, current_body, current_page)
    return chunks

def process_all_markdown(md_dir: Path, out_path: Path, max_files: int = None) -> None:
    all_chunks: List[Dict] = []
    md_files = sorted(md_dir.glob("*.md"))
    
    # Limit files if max_files is specified
    if max_files:
        md_files = md_files[:max_files]
        print(f"[chunking] Limitato a {max_files} file per test")
    
    for md_file in md_files:
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic chunking of markdown files")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limita il numero di file da processare (utile per test)"
    )
    args = parser.parse_args()
    
    process_all_markdown(MARKDOWN_DIR, CHUNKS_PATH, max_files=args.max_files)