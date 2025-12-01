import json
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from config import CHUNKS_PATH, STAGE1_RAW_PATH, STAGE1_2_PATH, WINDOW_SIZE
from llm_client import QwenClient

SYSTEM_PROMPT = (
    "Sei un esperto fiscalista italiano e docente per l'esame di abilitazione.\n"
    "Riceverai una sequenza di estratti (chunk) da manuali SEAC. "
    "Il tuo compito è generare dataset per il training di un'IA fiscale.\n"
    "Devi prestare massima attenzione a:\n"
    "1. Collegamenti logici tra i chunk (domande multi-hop).\n"
    "2. Aspetti temporali (abrogazioni, entrate in vigore, regimi transitori).\n"
    "3. Accuratezza normativa."
)

USER_TEMPLATE = """
CONTESTO NORMATIVO (Sliding Window di {window_size} chunk):

{context_text}

---

COMPITI (Rispondi ESCLUSIVAMENTE con un JSON valido):

Genera i seguenti tipi di Q&A basandoti sul testo fornito.
Focalizzati principalmente sul CHUNK CENTRALE (Chunk #{center_id}), ma usa i chunk circostanti per contesto, disambiguazione o domande di collegamento.

1. "simple_qas": 2 domande atomiche (definizioni, aliquote, scadenze) basate sul chunk centrale.
2. "complex_qas": 1 domanda complessa (calcolo, scenario pratico).
3. "temporal_qas": 1 domanda (SE APPLICABILE) che riguarda date, entrate in vigore, abrogazioni o validità temporale. Se non ci sono riferimenti temporali rilevanti, lascia la lista vuota.
4. "multi_hop_qas": 1 domanda (SE APPLICABILE) che richiede di collegare informazioni del chunk centrale con quello precedente o successivo.
5. "paraphrase": Una riscrittura discorsiva e completa del contenuto del CHUNK CENTRALE.

Schema JSON atteso:
{{
  "simple_qas": [{{"q": "...", "a": "..."}}, ...],
  "complex_qas": [{{"q": "...", "a": "..."}}],
  "temporal_qas": [{{"q": "...", "a": "..."}}],   // Opzionale, vuoto se non pertinente
  "multi_hop_qas": [{{"q": "...", "a": "..."}}],  // Opzionale, vuoto se non pertinente
  "paraphrase": "..."
}}
"""

def get_sliding_windows(chunks: List[Dict], window_size: int = 3) -> List[Tuple[List[Dict], int]]:
    """
    Genera finestre scorrevoli di chunk.
    Restituisce (window_chunks, center_index_in_window).
    Gestisce i bordi (padding o finestre ridotte).
    Qui usiamo un approccio semplice: per ogni chunk 'i', prendiamo [i-1, i, i+1] se esistono.
    """
    windows = []
    n = len(chunks)
    half = window_size // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = chunks[start:end]
        
        # Calcoliamo l'indice del chunk centrale relativo alla finestra
        # Se siamo all'inizio (i=0), il centro è 0. Se siamo a i=10, start=9, centro è 1.
        center_rel_idx = i - start
        
        windows.append((window, center_rel_idx))
    
    return windows

def format_window_text(window: List[Dict], center_idx: int) -> str:
    out = []
    for idx, ch in enumerate(window):
        marker = " (CHUNK CENTRALE)" if idx == center_idx else f" (Contesto {'Precedente' if idx < center_idx else 'Successivo'})"
        out.append(f"=== CHUNK #{ch['id']}{marker} ===\nFonte: {ch['source']}\nTitolo: {ch['title']}\n\n{ch['content']}\n")
    return "\n".join(out)

def build_records(chunk_center: Dict, llm_output: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Costruisce i record per Stage1 e Stage1_2.
    """
    s1_recs = []
    s1_2_recs = []
    
    text = chunk_center["content"]
    
    # Raccogliamo tutte le QA
    all_qas = []
    for key in ["simple_qas", "complex_qas", "temporal_qas", "multi_hop_qas"]:
        for item in (llm_output.get(key) or []):
            if "q" in item and "a" in item:
                # Aggiungiamo un tag al tipo di domanda per tracciabilità interna (opzionale)
                item["_type"] = key
                all_qas.append(item)

    # Stage 1: QA aggregate
    qa_questions = [x["q"].strip() for x in all_qas]
    qa_answers = [x["a"].strip() for x in all_qas]
    
    if qa_questions:
        s1_recs.append({
            "data_type": "qa",
            "question": qa_questions,
            "answers": qa_answers,
            "docs": [text],
            "pos_index": [0]
        })

    # Stage 1: Paraphrase
    para = (llm_output.get("paraphrase") or "").strip()
    if para:
        s1_recs.append({
            "data_type": "paraphrase",
            "question": "",
            "answers": [para],
            "docs": [text],
            "pos_index": [0]
        })

    # Stage 1_2: Instruction Tuning (1 record per QA)
    for item in all_qas:
        q, a = item["q"].strip(), item["a"].strip()
        if q and a:
            s1_2_recs.append({
                "question": q,
                "docs": [text],
                "gold_answer": a,
                "answer": a,
                "data_type": "qa", # Potremmo differenziare 'temporal' o 'multi_hop' se CLaRa lo supportasse
                "pos_index": [0]
            })

    return s1_recs, s1_2_recs

def main():
    client = QwenClient()
    chunks: List[Dict] = []

    if not CHUNKS_PATH.exists():
        print(f"[ERROR] {CHUNKS_PATH} non trovato. Esegui prima lo step 02.")
        return

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # Ordiniamo per source e id per garantire contiguità corretta
    # Assumiamo che i chunk siano già ordinati o che l'ID rifletta l'ordine
    chunks.sort(key=lambda x: (x["source"], x["id"]))

    windows = get_sliding_windows(chunks, WINDOW_SIZE)
    
    stage1_records: List[Dict] = []
    stage1_2_records: List[Dict] = []

    # Apriamo i file in append mode (o write se vogliamo sovrascrivere all'inizio, ma qui facciamo write per pulizia)
    # Attenzione: se il processo si interrompe e riparte, bisognerebbe gestire il resume.
    # Per ora sovrascriviamo all'inizio e appendiamo man mano.
    
    with open(STAGE1_RAW_PATH, "w", encoding="utf-8") as f1, open(STAGE1_2_PATH, "w", encoding="utf-8") as f2:
        pass # Clear files

    print(f"[info] Processo {len(windows)} finestre (sliding window size={WINDOW_SIZE})...")

    for window, center_idx in tqdm(windows, desc="Generazione Dataset"):
        center_chunk = window[center_idx]
        
        # Costruiamo il prompt con il contesto
        context_text = format_window_text(window, center_idx)
        user_prompt = USER_TEMPLATE.format(
            window_size=len(window),
            context_text=context_text,
            center_id=center_chunk["id"]
        )

        try:
            out = client.json_completion(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            print(f"[WARN] Errore LLM sul chunk {center_chunk['id']}: {e}")
            continue

        s1, s1_2 = build_records(center_chunk, out)
        
        # Scrittura incrementale
        with open(STAGE1_RAW_PATH, "a", encoding="utf-8") as f:
            for rec in s1:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
        with open(STAGE1_2_PATH, "a", encoding="utf-8") as f:
            for rec in s1_2:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[stage1] Output salvato in {STAGE1_RAW_PATH}")
    print(f"[stage1_2] Output salvato in {STAGE1_2_PATH}")

if __name__ == "__main__":
    main()
