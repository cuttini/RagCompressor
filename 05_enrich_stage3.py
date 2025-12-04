import json
import random
import string
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from config import CHUNKS_PATH, STAGE1_2_PATH, ARTIFACTS_DIR

# Italian language support for better BM25 retrieval
try:
    from nltk.stem.snowball import ItalianStemmer
    STEMMER = ItalianStemmer()
    USE_STEMMER = True
except ImportError:
    print("[WARNING] NLTK not available. BM25 will use simple tokenization without stemming.")
    print("         Install with: pip install nltk && python -c \"import nltk; nltk.download('punkt')\"")
    USE_STEMMER = False

def tokenize_italian(text: str) -> list:
    """
    Tokenize text for Italian language BM25 retrieval.
    
    Features:
    - Removes punctuation
    - Lowercase normalization
    - Italian stemming (if NLTK available) to match inflections:
      pagare/pagamento/pagato → pag
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of normalized/stemmed tokens
    """
    # Remove punctuation and lowercase
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    
    if USE_STEMMER:
        return [STEMMER.stem(token) for token in tokens]
    else:
        return tokens

# Configurazione
INPUT_FILE = STAGE1_2_PATH  # Output dello script 03 (stage1_2_instruction.jsonl)
CHUNKS_FILE = CHUNKS_PATH    # Il database di tutti i chunk (chunks.jsonl)
OUTPUT_FILE = ARTIFACTS_DIR / "stage3_end_to_end.jsonl"
TOTAL_DOCS = 20  # Target size del pool

def main():
    # 1. Carichiamo tutti i chunk per creare il pool di distrattori
    print("Caricamento database chunk...")
    all_chunks = []
    chunk_contents = []
    with open(CHUNKS_FILE, 'r') as f:
        for line in f:
            c = json.loads(line)
            all_chunks.append(c)
            # Use Italian-aware tokenization for better BM25 matching
            chunk_contents.append(tokenize_italian(c['content']))

    # Creiamo l'indice BM25 per trovare Hard Negatives velocemente
    print("Indicizzazione BM25 (per Hard Negatives)...")
    bm25 = BM25Okapi(chunk_contents)

    print("Arricchimento dataset per Stage 3...")
    with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'w') as fout:
        for line in tqdm(fin):
            record = json.loads(line)
            
            question = record['question']
            gold_docs = record['docs'] # Questi sono i chunk corretti
            
            # Identifichiamo il contenuto dei gold per escluderli dai negativi
            gold_contents = set(gold_docs)
            
            # --- A. TROVARE HARD NEGATIVES ---
            # Cerchiamo chunk simili alla domanda usando tokenizzazione italiana
            tokenized_query = tokenize_italian(question)
            # Ne prendiamo 50 per essere sicuri di trovarne esclusi i gold
            candidates = bm25.get_top_n(tokenized_query, all_chunks, n=50)
            
            hard_negatives = []
            for c in candidates:
                if c['content'] not in gold_contents:
                    hard_negatives.append(c['content'])
                    if len(hard_negatives) >= 7: # Ne vogliamo circa 7 "difficili"
                        break
            
            # --- B. TROVARE RANDOM NEGATIVES ---
            random_negatives = []
            while len(gold_docs) + len(hard_negatives) + len(random_negatives) < TOTAL_DOCS:
                rnd = random.choice(all_chunks)['content']
                if rnd not in gold_contents and rnd not in hard_negatives:
                    random_negatives.append(rnd)
            
            # --- C. COSTRUZIONE POOL FINALE ---
            # Uniamo tutto
            final_pool = gold_docs + hard_negatives + random_negatives
            # Tagliamo se per caso abbiamo sforato (ma il while sopra protegge)
            final_pool = final_pool[:TOTAL_DOCS]
            
            # Mescoliamo per non avere i Gold sempre in testa
            # (Ma dobbiamo tracciare dove finiscono i Gold!)
            
            # Creiamo coppie (contenuto, is_gold)
            tagged_pool = []
            for d in final_pool:
                tagged_pool.append((d, d in gold_contents))
            
            random.shuffle(tagged_pool)
            
            # Ricostruiamo la lista docs e ricalcoliamo pos_index
            new_docs = []
            new_pos_index = []
            
            for idx, (content, is_gold) in enumerate(tagged_pool):
                new_docs.append(content)
                if is_gold:
                    new_pos_index.append(idx)
            
            # Prepariamo il record Stage 3
            stage3_record = {
                "question": question,
                "docs": new_docs,
                "answer": record['answer'],
                "pos_index": new_pos_index,
                "data_type": "qa" # Stage 3 supporta solo qa di solito
            }
            
            fout.write(json.dumps(stage3_record, ensure_ascii=False) + "\n")

    print(f"Fatto! Dataset Stage 3 pronto in {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
