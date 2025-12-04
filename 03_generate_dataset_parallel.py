import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
from datetime import datetime

from config import CHUNKS_PATH, STAGE1_RAW_PATH, STAGE1_2_PATH, WINDOW_SIZE, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE
from llm_client import QwenClient

# Checkpoint file path
CHECKPOINT_PATH = Path("output/checkpoint.json")

# === PROMPTS ===
SYSTEM_PROMPT = (
    "Sei un esperto fiscalista italiano e docente per l'esame di abilitazione.\n"
    "Riceverai una sequenza di estratti (chunk) da manuali SEAC. "
    "Il tuo compito è generare dataset per il training di un'IA fiscale.\n"
    "Devi prestare massima attenzione a:\n"
    "1. REALISMO PROFESSIONALE: Le domande devono riflettere dubbi reali di commercialisti e operatori CAF. "
    "Usa un linguaggio tecnico ma pratico. Evita domande scolastiche o puramente definitorie.\n"
    "2. SPECIFICITÀ DEI MODELLI: Se il testo cita quadri, righi o codici (es. 'Quadro RN', 'Rigo 20', 'Codice 12'), "
    "la domanda DEVE essere specifica (es. 'Dove va indicato X?', 'Cosa inserire nel rigo Y?').\n"
    "3. SCENARI PRATICI: Prediligi domande basate su casi d'uso (es. 'Il mio cliente ha fatto X, può detrarre Y?').\n"
    "4. PRESERVAZIONE NUMERI: Tutti i numeri (date, importi, riferimenti normativi) "
    "presenti nella domanda DEVONO essere presenti nella risposta.\n"
    "5. NATURALEZZA: NON fare mai riferimento a 'chunk', 'contesto', 'estratto'. "
)

USER_TEMPLATE = """
CONTESTO NORMATIVO (Sliding Window di {window_size} chunk):

{context_text}

---

COMPITI (Rispondi ESCLUSIVAMENTE con un JSON valido):

Genera Q&A basate sul testo fornito, focalizzandoti sul CHUNK CENTRALE (Chunk #{center_id}).

LINEE GUIDA PER DOMANDE "REALI" (Basate su Benchmark):
- EVITA: "Cosa dice il testo riguardo X?", "Definisci X."
- PREFERISCI: "Un contribuente che ha X, come deve comportarsi?", "In quale rigo del modello Redditi va indicato X?", "La detrazione Y spetta anche se...?"

CATEGORIE RICHIESTE (Genera 2-3 domande miste scegliendo tra queste tipologie, se applicabili):

1. "form_loc_qa" (Form/Field Location):
   - CHIEDI DOVE VANNO INSERITI I DATI (Quadro, Rigo, Codice).
   - Esempio: "In quale rigo del quadro RN si indica l'eccedenza?"
   - *Usa SOLO se il testo cita quadri/righi.*

2. "deadline_qa" (Deadline/Time):
   - Domande su date, scadenze, termini o durate.
   - Esempio: "Entro quale data va inviata la comunicazione?", "Quanti giorni di preavviso servono?"
   - *Usa SOLO se il testo cita date/termini.*

3. "scenario_qa" (Scenario/Case Study):
   - Domande ipotetiche su casi concreti.
   - Esempio: "Se una SRL ha optato per..., può...?", "In caso di decesso del titolare, gli eredi devono...?"

4. "boolean_qa" (Yes/No with Explanation):
   - Domande su permessi, obblighi o possibilità.
   - Esempio: "È possibile detrarre le spese per...?", "Il contribuente è obbligato a...?"
   - *La risposta deve iniziare con "Sì," o "No," seguita dalla spiegazione.*

5. "procedure_qa" (Procedure/Calculation):
   - Domande su "come fare" o "come calcolare".
   - Esempio: "Come si determina la base imponibile per...?", "Qual è la procedura per richiedere...?"

ALTRE CATEGORIE (1 per tipo, se applicabile):
- "complex_qas": 1 domanda complessa che richiede ragionamento articolato.
- "multi_hop_qas": 1 domanda di collegamento tra sezioni diverse.
- "evolution_qas": Se nel contesto ci sono riferimenti a normative precedenti, cambi di anno o modifiche legislative, 
  genera 1 domanda che chieda esplicitamente la differenza o l'evoluzione nel tempo.
  Esempio: "Come è cambiata la deducibilità tra il 2023 e il 2024?", "Quali sono le differenze rispetto alla normativa precedente?"
  *Usa SOLO se il testo contiene riferimenti temporali/normativi multipli.*
- "paraphrase": Una riscrittura discorsiva e completa del contenuto principale.

REGOLA CRITICA - PRESERVAZIONE NUMERI:
Se la domanda contiene numeri (date, importi, articoli), la risposta DEVE includerli esattamente.

REGOLA CRITICA - NATURALEZZA:
MAI usare termini come "chunk", "testo", "estratto". Immagina di parlare con un collega.

Schema JSON atteso:
{{
  "mixed_qas": [
      {{"type": "form_loc_qa", "q": "...", "a": "..."}},
      {{"type": "scenario_qa", "q": "...", "a": "..."}}
      // Inserisci 2-3 domande di tipi diversi
  ],
  "complex_qas": [{{"q": "...", "a": "..."}}],
  "multi_hop_qas": [{{"q": "...", "a": "..."}}],
  "evolution_qas": [{{"q": "...", "a": "..."}}],
  "paraphrase": "..."
}}
"""

# === UTILITY FUNCTIONS ===
def extract_numbers(text: str) -> set:
    """
    Estrae tutti i numeri rilevanti da un testo fiscale con normalizzazione.
    
    Cattura:
    - Date (DD/MM/YYYY, DD-MM-YYYY, "1° gennaio 2024")
    - Importi (€ X.XXX, X euro, X,XX)
    - Riferimenti normativi (Art. X, D.L. X/YYYY, L. X/YYYY, comma X)
    - Percentuali (X%, X per cento)
    - Numeri in contesto ("entro X giorni", "X mesi")
    
    NORMALIZZAZIONE:
    - Rimuove punti delle migliaia: 1.000 → 1000
    - Normalizza decimali: 1,5 → 1.5
    - Espande riferimenti: Art. 10 → articolo 10
    
    Returns:
        Set di stringhe normalizzate contenenti i numeri trovati
    """
    import re
    numbers = set()
    
    # Date formato DD/MM/YYYY o DD-MM-YYYY
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    numbers.update(re.findall(date_pattern, text))
    
    # Date formato "1° gennaio 2024", "primo gennaio 2024"
    date_long_pattern = r'\b\d{1,2}°?\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}\b'
    numbers.update(re.findall(date_long_pattern, text, re.IGNORECASE))
    
    # Importi monetari: € 1.234,56 o 1.234 euro o 1234,56 euro
    # Normalizza rimuovendo punti migliaia e sostituendo virgola decimale
    money_pattern = r'€\s*[\d.,]+|[\d.,]+\s*(?:euro|EUR)'
    for match in re.findall(money_pattern, text, re.IGNORECASE):
        # Normalizza: rimuovi punti migliaia se presenti con virgola decimale
        normalized = match
        if ',' in match and '.' in match:
            # Formato italiano: 1.234,56 → 1234.56
            normalized = match.replace('.', '').replace(',', '.')
        elif ',' in match:
            # Solo virgola: 1234,56 → 1234.56
            normalized = match.replace(',', '.')
        numbers.add(normalized.lower())
    
    # Riferimenti normativi: Art. 123, D.L. 34/2020, L. 178/2020
    # Aggiungi sia forma abbreviata che estesa
    art_pattern = r'\b(?:Art\.|Articolo)\s*(\d+)(?:[- ](?:bis|ter|quater))?\b'
    for match in re.finditer(art_pattern, text, re.IGNORECASE):
        art_num = match.group(1)
        numbers.add(f"art. {art_num}")
        numbers.add(f"articolo {art_num}")
    
    dl_pattern = r'\b(?:D\.L\.|D\.Lgs\.|L\.)\s*\d+/\d{2,4}\b'
    numbers.update(re.findall(dl_pattern, text, re.IGNORECASE))
    
    # Comma, paragrafo
    comma_pattern = r'\bcomma\s+\d+\b'
    numbers.update(re.findall(comma_pattern, text, re.IGNORECASE))
    
    # Percentuali: 110% o 50 per cento
    perc_pattern = r'\b\d+(?:[.,]\d+)?\s*%|(\b\d+(?:[.,]\d+)?)\s+per\s+cento\b'
    for match in re.finditer(perc_pattern, text, re.IGNORECASE):
        if match.group(0).endswith('%'):
            numbers.add(match.group(0).replace(',', '.'))
        else:
            # Aggiungi entrambe le forme
            num = match.group(1).replace(',', '.')
            numbers.add(f"{num}%")
            numbers.add(f"{num} per cento")
    
    # Numeri con context fiscale specifico
    context_pattern = r'\b(\d+)(?:[.,](\d+))?\s+(giorni|mesi|anni|rate|annualità)\b'
    for match in re.finditer(context_pattern, text, re.IGNORECASE):
        full_match = match.group(0).replace(',', '.')
        numbers.add(full_match.lower())
        
        # Aggiungi anche forma scritta per numeri comuni
        num = int(match.group(1))
        unit = match.group(3).lower()
        if num <= 100:  # Solo per numeri piccoli
            numbers.add(f"{num} {unit}")
    
    # Anni standalone (solo 4 cifre che iniziano con 19xx o 20xx)
    year_pattern = r'\b(?:19|20)\d{2}\b'
    numbers.update(re.findall(year_pattern, text))
    
    # Normalizza: lowercase e rimuovi spazi multipli
    normalized = {n.lower().strip() for n in numbers if n.strip()}
    return normalized


def validate_number_preservation(question: str, answer: str) -> Tuple[bool, List[str]]:
    """
    Verifica che tutti i numeri nella domanda siano presenti nella risposta.
    
    Args:
        question: Testo della domanda
        answer: Testo della risposta
    
    Returns:
        (is_valid, missing_numbers): Tuple con flag di validità e lista numeri mancanti
    """
    q_numbers = extract_numbers(question)
    a_numbers = extract_numbers(answer)
    
    # Tutti i numeri della domanda devono essere nella risposta
    missing = q_numbers - a_numbers
    is_valid = len(missing) == 0
    
    return is_valid, sorted(list(missing))


def get_adaptive_sliding_windows(
    chunks: List[Dict], 
    min_window_size: int = 3,
    max_window_size: int = 7
) -> List[Tuple[List[Dict], int]]:
    """
    Genera finestre scorrevoli ADATTIVE basate su titolo/sezione.
    
    La window si espande finché i chunk hanno lo stesso 'title', rispettando
    i limiti min/max per evitare mixing di sezioni diverse e overflow di memoria.
    
    Args:
        chunks: Lista di chunk da processare
        min_window_size: Dimensione minima della window (default: 3)
        max_window_size: Dimensione massima della window (default: 7)
    
    Returns:
        Lista di tuple (window, center_relative_index)
    
    Esempi:
        - Sezione lunga (5+ chunk stesso title) → window = 5-7 chunk
        - Sezione breve (2 chunk) → window = 3 chunk (aggiunge contesto prev/next)
        - Boundary tra sezioni → window si ferma al cambio title
    """
    windows = []
    n = len(chunks)
    
    for i in range(n):
        center = chunks[i]
        center_title = center.get('title', '')
        
        # Inizia con il chunk centrale
        start = i
        end = i + 1
        
        # ESPANDI VERSO SINISTRA finché:
        # 1. Stesso title del centro
        # 2. Non superi max_window_size
        # 3. Non sei all'inizio della lista
        while (start > 0 and 
               end - start < max_window_size and
               chunks[start - 1].get('title', '') == center_title):
            start -= 1
        
        # ESPANDI VERSO DESTRA finché:
        # 1. Stesso title del centro
        # 2. Non superi max_window_size
        # 3. Non sei alla fine della lista
        while (end < n and 
               end - start < max_window_size and
               chunks[end].get('title', '') == center_title):
            end += 1
        
        # Se la window è troppo piccola, allarga con contesto aggiuntivo
        # (anche se titolo diverso, per avere minimo min_window_size chunk)
        current_size = end - start
        if current_size < min_window_size:
            deficit = min_window_size - current_size
            half_deficit = deficit // 2
            
            # Aggiungi deficit/2 a sinistra (se possibile)
            extra_left = min(half_deficit, start)
            start -= extra_left
            
            # Aggiungi rimanente a destra (se possibile)
            extra_right = min(deficit - extra_left, n - end)
            end += extra_right
            
            # Se ancora troppo piccolo (edge case: inizio/fine lista), 
            # prendi tutto quello che puoi
            if end - start < min_window_size:
                if start > 0:
                    start = max(0, end - min_window_size)
                elif end < n:
                    end = min(n, start + min_window_size)
        
        window = chunks[start:end]
        center_rel_idx = i - start
        windows.append((window, center_rel_idx))
    
    return windows

def clean_and_parse_json(llm_output_str: str) -> Dict:
    """Pulisce markdown code blocks se presenti."""
    cleaned = llm_output_str.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)

def format_window_text(window: List[Dict], center_idx: int) -> str:
    """Formatta il testo della finestra per il prompt."""
    out = []
    for idx, ch in enumerate(window):
        marker = " (CHUNK CENTRALE)" if idx == center_idx else f" (Contesto {'Precedente' if idx < center_idx else 'Successivo'})"
        out.append(f"=== CHUNK #{ch['id']}{marker} ===\nFonte: {ch['source']}\nTitolo: {ch['title']}\n\n{ch['content']}\n")
    return "\n".join(out)

def build_records(window: List[Dict], center_idx: int, llm_output: Dict) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Costruisce i record Stage 1 e Stage 1.2 con validazione HARD sui numeri.
    
    Returns:
        (s1_records, s1_2_records, validation_stats)
    """
    s1_recs = []
    s1_2_recs = []
    validation_stats = {
        'total_qas': 0,
        'rejected_qas': 0,
        'rejected_details': []  # [(question, answer, missing_numbers)]
    }
    
    center_chunk = window[center_idx]
    center_text = center_chunk["content"]
    full_window_texts = [chunk["content"] for chunk in window]
    
    # STAGE 1 - MIXED QAs (ex simple_qas)
    s1_qas = []
    # Supporta sia il vecchio formato "simple_qas" che il nuovo "mixed_qas" per compatibilità
    mixed_source = llm_output.get("mixed_qas") or llm_output.get("simple_qas") or []
    
    for item in mixed_source:
        if "q" in item and "a" in item:
            q, a = item["q"].strip(), item["a"].strip()
            validation_stats['total_qas'] += 1
            
            # HARD validation: verifica preservazione numeri
            is_valid, missing = validate_number_preservation(q, a)
            if not is_valid:
                validation_stats['rejected_qas'] += 1
                validation_stats['rejected_details'].append((q, a, missing))
                continue  # Scarta questa Q&A
            
            s1_qas.append(item)
            
    if s1_qas:
        s1_recs.append({
            "data_type": "qa",
            "question": [x["q"].strip() for x in s1_qas],
            "answers": [x["a"].strip() for x in s1_qas],
            "docs": [center_text],
            "pos_index": [0]
        })

    para = (llm_output.get("paraphrase") or "").strip()
    if para:
        s1_recs.append({
            "data_type": "paraphrase",
            "question": "",
            "answers": [para],
            "docs": [center_text],
            "pos_index": [0]
        })

    # STAGE 1.2
    # Includiamo anche "mixed_qas" per avere i singoli record
    all_categories = ["mixed_qas", "simple_qas", "complex_qas", "multi_hop_qas", "evolution_qas"]
    
    for cat in all_categories:
        source_list = llm_output.get(cat) or []
        for item in source_list:
            q, a = item.get("q", "").strip(), item.get("a", "").strip()
            qa_type = item.get("type", cat) # Usa il tipo specifico se presente (es. "form_loc_qa"), altrimenti la categoria
            
            if not q or not a:
                continue
            
            validation_stats['total_qas'] += 1
            
            # HARD validation: verifica preservazione numeri
            is_valid, missing = validate_number_preservation(q, a)
            if not is_valid:
                validation_stats['rejected_qas'] += 1
                validation_stats['rejected_details'].append((q, a, missing))
                continue  # Scarta questa Q&A
                
            is_context_heavy = cat in ["multi_hop_qas", "complex_qas", "evolution_qas"]
            current_docs = full_window_texts if is_context_heavy else [center_text]
            current_pos_index = [i for i in range(len(current_docs))] if is_context_heavy else [0]

            s1_2_recs.append({
                "question": q,
                "docs": current_docs, 
                "answer": a,
                "data_type": "qa",
                "pos_index": current_pos_index,
                "metadata": {"type": qa_type, "source_chunk_id": center_chunk["id"]}
            })

    return s1_recs, s1_2_recs, validation_stats


class DatasetGenerator:
    def __init__(self, max_workers: int = 4, timeout: int = 120, enable_resume: bool = True):
        """
        Args:
            max_workers: Numero di thread paralleli per chiamate LLM
            timeout: Timeout in secondi per ogni chiamata LLM
            enable_resume: Abilita il resume automatico tramite checkpoint
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.enable_resume = enable_resume
        self.write_lock = Lock()
        self.checkpoint_lock = Lock()
        self.stats = {
            'processed': 0,
            'errors': 0,
            'total_time': 0.0,
            'total_qas': 0,
            'rejected_qas': 0,
            'rejected_samples': [],  # Log fino a 10 esempi di rigetto
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0
        }
        self.processed_indices = set()  # Track processed window indices
    
    def process_window(
        self,
        window: List[Dict],
        center_idx: int,
        client: QwenClient
    ) -> Tuple[List[Dict], List[Dict], bool, Dict]:
        """
        Processa una singola finestra e restituisce i record generati.
        
        Returns:
            (s1_records, s1_2_records, success, validation_stats)
        """
        center_chunk = window[center_idx]
        
        # Costruiamo il prompt con il contesto
        context_text = format_window_text(window, center_idx)
        user_prompt = USER_TEMPLATE.format(
            window_size=len(window),
            context_text=context_text,
            center_id=center_chunk["id"]
        )
        
        start = datetime.now()
        try:
            raw_out, usage_stats = client.json_completion(SYSTEM_PROMPT, user_prompt)
            out = raw_out if isinstance(raw_out, dict) else clean_and_parse_json(raw_out)
            
            elapsed = (datetime.now() - start).total_seconds()
            
            # Costruisci i record con validazione
            s1, s1_2, val_stats = build_records(window, center_idx, out)
            
            # Aggiorna statistiche globali
            with self.write_lock:
                self.stats['processed'] += 1
                self.stats['total_time'] += elapsed
                self.stats['total_qas'] += val_stats['total_qas']
                self.stats['rejected_qas'] += val_stats['rejected_qas']
                self.stats['total_tokens'] += usage_stats['total_tokens']
                self.stats['prompt_tokens'] += usage_stats['prompt_tokens']
                self.stats['completion_tokens'] += usage_stats['completion_tokens']
                
                # Salva fino a 10 esempi di rigetto per debugging
                if len(self.stats['rejected_samples']) < 10:
                    for detail in val_stats['rejected_details']:
                        if len(self.stats['rejected_samples']) >= 10:
                            break
                        self.stats['rejected_samples'].append({
                            'chunk_id': center_chunk['id'],
                            'question': detail[0][:100],  # Tronca se troppo lungo
                            'answer': detail[1][:100],
                            'missing_numbers': detail[2]
                        })
            
            return s1, s1_2, True, val_stats
            
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds()
            with self.write_lock:
                self.stats['errors'] += 1
            print(f"\n[WARN] Errore chunk {center_chunk['id']}: {e}")
            return [], [], False, {'total_qas': 0, 'rejected_qas': 0, 'rejected_details': []}
    
    def write_records(
        self,
        s1_records: List[Dict],
        s1_2_records: List[Dict],
        s1_file: Path,
        s1_2_file: Path
    ):
        """Scrive i record in modo thread-safe."""
        with self.write_lock:
            with open(s1_file, "a", encoding="utf-8") as f1:
                for rec in s1_records:
                    f1.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            with open(s1_2_file, "a", encoding="utf-8") as f2:
                for rec in s1_2_records:
                    f2.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    def save_checkpoint(self, processed_idx: int, total_windows: int):
        """Salva il checkpoint con l'indice dell'ultima finestra processata."""
        if not self.enable_resume:
            return
            
        with self.checkpoint_lock:
            checkpoint_data = {
                'last_processed_idx': processed_idx,
                'total_windows': total_windows,
                'processed_count': len(self.processed_indices),
                'timestamp': datetime.now().isoformat(),
                'stats': {
                    'processed': self.stats['processed'],
                    'errors': self.stats['errors'],
                    'total_qas': self.stats['total_qas'],
                    'rejected_qas': self.stats['rejected_qas'],
                    'total_tokens': self.stats['total_tokens'],
                    'prompt_tokens': self.stats['prompt_tokens'],
                    'completion_tokens': self.stats['completion_tokens']
                }
            }
            
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def load_checkpoint() -> dict:
        """Carica il checkpoint salvato, se esiste."""
        if CHECKPOINT_PATH.exists():
            try:
                with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Impossibile caricare checkpoint: {e}")
                return None
        return None
    
    @staticmethod
    def clear_checkpoint():
        """Rimuove il file di checkpoint."""
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
            print("[INFO] Checkpoint cleared.")
    
    def generate_dataset(
        self,
        windows: List[Tuple[List[Dict], int]],
        start_idx: int = 0,
        limit: int = None
    ):
        """
        Genera il dataset con parallelizzazione.
        
        Args:
            windows: Lista di (window, center_idx)
            start_idx: Indice da cui iniziare (per resume)
            limit: Numero massimo di finestre da processare (None = tutte)
        """
        # Filtra le finestre da processare
        if limit:
            windows_to_process = windows[start_idx:start_idx + limit]
        else:
            windows_to_process = windows[start_idx:]
        
        total = len(windows_to_process)
        print(f"[INFO] Processando {total} finestre (da {start_idx} a {start_idx + total})...")
        print(f"[INFO] Workers paralleli: {self.max_workers}")
        print(f"[INFO] Timeout per richiesta: {self.timeout}s")
        
        # Crea un client per ogni worker
        clients = [QwenClient(timeout=self.timeout) for _ in range(self.max_workers)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Crea i futures
            futures = {}
            for i, (window, center_idx) in enumerate(windows_to_process):
                client = clients[i % self.max_workers]
                future = executor.submit(self.process_window, window, center_idx, client)
                futures[future] = (window, center_idx, start_idx + i)
            
            # Progress bar
            with tqdm(total=total, desc="Generazione Dataset", unit="win") as pbar:
                for future in as_completed(futures):
                    window, center_idx, original_idx = futures[future]
                    s1, s1_2, success, val_stats = future.result()
                    
                    if success:
                        # Scrivi i risultati
                        self.write_records(s1, s1_2, STAGE1_RAW_PATH, STAGE1_2_PATH)
                        
                        # Traccia indice processato e salva checkpoint
                        self.processed_indices.add(original_idx)
                        self.save_checkpoint(original_idx, len(windows))
                        
                        # Aggiorna progress bar con statistiche
                        avg_time = self.stats['total_time'] / self.stats['processed'] if self.stats['processed'] > 0 else 0
                        rejection_rate = (self.stats['rejected_qas'] / self.stats['total_qas'] * 100) if self.stats['total_qas'] > 0 else 0
                        tokens_per_sec = self.stats['total_tokens'] / self.stats['total_time'] if self.stats['total_time'] > 0 else 0
                        pbar.set_postfix({
                            'avg_time': f'{avg_time:.1f}s',
                            'tokens/s': f'{tokens_per_sec:.1f}',
                            'errors': self.stats['errors'],
                            'reject%': f'{rejection_rate:.1f}%',
                            'last_idx': original_idx
                        })
                    
                    pbar.update(1)
        
        # Stampa statistiche finali
        print(f"\n[STATS] Completato:")
        print(f"  - Processate: {self.stats['processed']}")
        print(f"  - Errori: {self.stats['errors']}")
        
        # Statistiche token
        print(f"\n[TOKENS] Utilizzo Token:")
        print(f"  - Token totali: {self.stats['total_tokens']:,}")
        print(f"  - Prompt tokens: {self.stats['prompt_tokens']:,}")
        print(f"  - Completion tokens: {self.stats['completion_tokens']:,}")
        if self.stats['total_time'] > 0:
            tokens_per_sec = self.stats['total_tokens'] / self.stats['total_time']
            print(f"  - Throughput: {tokens_per_sec:.1f} tokens/s")
        
        # Statistiche validazione numeri
        print(f"\n[VALIDATION] Preservazione Numeri (HARD mode):")
        print(f"  - Q&A totali generate: {self.stats['total_qas']}")
        print(f"  - Q&A accettate: {self.stats['total_qas'] - self.stats['rejected_qas']}")
        print(f"  - Q&A rigettate: {self.stats['rejected_qas']}")
        if self.stats['total_qas'] > 0:
            rejection_rate = (self.stats['rejected_qas'] / self.stats['total_qas']) * 100
            acceptance_rate = 100 - rejection_rate
            print(f"  - Tasso accettazione: {acceptance_rate:.1f}%")
            print(f"  - Tasso rigetto: {rejection_rate:.1f}%")
        
        # Mostra esempi di rigetto
        if self.stats['rejected_samples']:
            print(f"\n[VALIDATION] Esempi di Q&A rigettate (primi {len(self.stats['rejected_samples'])}):")
            for i, sample in enumerate(self.stats['rejected_samples'], 1):
                print(f"\n  #{i} - Chunk {sample['chunk_id']}:")
                print(f"    Q: {sample['question']}")
                print(f"    A: {sample['answer']}")
                print(f"    Numeri mancanti: {', '.join(sample['missing_numbers'])}")
        
        if self.stats['processed'] > 0:
            avg = self.stats['total_time'] / self.stats['processed']
            print(f"  - Tempo medio: {avg:.2f}s/finestra")
            remaining = len(windows) - (start_idx + total)
            if remaining > 0:
                eta_hours = (remaining * avg) / 3600
                print(f"  - ETA rimanenti ({remaining} finestre): {eta_hours:.1f} ore")

def main():
    parser = argparse.ArgumentParser(description="Genera dataset con parallelizzazione")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Numero di workers paralleli (default: 4)"
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="Timeout per richiesta LLM in secondi (default: 180)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Indice finestra da cui iniziare (per resume, default: 0)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Numero massimo di finestre da processare (default: tutte)"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Cancella output esistenti e ricomincia da zero"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Disabilita il resume automatico (default: resume abilitato)"
    )
    
    args = parser.parse_args()
    
    # Carica chunks
    if not CHUNKS_PATH.exists():
        print(f"[ERROR] {CHUNKS_PATH} non trovato. Esegui prima lo step 02.")
        return
    
    chunks: List[Dict] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    chunks.sort(key=lambda x: (x["source"], x["id"]))
    windows = get_adaptive_sliding_windows(chunks, MIN_WINDOW_SIZE, MAX_WINDOW_SIZE)
    
    # Calcola statistiche sulle window sizes
    window_sizes = [len(w[0]) for w in windows]
    avg_window_size = sum(window_sizes) / len(window_sizes) if window_sizes else 0
    min_win_size = min(window_sizes) if window_sizes else 0
    max_win_size = max(window_sizes) if window_sizes else 0
    
    print(f"[INFO] Caricati {len(chunks)} chunks")
    print(f"[INFO] Generate {len(windows)} adaptive sliding windows")
    print(f"[INFO] Window size stats: min={min_win_size}, max={max_win_size}, avg={avg_window_size:.1f}")
    print(f"[INFO] Configured range: MIN_WINDOW_SIZE={MIN_WINDOW_SIZE}, MAX_WINDOW_SIZE={MAX_WINDOW_SIZE}")
    
    # Gestisci resume automatico
    enable_resume = not args.no_resume
    resume_idx = args.start
    
    if args.clear:
        print("[INFO] Clearing output files and checkpoint...")
        with open(STAGE1_RAW_PATH, "w", encoding="utf-8") as f1:
            pass
        with open(STAGE1_2_PATH, "w", encoding="utf-8") as f2:
            pass
        DatasetGenerator.clear_checkpoint()
        resume_idx = 0
    elif enable_resume and args.start == 0:
        # Carica checkpoint automaticamente solo se non è stato specificato --start
        checkpoint = DatasetGenerator.load_checkpoint()
        if checkpoint:
            resume_idx = checkpoint['last_processed_idx'] + 1
            print(f"[INFO] Auto-resume attivato: ripresa dall'indice {resume_idx}")
            print(f"[INFO] Checkpoint precedente: {checkpoint['processed_count']} finestre processate")
            print(f"[INFO] Timestamp: {checkpoint['timestamp']}")
        else:
            print("[INFO] Nessun checkpoint trovato, partenza da zero")
            resume_idx = 0
    elif args.start > 0:
        print(f"[INFO] Resume manuale dall'indice {resume_idx}")
    else:
        print("[INFO] Resume disabilitato, partenza da zero")
        resume_idx = 0
    
    # Genera dataset
    generator = DatasetGenerator(
        max_workers=args.workers, 
        timeout=args.timeout,
        enable_resume=enable_resume
    )
    generator.generate_dataset(windows, start_idx=resume_idx, limit=args.limit)
    
    print(f"\n[SUCCESS] Output salvati in:")
    print(f"  - {STAGE1_RAW_PATH}")
    print(f"  - {STAGE1_2_PATH}")

if __name__ == "__main__":
    main()
