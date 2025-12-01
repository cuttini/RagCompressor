# Fiscal-CLaRa Data Factory

Progetto per la creazione automatica di un dataset di alta qualità specializzato sul dominio fiscale italiano, basato sui manuali SEAC.
L'obiettivo è addestrare CLaRa (modello RAG fiscale) tramite dati sintetici generati da Qwen 2.5-32B/72B Instruct.

## Obiettivi

1.  **Stage1 – Compression Pretraining**: Creazione di dati QA e parafrasi ancorati al testo SEAC.
2.  **Stage1_2 – Compression Instruction Tuning**: Conversione dei QA in formati "instruction".

## Pipeline

1.  **Estrazione PDF → Markdown**: Utilizzo di **Nanonets OCR** (modello VLM Nanonets-OCR2-3B) con GPU per preservare struttura e tabelle.
    - Conversione PDF in immagini tramite `pdf2image`
    - Estrazione testo e layout strutturato per pagina
    - Debug mode disponibile per salvare immagini intermedie
    - Limit opzionale di pagine processate per PDF
2.  **Chunking Semantico**: Divisione in chunk logici basati su sezioni e limiti di token (800 token max).
3.  **Generazione Sintetica**:
    - **Simple QA**: Domande atomiche su definizioni, aliquote, scadenze.
    - **Complex QA**: Casi pratici realistici.
    - **Paraphrase**: Riscritture fluenti del contenuto normativo.
4.  **Validazione Automatica**: Filtro anti-allucinazioni usando LLM come giudice.
5.  **Formattazione JSONL**: Output compatibile con CLaRa SFTDataset.
6.  **Debug Visualization** (Opzionale): Visualizzazione dei semantic chunks su pagine PDF.

## Struttura del Progetto

```text
fiscal-clara-data-factory/
├─ config.py                # Configurazioni globali
├─ llm_client.py            # Client per vLLM/Qwen
├─ nanonets_processor.py    # Wrapper per Nanonets OCR
├─ debug_utils.py           # Utility per debug e visualizzazione
├─ 01_pdf_to_markdown.py    # Conversione PDF -> MD (Nanonets OCR)
├─ 02_semantic_chunking.py  # Chunking semantico
├─ 03_generate_dataset.py   # Generazione QA e Parafrasi
├─ 04_validate_dataset.py   # Validazione dataset
├─ 05_debug_visualize.py    # Visualizzazione chunk su PDF
├─ README.md                # Questo file
├─ ebooks_seac/             # Directory per i PDF SEAC (Input)
├─ markdown_seac/           # Output conversione MD + layout JSON
└─ artifacts/               # Output finali (JSONL)
   ├─ chunks.jsonl
   ├─ stage1_raw.jsonl
   ├─ stage1_validated.jsonl
   └─ stage1_2_instruction.jsonl
```

## Utilizzo

1.  Posizionare i manuali SEAC (PDF) nella cartella `ebooks_seac/`.
2.  Eseguire la pipeline sequenzialmente:

```bash
# 1. Conversione PDF -> Markdown (con opzioni per debug e limite pagine)
python 01_pdf_to_markdown.py                    # Processa tutti i PDF
python 01_pdf_to_markdown.py --debug            # Salva immagini intermedie
python 01_pdf_to_markdown.py --max-pages 10     # Limita a 10 pagine per PDF

# 2. Creazione Chunk Semantici
python 02_semantic_chunking.py

# 3. Generazione Dataset Sintetico
python 03_generate_dataset.py

# 4. Validazione e Pulizia
python 04_validate_dataset.py

# Opzionale: Visualizzazione debug dei chunk su PDF
python 05_debug_visualize.py
```

## Requisiti

- Python 3.10+
- **GPU** (CUDA): Necessaria per Nanonets OCR
- **Librerie Python**:
  - `transformers` - Per Nanonets OCR model
  - `torch` - PyTorch con supporto CUDA
  - `pdf2image` - Conversione PDF in immagini
  - `Pillow` (PIL) - Manipolazione immagini
  - `openai` - Client API per LLM
  - `tenacity` - Retry logic
  - `tqdm` - Progress bars
- **Dipendenze di Sistema**:
  - `poppler-utils` - Richiesto da pdf2image
- **Server vLLM** attivo con modello Qwen 2.5-32B/72B (configurabile in `config.py`)
- **Modello OCR**: `nanonets/Nanonets-OCR2-3B` (scaricato automaticamente da HuggingFace)
