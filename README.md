# Fiscal-CLaRa Data Factory

Progetto per la creazione automatica di un dataset di alta qualità specializzato sul dominio fiscale italiano, basato sui manuali SEAC.
L'obiettivo è addestrare CLaRa (modello RAG fiscale) tramite dati sintetici generati da Qwen 2.5-32B/72B Instruct.

## Obiettivi

1.  **Stage1 – Compression Pretraining**: Creazione di dati QA e parafrasi ancorati al testo SEAC.
2.  **Stage1_2 – Compression Instruction Tuning**: Conversione dei QA in formati "instruction".

## Pipeline

1.  **Estrazione PDF → Markdown**: Utilizzo di `docling` per preservare struttura e tabelle.
2.  **Chunking Semantico**: Divisione in chunk logici basati su sezioni e limiti di token.
3.  **Generazione Sintetica**:
    *   **Simple QA**: Domande atomiche su definizioni, aliquote, scadenze.
    *   **Complex QA**: Casi pratici realistici.
    *   **Paraphrase**: Riscritture fluenti del contenuto normativo.
4.  **Validazione Automatica**: Filtro anti-allucinazioni usando LLM come giudice.
5.  **Formattazione JSONL**: Output compatibile con CLaRa SFTDataset.

## Struttura del Progetto

```text
fiscal-clara-data-factory/
├─ config.py               # Configurazioni globali
├─ llm_client.py           # Client per vLLM/Qwen
├─ 01_pdf_to_markdown.py   # Conversione PDF -> MD
├─ 02_semantic_chunking.py # Chunking semantico
├─ 03_generate_dataset.py  # Generazione QA e Parafrasi
├─ 04_validate_dataset.py  # Validazione dataset
├─ README.md               # Questo file
├─ ebooks_seac/            # Directory per i PDF SEAC (Input)
├─ markdown_seac/          # Output conversione MD
└─ artifacts/              # Output finali (JSONL)
   ├─ chunks.jsonl
   ├─ stage1_raw.jsonl
   ├─ stage1_validated.jsonl
   └─ stage1_2_instruction.jsonl
```

## Utilizzo

1.  Posizionare i manuali SEAC (PDF) nella cartella `ebooks_seac/`.
2.  Eseguire la pipeline sequenzialmente:

```bash
# 1. Conversione PDF -> Markdown
python 01_pdf_to_markdown.py

# 2. Creazione Chunk Semantici
python 02_semantic_chunking.py

# 3. Generazione Dataset Sintetico
python 03_generate_dataset.py

# 4. Validazione e Pulizia
python 04_validate_dataset.py
```

## Requisiti

*   Python 3.8+
*   Librerie: `docling`, `transformers`, `openai`, `tenacity`, `tqdm`
*   Server vLLM attivo con modello Qwen 2.5 (configurabile in `config.py`)
