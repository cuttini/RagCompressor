# Script di Enrichment per Stage 3 - CLaRa End-to-End Retrieval

## Scopo

Lo script `05_enrich_stage3.py` trasforma il dataset "puro" (positive-only) generato da `03_generate_dataset.py` in un dataset adatto per l'addestramento End-to-End dello **Stage 3 di CLaRa**.

### Problema

Il dataset generato da `03_generate_dataset.py` contiene solo documenti corretti (gold documents). Per addestrare il **Retriever (Query Reasoner)**, il modello deve imparare a distinguere tra:

- ✅ Documenti rilevanti (gold)
- ❌ Documenti simili ma errati (hard negatives)
- ❌ Documenti casuali (random negatives)

Senza distrattori, il gradiente diventa "pigro" e il retriever non impara a discriminare.

## Strategia di Enrichment

Per ogni campione nel dataset:

1. **Gold Documents (2-3)**: Mantiene i chunk corretti originali
2. **Hard Negatives (~7)**: Usa **BM25** per trovare chunk simili alla query ma che NON sono gold (distrattori difficili)
3. **Random Negatives (~10-11)**: Aggiunge chunk casuali dal database
4. **Shuffle**: Mescola tutti i 20 documenti per evitare bias posizionali
5. **Pos Index**: Ricalcola gli indici dei gold documents nella lista mescolata

## Utilizzo

```bash
python 05_enrich_stage3.py
```

### Input

- `artifacts/stage1_2_instruction.jsonl` (output di `03_generate_dataset.py`)
- `artifacts/chunks.jsonl` (database completo dei chunk)

### Output

- `artifacts/stage3_end_to_end.jsonl` (dataset pronto per Stage 3)

## Formato Output

Ogni riga del file di output è un JSON con:

```json
{
  "question": "Domanda fiscale...",
  "docs": ["doc1", "doc2", ..., "doc20"],
  "gold_answer": "Risposta corretta...",
  "pos_index": [3, 7, 15],
  "data_type": "qa"
}
```

Dove:

- `docs`: Array di 20 documenti (mescolati)
- `pos_index`: Indici dei documenti gold nella lista `docs`

## Note Tecniche

- **BM25**: Algoritmo di ranking basato su frequenza dei termini (non richiede embeddings o GPU)
- **Velocità**: ~100-200 campioni/secondo su CPU
- **Determinismo**: Risultati riproducibili con `random.seed()` se necessario

## Dipendenze

```bash
pip install rank-bm25
```
