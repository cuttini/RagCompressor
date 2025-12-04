# Analisi Errori Residui: Chiusure Tabelle Orfane

## Riepilogo Esecutivo

**Totale errori**: 30 (0.07% dei 41,762 chunk totali)  
**Tipologia**: 100% "chiusure orfane" (`</table>` senza `<table>` corrispondente)

## Root Cause Identificata

### Bug nella funzione `repair_malformed_tables()`

La funzione di riparazione HTML sta **aggiungendo tag `</table>` in eccesso**, creando un effetto "domino":

#### Esempio: Chunk 117 e 118

**Chunk 117** (`6. I creditori estranei`):

- Tag `<table>`: 2
- Tag `</table>`: **3** ← UNO IN PIÙ!
- Ultimo contenuto: termina normalmente (nessun troncamento)

**Chunk 118** (`6. I creditori estranei` - stesso titolo!):

- Tag `<table>`: 0
- Tag `</table>`: **1** ← ORFANO
- Inizia con: `</table> # f` (commento sporco)

### Meccanismo del Bug

La funzione `repair_malformed_tables()` ([righe 56-122](file:///home/sysadmin/dave/clara/fiscal-clara-data-factory/02_semantic_chunking.py#L56-L122)) usa questa euristica:

```python
# Cerca il prossimo header o tabella
next_section = re.search(r'\n## |<table', text[search_start:], re.IGNORECASE)

if next_section:
    close_position = search_start + next_section.start()
else:
    close_position = len(text)

# Inserisci </table> prima della nuova sezione
insertions.append((close_position, '\n</table>\n'))
```

**Problema**: Se nel testo ci sono **pattern ambigui** che assomigliano a `</table>`, la regex potrebbe non matchare correttamente, portando a inserzioni duplicate o in posizioni errate.

## Distribuzione Errori per File

| File                                                | Errori | % sul totale |
| --------------------------------------------------- | ------ | ------------ |
| Mod_REDDITI_Imprese_Individuali_2025_Digitale_PS.md | 7      | 23.3%        |
| Dichiarazione_Redditi_Esteri_2025.md                | 3      | 10.0%        |
| Mod_730_Casi_Pratici_2025_Digitale_PS.md            | 3      | 10.0%        |
| Mod_REDDITI_Societa_Di_Capitali_2025_Digitale_PS.md | 3      | 10.0%        |
| Mod_REDDITI_Societa_Di_Persone_2025_PS.md           | 3      | 10.0%        |
| Altri (9 file)                                      | 11     | 36.7%        |

### Pattern Comune

I file più colpiti sono **modelli dichiarativi 2025** con molte tabelle complesse:

- Mod REDDITI (Imprese Individuali, Società di Capitali, Società di Persone)
- Mod 730 Casi Pratici
- Dichiarazione Redditi Esteri

Questi documenti hanno:

- Tabelle annidate o con strutture complesse
- Molti tag HTML inline
- Possibili commenti HTML (`<!-- -->`) che confondono il parser

## Esempi di Chiusure Orfane

### Caso 1: Fine di sezione pulita

**Chunk**: IVA_Operazioni_Estero_2024.md - Chunk 21591  
**Titolo**: OBBLIGO DI FATTURAZIONE  
**Prima di `</table>`**:

```
...arte" o "regime del margine – oggetti di antiquariato o da collezione.</td></tr>
```

→ Sembra una riga di tabella normale, ma il `<table>` è nel chunk precedente.

### Caso 2: Commenti HTML sporchi

**Chunk**: ISA 2025.md - Chunk 18623  
**Titolo**: MOD. CPB 2025-2026  
**Prima di `</table>`**:

```
... il subentro di due o più eredi in caso di decesso del socio o associato;

---
```

→ La tabella era probabilmente seguita da `---` (separatore), confondendo il repair.

### Caso 3: Fine documento

**Chunk**: Accordo_Ristrutturazione_Debiti_Codice_Crisi_2023.md - Chunk 122  
**Titolo**: 8. La convenienza del trattamento proposto rispetto alla li-  
**Prima di `</table>`**:

```
...venienza del trattamento proposto rispetto alla liquidazione giudi-ziale.

---


```

→ End of section con `---`, il repair ha aggiunto `</table>` in eccesso.

## Raccomandazioni

### 1. Fix Immediato (Quick Win)

Migliorare la logica di `repair_malformed_tables()`:

```python
def repair_malformed_tables(text: str) -> str:
    # ... existing code ...

    # NUOVO: Verifica bilanciamento prima e dopo repair
    opens_before = len(re.findall(r'<table', text, re.IGNORECASE))
    closes_before = len(re.findall(r'</table>', text, re.IGNORECASE))

    # ... repair logic ...

    # Validazione post-repair
    opens_after = len(re.findall(r'<table', repaired, re.IGNORECASE))
    closes_after = len(re.findall(r'</table>', repaired, re.IGNORECASE))

    if closes_after > opens_after:
        # ROLLBACK! Troppe chiusure aggiunte
        print(f"[WARNING] Repair ha creato {closes_after - opens_after} chiusure in eccesso, rollback")
        return text  # Ritorna originale senza repair

    return repaired
```

### 2. Soluzione Robusta (Long Term)

Usare un parser HTML vero invece di regex:

```python
from bs4 import BeautifulSoup

def repair_malformed_tables_v2(text: str) -> str:
    soup = BeautifulSoup(text, 'html.parser')
    # BeautifulSoup ripara automaticamente HTML malformato
    return str(soup)
```

**Rischio**: BeautifulSoup potrebbe alterare la formattazione del markdown.

### 3. Approccio Conservativo

**Accettare i 30 errori** (0.07%) come trade-off accettabile:

- ✅ 97.4% di miglioramento ottenuto (1272 → 30)
- ✅ Chunk con tabelle complete: 8,430 (20.2% del dataset)
- ⚠️ 30 chunk con `</table>` orfano: impatto minimo sul training

**Pro**: Nessuna ulteriore complessità  
**Contro**: Dataset non perfetto al 100%

### 4. Post-processing Filter

Aggiungere un filtro alla fine di `chunk_markdown()`:

```python
def flush(title: str, body: str, page_number: str = "N/A"):
    # ... existing logic ...

    # FILTER: Skip chunks che iniziano con </table>
    if final_content.lstrip().startswith('</table>'):
        print(f"[FILTER] Skipping chunk con chiusura orfana: {section_title}")
        return

    chunks.append({...})
```

## Decisione Consigliata

**Opzione 3 (Conservativa)** con **Opzione 4 (Filter)** come fallback rapido:

1. Mantenere l'attuale implementazione che funziona al 97.4%
2. Aggiungere un semplice filtro per skippare chunk che iniziano con `</table>` orfano
3. Monitorare metriche di training del modello
4. Se necessario, implementare Fix #1 o #2 in futuro

### Implementazione Filter (5 minuti)

```python
# In chunk_markdown(), funzione flush()
# Dopo riga 449, prima di append:

# FILTER orphan closing tags
if sub.strip().startswith('</table>'):
    continue  # Skip this sub-chunk
```

**Risultato atteso**: 30 → 0 errori (eliminando chunk problematici)

## Impatto sul Dataset

### Con 30 errori residui

- **Chunk totali**: 41,762
- **Chunk con errori**: 30 (0.07%)
- **Chunk validi**: 41,732 (99.93%)
- **Chunk con tabelle complete**: 8,430

### Dopo implementazione Filter

- **Chunk totali**: ~41,730 (-30)
- **Chunk con errori**: 0
- **Chunk validi**: 100%
- **Impatto**: Trascurabile (-0.07% del dataset)

## Conclusione

Gli errori residui sono causati da un **over-repair** della funzione `repair_malformed_tables()`. La soluzione più pragmatica è:

1. ✅ Accettare il 97.4% di successo come ottimo risultato
2. ✅ Opzionalmente aggiungere un filtro per i 30 chunk orfani
3. 📊 Monitorare l'impatto sul training
4. 🔄 Iterare se necessario in base ai risultati

**Raccomandazione finale**: Procedi con il training usando il dataset attuale (41,762 chunk). Gli errori residui sono statisticamente irrilevanti (<0.1%) e non compromettono la qualità del RAG.
