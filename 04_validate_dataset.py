import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from config import STAGE1_RAW_PATH, STAGE1_VALIDATED_PATH
from llm_client import QwenClient

SYSTEM_PROMPT = (
    "Sei un revisore fiscale ESTREMAMENTE SEVERO specializzato in rilevamento di allucinazioni. "
    "Il tuo compito è verificare se OGNI SINGOLA informazione nella risposta è ESPLICITAMENTE "
    "presente nel testo normativo fornito. NON devi usare la tua conoscenza pregressa del mondo "
    "per valutare la correttezza della risposta. Una risposta può essere VERA nel mondo reale "
    "ma SBAGLIATA ai fini di questa valutazione se le informazioni non sono nel testo."
)

USER_TEMPLATE = """
=== TESTO NORMATIVO (UNICA FONTE DI VERITÀ) ===

\"\"\"{doc}\"\"\"

=== FINE TESTO NORMATIVO ===

DOMANDA:
{question}

RISPOSTA DA VERIFICARE:
{answer}

=== ISTRUZIONI DI VALUTAZIONE ANTI-ALLUCINAZIONE ===

Devi verificare se OGNI informazione nella risposta è ESPLICITAMENTE supportata dal TESTO NORMATIVO sopra.

REGOLE CRITICHE:
1. NON usare MAI la tua conoscenza pregressa (es. leggi che conosci, date, importi, scadenze).
2. Se la risposta contiene dettagli VERI nel mondo reale ma ASSENTI nel testo → punteggio 1 o 2.
3. Se la risposta cita leggi, decreti, articoli NON menzionati nel testo → punteggio 1 o 2.
4. Se la risposta aggiunge date, importi, percentuali NON presenti nel testo → punteggio 1 o 2.
5. Punteggio 4-5 SOLO se ogni singolo fatto è verificabile rileggendo il testo sopra.

SCALA DI VALUTAZIONE:
- 1 = Risposta contiene informazioni esterne/inventate o contraddice il testo
- 2 = Risposta contiene dettagli corretti nel mondo reale ma NON presenti nel testo (DATA LEAKAGE)
- 3 = Risposta parzialmente supportata, alcuni dettagli non verificabili nel testo
- 4 = Risposta ben supportata dal testo con minime inferenze ragionevoli
- 5 = Ogni informazione è ESPLICITAMENTE citata o direttamente deducibile dal testo

Rispondi SOLO con un JSON del tipo:

{{
  "score": 1-5,
  "justification": "spiega brevemente perché",
  "external_info": ["lista di informazioni nella risposta NON trovate nel testo, vuota se nessuna"]
}}
"""

def main():
    client = QwenClient()
    kept = 0
    total = 0

    out_f = open(STAGE1_VALIDATED_PATH, "w", encoding="utf-8")

    with open(STAGE1_RAW_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Validazione dati Stage1"):
            total += 1
            rec = json.loads(line)

            # solo QA; le parafrasi le teniamo sempre
            if rec.get("data_type") == "paraphrase":
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                continue

            doc = rec["docs"][0]
            questions = rec["question"]
            answers = rec["answers"]

            # se una delle QA cade sotto soglia, scartiamo l'intero record
            record_ok = True
            for q, a in zip(questions, answers):
                user = USER_TEMPLATE.format(doc=doc, question=q, answer=a)
                try:
                    res, _ = client.json_completion(SYSTEM_PROMPT, user)
                    score = int(res.get("score", 0))
                except Exception as e:
                    print(f"[WARN] errore validazione: {e}")
                    score = 0

                if score < 4:
                    record_ok = False
                    break

            if record_ok:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    out_f.close()
    print(f"[validate] tenuti {kept}/{total} record QA+paraphrase → {STAGE1_VALIDATED_PATH}")

if __name__ == "__main__":
    main()
