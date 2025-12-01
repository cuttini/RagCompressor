import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from config import STAGE1_RAW_PATH, STAGE1_VALIDATED_PATH
from llm_client import QwenClient

SYSTEM_PROMPT = (
    "Sei un revisore fiscale estremamente severo. "
    "Valuti se una risposta è COMPLETAMENTE supportata dal testo normativo fornito."
)

USER_TEMPLATE = """
TESTO NORMATIVO (SEAC):

\"\"\"{doc}\"\"\"


DOMANDA:
{question}

RISPOSTA PROPOSTA:
{answer}

Valuta da 1 a 5 quanto la risposta è supportata dal testo (1 = fuorviante, 5 = completamente corretta
e giustificata dal testo). Rispondi SOLO con un JSON del tipo:

{{
  "score": 1-5,
  "justification": "spiega in poche frasi perché"
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
                    res = client.json_completion(SYSTEM_PROMPT, user)
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
