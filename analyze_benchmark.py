import csv
import re
from collections import defaultdict, Counter

BENCHMARK_FILE = "goldquestions/benchmark.tsv"

def classify_question(question):
    q_lower = question.lower()
    
    # Heuristics for classification
    if re.search(r'\b(rigo|quadro|casella|codice|modello|dove va indicato)\b', q_lower):
        return "Form/Field Location"
    if re.search(r'\b(entro quando|scadenza|termine|data|giorni|quanti anni)\b', q_lower):
        return "Deadline/Time"
    if re.search(r'\b(quali sono|elenca|indicare|quali documenti|quali soggetti)\b', q_lower):
        return "List/Enumeration"
    if re.search(r'\b(come si|procedura|modalità|calcolo|come calcolare)\b', q_lower):
        return "Procedure/Calculation"
    if re.search(r'\b(è possibile|può|deve|possono|si può|è obbligatorio)\b', q_lower):
        return "Yes/No (Permission/Obligation)"
    if re.search(r'\b(cosa si intende|definizione|cos\'è|significato)\b', q_lower):
        return "Definition/Concept"
    if re.search(r'\b(se un|in caso di|un contribuente|una società)\b', q_lower):
        return "Scenario/Case Study"
    
    return "Other/General"

def analyze_benchmark():
    typologies = defaultdict(list)
    total_questions = 0
    
    try:
        with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 3: continue
                
                # Assuming question is in column 2 (index 2) based on previous inspection
                # Row format seems to be: ID, Filename, Question, Answer, ...
                question = row[2]
                
                if not question.strip(): continue
                
                category = classify_question(question)
                typologies[category].append(question)
                total_questions += 1
                
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Total Questions Analyzed: {total_questions}\n")
    print("--- Typology Distribution ---")
    
    sorted_categories = sorted(typologies.items(), key=lambda x: len(x[1]), reverse=True)
    
    for category, questions in sorted_categories:
        percentage = (len(questions) / total_questions) * 100
        print(f"\n{category}: {len(questions)} ({percentage:.1f}%)")
        print("Examples:")
        for q in questions[:3]:
            print(f"  - {q}")

if __name__ == "__main__":
    analyze_benchmark()
