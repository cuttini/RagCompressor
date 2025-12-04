#!/usr/bin/env python3
"""
Test script per validare la funzione di estrazione numeri e preservazione.
"""

import sys
sys.path.insert(0, '/home/sysadmin/dave/clara/fiscal-clara-data-factory')

from typing import Tuple, List

# Importa le funzioni dal modulo principale
import importlib.util
spec = importlib.util.spec_from_file_location("dataset_gen", "/home/sysadmin/dave/clara/fiscal-clara-data-factory/03_generate_dataset_parallel.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

extract_numbers = module.extract_numbers
validate_number_preservation = module.validate_number_preservation

def test_extract_numbers():
    """Test casi di estrazione numeri."""
    print("="*60)
    print("TEST: extract_numbers()")
    print("="*60)
    
    test_cases = [
        ("Il Superbonus 110% scade il 31/12/2023", 
         {"110%", "31/12/2023"}),
        ("L'Art. 119 del D.L. 34/2020 prevede detrazioni", 
         {"art. 119", "d.l. 34/2020"}),
        ("Pagamento di € 5.000 entro 30 giorni",
         {"€ 5.000", "30 giorni"}),
        ("Comma 3 della L. 178/2020 del 1° gennaio 2021",
         {"comma 3", "l. 178/2020", "1° gennaio 2021"}),
        ("Aliquota del 50 per cento per 10 anni",
         {"50 per cento", "10 anni"}),
    ]
    
    print("\nTest cases:")
    for i, (text, expected) in enumerate(test_cases, 1):
        found = extract_numbers(text)
        # Normalizza per confronto
        expected_norm = {s.lower() for s in expected}
        
        print(f"\n  Test #{i}:")
        print(f"    Input: {text}")
        print(f"    Trovati: {found}")
        print(f"    Attesi: {expected_norm}")
        
        # Check parziale (almeno alcuni dei numeri attesi devono esserci)
        match_count = len(expected_norm.intersection(found))
        if match_count > 0:
            print(f"    ✓ PASS ({match_count}/{len(expected_norm)} numeri matchati)")
        else:
            print(f"    ✗ FAIL (nessun numero matchato)")


def test_validate_preservation():
    """Test casi di validazione preservazione."""
    print("\n" + "="*60)
    print("TEST: validate_number_preservation()")
    print("="*60)
    
    test_cases = [
        # (question, answer, should_pass, description)
        (
            "Qual è la scadenza del Superbonus 110% al 31/12/2023?",
            "Il Superbonus 110% scade il 31/12/2023.",
            True,
            "Tutti i numeri preservati"
        ),
        (
            "Qual è la scadenza del Superbonus 110% al 31/12/2023?",
            "Il Superbonus scade a fine 2023.",
            False,
            "Numeri mancanti (110%, 31/12/2023)"
        ),
        (
            "Cosa prevede l'Art. 119 del D.L. 34/2020?",
            "L'Art. 119 del D.L. 34/2020 regola le detrazioni fiscali.",
            True,
            "Riferimenti normativi preservati"
        ),
        (
            "Quanto si paga secondo il comma 3?",
            "Si paga € 5.000 entro 30 giorni come da comma 3.",
            True,
            "Importi e riferimenti preservati"
        ),
        (
            "Quanto si paga € 5.000 secondo il comma 3?",
            "Si paga cinquemila euro come da comma 3.",
            False,
            "Importo numerico perso (€ 5.000 -> cinquemila)"
        ),
    ]
    
    print("\nTest cases:")
    for i, (q, a, should_pass, desc) in enumerate(test_cases, 1):
        is_valid, missing = validate_number_preservation(q, a)
        
        print(f"\n  Test #{i}: {desc}")
        print(f"    Q: {q}")
        print(f"    A: {a}")
        print(f"    Validazione: {'PASS' if is_valid else 'FAIL'}")
        if not is_valid:
            print(f"    Numeri mancanti: {missing}")
        
        if is_valid == should_pass:
            print(f"    ✓ Comportamento atteso")
        else:
            print(f"    ✗ Comportamento NON atteso (expected {should_pass}, got {is_valid})")


if __name__ == "__main__":
    print("\n" + "🔍 TEST VALIDAZIONE PRESERVAZIONE NUMERI" + "\n")
    
    try:
        test_extract_numbers()
        test_validate_preservation()
        
        print("\n" + "="*60)
        print("✅ Test completati!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE durante i test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
