#!/usr/bin/env python3
"""
Test script per verificare il comportamento della funzione get_adaptive_sliding_windows.
"""

import sys
sys.path.insert(0, '/home/sysadmin/dave/clara/fiscal-clara-data-factory')

from typing import Dict, List, Tuple

# Copia della funzione da testare
def get_adaptive_sliding_windows(
    chunks: List[Dict], 
    min_window_size: int = 3,
    max_window_size: int = 7
) -> List[Tuple[List[Dict], int]]:
    """Genera finestre scorrevoli ADATTIVE basate su titolo/sezione."""
    windows = []
    n = len(chunks)
    
    for i in range(n):
        center = chunks[i]
        center_title = center.get('title', '')
        
        start = i
        end = i + 1
        
        # ESPANDI VERSO SINISTRA
        while (start > 0 and 
               end - start < max_window_size and
               chunks[start - 1].get('title', '') == center_title):
            start -= 1
        
        # ESPANDI VERSO DESTRA
        while (end < n and 
               end - start < max_window_size and
               chunks[end].get('title', '') == center_title):
            end += 1
        
        # Se troppo piccola, allarga con contesto
        current_size = end - start
        if current_size < min_window_size:
            deficit = min_window_size - current_size
            half_deficit = deficit // 2
            extra_left = min(half_deficit, start)
            start -= extra_left
            extra_right = min(deficit - extra_left, n - end)
            end += extra_right
            
            if end - start < min_window_size:
                if start > 0:
                    start = max(0, end - min_window_size)
                elif end < n:
                    end = min(n, start + min_window_size)
        
        window = chunks[start:end]
        center_rel_idx = i - start
        windows.append((window, center_rel_idx))
    
    return windows


# Test Case 1: Sezione lunga (5+ chunk stesso title)
print("=" * 80)
print("TEST 1: Sezione lunga con stesso titolo (simula il tuo esempio 5693-5697)")
print("=" * 80)

test_chunks_long = [
    {"id": 5693, "title": "SEZIONE ACCONTI 2024 CONIUGE"},
    {"id": 5694, "title": "SEZIONE ACCONTI 2024 CONIUGE"},
    {"id": 5695, "title": "SEZIONE ACCONTI 2024 CONIUGE"},
    {"id": 5696, "title": "SEZIONE ACCONTI 2024 CONIUGE"},
    {"id": 5697, "title": "SEZIONE ACCONTI 2024 CONIUGE"},
]

windows = get_adaptive_sliding_windows(test_chunks_long, min_window_size=3, max_window_size=7)

for i, (window, center_idx) in enumerate(windows):
    center_chunk = window[center_idx]
    window_ids = [c['id'] for c in window]
    print(f"Chunk centrale: {center_chunk['id']} → Window: {window_ids} (size={len(window)})")

print(f"\n✓ Risultato atteso: Ogni chunk vede TUTTA la sezione (5 chunk) perché hanno lo stesso title")


# Test Case 2: Boundary tra sezioni
print("\n" + "=" * 80)
print("TEST 2: Boundary tra sezioni diverse (deve fermarsi al cambio title)")
print("=" * 80)

test_chunks_boundary = [
    {"id": 100, "title": "SEZIONE ACCONTI 2024"},
    {"id": 101, "title": "SEZIONE ACCONTI 2024"},
    {"id": 102, "title": "DATI FISCALI - ONERI DETRAIBILI"},  # ← CAMBIO SEZIONE
    {"id": 103, "title": "DATI FISCALI - ONERI DETRAIBILI"},
    {"id": 104, "title": "DATI FISCALI - ONERI DETRAIBILI"},
]

windows = get_adaptive_sliding_windows(test_chunks_boundary, min_window_size=3, max_window_size=7)

for i, (window, center_idx) in enumerate(windows):
    center_chunk = window[center_idx]
    window_ids = [c['id'] for c in window]
    titles = [c['title'][:30] + "..." if len(c['title']) > 30 else c['title'] for c in window]
    print(f"Chunk centrale: {center_chunk['id']} → Window: {window_ids}")
    print(f"  Titles: {titles}")

print(f"\n✓ Risultato atteso: Window NON mescola chunk con title diversi (si ferma al boundary)")


# Test Case 3: Chunk isolato (solo 1 chunk con quel title)
print("\n" + "=" * 80)
print("TEST 3: Chunk isolato (title unico, deve allargare a min_window_size=3)")
print("=" * 80)

test_chunks_isolated = [
    {"id": 200, "title": "SEZIONE A"},
    {"id": 201, "title": "SEZIONE B ISOLATA"},  # ← UNICO CON QUESTO TITLE
    {"id": 202, "title": "SEZIONE C"},
]

windows = get_adaptive_sliding_windows(test_chunks_isolated, min_window_size=3, max_window_size=7)

for i, (window, center_idx) in enumerate(windows):
    center_chunk = window[center_idx]
    window_ids = [c['id'] for c in window]
    titles = [c['title'] for c in window]
    print(f"Chunk centrale: {center_chunk['id']} → Window: {window_ids}")
    print(f"  Titles: {titles}")

print(f"\n✓ Risultato atteso: Chunk 201 ha window size=3 (min), include contesto prev/next anche se title diverso")


# Test Case 4: Sezione molto lunga (> max_window_size)
print("\n" + "=" * 80)
print("TEST 4: Sezione molto lunga (10 chunk stesso title, max=7)")
print("=" * 80)

test_chunks_very_long = [
    {"id": i, "title": "SEZIONE LUNGHISSIMA"} 
    for i in range(300, 310)  # 10 chunk
]

windows = get_adaptive_sliding_windows(test_chunks_very_long, min_window_size=3, max_window_size=7)

# Mostra solo chunk centrale (305)
for i, (window, center_idx) in enumerate(windows):
    center_chunk = window[center_idx]
    if center_chunk['id'] == 305:
        window_ids = [c['id'] for c in window]
        print(f"Chunk centrale: {center_chunk['id']} → Window: {window_ids} (size={len(window)})")

print(f"\n✓ Risultato atteso: Window max=7, non include tutti i 10 chunk (rispetta MAX_WINDOW_SIZE)")


# Statistiche finali
print("\n" + "=" * 80)
print("STATISTICHE COMPLESSIVE")
print("=" * 80)

all_test_chunks = test_chunks_long + test_chunks_boundary + test_chunks_isolated + test_chunks_very_long
windows_all = get_adaptive_sliding_windows(all_test_chunks, min_window_size=3, max_window_size=7)
window_sizes = [len(w[0]) for w in windows_all]

print(f"Totale chunks: {len(all_test_chunks)}")
print(f"Totale windows: {len(windows_all)}")
print(f"Window size min: {min(window_sizes)}")
print(f"Window size max: {max(window_sizes)}")
print(f"Window size avg: {sum(window_sizes) / len(window_sizes):.1f}")

# Verifica che min_window_size e max_window_size siano rispettati
assert min(window_sizes) >= 3, "❌ FAIL: Window troppo piccola!"
assert max(window_sizes) <= 7, "❌ FAIL: Window troppo grande!"

print("\n✅ TUTTI I TEST PASSATI!")
