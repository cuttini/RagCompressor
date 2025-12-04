#!/usr/bin/env python3
"""
Script di validazione per verificare che i chunk non contengano tabelle frammentate.
"""
import json
import re
from pathlib import Path

def validate_chunk_tables(chunks_file: Path):
    """Valida che i chunk non contengano tabelle HTML frammentate."""
    
    errors = []
    warnings = []
    table_chunks = []
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            content = chunk['content']
            chunk_id = chunk['id']
            source = chunk.get('source', 'unknown')
            
            # Check 1: Tabelle non chiuse (tag <table> senza </table>)
            if '<table' in content.lower() and '</table>' not in content.lower():
                errors.append({
                    'chunk_id': chunk_id,
                    'source': source,
                    'error': 'Tabella non chiusa (tag aperti senza chiusura)',
                    'severity': 'HIGH'
                })
            
            # Check 2: Contenuto HTML troncato (tag TD/TR aperti alla fine)
            if re.search(r'<t[dr][^>]*>\s*$', content.strip(), re.IGNORECASE):
                errors.append({
                    'chunk_id': chunk_id,
                    'source': source,
                    'error': 'Contenuto HTML troncato (tag TD/TR alla fine)',
                    'severity': 'HIGH'
                })
            
            # Check 3: Mezza tabella (</table> senza <table>)
            if '</table>' in content.lower() and '<table' not in content.lower():
                errors.append({
                    'chunk_id': chunk_id,
                    'source': source,
                    'error': 'Fine tabella senza inizio',
                    'severity': 'HIGH'
                })
            
            # Statistiche: conta chunk con tabelle complete
            if '<table' in content.lower() and '</table>' in content.lower():
                table_count = content.lower().count('<table')
                table_chunks.append({
                    'chunk_id': chunk_id,
                    'source': source,
                    'title': chunk.get('title', 'N/A'),
                    'table_count': table_count
                })
    
    # Report
    print("=" * 80)
    print("VALIDAZIONE CHUNK - TABELLE HTML")
    print("=" * 80)
    
    if errors:
        print(f"\n❌ ERRORI TROVATI: {len(errors)}")
        for err in errors[:10]:  # Mostra primi 10
            print(f"  - Chunk {err['chunk_id']} ({err['source']}): {err['error']}")
        if len(errors) > 10:
            print(f"  ... e altri {len(errors) - 10} errori")
    else:
        print("\n✅ NESSUN ERRORE: Nessuna tabella frammentata trovata!")
    
    print(f"\n📊 STATISTICHE:")
    print(f"  - Chunk totali processati: {chunk_id + 1}")
    print(f"  - Chunk con tabelle complete: {len(table_chunks)}")
    
    if table_chunks:
        print(f"\n📋 CHUNK CON TABELLE COMPLETE:")
        # Filtra solo il file problematico
        fiscal_tables = [t for t in table_chunks if 'Accordo_Ristrutturazione' in t['source']]
        if fiscal_tables:
            print(f"  File: Accordo_Ristrutturazione_Debiti_Codice_Crisi_2023.md")
            for t in fiscal_tables:
                print(f"    - Chunk {t['chunk_id']:4d}: {t['title'][:60]} ({t['table_count']} tabelle)")
    
    return len(errors) == 0

if __name__ == "__main__":
    chunks_path = Path(__file__).parent / "artifacts" / "chunks.jsonl"
    
    if not chunks_path.exists():
        print(f"❌ File non trovato: {chunks_path}")
        exit(1)
    
    success = validate_chunk_tables(chunks_path)
    exit(0 if success else 1)
