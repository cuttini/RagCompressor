#!/usr/bin/env python3
"""
Analisi dettagliata dei 33 errori residui di frammentazione tabelle.
"""
import json
from collections import defaultdict
from pathlib import Path

def analyze_residual_errors():
    chunks_file = Path('artifacts/chunks.jsonl')
    
    # Leggi tutti i chunk
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # Trova chunk con errori
    errors_by_type = {
        'unclosed_table': [],  # <table> senza </table>
        'orphan_close': [],    # </table> senza <table>
        'truncated_html': []   # Tag HTML troncati
    }
    
    for chunk in chunks:
        content = chunk['content']
        chunk_id = chunk['id']
        source = chunk.get('source', 'unknown')
        
        # Check 1: Tabelle non chiuse
        if '<table' in content.lower() and '</table>' not in content.lower():
            errors_by_type['unclosed_table'].append({
                'chunk_id': chunk_id,
                'source': source,
                'title': chunk.get('title', 'N/A'),
                'content_preview': content[-300:] if len(content) > 300 else content
            })
        
        # Check 2: Tag di chiusura orfani
        if '</table>' in content.lower() and '<table' not in content.lower():
            # Trova cosa c'è PRIMA del </table>
            table_close_pos = content.lower().find('</table>')
            before_close = content[max(0, table_close_pos-200):table_close_pos]
            
            errors_by_type['orphan_close'].append({
                'chunk_id': chunk_id,
                'source': source,
                'title': chunk.get('title', 'N/A'),
                'before_close': before_close,
                'after_close': content[table_close_pos:table_close_pos+100]
            })
        
        # Check 3: HTML troncato
        if content.strip().endswith(('<td>', '<tr>', '<td', '<tr', '<')):
            errors_by_type['truncated_html'].append({
                'chunk_id': chunk_id,
                'source': source,
                'title': chunk.get('title', 'N/A'),
                'last_chars': content[-150:]
            })
    
    # Analisi per file sorgente
    errors_by_file = defaultdict(lambda: {'unclosed': 0, 'orphan': 0, 'truncated': 0})
    
    for err in errors_by_type['unclosed_table']:
        errors_by_file[err['source']]['unclosed'] += 1
    for err in errors_by_type['orphan_close']:
        errors_by_file[err['source']]['orphan'] += 1
    for err in errors_by_type['truncated_html']:
        errors_by_file[err['source']]['truncated'] += 1
    
    # Report
    print("=" * 80)
    print("ANALISI ERRORI RESIDUI - TABELLE HTML")
    print("=" * 80)
    
    total_errors = (len(errors_by_type['unclosed_table']) + 
                   len(errors_by_type['orphan_close']) + 
                   len(errors_by_type['truncated_html']))
    
    print(f"\n📊 TOTALE ERRORI: {total_errors}\n")
    
    # Errori per tipologia
    print("🔍 ERRORI PER TIPOLOGIA:")
    print(f"  - Tabelle non chiuse (<table> senza </table>): {len(errors_by_type['unclosed_table'])}")
    print(f"  - Chiusure orfane (</table> senza <table>): {len(errors_by_type['orphan_close'])}")
    print(f"  - HTML troncato (tag incompleti): {len(errors_by_type['truncated_html'])}")
    
    # Errori per file
    print(f"\n📁 ERRORI PER FILE SORGENTE ({len(errors_by_file)} file):")
    for source, counts in sorted(errors_by_file.items(), key=lambda x: sum(x[1].values()), reverse=True):
        total = counts['unclosed'] + counts['orphan'] + counts['truncated']
        print(f"  {source[:60]:60s} | Tot:{total:2d} (NC:{counts['unclosed']:2d}, OR:{counts['orphan']:2d}, TR:{counts['truncated']:2d})")
    
    # Dettagli chiusure orfane (più interessanti)
    if errors_by_type['orphan_close']:
        print(f"\n🔎 DETTAGLIO CHIUSURE ORFANE ({len(errors_by_type['orphan_close'])} casi):")
        for err in errors_by_type['orphan_close'][:10]:  # Prime 10
            print(f"\n  📄 {err['source']} - Chunk {err['chunk_id']}")
            print(f"     Titolo: {err['title'][:60]}")
            print(f"     Prima di </table>: ...{err['before_close'][-80:]}")
    
    # Raccomandazioni
    print("\n" + "=" * 80)
    print("💡 RACCOMANDAZIONI")
    print("=" * 80)
    
    # Identifica file più problematici
    top_problem_files = sorted(errors_by_file.items(), 
                               key=lambda x: sum(x[1].values()), 
                               reverse=True)[:5]
    
    print("\n🎯 TOP 5 FILE PROBLEMATICI:")
    for source, counts in top_problem_files:
        total = counts['unclosed'] + counts['orphan'] + counts['truncated']
        print(f"  {source}")
        print(f"    → {total} errori totali - Considera ri-elaborazione OCR")
    
    # Pattern analysis
    orphan_sources = [err['source'] for err in errors_by_type['orphan_close']]
    if orphan_sources:
        from collections import Counter
        most_common = Counter(orphan_sources).most_common(3)
        print(f"\n⚠️ FILE CON PIÙ CHIUSURE ORFANE:")
        for source, count in most_common:
            print(f"  {source}: {count} chunk")
    
    return errors_by_type, errors_by_file

if __name__ == "__main__":
    analyze_residual_errors()
