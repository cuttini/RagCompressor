#!/usr/bin/env python3
"""
Sanity check for chunks.jsonl to identify involuntary splitting patterns.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

# Read all chunks
chunks_file = Path("artifacts/chunks.jsonl")
chunks = []

print("Loading chunks...")
with open(chunks_file, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks\n")

# Analysis patterns
issues = defaultdict(list)

def get_actual_text(content):
    """Extract actual text content after metadata header."""
    # Remove metadata header
    content = re.sub(r'^>.*?<\s*', '', content, flags=re.MULTILINE)
    # Remove markdown headers
    content = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
    return content.strip()

def starts_with_continuation(text):
    """Check if text starts with lowercase or continuation words."""
    text = text.lstrip()
    if not text:
        return False
    
    # Check if starts with lowercase
    if text[0].islower():
        return True
    
    # Check for continuation words
    continuation_words = [
        'che', 'di cui', 'inoltre', 'tuttavia', 'quindi', 'pertanto',
        'il quale', 'la quale', 'dei', 'delle', 'degli', 'dal', 'dalla',
        'e', 'o', 'ma', 'però', 'tuttavia', 'oppure'
    ]
    
    for word in continuation_words:
        if re.match(rf'^{word}\b', text, re.IGNORECASE):
            return True
    
    return False

def ends_mid_sentence(text):
    """Check if text ends without proper punctuation."""
    text = text.rstrip()
    if not text:
        return False
    
    # Check last character
    if text[-1] in ['.', '!', '?', ':', ';']:
        return False
    
    # Check if ends with incomplete structure
    if re.search(r'[,;]\s*$', text):
        return True
    
    return True

print("="*80)
print("ANALYSIS 1: Consecutive Same-Title Chunks")
print("="*80)

consecutive_titles = []
for i in range(len(chunks) - 1):
    if chunks[i]['title'] == chunks[i+1]['title']:
        consecutive_titles.append({
            'title': chunks[i]['title'],
            'chunk1_id': chunks[i]['id'],
            'chunk2_id': chunks[i+1]['id'],
            'source': chunks[i]['source']
        })

print(f"Found {len(consecutive_titles)} cases of consecutive same-title chunks")
if consecutive_titles:
    print("\nExamples (first 10):")
    for item in consecutive_titles[:10]:
        print(f"  - {item['source']}: '{item['title']}' (IDs: {item['chunk1_id']}, {item['chunk2_id']})")

issues['consecutive_same_title'] = consecutive_titles

print("\n" + "="*80)
print("ANALYSIS 2: Chunks Starting with Continuation Patterns")
print("="*80)

continuation_starts = []
for chunk in chunks:
    actual_text = get_actual_text(chunk['content'])
    if starts_with_continuation(actual_text):
        continuation_starts.append({
            'id': chunk['id'],
            'title': chunk['title'],
            'source': chunk['source'],
            'start_text': actual_text[:100]
        })

print(f"Found {len(continuation_starts)} chunks starting with continuation patterns")
if continuation_starts:
    print("\nExamples (first 10):")
    for item in continuation_starts[:10]:
        print(f"  - ID {item['id']}: {item['title']}")
        print(f"    Starts: {item['start_text'][:80]}...")

issues['continuation_starts'] = continuation_starts

print("\n" + "="*80)
print("ANALYSIS 3: Chunks Ending Mid-Sentence")
print("="*80)

mid_sentence_ends = []
for chunk in chunks:
    actual_text = get_actual_text(chunk['content'])
    if ends_mid_sentence(actual_text):
        mid_sentence_ends.append({
            'id': chunk['id'],
            'title': chunk['title'],
            'source': chunk['source'],
            'end_text': actual_text[-100:] if len(actual_text) > 100 else actual_text
        })

print(f"Found {len(mid_sentence_ends)} chunks ending mid-sentence")
if mid_sentence_ends:
    print("\nExamples (first 10):")
    for item in mid_sentence_ends[:10]:
        print(f"  - ID {item['id']}: {item['title']}")
        print(f"    Ends: ...{item['end_text'][-80:]}")

issues['mid_sentence_ends'] = mid_sentence_ends

print("\n" + "="*80)
print("ANALYSIS 4: Chunks with Orphaned Legal Subsections")
print("="*80)

# Look for patterns like "a)" at very start or "continua da precedente"
orphaned_subsections = []
for chunk in chunks:
    actual_text = get_actual_text(chunk['content'])
    
    # Check if starts with isolated legal marker
    if re.match(r'^\*?[a-z]\)\s*\*?\s+[a-z]', actual_text, re.IGNORECASE):
        orphaned_subsections.append({
            'id': chunk['id'],
            'title': chunk['title'],
            'source': chunk['source'],
            'start_text': actual_text[:100]
        })

print(f"Found {len(orphaned_subsections)} chunks with potential orphaned subsections")
if orphaned_subsections:
    print("\nExamples (first 10):")
    for item in orphaned_subsections[:10]:
        print(f"  - ID {item['id']}: {item['title']}")
        print(f"    Starts: {item['start_text'][:80]}...")

issues['orphaned_subsections'] = orphaned_subsections

print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)
print(f"Total chunks analyzed: {len(chunks)}")
print(f"\nIssues found:")
print(f"  1. Consecutive same-title chunks: {len(consecutive_titles)}")
print(f"  2. Continuation pattern starts: {len(continuation_starts)}")
print(f"  3. Mid-sentence endings: {len(mid_sentence_ends)}")
print(f"  4. Orphaned subsections: {len(orphaned_subsections)}")

print(f"\nPercentages:")
print(f"  1. {len(consecutive_titles)/len(chunks)*100:.2f}% have consecutive same title")
print(f"  2. {len(continuation_starts)/len(chunks)*100:.2f}% start with continuation")
print(f"  3. {len(mid_sentence_ends)/len(chunks)*100:.2f}% end mid-sentence")
print(f"  4. {len(orphaned_subsections)/len(chunks)*100:.2f}% have orphaned subsections")

# Save detailed report
report_file = Path("artifacts/chunking_sanity_check.json")
with open(report_file, "w", encoding="utf-8") as f:
    json.dump({
        'total_chunks': len(chunks),
        'issues': {
            'consecutive_same_title': {
                'count': len(consecutive_titles),
                'examples': consecutive_titles[:20]
            },
            'continuation_starts': {
                'count': len(continuation_starts),
                'examples': continuation_starts[:20]
            },
            'mid_sentence_ends': {
                'count': len(mid_sentence_ends),
                'examples': mid_sentence_ends[:20]
            },
            'orphaned_subsections': {
                'count': len(orphaned_subsections),
                'examples': orphaned_subsections[:20]
            }
        }
    }, f, ensure_ascii=False, indent=2)

print(f"\n✓ Detailed report saved to: {report_file}")
