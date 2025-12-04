#!/usr/bin/env python3
"""
Verification script for CLaRa Training Pipeline fixes.

Tests:
1. JSON schema validation (answer key exists)
2. Orphaned table tag detection
3. Italian stemmer functionality
4. Number validation normalization

Run this after implementing all fixes to ensure correctness.
"""

import json
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_json_schema():
    """Test that generated datasets have correct 'answer' key."""
    print("\n[TEST 1] JSON Schema Validation")
    print("=" * 50)
    
    stage1_2_path = Path("artifacts/stage1_2_instruction.jsonl")
    
    if not stage1_2_path.exists():
        print("⚠ Skip: artifacts/stage1_2_instruction.jsonl not found")
        print("  Run: python 03_generate_dataset_parallel.py --limit 5 --clear")
        return
    
    errors = []
    with open(stage1_2_path) as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            
            if 'answer' not in rec:
                errors.append(f"Line {i}: Missing 'answer' key")
            if 'gold_answer' in rec:
                errors.append(f"Line {i}: Old 'gold_answer' key still present")
    
    if errors:
        print("✗ FAILED")
        for err in errors:
            print(f"  - {err}")
    else:
        print("✓ PASSED: All records have correct 'answer' key")


def test_orphaned_tables():
    """Test that no chunks start with orphaned </table> tags."""
    print("\n[TEST 2] Orphaned Table Tag Filter")
    print("=" * 50)
    
    chunks_path = Path("artifacts/chunks.jsonl")
    
    if not chunks_path.exists():
        print("⚠ Skip: artifacts/chunks.jsonl not found")
        print("  Run: python 02_semantic_chunking.py")
        return
    
    errors = []
    with open(chunks_path) as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            content = chunk['content'].strip()
            
            if content.startswith('</table>'):
                errors.append(f"Chunk {chunk['id']}: Starts with orphaned </table>")
    
    if errors:
        print("✗ FAILED")
        for err in errors[:10]:  # Show max 10 errors
            print(f"  - {err}")
    else:
        print("✓ PASSED: No orphaned table tags found")


def test_italian_stemmer():
    """Test that Italian stemmer works correctly."""
    print("\n[TEST 3] Italian BM25 Tokenization")
    print("=" * 50)
    
    try:
        from nltk.stem.snowball import ItalianStemmer
        stemmer = ItalianStemmer()
        
        # Test that inflections map to same stem
        test_cases = [
            (['pagare', 'pagamento', 'pagato'], 'pag'),
            (['fiscale', 'fiscali', 'fiscalità'], 'fiscal'),
            (['dichiarare', 'dichiarazione', 'dichiarato'], 'dichiar'),
        ]
        
        errors = []
        for words, expected_stem_prefix in test_cases:
            stems = [stemmer.stem(w) for w in words]
            # Check that all stems start with expected prefix (stems might have suffixes)
            if not all(s.startswith(expected_stem_prefix) for s in stems):
                errors.append(f"Inflections {words} got different stems: {stems}")
        
        if errors:
            print("✗ FAILED")
            for err in errors:
                print(f"  - {err}")
        else:
            print("✓ PASSED: Italian stemmer working correctly")
            print(f"  Examples: pagare/pagamento/pagato → {stemmer.stem('pagare')}")
    
    except ImportError:
        print("⚠ WARNING: NLTK not installed, using fallback tokenization")
        print("  Install with: pip install nltk")


def test_number_normalization():
    """Test enhanced number validation with normalization."""
    print("\n[TEST 4] Number Validation Normalization")
    print("=" * 50)
    
    # Import from the actual script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_dataset", 
        Path(__file__).parent / "03_generate_dataset_parallel.py"
    )
    gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_module)
    extract_numbers = gen_module.extract_numbers
    
    # Test cases that should now match
    test_cases = [
        # (text1, text2, description)
        ("Art. 10", "articolo 10", "Normative reference abbrev vs full"),
        ("€ 1.000", "1000 euro", "Money with thousands separator"),
        ("1.234,56 euro", "1234.56 euro", "Italian decimal format"),
        ("30 giorni", "30 giorni", "Context numbers"),
        ("50%", "50 per cento", "Percentage formats"),
    ]
    
    passed = 0
    failed = 0
    
    for text1, text2, desc in test_cases:
        nums1 = extract_numbers(text1)
        nums2 = extract_numbers(text2)
        
        # Check if there's overlap (normalization working)
        overlap = nums1 & nums2
        
        if overlap:
            print(f"  ✓ {desc}")
            print(f"    '{text1}' ∩ '{text2}' = {overlap}")
            passed += 1
        else:
            print(f"  ✗ {desc}")
            print(f"    '{text1}' → {nums1}")
            print(f"    '{text2}' → {nums2}")
            failed += 1
    
    if failed == 0:
        print(f"\n✓ PASSED: All {passed} normalization tests passed")
    else:
        print(f"\n⚠ {failed} tests failed, {passed} passed")


def main():
    print("=" * 50)
    print("CLaRa Training Pipeline - Verification Suite")
    print("=" * 50)
    
    test_json_schema()
    test_orphaned_tables()
    test_italian_stemmer()
    test_number_normalization()
    
    print("\n" + "=" * 50)
    print("Verification Complete")
    print("=" * 50)


if __name__ == "__main__":
    main()
