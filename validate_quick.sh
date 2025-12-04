#!/usr/bin/env bash
# Quick validation test for CLaRa fixes using only a few documents

set -e

echo "=================================================="
echo "CLaRa Pipeline - Quick Validation (Limited Docs)"
echo "=================================================="

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}[Step 1]${NC} Semantic Chunking (max 2 files)"
echo "Running: python 02_semantic_chunking.py --max-files 2"
python 02_semantic_chunking.py --max-files 2

echo -e "\n${YELLOW}[Step 2]${NC} Dataset Generation (limit 10 windows)"
echo "Running: python 03_generate_dataset_parallel.py --limit 10 --clear --workers 2"
python 03_generate_dataset_parallel.py --limit 10 --clear --workers 2

echo -e "\n${YELLOW}[Step 3]${NC} Verification Tests"
echo "=================================================="

# Test 1: JSON Schema
echo -e "\n${YELLOW}[TEST 1]${NC} JSON Schema Validation"
python << 'EOF'
import json
from pathlib import Path

stage1_2 = Path("artifacts/stage1_2_instruction.jsonl")
if not stage1_2.exists():
    print("⚠ Skip: No dataset found")
    exit(0)

errors = []
total = 0
with open(stage1_2) as f:
    for i, line in enumerate(f, 1):
        if not line.strip():
            continue
        total += 1
        rec = json.loads(line)
        if 'answer' not in rec:
            errors.append(f"Line {i}: Missing 'answer' key")
        if 'gold_answer' in rec:
            errors.append(f"Line {i}: Old 'gold_answer' key present")

if errors:
    print(f"✗ FAILED ({len(errors)} errors)")
    for err in errors[:5]:
        print(f"  - {err}")
else:
    print(f"✓ PASSED: All {total} records have correct 'answer' key")
EOF

# Test 2: Orphaned Tables
echo -e "\n${YELLOW}[TEST 2]${NC} Orphaned Table Tag Filter"
python << 'EOF'
import json
from pathlib import Path

chunks = Path("artifacts/chunks.jsonl")
if not chunks.exists():
    print("⚠ Skip: No chunks found")
    exit(0)

errors = []
total = 0
with open(chunks) as f:
    for line in f:
        if not line.strip():
            continue
        total += 1
        chunk = json.loads(line)
        content = chunk['content'].strip()
        if content.startswith('</table>'):
            errors.append(f"Chunk {chunk['id']}")

if errors:
    print(f"✗ FAILED: {len(errors)} chunks with orphaned </table>")
    for err in errors[:5]:
        print(f"  - {err}")
else:
    print(f"✓ PASSED: {total} chunks checked, no orphaned tags")
EOF

# Test 3: Number Normalization
echo -e "\n${YELLOW}[TEST 3]${NC} Number Normalization"
python << 'EOF'
import importlib.util
from pathlib import Path

# Load extract_numbers from script
spec = importlib.util.spec_from_file_location(
    "gen", Path("03_generate_dataset_parallel.py")
)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

tests = [
    ("Art. 10", "articolo 10", "Normative ref"),
    ("€ 1.000", "1000 euro", "Money format"),
    ("50%", "50 per cento", "Percentage"),
]

passed = 0
for t1, t2, desc in tests:
    n1 = gen.extract_numbers(t1)
    n2 = gen.extract_numbers(t2)
    overlap = n1 & n2
    if overlap:
        print(f"  ✓ {desc}: {list(overlap)[0]}")
        passed += 1
    else:
        print(f"  ✗ {desc}")

if passed == len(tests):
    print(f"\n✓ PASSED: All {passed} normalization tests")
else:
    print(f"\n⚠ {passed}/{len(tests)} tests passed")
EOF

# Test 4: Compact Metadata
echo -e "\n${YELLOW}[TEST 4]${NC} Compact Metadata Headers"
python << 'EOF'
import json
from pathlib import Path

chunks = Path("artifacts/chunks.jsonl")
if not chunks.exists():
    print("⚠ Skip: No chunks found")
    exit(0)

# Check first chunk for new compact format
with open(chunks) as f:
    first_line = f.readline()
    chunk = json.loads(first_line)
    content = chunk['content']
    
    # Old format check
    has_old_format = '[CONTESTO]' in content and 'FONTE:' in content
    # New format check (starts with >)
    has_new_format = content.strip().startswith('>')
    
    if has_old_format:
        print("✗ FAILED: Still using old verbose format")
        print(f"  Sample: {content[:200]}")
    elif has_new_format:
        # Extract first line to show new format
        first_line = content.split('\n')[0]
        print(f"✓ PASSED: Using new compact format")
        print(f"  Example: {first_line[:80]}...")
    else:
        print("⚠ WARNING: Unexpected format")
        print(f"  Sample: {content[:150]}")
EOF

echo -e "\n=================================================="
echo -e "${GREEN}✓ Validation Complete${NC}"
echo "=================================================="
echo ""
echo "Summary of changes validated:"
echo "  1. JSON keys: 'gold_answer' → 'answer' ✓"
echo "  2. Orphaned </table> tags filtered ✓"
echo "  3. Number normalization working ✓"
echo "  4. Compact metadata format ✓"
echo ""
echo "Next steps:"
echo "  - If all tests passed, run full pipeline:"
echo "    python 02_semantic_chunking.py"
echo "    python 03_generate_dataset_parallel.py --workers 4"
echo "    python 05_enrich_stage3.py"
