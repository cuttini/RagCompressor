#!/usr/bin/env python3
"""
Chunk Reviewer - Web service for human review of CLaRa training chunks.
Browse and review stage1_raw and stage1_2 chunks.
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configuration
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
STAGE1_RAW = ARTIFACTS_DIR / "stage1_raw.jsonl"
STAGE1_2 = ARTIFACTS_DIR / "stage1_2_instruction.jsonl"
STAGE3_E2E = ARTIFACTS_DIR / "stage3_end_to_end.jsonl"
REVIEW_STATUS_FILE = ARTIFACTS_DIR / "review_status.json"
ITEMS_PER_PAGE = 10

# Global data store
chunks = []
review_status = {}


def load_chunks():
    """Load chunks from all JSONL files."""
    global chunks
    chunks = []
    
    # Load stage1_raw
    if STAGE1_RAW.exists():
        with open(STAGE1_RAW, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    chunk = json.loads(line)
                    chunk['_id'] = f"raw_{idx}"
                    chunk['_stage'] = 'stage1_raw'
                    chunks.append(chunk)
    
    # Load stage1_2
    if STAGE1_2.exists():
        with open(STAGE1_2, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    chunk = json.loads(line)
                    chunk['_id'] = f"s2_{idx}"
                    chunk['_stage'] = 'stage1_2'
                    chunks.append(chunk)
    
    # Load stage3_end_to_end
    if STAGE3_E2E.exists():
        with open(STAGE3_E2E, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if line.strip():
                    chunk = json.loads(line)
                    chunk['_id'] = f"s3_{idx}"
                    chunk['_stage'] = 'stage3_e2e'
                    chunks.append(chunk)
    
    print(f"Loaded {len(chunks)} chunks total")


def load_review_status():
    """Load existing review status from JSON file."""
    global review_status
    if REVIEW_STATUS_FILE.exists():
        with open(REVIEW_STATUS_FILE, 'r', encoding='utf-8') as f:
            review_status = json.load(f)
    else:
        review_status = {}


def save_review_status():
    """Save review status to JSON file."""
    with open(REVIEW_STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(review_status, f, ensure_ascii=False, indent=2)


@app.route('/')
def index():
    """Main chunk browser view."""
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    stage_filter = request.args.get('stage', 'all')
    type_filter = request.args.get('type', 'all')
    status_filter = request.args.get('status', 'all')
    
    # Filter chunks
    filtered = chunks
    
    if stage_filter != 'all':
        filtered = [c for c in filtered if c['_stage'] == stage_filter]
    
    if type_filter != 'all':
        filtered = [c for c in filtered if c.get('data_type') == type_filter]
    
    if status_filter != 'all':
        if status_filter == 'pending':
            filtered = [c for c in filtered if c['_id'] not in review_status]
        else:
            filtered = [c for c in filtered if review_status.get(c['_id']) == status_filter]
    
    # Pagination
    total = len(filtered)
    total_pages = max(1, (total + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    page_chunks = filtered[start:end]
    
    # Add review status to each chunk
    for chunk in page_chunks:
        chunk['_status'] = review_status.get(chunk['_id'], 'pending')
    
    # Stats
    stats = {
        'total': len(chunks),
        'approved': sum(1 for s in review_status.values() if s == 'approved'),
        'rejected': sum(1 for s in review_status.values() if s == 'rejected'),
        'flagged': sum(1 for s in review_status.values() if s == 'flagged'),
    }
    stats['pending'] = stats['total'] - stats['approved'] - stats['rejected'] - stats['flagged']
    
    return render_template('index.html',
                           chunks=page_chunks,
                           page=page,
                           total_pages=total_pages,
                           total=total,
                           stats=stats,
                           stage_filter=stage_filter,
                           type_filter=type_filter,
                           status_filter=status_filter)


@app.route('/api/review', methods=['POST'])
def update_review():
    """Update review status for a chunk."""
    data = request.json
    chunk_id = data.get('chunk_id')
    status = data.get('status')
    
    if not chunk_id or status not in ('approved', 'rejected', 'flagged', 'pending'):
        return jsonify({'error': 'Invalid request'}), 400
    
    if status == 'pending':
        review_status.pop(chunk_id, None)
    else:
        review_status[chunk_id] = status
    
    save_review_status()
    
    # Return updated stats
    stats = {
        'total': len(chunks),
        'approved': sum(1 for s in review_status.values() if s == 'approved'),
        'rejected': sum(1 for s in review_status.values() if s == 'rejected'),
        'flagged': sum(1 for s in review_status.values() if s == 'flagged'),
    }
    stats['pending'] = stats['total'] - stats['approved'] - stats['rejected'] - stats['flagged']
    
    return jsonify({'success': True, 'stats': stats})


@app.route('/api/export')
def export_approved():
    """Export approved chunks as JSONL."""
    approved_chunks = []
    for chunk in chunks:
        if review_status.get(chunk['_id']) == 'approved':
            # Remove internal fields
            export_chunk = {k: v for k, v in chunk.items() if not k.startswith('_')}
            approved_chunks.append(export_chunk)
    
    return jsonify({'count': len(approved_chunks), 'chunks': approved_chunks})


if __name__ == '__main__':
    load_chunks()
    load_review_status()
    print("Starting Chunk Reviewer on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
