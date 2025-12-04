#!/usr/bin/env python3
"""Debug script to inspect problematic chunks"""
import json

with open('artifacts/chunks.jsonl') as f:
    for line in f:
        chunk = json.loads(line)
        if chunk['id'] == 35:
            content = chunk['content']
            
            print("=== CHUNK 35 ===")
            print(f"Title: {chunk['title']}")
            print(f"Source: {chunk['source']}")
            print(f"\nContent length: {len(content)}")
            
            # Count tables
            table_opens = content.count('<table')
            table_closes = content.count('</table>')
            print(f"<table> tags: {table_opens}")
            print(f"</table> tags: {table_closes}")
            
            # Show last 500 chars
            print(f"\n... LAST 500 CHARS ...\n{content[-500:]}")
            break
