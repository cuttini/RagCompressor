#!/usr/bin/env python3
"""
Test script to measure VLLM inference performance (tokens/second)
"""

import time
import requests
import json

# VLLM endpoint configuration
VLLM_BASE_URL = "http://192.168.2.184:5000/v1"

def test_inference():
    """Perform a test inference and calculate tokens/second"""
    
    # Test prompt - using a moderately sized prompt
    test_prompt = """Analizza il seguente testo normativo e rispondi alla domanda.

Testo: L'articolo 12 del decreto legislativo n. 471/1997 stabilisce che in caso di omesso versamento delle ritenute alla fonte, si applica una sanzione amministrativa pari al 30% dell'importo non versato. La sanzione può essere ridotta mediante ravvedimento operoso entro i termini previsti dalla legge.

Domanda: Qual è la percentuale della sanzione per omesso versamento delle ritenute alla fonte?"""

    # Prepare the request
    url = f"{VLLM_BASE_URL}/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen3-32B-AWQ",  # Model available on the VLLM endpoint
        "prompt": test_prompt,
        "max_tokens": 500,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    print("=" * 80)
    print("VLLM Inference Performance Test")
    print("=" * 80)
    print(f"\nEndpoint: {VLLM_BASE_URL}")
    print(f"Prompt length: {len(test_prompt)} characters")
    print("\nSending request...")
    
    # Measure inference time
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Parse response
        result = response.json()
        
        # Extract completion and usage stats
        if "choices" in result and len(result["choices"]) > 0:
            completion_text = result["choices"][0].get("text", "")
            print(f"\n{'='*80}")
            print("Generated Response:")
            print(f"{'='*80}")
            print(completion_text)
            print(f"{'='*80}")
        
        # Calculate tokens/second
        if "usage" in result:
            usage = result["usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            print(f"\n{'='*80}")
            print("Performance Metrics:")
            print(f"{'='*80}")
            print(f"Prompt tokens:      {prompt_tokens}")
            print(f"Completion tokens:  {completion_tokens}")
            print(f"Total tokens:       {total_tokens}")
            print(f"Elapsed time:       {elapsed_time:.2f} seconds")
            print(f"\nTokens/second:      {total_tokens / elapsed_time:.2f}")
            print(f"Generation speed:   {completion_tokens / elapsed_time:.2f} tokens/sec")
            print(f"{'='*80}")
            
        else:
            print("\nWarning: No usage statistics in response")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print("\nFull response:")
            print(json.dumps(result, indent=2))
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error during inference: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return None

if __name__ == "__main__":
    test_inference()
