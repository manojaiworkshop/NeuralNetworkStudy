#!/usr/bin/env python3
"""
Python tokenization server for C++ inference
Provides proper BERT tokenization via JSON RPC
"""

import sys
import json
from transformers import AutoTokenizer

# Load FinBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')

def tokenize(text, max_length=64):
    """Tokenize text and return input_ids and attention_mask"""
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors=None
    )
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'tokens': tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    }

if __name__ == "__main__":
    # Read from stdin, write to stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            text = request.get('text', '')
            max_length = request.get('max_length', 64)
            
            result = tokenize(text, max_length)
            print(json.dumps(result), flush=True)
            
        except Exception as e:
            error_result = {'error': str(e)}
            print(json.dumps(error_result), flush=True)
