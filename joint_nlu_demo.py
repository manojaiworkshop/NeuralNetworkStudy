#!/usr/bin/env python3
"""
Joint NLU Inference - Interactive Demo
Load trained model and perform complete NLU analysis
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json
from joint_nlu_finbert import JointNLUModel
import re

class JointNLUInference:
    def __init__(self, model_path="joint_nlu_finbert_model.pt", model_name="ProsusAI/finbert"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load vocabularies
        self.intent_to_id = checkpoint['intent_to_id']
        self.slot_to_id = checkpoint['slot_to_id']
        self.entity_to_id = checkpoint['entity_to_id']
        self.id_to_intent = checkpoint['id_to_intent']
        self.id_to_slot = checkpoint['id_to_slot']
        self.id_to_entity = checkpoint['id_to_entity']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = JointNLUModel(
            model_name,
            len(self.intent_to_id),
            len(self.slot_to_id),
            len(self.entity_to_id),
            freeze_bert=True
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"🤖 Joint NLU Model Loaded on {self.device}")
        print(f"📊 Vocabularies: {len(self.intent_to_id)} intents, {len(self.slot_to_id)} slots, {len(self.entity_to_id)} entities")
    
    def analyze(self, text, max_length=64):
        """Perform complete NLU analysis on input text"""
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        offset_mapping = encoded['offset_mapping']
        
        # Model inference
        with torch.no_grad():
            intent_logits, slot_logits, entity_logits = self.model(input_ids, attention_mask)
        
        # Debug: print top predictions
        print(f"\n🔧 Debug Info:")
        print(f"  Input tokens: {len(input_ids[0])} tokens")
        print(f"  Active tokens: {attention_mask.sum().item()} tokens")
        
        # Show top slot predictions
        active_slots = slot_logits[attention_mask.bool()]
        top_slot_ids = torch.topk(active_slots.flatten(), k=5)[1]
        print(f"  Top slot predictions: {[self.id_to_slot.get(idx.item(), 'UNK') for idx in top_slot_ids]}")
        
        # Show top entity predictions  
        active_entities = entity_logits[attention_mask.bool()]
        top_entity_ids = torch.topk(active_entities.flatten(), k=5)[1]
        print(f"  Top entity predictions: {[self.id_to_entity.get(idx.item(), 'UNK') for idx in top_entity_ids]}")
        
        # Get predictions
        intent_pred = torch.argmax(intent_logits, dim=-1).item()
        slot_preds = torch.argmax(slot_logits, dim=-1).squeeze()
        entity_preds = torch.argmax(entity_logits, dim=-1).squeeze()
        
        # Get confidences
        intent_conf = torch.softmax(intent_logits, dim=-1).max().item()
        slot_confs = torch.softmax(slot_logits, dim=-1).max(dim=-1)[0].squeeze()
        entity_confs = torch.softmax(entity_logits, dim=-1).max(dim=-1)[0].squeeze()
        
        # Map predictions to text
        intent_name = self.id_to_intent[intent_pred]
        
        # Align subword tokens back to original text
        tokens_info = self._align_predictions_to_text(
            text, encoded, slot_preds, entity_preds, slot_confs, entity_confs
        )
        
        return {
            'text': text,
            'intent': {
                'name': intent_name,
                'confidence': intent_conf
            },
            'slots': tokens_info['slots'],
            'entities': tokens_info['entities'],
            'tokens': tokens_info['tokens']
        }
    
    def _align_predictions_to_text(self, text, encoded, slot_preds, entity_preds, slot_confs, entity_confs):
        """Align BERT subword predictions back to original text"""
        
        input_ids = encoded['input_ids'].squeeze()
        offset_mapping = encoded['offset_mapping'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # Convert tokens back to text with predictions
        token_predictions = []
        
        for i, (start, end) in enumerate(offset_mapping):
            if not attention_mask[i] or start >= end or start == 0 and end == 0:
                continue  # Skip special tokens and padding
                
            if start < len(text) and end <= len(text):
                token_text = text[start:end]
                slot_id = slot_preds[i].item()
                entity_id = entity_preds[i].item()
                
                slot_name = self.id_to_slot.get(slot_id, 'O')
                entity_name = self.id_to_entity.get(entity_id, 'O')
                
                token_predictions.append({
                    'text': token_text,
                    'start': start,
                    'end': end,
                    'slot': slot_name,
                    'entity': entity_name,
                    'slot_confidence': slot_confs[i].item(),
                    'entity_confidence': entity_confs[i].item()
                })
        
        # Group subword tokens back into words
        words = []
        current_word = {'text': '', 'start': 0, 'end': 0, 'tokens': []}
        
        for token_pred in token_predictions:
            # Check if this token starts a new word (has leading space or is first token)
            if (token_pred['text'].startswith(' ') or 
                token_pred['start'] == 0 or 
                (current_word['tokens'] and token_pred['start'] > current_word['end'])):
                
                # Save previous word if it exists
                if current_word['tokens']:
                    words.append(current_word)
                
                # Start new word
                current_word = {
                    'text': token_pred['text'].strip(),
                    'start': token_pred['start'],
                    'end': token_pred['end'],
                    'tokens': [token_pred]
                }
            else:
                # Continue current word
                current_word['text'] += token_pred['text']
                current_word['end'] = token_pred['end']
                current_word['tokens'].append(token_pred)
        
        # Add the last word
        if current_word['tokens']:
            words.append(current_word)
        
        # Aggregate predictions for each word
        tokens = []
        for word in words:
            # Use the most confident prediction for the word
            best_slot_token = max(word['tokens'], key=lambda t: t['slot_confidence'])
            best_entity_token = max(word['tokens'], key=lambda t: t['entity_confidence'])
            
            tokens.append({
                'text': word['text'],
                'slot': best_slot_token['slot'],
                'entity': best_entity_token['entity'],
                'slot_confidence': best_slot_token['slot_confidence'],
                'entity_confidence': best_entity_token['entity_confidence']
            })
        
        # Extract structured slots and entities
        structured_slots = self._extract_structured_info(tokens, 'slot')
        structured_entities = self._extract_structured_info(tokens, 'entity')
        
        return {
            'tokens': tokens,
            'slots': structured_slots,
            'entities': structured_entities
        }
    
    def _extract_structured_info(self, tokens, info_type):
        """Extract structured slot or entity information"""
        structured = []
        current_entity = None
        current_tokens = []
        
        for token in tokens:
            tag = token[info_type]
            
            if tag.startswith('B-'):
                # Begin new entity
                if current_entity:
                    structured.append({
                        'type': current_entity,
                        'value': ' '.join([t['text'] for t in current_tokens]),
                        'tokens': current_tokens.copy(),
                        'confidence': sum(t[f'{info_type}_confidence'] for t in current_tokens) / len(current_tokens)
                    })
                
                current_entity = tag[2:]
                current_tokens = [token]
                
            elif tag.startswith('I-') and current_entity == tag[2:]:
                # Continue current entity
                current_tokens.append(token)
                
            else:
                # End current entity (O tag or different entity)
                if current_entity:
                    structured.append({
                        'type': current_entity,
                        'value': ' '.join([t['text'] for t in current_tokens]),
                        'tokens': current_tokens.copy(),
                        'confidence': sum(t[f'{info_type}_confidence'] for t in current_tokens) / len(current_tokens)
                    })
                    current_entity = None
                    current_tokens = []
        
        # Handle last entity
        if current_entity:
            structured.append({
                'type': current_entity,
                'value': ' '.join([t['text'] for t in current_tokens]),
                'tokens': current_tokens.copy(),
                'confidence': sum(t[f'{info_type}_confidence'] for t in current_tokens) / len(current_tokens)
            })
        
        return structured

def format_analysis_output(result):
    """Format analysis result for display"""
    print(f"\n🔍 NLU Analysis for: \"{result['text']}\"")
    print("=" * 60)
    
    # Intent
    intent = result['intent']
    print(f"🎯 Intent: {intent['name']} (confidence: {intent['confidence']:.3f})")
    
    # Entities
    if result['entities']:
        print(f"\n🎭 Entities:")
        for entity in result['entities']:
            print(f"  • {entity['type']}: \"{entity['value']}\" (conf: {entity['confidence']:.3f})")
    else:
        print(f"\n🎭 Entities: None detected")
    
    # Slots
    if result['slots']:
        print(f"\n🏷️  Slots:")
        for slot in result['slots']:
            print(f"  • {slot['type']}: \"{slot['value']}\" (conf: {slot['confidence']:.3f})")
    else:
        print(f"\n🏷️  Slots: None detected")
    
    # Token-level analysis
    print(f"\n📝 Token Analysis:")
    for token in result['tokens']:
        slot_display = token['slot'] if token['slot'] != 'O' else 'O'
        entity_display = token['entity'] if token['entity'] not in ['aircraft_code', 'O'] else 'O'
        
        # Show all tokens, but highlight non-O predictions
        if token['slot'] != 'O' or token['entity'] not in ['aircraft_code', 'O']:
            print(f"  \"{token['text']}\" → Slot:{slot_display} ({token['slot_confidence']:.3f}), Entity:{entity_display} ({token['entity_confidence']:.3f})")
        else:
            print(f"  \"{token['text']}\" → O")

def interactive_demo():
    """Interactive demo of the joint NLU system"""
    
    print("🤖 Interactive Joint NLU Demo")
    print("=" * 50)
    print("Enter flight-related queries (or 'quit' to exit)")
    
    try:
        nlu = JointNLUInference()
        
        # Sample queries for demonstration
        sample_queries = [
            "I want to book a flight from New York to Los Angeles on Monday",
            "What's the cheapest flight from Boston to Chicago departing at 8 AM?",
            "Show me flights from Dallas to Miami next Tuesday",
            "I need a one-way ticket from Seattle to Denver",
            "Book me on American Airlines flight 1234 tomorrow morning"
        ]
        
        print("\n💡 Sample queries you can try:")
        for i, query in enumerate(sample_queries, 1):
            print(f"  {i}. {query}")
        
        while True:
            user_input = input("\n🗣️  Enter your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                # Analyze the query
                result = nlu.analyze(user_input)
                format_analysis_output(result)
                
            except Exception as e:
                print(f"❌ Error analyzing query: {e}")
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure you've run the training script first to create joint_nlu_finbert_model.pt")

if __name__ == "__main__":
    interactive_demo()