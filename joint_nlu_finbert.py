#!/usr/bin/env python3
"""
Joint Intent + Slot + Entity Detection with FinBERT
Multi-task learning for complete NLU pipeline
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup,
    AutoConfig
)
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import os
import logging
from collections import Counter
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JointATISDataset(Dataset):
    """Dataset for joint intent + slot + entity detection"""
    
    def __init__(self, data_path, tokenizer, intent_to_id=None, slot_to_id=None, entity_to_id=None, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Build vocabularies
        if intent_to_id is None:
            self.intent_to_id, self.slot_to_id, self.entity_to_id = self._build_vocabularies(raw_data)
        else:
            self.intent_to_id = intent_to_id
            self.slot_to_id = slot_to_id
            self.entity_to_id = entity_to_id
        
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
        self.id_to_slot = {i: slot for slot, i in self.slot_to_id.items()}
        self.id_to_entity = {i: entity for entity, i in self.entity_to_id.items()}
        
        # Process data
        for item in raw_data:
            if item['intent'] in self.intent_to_id:
                processed_item = self._process_item(item)
                if processed_item:
                    self.data.append(processed_item)
        
        logger.info(f"Loaded {len(self.data)} examples")
        logger.info(f"Vocabularies - Intents: {len(self.intent_to_id)}, Slots: {len(self.slot_to_id)}, Entities: {len(self.entity_to_id)}")
    
    def _build_vocabularies(self, raw_data):
        """Build intent, slot, and entity vocabularies"""
        intents = []
        slots = []
        entities = []
        
        for item in raw_data:
            intents.append(item['intent'])
            slots.extend(item['slots'])
            
            # Extract entities from slot tags
            for slot in item['slots']:
                if slot.startswith('B-') or slot.startswith('I-'):
                    entity_type = slot[2:]  # Remove B- or I- prefix
                    entities.append(entity_type)
        
        # Filter rare classes
        intent_counts = Counter(intents)
        slot_counts = Counter(slots)
        entity_counts = Counter(entities)
        
        filtered_intents = [intent for intent, count in intent_counts.items() if count >= 2]
        filtered_slots = [slot for slot, count in slot_counts.items() if count >= 1]
        filtered_entities = [entity for entity, count in entity_counts.items() if count >= 1]
        
        intent_to_id = {intent: i for i, intent in enumerate(sorted(filtered_intents))}
        slot_to_id = {slot: i for i, slot in enumerate(sorted(filtered_slots))}
        entity_to_id = {entity: i for i, entity in enumerate(sorted(filtered_entities))}
        
        return intent_to_id, slot_to_id, entity_to_id
    
    def _process_item(self, item):
        """Process a single item and align tokens with labels"""
        text = item['text']
        intent = item['intent']
        tokens = item['tokens']
        slots = item['slots']
        
        if len(tokens) != len(slots):
            return None
        
        # Tokenize with BERT tokenizer
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        offset_mapping = encoded['offset_mapping'].squeeze()
        
        # Align original tokens with BERT subword tokens
        slot_labels = self._align_tokens_with_labels(tokens, slots, offset_mapping, text)
        entity_labels = self._extract_entity_labels(slot_labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'intent_label': self.intent_to_id.get(intent, 0),
            'slot_labels': slot_labels,
            'entity_labels': entity_labels,
            'text': text,
            'original_tokens': tokens,
            'original_slots': slots
        }
    
    def _align_tokens_with_labels(self, tokens, slots, offset_mapping, text):
        """Align BERT subword tokens with original slot labels"""
        slot_labels = [0] * len(offset_mapping)  # Default to first slot (usually O)
        
        # Create character-to-slot mapping
        char_to_slot = [0] * len(text)
        char_pos = 0
        
        for token, slot in zip(tokens, slots):
            token_start = text.find(token, char_pos)
            if token_start != -1:
                slot_id = self.slot_to_id.get(slot, 0)
                for i in range(token_start, min(token_start + len(token), len(char_to_slot))):
                    char_to_slot[i] = slot_id
                char_pos = token_start + len(token)
        
        # Map to BERT tokens
        for i, (start, end) in enumerate(offset_mapping):
            if start < len(char_to_slot) and end <= len(char_to_slot) and start < end:
                slot_labels[i] = char_to_slot[start]
        
        return slot_labels
    
    def _extract_entity_labels(self, slot_labels):
        """Extract entity labels from slot labels"""
        entity_labels = [0] * len(slot_labels)
        
        for i, slot_id in enumerate(slot_labels):
            slot_name = self.id_to_slot.get(slot_id, "O")
            if slot_name.startswith('B-') or slot_name.startswith('I-'):
                entity_type = slot_name[2:]
                entity_id = self.entity_to_id.get(entity_type, 0)
                entity_labels[i] = entity_id
        
        return entity_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class JointNLUModel(nn.Module):
    """Joint model for intent classification, slot detection, and entity recognition"""
    
    def __init__(self, model_name, num_intents, num_slots, num_entities, dropout=0.1, freeze_bert=True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific heads
        self.intent_classifier = nn.Linear(self.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.config.hidden_size, num_slots)
        self.entity_classifier = nn.Linear(self.config.hidden_size, num_entities)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get representations
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        # Task predictions
        intent_logits = self.intent_classifier(pooled_output)  # Sentence-level
        slot_logits = self.slot_classifier(sequence_output)    # Token-level
        entity_logits = self.entity_classifier(sequence_output)  # Token-level
        
        return intent_logits, slot_logits, entity_logits

def compute_joint_loss(intent_logits, slot_logits, entity_logits, 
                      intent_labels, slot_labels, entity_labels, 
                      attention_mask, intent_weight=0.4, slot_weight=0.3, entity_weight=0.3):
    """Compute weighted joint loss"""
    loss_fct = nn.CrossEntropyLoss()
    
    # Intent loss (sentence level)
    intent_loss = loss_fct(intent_logits, intent_labels)
    
    # Slot loss (token level, masked)
    active_positions = attention_mask.view(-1) == 1
    active_slot_logits = slot_logits.view(-1, slot_logits.size(-1))[active_positions]
    active_slot_labels = slot_labels.view(-1)[active_positions]
    slot_loss = loss_fct(active_slot_logits, active_slot_labels)
    
    # Entity loss (token level, masked)
    active_entity_logits = entity_logits.view(-1, entity_logits.size(-1))[active_positions]
    active_entity_labels = entity_labels.view(-1)[active_positions]
    entity_loss = loss_fct(active_entity_logits, active_entity_labels)
    
    # Weighted total loss
    total_loss = intent_weight * intent_loss + slot_weight * slot_loss + entity_weight * entity_loss
    
    return total_loss, intent_loss, slot_loss, entity_loss

def evaluate_joint_model(model, dataloader, device, dataset):
    """Evaluate the joint model on all tasks"""
    model.eval()
    
    intent_correct = 0
    slot_correct = 0
    entity_correct = 0
    total_samples = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
            intent_labels = torch.tensor([item['intent_label'] for item in batch]).to(device)
            slot_labels = torch.stack([torch.tensor(item['slot_labels']) for item in batch]).to(device)
            entity_labels = torch.stack([torch.tensor(item['entity_labels']) for item in batch]).to(device)
            
            intent_logits, slot_logits, entity_logits = model(input_ids, attention_mask)
            
            # Intent accuracy
            intent_preds = torch.argmax(intent_logits, dim=-1)
            intent_correct += (intent_preds == intent_labels).sum().item()
            total_samples += len(intent_labels)
            
            # Slot accuracy (only active positions)
            active_positions = attention_mask == 1
            slot_preds = torch.argmax(slot_logits, dim=-1)
            slot_correct += ((slot_preds == slot_labels) & active_positions).sum().item()
            
            # Entity accuracy (only active positions)  
            entity_preds = torch.argmax(entity_logits, dim=-1)
            entity_correct += ((entity_preds == entity_labels) & active_positions).sum().item()
            
            total_tokens += active_positions.sum().item()
    
    intent_acc = intent_correct / total_samples * 100
    slot_acc = slot_correct / total_tokens * 100
    entity_acc = entity_correct / total_tokens * 100
    
    return intent_acc, slot_acc, entity_acc

def main():
    print("🤖 Joint Intent + Slot + Entity Detection with FinBERT")
    print("=" * 60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    model_name = "ProsusAI/finbert"
    print(f"\n📥 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets
    print("\n📚 Loading ATIS datasets...")
    train_dataset = JointATISDataset('data/train.json', tokenizer, max_length=64)
    val_dataset = JointATISDataset('data/validation.json', tokenizer, 
                                   train_dataset.intent_to_id, 
                                   train_dataset.slot_to_id, 
                                   train_dataset.entity_to_id, max_length=64)
    test_dataset = JointATISDataset('data/test.json', tokenizer,
                                    train_dataset.intent_to_id,
                                    train_dataset.slot_to_id,
                                    train_dataset.entity_to_id, max_length=64)
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"🎯 Vocabularies: Intents={len(train_dataset.intent_to_id)}, Slots={len(train_dataset.slot_to_id)}, Entities={len(train_dataset.entity_to_id)}")
    
    # Show some examples of each task
    print(f"\n📈 Sample Intent Distribution:")
    train_intents = [train_dataset.data[i]['intent_label'] for i in range(min(1000, len(train_dataset)))]
    intent_counts = Counter([train_dataset.id_to_intent[i] for i in train_intents])
    for intent, count in intent_counts.most_common(5):
        print(f"  {intent}: {count}")
    
    print(f"\n🏷️  Sample Slot Types:")
    for i, (slot, slot_id) in enumerate(list(train_dataset.slot_to_id.items())[:10]):
        print(f"  {slot}")
    
    print(f"\n🎭 Sample Entity Types:")
    for i, (entity, entity_id) in enumerate(list(train_dataset.entity_to_id.items())[:10]):
        print(f"  {entity}")
    
    # Create data loaders
    batch_size = 16  # Smaller batch size due to sequence labeling
    
    def collate_fn(batch):
        return batch
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    print(f"\n🏗️  Initializing Joint NLU Model...")
    model = JointNLUModel(
        model_name, 
        len(train_dataset.intent_to_id),
        len(train_dataset.slot_to_id), 
        len(train_dataset.entity_to_id),
        dropout=0.1, 
        freeze_bert=True
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Training setup
    epochs = 15
    lr = 1e-3
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    print(f"\n🚀 Training Details:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Loss weights: Intent=0.4, Slot=0.3, Entity=0.3")
    
    # Training loop
    print(f"\n🔥 Starting Joint Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        intent_loss_sum = 0
        slot_loss_sum = 0
        entity_loss_sum = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Prepare batch tensors
            input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
            intent_labels = torch.tensor([item['intent_label'] for item in batch]).to(device)
            slot_labels = torch.stack([torch.tensor(item['slot_labels']) for item in batch]).to(device)
            entity_labels = torch.stack([torch.tensor(item['entity_labels']) for item in batch]).to(device)
            
            # Forward pass
            intent_logits, slot_logits, entity_logits = model(input_ids, attention_mask)
            
            # Compute joint loss
            loss, intent_loss, slot_loss, entity_loss = compute_joint_loss(
                intent_logits, slot_logits, entity_logits,
                intent_labels, slot_labels, entity_labels,
                attention_mask
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            intent_loss_sum += intent_loss.item()
            slot_loss_sum += slot_loss.item()
            entity_loss_sum += entity_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Intent': f'{intent_loss.item():.3f}',
                'Slot': f'{slot_loss.item():.3f}',
                'Entity': f'{entity_loss.item():.3f}'
            })
        
        # Validation
        if (epoch + 1) % 3 == 0:  # Evaluate every 3 epochs
            intent_acc, slot_acc, entity_acc = evaluate_joint_model(model, val_loader, device, val_dataset)
            print(f"\nEpoch {epoch+1} Validation:")
            print(f"  Intent Accuracy: {intent_acc:.2f}%")
            print(f"  Slot Accuracy: {slot_acc:.2f}%")
            print(f"  Entity Accuracy: {entity_acc:.2f}%")
            print(f"  Avg Loss: {total_loss/num_batches:.4f}")
            
            # Early stopping if all tasks are performing well
            if intent_acc > 75 and slot_acc > 70 and entity_acc > 65:
                print(f"🎯 Early stopping - good performance achieved!")
                break
    
    # Final evaluation
    print(f"\n🎯 Final Evaluation:")
    intent_acc, slot_acc, entity_acc = evaluate_joint_model(model, test_loader, device, test_dataset)
    print(f"  Intent Accuracy: {intent_acc:.2f}%")
    print(f"  Slot Accuracy: {slot_acc:.2f}%") 
    print(f"  Entity Accuracy: {entity_acc:.2f}%")
    
    # Show sample predictions
    print(f"\n🔍 Sample Joint Predictions:")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 1:  # Just show first batch
                break
                
            input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
            
            intent_logits, slot_logits, entity_logits = model(input_ids, attention_mask)
            
            for j in range(min(3, len(batch))):
                item = batch[j]
                
                # Predictions
                intent_pred = torch.argmax(intent_logits[j]).item()
                slot_preds = torch.argmax(slot_logits[j], dim=-1)
                entity_preds = torch.argmax(entity_logits[j], dim=-1)
                
                # Get text and tokens
                text = item['text']
                tokens = item['original_tokens']
                true_intent = train_dataset.id_to_intent[item['intent_label']]
                pred_intent = train_dataset.id_to_intent[intent_pred]
                
                print(f"\n{j+1}. Text: \"{text}\"")
                print(f"   Intent: {pred_intent} (true: {true_intent})")
                
                # Show first few token predictions
                print("   Token predictions:")
                for k in range(min(8, len(tokens))):
                    if k < len(slot_preds):
                        slot_pred = train_dataset.id_to_slot.get(slot_preds[k].item(), 'UNK')
                        entity_pred = train_dataset.id_to_entity.get(entity_preds[k].item(), 'UNK')
                        true_slot = item['original_slots'][k] if k < len(item['original_slots']) else 'O'
                        print(f"     {tokens[k]} → Slot:{slot_pred} Entity:{entity_pred} (true:{true_slot})")
    
    # Save model
    model_save_path = "joint_nlu_finbert_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'intent_to_id': train_dataset.intent_to_id,
        'slot_to_id': train_dataset.slot_to_id,
        'entity_to_id': train_dataset.entity_to_id,
        'id_to_intent': train_dataset.id_to_intent,
        'id_to_slot': train_dataset.id_to_slot,
        'id_to_entity': train_dataset.id_to_entity,
        'model_name': model_name,
        'num_intents': len(train_dataset.intent_to_id),
        'num_slots': len(train_dataset.slot_to_id),
        'num_entities': len(train_dataset.entity_to_id)
    }, model_save_path)
    
    print(f"\n💾 Joint model saved to: {model_save_path}")
    print(f"🎉 Training completed!")
    
    return intent_acc, slot_acc, entity_acc

if __name__ == "__main__":
    main()