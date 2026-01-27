#!/usr/bin/env python3
"""
Fast FinBERT Intent Classification - Optimized Version
Better data alignment and reduced overfitting
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ATISIntentDataset(Dataset):
    def __init__(self, data_path, tokenizer, intent_to_id=None, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Build or use existing intent vocabulary
        if intent_to_id is None:
            intents = [item['intent'] for item in raw_data]
            intent_counts = Counter(intents)
            
            # Only keep intents with at least 2 examples to avoid rare classes
            filtered_intents = [intent for intent, count in intent_counts.items() if count >= 2]
            
            self.intent_to_id = {intent: i for i, intent in enumerate(sorted(filtered_intents))}
            self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
        else:
            self.intent_to_id = intent_to_id
            self.id_to_intent = {i: intent for intent, i in intent_to_id.items()}
        
        # Filter data to only include known intents
        for item in raw_data:
            if item['intent'] in self.intent_to_id:
                self.data.append({
                    'text': item['text'],
                    'intent': self.intent_to_id[item['intent']],
                    'intent_name': item['intent']
                })
        
        logger.info(f"Loaded {len(self.data)} examples with {len(self.intent_to_id)} intents")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['intent']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }

class FinBERTIntentClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3, freeze_bert=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Optionally freeze BERT weights for faster training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def evaluate_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    return accuracy * 100

def main():
    print("⚡ Fast FinBERT Intent Classification")
    print("=" * 50)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    model_name = "ProsusAI/finbert"
    print(f"\n📥 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets with consistent intent mapping
    print("\n📚 Loading ATIS datasets...")
    train_dataset = ATISIntentDataset('data/train.json', tokenizer, max_length=64)
    
    # Use training intent mapping for validation and test
    val_dataset = ATISIntentDataset('data/validation.json', tokenizer, train_dataset.intent_to_id, max_length=64)
    test_dataset = ATISIntentDataset('data/test.json', tokenizer, train_dataset.intent_to_id, max_length=64)
    
    num_classes = len(train_dataset.intent_to_id)
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"🎯 Number of intent classes: {num_classes}")
    
    # Show intent distribution
    train_intents = [train_dataset.data[i]['intent_name'] for i in range(len(train_dataset))]
    intent_counts = Counter(train_intents)
    print("\n📈 Top 10 intents:")
    for intent, count in intent_counts.most_common(10):
        print(f"  {intent}: {count}")
    
    # Create data loaders
    batch_size = 32  # Larger batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with frozen BERT for faster training
    print(f"\n🏗️  Initializing FinBERT classifier...")
    model = FinBERTIntentClassifier(model_name, num_classes, dropout=0.1, freeze_bert=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Training setup
    epochs = 10
    lr = 1e-3  # Higher learning rate since we're only training classifier
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"\n🚀 Training Details:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  BERT frozen: True (only training classifier)")
    
    # Training loop
    print(f"\n🔥 Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Train Acc = {100.*correct/total:.2f}%, Val Acc = {val_acc:.2f}%")
        
        # Early stopping if validation accuracy is good
        if val_acc > 85:
            print(f"🎯 Early stopping - good validation accuracy achieved!")
            break
    
    # Final test
    test_acc = evaluate_model(model, test_loader, device)
    
    print(f"\n🎯 Final Results:")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    
    # Show sample predictions
    print(f"\n🔍 Sample Predictions:")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 1:  # Just show first batch
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['text']
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            confidences = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            
            for j in range(min(5, len(texts))):
                pred_intent = train_dataset.id_to_intent[predictions[j].item()]
                true_intent = train_dataset.id_to_intent[labels[j].item()]
                confidence = confidences[j].item()
                
                status = "✅" if pred_intent == true_intent else "❌"
                print(f"{j+1}. {status} \"{texts[j][:50]}{'...' if len(texts[j]) > 50 else ''}\"")
                print(f"    Predicted: {pred_intent} ({confidence:.3f})")
                print(f"    True: {true_intent}")
                print()
    
    # Save model
    model_save_path = "finbert_fast_intent_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'intent_to_id': train_dataset.intent_to_id,
        'id_to_intent': train_dataset.id_to_intent,
        'model_name': model_name,
        'num_classes': num_classes
    }, model_save_path)
    
    print(f"💾 Model saved to: {model_save_path}")
    return test_acc

if __name__ == "__main__":
    main()