#!/usr/bin/env python3
"""
FinBERT Fine-tuning for Intent Classification
Using ProsusAI/finbert model for ATIS intent classification
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ATISIntentDataset(Dataset):
    """Dataset class for ATIS intent classification"""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.intent_to_id = {}
        self.id_to_intent = {}
        
        # Load data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Build intent vocabulary
        intents = list(set([item['intent'] for item in raw_data]))
        for i, intent in enumerate(sorted(intents)):
            self.intent_to_id[intent] = i
            self.id_to_intent[i] = intent
        
        # Process data
        for item in raw_data:
            self.data.append({
                'text': item['text'],
                'intent': self.intent_to_id[item['intent']],
                'intent_name': item['intent']
            })
        
        logger.info(f"Loaded {len(self.data)} examples with {len(intents)} intents")
    
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
    """FinBERT model for intent classification"""
    
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def download_and_setup_finbert():
    """Download FinBERT model and tokenizer"""
    model_name = "ProsusAI/finbert"
    
    print("🔄 Downloading FinBERT model and tokenizer...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ FinBERT tokenizer loaded: {len(tokenizer)} vocab size")
        return tokenizer, model_name
    
    except Exception as e:
        print(f"❌ Error downloading FinBERT: {e}")
        # Fallback to BERT base
        fallback_model = "bert-base-uncased"
        print(f"🔄 Using fallback model: {fallback_model}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        return tokenizer, fallback_model

def train_model(model, train_loader, val_loader, epochs=5, lr=2e-5):
    """Train the FinBERT intent classifier"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"\n🚀 Training on {device}")
    print(f"📊 Total steps: {total_steps}")
    print(f"🔥 Learning rate: {lr}")
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")
        
        # Training
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in train_pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_acc, val_loss = evaluate_model(model, val_loader, loss_fn, device)
        
        print(f"\n📈 Epoch {epoch + 1} Results:")
        print(f"  Training   - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

def evaluate_model(model, dataloader, loss_fn, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss

def test_model(model, test_loader, tokenizer, intent_dataset):
    """Test the trained model and show predictions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    sample_predictions = []
    
    print("\n🔍 Testing trained model...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['text']
            
            logits = model(input_ids, attention_mask)
            _, predicted = torch.max(logits.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Collect sample predictions
            for i in range(len(texts)):
                if len(sample_predictions) < 10:
                    pred_intent = intent_dataset.id_to_intent[predicted[i].item()]
                    true_intent = intent_dataset.id_to_intent[labels[i].item()]
                    confidence = torch.softmax(logits[i], dim=0).max().item()
                    
                    sample_predictions.append({
                        'text': texts[i],
                        'predicted': pred_intent,
                        'true': true_intent,
                        'confidence': confidence,
                        'correct': pred_intent == true_intent
                    })
    
    accuracy = 100. * correct / total
    
    print(f"\n🎯 Test Results:")
    print(f"  Final Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print(f"\n🔍 Sample Predictions:")
    for i, pred in enumerate(sample_predictions):
        status = "✅" if pred['correct'] else "❌"
        print(f"{i+1}. {status} \"{pred['text'][:60]}{'...' if len(pred['text']) > 60 else ''}\"")
        print(f"    Predicted: {pred['predicted']} (confidence: {pred['confidence']:.3f})")
        print(f"    True: {pred['true']}")
        print()
    
    return accuracy

def main():
    print("🤖 FinBERT Intent Classification Training")
    print("=" * 60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    # Download model
    tokenizer, model_name = download_and_setup_finbert()
    
    # Load datasets
    print("\n📚 Loading ATIS datasets...")
    train_dataset = ATISIntentDataset('data/train.json', tokenizer)
    val_dataset = ATISIntentDataset('data/validation.json', tokenizer)
    val_dataset.intent_to_id = train_dataset.intent_to_id  # Use same mapping
    val_dataset.id_to_intent = train_dataset.id_to_intent
    
    test_dataset = ATISIntentDataset('data/test.json', tokenizer)
    test_dataset.intent_to_id = train_dataset.intent_to_id  # Use same mapping
    test_dataset.id_to_intent = train_dataset.id_to_intent
    
    num_classes = len(train_dataset.intent_to_id)
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"🎯 Number of intent classes: {num_classes}")
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"\n🏗️  Initializing FinBERT classifier...")
    model = FinBERTIntentClassifier(model_name, num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train model
    epochs = 3  # Start with fewer epochs for faster training
    train_model(model, train_loader, val_loader, epochs=epochs)
    
    # Test model
    final_accuracy = test_model(model, test_loader, tokenizer, train_dataset)
    
    # Save model
    model_save_path = "finbert_intent_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'intent_to_id': train_dataset.intent_to_id,
        'id_to_intent': train_dataset.id_to_intent,
        'model_name': model_name,
        'num_classes': num_classes
    }, model_save_path)
    
    print(f"\n💾 Model saved to: {model_save_path}")
    print(f"🎉 Final test accuracy: {final_accuracy:.2f}%")
    
    return model, tokenizer, train_dataset.intent_to_id

if __name__ == "__main__":
    main()