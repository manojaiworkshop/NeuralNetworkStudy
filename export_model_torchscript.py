#!/usr/bin/env python3
"""
Export trained Joint NLU model to TorchScript format for C++ inference
"""

import torch
import json
from joint_nlu_finbert import JointNLUModel
from transformers import AutoTokenizer

def export_model_to_torchscript():
    """Export the trained model to TorchScript format"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔄 Loading trained model...")
    
    # Load checkpoint
    checkpoint = torch.load('joint_nlu_finbert_model.pt', map_location=device)
    
    # Extract model info
    model_name = checkpoint['model_name']
    num_intents = checkpoint['num_intents']
    num_slots = checkpoint['num_slots']
    num_entities = checkpoint['num_entities']
    
    print(f"📊 Model info: {num_intents} intents, {num_slots} slots, {num_entities} entities")
    
    # Create model
    model = JointNLUModel(model_name, num_intents, num_slots, num_entities, freeze_bert=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    
    # Create example input
    batch_size = 1
    max_length = 64
    example_input_ids = torch.randint(0, 30522, (batch_size, max_length)).to(device)
    example_attention_mask = torch.ones(batch_size, max_length).to(device)
    
    print(f"🔄 Tracing model with example input...")
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(model, (example_input_ids, example_attention_mask))
        print(f"✅ Model traced successfully")
        
        # Save traced model
        traced_model.save('joint_nlu_model_traced.pt')
        print(f"💾 Traced model saved to: joint_nlu_model_traced.pt")
        
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        print(f"🔄 Trying scripting instead...")
        
        # Try scripting if tracing fails
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save('joint_nlu_model_scripted.pt')
            print(f"💾 Scripted model saved to: joint_nlu_model_scripted.pt")
        except Exception as e2:
            print(f"❌ Scripting also failed: {e2}")
            return False
    
    # Save vocabularies as JSON for C++ to load
    vocab_data = {
        'intent_to_id': checkpoint['intent_to_id'],
        'id_to_intent': {str(k): v for k, v in checkpoint['id_to_intent'].items()},
        'slot_to_id': checkpoint['slot_to_id'],
        'id_to_slot': {str(k): v for k, v in checkpoint['id_to_slot'].items()},
        'entity_to_id': checkpoint['entity_to_id'],
        'id_to_entity': {str(k): v for k, v in checkpoint['id_to_entity'].items()},
        'model_name': model_name,
        'num_intents': num_intents,
        'num_slots': num_slots,
        'num_entities': num_entities,
        'max_length': max_length
    }
    
    with open('joint_nlu_vocab.json', 'w') as f:
        json.dump(vocab_data, f, indent=2)
    
    print(f"💾 Vocabularies saved to: joint_nlu_vocab.json")
    
    # Test the traced model
    print(f"\n🧪 Testing traced model...")
    with torch.no_grad():
        original_output = model(example_input_ids, example_attention_mask)
        traced_output = traced_model(example_input_ids, example_attention_mask)
        
        # Compare outputs
        for i, (orig, traced) in enumerate(zip(original_output, traced_output)):
            diff = torch.abs(orig - traced).max().item()
            print(f"  Output {i} max difference: {diff:.6f}")
    
    print(f"\n🎉 Export completed successfully!")
    return True

if __name__ == "__main__":
    export_model_to_torchscript()
