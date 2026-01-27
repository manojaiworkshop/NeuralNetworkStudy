#!/usr/bin/env python3
"""
Download ATIS dataset from Hugging Face and convert to our JSON format
"""

import json
import os
from datasets import load_dataset

def download_atis_dataset():
    """Download ATIS dataset from Hugging Face and save as JSON"""
    
    print("📥 Downloading ATIS dataset from Hugging Face...")
    
    try:
        # Load the ATIS dataset from the correct source
        dataset = load_dataset("tuetschek/atis")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Convert each split
        for split_name, split_data in dataset.items():
            print(f"Processing {split_name} split ({len(split_data)} examples)...")
            
            examples = []
            for i, item in enumerate(split_data):
                try:
                    # Check available fields in the first item
                    if i == 0:
                        print(f"Available fields: {list(item.keys())}")
                        print(f"Sample item: {item}")
                    
                    # Handle the real ATIS dataset format
                    text = item.get("text", "")
                    slots_str = item.get("slots", "")
                    
                    # Tokenize text and slots
                    tokens = text.split()
                    slots = slots_str.split()
                    
                    # Ensure tokens and slots have same length
                    if len(tokens) != len(slots):
                        print(f"Warning: Length mismatch at item {i}: tokens={len(tokens)}, slots={len(slots)}")
                        # If slots are shorter, pad with 'O'
                        if len(slots) < len(tokens):
                            slots.extend(["O"] * (len(tokens) - len(slots)))
                        # If slots are longer, truncate
                        else:
                            slots = slots[:len(tokens)]
                    
                    example = {
                        "text": text,
                        "intent": item.get("intent", "unknown"),
                        "tokens": tokens,
                        "slots": slots
                    }
                    
                    examples.append(example)
                    
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    continue
            
            # Save to JSON file
            output_file = f"data/{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(examples, f, indent=2)
            
            print(f"✅ Saved {len(examples)} examples to {output_file}")
        
        print("\n🎉 ATIS dataset downloaded successfully!")
        
        # Create validation split from training data (10%)
        if os.path.exists("data/train.json"):
            print("\n📊 Creating validation split from training data...")
            
            with open("data/train.json", 'r') as f:
                train_data = json.load(f)
            
            # Split training data: 90% train, 10% validation
            val_size = len(train_data) // 10
            val_data = train_data[:val_size]
            train_data = train_data[val_size:]
            
            # Save updated splits
            with open("data/train.json", 'w') as f:
                json.dump(train_data, f, indent=2)
            
            with open("data/validation.json", 'w') as f:
                json.dump(val_data, f, indent=2)
            
            print(f"✅ Created validation split with {len(val_data)} examples")
            print(f"✅ Updated training split with {len(train_data)} examples")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("📝 Creating sample ATIS-style dataset instead...")
        create_sample_dataset()

def create_sample_dataset():
    """Create a sample ATIS-style dataset for testing"""
    
    sample_data = {
        "train": [
            {
                "text": "book a flight from boston to denver",
                "intent": "book_flight",
                "tokens": ["book", "a", "flight", "from", "boston", "to", "denver"],
                "slots": ["O", "O", "O", "O", "B-from_city", "O", "B-to_city"]
            },
            {
                "text": "i want to book a flight from new york to chicago",
                "intent": "book_flight", 
                "tokens": ["i", "want", "to", "book", "a", "flight", "from", "new", "york", "to", "chicago"],
                "slots": ["O", "O", "O", "O", "O", "O", "O", "B-from_city", "I-from_city", "O", "B-to_city"]
            },
            {
                "text": "cancel my flight to miami",
                "intent": "cancel_flight",
                "tokens": ["cancel", "my", "flight", "to", "miami"],
                "slots": ["O", "O", "O", "O", "B-to_city"]
            },
            {
                "text": "what is the weather in seattle",
                "intent": "get_weather",
                "tokens": ["what", "is", "the", "weather", "in", "seattle"],
                "slots": ["O", "O", "O", "O", "O", "B-city"]
            },
            {
                "text": "show me flights from dallas to atlanta",
                "intent": "book_flight",
                "tokens": ["show", "me", "flights", "from", "dallas", "to", "atlanta"],
                "slots": ["O", "O", "O", "O", "B-from_city", "O", "B-to_city"]
            },
            {
                "text": "get fare from san francisco to los angeles",
                "intent": "get_fare",
                "tokens": ["get", "fare", "from", "san", "francisco", "to", "los", "angeles"],
                "slots": ["O", "O", "O", "B-from_city", "I-from_city", "O", "B-to_city", "I-to_city"]
            }
        ],
        "validation": [
            {
                "text": "book flight from chicago to denver",
                "intent": "book_flight",
                "tokens": ["book", "flight", "from", "chicago", "to", "denver"],
                "slots": ["O", "O", "O", "B-from_city", "O", "B-to_city"]
            },
            {
                "text": "cancel reservation to boston",
                "intent": "cancel_flight", 
                "tokens": ["cancel", "reservation", "to", "boston"],
                "slots": ["O", "O", "O", "B-to_city"]
            }
        ],
        "test": [
            {
                "text": "i need a flight from denver to new york",
                "intent": "book_flight",
                "tokens": ["i", "need", "a", "flight", "from", "denver", "to", "new", "york"],
                "slots": ["O", "O", "O", "O", "O", "B-from_city", "O", "B-to_city", "I-to_city"]
            }
        ]
    }
    
    # Expand the dataset with more examples
    expanded_train = []
    cities = ["boston", "denver", "chicago", "miami", "seattle", "dallas", "atlanta", "phoenix", "philadelphia"]
    
    # Generate more flight booking examples
    for from_city in cities[:4]:
        for to_city in cities[4:]:
            expanded_train.extend([
                {
                    "text": f"book a flight from {from_city} to {to_city}",
                    "intent": "book_flight",
                    "tokens": ["book", "a", "flight", "from", from_city, "to", to_city],
                    "slots": ["O", "O", "O", "O", "B-from_city", "O", "B-to_city"]
                },
                {
                    "text": f"i want to fly from {from_city} to {to_city}",
                    "intent": "book_flight", 
                    "tokens": ["i", "want", "to", "fly", "from", from_city, "to", to_city],
                    "slots": ["O", "O", "O", "O", "O", "B-from_city", "O", "B-to_city"]
                },
                {
                    "text": f"cancel flight to {to_city}",
                    "intent": "cancel_flight",
                    "tokens": ["cancel", "flight", "to", to_city],
                    "slots": ["O", "O", "O", "B-to_city"]
                },
                {
                    "text": f"weather in {to_city}",
                    "intent": "get_weather",
                    "tokens": ["weather", "in", to_city],
                    "slots": ["O", "O", "B-city"]
                },
                {
                    "text": f"fare from {from_city} to {to_city}",
                    "intent": "get_fare",
                    "tokens": ["fare", "from", from_city, "to", to_city],
                    "slots": ["O", "O", "B-from_city", "O", "B-to_city"]
                }
            ])
    
    sample_data["train"].extend(expanded_train)
    
    # Save sample dataset
    os.makedirs("data", exist_ok=True)
    for split, examples in sample_data.items():
        with open(f"data/{split}.json", 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"✅ Created {split}.json with {len(examples)} examples")

if __name__ == "__main__":
    download_atis_dataset()