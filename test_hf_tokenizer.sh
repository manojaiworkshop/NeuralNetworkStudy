#!/bin/bash
cd /home/manoj/Documents/CODES/NeuralNetworkStudy
source finbert_env/bin/activate

echo "🧪 Testing C++ Inference with HuggingFace BERT Tokenizer"
echo ""

# Single test query
echo "I want to fly from Boston to Denver on Monday" | timeout 30 ./build/joint_nlu_inference 2>&1 | tail -20
