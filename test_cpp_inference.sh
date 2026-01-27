#!/bin/bash
# Test C++ inference with sample queries

cd /home/manoj/Documents/CODES/NeuralNetworkStudy

echo "🧪 Testing C++ GPU Inference..."
echo ""

# Run inference with test queries
cat << 'EOF' | ./build/joint_nlu_inference
I want to fly from Boston to Denver on Monday
Show me American Airlines flights to Chicago
What's the cheapest flight from New York to LA?
quit
EOF
