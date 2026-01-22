#include "nn/attention.h"
#include "nn/attention_rnn.h"
#include "nn/matrix.h"
#include <iostream>
#include <vector>
#include <iomanip>

void print_attention_weights(const Matrix& weights) {
    std::cout << "\nAttention Weights (";
    std::cout << weights.getRows() << " x " << weights.getCols() << "):\n";
    
    for (size_t i = 0; i < weights.getRows(); i++) {
        std::cout << "  [";
        for (size_t j = 0; j < weights.getCols(); j++) {
            std::cout << std::fixed << std::setprecision(3) << std::setw(6) << weights.get(i, j);
            if (j < weights.getCols() - 1) std::cout << " ";
        }
        std::cout << "]\n";
    }
}

void demo_dot_product_attention() {
    std::cout << "\n========================================\n";
    std::cout << "DOT-PRODUCT ATTENTION DEMO\n";
    std::cout << "========================================\n";
    
    size_t hidden_dim = 4;
    size_t seq_length = 3;
    
    DotProductAttention attention;
    
    // Create query (decoder state)
    Matrix query(1, hidden_dim);
    query.randomNormal();
    
    // Create keys and values (encoder states)
    std::vector<Matrix> keys, values;
    for (size_t i = 0; i < seq_length; i++) {
        Matrix key(1, hidden_dim);
        Matrix value(1, hidden_dim);
        key.randomNormal();
        value.randomNormal();
        keys.push_back(key);
        values.push_back(value);
    }
    
    // Compute attention
    auto [context, weights] = attention.forward(query, keys, values);
    
    std::cout << "\nQuery: [" << query.get(0,0) << ", " << query.get(0,1) 
              << ", " << query.get(0,2) << ", " << query.get(0,3) << "]\n";
    std::cout << "\nEncoder Sequence Length: " << seq_length << "\n";
    std::cout << "Hidden Dimension: " << hidden_dim << "\n";
    
    print_attention_weights(weights);
    
    std::cout << "\nContext Vector: [";
    for (size_t i = 0; i < context.getCols(); i++) {
        std::cout << std::fixed << std::setprecision(3) << context.get(0, i);
        if (i < context.getCols() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void demo_additive_attention() {
    std::cout << "\n========================================\n";
    std::cout << "ADDITIVE (BAHDANAU) ATTENTION DEMO\n";
    std::cout << "========================================\n";
    
    size_t hidden_dim = 4;
    size_t seq_length = 3;
    
    AdditiveAttention attention(hidden_dim, hidden_dim, hidden_dim);
    
    // Create query and encoder states
    Matrix query(1, hidden_dim);
    query.randomNormal();
    
    std::vector<Matrix> keys, values;
    for (size_t i = 0; i < seq_length; i++) {
        Matrix key(1, hidden_dim);
        Matrix value(1, hidden_dim);
        key.randomNormal();
        value.randomNormal();
        keys.push_back(key);
        values.push_back(value);
    }
    
    // Compute attention
    auto [context, weights] = attention.forward(query, keys, values);
    
    std::cout << "\nLearnable Parameters:\n";
    std::cout << "  W_query: " << hidden_dim << " x " << hidden_dim << "\n";
    std::cout << "  W_key:   " << hidden_dim << " x " << hidden_dim << "\n";
    std::cout << "  v:       " << hidden_dim << " x 1\n";
    std::cout << "\nEncoder Sequence Length: " << seq_length << "\n";
    
    print_attention_weights(weights);
    
    std::cout << "\nContext Vector: [";
    for (size_t i = 0; i < context.getCols(); i++) {
        std::cout << std::fixed << std::setprecision(3) << context.get(0, i);
        if (i < context.getCols() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

void demo_scaled_attention() {
    std::cout << "\n========================================\n";
    std::cout << "SCALED DOT-PRODUCT ATTENTION DEMO\n";
    std::cout << "========================================\n";
    
    size_t hidden_dim = 8;  // Larger dimension to show scaling benefit
    size_t seq_length = 4;
    
    ScaledDotProductAttention attention(hidden_dim);
    
    // Create query and encoder states
    Matrix query(1, hidden_dim);
    query.randomNormal();
    
    std::vector<Matrix> keys, values;
    for (size_t i = 0; i < seq_length; i++) {
        Matrix key(1, hidden_dim);
        Matrix value(1, hidden_dim);
        key.randomNormal();
        value.randomNormal();
        keys.push_back(key);
        values.push_back(value);
    }
    
    // Compute attention
    auto [context, weights] = attention.forward(query, keys, values);
    
    std::cout << "\nScaling Factor: 1/√" << hidden_dim << " = " 
              << std::fixed << std::setprecision(4) << (1.0 / std::sqrt(hidden_dim)) << "\n";
    std::cout << "Encoder Sequence Length: " << seq_length << "\n";
    
    print_attention_weights(weights);
    
    std::cout << "\nContext Vector: [";
    for (size_t i = 0; i < std::min(size_t(4), context.getCols()); i++) {
        std::cout << std::fixed << std::setprecision(3) << context.get(0, i);
        if (i < std::min(size_t(4), context.getCols()) - 1) std::cout << ", ";
    }
    std::cout << " ...]\n";
}

void demo_attention_comparison() {
    std::cout << "\n========================================\n";
    std::cout << "ATTENTION MECHANISMS COMPARISON\n";
    std::cout << "========================================\n";
    
    std::cout << "\n┌─────────────────────┬──────────────┬─────────────┬────────────┐\n";
    std::cout << "│ Attention Type      │ Parameters   │ Complexity  │ Best For   │\n";
    std::cout << "├─────────────────────┼──────────────┼─────────────┼────────────┤\n";
    std::cout << "│ Dot-Product         │ 0            │ O(n)        │ Speed      │\n";
    std::cout << "│ Additive (Bahdanau) │ O(d²)        │ O(n*d²)     │ Learning   │\n";
    std::cout << "│ Scaled Dot-Product  │ 0            │ O(n)        │ Stability  │\n";
    std::cout << "│ Multi-Head          │ O(h*d²)      │ O(h*n*d²)   │ Complex    │\n";
    std::cout << "└─────────────────────┴──────────────┴─────────────┴────────────┘\n";
    
    std::cout << "\nWhere:\n";
    std::cout << "  d = hidden dimension\n";
    std::cout << "  n = sequence length\n";
    std::cout << "  h = number of attention heads\n";
    
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ATTENTION USE CASES                                   ║\n";
    std::cout << "╠════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Machine Translation    → Additive or Multi-Head       ║\n";
    std::cout << "║  Text Summarization     → Scaled Dot-Product           ║\n";
    std::cout << "║  Speech Recognition     → Additive (location-aware)    ║\n";
    std::cout << "║  Image Captioning       → Multi-Head                   ║\n";
    std::cout << "║  Time Series Forecasting→ Dot-Product (lightweight)    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                          ║\n";
    std::cout << "║        ATTENTION MECHANISMS DEMONSTRATION                ║\n";
    std::cout << "║                                                          ║\n";
    std::cout << "║  Attention solves the information bottleneck problem    ║\n";
    std::cout << "║  in encoder-decoder architectures by allowing the       ║\n";
    std::cout << "║  decoder to focus on relevant encoder states.           ║\n";
    std::cout << "║                                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    try {
        // Demo different attention mechanisms
        demo_dot_product_attention();
        demo_additive_attention();
        demo_scaled_attention();
        demo_attention_comparison();
        
        std::cout << "\n========================================\n";
        std::cout << "SUCCESS: All attention mechanisms tested!\n";
        std::cout << "========================================\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << "\n\n";
        return 1;
    }
    
    return 0;
}
