# Transformer Architecture Complete Analysis

## Overview

Your workspace contains a **complete Transformer implementation** following the "Attention Is All You Need" (Vaswani et al., 2017) architecture. This document explains how all components interact with the core Matrix library and other modules.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Structure](#file-structure)
3. [Core Module Dependencies](#core-module-dependencies)
4. [Component Deep Dive](#component-deep-dive)
5. [Matrix Library Usage](#matrix-library-usage)
6. [Data Flow Analysis](#data-flow-analysis)
7. [Code Examples](#code-examples)

---

## Architecture Overview

### Transformer Structure

```
Input Tokens (src)          Input Tokens (tgt)
       ↓                            ↓
  [Embedding]                  [Embedding]
       ↓                            ↓
[Positional Encoding]       [Positional Encoding]
       ↓                            ↓
  ┌─────────┐                 ┌─────────┐
  │ ENCODER │                 │ DECODER │
  │         │                 │         │
  │ Layer 1 │                 │ Layer 1 │ ←── Encoder Output
  │ Layer 2 │                 │ Layer 2 │ ←── (Cross-Attention)
  │   ...   │                 │   ...   │
  │ Layer N │                 │ Layer N │
  └─────────┘                 └─────────┘
       ↓                            ↓
       └────────────────────────────┘
                    ↓
            [Output Projection]
                    ↓
            Vocabulary Logits
```

### Key Components

| Component | Purpose | Matrix Operations |
|-----------|---------|-------------------|
| **TokenEmbedding** | Token ID → Dense Vector | Lookup table (vocab_size × d_model) |
| **PositionalEncoding** | Add position info | Sinusoidal functions, matrix addition |
| **MultiHeadAttention** | Parallel attention | Q, K, V projections + scaled dot-product |
| **FeedForward** | Non-linearity | Two linear layers with ReLU |
| **LayerNormalization** | Stabilize training | Mean/variance normalization |
| **EncoderLayer** | Encoder block | Self-attention + FFN + residuals |
| **DecoderLayer** | Decoder block | Self-attn + cross-attn + FFN |

---

## File Structure

### Header Files (include/nn/transformer/)

```
transformer/
├── attention.h          - Multi-head attention mechanism
├── decoder.h           - Decoder layer and stack
├── embedding.h         - Token and positional embeddings
├── encoder.h           - Encoder layer and stack
├── feedforward.h       - Position-wise feed-forward network
├── layer_norm.h        - Layer normalization
├── model_saver.h       - Model persistence (weights save/load)
├── tokenizer.h         - Text tokenization utilities
└── transformer.h       - Complete encoder-decoder model
```

### Implementation Files (src/transformer/)

```
transformer/
├── attention.cpp       - Scaled dot-product + multi-head attention
├── decoder.cpp         - Decoder implementation with masked attention
├── embedding.cpp       - Token embeddings + positional encoding
├── encoder.cpp         - Encoder with self-attention
├── feedforward.cpp     - Two-layer FFN with ReLU/GELU
├── layer_norm.cpp      - Layer normalization implementation
├── tokenizer.cpp       - BPE/WordPiece tokenization
└── transformer.cpp     - End-to-end training and inference
```

---

## Core Module Dependencies

### Dependency Graph

```
Transformer (transformer.h)
    ├── Matrix (matrix.h)               ← Core matrix operations
    ├── Embedding (embedding.h)
    │   └── Matrix
    ├── TransformerEncoder (encoder.h)
    │   ├── EncoderLayer
    │   │   ├── MultiHeadAttention (attention.h)
    │   │   │   ├── ScaledDotProductAttention
    │   │   │   │   └── Matrix
    │   │   │   └── Matrix (W_Q, W_K, W_V, W_O)
    │   │   ├── PositionWiseFeedForward (feedforward.h)
    │   │   │   ├── Matrix (W1, b1, W2, b2)
    │   │   │   ├── Layer (layer.h)
    │   │   │   └── Activation (activation.h)
    │   │   └── LayerNormalization (layer_norm.h)
    │   │       ├── Layer (layer.h)
    │   │       └── Matrix (gamma, beta)
    │   └── Matrix
    └── TransformerDecoder (decoder.h)
        └── DecoderLayer
            ├── MultiHeadAttention (self-attention)
            ├── MultiHeadAttention (cross-attention)
            ├── PositionWiseFeedForward
            └── LayerNormalization (×3)
```

### Core Libraries Used

1. **Matrix Library** (`matrix.h`, `matrix.cpp`)
   - All weight matrices
   - All activations and gradients
   - Matrix multiplication for projections
   - Transpose for attention

2. **Activation Library** (`activation.h`, `activation.cpp`)
   - ReLU in feed-forward networks
   - Optional GELU for better performance
   - Softmax for attention weights

3. **Layer Library** (`layer.h`)
   - Base class for LayerNormalization
   - Parameter management interface
   - Forward/backward abstractions

---

## Component Deep Dive

### 1. ScaledDotProductAttention

**Purpose**: Core attention mechanism with scaling

**Matrix Operations**:
```cpp
// File: src/transformer/attention.cpp

Matrix ScaledDotProductAttention::forward(const Matrix& Q, const Matrix& K, const Matrix& V, ...) {
    // 1. Matrix transpose (Matrix lib method)
    Matrix K_T = K.transpose();
    
    // 2. Matrix multiplication (QK^T)
    Matrix scores = Q * K_T;
    
    // 3. Element-wise scaling (1/√d_k)
    for (size_t i = 0; i < scores.getRows(); i++) {
        for (size_t j = 0; j < scores.getCols(); j++) {
            scores.set(i, j, scores.get(i, j) * scale_factor);
        }
    }
    
    // 4. Softmax (using Matrix get/set methods)
    for (size_t i = 0; i < scores.getRows(); i++) {
        double max_score = scores.get(i, 0);
        for (size_t j = 1; j < scores.getCols(); j++) {
            max_score = std::max(max_score, scores.get(i, j));
        }
        
        double sum_exp = 0.0;
        for (size_t j = 0; j < scores.getCols(); j++) {
            double val = std::exp(scores.get(i, j) - max_score);
            cached_attention_weights.set(i, j, val);
            sum_exp += val;
        }
        
        for (size_t j = 0; j < scores.getCols(); j++) {
            cached_attention_weights.set(i, j, 
                cached_attention_weights.get(i, j) / sum_exp);
        }
    }
    
    // 5. Final matrix multiplication (attention_weights × V)
    Matrix output = cached_attention_weights * V;
    
    return output;
}
```

**Matrix Library Usage**:
- `Matrix::transpose()` - K^T computation
- `Matrix::operator*()` - Matrix multiplication (QK^T and attn×V)
- `Matrix::get()` / `Matrix::set()` - Element access for scaling and softmax
- `Matrix::getRows()` / `Matrix::getCols()` - Dimension queries

---

### 2. MultiHeadAttention

**Purpose**: Parallel attention heads with learned projections

**Matrix Operations**:
```cpp
// File: src/transformer/attention.cpp (lines 200+)

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads, double dropout)
    : d_model(d_model), num_heads(num_heads), 
      d_k(d_model / num_heads), d_v(d_model / num_heads) {
    
    // Projection matrices (using Matrix constructor)
    W_Q.resize(num_heads);
    W_K.resize(num_heads);
    W_V.resize(num_heads);
    
    for (size_t h = 0; h < num_heads; h++) {
        W_Q[h] = Matrix(d_model, d_k);  // Query projection
        W_K[h] = Matrix(d_model, d_k);  // Key projection
        W_V[h] = Matrix(d_model, d_v);  // Value projection
        
        // Xavier initialization using Matrix methods
        W_Q[h].xavierInit(d_model, d_k);
        W_K[h].xavierInit(d_model, d_k);
        W_V[h].xavierInit(d_model, d_v);
    }
    
    // Output projection
    W_O = Matrix(d_model, d_model);
    W_O.xavierInit(d_model, d_model);
}

Matrix MultiHeadAttention::forward(const Matrix& Q, const Matrix& K, const Matrix& V, ...) {
    std::vector<Matrix> head_outputs;
    
    // For each attention head
    for (size_t h = 0; h < num_heads; h++) {
        // Project Q, K, V using learned matrices
        Matrix Q_proj = Q * W_Q[h];  // Matrix multiplication
        Matrix K_proj = K * W_K[h];
        Matrix V_proj = V * W_V[h];
        
        // Apply scaled dot-product attention
        Matrix head_output = attention_heads[h]->forward(Q_proj, K_proj, V_proj, mask, training);
        head_outputs.push_back(head_output);
    }
    
    // Concatenate heads (using Matrix operations)
    Matrix concat = concatenateHeads(head_outputs);  // Custom concat
    
    // Output projection
    Matrix output = concat * W_O;  // Matrix multiplication
    
    return output;
}
```

**Key Matrix Operations**:
1. **Projection**: `Q * W_Q[h]` - Transform input to head-specific subspace
2. **Concatenation**: Combine head outputs (d_v × num_heads → d_model)
3. **Output Projection**: `concat * W_O` - Mix information from all heads

**Matrix Library Features Used**:
- `Matrix(rows, cols)` - Constructor for weight matrices
- `Matrix::xavierInit()` - Proper weight initialization
- `Matrix::operator*()` - All projections
- `Matrix::getRows()`, `getCols()` - Shape management

---

### 3. PositionWiseFeedForward

**Purpose**: Two-layer MLP applied to each position independently

**Matrix Operations**:
```cpp
// File: src/transformer/feedforward.cpp

Matrix PositionWiseFeedForward::forward(const Matrix& input, bool training) {
    size_t batch_seq = input.getRows();
    
    // Cache input for backward pass
    cached_input = input;
    
    // First linear layer: input * W1 + b1
    cached_hidden_pre = input * W1;  // Matrix multiplication
    
    // Add bias (using Matrix get/set)
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t j = 0; j < d_ff; j++) {
            cached_hidden_pre.set(i, j, 
                cached_hidden_pre.get(i, j) + b1.get(0, j));
        }
    }
    
    // Apply ReLU activation (using Activation module)
    cached_hidden = activation->forward(cached_hidden_pre);
    
    // Apply dropout if training
    Matrix hidden_after_dropout = cached_hidden;
    if (training && dropout_rate > 0.0) {
        // ... dropout logic using Matrix get/set ...
    }
    
    // Second linear layer: hidden * W2 + b2
    Matrix output = hidden_after_dropout * W2;  // Matrix multiplication
    
    // Add bias
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t j = 0; j < d_model; j++) {
            output.set(i, j, output.get(i, j) + b2.get(0, j));
        }
    }
    
    return output;
}
```

**Architecture**:
```
Input (batch × seq_len × d_model)
    ↓
Linear (d_model → d_ff)         [Matrix: W1, b1]
    ↓
ReLU Activation                  [Activation module]
    ↓
Dropout (optional)
    ↓
Linear (d_ff → d_model)         [Matrix: W2, b2]
    ↓
Output (batch × seq_len × d_model)
```

**Matrix Library Integration**:
- `Matrix::operator*()` - Both linear transformations
- `Matrix::get()` / `set()` - Bias addition
- `Activation::forward()` - ReLU activation
- Gradient computation in `backward()` uses `Matrix::transpose()`

**Typical Dimensions**:
- `W1`: 512 × 2048 (expansion by 4×)
- `W2`: 2048 × 512 (back to original dimension)

---

### 4. LayerNormalization

**Purpose**: Normalize activations across features (stabilizes training)

**Matrix Operations**:
```cpp
// File: src/transformer/layer_norm.cpp

Matrix LayerNormalization::forward(const Matrix& input) {
    size_t batch_seq = input.getRows();
    size_t features = input.getCols();
    
    cached_input = input;
    cached_mean = Matrix(batch_seq, 1, 0.0);
    cached_std = Matrix(batch_seq, 1, 0.0);
    
    // Compute mean for each sample (using Matrix methods)
    for (size_t i = 0; i < batch_seq; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < features; j++) {
            sum += input.get(i, j);
        }
        cached_mean.set(i, 0, sum / features);
    }
    
    // Compute variance
    for (size_t i = 0; i < batch_seq; i++) {
        double sum_sq = 0.0;
        double mean = cached_mean.get(i, 0);
        for (size_t j = 0; j < features; j++) {
            double diff = input.get(i, j) - mean;
            sum_sq += diff * diff;
        }
        double variance = sum_sq / features;
        cached_std.set(i, 0, std::sqrt(variance + epsilon));
    }
    
    // Normalize: (x - μ) / σ
    cached_normalized = Matrix(batch_seq, features);
    for (size_t i = 0; i < batch_seq; i++) {
        double mean = cached_mean.get(i, 0);
        double std = cached_std.get(i, 0);
        for (size_t j = 0; j < features; j++) {
            double normalized = (input.get(i, j) - mean) / std;
            cached_normalized.set(i, j, normalized);
        }
    }
    
    // Scale and shift: γ * normalized + β
    Matrix output(batch_seq, features);
    for (size_t i = 0; i < batch_seq; i++) {
        for (size_t j = 0; j < features; j++) {
            output.set(i, j, 
                gamma.get(0, j) * cached_normalized.get(i, j) + beta.get(0, j));
        }
    }
    
    return output;
}
```

**Formula**: `LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β`

**Matrix Library Usage**:
- All statistics computed using `Matrix::get()` / `set()`
- `gamma`, `beta` are learnable Matrix objects (1 × features)
- Normalization happens per-sample across all features

---

### 5. TokenEmbedding

**Purpose**: Convert token IDs to dense vectors

**Matrix Operations**:
```cpp
// File: src/transformer/embedding.cpp

TokenEmbedding::TokenEmbedding(size_t vocab_size, size_t embedding_dim)
    : vocab_size(vocab_size), embedding_dim(embedding_dim) {
    
    // Embedding lookup table (Matrix)
    embeddings = Matrix(vocab_size, embedding_dim);
    embeddings.randomNormal(0.0, 0.02);  // Small random initialization
    
    gradients = Matrix(vocab_size, embedding_dim, 0.0);
}

Matrix TokenEmbedding::forward(const std::vector<std::vector<int>>& token_ids) {
    size_t batch_size = token_ids.size();
    size_t seq_len = token_ids[0].size();
    
    // Output: (batch_size * seq_len) × embedding_dim
    Matrix output(batch_size * seq_len, embedding_dim);
    
    // Lookup embeddings (using Matrix get/set)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            int token_id = token_ids[b][t];
            size_t out_idx = b * seq_len + t;
            
            // Copy embedding vector
            for (size_t d = 0; d < embedding_dim; d++) {
                output.set(out_idx, d, embeddings.get(token_id, d));
            }
        }
    }
    
    return output;
}
```

**Matrix Structure**:
```
Embedding Matrix: vocab_size × embedding_dim
┌─────────────────────────────┐
│ Token 0:  [0.1, -0.2, ...]  │ ← Each row is a token embedding
│ Token 1:  [0.3,  0.1, ...]  │
│ Token 2:  [-0.1, 0.4, ...]  │
│    ...                       │
│ Token V:  [0.2, -0.3, ...]  │
└─────────────────────────────┘
```

**Matrix Library Features**:
- `Matrix(vocab_size, embedding_dim)` - Embedding table
- `Matrix::randomNormal()` - Initialization
- `Matrix::get()` - Lookup by token_id (row index)
- `Matrix::set()` - Update during training

---

### 6. PositionalEncoding

**Purpose**: Add position information (Transformers have no inherent order)

**Matrix Operations**:
```cpp
// File: src/transformer/embedding.cpp

PositionalEncoding::PositionalEncoding(size_t max_seq_len, size_t d_model)
    : max_seq_len(max_seq_len), d_model(d_model) {
    
    // Pre-compute positional encoding matrix
    encoding = Matrix(max_seq_len, d_model);
    
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < d_model; i++) {
            double angle = pos / std::pow(10000.0, (2.0 * i) / d_model);
            
            if (i % 2 == 0) {
                encoding.set(pos, i, std::sin(angle));  // Even dimensions
            } else {
                encoding.set(pos, i, std::cos(angle));  // Odd dimensions
            }
        }
    }
}

Matrix PositionalEncoding::forward(const Matrix& input) {
    size_t batch_seq = input.getRows();
    size_t features = input.getCols();
    
    Matrix output = input;  // Copy input
    
    // Add positional encoding (element-wise addition)
    for (size_t i = 0; i < batch_seq; i++) {
        size_t pos = i % max_seq_len;  // Position in sequence
        for (size_t j = 0; j < features; j++) {
            output.set(i, j, output.get(i, j) + encoding.get(pos, j));
        }
    }
    
    return output;
}
```

**Formulas**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why it works**:
- Sine/cosine at different frequencies encode position
- Model can learn to attend to relative positions
- Fixed (not learned) but effective

**Matrix Library Usage**:
- `Matrix(max_seq_len, d_model)` - Pre-computed encoding
- `std::sin()`, `std::cos()` - Standard math functions
- `Matrix::operator+()` could be used, but manual addition with `get/set` for clarity

---

## Matrix Library Usage

### Core Matrix Operations Used

#### 1. Construction and Initialization

```cpp
// From various transformer files:

// Basic construction
Matrix W1(d_model, d_ff);           // Uninitialized
Matrix b1(1, d_ff, 0.0);            // Initialized to zero
Matrix embeddings(vocab_size, dim); // Lookup table

// Initialization methods (from matrix.h)
W1.randomNormal(0.0, 0.02);         // Normal distribution
W1.xavierInit(fan_in, fan_out);     // Xavier/Glorot initialization
W2.heInit(fan_in);                  // He initialization (for ReLU)
```

#### 2. Matrix Multiplication

```cpp
// Used extensively in all components:

// Attention: Q, K, V projections
Matrix Q_proj = Q * W_Q;            // (batch×seq×d_model) × (d_model×d_k)

// Feed-forward layers
Matrix hidden = input * W1;          // (batch×seq×d_model) × (d_model×d_ff)
Matrix output = hidden * W2;         // (batch×seq×d_ff) × (d_ff×d_model)

// Attention scores
Matrix scores = Q * K.transpose();   // (batch×seq_q×d_k) × (d_k×seq_k)
```

**Performance Note**: Matrix multiplication is the most expensive operation. For a 6-layer transformer with d_model=512:
- Each attention has 4 projections (Q, K, V, O)
- Each FFN has 2 projections (W1, W2)
- Total: **6 × (4 + 2) = 36 matrix multiplications per forward pass**

#### 3. Element-wise Access

```cpp
// Used for:
// - Bias addition
// - Softmax computation
// - Layer normalization
// - Dropout masks

// Bias addition in feed-forward
for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < d_ff; j++) {
        output.set(i, j, output.get(i, j) + bias.get(0, j));
    }
}

// Softmax in attention
for (size_t i = 0; i < rows; i++) {
    double max_val = scores.get(i, 0);
    for (size_t j = 1; j < cols; j++) {
        max_val = std::max(max_val, scores.get(i, j));
    }
    // ... softmax computation using get/set ...
}
```

#### 4. Transpose

```cpp
// Used in attention and backward pass:

Matrix K_T = K.transpose();          // Attention: QK^T
Matrix grad_W = input.transpose() * grad_output;  // Backward pass
```

#### 5. Dimension Queries

```cpp
// Shape checking and iteration:

size_t batch_size = input.getRows();
size_t features = input.getCols();

// Validate dimensions before operations
if (Q.getCols() != K.getCols()) {
    throw std::runtime_error("Dimension mismatch in attention");
}
```

---

## Data Flow Analysis

### End-to-End Forward Pass

Let's trace a single token through the entire transformer:

```cpp
// File: src/transformer/transformer.cpp

Matrix Transformer::forward(const std::vector<std::vector<int>>& src_tokens,
                           const std::vector<std::vector<int>>& tgt_tokens,
                           bool training) {
    
    // STEP 1: EMBEDDING (Token ID → Dense Vector)
    // ============================================
    // Input: src_tokens = [[5, 123, 47, ...]]  (token IDs)
    // Output: Matrix (batch×seq_len × d_model)
    Matrix src_embedded = src_embedding->forward(src_tokens, training);
    // Uses: TokenEmbedding::embeddings.get(token_id, dim)
    
    // STEP 2: ENCODER (Process source sequence)
    // ============================================
    // Input: src_embedded (batch×seq_len × d_model)
    // Output: encoder_output (same shape)
    cached_encoder_output = encoder->forward(src_embedded, &src_mask, training);
    
    // Inside encoder->forward():
    for (size_t layer = 0; layer < num_encoder_layers; layer++) {
        // 2a. Multi-Head Self-Attention
        Matrix attn_output = layers[layer]->self_attention->forward(x, x, x, mask, training);
        // Uses: Matrix multiplication for Q, K, V projections
        //       Q * K.transpose() for attention scores
        //       attention_weights * V for output
        
        // 2b. Add & Norm (Residual + Layer Normalization)
        Matrix attn_residual = x + attn_output;  // Residual connection
        Matrix attn_normalized = layers[layer]->norm1->forward(attn_residual);
        // Uses: Matrix get/set for normalization statistics
        
        // 2c. Feed-Forward Network
        Matrix ffn_output = layers[layer]->feed_forward->forward(attn_normalized, training);
        // Uses: Matrix multiplication (input * W1, hidden * W2)
        //       Activation module for ReLU
        
        // 2d. Add & Norm (again)
        Matrix ffn_residual = attn_normalized + ffn_output;
        x = layers[layer]->norm2->forward(ffn_residual);
    }
    
    // STEP 3: DECODER (Generate target sequence)
    // ============================================
    Matrix tgt_embedded = tgt_embedding->forward(tgt_tokens, training);
    cached_decoder_output = decoder->forward(tgt_embedded, cached_encoder_output,
                                            &tgt_mask, &src_mask, training);
    
    // Inside decoder->forward():
    for (size_t layer = 0; layer < num_decoder_layers; layer++) {
        // 3a. Masked Self-Attention (causal mask prevents looking ahead)
        Matrix self_attn = layers[layer]->self_attention->forward(x, x, x, tgt_mask, training);
        x = layers[layer]->norm1->forward(x + self_attn);
        
        // 3b. Cross-Attention (attend to encoder output)
        Matrix cross_attn = layers[layer]->cross_attention->forward(
            x,                      // Query: decoder state
            cached_encoder_output,  // Key: encoder output
            cached_encoder_output,  // Value: encoder output
            src_mask, training
        );
        x = layers[layer]->norm2->forward(x + cross_attn);
        
        // 3c. Feed-Forward Network
        Matrix ffn = layers[layer]->feed_forward->forward(x, training);
        x = layers[layer]->norm3->forward(x + ffn);
    }
    
    // STEP 4: OUTPUT PROJECTION (d_model → vocab_size)
    // ============================================
    Matrix logits = cached_decoder_output * output_projection;
    // Add bias
    for (size_t i = 0; i < logits.getRows(); i++) {
        for (size_t j = 0; j < logits.getCols(); j++) {
            logits.set(i, j, logits.get(i, j) + output_bias.get(0, j));
        }
    }
    
    return logits;  // Shape: (batch×seq_len) × vocab_size
}
```

### Matrix Dimensions at Each Step

**Example**: Translate "Hello world" (2 tokens) to French

```
Input Tokens: [15, 234]  (vocab IDs)

ENCODER:
─────────
Token Embedding:    [15, 234]  →  Matrix (1×2, 512)
                    ↓ Lookup from (vocab_size×512) embedding table
Positional Encoding: Matrix (1×2, 512) [sin/cos patterns added]
                    ↓
Encoder Layer 1:
  Self-Attention:
    Q = input × W_Q     (1×2, 512) × (512, 512) = (1×2, 512)
    K = input × W_K     (1×2, 512) × (512, 512) = (1×2, 512)
    V = input × W_V     (1×2, 512) × (512, 512) = (1×2, 512)
    scores = Q × K^T    (1×2, 512) × (512, 2×1) = (1×2, 2)
    attn = softmax(scores) × V  (1×2, 2) × (1×2, 512) = (1×2, 512)
  Feed-Forward:
    hidden = input × W1 (1×2, 512) × (512, 2048) = (1×2, 2048)
    output = hidden × W2 (1×2, 2048) × (2048, 512) = (1×2, 512)
... (repeat for 6 layers)
Final Encoder Output: Matrix (1×2, 512)

DECODER:
─────────
Target Tokens: [1, 45] (BOS + first word)
Target Embedding: Matrix (1×2, 512)
                    ↓
Decoder Layer 1:
  Masked Self-Attention: (1×2, 512) → (1×2, 512)
  Cross-Attention:
    Q = decoder × W_Q    (1×2, 512) × (512, 512) = (1×2, 512)
    K = encoder_out × W_K (1×2, 512) × (512, 512) = (1×2, 512)
    V = encoder_out × W_V (1×2, 512) × (512, 512) = (1×2, 512)
  Feed-Forward: (1×2, 512) → (1×2, 2048) → (1×2, 512)
... (repeat for 6 layers)
Final Decoder Output: Matrix (1×2, 512)

OUTPUT PROJECTION:
──────────────────
Logits = decoder_output × W_out  (1×2, 512) × (512, 32000) = (1×2, 32000)
                                               ↑ One probability per vocab word
```

---

## Code Examples

### Example 1: Creating a Transformer

```cpp
#include "nn/transformer/transformer.h"

int main() {
    // Configuration
    size_t src_vocab = 10000;  // English vocabulary
    size_t tgt_vocab = 10000;  // French vocabulary
    size_t d_model = 512;      // Model dimension
    size_t num_heads = 8;      // Attention heads
    size_t num_layers = 6;     // Encoder/decoder layers
    size_t d_ff = 2048;        // Feed-forward dimension
    size_t max_len = 512;      // Maximum sequence length
    double dropout = 0.1;
    
    // Create transformer
    Transformer model(src_vocab, tgt_vocab, d_model, num_heads,
                     num_layers, num_layers, d_ff, max_len, dropout);
    
    // Training data
    std::vector<std::vector<int>> src_tokens = {
        {5, 123, 47, 89, 2},  // "Hello world" + EOS
        {12, 45, 78, 2}       // "Good morning" + EOS
    };
    
    std::vector<std::vector<int>> tgt_tokens = {
        {1, 234, 567, 2},     // BOS + "Bonjour monde" + EOS
        {1, 890, 123, 2}      // BOS + "Bonjour" + EOS
    };
    
    // Forward pass
    Matrix logits = model.forward(src_tokens, tgt_tokens, true);
    
    // Compute loss
    double loss = model.computeLoss(logits, tgt_tokens);
    
    // Backward pass
    model.backward(logits, tgt_tokens);
    
    // Update parameters
    model.updateParameters(0.0001);  // Learning rate
    
    std::cout << "Loss: " << loss << "\n";
    
    return 0;
}
```

### Example 2: Inference (Translation)

```cpp
#include "nn/transformer/transformer.h"

int main() {
    // Load pre-trained model
    Transformer model(/* ... */);
    model.loadWeights("transformer_weights.bin");
    
    // Source sentence: "Hello world"
    std::vector<int> src_tokens = {5, 123, 47, 2};  // + EOS
    
    // Greedy decoding
    std::vector<int> translation = model.greedyDecode(src_tokens, 50);
    
    // Or beam search for better quality
    std::vector<int> translation_beam = model.beamSearch(src_tokens, 5, 50);
    
    // translation_beam = [234, 567, 2]  // "Bonjour monde" + EOS
    
    return 0;
}
```

### Example 3: Visualize Attention Weights

```cpp
#include "nn/transformer/transformer.h"

int main() {
    Transformer model(/* ... */);
    
    // Forward pass
    Matrix logits = model.forward(src_tokens, tgt_tokens, false);
    
    // Get attention weights from encoder
    auto encoder_attention = model.getEncoder()->getAllAttentionWeights();
    
    // encoder_attention[layer][head] = Matrix (seq_len × seq_len)
    
    std::cout << "Encoder Layer 0, Head 0 Attention:\n";
    Matrix attn = encoder_attention[0][0];
    
    for (size_t i = 0; i < attn.getRows(); i++) {
        for (size_t j = 0; j < attn.getCols(); j++) {
            double weight = attn.get(i, j);
            
            // ASCII visualization
            if (weight > 0.5) std::cout << "██";
            else if (weight > 0.3) std::cout << "▓▓";
            else if (weight > 0.1) std::cout << "▒▒";
            else std::cout << "░░";
        }
        std::cout << "\n";
    }
    
    return 0;
}
```

---

## Performance Characteristics

### Computational Complexity

**Per Layer**:
- Multi-Head Attention: O(n² × d_model) where n = sequence length
- Feed-Forward Network: O(n × d_model × d_ff)

**Total (6-layer transformer)**:
- Time: O(6 × (n² × d_model + n × d_model × d_ff))
- Space: O(6 × (n × d_model + d_model²))

**Bottleneck**: Attention's O(n²) complexity makes long sequences expensive

### Matrix Operations Count

For a single forward pass (batch_size=1, seq_len=10, d_model=512, d_ff=2048):

```
ENCODER (6 layers):
  Attention per layer:
    - Q, K, V projections: 3 × (10×512 × 512×512) = 3 × 2.6M ops
    - Attention scores: 10×512 × 512×10 = 2.6M ops
    - Output projection: 10×512 × 512×512 = 2.6M ops
    Subtotal: 13M ops per layer
    
  Feed-Forward per layer:
    - First linear: 10×512 × 512×2048 = 10.5M ops
    - Second linear: 10×2048 × 2048×512 = 21M ops
    Subtotal: 31.5M ops per layer
    
  Total per layer: 44.5M ops
  6 layers: 267M ops

DECODER (6 layers, similar): 267M ops

OUTPUT PROJECTION: 10×512 × 512×vocab_size

TOTAL: ~550M operations + softmax/normalization
```

### Memory Usage

**Weights**:
- Embeddings: 2 × vocab_size × d_model = 2 × 10000 × 512 = 10M parameters
- Encoder: 6 × (4 × d_model² + 2 × d_model × d_ff) = 6 × (4×512² + 2×512×2048) = 18.9M
- Decoder: Same as encoder = 18.9M
- **Total: ~48M parameters**

**Activations** (stored for backward pass):
- Each layer stores attention matrices (batch × heads × seq × seq)
- For batch=32, seq=128, heads=8: 32 × 8 × 128 × 128 = 4M values
- **Total: ~100MB per forward pass**

---

## Summary

### How Transformer Uses Core Libraries

| Component | Matrix Lib | Activation Lib | Layer Lib |
|-----------|-----------|---------------|-----------|
| **TokenEmbedding** | ✓ (lookup table) | ✗ | ✗ |
| **PositionalEncoding** | ✓ (sin/cos storage) | ✗ | ✗ |
| **MultiHeadAttention** | ✓✓✓ (Q,K,V,O projections) | ✗ | ✗ |
| **FeedForward** | ✓✓ (W1, W2) | ✓ (ReLU) | ✓ (inherits Layer) |
| **LayerNorm** | ✓ (gamma, beta) | ✗ | ✓ (inherits Layer) |
| **EncoderLayer** | ✓ (via sub-components) | ✓ (via FFN) | ✓ (via LayerNorm) |
| **DecoderLayer** | ✓ (via sub-components) | ✓ (via FFN) | ✓ (via LayerNorm) |

### Key Takeaways

1. **Matrix Library is Central**: Every component uses Matrix for:
   - Weight storage (W_Q, W_K, W_V, W_O, W1, W2, embeddings)
   - Forward computation (matrix multiplication, transpose)
   - Gradient computation (backward pass)

2. **Activation Library**: Used only in Feed-Forward networks (ReLU/GELU)

3. **Layer Library**: Provides abstraction for LayerNormalization to fit into neural network framework

4. **Clean Architecture**: Each component is self-contained:
   - Manages its own weights (Matrix objects)
   - Implements forward/backward passes
   - Handles parameter updates

5. **Efficient Memory**: 
   - Shared Matrix operations (no duplication)
   - Cache intermediate values for backward pass
   - Reuse attention weights across heads

The transformer demonstrates excellent software engineering: modular design, clear interfaces, and optimal use of the core Matrix library!
