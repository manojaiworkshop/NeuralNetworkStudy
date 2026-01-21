#include "nn/rnn.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================================
// RNN CELL IMPLEMENTATION
// ============================================================================

RNNCell::RNNCell(size_t input_size, size_t hidden_size, Activation* activation)
    : input_size(input_size), 
      hidden_size(hidden_size),
      W_xh(hidden_size, input_size),
      W_hh(hidden_size, hidden_size),
      b_h(hidden_size, 1),
      dW_xh(hidden_size, input_size),
      dW_hh(hidden_size, hidden_size),
      db_h(hidden_size, 1),
      activation(activation ? std::unique_ptr<Activation>(activation) 
                           : std::make_unique<Tanh>()) {
    
    initializeWeights("xavier");
    resetGradients();
}

void RNNCell::initializeWeights(const std::string& strategy) {
    if (strategy == "xavier") {
        W_xh.xavierInit(input_size, hidden_size);
        W_hh.xavierInit(hidden_size, hidden_size);
    } else if (strategy == "he") {
        W_xh.heInit(input_size);
        W_hh.heInit(hidden_size);
    } else {
        W_xh.randomize(-0.1, 0.1);
        W_hh.randomize(-0.1, 0.1);
    }
    b_h.zeros();
}

Matrix RNNCell::forward(const Matrix& input, const Matrix& prev_hidden) {
    // Cache for backward pass
    cached_input = input;
    cached_prev_hidden = prev_hidden;
    
    // Compute new hidden state:
    // h(t) = activation(W_xh * x(t) + W_hh * h(t-1) + b_h)
    
    // W_xh * x(t)^T
    Matrix xh_term = input * W_xh.transpose();  // (batch_size x hidden_size)
    
    // W_hh * h(t-1)^T
    Matrix hh_term = prev_hidden * W_hh.transpose();  // (batch_size x hidden_size)
    
    // Add bias
    Matrix pre_activation = xh_term + hh_term;
    for (size_t i = 0; i < pre_activation.getRows(); ++i) {
        for (size_t j = 0; j < pre_activation.getCols(); ++j) {
            pre_activation.set(i, j, pre_activation.get(i, j) + b_h.get(j, 0));
        }
    }
    
    // Apply activation
    cached_hidden = activation->forward(pre_activation);
    
    return cached_hidden;
}

Matrix RNNCell::backward(const Matrix& grad_hidden, const Matrix& grad_next_hidden) {
    // Total gradient w.r.t. hidden state
    Matrix total_grad = grad_hidden + grad_next_hidden;
    
    // Gradient through activation
    Matrix grad_pre_activation = activation->backward(cached_hidden, total_grad);
    
    // Gradient w.r.t. W_xh: dL/dW_xh = grad_pre_activation^T * input
    dW_xh = dW_xh + grad_pre_activation.transpose() * cached_input;
    
    // Gradient w.r.t. W_hh: dL/dW_hh = grad_pre_activation^T * prev_hidden
    dW_hh = dW_hh + grad_pre_activation.transpose() * cached_prev_hidden;
    
    // Gradient w.r.t. b_h: sum across batch
    for (size_t j = 0; j < hidden_size; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < grad_pre_activation.getRows(); ++i) {
            sum += grad_pre_activation.get(i, j);
        }
        db_h.set(j, 0, db_h.get(j, 0) + sum);
    }
    
    // Gradient w.r.t. input
    Matrix grad_input = grad_pre_activation * W_xh;
    
    // Gradient w.r.t. previous hidden (for BPTT)
    Matrix grad_prev_hidden = grad_pre_activation * W_hh;
    
    return grad_input;
}

void RNNCell::updateParameters(double learning_rate) {
    W_xh = W_xh - dW_xh * learning_rate;
    W_hh = W_hh - dW_hh * learning_rate;
    b_h = b_h - db_h * learning_rate;
    resetGradients();
}

void RNNCell::resetGradients() {
    dW_xh.zeros();
    dW_hh.zeros();
    db_h.zeros();
}

// ============================================================================
// RNN LAYER IMPLEMENTATION
// ============================================================================

RNNLayer::RNNLayer(size_t input_size, size_t hidden_size, size_t output_size,
                   bool return_sequences,
                   Activation* hidden_activation,
                   Activation* output_activation)
    : input_size(input_size),
      hidden_size(hidden_size),
      output_size(output_size),
      return_sequences(return_sequences),
      cell(input_size, hidden_size, hidden_activation),
      W_hy(output_size, hidden_size),
      b_y(output_size, 1),
      dW_hy(output_size, hidden_size),
      db_y(output_size, 1),
      output_activation(output_activation ? std::unique_ptr<Activation>(output_activation)
                                         : std::make_unique<Linear>()) {
    
    initializeWeights("xavier");
}

void RNNLayer::initializeWeights(const std::string& strategy) {
    cell.initializeWeights(strategy);
    if (strategy == "xavier") {
        W_hy.xavierInit(hidden_size, output_size);
    } else {
        W_hy.randomize(-0.1, 0.1);
    }
    b_y.zeros();
}

Matrix RNNLayer::forward(const std::vector<Matrix>& sequence, 
                        const Matrix& initial_hidden) {
    
    if (sequence.empty()) {
        throw std::invalid_argument("Empty sequence in RNNLayer::forward");
    }
    
    // Initialize hidden state
    size_t batch_size = sequence[0].getRows();
    Matrix hidden = initial_hidden;
    if (hidden.getRows() == 0) {
        hidden = Matrix(batch_size, hidden_size);
        hidden.zeros();
    }
    
    // Clear caches
    hidden_states.clear();
    inputs.clear();
    
    // Process sequence
    std::vector<Matrix> outputs;
    for (const auto& input : sequence) {
        inputs.push_back(input);
        
        // Update hidden state
        hidden = cell.forward(input, hidden);
        hidden_states.push_back(hidden);
        
        // Compute output: y = W_hy * h + b_y
        Matrix output = hidden * W_hy.transpose();
        for (size_t i = 0; i < output.getRows(); ++i) {
            for (size_t j = 0; j < output.getCols(); ++j) {
                output.set(i, j, output.get(i, j) + b_y.get(j, 0));
            }
        }
        
        output = output_activation->forward(output);
        outputs.push_back(output);
    }
    
    // Return all outputs or just the last one
    if (return_sequences) {
        // Concatenate all outputs
        size_t total_rows = 0;
        for (const auto& out : outputs) {
            total_rows += out.getRows();
        }
        
        Matrix result(total_rows, output_size);
        size_t row_offset = 0;
        for (const auto& out : outputs) {
            for (size_t i = 0; i < out.getRows(); ++i) {
                for (size_t j = 0; j < out.getCols(); ++j) {
                    result.set(row_offset + i, j, out.get(i, j));
                }
            }
            row_offset += out.getRows();
        }
        return result;
    } else {
        return outputs.back();
    }
}

std::vector<Matrix> RNNLayer::backward(const std::vector<Matrix>& grad_output) {
    std::vector<Matrix> grad_inputs;
    
    // Initialize gradient for next time step
    size_t batch_size = hidden_states[0].getRows();
    Matrix grad_h_next(batch_size, hidden_size);
    grad_h_next.zeros();
    
    // Backpropagate through time (BPTT)
    for (int t = hidden_states.size() - 1; t >= 0; --t) {
        // Gradient from output
        Matrix grad_out = grad_output[t];
        
        // Gradient through output activation
        Matrix grad_pre_out = output_activation->backward(
            hidden_states[t] * W_hy.transpose(), grad_out);
        
        // Gradient w.r.t. W_hy
        dW_hy = dW_hy + grad_pre_out.transpose() * hidden_states[t];
        
        // Gradient w.r.t. b_y
        for (size_t j = 0; j < output_size; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < grad_pre_out.getRows(); ++i) {
                sum += grad_pre_out.get(i, j);
            }
            db_y.set(j, 0, db_y.get(j, 0) + sum);
        }
        
        // Gradient w.r.t. hidden state
        Matrix grad_h = grad_pre_out * W_hy + grad_h_next;
        
        // Backprop through RNN cell
        Matrix prev_hidden = (t > 0) ? hidden_states[t-1] : Matrix(batch_size, hidden_size);
        if (t == 0) prev_hidden.zeros();
        
        Matrix grad_input = cell.backward(grad_h, Matrix(batch_size, hidden_size));
        grad_inputs.insert(grad_inputs.begin(), grad_input);
        
        // Update grad_h_next for next iteration
        grad_h_next = grad_pre_out * W_hy;
    }
    
    return grad_inputs;
}

void RNNLayer::updateParameters(double learning_rate) {
    cell.updateParameters(learning_rate);
    W_hy = W_hy - dW_hy * learning_rate;
    b_y = b_y - db_y * learning_rate;
    resetGradients();
}

void RNNLayer::resetGradients() {
    cell.resetGradients();
    dW_hy.zeros();
    db_y.zeros();
}

// ============================================================================
// RNN NETWORK IMPLEMENTATION
// ============================================================================

void RNNNetwork::addLayer(RNNLayer* layer) {
    layers.push_back(std::unique_ptr<RNNLayer>(layer));
}

Matrix RNNNetwork::forward(const std::vector<Matrix>& sequence) {
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    // For now, single layer forward pass
    return layers[0]->forward(sequence);
}

void RNNNetwork::train(const std::vector<std::vector<Matrix>>& sequences,
                      const std::vector<Matrix>& targets,
                      int epochs,
                      double learning_rate,
                      bool verbose) {
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < sequences.size(); ++i) {
            // Forward pass
            Matrix prediction = forward(sequences[i]);
            
            // Compute loss (MSE)
            Matrix diff = prediction - targets[i];
            double loss = 0.0;
            for (size_t r = 0; r < diff.getRows(); ++r) {
                for (size_t c = 0; c < diff.getCols(); ++c) {
                    loss += diff.get(r, c) * diff.get(r, c);
                }
            }
            loss /= (diff.getRows() * diff.getCols());
            total_loss += loss;
            
            // Backward pass
            Matrix grad = diff * (2.0 / (diff.getRows() * diff.getCols()));
            
            // Create gradient for each time step
            std::vector<Matrix> grad_seq;
            size_t seq_len = sequences[i].size();
            for (size_t t = 0; t < seq_len; ++t) {
                if (t == seq_len - 1) {
                    grad_seq.push_back(grad);
                } else {
                    Matrix zero_grad(grad.getRows(), grad.getCols());
                    zero_grad.zeros();
                    grad_seq.push_back(zero_grad);
                }
            }
            
            layers[0]->backward(grad_seq);
            layers[0]->updateParameters(learning_rate);
        }
        
        if (verbose && (epoch % 100 == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << std::setw(4) << epoch 
                     << " | Loss: " << std::fixed << std::setprecision(6) 
                     << total_loss / sequences.size() << std::endl;
        }
    }
}

Matrix RNNNetwork::predict(const std::vector<Matrix>& sequence) {
    return forward(sequence);
}

void RNNNetwork::summary() const {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    RNN NETWORK SUMMARY                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
    
    int total_params = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i << ": RNN\n";
        std::cout << "  Input size:     " << layers[i]->getInputSize() << "\n";
        std::cout << "  Hidden size:    " << layers[i]->getHiddenSize() << "\n";
        std::cout << "  Output size:    " << layers[i]->getOutputSize() << "\n";
        std::cout << "  Parameters:     " << layers[i]->getParameterCount() << "\n";
        std::cout << "  Return sequences: " << (layers[i]->getReturnSequences() ? "Yes" : "No") << "\n\n";
        total_params += layers[i]->getParameterCount();
    }
    
    std::cout << "Total parameters: " << total_params << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
}
