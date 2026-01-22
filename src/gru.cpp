#include "nn/gru.h"
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>

// ═══════════════════════════════════════════════════════════════════════════
// GRUCell Implementation
// ═══════════════════════════════════════════════════════════════════════════

GRUCell::GRUCell(size_t input_size, size_t hidden_size)
    : input_size(input_size), hidden_size(hidden_size),
      // Initialize weight matrices (3 sets: reset, update, candidate)
      W_r(hidden_size, input_size), W_z(hidden_size, input_size),
      W_h(hidden_size, input_size),
      U_r(hidden_size, hidden_size), U_z(hidden_size, hidden_size),
      U_h(hidden_size, hidden_size),
      b_r(hidden_size, 1), b_z(hidden_size, 1), b_h(hidden_size, 1),
      // Initialize gradient matrices
      dW_r(hidden_size, input_size), dW_z(hidden_size, input_size),
      dW_h(hidden_size, input_size),
      dU_r(hidden_size, hidden_size), dU_z(hidden_size, hidden_size),
      dU_h(hidden_size, hidden_size),
      db_r(hidden_size, 1), db_z(hidden_size, 1), db_h(hidden_size, 1),
      // Activations
      sigmoid(std::make_unique<Sigmoid>()),
      tanh_activation(std::make_unique<Tanh>()) {
    
    initializeWeights("xavier");
}

void GRUCell::initializeWeights(const std::string& strategy) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Xavier/Glorot initialization
    double limit_input = std::sqrt(6.0 / (input_size + hidden_size));
    double limit_hidden = std::sqrt(6.0 / (2 * hidden_size));
    
    std::uniform_real_distribution<> dis_input(-limit_input, limit_input);
    std::uniform_real_distribution<> dis_hidden(-limit_hidden, limit_hidden);
    
    // Initialize all weight matrices
    auto init_matrix = [&](Matrix& m, std::uniform_real_distribution<>& dis) {
        for (size_t i = 0; i < m.getRows(); ++i) {
            for (size_t j = 0; j < m.getCols(); ++j) {
                m.set(i, j, dis(gen));
            }
        }
    };
    
    init_matrix(W_r, dis_input); init_matrix(W_z, dis_input); init_matrix(W_h, dis_input);
    init_matrix(U_r, dis_hidden); init_matrix(U_z, dis_hidden); init_matrix(U_h, dis_hidden);
    
    // Initialize biases to zero
    b_r.zeros(); b_z.zeros(); b_h.zeros();
}

Matrix GRUCell::forward(const Matrix& input, const Matrix& prev_hidden) {
    // Cache inputs for backward pass
    cached_input = input;
    cached_prev_hidden = prev_hidden;
    
    // ──────────────────────────────────────────────────────────────────────
    // Step 1: RESET GATE - Decide how much past to forget
    // r(t) = σ(W_r·x(t) + U_r·h(t-1) + b_r)
    // ──────────────────────────────────────────────────────────────────────
    Matrix r_input_term = input * W_r.transpose();
    Matrix r_hidden_term = prev_hidden * U_r.transpose();
    Matrix r_pre = r_input_term + r_hidden_term;
    
    // Add bias
    for (size_t i = 0; i < r_pre.getRows(); ++i) {
        for (size_t j = 0; j < r_pre.getCols(); ++j) {
            r_pre.set(i, j, r_pre.get(i, j) + b_r.get(j, 0));
        }
    }
    cached_reset_gate = sigmoid->forward(r_pre);
    
    // ──────────────────────────────────────────────────────────────────────
    // Step 2: UPDATE GATE - Decide how much to update
    // z(t) = σ(W_z·x(t) + U_z·h(t-1) + b_z)
    // ──────────────────────────────────────────────────────────────────────
    Matrix z_input_term = input * W_z.transpose();
    Matrix z_hidden_term = prev_hidden * U_z.transpose();
    Matrix z_pre = z_input_term + z_hidden_term;
    
    for (size_t i = 0; i < z_pre.getRows(); ++i) {
        for (size_t j = 0; j < z_pre.getCols(); ++j) {
            z_pre.set(i, j, z_pre.get(i, j) + b_z.get(j, 0));
        }
    }
    cached_update_gate = sigmoid->forward(z_pre);
    
    // ──────────────────────────────────────────────────────────────────────
    // Step 3: CANDIDATE - Compute new candidate hidden state
    // h̃(t) = tanh(W_h·x(t) + U_h·(r(t) ⊙ h(t-1)) + b_h)
    // ──────────────────────────────────────────────────────────────────────
    cached_reset_hidden = cached_reset_gate.hadamard(prev_hidden);
    
    Matrix h_input_term = input * W_h.transpose();
    Matrix h_hidden_term = cached_reset_hidden * U_h.transpose();
    Matrix h_pre = h_input_term + h_hidden_term;
    
    for (size_t i = 0; i < h_pre.getRows(); ++i) {
        for (size_t j = 0; j < h_pre.getCols(); ++j) {
            h_pre.set(i, j, h_pre.get(i, j) + b_h.get(j, 0));
        }
    }
    cached_candidate = tanh_activation->forward(h_pre);
    
    // ──────────────────────────────────────────────────────────────────────
    // Step 4: FINAL HIDDEN STATE - Interpolate between old and new
    // h(t) = (1 - z(t)) ⊙ h(t-1) + z(t) ⊙ h̃(t)
    // ──────────────────────────────────────────────────────────────────────
    Matrix one_minus_z = cached_update_gate.apply([](double x) { return 1.0 - x; });
    cached_hidden = one_minus_z.hadamard(prev_hidden) + 
                    cached_update_gate.hadamard(cached_candidate);
    
    return cached_hidden;
}

std::pair<Matrix, Matrix> GRUCell::backward(const Matrix& grad_hidden) {
    // ──────────────────────────────────────────────────────────────────────
    // BACKWARD PASS - Chain rule through all gates
    // ──────────────────────────────────────────────────────────────────────
    
    // Gradient through final interpolation
    // h(t) = (1-z) ⊙ h(t-1) + z ⊙ h̃
    Matrix one_minus_z = cached_update_gate.apply([](double x) { return 1.0 - x; });
    
    // ∂L/∂h̃ = ∂L/∂h · z
    Matrix grad_candidate = grad_hidden.hadamard(cached_update_gate);
    
    // ∂L/∂z = ∂L/∂h · (h̃ - h(t-1))
    Matrix diff = cached_candidate - cached_prev_hidden;
    Matrix grad_z = grad_hidden.hadamard(diff);
    
    // ∂L/∂h(t-1) from interpolation = ∂L/∂h · (1-z)
    Matrix grad_prev_from_interp = grad_hidden.hadamard(one_minus_z);
    
    // Backward through candidate (tanh)
    Matrix grad_candidate_pre = tanh_activation->backward(cached_candidate, grad_candidate);
    
    // Backward through update gate (sigmoid)
    Matrix grad_z_pre = sigmoid->backward(cached_update_gate, grad_z);
    
    // Gradient w.r.t. reset-gated hidden: ∂L/∂(r ⊙ h(t-1))
    Matrix grad_reset_hidden = grad_candidate_pre * U_h;
    
    // Gradient through reset gate
    Matrix grad_r = grad_reset_hidden.hadamard(cached_prev_hidden);
    Matrix grad_r_pre = sigmoid->backward(cached_reset_gate, grad_r);
    
    // Gradient w.r.t. previous hidden from reset gate
    Matrix grad_prev_from_reset = grad_reset_hidden.hadamard(cached_reset_gate);
    
    // Total gradient w.r.t. previous hidden
    Matrix grad_prev_hidden = grad_prev_from_interp + grad_prev_from_reset;
    
    // Accumulate weight gradients
    dW_r = dW_r + grad_r_pre.transpose() * cached_input;
    dW_z = dW_z + grad_z_pre.transpose() * cached_input;
    dW_h = dW_h + grad_candidate_pre.transpose() * cached_input;
    
    dU_r = dU_r + grad_r_pre.transpose() * cached_prev_hidden;
    dU_z = dU_z + grad_z_pre.transpose() * cached_prev_hidden;
    dU_h = dU_h + grad_candidate_pre.transpose() * cached_reset_hidden;
    
    // Bias gradients (sum across batch)
    for (size_t j = 0; j < hidden_size; ++j) {
        double sum_r = 0.0, sum_z = 0.0, sum_h = 0.0;
        for (size_t i = 0; i < grad_r_pre.getRows(); ++i) {
            sum_r += grad_r_pre.get(i, j);
            sum_z += grad_z_pre.get(i, j);
            sum_h += grad_candidate_pre.get(i, j);
        }
        db_r.set(j, 0, db_r.get(j, 0) + sum_r);
        db_z.set(j, 0, db_z.get(j, 0) + sum_z);
        db_h.set(j, 0, db_h.get(j, 0) + sum_h);
    }
    
    // Gradient w.r.t. input
    Matrix grad_input = grad_r_pre * W_r + grad_z_pre * W_z + grad_candidate_pre * W_h;
    
    // Add gradient w.r.t. previous hidden from update and reset gates
    grad_prev_hidden = grad_prev_hidden + grad_r_pre * U_r + grad_z_pre * U_z;
    
    return {grad_input, grad_prev_hidden};
}

void GRUCell::updateParameters(double learning_rate) {
    W_r = W_r - dW_r * learning_rate;
    W_z = W_z - dW_z * learning_rate;
    W_h = W_h - dW_h * learning_rate;
    
    U_r = U_r - dU_r * learning_rate;
    U_z = U_z - dU_z * learning_rate;
    U_h = U_h - dU_h * learning_rate;
    
    b_r = b_r - db_r * learning_rate;
    b_z = b_z - db_z * learning_rate;
    b_h = b_h - db_h * learning_rate;
    
    resetGradients();
}

void GRUCell::resetGradients() {
    dW_r.zeros(); dW_z.zeros(); dW_h.zeros();
    dU_r.zeros(); dU_z.zeros(); dU_h.zeros();
    db_r.zeros(); db_z.zeros(); db_h.zeros();
}

// ═══════════════════════════════════════════════════════════════════════════
// GRULayer Implementation
// ═══════════════════════════════════════════════════════════════════════════

GRULayer::GRULayer(size_t input_size, size_t hidden_size, size_t output_size,
                   bool return_sequences,
                   Activation* output_activation)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
      return_sequences(return_sequences),
      cell(input_size, hidden_size),
      W_hy(output_size, hidden_size),
      b_y(output_size, 1),
      dW_hy(output_size, hidden_size),
      db_y(output_size, 1),
      output_activation(output_activation ? std::unique_ptr<Activation>(output_activation)
                                          : std::make_unique<Linear>()) {
    
    initializeWeights("xavier");
}

void GRULayer::initializeWeights(const std::string& strategy) {
    cell.initializeWeights(strategy);
    W_hy.xavierInit(hidden_size, output_size);
    b_y.zeros();
}

Matrix GRULayer::forward(const std::vector<Matrix>& sequence,
                         const Matrix& initial_hidden) {
    size_t seq_length = sequence.size();
    size_t batch_size = sequence[0].getRows();
    
    // Initialize hidden state
    Matrix h = initial_hidden.getRows() > 0 ? initial_hidden : Matrix(batch_size, hidden_size);
    
    // Clear cached states
    hidden_states.clear();
    inputs.clear();
    
    // Process sequence
    for (size_t t = 0; t < seq_length; ++t) {
        inputs.push_back(sequence[t]);
        h = cell.forward(sequence[t], h);
        hidden_states.push_back(h);
    }
    
    // Compute output
    if (return_sequences) {
        // Return output for all time steps
        Matrix all_outputs(seq_length * batch_size, output_size);
        for (size_t t = 0; t < seq_length; ++t) {
            Matrix y = hidden_states[t] * W_hy.transpose();
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < output_size; ++j) {
                    y.set(i, j, y.get(i, j) + b_y.get(j, 0));
                }
            }
            y = output_activation->forward(y);
            
            // Copy to result
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < output_size; ++j) {
                    all_outputs.set(t * batch_size + i, j, y.get(i, j));
                }
            }
        }
        return all_outputs;
    } else {
        // Return only last time step
        Matrix y = h * W_hy.transpose();
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                y.set(i, j, y.get(i, j) + b_y.get(j, 0));
            }
        }
        return output_activation->forward(y);
    }
}

std::vector<Matrix> GRULayer::backward(const std::vector<Matrix>& grad_output) {
    size_t seq_length = hidden_states.size();
    size_t batch_size = hidden_states[0].getRows();
    
    std::vector<Matrix> grad_inputs(seq_length);
    Matrix grad_h_next(batch_size, hidden_size);
    
    dW_hy.zeros();
    db_y.zeros();
    
    // Backpropagate through time
    for (int t = seq_length - 1; t >= 0; --t) {
        // Gradient from output
        Matrix grad_y = grad_output[t];
        grad_y = output_activation->backward(hidden_states[t], grad_y);
        
        // Accumulate output layer gradients
        dW_hy = dW_hy + grad_y.transpose() * hidden_states[t];
        for (size_t j = 0; j < output_size; ++j) {
            double sum = 0.0;
            for (size_t i = 0; i < batch_size; ++i) {
                sum += grad_y.get(i, j);
            }
            db_y.set(j, 0, db_y.get(j, 0) + sum);
        }
        
        // Gradient w.r.t. hidden state
        Matrix grad_h = grad_y * W_hy + grad_h_next;
        
        // Backward through GRU cell
        auto [grad_input, grad_prev_h] = cell.backward(grad_h);
        
        grad_inputs[t] = grad_input;
        grad_h_next = grad_prev_h;
    }
    
    return grad_inputs;
}

void GRULayer::updateParameters(double learning_rate) {
    cell.updateParameters(learning_rate);
    W_hy = W_hy - dW_hy * learning_rate;
    b_y = b_y - db_y * learning_rate;
    resetGradients();
}

void GRULayer::resetGradients() {
    cell.resetGradients();
    dW_hy.zeros();
    db_y.zeros();
}

// ═══════════════════════════════════════════════════════════════════════════
// GRUNetwork Implementation
// ═══════════════════════════════════════════════════════════════════════════

void GRUNetwork::addLayer(GRULayer* layer) {
    layers.push_back(std::unique_ptr<GRULayer>(layer));
}

Matrix GRUNetwork::forward(const std::vector<Matrix>& sequence) {
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    return layers[0]->forward(sequence);
}

void GRUNetwork::train(const std::vector<std::vector<Matrix>>& sequences,
                       const std::vector<Matrix>& targets,
                       int epochs,
                       double learning_rate,
                       bool verbose) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < sequences.size(); ++i) {
            // Forward pass
            Matrix prediction = forward(sequences[i]);
            
            // Calculate loss (MSE)
            Matrix error = prediction - targets[i];
            double loss = 0.0;
            for (size_t r = 0; r < error.getRows(); ++r) {
                for (size_t c = 0; c < error.getCols(); ++c) {
                    loss += error.get(r, c) * error.get(r, c);
                }
            }
            total_loss += loss / (error.getRows() * error.getCols());
            
            // Backward pass
            std::vector<Matrix> grad_output = {error};
            for (auto& layer : layers) {
                grad_output = layer->backward(grad_output);
            }
            
            // Update parameters
            for (auto& layer : layers) {
                layer->updateParameters(learning_rate);
            }
        }
        
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << std::setw(4) << epoch 
                     << " | Loss: " << std::fixed << std::setprecision(6) 
                     << total_loss / sequences.size() << std::endl;
        }
    }
}

Matrix GRUNetwork::predict(const std::vector<Matrix>& sequence) {
    return forward(sequence);
}

void GRUNetwork::summary() const {
    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "                   GRU NETWORK SUMMARY                  \n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    
    int total_params = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        int params = layers[i]->getParameterCount();
        total_params += params;
        std::cout << "Layer " << i << ": GRU\n";
        std::cout << "  Input: " << layers[i]->getInputSize() << "\n";
        std::cout << "  Hidden: " << layers[i]->getHiddenSize() << "\n";
        std::cout << "  Output: " << layers[i]->getOutputSize() << "\n";
        std::cout << "  Parameters: " << params << "\n\n";
    }
    
    std::cout << "Total parameters: " << total_params << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
}
