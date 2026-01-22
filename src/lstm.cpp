#include "nn/lstm.h"
#include <cmath>
#include <iostream>
#include <random>
#include <iomanip>

// ═══════════════════════════════════════════════════════════════════════════
// LSTMCell Implementation
// ═══════════════════════════════════════════════════════════════════════════

LSTMCell::LSTMCell(size_t input_size, size_t hidden_size)
    : input_size(input_size), hidden_size(hidden_size),
      // Initialize weight matrices
      W_f(hidden_size, input_size), W_i(hidden_size, input_size),
      W_c(hidden_size, input_size), W_o(hidden_size, input_size),
      U_f(hidden_size, hidden_size), U_i(hidden_size, hidden_size),
      U_c(hidden_size, hidden_size), U_o(hidden_size, hidden_size),
      b_f(hidden_size, 1), b_i(hidden_size, 1),
      b_c(hidden_size, 1), b_o(hidden_size, 1),
      // Initialize gradient matrices
      dW_f(hidden_size, input_size), dW_i(hidden_size, input_size),
      dW_c(hidden_size, input_size), dW_o(hidden_size, input_size),
      dU_f(hidden_size, hidden_size), dU_i(hidden_size, hidden_size),
      dU_c(hidden_size, hidden_size), dU_o(hidden_size, hidden_size),
      db_f(hidden_size, 1), db_i(hidden_size, 1),
      db_c(hidden_size, 1), db_o(hidden_size, 1),
      // Activations
      sigmoid(std::make_unique<Sigmoid>()),
      tanh_activation(std::make_unique<Tanh>()) {
    
    initializeWeights("xavier");
}

void LSTMCell::initializeWeights(const std::string& strategy) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Xavier/Glorot initialization for better gradient flow
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
    
    init_matrix(W_f, dis_input); init_matrix(W_i, dis_input);
    init_matrix(W_c, dis_input); init_matrix(W_o, dis_input);
    init_matrix(U_f, dis_hidden); init_matrix(U_i, dis_hidden);
    init_matrix(U_c, dis_hidden); init_matrix(U_o, dis_hidden);
    
    // Initialize biases
    // Forget gate bias to 1.0 (helps learning at start - remember more initially)
    for (size_t i = 0; i < hidden_size; ++i) {
        b_f.set(i, 0, 1.0);  // Bias forget gate toward remembering
        b_i.set(i, 0, 0.0);
        b_c.set(i, 0, 0.0);
        b_o.set(i, 0, 0.0);
    }
}

std::pair<Matrix, Matrix> LSTMCell::forward(const Matrix& input,
                                            const Matrix& prev_hidden,
                                            const Matrix& prev_cell) {
    // Cache inputs for backward pass
    cached_input = input;
    cached_prev_hidden = prev_hidden;
    cached_prev_cell = prev_cell;
    
    // ──────────────────────────────────────────────────────────────────────
    // Step 1: FORGET GATE - Decide what to forget from cell state
    // f(t) = σ(W_f·x(t) + U_f·h(t-1) + b_f)
    // ──────────────────────────────────────────────────────────────────────
    Matrix f_input_term = input * W_f.transpose();
    Matrix f_hidden_term = prev_hidden * U_f.transpose();
    Matrix f_pre = f_input_term + f_hidden_term;
    
    // Add bias
    for (size_t i = 0; i < f_pre.getRows(); ++i) {
        for (size_t j = 0; j < f_pre.getCols(); ++j) {
            f_pre.set(i, j, f_pre.get(i, j) + b_f.get(j, 0));
        }
    }
    cached_forget_gate = sigmoid->forward(f_pre);
    
    // ──────────────────────────────────────────────────────────────────────
    // Step 2: INPUT GATE - Decide what new information to store
    // i(t) = σ(W_i·x(t) + U_i·h(t-1) + b_i)
    // C̃(t) = tanh(W_c·x(t) + U_c·h(t-1) + b_c)
    // ──────────────────────────────────────────────────────────────────────
    Matrix i_input_term = input * W_i.transpose();
    Matrix i_hidden_term = prev_hidden * U_i.transpose();
    Matrix i_pre = i_input_term + i_hidden_term;
    for (size_t i = 0; i < i_pre.getRows(); ++i) {
        for (size_t j = 0; j < i_pre.getCols(); ++j) {
            i_pre.set(i, j, i_pre.get(i, j) + b_i.get(j, 0));
        }
    }
    cached_input_gate = sigmoid->forward(i_pre);
    
    Matrix c_input_term = input * W_c.transpose();
    Matrix c_hidden_term = prev_hidden * U_c.transpose();
    Matrix c_pre = c_input_term + c_hidden_term;
    for (size_t i = 0; i < c_pre.getRows(); ++i) {
        for (size_t j = 0; j < c_pre.getCols(); ++j) {
            c_pre.set(i, j, c_pre.get(i, j) + b_c.get(j, 0));
        }
    }
    cached_candidate = tanh_activation->forward(c_pre);
    
    // ──────────────────────────────────────────────────────────────────────
    // Step 3: UPDATE CELL STATE - The memory highway!
    // C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)
    // ──────────────────────────────────────────────────────────────────────
    cached_cell = cached_forget_gate.hadamard(prev_cell) + 
                  cached_input_gate.hadamard(cached_candidate);
    
    // ──────────────────────────────────────────────────────────────────────
    // Step 4: OUTPUT GATE - Decide what to output
    // o(t) = σ(W_o·x(t) + U_o·h(t-1) + b_o)
    // h(t) = o(t) ⊙ tanh(C(t))
    // ──────────────────────────────────────────────────────────────────────
    Matrix o_input_term = input * W_o.transpose();
    Matrix o_hidden_term = prev_hidden * U_o.transpose();
    Matrix o_pre = o_input_term + o_hidden_term;
    for (size_t i = 0; i < o_pre.getRows(); ++i) {
        for (size_t j = 0; j < o_pre.getCols(); ++j) {
            o_pre.set(i, j, o_pre.get(i, j) + b_o.get(j, 0));
        }
    }
    cached_output_gate = sigmoid->forward(o_pre);
    
    Matrix cell_tanh = tanh_activation->forward(cached_cell);
    cached_hidden = cached_output_gate.hadamard(cell_tanh);
    
    return {cached_hidden, cached_cell};
}

std::tuple<Matrix, Matrix, Matrix> LSTMCell::backward(const Matrix& grad_hidden,
                                                       const Matrix& grad_cell) {
    // ──────────────────────────────────────────────────────────────────────
    // BACKWARD PASS - Chain rule through all gates
    // ──────────────────────────────────────────────────────────────────────
    
    // Gradient through output gate
    Matrix cell_tanh = tanh_activation->forward(cached_cell);
    Matrix grad_o = grad_hidden.hadamard(cell_tanh);
    Matrix grad_o_pre = sigmoid->backward(cached_output_gate, grad_o);
    
    // Gradient w.r.t. cell from output
    Matrix grad_cell_from_output = grad_hidden.hadamard(cached_output_gate);
    Matrix grad_cell_tanh = tanh_activation->backward(cached_cell, grad_cell_from_output);
    Matrix total_grad_cell = grad_cell_tanh + grad_cell;
    
    // Gradient through input gate and candidate
    Matrix grad_candidate = total_grad_cell.hadamard(cached_input_gate);
    Matrix grad_candidate_pre = tanh_activation->backward(cached_candidate, grad_candidate);
    
    Matrix grad_i = total_grad_cell.hadamard(cached_candidate);
    Matrix grad_i_pre = sigmoid->backward(cached_input_gate, grad_i);
    
    // Gradient through forget gate
    Matrix grad_f = total_grad_cell.hadamard(cached_prev_cell);
    Matrix grad_f_pre = sigmoid->backward(cached_forget_gate, grad_f);
    
    // Gradient w.r.t. previous cell state
    Matrix grad_prev_cell = total_grad_cell.hadamard(cached_forget_gate);
    
    // Accumulate weight gradients
    dW_f = dW_f + grad_f_pre.transpose() * cached_input;
    dW_i = dW_i + grad_i_pre.transpose() * cached_input;
    dW_c = dW_c + grad_candidate_pre.transpose() * cached_input;
    dW_o = dW_o + grad_o_pre.transpose() * cached_input;
    
    dU_f = dU_f + grad_f_pre.transpose() * cached_prev_hidden;
    dU_i = dU_i + grad_i_pre.transpose() * cached_prev_hidden;
    dU_c = dU_c + grad_candidate_pre.transpose() * cached_prev_hidden;
    dU_o = dU_o + grad_o_pre.transpose() * cached_prev_hidden;
    
    // Bias gradients
    for (size_t j = 0; j < hidden_size; ++j) {
        double sum_f = 0.0, sum_i = 0.0, sum_c = 0.0, sum_o = 0.0;
        for (size_t i = 0; i < grad_f_pre.getRows(); ++i) {
            sum_f += grad_f_pre.get(i, j);
            sum_i += grad_i_pre.get(i, j);
            sum_c += grad_candidate_pre.get(i, j);
            sum_o += grad_o_pre.get(i, j);
        }
        db_f.set(j, 0, db_f.get(j, 0) + sum_f);
        db_i.set(j, 0, db_i.get(j, 0) + sum_i);
        db_c.set(j, 0, db_c.get(j, 0) + sum_c);
        db_o.set(j, 0, db_o.get(j, 0) + sum_o);
    }
    
    // Gradient w.r.t. input
    Matrix grad_input = grad_f_pre * W_f + grad_i_pre * W_i + 
                       grad_candidate_pre * W_c + grad_o_pre * W_o;
    
    // Gradient w.r.t. previous hidden
    Matrix grad_prev_hidden = grad_f_pre * U_f + grad_i_pre * U_i + 
                             grad_candidate_pre * U_c + grad_o_pre * U_o;
    
    return {grad_input, grad_prev_hidden, grad_prev_cell};
}

void LSTMCell::updateParameters(double learning_rate) {
    W_f = W_f - dW_f * learning_rate;
    W_i = W_i - dW_i * learning_rate;
    W_c = W_c - dW_c * learning_rate;
    W_o = W_o - dW_o * learning_rate;
    
    U_f = U_f - dU_f * learning_rate;
    U_i = U_i - dU_i * learning_rate;
    U_c = U_c - dU_c * learning_rate;
    U_o = U_o - dU_o * learning_rate;
    
    b_f = b_f - db_f * learning_rate;
    b_i = b_i - db_i * learning_rate;
    b_c = b_c - db_c * learning_rate;
    b_o = b_o - db_o * learning_rate;
    
    resetGradients();
}

void LSTMCell::resetGradients() {
    dW_f.zeros(); dW_i.zeros(); dW_c.zeros(); dW_o.zeros();
    dU_f.zeros(); dU_i.zeros(); dU_c.zeros(); dU_o.zeros();
    db_f.zeros(); db_i.zeros(); db_c.zeros(); db_o.zeros();
}

// ═══════════════════════════════════════════════════════════════════════════
// LSTMLayer Implementation
// ═══════════════════════════════════════════════════════════════════════════

LSTMLayer::LSTMLayer(size_t input_size, size_t hidden_size, size_t output_size,
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

void LSTMLayer::initializeWeights(const std::string& strategy) {
    cell.initializeWeights(strategy);
    W_hy.xavierInit(hidden_size, output_size);
    b_y.zeros();
}

Matrix LSTMLayer::forward(const std::vector<Matrix>& sequence,
                          const Matrix& initial_hidden,
                          const Matrix& initial_cell) {
    size_t seq_length = sequence.size();
    size_t batch_size = sequence[0].getRows();
    
    // Initialize hidden and cell states
    Matrix h = initial_hidden.getRows() > 0 ? initial_hidden : Matrix(batch_size, hidden_size);
    Matrix c = initial_cell.getRows() > 0 ? initial_cell : Matrix(batch_size, hidden_size);
    
    // Clear cached states
    hidden_states.clear();
    cell_states.clear();
    inputs.clear();
    
    // Process sequence
    for (size_t t = 0; t < seq_length; ++t) {
        inputs.push_back(sequence[t]);
        auto [new_h, new_c] = cell.forward(sequence[t], h, c);
        h = new_h;
        c = new_c;
        hidden_states.push_back(h);
        cell_states.push_back(c);
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

std::vector<Matrix> LSTMLayer::backward(const std::vector<Matrix>& grad_output) {
    size_t seq_length = hidden_states.size();
    size_t batch_size = hidden_states[0].getRows();
    
    std::vector<Matrix> grad_inputs(seq_length);
    Matrix grad_h_next(batch_size, hidden_size);
    Matrix grad_c_next(batch_size, hidden_size);
    
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
        
        // Gradient w.r.t. cell state
        Matrix grad_c = grad_c_next;
        
        // Get previous states
        Matrix prev_h = (t > 0) ? hidden_states[t-1] : Matrix(batch_size, hidden_size);
        Matrix prev_c = (t > 0) ? cell_states[t-1] : Matrix(batch_size, hidden_size);
        
        // Backward through LSTM cell
        auto [grad_input, grad_prev_h, grad_prev_c] = cell.backward(grad_h, grad_c);
        
        grad_inputs[t] = grad_input;
        grad_h_next = grad_prev_h;
        grad_c_next = grad_prev_c;
    }
    
    return grad_inputs;
}

void LSTMLayer::updateParameters(double learning_rate) {
    cell.updateParameters(learning_rate);
    W_hy = W_hy - dW_hy * learning_rate;
    b_y = b_y - db_y * learning_rate;
    resetGradients();
}

void LSTMLayer::resetGradients() {
    cell.resetGradients();
    dW_hy.zeros();
    db_y.zeros();
}

// ═══════════════════════════════════════════════════════════════════════════
// LSTMNetwork Implementation
// ═══════════════════════════════════════════════════════════════════════════

void LSTMNetwork::addLayer(LSTMLayer* layer) {
    layers.push_back(std::unique_ptr<LSTMLayer>(layer));
}

Matrix LSTMNetwork::forward(const std::vector<Matrix>& sequence) {
    if (layers.empty()) {
        throw std::runtime_error("Network has no layers");
    }
    
    return layers[0]->forward(sequence);
}

void LSTMNetwork::train(const std::vector<std::vector<Matrix>>& sequences,
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

Matrix LSTMNetwork::predict(const std::vector<Matrix>& sequence) {
    return forward(sequence);
}

void LSTMNetwork::summary() const {
    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "                  LSTM NETWORK SUMMARY                  \n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    
    int total_params = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        int params = layers[i]->getParameterCount();
        total_params += params;
        std::cout << "Layer " << i << ": LSTM\n";
        std::cout << "  Input: " << layers[i]->getInputSize() << "\n";
        std::cout << "  Hidden: " << layers[i]->getHiddenSize() << "\n";
        std::cout << "  Output: " << layers[i]->getOutputSize() << "\n";
        std::cout << "  Parameters: " << params << "\n\n";
    }
    
    std::cout << "Total parameters: " << total_params << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
}
