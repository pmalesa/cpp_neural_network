#include "neural_network.h"
#include "spdlog/spdlog.h"
#include "activation.h"
#include "loss.h"

#include <stdexcept>
#include <sstream>

using std::string;

NeuralNetwork::NeuralNetwork()
    : shape_({}), activation_functions_({}),
      weights_({}), built_(false), A_values_({}), Z_values_({}) {}

NeuralNetwork::NeuralNetwork(const vector<size_t>& new_shape, const vector<ActivationFunction>& new_activation_functions) 
    : shape_(new_shape), activation_functions_(new_activation_functions), 
      weights_({}), built_(false), A_values_({}), Z_values_({}) {
/*
    Network layer's sizes including input and output layers
*/
    if (new_shape.size() < 2 ) {
        throw std::logic_error("A neural network requires at least an input and output layer!");
    }

    if (new_activation_functions.size() != new_shape.size() - 1) {
        throw std::logic_error("A neural network requires one activation per non-input layer!");
    }

    for (size_t n_neurons : new_shape) {
        if (n_neurons == 0) {
            throw std::invalid_argument("Layer must contain at least one neuron!");
        }
    }
}

NeuralNetwork& NeuralNetwork::erase() {
    shape_.clear();
    activation_functions_.clear();
    weights_.clear();
    A_values_.clear();
    Z_values_.clear();
    built_ = false;
    return *this;
}

NeuralNetwork& NeuralNetwork::build() {
    spdlog::info("Building neural network...");
    if (built_) {
        string err_msg = "Network has already been built!";
        spdlog::error(err_msg);
        throw std::logic_error(err_msg); 
    }

    if (get_n_layers() < 2) {
        string err_msg = "Cannot build a network with fewer than 2 layers!";
        spdlog::error(err_msg);
        throw std::logic_error(err_msg); 
    }

    if (activation_functions_.size() != shape_.size() - 1) {
        string err_msg = "Invalid number of activation functions!";
        spdlog::error(err_msg);
        throw std::logic_error(err_msg);
    }

    for (size_t layer = 0; layer < get_n_layers() - 1; ++layer) {
        Matrix layer_weights(shape_[layer] + 1, shape_[layer + 1]);
        layer_weights.fill_random();
        weights_.push_back(layer_weights); 
    }
    built_ = true;

    spdlog::info("Neural network built successfully.");

    std::ostringstream log_msg_oss;
    log_msg_oss << "Number of hidden layers: " << (get_n_layers() > 2 ? get_n_layers() - 2 : 0) << " | Network structure: [";
    for (size_t layer = 0; layer < shape_.size(); ++layer) {
        log_msg_oss << shape_[layer];
        if (layer < shape_.size() - 1) {
            log_msg_oss << ", ";
        }
    }

    log_msg_oss << "]";
    spdlog::info(log_msg_oss.str());

    return *this;
}

Matrix NeuralNetwork::forward(const Matrix& input, bool learning) {
/*
It takes batch of column vectors on input.
*/
    if (!built_) {
        throw std::logic_error("Network must be built before forward pass!");
    }

    if (input.get_rows() != shape_.front()) {
        throw std::invalid_argument("Input feature count does not match the input layer!");
    }

    if (input.get_cols() == 0) {
        throw std::invalid_argument("Input batch cannot be empty!");
    }

    if (learning) {
        Z_values_.clear();
        A_values_.clear();
        A_values_.push_back(input); 
    }

    Matrix X = input;

    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        Matrix X_bias = X.add_row();
        const Matrix& W = weights_[layer];
        
        Matrix Z = W.transpose() * X_bias;

        if (learning) {
            Z_values_.push_back(Z);
        }

        X = apply_activation_(Z, layer);

        if (learning) {
            A_values_.push_back(X);
        }
    }

    return X;
}

void NeuralNetwork::backward(const Matrix& input, const Matrix& target, double learning_rate, LossFunction loss) {
    if (A_values_.empty() || Z_values_.empty()) {
        throw std::logic_error("Can not perform backward pass before the forward pass!");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive!");
    }

    if (A_values_.front().get_rows() != input.get_rows() || 
        A_values_.front().get_cols() != input.get_cols()) {
        throw std::invalid_argument("Input does not match the cached forward-pass input!");
    }

    const Matrix& prediction = A_values_.back();

    Matrix dA = apply_loss_derivative_(target, prediction, loss);

    for (size_t layer = weights_.size(); layer-- > 0;) {
        // Convert dA[layer + 1] to dZ[layer]
        Matrix dZ = activation_backward_(dA, Z_values_[layer], A_values_[layer + 1], layer);

        // dW[layer] = A_bias[layer] * dZ[layer]^T
        Matrix A_prev_bias = A_values_[layer].add_row();
        Matrix dW = A_prev_bias * dZ.transpose();

        // Use the old weights when propagating the gradient
        if (layer > 0) {
            Matrix W_no_bias = weights_[layer].remove_row();
            dA = W_no_bias * dZ; 
        }

        weights_[layer] = weights_[layer] - learning_rate * dW;
    }

    // Clear cached values which correspond to the previous weights
    A_values_.clear();
    Z_values_.clear();
}

NeuralNetwork& NeuralNetwork::add_input_layer(size_t n_neurons) {
    if (built_) {
        throw std::logic_error("Cannot add layers after building the network!");
    }

    if (!shape_.empty()) {
        throw std::logic_error("Input layer has already been added!");
    }

    if (n_neurons == 0) {
        throw std::invalid_argument("Layer must contain at least one neuron!");
    }

    shape_.push_back(n_neurons);

    return *this;
}

NeuralNetwork& NeuralNetwork::add_layer(size_t n_neurons) {
    return add_layer(n_neurons, ActivationFunction::ReLU);
}

NeuralNetwork& NeuralNetwork::add_layer(size_t n_neurons, ActivationFunction activation_function) {
    if (shape_.empty()) {
        throw std::logic_error("Input layer must be added first!");
    }

    if (built_) {
        throw std::logic_error("Cannot add layers after building the network!");
    }

    if (n_neurons == 0) {
        throw std::invalid_argument("Layer must contain at least one neuron!");
    }

    shape_.push_back(n_neurons);
    activation_functions_.push_back(activation_function);

    return *this;
}

void NeuralNetwork::randomize_weights_() {
    if (built_ == false) {
        throw std::logic_error("Cannot randomize weights of an unbuilt network!"); 
    }
    for (Matrix& weights : weights_) {
        weights.fill_random();
    }
}

Matrix NeuralNetwork::apply_activation_(const Matrix& Z, size_t layer) {
    switch (activation_functions_[layer]) {
        case ActivationFunction::ReLU:
            return Activation::relu(Z);
        case ActivationFunction::Tanh:
            return Activation::tanh(Z);
        case ActivationFunction::Sigmoid:
            return Activation::sigmoid(Z);
        case ActivationFunction::Softmax:
            return Activation::softmax(Z);
        default:
            throw std::logic_error("Unknown activation function!");
    }
}

Matrix NeuralNetwork::apply_loss_derivative_(const Matrix& target, const Matrix& pred, LossFunction loss) {
    switch (loss) {
        case LossFunction::MSE:
            return Loss::mse_derivative(target, pred);
        case LossFunction::MAE:
            return Loss::mae_derivative(target, pred);
        case LossFunction::BinaryCrossEntropy:
            return Loss::binary_cross_entropy_derivative(target, pred);
        case LossFunction::CategoricalCrossEntropy:
            return Loss::categorical_cross_entropy_derivative(target, pred);
        default:
            throw std::logic_error("Unknown loss function!");
    }
}

Matrix NeuralNetwork::activation_backward_(const Matrix& dA, const Matrix& Z, const Matrix& A, size_t layer) {
    switch (activation_functions_[layer]) {
        case ActivationFunction::ReLU:
            return dA.elementwise_mul(Activation::relu_derivative(Z));
        case ActivationFunction::Tanh:
            return dA.elementwise_mul(Activation::tanh_derivative(Z));
        case ActivationFunction::Sigmoid:
            return dA.elementwise_mul(Activation::sigmoid_derivative(Z));
        case ActivationFunction::Softmax:
            return softmax_backward_(dA, A);
        default:
            throw std::logic_error("Unknown activation function!");
    }
}

Matrix NeuralNetwork::softmax_backward_(const Matrix& dA, const Matrix& A) {
    Matrix dZ(A.get_rows(), A.get_cols());

    for (size_t col = 0; col < A.get_cols(); ++col) {
        double dot = 0.0;
        for (size_t row = 0; row < A.get_rows(); ++row) {
            dot += dA[row][col] * A[row][col];
        }
        for (size_t row = 0; row < A.get_rows(); ++row) {
            dZ[row][col] = A[row][col] * (dA[row][col] - dot);
        }
    }

    return dZ;
}
