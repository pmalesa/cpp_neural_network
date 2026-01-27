#include "neural_network.h"
#include "spdlog/spdlog.h"
#include "activation.h"
#include "loss.h"

#include <stdexcept>
#include <sstream>

using std::string;

NeuralNetwork::NeuralNetwork()
    : n_layers_(0), shape_({}), activation_functions_({}),
      weights_({}), built_(false), A_values_({}), Z_values_({}) {}

NeuralNetwork::NeuralNetwork(const vector<size_t>& new_shape, const vector<ActivationFunction>& new_activation_functions) 
    : n_layers_(new_shape.size()), shape_(new_shape), activation_functions_(new_activation_functions), 
      weights_({}), built_(false), A_values_({}), Z_values_({}) {
/*
    Network layer's sizes including input and output layers
*/
    if (new_shape.size() != new_activation_functions.size()) {
        n_layers_ = 0;
        shape_.clear();
        activation_functions_.clear();
        throw std::logic_error("Number of layers and number of activation functions cannot be different!");
    }
}

NeuralNetwork& NeuralNetwork::erase() {
    n_layers_ = 0;
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
    if (n_layers_ < 2) {
        string err_msg = "Cannot build a network with fewer than 2 layers!.";
        spdlog::error(err_msg);
        throw std::logic_error(err_msg); 
    }
    for (size_t layer = 0; layer < n_layers_ - 1; ++layer) {
        Matrix layer_weights(shape_[layer] + 1, shape_[layer + 1]);
        layer_weights.fill_random();
        weights_.push_back(layer_weights); 
    }
    built_ = true;

    spdlog::info("Neural network built successfully.");
    std::ostringstream log_msg_oss;
    log_msg_oss << "Number of hidden layers: " << (n_layers_ > 2 ? n_layers_ - 2 : 0) << " | Network structure: [";
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

    if (learning) {
        Z_values_.clear();
        A_values_.clear();
    }

    Matrix X = input;

    for (size_t layer = 0; layer < weights_.size(); ++layer) {
        Matrix X_bias = X.add_bias_row();
        const Matrix& W = weights_[layer];
        Matrix Z = W.transpose() * X_bias;
        if (learning) {
            Z_values_.push_back(Z);
        }
        switch (activation_functions_[layer]) {
            case ActivationFunction::ReLU:
                X = Activation::relu(Z);
                break;
            case ActivationFunction::Tanh:
                X = Activation::tanh(Z);
                break;
            case ActivationFunction::Sigmoid:
                X = Activation::sigmoid(Z);
                break;
            case ActivationFunction::Softmax:
                X = Activation::softmax(Z);
                break;
            default:
                throw std::logic_error("Unknown activation function!");
        }
    }

    if (learning) {
        A_values_.push_back(X);
    }

    return X;
}

/* TODO */
void NeuralNetwork::backward(const Matrix& input, const Matrix& target, double learning_rate) {
    if (A_values_.empty() || Z_values_.empty()) {
        throw std::logic_error("Can not perform backward pass before the forward pass!");
    }

    // const size_t L = weights.size();

    // 1) Compute loss derivative (dA[L])
    Matrix dA = Loss::mse_derivative(target, A_values_.back());

    // ...

}

NeuralNetwork& NeuralNetwork::add_layer(size_t n_neurons, LayerType layer_type, ActivationFunction activation_function) {
    ++n_layers_;
    shape_.push_back(n_neurons);
    if (layer_type != LayerType::Output) {
        activation_functions_.push_back(activation_function);
    }
    return *this;
}

NeuralNetwork& NeuralNetwork::add_layer(size_t n_neurons) {
    return add_layer(n_neurons, LayerType::Hidden, ActivationFunction::ReLU);
}

NeuralNetwork& NeuralNetwork::add_layer(size_t n_neurons, LayerType layer_type) {
    return add_layer(n_neurons, layer_type, ActivationFunction::ReLU);
}

NeuralNetwork& NeuralNetwork::add_layer(size_t n_neurons, ActivationFunction activation_function) {
    return add_layer(n_neurons, LayerType::Hidden, activation_function);
}

void NeuralNetwork::randomize_weights_() {
    if (built_ == false) {
        throw std::logic_error("Cannot randomize weights of an unbuilt network!"); 
    }
    for (Matrix& weights : weights_) {
        weights.fill_random();
    }
}