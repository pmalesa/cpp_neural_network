#include "neural_network.h"
#include "logger.h"
#include "activation.h"
#include "loss.h"

#include <stdexcept>
#include <sstream>

static Logger& logger = Logger::instance();

NeuralNetwork::NeuralNetwork()
    : n_layers(0), shape({}), activation_functions({}),
      weights({}), built_(false), A_values_({}), Z_values_({}) {
    logger.set_level(Logger::Level::Error);
}

NeuralNetwork::NeuralNetwork(const vector<size_t>& new_shape, const vector<ActivationFunction>& new_activation_functions) 
    : n_layers(new_shape.size()), shape(new_shape), activation_functions(new_activation_functions), 
      weights({}), built_(false), A_values_({}), Z_values_({}) {
/*
    Network layer's sizes including input and output layers
*/
    logger.set_level(Logger::Level::Error);
    if (new_shape.size() != new_activation_functions.size()) {
        n_layers = 0;
        shape.clear();
        activation_functions.clear();
        throw std::logic_error("Number of layers and number of activation functions cannot be different!");
    }
}

NeuralNetwork& NeuralNetwork::erase() {
    n_layers = 0;
    shape.clear();
    activation_functions.clear();
    weights.clear();
    A_values_.clear();
    Z_values_.clear();
    built_ = false;
    return *this;
}

NeuralNetwork& NeuralNetwork::build() {
    logger.log("Building neural network...", Logger::Level::Info);
    if (n_layers < 2) {
        string err_msg = "Cannot build a network with fewer than 2 layers!.";
        logger.log(err_msg, Logger::Level::Error);
        throw std::logic_error(err_msg); 
    }
    for (size_t layer = 0; layer < n_layers - 1; ++layer) {
        Matrix layer_weights(shape[layer] + 1, shape[layer + 1]);
        layer_weights.fill_random();
        weights.push_back(layer_weights); 
    }
    built_ = true;

    logger.log("Neural network built successfully.", Logger::Level::Info);
    std::ostringstream log_msg_oss;
    log_msg_oss << "Number of hidden layers: " << (n_layers > 2 ? n_layers - 2 : 0) << " | Network structure: [";
    for (size_t layer = 0; layer < shape.size(); ++layer) {
        log_msg_oss << shape[layer];
        if (layer < shape.size() - 1) {
            log_msg_oss << ", ";
        }
    }
    log_msg_oss << "]";
    logger.log(log_msg_oss.str(), Logger::Level::Info);

    return *this;
}

void NeuralNetwork::randomize_weights_() {
    if (built_ == false) {
        throw std::logic_error("Cannot randomize weights of an unbuilt network!"); 
    }
    for (Matrix& weights : weights) {
        weights.fill_random();
    }
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

    for (size_t layer = 0; layer < weights.size(); ++layer) {
        Matrix X_bias = X.add_bias_row();
        const Matrix& W = weights[layer];
        Matrix Z = W.transpose() * X_bias;
        if (learning) {
            Z_values_.push_back(Z);
        }
        switch (activation_functions[layer]) {
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