#include "neural_network.h"
#include "logger.h"
#include "activation.h"

#include <stdexcept>
#include <sstream>

static Logger& logger = Logger::instance();

NeuralNetwork::NeuralNetwork()
    : n_layers(0), shape({}), activation_functions({}),
      weights({}), built_(false) {
    logger.set_level(Logger::Level::Error);
}

NeuralNetwork::NeuralNetwork(const vector<size_t>& new_shape, const vector<ActivationFunction>& new_activation_functions) 
    : n_layers(new_shape.size()), shape(new_shape), activation_functions(new_activation_functions), 
      weights({}), built_(false) {
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
    built_ = false;
    return *this;
}

NeuralNetwork& NeuralNetwork::build() {
    if (n_layers < 2) {
        throw std::logic_error("Cannot build a network with fewer than 2 layers!"); 
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

Matrix NeuralNetwork::forward(const Matrix& input) const {
/*
It takes batch of column vectors on input.
*/
    if (!built_) {
        throw std::logic_error("Network must be built before forward pass!");
    }

    Matrix X = input;
    for (size_t layer = 0; layer < weights.size(); ++layer) {
        Matrix X_bias = X.add_bias_row();
        const Matrix& W = weights[layer];
        Matrix Z = W.transpose() * X_bias;

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

    return X;
}

/* TODO */
void NeuralNetwork::backward(const Matrix& input, const Matrix& target, double learning_rate) {

}