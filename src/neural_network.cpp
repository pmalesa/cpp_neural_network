#include "neural_network.h"
#include "logger.h"
#include "activation.h"

#include <stdexcept>
#include <sstream>

static Logger& logger = Logger::instance();

NeuralNetwork::NeuralNetwork()
    : n_layers_(0), network_shape_({}), activation_functions_({}),
      layer_weights_({}), task_type_(TaskType::Classification), initialized_(false), built_(false) {
    logger.set_level(Logger::Level::Error);
}

NeuralNetwork::NeuralNetwork(const vector<size_t>& network_shape, const vector<ActivationFunction>& activation_functions) 
    : n_layers_(network_shape.size()), network_shape_(network_shape), activation_functions_(activation_functions), 
      layer_weights_({}), task_type_(TaskType::Classification), initialized_(false), built_(false) {
/*
    Network layer's sizes including input and output layers
*/
    logger.set_level(Logger::Level::Error);
    if (network_shape.size() != activation_functions.size()) {
        n_layers_ = 0;
        network_shape_.clear();
        activation_functions_.clear();
        throw std::logic_error("Number of layers and number of activation functions cannot be different!");
    }
}

NeuralNetwork& NeuralNetwork::erase() {
    n_layers_ = 0;
    network_shape_.clear();
    activation_functions_.clear();
    layer_weights_.clear();
    task_type_ = TaskType::Classification;
    initialized_ = false;
    built_ = false;
    return *this;
}

NeuralNetwork& NeuralNetwork::init(size_t input_size, ActivationFunction activation_function) {
    erase();
    n_layers_ = 1;
    network_shape_.push_back(input_size);
    activation_functions_.push_back(activation_function);
    initialized_ = true;
    logger.log("Neural network initialized successfully.", Logger::Level::Info);
    return *this;
}

NeuralNetwork& NeuralNetwork::add_layer(size_t n_neurons, ActivationFunction activation_function) {
    if (built_) {
        throw std::logic_error("Cannot add a layer to an already built network!");
    }
    if (initialized_ == false) {
        throw std::logic_error("Cannot add a layer to an unitialized network!");
    }
    if (n_neurons < 1) {
        return *this;
    }
    ++n_layers_;
    network_shape_.push_back(n_neurons);
    activation_functions_.push_back(activation_function);
    return *this;
}

NeuralNetwork& NeuralNetwork::build() {
    if (initialized_ == false) {
        throw std::logic_error("Cannot build an unitialized network!"); 
    }
    if (n_layers_ < 2) {
        throw std::logic_error("Cannot build a network with fewer than 2 layers!"); 
    }
    for (size_t layer = 0; layer < n_layers_ - 1; ++layer) {
        Matrix weights(network_shape_[layer] + 1, network_shape_[layer + 1]);
        weights.fill_random();
        layer_weights_.push_back(weights); 
    }
    built_ = true;

    logger.log("Neural network built successfully.", Logger::Level::Info);
    std::ostringstream log_msg_oss;
    log_msg_oss << "Number of hidden layers: " << (n_layers_ > 2 ? n_layers_ - 2 : 0) << " | Network structure: [";
    for (size_t layer = 0; layer < network_shape_.size(); ++layer) {
        log_msg_oss << network_shape_[layer];
        if (layer < network_shape_.size() - 1) {
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
    for (Matrix& weights : layer_weights_) {
        weights.fill_random();
    }
}

Matrix NeuralNetwork::forward_(const Matrix& input) const {
/*
It takes batch of column vectors on input.
*/
    if (!built_) {
        throw std::logic_error("Network must be built before forward pass!");
    }

    Matrix X = input;
    for (size_t layer = 0; layer < layer_weights_.size(); ++layer) {
        Matrix X_bias = X.add_bias_row();
        const Matrix& W = layer_weights_[layer];
        Matrix Z = W.transpose() * X_bias;

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

    return X;
}

/* TODO */
void NeuralNetwork::backward_(const Matrix& input, const Matrix& target, double learning_rate) {

}