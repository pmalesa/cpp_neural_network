#include "neural_network.h"
#include <iostream>

using std::cout;

NeuralNetwork::NeuralNetwork(const vector<size_t>& network_shape) 
    : n_layers_(network_shape.size()), network_shape_(network_shape), activation_functions_({}), 
      layer_weights_({}), classification_(false), built_(false) {}

NeuralNetwork& NeuralNetwork::erase() {
    n_layers_ = 0;
    network_shape_.clear();
    activation_functions_.clear();
    layer_weights_.clear();
    classification_ = false;
    built_ = false;
    return *this;
}

NeuralNetwork& NeuralNetwork::init(size_t input_size, ActivationFunction activation_function) {
    n_layers_ = 1;
    network_shape_.clear();
    network_shape_.push_back(input_size);
    activation_functions_.clear();
    activation_functions_.push_back(activation_function);
    layer_weights_.clear();
    classification_ = false;
    built_ = false;
    return *this;
}