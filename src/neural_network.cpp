#include "neural_network.h"
#include <stdexcept>

NeuralNetwork::NeuralNetwork()
    : n_layers_(0), network_shape_({}), activation_functions_({}),
      layer_weights_({}), task_type_(TaskType::Classification), initialized_(false), built_(false) {}

NeuralNetwork::NeuralNetwork(const vector<size_t>& network_shape) 
    : n_layers_(network_shape.size()), network_shape_(network_shape), activation_functions_({}), 
      layer_weights_({}), task_type_(TaskType::Classification), initialized_(false), built_(false) {}

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
    for (size_t layer = 0; layer < n_layers_ + 1; ++layer) {
        Matrix weights(network_shape_[layer] + 1, network_shape_[layer + 1]);
        weights.fill_random();
        layer_weights_.push_back(weights); 
    }
    built_ = true;
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

/* TODO */
Matrix NeuralNetwork::forward_(const Matrix& input) const {
/* It takes batch of column vectors on input.
   If one of the dimensions is equal to 1, then
   the matrix is converted to column vector
   if it is necessary.
*/

    return Matrix();
}

/* TODO */
void NeuralNetwork::backward_(const Matrix& input, const Matrix& target, double learning_rate) {

}