#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "matrix.h"
#include "types.h"

using std::vector;
using types::ActivationFunction;
using types::LayerType;

class NeuralNetwork {
public:
    NeuralNetwork();
    NeuralNetwork(const vector<size_t>& shape, const vector<ActivationFunction>& activation_functions);
    ~NeuralNetwork() = default;

    NeuralNetwork& erase(); // Erases all layers, sizes and weights (state as after default constructor)
    NeuralNetwork& build();

    Matrix forward(const Matrix& input, bool learning = false);
    void backward(const Matrix& input, const Matrix& target, double learning_rate);
    bool is_built() const { return built_; }
    
    NeuralNetwork& add_layer(size_t n_neurons, LayerType layer_type, ActivationFunction activation_function);
    // If layer_type is equal to LayerType::Output, then activation_function parameter can be ommitted (it will not be used)
    
    // Overloads for "Python-like" parameter passing
    NeuralNetwork& add_layer(size_t n_neurons);
    NeuralNetwork& add_layer(size_t n_neurons, LayerType layer_type);
    NeuralNetwork& add_layer(size_t n_neurons, ActivationFunction activation_function);
    
    size_t get_n_layers() const { return n_layers_; }
    vector<size_t> get_shape() const { return shape_; }
    vector<ActivationFunction>& get_activation_functions() { return activation_functions_; }
    const vector<ActivationFunction>& get_activation_functions() const { return activation_functions_; }
    vector<Matrix>& get_weights() { return weights_; }
    const vector<Matrix>& get_weights() const { return weights_; }
    const vector<Matrix>& get_activations() const { return A_values_; }
    const vector<Matrix>& get_pre_activations() const { return Z_values_; }
    
private:
    void randomize_weights_();
    
    size_t n_layers_;
    vector<size_t> shape_;
    vector<ActivationFunction> activation_functions_;
    vector<Matrix> weights_;
    bool built_;
    vector<Matrix> A_values_;
    vector<Matrix> Z_values_;
};

#endif // NEURAL_NETWORK_H
