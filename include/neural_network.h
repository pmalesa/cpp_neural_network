#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "matrix.h"
#include "types.h"

using std::vector;
using types::ActivationFunction;

class NeuralNetwork {
public:
    NeuralNetwork();
    NeuralNetwork(const vector<size_t>& shape, const vector<ActivationFunction>& activation_functions);
    ~NeuralNetwork() = default;

    NeuralNetwork& erase(); // Erases all layers, sizes and weights (state as after default constructor)
    NeuralNetwork& build();

    void randomize_weights_();
    Matrix forward(const Matrix& input, bool learning = false);
    void backward(const Matrix& input, const Matrix& target, double learning_rate);
    bool is_built() const { return built_; }

    size_t n_layers;
    vector<size_t> shape;
    vector<ActivationFunction> activation_functions;
    vector<Matrix> weights;

private:
    bool built_;
    vector<Matrix> A_values_;
    vector<Matrix> Z_values_;
};


#endif // NEURAL_NETWORK_H

/*
    TODO:
     - backward():
        - Derivatives of activation functions
        - Store A values and Z values (post and pre activation values) in a private variable
*/