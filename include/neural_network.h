#include <vector>
#include "matrix.h"

using std::vector;

enum class ActivationFunction { ReLU, Tanh, Softmax };

class NeuralNetwork {
public:
    NeuralNetwork();
    NeuralNetwork(const vector<size_t>& network_shape); // Network layer's sizes including input and output layers
    ~NeuralNetwork();

    NeuralNetwork& erase(); // Erases all layers, sizes and weights (state as after default constructor)
    NeuralNetwork& init(size_t input_size, ActivationFunction activation_function = ActivationFunction::ReLU); // Calls erase()
    NeuralNetwork& add_layer(size_t n_neurons, ActivationFunction activation_function = ActivationFunction::ReLU, bool output = false);
    NeuralNetwork& build();

private:
    void randomize_weights_();

    size_t n_layers_;
    vector<size_t> newtork_shape_;
    vector<ActivationFunction> activation_functions_;
    vector<Matrix> layer_weights_;
};
