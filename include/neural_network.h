#include <vector>
#include <string>
#include "matrix.h"

using std::vector;
using std::string;

enum class ActivationFunction { ReLU, Tanh, Softmax };
enum class LossFunction { MSE, CrossEntropy };

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    NeuralNetwork(const vector<size_t>& network_shape); // Network layer's sizes including input and output layers
    ~NeuralNetwork() = default;

    NeuralNetwork& erase(); // Erases all layers, sizes and weights (state as after default constructor)
    NeuralNetwork& init(size_t input_size, ActivationFunction activation_function = ActivationFunction::ReLU); // Calls erase()
    NeuralNetwork& add_layer(size_t n_neurons, ActivationFunction activation_function = ActivationFunction::ReLU, bool output = false);
    NeuralNetwork& build();
    NeuralNetwork& fit(const Matrix& X, const Matrix& y,
                       size_t epochs = 100, double learning_rate = 0.01,
                       LossFunction loss = LossFunction::CrossEntropy);

    Matrix predict(const Matrix& input) const;
    void save(const string& filename) const;
    void load(const string& filename);
    
private:
    void randomize_weights_();
    Matrix forward_(const Matrix& input) const;
    void backward_(const Matrix& input, const Matrix& target, double learning_rate);

    size_t n_layers_;
    vector<size_t> network_shape_;
    vector<ActivationFunction> activation_functions_;
    vector<Matrix> layer_weights_;
    bool classification_;
    bool built_;
};
