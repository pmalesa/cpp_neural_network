#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include "matrix.h"

using std::vector;
using std::string;

enum class ActivationFunction { ReLU, Tanh, Softmax, Sigmoid };
enum class LossFunction { MSE, MAE, BinaryCrossEntropy, CategoricalCrossEntropy };
enum class TaskType { Classification, Regression };

class NeuralNetwork {
public:
    NeuralNetwork();
    NeuralNetwork(const vector<size_t>& network_shape, const vector<ActivationFunction>& activation_functions);
    ~NeuralNetwork() = default;

    NeuralNetwork& erase(); // Erases all layers, sizes and weights (state as after default constructor)
    NeuralNetwork& init(size_t input_size, ActivationFunction activation_function = ActivationFunction::ReLU);
    NeuralNetwork& add_layer(size_t n_neurons, ActivationFunction activation_function = ActivationFunction::ReLU);
    NeuralNetwork& build();
    NeuralNetwork& fit(const Matrix& X, const Matrix& y,
                       size_t epochs = 100, double learning_rate = 0.01,
                       LossFunction loss = LossFunction::BinaryCrossEntropy);

    Matrix predict(const Matrix& input) const;
    void save(const string& filename) const;
    void load(const string& filename);

    vector<size_t> get_shape() const { return network_shape_; }
    vector<ActivationFunction> get_activation_functions() const { return activation_functions_; }
    vector<Matrix> get_weights() const { return layer_weights_; }
    bool is_built() const { return built_; }
    
private:
    void randomize_weights_();
    Matrix forward_(const Matrix& input) const;
    void backward_(const Matrix& input, const Matrix& target, double learning_rate);

    size_t n_layers_;
    vector<size_t> network_shape_;
    vector<ActivationFunction> activation_functions_;
    vector<Matrix> layer_weights_;
    TaskType task_type_;
    bool initialized_;
    bool built_;
};

#endif // NEURAL_NETWORK_H

/*
    TODO:
    - DataSet representation 
      (CSV, maybe create some class to process CSV files, but it will be used by some trainer class, not the nn itself, nn uses Matrices)
    - forward()
    - backward()
    - fit()
    - predict()
    - save() and load()
*/