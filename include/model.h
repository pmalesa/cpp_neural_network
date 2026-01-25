#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "neural_network.h"
#include "matrix.h"
#include "types.h"

using std::vector;
using std::string;
using types::ActivationFunction;
using types::LayerType;
using types::LossFunction;
using types::TaskType;

class Model {
public:
    Model();
    Model(const vector<size_t>& shape, const vector<ActivationFunction>& activation_functions);
    ~Model() = default;

    Model& add_layer(
        size_t n_neurons, 
        LayerType layer_type, 
        ActivationFunction activation_function); 
    // If layer_type is equal to LayerType::Output, then activation_function parameter can be ommitted (it will not be used)

    // Overloads for "Python-like" parameter passing
    Model& add_layer(size_t n_neurons);
    Model& add_layer(size_t n_neurons, LayerType layer_type);
    Model& add_layer(size_t n_neurons, ActivationFunction activation_function);

    Model& fit(const Matrix& X, const Matrix& y,
                       size_t epochs = 100, double learning_rate = 0.01,
                       LossFunction loss = LossFunction::BinaryCrossEntropy);

    Matrix predict(const Matrix& input) const;
    Model& clear();
    void save(const string& filename) const;
    void load(const string& filename);
    vector<size_t> shape() const { return nn_.get_shape(); }

private:
    NeuralNetwork nn_;
    TaskType task_type_;
    bool fit_;
};

#endif // MODEL_H
