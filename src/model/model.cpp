#include "model.h"
#include "logger.h"
#include "activation.h"

#include <stdexcept>
#include <sstream>

static Logger& logger = Logger::instance();

Model::Model()
    : nn_(NeuralNetwork()), task_type_(TaskType::Classification), fit_(false) {
    logger.set_level(Logger::Level::Error);
}

Model::Model(const vector<size_t>& shape, const vector<ActivationFunction>& activation_functions) 
    : nn_(NeuralNetwork(shape, activation_functions)), task_type_(TaskType::Classification), fit_(false) {
/*
    Network layer's sizes including input and output layers
*/
    logger.set_level(Logger::Level::Error);
    if (shape.size() != activation_functions.size()) {
        nn_.n_layers = 0;
        nn_.shape.clear();
        nn_.activation_functions.clear();
        throw std::logic_error("Number of layers and number of activation functions cannot be different!");
    }
}

Model& Model::add_layer(size_t n_neurons, LayerType layer_type, ActivationFunction activation_function) {
    if (fit_) {
        throw std::logic_error("Cannot add a layer to the network after calling fit()!");
    }
    if (n_neurons < 1) {
        return *this;
    }
    ++nn_.n_layers;
    nn_.shape.push_back(n_neurons);
    if (layer_type != LayerType::Output) {
        nn_.activation_functions.push_back(activation_function);
    }
    return *this;
}

Model& Model::add_layer(size_t n_neurons) {
    return add_layer(n_neurons, LayerType::Hidden, ActivationFunction::ReLU);
}

Model& Model::add_layer(size_t n_neurons, LayerType layer_type) {
    return add_layer(n_neurons, layer_type, ActivationFunction::ReLU);
}

Model& Model::add_layer(size_t n_neurons, ActivationFunction activation_function) {
    return add_layer(n_neurons, LayerType::Hidden, activation_function);
}

Model& Model::fit(const Matrix& X, const Matrix& y, size_t epochs, double learning_rate, LossFunction loss) {
    logger.log("Model training started.", Logger::Level::Info);
    fit_ = true;
    if (!nn_.is_built()) {
        nn_.build();
    }
    // calls forward() and backward()

    logger.log("Model training finished.", Logger::Level::Info);
    return *this;
}

Matrix Model::predict(const Matrix& input) const {
    // calls forward()
    return {};
}

Model& Model::clear() {
    fit_ = false;
    nn_.erase();
    return *this;
}

void Model::save(const string& filename) const {
    // ...

    std::ostringstream log_msg_oss;
    log_msg_oss << "Model saved to: '" << filename << "'.";
    logger.log(log_msg_oss.str(), Logger::Level::Info);
}

void Model::load(const string& filename) {
    // ...

    std::ostringstream log_msg_oss;
    log_msg_oss << "Model loaded from: '" << filename << "'.";
    logger.log(log_msg_oss.str(), Logger::Level::Info);
}