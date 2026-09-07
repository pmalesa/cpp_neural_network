#include "model.h"
#include "spdlog/spdlog.h"

#include <stdexcept>
#include <sstream>

Model::Model()
    : nn_(NeuralNetwork()), task_type_(TaskType::Classification), fit_(false) {}

Model::Model(const vector<size_t>& shape, const vector<ActivationFunction>& activation_functions) 
    : nn_(NeuralNetwork(shape, activation_functions)), task_type_(TaskType::Classification), fit_(false) {}

Model& Model::add_input_layer(size_t n_neurons) {
    nn_.add_input_layer(n_neurons);
    return *this;
}

Model& Model::add_layer(size_t n_neurons) {
    return add_layer(n_neurons, ActivationFunction::ReLU);
}

Model& Model::add_layer(size_t n_neurons, ActivationFunction activation_function) {
    nn_.add_layer(n_neurons, activation_function);
    return *this;
}

Model& Model::fit(const Matrix& X, const Matrix& y, size_t epochs, double learning_rate, LossFunction loss) {
    const vector<size_t> network_shape = nn_.get_shape();

    if (network_shape.size() < 2) {
        throw std::logic_error("Model requires an input and output layer before training!");
    }

    if (X.is_empty()) {
        throw std::invalid_argument("Input data can not be empty!");
    }

    if (y.is_empty()) {
        throw std::invalid_argument("Target data can not be empty!");
    }

    if (X.get_cols() != y.get_cols()) {
        throw std::invalid_argument("Number of examples in input and target data must be equal!");
    }

    if (epochs == 0) {
        throw std::invalid_argument("Number of epochs must be greater than zero!");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive!");
    }

    if (X.get_rows() != network_shape.front()) {
        throw std::invalid_argument("Input feature count does not match the model input layer!");
    }

    if (y.get_rows() != network_shape.back()) {
        throw std::invalid_argument("Target count does not match the model output layer!");
    }

    spdlog::info("Model training started.");
    if (!nn_.is_built()) {
        nn_.build();
    }
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        Matrix prediction = nn_.forward(X, true);
        double epoch_loss = nn_.get_loss(y, prediction, loss);
        nn_.backward(X, y, learning_rate, loss);

        std::ostringstream log_msg_oss;
        log_msg_oss << "[EPOCH " << epoch + 1 << "] Loss: " << epoch_loss;
        spdlog::info(log_msg_oss.str());
    }

    fit_ = true;
    spdlog::info("Model training finished.");
    return *this;
}

Matrix Model::predict(const Matrix& input) {
    if (!fit_) {
        throw std::logic_error("Model must be fitted before prediction!");
    }

    return nn_.forward(input, false);
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
    spdlog::info(log_msg_oss.str());
}

void Model::load(const string& filename) {
    // ...

    std::ostringstream log_msg_oss;
    log_msg_oss << "Model loaded from: '" << filename << "'.";
    spdlog::info(log_msg_oss.str());
}