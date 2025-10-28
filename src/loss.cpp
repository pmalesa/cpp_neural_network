#include "loss.h"
#include <cmath>

namespace Loss {

double mse(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.get_rows() != y_pred.get_rows() || y_true.get_cols() != y_pred.get_cols()) {
        throw std::invalid_argument("Shape mismatch in MSE calculation!");
    }

    long double sum = 0.0L;
    double count = static_cast<double>(y_true.get_rows() * y_true.get_cols());

    for (size_t row = 0; row < y_true.get_rows(); ++row) {
        for (size_t col = 0; col < y_true.get_cols(); ++col) {
            double diff = y_true[row][col] - y_pred[row][col];
            sum += static_cast<long double>(diff) * diff;
        }
    }

    return static_cast<double>(sum / count);
}

Matrix mse_derivative(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.get_rows() != y_pred.get_rows() || y_true.get_cols() != y_pred.get_cols()) {
        throw std::invalid_argument("Shape mismatch in MSE derivative calculation!");
    }

    Matrix grads(y_true.get_rows(), y_true.get_cols());
    double scale = 2.0 / static_cast<double>(y_true.get_rows() * y_true.get_cols());

    for (size_t row = 0; row < y_true.get_rows(); ++row) {
        for (size_t col = 0; col < y_true.get_cols(); ++col) {
            grads[row][col] = scale * (y_pred[row][col] - y_true[row][col]);
        }
    }

    return grads;
}

double mae(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.get_rows() != y_pred.get_rows() || y_true.get_cols() != y_pred.get_cols()) {
        throw std::invalid_argument("Shape mismatch in MSE calculation!");
    }

    long double sum = 0.0L;
    double count = static_cast<double>(y_true.get_rows() * y_true.get_cols());

    for (size_t row = 0; row < y_true.get_rows(); ++row) {
        for (size_t col = 0; col < y_true.get_cols(); ++col) {
            sum += static_cast<long double>(std::abs(y_true[row][col] - y_pred[row][col]));
        }
    }

    return static_cast<double>(sum / count);
}

Matrix mae_derivative(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.get_rows() != y_pred.get_rows() || y_true.get_cols() != y_pred.get_cols()) {
        throw std::invalid_argument("Shape mismatch in MSE derivative calculation!");
    }

    Matrix grads(y_true.get_rows(), y_true.get_cols());
    double scale = 1.0 / static_cast<double>(y_true.get_rows() * y_true.get_cols());

    for (size_t row = 0; row < y_true.get_rows(); ++row) {
        for (size_t col = 0; col < y_true.get_cols(); ++col) {
            double diff = y_pred[row][col] - y_true[row][col];
            grads[row][col] = scale * ((diff > 0) - (diff < 0));    // sign(diff)
        }
    }

    return grads;
}

double categorical_cross_entropy(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.get_rows() != y_pred.get_rows() ||
        y_true.get_cols() != y_pred.get_cols()) {
        throw std::invalid_argument("Shape mismatch in Categorical Cross Entropy calculation!");
    }

    long double total_loss = 0.0;
    const size_t n_examples = y_true.get_cols();
    const size_t n_classes = y_true.get_rows();

    for (size_t i = 0; i < n_examples; ++i) {
        for (size_t k = 0; k < n_classes; ++k) {
            double y_t = y_true[k][i];
            double y_p = std::max(y_pred[k][i], 1e-15);
            total_loss += static_cast<long double>(-y_t * std::log(y_p));
        }
    }

    return static_cast<double>(total_loss / n_examples);
}

Matrix categorical_cross_entropy_derivative(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.get_rows() != y_pred.get_rows() ||
        y_true.get_cols() != y_pred.get_cols()) {
        throw std::invalid_argument("Shape mismatch in Categorical Cross Entropy derivative calculation!");
    }

    const double scale = 1.0 / static_cast<double>(y_true.get_cols());
    Matrix grads(y_true.get_rows(), y_true.get_cols());
    for (size_t row = 0; row < grads.get_rows(); ++row) {
        for (size_t col = 0; col < grads.get_cols(); ++col) {
            grads[row][col] = scale * (y_pred[row][col] - y_true[row][col]);
        }
    }
    return grads;
}

}