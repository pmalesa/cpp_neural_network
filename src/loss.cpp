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

}