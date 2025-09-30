#include "activation.h"
#include <cmath>

namespace Activation {

double relu(double x) {
    return x > 0 ? x : 0;
}

Matrix relu(const Matrix& mat) {
    Matrix result(mat.get_rows(), mat.get_cols());
    for (size_t row = 0; row < result.get_rows(); ++row) {
        for (size_t col = 0; col < result.get_cols(); ++col) {
            result[row][col] = relu(mat[row][col]);
        }
    }
    return result;
}

double softmax(double x) {
    return 0.0;
}

Matrix softmax(const Matrix& mat) {
    return Matrix();
}

double tanh(double x) {
    return 0.0;
}

Matrix tanh(const Matrix& mat) {
    return Matrix();
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

Matrix sigmoid(const Matrix& mat) {
    Matrix result(mat.get_rows(), mat.get_cols());
    for (size_t row = 0; row < result.get_rows(); ++row) {
        for (size_t col = 0; col < result.get_cols(); ++col) {
            result[row][col] = sigmoid(mat[row][col]);
        }
    }
    return result;
}

} // namespace Activation