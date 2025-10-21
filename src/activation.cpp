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

double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

Matrix relu_derivative(const Matrix& mat) {
    Matrix result(mat.get_rows(), mat.get_cols());
    for (size_t row = 0; row < result.get_rows(); ++row) {
        for (size_t col = 0; col < result.get_cols(); ++col) {
            result[row][col] = mat[row][col] > 0.0 ? 1.0 : 0.0;
        }
    }
    return result;
}

Matrix softmax(const Matrix& mat) {
    if (mat.get_rows() == 0 || mat.get_cols() == 0) {
        throw std::domain_error("Softmax input cannot be empty!");
    }

    Matrix result(mat.get_rows(), mat.get_cols());

    // Single vector case (1xN or Nx1)
    if (mat.get_rows() == 1 || mat.get_cols() == 1) {
        
        // Find maximal value
        double max_val = mat[0][0];
        for (size_t row = 0; row < mat.get_rows(); ++row) {
            for (size_t col = 0; col < mat.get_cols(); ++col) {
                max_val = std::max(max_val, mat[row][col]);
            }
        }

        double sum = 0.0;
        for (size_t row = 0; row < mat.get_rows(); ++row) {
            for (size_t col = 0; col < mat.get_cols(); ++col) {
                // Subtract max_val to prevent numerical overflow (it makes all exponents <= 0)
                // Softmax depends only on relative differences, not on absolute values
                result[row][col] = std::exp(mat[row][col] - max_val);
                sum += result[row][col];
            }
        }

        for (size_t row = 0; row < mat.get_rows(); ++row) {
            for (size_t col = 0; col < mat.get_cols(); ++col) {
                result[row][col] /= sum;
            }
        }
    } else { // Batch case (apply column-wise softmax)
        for (size_t col = 0; col < mat.get_cols(); ++col) {
            double max_val = mat[0][col];
            for (size_t row = 1; row < mat.get_rows(); ++row) {
                max_val = std::max(max_val, mat[row][col]);
            }

            double sum = 0.0;
            for (size_t row = 0; row < mat.get_rows(); ++row) {
                // Subtract max_val to prevent numerical overflow (it makes all exponents <= 0)
                // Softmax depends only on relative differences, not on absolute values
                result[row][col] = std::exp(mat[row][col] - max_val);
                sum += result[row][col];
            }

            for (size_t row = 0; row < mat.get_rows(); ++row) {
                result[row][col] /= sum;
            }
        }
    }

    return result;
}

double tanh(double x) {
    return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
}

Matrix tanh(const Matrix& mat) {
    Matrix result(mat.get_rows(), mat.get_cols());
    for (size_t row = 0; row < result.get_rows(); ++row) {
        for (size_t col = 0; col < result.get_cols(); ++col) {
            result[row][col] = tanh(mat[row][col]);
        }
    }
    return result;
}

// double tanh_derivative(double x) {

// }

// Matrix tanh_derivative(const Matrix& mat) {

// }

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

// double sigmoid_derivative(double x) {

// }

// Matrix sigmoid_derivative(const Matrix& mat) {

// }

} // namespace Activation
