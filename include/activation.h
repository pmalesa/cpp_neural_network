#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"

namespace Activation {
    double relu(double x);
    Matrix relu(const Matrix& mat);
    double relu_derivative(double x);
    Matrix relu_derivative(const Matrix& mat);

    Matrix softmax(const Matrix& mat);

    double tanh(double x);
    Matrix tanh(const Matrix& mat);
    double tanh_derivative(double x);
    Matrix tanh_derivative(const Matrix& mat);

    double sigmoid(double x);
    Matrix sigmoid(const Matrix& mat);
    double sigmoid_derivative(double x);
    Matrix sigmoid_derivative(const Matrix& mat);
}

#endif // ACTIVATION_H