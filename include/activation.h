#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"

namespace Activation {
    double relu(double x);
    Matrix relu(const Matrix& mat);

    double softmax(double x);
    Matrix softmax(const Matrix& mat);

    double tanh(double x);
    Matrix tanh(const Matrix& mat);

    double sigmoid(double x);
    Matrix sigmoid(const Matrix& mat);
}

#endif // ACTIVATION_H