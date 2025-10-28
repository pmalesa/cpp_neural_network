#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

namespace Loss {
    double mse(const Matrix& y_true, const Matrix& y_pred);
    Matrix mse_derivative(const Matrix& y_true, const Matrix& y_pred);
    
    double mae(const Matrix& y_true, const Matrix& y_pred);
    Matrix mae_derivative(const Matrix& y_true, const Matrix& y_pred);

    double categorical_cross_entropy(const Matrix& y_true, const Matrix& y_pred);
    Matrix categorical_cross_entropy_derivative(const Matrix& y_true, const Matrix& y_pred);
    
}

#endif // LOSS_H



/*
TODO:
    - BCCE
    - CCE
    - BCCE unittest
    - CCE unittest

*/
