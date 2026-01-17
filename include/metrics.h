#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"
#include <vector>

using std::vector;

namespace Metrics {
    struct ConfusionMatrix {
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;
    };

    double accuracy(const Matrix& y_pred, const Matrix& y_true);
    double precision(const Matrix& y_pred, const Matrix& y_true);
    double recall(const Matrix& y_pred, const Matrix& y_true);
    double f1(const Matrix& y_pred, const Matrix& y_true);
    ConfusionMatrix get_confusion_matrix_values(const Matrix& y_pred, const Matrix& y_true);
}

#endif // METRICS_H
