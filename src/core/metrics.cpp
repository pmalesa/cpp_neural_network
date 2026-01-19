#include "metrics.h"

#include <cmath>

namespace Metrics {

double accuracy(const Matrix& y_pred, const Matrix& y_true) {
    const ConfusionMatrix cm = get_confusion_matrix_values(y_pred, y_true);
    const double nom = static_cast<double>(cm.tp + cm.tn);
    const double denom = static_cast<double>(cm.tp + cm.fp + cm.tn + cm.fn);
    if (denom < 1.0) {
        return 0.0;
    }
    return nom / denom;
}

double precision(const Matrix& y_pred, const Matrix& y_true) {
    const ConfusionMatrix cm = get_confusion_matrix_values(y_pred, y_true);
    const double denom = static_cast<double>(cm.tp + cm.fp);
    if (denom < 1.0) {
        return 0.0;
    }
    return static_cast<double>(cm.tp) / denom;
}

double recall(const Matrix& y_pred, const Matrix& y_true) {
    const ConfusionMatrix cm = get_confusion_matrix_values(y_pred, y_true);
    const double denom = static_cast<double>(cm.tp + cm.fn);
    if (denom < 1.0) {
        return 0.0;
    }
    return static_cast<double>(cm.tp) / denom;
}

// TODO
double f1(const Matrix& y_pred, const Matrix& y_true) {
    return 0.0;
}

ConfusionMatrix get_confusion_matrix_values(const Matrix& y_pred, const Matrix& y_true) {
    if (y_pred.get_rows() != y_true.get_rows() || y_pred.get_cols() != y_true.get_cols()) {
        throw std::logic_error("Matrices need to have the same sizes.");
    }

    const size_t n_rows = y_pred.get_rows();
    const size_t n_cols = y_pred.get_cols();
    ConfusionMatrix cm;

    if (n_rows == 0 || n_cols == 0) {
        return cm;
    }

    if (n_rows != 1 && n_cols != 1) {
        throw std::logic_error("Matrix objects need to be in a form of either column or row vectors (1 x N or N x 1).");
    }

    const size_t n = (n_rows == 1) ? n_cols : n_rows;
    auto doubles_equal = [](double a, double b, double tolerance = 1e-9) {
        return fabs(a - b) < tolerance;
    };

    for (size_t i = 0; i < n; ++i) {
        double t = (n_rows == 1) ? y_true[0][i] : y_true[i][0];
        double p = (n_rows == 1) ? y_pred[0][i] : y_pred[i][0];

        if (doubles_equal(p, 1.0) && doubles_equal(t, 1.0)) {
            ++cm.tp;
        } else if (doubles_equal(p, 1.0) && doubles_equal(t, 0.0)) {
            ++cm.fp;
        } else if (doubles_equal(p, 0.0) && doubles_equal(t, 1.0)) {
            ++cm.fn;
        } else {
            ++cm.tn;
        }
    }

    return cm;
}

}