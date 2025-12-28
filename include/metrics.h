#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"

namespace Metrics {
    double precision(const Matrix& x, const Matrix& y);
    double recall(const Matrix& x, const Matrix& y);
    double f1(const Matrix& x, const Matrix& y);
}

#endif // METRICS_H
