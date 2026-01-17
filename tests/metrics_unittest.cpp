#include "matrix.h"
#include "metrics.h"
#include <gtest/gtest.h>

#include <iostream>

using il = std::initializer_list<std::initializer_list<double>>;

class LossTest : public testing::Test {
public:
    LossTest() {}

protected:
    bool double_values_equal_(double lhs, double rhs, double tolerance = 1e-9) {
        return abs(lhs - rhs) < tolerance;
    }
};

TEST_F(LossTest, GetConfusionMatrixValuesTest) {

}

TEST_F(LossTest, BinaryClassificationPrecisionRowVectorTest) {
    // Incorrect sizes
    Matrix pred_incorrect = { {1.0, 1.0, 0.0} };
    Matrix gt_incorrect = { {1.0, 1.0} };
    EXPECT_THROW(Metrics::precision(pred_incorrect, gt_incorrect), std::logic_error);

    pred_incorrect = { {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0} };
    gt_incorrect = { {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0} };
    EXPECT_THROW(Metrics::precision(pred_incorrect, gt_incorrect), std::logic_error);

    // Row vector
    Matrix pred = { {1.0, 0.0, 1.0, 0.0, 0.0, 1.0 } };
    Matrix gt = { {1.0, 0.0, 1.0, 1.0, 0.0, 0.0} };
    double result = 2.0 / 3.0;
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), result));

    // Column vector
    pred = { {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0} };
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), result));
}

TEST_F(LossTest, RecallTest) {

}

TEST_F(LossTest, F1Test) {

}
