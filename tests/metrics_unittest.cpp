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

// TEST_F(LossTest, GetConfusionMatrixValuesTest) {

// }

TEST_F(LossTest, BinaryClassificationAccuracyTest) {
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
    double result = 4.0 / 6.0;
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), result));

    // Column vector
    pred = { {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0} };
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), result));

    // No true positives and no false positives (denominator == 0.0)
    pred = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0} };
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), 0.0));

    // Empty vectors (row/column matrices)
    pred = { };
    gt = { };
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), 0.0));
    pred = { {} };
    gt = { {} };
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), 0.0));    
}

TEST_F(LossTest, BinaryClassificationPrecisionTest) {
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

    // No true positives and no false positives (denominator == 0.0)
    pred = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0} };
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), 0.0));

    // Empty vectors (row/column matrices)
    pred = { };
    gt = { };
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), 0.0));
    pred = { {} };
    gt = { {} };
    EXPECT_TRUE(double_values_equal_(Metrics::precision(pred, gt), 0.0));
}

TEST_F(LossTest, RecallTest) {
    // Incorrect sizes
    Matrix pred_incorrect = { {1.0, 1.0, 0.0} };
    Matrix gt_incorrect = { {1.0, 1.0} };
    EXPECT_THROW(Metrics::recall(pred_incorrect, gt_incorrect), std::logic_error);

    pred_incorrect = { {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0} };
    gt_incorrect = { {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0} };
    EXPECT_THROW(Metrics::recall(pred_incorrect, gt_incorrect), std::logic_error);

    // Row vector
    Matrix pred = { {1.0, 0.0, 1.0, 0.0, 0.0, 1.0 } };
    Matrix gt = { {1.0, 0.0, 1.0, 1.0, 0.0, 0.0} };
    double result = 2.0 / 3.0;
    EXPECT_TRUE(double_values_equal_(Metrics::recall(pred, gt), result));

    // Column vector
    pred = { {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0} };
    EXPECT_TRUE(double_values_equal_(Metrics::recall(pred, gt), result));

    // No true positives and no false negatives (denominator == 0.0)
    pred = { {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {0.0} };
    gt = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    EXPECT_TRUE(double_values_equal_(Metrics::recall(pred, gt), 0.0));

    // Empty vectors (row/column matrices)
    pred = { };
    gt = { };
    EXPECT_TRUE(double_values_equal_(Metrics::recall(pred, gt), 0.0));
    pred = { {} };
    gt = { {} };
    EXPECT_TRUE(double_values_equal_(Metrics::recall(pred, gt), 0.0));
}

TEST_F(LossTest, F1Test) {
    // Incorrect sizes
    Matrix pred_incorrect = { {1.0, 1.0, 0.0} };
    Matrix gt_incorrect = { {1.0, 1.0} };
    EXPECT_THROW(Metrics::f1(pred_incorrect, gt_incorrect), std::logic_error);

    pred_incorrect = { {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0} };
    gt_incorrect = { {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0} };
    EXPECT_THROW(Metrics::f1(pred_incorrect, gt_incorrect), std::logic_error);

    // Row vector
    Matrix pred = { {1.0, 0.0, 1.0, 0.0, 0.0, 1.0 } };
    Matrix gt = { {1.0, 0.0, 1.0, 1.0, 0.0, 0.0} };
    double result = (2.0 * (2.0 / 3.0) * (2.0 / 3.0)) / ((2.0 / 3.0) + (2.0 / 3.0));
    EXPECT_TRUE(double_values_equal_(Metrics::f1(pred, gt), result));

    // Column vector
    pred = { {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0} };
    EXPECT_TRUE(double_values_equal_(Metrics::f1(pred, gt), result));

    // No true positives, no false positives and no false negatives (denominator == 0.0)
    pred = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    gt = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    EXPECT_TRUE(double_values_equal_(Metrics::f1(pred, gt), 0.0));

    // Empty vectors (row/column matrices)
    pred = { };
    gt = { };
    EXPECT_TRUE(double_values_equal_(Metrics::f1(pred, gt), 0.0));
    pred = { {} };
    gt = { {} };
    EXPECT_TRUE(double_values_equal_(Metrics::f1(pred, gt), 0.0));
}
