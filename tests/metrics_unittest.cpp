#include "matrix.h"
#include "metrics.h"
#include <gtest/gtest.h>

#include <iostream>

using il = std::initializer_list<std::initializer_list<double>>;

class MetricsTest : public testing::Test {
public:
    MetricsTest() {}

protected:
    bool double_values_equal_(double lhs, double rhs, double tolerance = 1e-9) {
        return abs(lhs - rhs) < tolerance;
    }
};

TEST_F(MetricsTest, GetConfusionMatrixValuesTest) {
    // Incorrect sizes
    Matrix pred_incorrect = { {1.0, 1.0, 0.0} };
    Matrix gt_incorrect = { {1.0, 1.0} };
    EXPECT_THROW(Metrics::precision(pred_incorrect, gt_incorrect), std::logic_error);

    pred_incorrect = { {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0} };
    gt_incorrect = { {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0} };
    EXPECT_THROW(Metrics::precision(pred_incorrect, gt_incorrect), std::logic_error);

    // Row vector
    Matrix pred = { {1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0 } };
    Matrix gt = { {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0} };
    Metrics::ConfusionMatrix cm = Metrics::get_confusion_matrix_values(pred, gt);
    EXPECT_TRUE(cm.tp == 2 && cm.tn == 2 && cm.fp == 2 && cm.fn == 1);
    pred = { {1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0} };
    gt = { {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);
    EXPECT_TRUE(cm.tp == 3 && cm.tn == 4 && cm.fp == 2 && cm.fn == 1);

    // Column vector
    pred = { {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0}, {1.0} };
    gt = { {1.0}, {0.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);
    EXPECT_TRUE(cm.tp == 2 && cm.tn == 2 && cm.fp == 2 && cm.fn == 1);
    pred = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0}, {1.0}, {0.0}, {1.0}, {0.0} };
    gt = { {1.0}, {0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {0.0} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);
    EXPECT_TRUE(cm.tp == 3 && cm.tn == 4 && cm.fp == 2 && cm.fn == 1);

    // Floating-point fuzz tolerance test
    pred = { {0.9999999999, 0.0000000001, 1.0000000001, -0.0000000001} };    
    gt = { {1.0, 0.0, 0.0, 0.0} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);
    EXPECT_TRUE(cm.tp == 1 && cm.tn == 2 && cm.fp == 1 && cm.fn == 0);

    // Empty vectors (row/column matrices)
    pred = { };
    gt = { };
    cm = Metrics::get_confusion_matrix_values(pred, gt);
    EXPECT_TRUE(cm.tp == 0 && cm.tn == 0 && cm.fp == 0 && cm.fn == 0);
    pred = { {} };
    gt = { {} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);
    EXPECT_TRUE(cm.tp == 0 && cm.tn == 0 && cm.fp == 0 && cm.fn == 0); 

    // Invalid labels
    pred = { {1.0, 2.0, -1.0, 0.0} };
    gt = { {1.0, 0.0, 0.0, 1.0} };
    EXPECT_THROW(Metrics::get_confusion_matrix_values(pred, gt), std::logic_error);
}

TEST_F(MetricsTest, BinaryClassificationAccuracyTest) {
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

TEST_F(MetricsTest, BinaryClassificationPrecisionTest) {
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

TEST_F(MetricsTest, RecallTest) {
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

TEST_F(MetricsTest, F1Test) {
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
