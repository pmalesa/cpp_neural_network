#include "matrix.h"
#include "metrics.h"
#include <gtest/gtest.h>

#include <iostream>

using il = std::initializer_list<std::initializer_list<double>>;

class MetricsTest : public testing::Test {
public:
    MetricsTest() {}

protected:
    static constexpr double tolerance_ = 1e-9;

};

TEST_F(MetricsTest, GetConfusionMatrixValuesTest) {
    // Incorrect sizes
    Matrix pred_incorrect = { {1.0, 1.0, 0.0} };
    Matrix gt_incorrect = { {1.0, 1.0} };

    EXPECT_THROW(Metrics::get_confusion_matrix_values(pred_incorrect, gt_incorrect), std::logic_error);

    pred_incorrect = { {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0} };
    gt_incorrect = { {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0} };

    EXPECT_THROW(Metrics::get_confusion_matrix_values(pred_incorrect, gt_incorrect), std::logic_error);

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

    // Only true positives
    pred = { {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0} };
    gt = { {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);

    EXPECT_TRUE(cm.tp == 6 && cm.tn == 0 && cm.fp == 0 && cm.fn == 0);

    // Only true negatives
    pred = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    gt = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);

    EXPECT_TRUE(cm.tp == 0 && cm.tn == 6 && cm.fp == 0 && cm.fn == 0);

    // Only false positives
    pred = { {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0} };
    gt = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);

    EXPECT_TRUE(cm.tp == 0 && cm.tn == 0 && cm.fp == 6 && cm.fn == 0);

    // Only false negatives
    pred = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    gt = { {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0} };
    cm = Metrics::get_confusion_matrix_values(pred, gt);

    EXPECT_TRUE(cm.tp == 0 && cm.tn == 0 && cm.fp == 0 && cm.fn == 6);

    // Floating-point fuzz tolerance test
    // Slight numeric deviations should still classify correctly (TP=1, TN=2, FP=1, FN = 0)
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

TEST_F(MetricsTest, BinaryMetricsTest) {
    // Incorrect sizes
    Matrix pred_incorrect = { {1.0, 1.0, 0.0} };
    Matrix gt_incorrect = { {1.0, 1.0} };

    EXPECT_THROW(Metrics::accuracy(pred_incorrect, gt_incorrect), std::logic_error);
    EXPECT_THROW(Metrics::precision(pred_incorrect, gt_incorrect), std::logic_error);
    EXPECT_THROW(Metrics::recall(pred_incorrect, gt_incorrect), std::logic_error);
    EXPECT_THROW(Metrics::f1(pred_incorrect, gt_incorrect), std::logic_error);

    pred_incorrect = { {1.0, 1.0, 0.0}, {0.0, 1.0, 1.0} };
    gt_incorrect = { {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0} };

    EXPECT_THROW(Metrics::accuracy(pred_incorrect, gt_incorrect), std::logic_error);
    EXPECT_THROW(Metrics::precision(pred_incorrect, gt_incorrect), std::logic_error);
    EXPECT_THROW(Metrics::recall(pred_incorrect, gt_incorrect), std::logic_error);
    EXPECT_THROW(Metrics::f1(pred_incorrect, gt_incorrect), std::logic_error);

    // Row vector
    Matrix pred = { {1.0, 0.0, 1.0, 0.0, 0.0, 1.0 } };
    Matrix gt = { {1.0, 0.0, 1.0, 1.0, 0.0, 0.0} };

    EXPECT_NEAR(Metrics::accuracy(pred, gt), 4.0 / 6.0, tolerance_);
    EXPECT_NEAR(Metrics::precision(pred, gt), 2.0 / 3.0, tolerance_);
    EXPECT_NEAR(Metrics::recall(pred, gt), 2.0 / 3.0, tolerance_);
    EXPECT_NEAR(Metrics::f1(pred, gt), 2.0 / 3.0, tolerance_);

    // Column vector
    pred = { {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0} };

    EXPECT_NEAR(Metrics::accuracy(pred, gt), 4.0 / 6.0, tolerance_);
    EXPECT_NEAR(Metrics::precision(pred, gt), 2.0 / 3.0, tolerance_);
    EXPECT_NEAR(Metrics::recall(pred, gt), 2.0 / 3.0, tolerance_);
    EXPECT_NEAR(Metrics::f1(pred, gt), 2.0 / 3.0, tolerance_);

    // Perfect case (precision == recall == f1 == accuracy == 1.0)
    pred = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0} };

    EXPECT_NEAR(Metrics::accuracy(pred, gt), 1.0, tolerance_);
    EXPECT_NEAR(Metrics::precision(pred, gt), 1.0, tolerance_);
    EXPECT_NEAR(Metrics::recall(pred, gt), 1.0, tolerance_);
    EXPECT_NEAR(Metrics::f1(pred, gt), 1.0, tolerance_);

    // Worst case (precision == recall == f1 == accuracy == 0.0)
    pred = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0} };
    gt = { {0.0}, {1.0}, {0.0}, {0.0}, {1.0}, {0.0} };
    
    EXPECT_NEAR(Metrics::accuracy(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::precision(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::recall(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::f1(pred, gt), 0.0, tolerance_);

    // No true positives and no false positives (denominator == 0.0)
    pred = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    gt = { {1.0}, {0.0}, {1.0}, {1.0}, {0.0}, {0.0} };

    EXPECT_NEAR(Metrics::precision(pred, gt), 0.0, tolerance_);

    // No true positives and no false negatives (denominator == 0.0)
    pred = { {1.0}, {0.0}, {1.0}, {0.0}, {0.0}, {0.0} };
    gt = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };

    EXPECT_NEAR(Metrics::recall(pred, gt), 0.0, tolerance_);

    // No true positives, no false positives and no false negatives (denominator == 0.0)
    pred = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };
    gt = { {0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0} };

    EXPECT_NEAR(Metrics::f1(pred, gt), 0.0, tolerance_);

    // Empty vectors (row/column matrices)
    pred = { };
    gt = { };
    
    EXPECT_NEAR(Metrics::accuracy(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::precision(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::recall(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::f1(pred, gt), 0.0, tolerance_);

    pred = { {} };
    gt = { {} };

    EXPECT_NEAR(Metrics::accuracy(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::precision(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::recall(pred, gt), 0.0, tolerance_);
    EXPECT_NEAR(Metrics::f1(pred, gt), 0.0, tolerance_);
}
