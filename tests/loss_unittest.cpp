#include "matrix.h"
#include "loss.h"
#include <gtest/gtest.h>

using il = std::initializer_list<std::initializer_list<double>>;

class LossTest : public testing::Test {
public:
    LossTest() {}

protected:
};

TEST_F(LossTest, MSETest) {
    Matrix y_incorrect = { {1.0}, {2.0} };
    Matrix y_true_1 = { {1.0}, {2.0}, {3.0} };
    Matrix y_pred_1 = { {2.0}, {2.0}, {4.0} };
    EXPECT_THROW(Loss::mse(y_incorrect, y_pred_1), std::invalid_argument);
    EXPECT_TRUE(Loss::mse(y_true_1, y_pred_1) == 2.0 / 3.0);
    Matrix y_true_2 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    Matrix y_pred_2 = {
        {1.5, 1.5, 3.0},
        {4.5, 4.5, 6.0},
        {7.5, 7.5, 9.0}
    };
    EXPECT_THROW(Loss::mse(y_true_2, y_incorrect), std::invalid_argument);
    EXPECT_TRUE(Loss::mse(y_true_2, y_pred_2) == 1.5 / 9.0);
}

TEST_F(LossTest, MSEDerivativeTest) {
    Matrix y_incorrect = { {1.0}, {2.0} };
    Matrix y_true_1 = { {1.0}, {2.0}, {3.0} };
    Matrix y_pred_1 = { {2.0}, {2.0}, {4.0} };
    Matrix grad_1 = Loss::mse_derivative(y_true_1, y_pred_1);
    EXPECT_THROW(Loss::mse_derivative(y_incorrect, y_pred_1), std::invalid_argument);
    EXPECT_TRUE(grad_1.get_rows() == 3);
    EXPECT_TRUE(grad_1.get_cols() == 1);
    EXPECT_TRUE((grad_1 == il{ {2.0 / 3.0}, {0.0}, {2.0 / 3.0} }));

    Matrix y_true_2 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    Matrix y_pred_2 = {
        {1.5, 1.5, 3.0},
        {4.5, 4.5, 6.0},
        {7.5, 7.5, 9.0}
    };
    Matrix grad_2 = Loss::mse_derivative(y_true_2, y_pred_2);
    EXPECT_TRUE(grad_2.get_rows() == 3);
    EXPECT_TRUE(grad_2.get_cols() == 3);
    EXPECT_THROW(Loss::mse_derivative(y_incorrect, y_pred_2), std::invalid_argument);
    EXPECT_TRUE((grad_2 == il{ {0.1111111, 0.1111111, 0.1111111}, {-0.1111111, -0.1111111, -0.1111111}, {0.0, 0.0, 0.0} }));
}

TEST_F(LossTest, MAETest) {
    Matrix y_incorrect = { {1.0}, {2.0} };
    Matrix y_true_1 = { {1.0}, {2.0}, {3.0} };
    Matrix y_pred_1 = { {2.0}, {2.0}, {4.0} };
    EXPECT_THROW(Loss::mae(y_incorrect, y_pred_1), std::invalid_argument);
    EXPECT_TRUE(Loss::mae(y_true_1, y_pred_1) == 2.0 / 3.0);

    Matrix y_true_2 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    Matrix y_pred_2 = {
        {1.5, 1.5, 3.0},
        {4.5, 4.5, 6.0},
        {7.5, 7.5, 9.0}
    };
    EXPECT_THROW(Loss::mae(y_true_2, y_incorrect), std::invalid_argument);
    EXPECT_TRUE(Loss::mae(y_true_2, y_pred_2) == 3.0 / 9.0);
}

TEST_F(LossTest, MAEDerivativeTest) {
    Matrix y_incorrect = { {1.0}, {2.0} };
    Matrix y_true_1 = { {1.0}, {2.0}, {3.0} };
    Matrix y_pred_1 = { {2.0}, {2.0}, {4.0} };
    Matrix grad_1 = Loss::mae_derivative(y_true_1, y_pred_1);
    EXPECT_THROW(Loss::mae_derivative(y_incorrect, y_pred_1), std::invalid_argument);
    EXPECT_TRUE(grad_1.get_rows() == 3);
    EXPECT_TRUE(grad_1.get_cols() == 1);
    EXPECT_TRUE((grad_1 == il{ {1.0 / 3.0}, {0.0}, {1.0 / 3.0} }));

    Matrix y_true_2 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    Matrix y_pred_2 = {
        {1.5, 1.5, 3.0},
        {4.5, 4.5, 6.0},
        {7.5, 7.5, 9.0}
    };

    Matrix grad_2 = Loss::mae_derivative(y_true_2, y_pred_2);
    EXPECT_EQ(grad_2.get_rows(), 3);
    EXPECT_EQ(grad_2.get_cols(), 3);
    EXPECT_TRUE((grad_2 == il{ {0.1111111, 0.1111111, 0.1111111}, {-0.1111111, -0.1111111, -0.1111111}, {0.0, 0.0, 0.0} }));
}